#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <istream>
#include <ostream>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/FullyConnected.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_support/Compiler.hpp>
#include "TestDevice.hpp"
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popops;
using poplibs_test::Pass;

const OptionFlags defaultEngineOptions {
  {"target.textSectionSizeInBytes", "0xa000"},
  {"target.workerStackSizeInBytes", "0x200"},
};

const OptionFlags simDebugOptions {
  {"debug.trace", "false"}
};

// Default tolerances used in tests
#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.3
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  unsigned numGroups;
  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  bool inPlaceUpdate = true;
  bool reportPlan;
  Type dataType;
  Type partialsType;
  double relativeTolerance, absoluteTolerance;
  IPUModel ipuModel;
  Pass pass = Pass::ALL;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of output channels")
    ("data-type",
     po::value<Type>(&dataType)->default_value(HALF),
     "Type of the data and the parameters")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(FLOAT),
     "Type of the partials")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
                           default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&ipuModel.numIPUs)->
                           default_value(ipuModel.numIPUs),
     "Number of IPUs")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("num-groups",
     po::value<unsigned>(&numGroups)->default_value(1),
     "Number of groups in grouped matrix multiplication")
    ("in-place-update",
     po::value<bool>(&inPlaceUpdate)->default_value(true),
     "Perform param update in place")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  bool inferenceOnly = vm.count("inference-only");
  if (inferenceOnly && pass != Pass::ALL && pass != Pass::FWD) {
    std::cerr << "pass=" << pass << " specified with --inference-only\n";
    return 1;
  }

  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::BWD);
  bool doWuPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::WU);

  if (vm["tolerance"].empty()) {
    if (dataType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (dataType == FLOAT) {
    absoluteTolerance = FLOAT_ABS_TOL;
   } else {
    absoluteTolerance = HALF_ABS_TOL;
  }

  const auto learningRate = 0.5;
  auto device = createTestDevice(deviceType, ipuModel.numIPUs,
                                  ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();

  Graph graph(device );
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplar::OptionFlags fwdOptions{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" :
                                            "TRAINING_FWD" }
  };

  matmul::PlanningCache cache;
  Tensor prevAct =
      createMatMulGroupedInputLHS(graph, dataType,
                                  {numGroups, batchSize, inputSize},
                                  {numGroups, inputSize, outputSize},
                                  "prevAct",
                                  fwdOptions, &cache);
  Tensor weights =
      createMatMulGroupedInputRHS(graph, dataType,
                                  {numGroups, batchSize, inputSize},
                                  {numGroups, inputSize, outputSize},
                                  "weights",
                                  fwdOptions, &cache);
  auto biases = graph.addVariable(dataType, {numGroups, outputSize}, "biases");
  mapTensorLinearly(graph, biases);

  auto bwdOptions = fwdOptions;
  bwdOptions.set("fullyConnectedPass", "TRAINING_BWD");

  auto fwdProg = Sequence();
  auto bwdProg = Sequence();

  Tensor nextAct;
  if (doFwdPass) {
    nextAct = poplin::matMulGrouped(graph, prevAct, weights, fwdProg, "",
                                    fwdOptions, &cache);
    if (reportPlan) {
      std::cout << "Forward plan:\n";
      poplin::matMulGroupedReportPlan(std::cout, graph, dataType,
                                      prevAct.shape(), weights.shape(),
                                      fwdOptions, &cache);
    }
    auto bBiases = biases.reshape({numGroups, 1, outputSize})
                         .broadcast(batchSize, 1);
    addInPlace(graph, nextAct, bBiases, fwdProg);
  } else {
    nextAct = graph.addVariable(dataType, {numGroups, batchSize, outputSize},
                                "nextAct");
    mapTensorLinearly(graph, nextAct);
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct =
      allocateHostMemoryForTensor(prevAct, "prevAct", graph, uploadProg,
                                  downloadProg, tmap);
  auto rawHostWeights =
      allocateHostMemoryForTensor(weights, "weights", graph, uploadProg,
                                  downloadProg, tmap);
  auto rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                   uploadProg, downloadProg,
                                                   tmap);
  auto rawHostNextAct =
      allocateHostMemoryForTensor(nextAct, "nextAct", graph, uploadProg,
                                  downloadProg, tmap);

  Tensor zDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas =
        poplin::createMatMulGroupedInputLHS(graph, dataType,
                                            {numGroups, batchSize, outputSize},
                                            {numGroups, outputSize, inputSize},
                                            "zDeltas", bwdOptions, &cache);
    rawHostZDeltas =
        allocateHostMemoryForTensor(zDeltas, "zDeltas", graph, uploadProg,
                                    downloadProg, tmap);
  }
  Tensor prevDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass) {
    auto weightsTransposed = poplin::transposeGroupedMatrix(weights);
    prevDeltas =
        poplin::matMulGrouped(graph, zDeltas, weightsTransposed, bwdProg, "",
                              bwdOptions, &cache);
    if (reportPlan) {
      std::cout << "Backward plan:\n";
      poplin::matMulGroupedReportPlan(std::cout, graph, dataType,
                                      zDeltas.shape(),
                                      weightsTransposed.shape(),
                                      bwdOptions, &cache);
    }
    rawHostPrevDeltas =
        allocateHostMemoryForTensor(prevDeltas, "prevDeltas", graph, uploadProg,
                                    downloadProg, tmap);
  }
  if (doWuPass) {
    auto wuOptions = fwdOptions;
    wuOptions.set("fullyConnectedPass", "TRAINING_WU");
    auto prevActTransposed = poplin::transposeGroupedMatrix(prevAct);
    poplin::matMulGroupedAcc(graph, weights, -learningRate,
                             prevActTransposed, zDeltas, bwdProg, "",
                             wuOptions);
    if (reportPlan) {
      std::cout << "WU plan:\n";
      poplin::matMulGroupedReportPlan(std::cout, graph, dataType,
                                      prevActTransposed.shape(),
                                      zDeltas.shape(), wuOptions, &cache);
    }
    auto biasDeltas = reduce(graph, zDeltas, {1}, popops::Operation::ADD,
                             bwdProg);
    scaledAddTo(graph, biases, biasDeltas, -learningRate, bwdProg);
  }

  std::vector<Program> programs;
  const auto fwdProgIndex = programs.size();
  programs.push_back(std::move(fwdProg));
  const auto bwdProgIndex = programs.size();
  programs.push_back(std::move(bwdProg));
  const auto uploadProgIndex = programs.size();
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size();
  programs.push_back(std::move(downloadProg));
  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.executionProfile", "compute_sets");
  }
  Engine engine(graph, std::move(programs), engineOptions);
  engine.load(device);
  attachStreams(engine, tmap);

  boost::multi_array<double, 3>
      hostPrevAct(boost::extents[numGroups][batchSize][inputSize]);
  boost::multi_array<double, 3>
      hostWeights(boost::extents[numGroups][inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostBiases(boost::extents[numGroups][outputSize]);
  boost::multi_array<double, 3>
      hostNextAct(boost::extents[numGroups][batchSize][outputSize]);
  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostPrevAct, -4.0, 4.0, randomEngine);
  writeRandomValues(target, dataType, hostWeights, -3.0, 3.0, randomEngine);
  writeRandomValues(target, dataType, hostBiases, -4.0, 4.0, randomEngine);
  copy(target, hostPrevAct, dataType, rawHostPrevAct.get());
  copy(target, hostWeights, dataType, rawHostWeights.get());
  copy(target, hostBiases, dataType, rawHostBiases.get());
  // Run the forward pass.
  engine.run(uploadProgIndex);
  engine.run(fwdProgIndex); // Run.
  engine.run(downloadProgIndex);
  copy(target, dataType, rawHostNextAct.get(), hostNextAct);

  // Validate against a reference model.
  bool matchesModel = true;
  if (doFwdPass) {
    boost::multi_array<double, 3>
        modelNextAct(boost::extents[numGroups][batchSize][outputSize]);
    poplibs_test::fc::fullyConnected(hostPrevAct, hostWeights, hostBiases,
                                    modelNextAct);
    matchesModel &= checkIsClose("fwd", hostNextAct, modelNextAct,
                                 relativeTolerance, absoluteTolerance);

  }
  if (doBwdPass || doWuPass) {
    boost::multi_array<double, 3> hostZDeltas(
      boost::extents[numGroups][batchSize][outputSize]
    );
    boost::multi_array<double, 3> hostPrevDeltas(
      boost::extents[numGroups][batchSize][inputSize]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards pass.
    writeRandomValues(target, dataType, hostZDeltas, -5.0, 5.0, randomEngine);
    copy(target, hostZDeltas, dataType, rawHostZDeltas.get());
    engine.run(uploadProgIndex);
    engine.run(bwdProgIndex); // Run.
    engine.run(downloadProgIndex);

    // Validate against a reference model.
    if (doBwdPass) {
      copy(target, dataType, rawHostPrevDeltas.get(), hostPrevDeltas);
      boost::multi_array<double, 3>
          modelPrevDeltas(boost::extents[numGroups][batchSize][inputSize]);
      poplibs_test::fc::fullyConnectedBackward(hostZDeltas, modelWeights,
                                              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance, absoluteTolerance);
    }
    if (doWuPass) {
      copy(target, dataType, rawHostWeights.get(), hostWeights);
      copy(target, dataType, rawHostBiases.get(), hostBiases);
      poplibs_test::fc::fullyConnectedWeightUpdate(learningRate, hostPrevAct,
                                                  hostZDeltas, modelWeights,
                                                  modelBiases);
      matchesModel &= checkIsClose("weights",
                                   hostWeights, modelWeights,
                                   relativeTolerance, absoluteTolerance);
      matchesModel &= checkIsClose("biases",
                                   hostBiases, modelBiases, relativeTolerance,
                                   absoluteTolerance);
    }
  }

  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    engine.printSummary(std::cout, OptionFlags{
                          { "doLayerWiseBreakdown", "true" }
                        });
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
