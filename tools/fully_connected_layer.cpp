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
#include "popops/Cast.hpp"
#include "poputil/exceptions.hpp"
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popops;
using poplibs_test::Pass;

const OptionFlags defaultEngineOptions {
  {"target.workerStackSizeInBytes", "0x200"},
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
  bool bias;
  bool inPlaceUpdate = true;
  bool reportPlan;
  Type inputType;
  Type outputType;
  Type partialsType;
  double relativeTolerance, absoluteTolerance;
  IPUModel ipuModel;
  Pass pass = Pass::ALL;
  bool useAggressiveRegrouping = false;
  bool reportVarStorage = false;
  double maxOutputMemoryProportion = 0;
  std::string tempMemoryBudget = "0";

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
    ("bias", po::value<bool>(&bias)->default_value(true),
     "Add a bias to each output")
    ("data-type",
     po::value<Type>(&inputType)->default_value(HALF),
     "Type of the input and output data")
    ("input-type",
     po::value<Type>(&inputType),
     "Type of the input data")
    ("output-type",
     po::value<Type>(&outputType),
     "Type of the output data")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(FLOAT),
     "Type of the partials")
    ("use-aggressive-regrouping",
      po::value<bool>(&useAggressiveRegrouping)->
                       default_value(useAggressiveRegrouping),
      "Use aggressive regrouping in acts/weights rearrangements")
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
    ("report-var-storage",
     po::value<bool>(
       &reportVarStorage
     )->default_value(reportVarStorage),
     "Report tensor storage information "
    )
    ("tempMemoryBudget",
       po::value<std::string>(&tempMemoryBudget)
           ->default_value(tempMemoryBudget),
     "Constrain the planner to limit the expected memory use. "
     "If 0, memory usage is unconstrained.")
    ("max-output-mem-prop",
     po::value<double>(&maxOutputMemoryProportion)->default_value(0),
     "Proportion of tile usage that is deemed \"large\" for outputs such "
     "that the convolution planner will try and serialize the computation. "
     "default behaviour if 0 is used.")
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

  if ((vm["output-type"].empty() != vm["input-type"].empty()) ||
      (!vm["data-type"].defaulted() && !vm["output-type"].empty())) {
    throw poputil::poplibs_error("Please specify either --data-type OR "
                                 "(--input-type AND --output-type), not both.");
  }
  if (vm["output-type"].empty()) {
    outputType = inputType;
  }

  if (vm["tolerance"].empty()) {
    if (outputType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (outputType == FLOAT) {
    absoluteTolerance = FLOAT_ABS_TOL;
   } else {
    absoluteTolerance = HALF_ABS_TOL;
  }

  const auto learningRate = 0.5;
  auto device = createTestDevice(deviceType, ipuModel.numIPUs,
                                  ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();

  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplar::OptionFlags fwdOptions{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" :
                                            "TRAINING_FWD" },
    { "useAggressiveRegrouping", useAggressiveRegrouping ? "true" : "false" },
    { "tempMemoryBudget", tempMemoryBudget }
  };

  if (maxOutputMemoryProportion != 0) {
    fwdOptions.set("maxOutputMemoryProportion",
                   std::to_string(maxOutputMemoryProportion));
  }

  matmul::PlanningCache cache;
  Tensor prevAct =
      createMatMulGroupedInputLHS(graph,
                                  inputType,
                                  outputType,
                                  {numGroups, batchSize, inputSize},
                                  {numGroups, inputSize, outputSize},
                                  "prevAct",
                                  fwdOptions,
                                  &cache);
  Tensor weights =
      createMatMulGroupedInputRHS(graph,
                                  inputType,
                                  outputType,
                                  {numGroups, batchSize, inputSize},
                                  {numGroups, inputSize, outputSize},
                                  "weights",
                                  fwdOptions,
                                  &cache);
  Tensor biases;
  if (bias) {
    biases = graph.addVariable(outputType, {numGroups, outputSize}, "biases");
    mapTensorLinearly(graph, biases);
  }

  auto bwdOptions = fwdOptions;
  bwdOptions.set("fullyConnectedPass", "TRAINING_BWD");
  bwdOptions.set("useAggressiveRegrouping", useAggressiveRegrouping ?
                 "true" : "false");

  auto fwdProg = Sequence();
  auto bwdProg = Sequence();

  Tensor nextAct;
  if (doFwdPass) {
    nextAct = poplin::matMulGrouped(graph,
                                    prevAct,
                                    weights,
                                    fwdProg,
                                    outputType,
                                    "Fwd",
                                    fwdOptions,
                                    &cache);
    if (reportPlan) {
      std::cout << "Forward plan:\n";
      poplin::matMulGroupedReportPlan(std::cout,
                                      graph,
                                      inputType,
                                      outputType,
                                      prevAct.shape(),
                                      weights.shape(),
                                      fwdOptions,
                                      &cache);
    }
    if (bias) {
      auto bBiases = biases.reshape({numGroups, 1, outputSize})
                           .broadcast(batchSize, 1);
      addInPlace(graph, nextAct, bBiases, fwdProg);
    }
  } else {
    nextAct = graph.addVariable(
      inputType, {numGroups, batchSize, outputSize}, "nextAct");
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
  std::unique_ptr<char []> rawHostBiases;
  if (bias) {
    rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                uploadProg, downloadProg, tmap);
  }
  auto rawHostNextAct =
      allocateHostMemoryForTensor(nextAct, "nextAct", graph, uploadProg,
                                  downloadProg, tmap);

  Tensor zDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas =
        poplin::createMatMulGroupedInputLHS(graph,
                                            inputType,
                                            outputType,
                                            {numGroups, batchSize, outputSize},
                                            {numGroups, outputSize, inputSize},
                                            "zDeltas",
                                            bwdOptions,
                                            &cache);
    rawHostZDeltas =
        allocateHostMemoryForTensor(zDeltas, "zDeltas", graph, uploadProg,
                                    downloadProg, tmap);
  }
  Tensor prevDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass) {
    auto weightsTransposed = poplin::transposeGroupedMatrix(weights);
    prevDeltas = poplin::matMulGrouped(graph,
                                       zDeltas,
                                       weightsTransposed,
                                       bwdProg,
                                       outputType,
                                       "Bwd",
                                       bwdOptions,
                                       &cache);
    if (reportPlan) {
      std::cout << "Backward plan:\n";
      poplin::matMulGroupedReportPlan(std::cout,
                                      graph,
                                      inputType,
                                      outputType,
                                      zDeltas.shape(),
                                      weightsTransposed.shape(),
                                      bwdOptions,
                                      &cache);
    }
    rawHostPrevDeltas =
        allocateHostMemoryForTensor(prevDeltas, "prevDeltas", graph, uploadProg,
                                    downloadProg, tmap);
  }
  if (doWuPass) {
    auto wuOptions = fwdOptions;
    wuOptions.set("fullyConnectedPass", "TRAINING_WU");
    wuOptions.set("useAggressiveRegrouping",
                  useAggressiveRegrouping ? "true" : "false");
    auto prevActTransposed = poplin::transposeGroupedMatrix(prevAct);
    auto scale = graph.addConstant(weights.elementType(), {}, -learningRate);
    // the check on groups is done to exercise both grouped and ungrouped
    // variants of matmul
    if (numGroups == 1) {
      poplin::matMulAcc(graph, weights.squeeze({0}), scale,
                        prevActTransposed.squeeze({0}), zDeltas.squeeze({0}),
                        bwdProg, "Wu", wuOptions, &cache);
      weights.expand({0});
    } else {
      poplin::matMulGroupedAcc(graph,
                               weights,
                               scale,
                               prevActTransposed,
                               zDeltas,
                               bwdProg,
                               "Wu",
                               wuOptions,
                               &cache);
    }
    if (reportPlan) {
      std::cout << "WU plan:\n";
      poplin::matMulGroupedReportPlan(std::cout,
                                      graph,
                                      inputType,
                                      outputType,
                                      prevActTransposed.shape(),
                                      zDeltas.shape(),
                                      wuOptions,
                                      &cache);
    }
    if (bias) {
      auto biasDeltas = reduce(graph, zDeltas, {1}, popops::Operation::ADD,
                               bwdProg);
      const auto biasDeltasOut = cast(graph, biasDeltas, outputType, bwdProg);
      scaledAddTo(graph, biases, biasDeltasOut, -learningRate, bwdProg);
    }
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
  writeRandomValues(target, inputType, hostPrevAct, -4.0, 4.0, randomEngine);
  writeRandomValues(target, inputType, hostWeights, -3.0, 3.0, randomEngine);
  if (bias) {
    writeRandomValues(target, outputType, hostBiases, -4.0, 4.0, randomEngine);
  } else {
    std::fill(hostBiases.data(), hostBiases.data() + hostBiases.num_elements(),
              0.0);
  }
  copy(target, hostPrevAct, inputType, rawHostPrevAct.get());
  copy(target, hostWeights, inputType, rawHostWeights.get());
  if (bias) {
    copy(target, hostBiases, outputType, rawHostBiases.get());
  }
  // Run the forward pass.
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(uploadProgIndex);
    engine.run(fwdProgIndex); // Run.
    engine.run(downloadProgIndex);
  });
  copy(target, outputType, rawHostNextAct.get(), hostNextAct);

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
    writeRandomValues(target, inputType, hostZDeltas, -5.0, 5.0, randomEngine);
    copy(target, hostZDeltas, inputType, rawHostZDeltas.get());
    device.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(bwdProgIndex); // Run.
      engine.run(downloadProgIndex);
    });

    // Validate against a reference model.
    if (doBwdPass) {
      copy(target, outputType, rawHostPrevDeltas.get(), hostPrevDeltas);
      boost::multi_array<double, 3>
          modelPrevDeltas(boost::extents[numGroups][batchSize][inputSize]);
      poplibs_test::fc::fullyConnectedBackward(hostZDeltas, modelWeights,
                                              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance, absoluteTolerance);
    }
    if (doWuPass) {
      copy(target, inputType, rawHostWeights.get(), hostWeights);
      if (bias) {
        copy(target, outputType, rawHostBiases.get(), hostBiases);
      }
      poplibs_test::fc::fullyConnectedWeightUpdate(learningRate, hostPrevAct,
                                                  hostZDeltas, modelWeights,
                                                  modelBiases);
      matchesModel &= checkIsClose("weights",
                                   hostWeights, modelWeights,
                                   relativeTolerance, absoluteTolerance);
      if (bias) {
        matchesModel &= checkIsClose("biases",
                                     hostBiases, modelBiases, relativeTolerance,
                                     absoluteTolerance);
      }
    }
  }

  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    // Rerun the program to get cycles excluding host copies.
    engine.resetExecutionProfile();
    if (doFwdPass) {
      engine.run(fwdProgIndex);
    }
    if (doBwdPass || doWuPass) {
      engine.run(bwdProgIndex);
    }
    OptionFlags opt = {{ "showExecutionSteps", "true" }};
    if (reportVarStorage) {
      opt.set("showVarStorage", "true");
    }

    engine.printProfileSummary(std::cout, opt);
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
