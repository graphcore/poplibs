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
#include <popstd/TileMapping.hpp>
#include <popconv/Convolution.hpp>
#include <popconv/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popreduce/Reduce.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplib_test/FullyConnected.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Pass.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace poplin;
using namespace popstd;
using namespace popreduce;
using poplib_test::Pass;

// Default tolerances used in tests
#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.3
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  MatMulOptions fwdOptions;
  unsigned numGroups;
  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  bool inPlaceUpdate = true;
  Type dataType;
  Type partialsType;
  double relativeTolerance, absoluteTolerance;
  IPUModel ipuModel;
  ipuModel.IPUExchangeType =
      IPUModel::ExchangeType::AGGRESSIVE_MULTICAST;
  Pass pass = Pass::ALL;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
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
  auto device  = ipuModel.createDevice();
  const auto &target = device.getTarget();

  Graph graph(device );
  popconv::addCodelets(graph);
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);

  fwdOptions.partialsType = partialsType;

  PlanningCache cache;
  fwdOptions.cache = &cache;
  fwdOptions.fullyConnectedPass =
      inferenceOnly ? FullyConnectedPass::INFERENCE_FWD :
                      FullyConnectedPass::TRAINING_FWD;
  Tensor prevAct =
      createMatMulGroupedInputLHS(graph, dataType,
                                  {numGroups, batchSize, inputSize},
                                  {numGroups, inputSize, outputSize},
                                  "prevAct",
                                  fwdOptions);
  Tensor weights =
      createMatMulGroupedInputRHS(graph, dataType,
                                  {numGroups, batchSize, inputSize},
                                  {numGroups, inputSize, outputSize},
                                  "weights",
                                  fwdOptions);
  auto biases = graph.addVariable(dataType, {numGroups, outputSize}, "biases");
  mapTensorLinearly(graph, biases);

  auto bwdOptions = fwdOptions;
  bwdOptions.fullyConnectedPass = FullyConnectedPass::TRAINING_BWD;

  auto fwdProg = Sequence();
  auto bwdProg = Sequence();

  Tensor nextAct;
  if (doFwdPass) {
    nextAct = poplin::matMulGrouped(graph, prevAct, weights, fwdProg, "",
                                    fwdOptions);
    auto bBiases = biases.reshape({numGroups, 1, outputSize})
                         .broadcast(batchSize, 1);
    addTo(graph, nextAct, bBiases, 1, fwdProg);
  } else {
    nextAct = graph.addVariable(dataType, {numGroups, batchSize, outputSize},
                                "nextAct");
    mapTensorLinearly(graph, nextAct);
  }

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct =
      allocateHostMemoryForTensor(prevAct, "prevAct", graph, tmap);
  auto rawHostWeights =
      allocateHostMemoryForTensor(weights, "weights", graph, tmap);
  auto rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                   tmap);
  auto rawHostNextAct =
      allocateHostMemoryForTensor(nextAct, "nextAct", graph, tmap);

  Tensor zDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas =
        poplin::createMatMulGroupedInputLHS(graph, dataType,
                                            {numGroups, batchSize, outputSize},
                                            {numGroups, outputSize, inputSize},
                                            "zDeltas", bwdOptions);
    rawHostZDeltas =
        allocateHostMemoryForTensor(zDeltas, "zDeltas", graph, tmap);
  }
  Tensor prevDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass) {
    prevDeltas =
        poplin::matMulGrouped(graph, zDeltas,
                              poplin::transposeGroupedMatrix(weights),
                              bwdProg, "", bwdOptions);
    rawHostPrevDeltas =
        allocateHostMemoryForTensor(prevDeltas, "prevDeltas", graph, tmap);
  }
  if (doWuPass) {
    auto wuOptions = fwdOptions;
    wuOptions.fullyConnectedPass = FullyConnectedPass::TRAINING_WU;
    poplin::matMulGroupedAcc(graph, weights, -learningRate,
                             poplin::transposeGroupedMatrix(prevAct),
                             zDeltas, bwdProg, "", wuOptions);
    auto biasDeltas = reduce(graph, zDeltas, {1}, popreduce::Operation::ADD,
                             bwdProg);
    addTo(graph, biases, biasDeltas, -learningRate, bwdProg);
  }

  Engine engine(device , graph, {std::move(fwdProg), std::move(bwdProg)});

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
  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);
  copy(target, dataType, rawHostNextAct.get(), hostNextAct);

  // Validate against a reference model.
  bool matchesModel = true;
  if (doFwdPass) {
    boost::multi_array<double, 3>
        modelNextAct(boost::extents[numGroups][batchSize][outputSize]);
    poplib_test::fc::fullyConnected(hostPrevAct, hostWeights, hostBiases,
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
    upload(engine, tmap);
    engine.run(1); // Run.
    download(engine, tmap);

    // Validate against a reference model.
    if (doBwdPass) {
      copy(target, dataType, rawHostPrevDeltas.get(), hostPrevDeltas);
      boost::multi_array<double, 3>
          modelPrevDeltas(boost::extents[numGroups][batchSize][inputSize]);
      poplib_test::fc::fullyConnectedBackward(hostZDeltas, modelWeights,
                                              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance, absoluteTolerance);
    }
    if (doWuPass) {
      copy(target, dataType, rawHostWeights.get(), hostWeights);
      copy(target, dataType, rawHostBiases.get(), hostBiases);
      poplib_test::fc::fullyConnectedWeightUpdate(learningRate, hostPrevAct,
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

  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.report(std::cout, opt);
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
