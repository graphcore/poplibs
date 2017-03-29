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
#include <popstd/ActivationMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popreduce/Reduce.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplib_test/FullyConnected.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace poplin;
using namespace popstd;
using namespace popreduce;

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  bool inPlaceUpdate = true;
  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;
  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of output channels")
    ("data-type",
     po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
     "Type of the data and the parameters")
    ("partials-type",
     po::value<FPDataType>(&partialsType)->default_value(FPDataType::FLOAT),
     "Type of the partials")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&info.numIPUs)->default_value(info.numIPUs),
     "Number of IPUs")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("in-place-update",
     po::value<bool>(&inPlaceUpdate)->default_value(true),
     "Perform param update in place")
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
  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));


  // Create tensors.
  Tensor prevAct = graph.addTensor(dataTypeStr, {batchSize, inputSize},
                                   "prevAct");
  mapActivations(graph, prevAct);
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = !inferenceOnly;
  mmOpt.cache = &cache;
  auto weights = createMatMulInputA(graph, dataTypeStr,
                                    {outputSize, inputSize},
                                    prevAct.transpose(),
                                    "weights", mmOpt);
  auto biases = graph.addTensor(dataTypeStr, {outputSize}, "biases");
  mapTensor(graph, biases);
  Tensor prevDeltas, zDeltas;
  if (!inferenceOnly) {
    zDeltas = graph.addTensor(dataTypeStr, {batchSize, outputSize}, "zDeltas");
    mapActivations(graph, zDeltas);
  }

  auto fwdProg = Sequence();
  auto nextAct = matMul(graph, prevAct, weights.transpose(), fwdProg,
                        "fc", mmOpt);
  auto bBiases = biases.broadcast(batchSize, 0)
                       .reshape({batchSize, outputSize});
  addTo(graph, nextAct, bBiases, 1, fwdProg);
  auto bwdProg = Sequence();
  const auto learningRate = 0.5;
  if (!inferenceOnly) {
    prevDeltas = matMul(graph, zDeltas, weights, bwdProg, "fc", mmOpt);
    auto weightDeltas = matMul(graph, zDeltas.transpose(), prevAct, bwdProg,
                               "fc", mmOpt);
    addTo(graph, weights, weightDeltas, -learningRate, bwdProg);
    auto biasDeltas = reduce(graph, zDeltas, bwdProg);
    addTo(graph, biases, biasDeltas, -learningRate, bwdProg);
  }

  auto upload = Sequence();
  auto download = Sequence();
  auto rawHostPrevAct = allocateHostMemoryForTensor(graph, prevAct, upload,
                                                    download);
  auto rawHostWeights = allocateHostMemoryForTensor(graph, weights, upload,
                                                    download);
  auto rawHostBiases = allocateHostMemoryForTensor(graph, biases, upload,
                                                   download);
  auto rawHostNextAct = allocateHostMemoryForTensor(graph, nextAct, upload,
                                                    download);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (!inferenceOnly) {
    rawHostZDeltas = allocateHostMemoryForTensor(graph, zDeltas, upload,
                                                 download);
    rawHostPrevDeltas = allocateHostMemoryForTensor(graph, prevDeltas, upload,
                                                    download);
  }

  Engine engine(graph, {std::move(upload), std::move(download),
                        std::move(fwdProg), std::move(bwdProg)});

  boost::multi_array<double, 2>
      hostPrevAct(boost::extents[batchSize][inputSize]);
  boost::multi_array<double, 2>
      hostWeights(boost::extents[outputSize][inputSize]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[outputSize]);
  boost::multi_array<double, 2>
      hostNextAct(boost::extents[batchSize][outputSize]);
  std::mt19937 randomEngine;
  writeRandomValues(hostPrevAct, -4.0, 4.0, randomEngine);
  writeRandomValues(hostWeights, -3.0, 3.0, randomEngine);
  writeRandomValues(hostBiases, -4.0, 4.0, randomEngine);
  copy(hostPrevAct, dataTypeStr, rawHostPrevAct.get());
  copy(hostWeights, dataTypeStr, rawHostWeights.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());
  // Run the forward pass.
  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.
  copy(dataTypeStr, rawHostNextAct.get(), hostNextAct);

  // Validate against a reference model.
  boost::multi_array<double, 2>
      modelNextAct(boost::extents[batchSize][outputSize]);
  poplib_test::fc::fullyConnected(hostPrevAct, hostWeights, hostBiases,
                                  modelNextAct);
  bool matchesModel = checkIsClose("fwd", hostNextAct, modelNextAct,
                                   relativeTolerance);

  if (!inferenceOnly) {
    boost::multi_array<double, 2> hostZDeltas(
      boost::extents[batchSize][outputSize]
    );
    boost::multi_array<double, 2> hostPrevDeltas(
      boost::extents[batchSize][inputSize]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards pass.
    writeRandomValues(hostZDeltas, -5.0, 5.0, randomEngine);
    copy(hostZDeltas, dataTypeStr, rawHostZDeltas.get());
    engine.run(0); // Upload.
    engine.run(3); // Run.
    engine.run(1); // Download.
    copy(dataTypeStr, rawHostPrevDeltas.get(), hostPrevDeltas);
    copy(dataTypeStr, rawHostWeights.get(), hostWeights);
    copy(dataTypeStr, rawHostBiases.get(), hostBiases);


    // Validate against a reference model.
    boost::multi_array<double, 2>
        modelPrevDeltas(boost::extents[batchSize][inputSize]);
    poplib_test::fc::fullyConnectedBackward(hostZDeltas, modelWeights,
                                            modelPrevDeltas);
    matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                 relativeTolerance);
    poplib_test::fc::fullyConnectedWeightUpdate(learningRate, hostPrevAct,
                                                hostZDeltas,
                                                modelWeights, modelBiases);
    matchesModel &= checkIsClose("weights",
                                 hostWeights, modelWeights, relativeTolerance);
    matchesModel &= checkIsClose("biases",
                                 hostBiases, modelBiases, relativeTolerance);
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
