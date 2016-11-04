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
#include <popnn/ActivationMapping.hpp>
#include <popnn/FullyConnected.hpp>
#include <popnn/FullyConnectedPlan.hpp>
#include <poplar/HalfFloat.hpp>
#include <popnn/Net.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn_ref/FullyConnected.hpp>
#include <popnn_ref/NonLinearity.hpp>
#include <popnn_ref/Util.hpp>
#include <popnn/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace ref::util;

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  FPDataType dataType;
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
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
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
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env, createIPUModelDevice(info));

  std::string dataTypeStr(asString(dataType));

  // Create tensors.
  Tensor prevAct = graph.addTensor(dataTypeStr, {batchSize, inputSize},
                                   "prevAct");
  mapActivations(graph, prevAct);
  Tensor weights, biases;
  std::tie(weights, biases) = fc::createParams(graph, dataTypeStr,
                                               inputSize, outputSize);
  Tensor nextAct = graph.addTensor(dataTypeStr, {batchSize, outputSize},
                                   "nextAct");
  mapActivations(graph, nextAct);
  Tensor prevDeltas, zDeltas;
  if (!inferenceOnly) {
    zDeltas =
        graph.addTensor(dataTypeStr, {batchSize, outputSize},
                        "zDeltas");
    mapActivations(graph, zDeltas);
    prevDeltas =
        graph.addTensor(dataTypeStr, {batchSize, inputSize},
                        "prevDeltas");
    mapActivations(graph, prevDeltas);
  }

  auto outMapping = computeActivationsMapping(graph, nextAct[0], 0, 1);
  auto plan = fc::createPlan(graph, dataTypeStr, inputSize,
                             outMapping, inferenceOnly);

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

  auto fwdProg = Sequence();
  fwdProg.add(fc::fullyConnected(graph, outputSize,
                                 NON_LINEARITY_NONE,
                                 prevAct, weights, biases, nextAct, plan));

  auto bwdProg = Sequence();
  const auto learningRate = 0.5;
  if (!inferenceOnly) {
    bwdProg.add(
      fc::fullyConnectedBackward(graph, zDeltas, weights, prevDeltas, plan)
    );
    bwdProg.add(
      fc::fullyConnectedWeightUpdate(graph, zDeltas, prevAct, weights,
                                     biases, learningRate, plan)
    );
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
  writeRandomValues(hostPrevAct, 0.0, 1.0, randomEngine);
  writeRandomValues(hostWeights, 0.0, 1.0, randomEngine);
  writeRandomValues(hostBiases, 0.0, 1.0, randomEngine);
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
  ref::fc::fullyConnected(hostPrevAct, hostWeights, hostBiases, modelNextAct);
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
    writeRandomValues(hostZDeltas, 0.0, 1.0, randomEngine);
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
    ref::fc::fullyConnectedBackward(hostZDeltas, modelWeights, modelPrevDeltas);
    matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                 relativeTolerance);
    ref::fc::fullyConnectedWeightUpdate(learningRate, hostPrevAct, hostZDeltas,
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
