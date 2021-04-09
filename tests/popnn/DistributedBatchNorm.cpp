// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// For the collectives, we rely on whatever the poplibs collectives support
// and are thus constrained by what constrains it imposes. This test is only
// meant to be a functional test and should not be used for performance
// assessment.

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <functional>
#include <iostream>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Norms.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_test/exceptions.hpp>
#include <poplin/codelets.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/Norms.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/TensorCollectives.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <tuple>

// Tolerances used in tests
#define FLOAT_REL_TOL 0.1
#define HALF_REL_TOL 0.2
#define FLOAT_ABS_TOL 1e-5
#define HALF_ABS_TOL 7e-2

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popnn;
using namespace popops;
using namespace poplibs_support;

const OptionFlags engineOptions;

// We don't use group size here because we do an all reduce over all the
// replicas.
std::vector<Tensor> multiTensorAllReduce(Graph &graph,
                                         const std::vector<Tensor> &inputs,
                                         Sequence &prog, unsigned groupSize,
                                         const DebugContext &debugContext,
                                         const OptionFlags &options) {
  auto topLevelGraph = graph.getTopLevelGraph();
  std::vector<Tensor> outputs;
  for (const auto &t : inputs) {
    // Must convert to non-replicated because poplibs collectives requires it.
    // If gcl collectives are used then we can directly pass the replicated
    // tensor to the gcl reduce-add opp and it should return a replicated
    // all-reduced tensor.
    auto tNonReplicated = topLevelGraph.getNonReplicatedTensor(t);
    auto result =
        allReduce(topLevelGraph, tNonReplicated,
                  popops::CollectiveOperator::ADD, prog, debugContext, options);
    auto replicaResult =
        graph.addVariable(result.elementType(), result[0].shape());
    // We don't really care for the mapping here as this is just a functional
    // test, so just map linearly
    mapTensorLinearly(graph, replicaResult, 0, 2);
    prog.add(Copy(result, topLevelGraph.getNonReplicatedTensor(replicaResult)));
    outputs.push_back(replicaResult);
  }
  return outputs;
}

// create randomised "activation" tensors in replicated graphs and calculated
// the BN stats across them. The resulting stats are compared against reference
// values
static bool normTest(const DeviceType &deviceType,
                     const std::vector<std::size_t> dims, float eps,
                     unsigned tilesPerIPU, unsigned numReplicas,
                     const Type &dataType, bool unbiasedVarEstimate,
                     bool stableAlgo, const Type &partialsType,
                     bool dumpProfile, bool compile_only) {
  assert(dims.size() >= 2);

  const auto batchSize = dims[0];
  const auto numChannels = dims[1];
  const auto fullBatchSize = batchSize * numReplicas;

  auto device = createTestDevice(deviceType, numReplicas, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  poplin::addCodelets(graph);

  const auto fieldSize = std::accumulate(dims.begin() + 2, dims.end(), 1U,
                                         std::multiplies<std::size_t>());

  std::vector<std::size_t> actDims;
  actDims.push_back(batchSize);
  actDims.resize(dims.size() - 1);
  std::copy(dims.begin() + 2, dims.end(), actDims.begin() + 1);
  actDims.push_back(dims[1]);

  auto replicatedGraph = graph.createReplicatedGraph(numReplicas);
  auto acts = replicatedGraph.addVariable(dataType, actDims, "act");
  poputil::mapTensorLinearly(replicatedGraph, acts);
  // Channel dimension as the second dimension. Statistics are computed over all
  // dimensions other than dimension 1.
  acts = acts.dimShufflePartial({acts.rank() - 1}, {1});

  auto prog = Sequence();

  auto [mean, invStdDev] = bn::distributedBatchNormStatistics(
      replicatedGraph, acts, eps, prog, unbiasedVarEstimate,
      multiTensorAllReduce, numReplicas * acts.dim(0), stableAlgo,
      partialsType);
  auto [gamma, beta] = popnn::createNormParams(replicatedGraph, acts);
  auto [actsBN, actsWhitened] = bn::batchNormalise(replicatedGraph, acts, gamma,
                                                   beta, mean, invStdDev, prog);
  (void)actsWhitened;

  auto gradsIn = graph.clone(actsBN);

  auto gradsOut = bn::distributedBatchNormGradients(
      replicatedGraph, acts, gradsIn, mean, invStdDev, gamma, prog,
      multiTensorAllReduce, numReplicas * acts.dim(0), partialsType);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  auto nonReplicatedActs =
      graph.getNonReplicatedTensor(acts).reshapePartial(0, 2, {fullBatchSize});
  auto rawHostActs = allocateHostMemoryForTensor(
      nonReplicatedActs, "acts", graph, uploadProg, downloadProg, tmap);

  auto nonReplicatedActsBn =
      graph.getNonReplicatedTensor(actsBN).reshapePartial(0, 2,
                                                          {fullBatchSize});
  auto rawHostActsBN = allocateHostMemoryForTensor(
      nonReplicatedActsBn, "actsBN", graph, uploadProg, downloadProg, tmap);

  auto nonReplicatedGradsIn =
      graph.getNonReplicatedTensor(gradsIn).reshapePartial(0, 2,
                                                           {fullBatchSize});
  auto rawHostGradsIn = allocateHostMemoryForTensor(
      nonReplicatedGradsIn, "gradsIn", graph, uploadProg, downloadProg, tmap);

  auto nonReplicatedGradsOut =
      graph.getNonReplicatedTensor(gradsOut).reshapePartial(0, 2,
                                                            {fullBatchSize});
  auto rawHostGradsOut = allocateHostMemoryForTensor(
      nonReplicatedGradsOut, "gradsOut", graph, uploadProg, downloadProg, tmap);

  auto nonReplicatedMean = graph.getNonReplicatedTensor(mean)[0];
  auto rawHostMean = allocateHostMemoryForTensor(
      nonReplicatedMean, "mean", graph, uploadProg, downloadProg, tmap);

  auto nonReplicatedInvStdDev = graph.getNonReplicatedTensor(invStdDev)[0];
  auto rawHostInvStdDev =
      allocateHostMemoryForTensor(nonReplicatedInvStdDev, "invStdDev", graph,
                                  uploadProg, downloadProg, tmap);

  auto rawHostGamma = allocateHostMemoryForTensor(
      gamma, "gamma", replicatedGraph, uploadProg, downloadProg, tmap);

  auto rawHostBeta = allocateHostMemoryForTensor(
      beta, "beta", replicatedGraph, uploadProg, downloadProg, tmap);

  unsigned numStatsElems = numChannels;

  boost::multi_array<double, 3> hostActs(
      boost::extents[fullBatchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostActsBN(
      boost::extents[fullBatchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostGradsIn(
      boost::extents[fullBatchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostGradsOut(
      boost::extents[fullBatchSize][numChannels][fieldSize]);
  boost::multi_array<double, 1> hostMean(boost::extents[numStatsElems]);
  boost::multi_array<double, 1> hostInvStdDev(boost::extents[numStatsElems]);
  boost::multi_array<double, 1> hostGamma(boost::extents[numChannels]);
  boost::multi_array<double, 1> hostBeta(boost::extents[numChannels]);

  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostActs, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostGamma, 0., +6.0, randomEngine);
  writeRandomValues(target, dataType, hostBeta, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostGradsIn, 0., +4.0, randomEngine);

  auto modelGamma = hostGamma;
  auto modelBeta = hostBeta;
  auto modelGradsIn = hostGradsIn;

  copy(target, hostActs, dataType, rawHostActs.get());
  for (unsigned i = 0; i != numReplicas; ++i) {
    void *dstGamma =
        rawHostGamma.get() +
        i * hostGamma.num_elements() * target.getTypeSize(gamma.elementType());
    copy(target, hostGamma, dataType, dstGamma);
    void *dstBeta =
        rawHostBeta.get() +
        i * hostBeta.num_elements() * target.getTypeSize(beta.elementType());
    copy(target, hostBeta, dataType, dstBeta);
  }
  copy(target, hostGradsIn, dataType, rawHostGradsIn.get());

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);

  if (compile_only)
    return 0;

  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  copy(target, dataType, rawHostMean.get(), hostMean);
  copy(target, dataType, rawHostInvStdDev.get(), hostInvStdDev);
  copy(target, dataType, rawHostActsBN.get(), hostActsBN);
  copy(target, dataType, rawHostGradsOut.get(), hostGradsOut);

  bool matchesModel = true;

  boost::multi_array<double, 3> modelActsWhitened(
      boost::extents[fullBatchSize][numChannels][fieldSize]);
  boost::multi_array<double, 1> modelMean(boost::extents[numStatsElems]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numStatsElems]);
  const auto normType = poplibs_test::norm::NormType::BatchNorm;

  poplibs_test::norm::normStatistics(hostActs, eps, unbiasedVarEstimate,
                                     stableAlgo, modelMean, modelInvStdDev,
                                     normType, false);

  boost::multi_array<double, 3> modelActsBN(
      boost::extents[fullBatchSize][numChannels][fieldSize]);
  poplibs_test::norm::normalise(hostActs, modelGamma, modelBeta, modelMean,
                                modelInvStdDev, modelActsBN, modelActsWhitened,
                                normType, false);
  boost::multi_array<double, 3> modelGradsOut(
      boost::extents[fullBatchSize][numChannels][fieldSize]);

  poplibs_test::norm::normGradients(modelActsWhitened, modelGradsIn,
                                    modelInvStdDev, modelGamma, modelGradsOut,
                                    normType, false);

  const double relativeTolerance =
      dataType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  const double absoluteTolerance =
      dataType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  matchesModel &= checkIsClose("mean", hostMean, modelMean, relativeTolerance,
                               absoluteTolerance);
  matchesModel &= checkIsClose("InvStdDev", hostInvStdDev, modelInvStdDev,
                               relativeTolerance, absoluteTolerance);
  matchesModel &= checkIsClose("actsBN", hostActsBN, modelActsBN,
                               relativeTolerance, absoluteTolerance);
  matchesModel &= checkIsClose("gradsOut", hostGradsOut, modelGradsOut,
                               relativeTolerance, absoluteTolerance);

  if (deviceType != DeviceType::Cpu && dumpProfile) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  return matchesModel;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  float eps;
  Type dataType;
  Type partialsType;
  unsigned tilesPerIPU;
  unsigned numReplicas = 1;
  ShapeOption<std::size_t> dims;
  bool unbiasedVarEstimate = false;
  bool stableAlgo = false;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("eps",
     po::value<float>(&eps)->required(),
     "eps")
    ("stable-algo-for-stats",
     po::value<bool>(&stableAlgo)->default_value(stableAlgo),
     "use stable algorithms for computing statistics")
    ("profile", "Output profiling report")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data Type")
    ("partials-type",
     po::value<Type>(&partialsType)->required(),
     "Partials Type")
    ("tiles-per-ipu",
     po::value<unsigned>(&tilesPerIPU)->required(),
     "Tiles per IPU")
    ("num-replicas",
     po::value<unsigned>(&numReplicas)->required(),
     "Number of replicas")
    ("dims",
     po::value<ShapeOption<std::size_t>>(&dims)->required(),
     "Dimensions : {batch,channels, ....field....}, where field could be "
     "empty or have any dimension, total batch size is multiplied by number "
     "of replicas")
    ("unbiased-var-estimate",
     po::value<bool>(&unbiasedVarEstimate)->default_value(unbiasedVarEstimate),
     "Use unbiased variance estimate");
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  bool dumpProfile = vm.count("profile");

  std::cerr << "\n Starting Distributed batch norm test with ";
  std::cerr << numReplicas << " replicas";

  if (dims.val.size() < 2) {
    std::cerr << "error: norm test must have tensor dimensions of at least 2";
    return 1;
  }
  auto matchesModel =
      normTest(deviceType, dims.val, eps, tilesPerIPU, numReplicas, dataType,
               unbiasedVarEstimate, stableAlgo, partialsType, dumpProfile,
               vm.count("compile-only"));
  return matchesModel ? 0 : 1;
}
