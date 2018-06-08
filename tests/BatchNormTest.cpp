#define BOOST_TEST_MODULE BatchNormTests

#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
#include <popconv/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/FullyConnected.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/BatchNorm.hpp>
#include <iostream>
#include <functional>
#include <boost/multi_array.hpp>

// Tolerances used in tests
#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.2
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popnn;
using namespace popops;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

static bool BatchNormConv(const std::vector<unsigned> dims,
                          float eps,
                          float learningRate,
                          unsigned tilesPerIPU,
                          const Type &dataType,
                          const Type &partialsType) {
  assert(dims.size() == 4);
  const auto batchSize = dims[0];
  const auto dimY = dims[1];
  const auto dimX = dims[2];
  const auto numChannels = dims[3];

  auto device = createTestDevice(TEST_TARGET, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(device);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  popconv::addCodelets(graph);

  auto acts = graph.addVariable(dataType, {batchSize, dimY, dimX, numChannels},
                                "act");
  poputil::mapTensorLinearly(graph, acts);
  acts = acts.dimShufflePartial({3}, {1});

  auto prog = Sequence();

  Tensor mean, invStdDev;
  std::tie(mean, invStdDev) =
      bn::batchNormEstimates(graph, acts, eps, prog, partialsType);
  Tensor gamma, beta;
  std::tie(gamma, beta) =
      bn::createBatchNormParams(graph, acts);

  // create combined parameters for inference
  const auto combinedScale = mul(graph, gamma, invStdDev, prog);
  const auto addendPart = mul(graph, mean, combinedScale, prog);
  const auto addend = sub(graph, beta, addendPart, prog);
  auto actsBNInf = bn::batchNormalise(graph, acts, combinedScale, addend, prog);

  Tensor actsWhitened, actsBN;
  std::tie(actsBN, actsWhitened) =
      bn::batchNormalise(graph, acts, gamma, beta, mean, invStdDev, prog);

  auto gradsIn = graph.clone(actsWhitened);

  Tensor gammaDelta, betaDelta;
  std::tie(gammaDelta, betaDelta) =
    bn::batchNormDeltas(graph, actsWhitened, gradsIn, prog);

  auto gradsOut =
    bn::batchNormGradients(graph, actsWhitened, gradsIn, gammaDelta, betaDelta,
                           invStdDev, gamma, prog);

  bn::batchNormParamUpdate(graph, gammaDelta, betaDelta, learningRate, gamma,
                           beta, prog);

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActs =
      allocateHostMemoryForTensor(acts, "acts", graph, tmap);
  auto rawHostActsBN =
          allocateHostMemoryForTensor(actsBN, "actsBN", graph, tmap);
  auto rawHostActsBNInf =
          allocateHostMemoryForTensor(actsBNInf, "actsBNInf", graph, tmap);
  auto rawHostGradsIn =
          allocateHostMemoryForTensor(gradsIn, "gradsIn", graph, tmap);
  auto rawHostGradsOut =
          allocateHostMemoryForTensor(gradsOut, "gradsOut", graph, tmap);
  auto rawHostActsWhitened =
          allocateHostMemoryForTensor(actsWhitened, "actsWhitened",
                                      graph, tmap);
  auto rawHostMean =
          allocateHostMemoryForTensor(mean, "mean", graph, tmap);
  auto rawHostInvStdDev =
          allocateHostMemoryForTensor(invStdDev, "invStdDev", graph, tmap);
  auto rawHostGamma =
          allocateHostMemoryForTensor(gamma, "gamma", graph, tmap);
  auto rawHostBeta =
          allocateHostMemoryForTensor(beta, "beta", graph, tmap);

  boost::multi_array<double, 4>
      hostActs(boost::extents[batchSize][numChannels][dimY][dimX]);
  boost::multi_array<double, 4>
      hostActsBN(boost::extents[batchSize][numChannels][dimY][dimX]);
  boost::multi_array<double, 4>
      hostActsBNInf(boost::extents[batchSize][numChannels][dimY][dimX]);
  boost::multi_array<double, 4>
      hostGradsIn(boost::extents[batchSize][numChannels][dimY][dimX]);
  boost::multi_array<double, 4>
      hostGradsOut(boost::extents[batchSize][numChannels][dimY][dimX]);
  boost::multi_array<double, 4>
      hostActsWhitened(boost::extents[batchSize][numChannels][dimY][dimX]);
  boost::multi_array<double, 1>
      hostMean(boost::extents[numChannels]);
  boost::multi_array<double, 1>
      hostInvStdDev(boost::extents[numChannels]);
  boost::multi_array<double, 1>
      hostGamma(boost::extents[numChannels]);
  boost::multi_array<double, 1>
      hostBeta(boost::extents[numChannels]);

  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostActs, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostGamma, 0, +6.0, randomEngine);
  writeRandomValues(target, dataType, hostBeta, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostGradsIn, 0, +4.0, randomEngine);
  auto modelGamma = hostGamma;
  auto modelBeta = hostBeta;
  auto modelGradsIn = hostGradsIn;

  copy(target, hostActs, dataType, rawHostActs.get());
  copy(target, hostGamma, dataType, rawHostGamma.get());
  copy(target, hostBeta, dataType, rawHostBeta.get());
  copy(target, hostGradsIn, dataType, rawHostGradsIn.get());

  Engine engine(device, graph, prog);

  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);

  copy(target, dataType, rawHostActsWhitened.get(), hostActsWhitened);
  copy(target, dataType, rawHostMean.get(), hostMean);
  copy(target, dataType, rawHostInvStdDev.get(), hostInvStdDev);
  copy(target, dataType, rawHostActsBN.get(), hostActsBN);
  copy(target, dataType, rawHostActsBNInf.get(), hostActsBNInf);
  copy(target, dataType, rawHostGradsOut.get(), hostGradsOut);
  copy(target, dataType, rawHostBeta.get(), hostBeta);
  copy(target, dataType, rawHostGamma.get(), hostGamma);

  bool matchesModel = true;

  boost::multi_array<double, 4> modelActsWhitened(boost::extents[batchSize]
                                                                [numChannels]
                                                                [dimY][dimX]);
  boost::multi_array<double, 1> modelMean(boost::extents[numChannels]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numChannels]);

  poplibs_test::conv::batchNormEstimates(hostActs, eps, modelMean,
                                        modelInvStdDev);

  boost::multi_array<double, 4>
      modelActsBN(boost::extents[batchSize][numChannels][dimY][dimX]);
  poplibs_test::conv::batchNormalise(hostActs, modelGamma, modelBeta, modelMean,
                                    modelInvStdDev, modelActsBN,
                                    modelActsWhitened);
  boost::multi_array<double, 4>
      modelGradsOut(boost::extents[batchSize][numChannels][dimY][dimX]);

  poplibs_test::conv::batchNormGradients(modelActsWhitened, modelGradsIn,
                                        modelInvStdDev, modelGamma,
                                        modelGradsOut);
  poplibs_test::conv::batchNormParamUpdate(modelActsWhitened, modelGradsIn,
                                          learningRate, modelGamma, modelBeta);

  const double relativeTolerance = dataType == FLOAT
                                   ? FLOAT_REL_TOL : HALF_REL_TOL;
  const double absoluteTolerance = dataType == FLOAT
                                   ? FLOAT_ABS_TOL : HALF_ABS_TOL;
  matchesModel &=
    checkIsClose("actsWhitened", hostActsWhitened, modelActsWhitened,
                 relativeTolerance, absoluteTolerance);
  matchesModel &=
    checkIsClose("mean", hostMean, modelMean, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("InvStdDev", hostInvStdDev, modelInvStdDev, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("actsBN", hostActsBN, modelActsBN, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("actsBNInf", hostActsBNInf, modelActsBN, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("gradsOut", hostGradsOut, modelGradsOut, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("beta", hostBeta, modelBeta, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("gamma", hostGamma, modelGamma, relativeTolerance,
                 absoluteTolerance);

  return matchesModel;
}

static bool BatchNormFc(const std::vector<unsigned> dims,
                        float eps,
                        float learningRate,
                        unsigned tilesPerIPU,
                        const Type &dataType,
                        const Type &partialsType) {
  auto device = createTestDevice(TEST_TARGET, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(device);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  assert(dims.size() == 2);
  const unsigned batchSize = dims[0];
  const unsigned numActs = dims[1];

  auto acts = graph.addVariable(dataType, {batchSize, numActs}, "act");
  poputil::mapTensorLinearly(graph, acts);
  auto gradsIn = graph.addVariable(dataType, {batchSize, numActs}, "gradsIn");
  poputil::mapTensorLinearly(graph, gradsIn);
  auto prog = Sequence();

  Tensor mean, invStdDev;
  std::tie(mean, invStdDev) =
      popnn::bn::batchNormEstimates(graph, acts, eps, prog, partialsType);

  Tensor gamma, beta;
  std::tie(gamma, beta) =
      bn::createBatchNormParams(graph, acts);

  // create combined parameters for inference
  const auto combinedScale = mul(graph, gamma, invStdDev, prog);
  const auto addendPart = mul(graph, mean, combinedScale, prog);
  const auto addend = sub(graph, beta, addendPart, prog);
  auto actsBNInf = bn::batchNormalise(graph, acts, combinedScale, addend, prog);

  Tensor actsWhitened, actsBN;
  std::tie(actsBN, actsWhitened) =
      bn::batchNormalise(graph, acts, gamma, beta, mean, invStdDev, prog);

  Tensor gammaDelta, betaDelta;
  std::tie(gammaDelta, betaDelta) =
    bn::batchNormDeltas(graph, actsWhitened, gradsIn, prog);

  auto gradsOut =
    bn::batchNormGradients(graph, actsWhitened, gradsIn, gammaDelta, betaDelta,
                           invStdDev, gamma, prog);

  bn::batchNormParamUpdate(graph, gammaDelta, betaDelta, learningRate, gamma,
                           beta, prog);

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActs = allocateHostMemoryForTensor(acts, "acts", graph, tmap);
  auto rawHostActsBN = allocateHostMemoryForTensor(actsBN, "actsBN",
                                                   graph, tmap);
  auto rawHostActsBNInf = allocateHostMemoryForTensor(actsBNInf, "actsBNInf",
                                                   graph, tmap);
  auto rawHostActsWhitened =
          allocateHostMemoryForTensor(actsWhitened, "actsWhitened",
                                      graph, tmap);
  auto rawHostGradsIn =
          allocateHostMemoryForTensor(gradsIn, "gradsIn", graph, tmap);
  auto rawHostGradsOut =
          allocateHostMemoryForTensor(gradsOut, "gradsOut", graph, tmap);
  auto rawHostMean = allocateHostMemoryForTensor(mean, "mean", graph, tmap);
  auto rawHostInvStdDev = allocateHostMemoryForTensor(invStdDev, "invStdDev",
                                                      graph, tmap);
  auto rawHostGamma = allocateHostMemoryForTensor(gamma, "gamma", graph, tmap);
  auto rawHostBeta = allocateHostMemoryForTensor(beta, "beta", graph, tmap);

  boost::multi_array<double, 2> hostActs(boost::extents[batchSize][numActs]);
  boost::multi_array<double, 2> hostActsBN(boost::extents[batchSize][numActs]);
  boost::multi_array<double, 2> hostActsBNInf(boost::extents[batchSize]
                                                            [numActs]);
  boost::multi_array<double, 2> hostActsWhitened(boost::extents[batchSize]
                                                               [numActs]);
  boost::multi_array<double, 2> hostGradsIn(boost::extents[batchSize]
                                                          [numActs]);
  boost::multi_array<double, 2> hostGradsOut(boost::extents[batchSize]
                                                          [numActs]);
  boost::multi_array<double, 1> hostMean(boost::extents[numActs]);
  boost::multi_array<double, 1> hostInvStdDev(boost::extents[numActs]);
  boost::multi_array<double, 1> hostGamma(boost::extents[numActs]);
  boost::multi_array<double, 1> hostBeta(boost::extents[numActs]);

  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostActs, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostGradsIn, 0, +4.0, randomEngine);
  writeRandomValues(target, dataType, hostGamma, 0, +6.0, randomEngine);
  writeRandomValues(target, dataType, hostBeta, -1.0, +5.0, randomEngine);
  auto modelBeta = hostBeta;
  auto modelGamma = hostGamma;

  copy(target, hostActs, dataType, rawHostActs.get());
  copy(target, hostGradsIn, dataType, rawHostGradsIn.get());
  copy(target, hostGamma, dataType, rawHostGamma.get());
  copy(target, hostBeta, dataType, rawHostBeta.get());

  Engine engine(device, graph, prog);

  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap); // Download.

  copy(target, dataType, rawHostActsWhitened.get(), hostActsWhitened);
  copy(target, dataType, rawHostGradsOut.get(), hostGradsOut);
  copy(target, dataType, rawHostMean.get(), hostMean);
  copy(target, dataType, rawHostInvStdDev.get(), hostInvStdDev);
  copy(target, dataType, rawHostActsBN.get(), hostActsBN);
  copy(target, dataType, rawHostActsBNInf.get(), hostActsBNInf);
  copy(target, dataType, rawHostBeta.get(), hostBeta);
  copy(target, dataType, rawHostGamma.get(), hostGamma);

  bool matchesModel = true;

  boost::multi_array<double, 2> modelActsWhitened(boost::extents[batchSize]
                                                                [numActs]);
  boost::multi_array<double, 1> modelMean(boost::extents[numActs]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numActs]);

  poplibs_test::fc::batchNormEstimates(hostActs, eps,
                                       modelMean, modelInvStdDev);

  boost::multi_array<double, 2> modelActsBN(boost::extents[batchSize][numActs]);
  poplibs_test::fc::batchNormalise(hostActs, modelGamma, modelBeta, modelMean,
                                  modelInvStdDev, modelActsBN,
                                  modelActsWhitened);

  boost::multi_array<double, 2> modelGradsOut(boost::extents[batchSize]
                                                            [numActs]);
  poplibs_test::fc::batchNormGradients(hostActsWhitened, hostGradsIn,
                                      modelInvStdDev, modelGamma,
                                      modelGradsOut);
  poplibs_test::fc::batchNormParamUpdate(modelActsWhitened, hostGradsIn,
                                        learningRate, modelGamma, modelBeta);

  const double relativeTolerance = dataType == FLOAT
                                   ? FLOAT_REL_TOL : HALF_REL_TOL;
  const double absoluteTolerance = dataType == FLOAT
                                   ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  matchesModel &=
    checkIsClose("actsWhitened", hostActsWhitened, modelActsWhitened,
                 relativeTolerance, absoluteTolerance);
  matchesModel &=
    checkIsClose("invStdDev", hostInvStdDev, modelInvStdDev, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("actsBN", hostActsBN, modelActsBN, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("actsBN", hostActsBNInf, modelActsBN, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("gradsOut", hostGradsOut, modelGradsOut, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("beta", hostBeta, modelBeta, relativeTolerance,
                 absoluteTolerance);
  matchesModel &=
    checkIsClose("gamma", hostGamma, modelGamma, relativeTolerance,
                 absoluteTolerance);
  return matchesModel;
}


BOOST_AUTO_TEST_CASE(BatchNormConv_Batch2_Dim28x28_Ch32_SmallEps){
  const float eps = 0.000001;
  const float learningRate = 0.1;
  const Type dataType = HALF;
  const Type partialsType = FLOAT;
  const unsigned tilesPerIPU = 16;

  auto matchesModel = BatchNormConv({2, 28, 28, 32}, eps, learningRate,
                                    tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormConv_Batch4_Dim56x56_Ch64_LargeEps){
  const float eps = 0.01;
  const float learningRate = 0.1;
  const Type dataType = HALF;
  const Type partialsType = FLOAT;

  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormConv({4, 56, 56, 64}, eps, learningRate,
                                    tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormConv_Batch16_Dim7x7_Ch8){
  const float eps = 0.001;
  const float learningRate = 0.1;
  const Type dataType = HALF;
  const Type partialsType = FLOAT;

  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormConv({16, 7, 7, 8}, eps, learningRate,
                                    tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormConv_Batch1_DataFloat_PartialsFloat){
  const float eps = 0.0001;
  const float learningRate = 0.1;
  const Type dataType = FLOAT;
  const Type partialsType = FLOAT;

  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormConv({1, 56, 56, 8}, eps, learningRate,
                                    tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch4_Acts2048) {
  const float eps = 0.001;
  const float learningRate = 0.1;
  const Type dataType = HALF;
  const Type partialsType = FLOAT;
  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormFc({4, 2048}, eps, learningRate,
                                  tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch16_Acts256_SmallEps) {
  const float eps = 0.00001;
  const float learningRate = 0.1;
  const Type dataType = HALF;
  const Type partialsType = FLOAT;
  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormFc({16, 256}, eps, learningRate,
                                  tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch8_Acts512_LargeEps) {
  const float eps = 0.01;
  const float learningRate = 0.1;
  const Type dataType = HALF;
  const Type partialsType = FLOAT;
  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormFc({8, 256}, eps, learningRate,
                                  tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch8_Acts512_DataFloat_PartialsFloat) {
  const float eps = 0.001;
  const float learningRate = 0.1;
  const Type dataType = FLOAT;
  const Type partialsType = FLOAT;
  const unsigned tilesPerIPU = 16;
  auto matchesModel = BatchNormFc({8, 256}, eps, learningRate,
                                  tilesPerIPU, dataType, partialsType);
  BOOST_TEST(matchesModel == true);
}

