#define BOOST_TEST_MODULE BatchNormTests

#include <boost/test/unit_test.hpp>
#include <popstd/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popconv/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplib_test/Convolution.hpp>
#include <poplib_test/FullyConnected.hpp>
#include <poplib_test/Util.hpp>
#include <popnn/BatchNorm.hpp>
#include <iostream>
#include <functional>
#include <boost/multi_array.hpp>

// Tolerances used in tests
#define FLOAT_REL_TOL  0.01
#define HALF_REL_TOL   0.1
#define FLOAT_ABS_TOL  1e-6
#define HALF_ABS_TOL   1e-5

using namespace poplar;
using namespace poplar::program;
using namespace popstd;
using namespace poplib_test::util;
using namespace popnn;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

static bool BatchNormConv(const std::vector<unsigned> dims,
                          float eps,
                          float learningRate,
                          unsigned tilesPerIPU,
                          const std::string &dataTypeStr,
                          const std::string &partialsTypeStr) {
  assert(dims.size() == 4);
  const auto batchSize = dims[0];
  const auto dimY = dims[1];
  const auto dimX = dims[2];
  const auto numChannels = dims[3];

  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::AGGRESSIVE_MULTICAST;
  info.tilesPerIPU = tilesPerIPU;
  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  popnn::addCodelets(graph);
  popreduce::addCodelets(graph);
  popconv::addCodelets(graph);

  auto acts = graph.addTensor(dataTypeStr, {batchSize, dimY, dimX, numChannels},
                              "act");
  popstd::mapTensorLinearly(graph, acts);

  auto prog = Sequence();

  Tensor mean, invStdDev;
  std::tie(mean, invStdDev) =
      bn::batchNormEstimates(graph, acts, eps, prog, partialsTypeStr);
  Tensor gamma, beta;
  std::tie(gamma, beta) =
      bn::createBatchNormParams(graph, acts);

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

  auto upload = Sequence();
  auto download = Sequence();

  auto rawHostActs =
          allocateHostMemoryForTensor(acts, upload, download);
  auto rawHostActsBN =
          allocateHostMemoryForTensor(actsBN, upload, download);
  auto rawHostGradsIn =
          allocateHostMemoryForTensor(gradsIn, upload, download);
  auto rawHostGradsOut =
          allocateHostMemoryForTensor(gradsOut, upload, download);
  auto rawHostActsWhitened =
          allocateHostMemoryForTensor(actsWhitened, upload, download);
  auto rawHostMean =
          allocateHostMemoryForTensor(mean, upload, download);
  auto rawHostInvStdDev =
          allocateHostMemoryForTensor(invStdDev, upload, download);
  auto rawHostGamma =
          allocateHostMemoryForTensor(gamma, upload, download);
  auto rawHostBeta =
          allocateHostMemoryForTensor(beta, upload, download);

  boost::multi_array<double, 4>
      hostActs(boost::extents[batchSize][dimY][dimX][numChannels]);
  boost::multi_array<double, 4>
      hostActsBN(boost::extents[batchSize][dimY][dimX][numChannels]);
  boost::multi_array<double, 4>
      hostGradsIn(boost::extents[batchSize][dimY][dimX][numChannels]);
  boost::multi_array<double, 4>
      hostGradsOut(boost::extents[batchSize][dimY][dimX][numChannels]);
  boost::multi_array<double, 4>
      hostActsWhitened(boost::extents[batchSize][dimY][dimX][numChannels]);
  boost::multi_array<double, 1>
      hostMean(boost::extents[numChannels]);
  boost::multi_array<double, 1>
      hostInvStdDev(boost::extents[numChannels]);
  boost::multi_array<double, 1>
      hostGamma(boost::extents[numChannels]);
  boost::multi_array<double, 1>
      hostBeta(boost::extents[numChannels]);

  std::mt19937 randomEngine;
  writeRandomValues(hostActs, -1.0, +5.0, randomEngine);
  writeRandomValues(hostGamma, 0, +6.0, randomEngine);
  writeRandomValues(hostBeta, -1.0, +5.0, randomEngine);
  writeRandomValues(hostGradsIn, 0, +4.0, randomEngine);
  auto modelGamma = hostGamma;
  auto modelBeta = hostBeta;
  auto modelGradsIn = hostGradsIn;

  copy(hostActs, dataTypeStr, rawHostActs.get());
  copy(hostGamma, dataTypeStr, rawHostGamma.get());
  copy(hostBeta, dataTypeStr, rawHostBeta.get());
  copy(hostGradsIn, dataTypeStr, rawHostGradsIn.get());

  Engine engine(graph, {std::move(upload), std::move(download),
                        std::move(prog)});

  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.

  copy(dataTypeStr, rawHostActsWhitened.get(), hostActsWhitened);
  copy(dataTypeStr, rawHostMean.get(), hostMean);
  copy(dataTypeStr, rawHostInvStdDev.get(), hostInvStdDev);
  copy(dataTypeStr, rawHostActsBN.get(), hostActsBN);
  copy(dataTypeStr, rawHostGradsOut.get(), hostGradsOut);
  copy(dataTypeStr, rawHostBeta.get(), hostBeta);
  copy(dataTypeStr, rawHostGamma.get(), hostGamma);

  bool matchesModel = true;

  boost::multi_array<double, 4> modelActsWhitened(boost::extents[batchSize]
                                                                [dimY][dimX]
                                                                [numChannels]);
  boost::multi_array<double, 1> modelMean(boost::extents[numChannels]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numChannels]);

  poplib_test::conv::batchNormEstimates(hostActs, eps, modelMean,
                                        modelInvStdDev);

  boost::multi_array<double, 4>
      modelActsBN(boost::extents[batchSize][dimY][dimX][numChannels]);
  poplib_test::conv::batchNormalise(hostActs, modelGamma, modelBeta, modelMean,
                                    modelInvStdDev, modelActsBN,
                                    modelActsWhitened);
  boost::multi_array<double, 4>
      modelGradsOut(boost::extents[batchSize][dimY][dimX][numChannels]);

  poplib_test::conv::batchNormGradients(modelActsWhitened, modelGradsIn,
                                        modelInvStdDev, modelGamma,
                                        modelGradsOut);
  poplib_test::conv::batchNormParamUpdate(modelActsWhitened, modelGradsIn,
                                          learningRate, modelGamma, modelBeta);

  const double relativeTolerance = dataTypeStr == "float"
                                   ? FLOAT_REL_TOL : HALF_REL_TOL;
  const double absoluteTolerance = dataTypeStr == "float"
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
                        const std::string &dataTypeStr,
                        const std::string &partialsTypeStr) {
  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::AGGRESSIVE_MULTICAST;
  info.tilesPerIPU = tilesPerIPU;
  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  popnn::addCodelets(graph);
  popreduce::addCodelets(graph);

  assert(dims.size() == 2);
  const unsigned batchSize = dims[0];
  const unsigned numActs = dims[1];

  auto acts = graph.addTensor(dataTypeStr, {batchSize, numActs}, "act");
  popstd::mapTensorLinearly(graph, acts);
  auto gradsIn = graph.addTensor(dataTypeStr, {batchSize, numActs}, "gradsIn");
  popstd::mapTensorLinearly(graph, gradsIn);
  auto prog = Sequence();

  Tensor mean, invStdDev;
  std::tie(mean, invStdDev) =
      popnn::bn::batchNormEstimates(graph, acts, eps, prog, partialsTypeStr);

  Tensor gamma, beta;
  std::tie(gamma, beta) =
      bn::createBatchNormParams(graph, acts);

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

  auto upload = Sequence();
  auto download = Sequence();
  auto rawHostActs = allocateHostMemoryForTensor(acts, upload, download);
  auto rawHostActsBN = allocateHostMemoryForTensor(actsBN, upload, download);
  auto rawHostActsWhitened =
          allocateHostMemoryForTensor(actsWhitened, upload, download);
  auto rawHostGradsIn =
          allocateHostMemoryForTensor(gradsIn, upload, download);
  auto rawHostGradsOut =
          allocateHostMemoryForTensor(gradsOut, upload, download);
  auto rawHostMean = allocateHostMemoryForTensor(mean, upload, download);
  auto rawHostInvStdDev = allocateHostMemoryForTensor(invStdDev, upload,
                                                      download);
  auto rawHostGamma = allocateHostMemoryForTensor(gamma, upload, download);
  auto rawHostBeta = allocateHostMemoryForTensor(beta, upload, download);

  boost::multi_array<double, 2> hostActs(boost::extents[batchSize][numActs]);
  boost::multi_array<double, 2> hostActsBN(boost::extents[batchSize][numActs]);
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
  writeRandomValues(hostActs, -1.0, +5.0, randomEngine);
  writeRandomValues(hostGradsIn, 0, +4.0, randomEngine);
  writeRandomValues(hostGamma, 0, +6.0, randomEngine);
  writeRandomValues(hostBeta, -1.0, +5.0, randomEngine);
  auto modelBeta = hostBeta;
  auto modelGamma = hostGamma;

  copy(hostActs, dataTypeStr, rawHostActs.get());
  copy(hostGradsIn, dataTypeStr, rawHostGradsIn.get());
  copy(hostGamma, dataTypeStr, rawHostGamma.get());
  copy(hostBeta, dataTypeStr, rawHostBeta.get());

  Engine engine(graph, {std::move(upload), std::move(download),
                        std::move(prog)});

  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.


  copy(dataTypeStr, rawHostActsWhitened.get(), hostActsWhitened);
  copy(dataTypeStr, rawHostGradsOut.get(), hostGradsOut);
  copy(dataTypeStr, rawHostMean.get(), hostMean);
  copy(dataTypeStr, rawHostInvStdDev.get(), hostInvStdDev);
  copy(dataTypeStr, rawHostActsBN.get(), hostActsBN);
  copy(dataTypeStr, rawHostBeta.get(), hostBeta);
  copy(dataTypeStr, rawHostGamma.get(), hostGamma);

  bool matchesModel = true;

  boost::multi_array<double, 2> modelActsWhitened(boost::extents[batchSize]
                                                                [numActs]);
  boost::multi_array<double, 1> modelMean(boost::extents[numActs]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numActs]);

  poplib_test::fc::batchNormEstimates(hostActs, eps, modelMean, modelInvStdDev);

  boost::multi_array<double, 2> modelActsBN(boost::extents[batchSize][numActs]);
  poplib_test::fc::batchNormalise(hostActs, modelGamma, modelBeta, modelMean,
                                  modelInvStdDev, modelActsBN,
                                  modelActsWhitened);

  boost::multi_array<double, 2> modelGradsOut(boost::extents[batchSize]
                                                            [numActs]);
  poplib_test::fc::batchNormGradients(hostActsWhitened, hostGradsIn,
                                      modelInvStdDev, modelGamma,
                                      modelGradsOut);
  poplib_test::fc::batchNormParamUpdate(modelActsWhitened, hostGradsIn,
                                        learningRate, modelGamma, modelBeta);

  const double relativeTolerance = dataTypeStr == "float"
                                   ? FLOAT_REL_TOL : HALF_REL_TOL;
  const double absoluteTolerance = dataTypeStr == "float"
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
  const std::string dataTypeStr = "half";
  const std::string partialsTypeStr = "float";
  const unsigned tilesPerIPU = 128;

  auto matchesModel = BatchNormConv({2, 28, 28, 32}, eps, learningRate,
                                    tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormConv_Batch4_Dim56x56_Ch64_LargeEps){
  const float eps = 0.01;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "half";
  const std::string partialsTypeStr = "float";

  const unsigned tilesPerIPU = 64;
  auto matchesModel = BatchNormConv({4, 56, 56, 64}, eps, learningRate,
                                    tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormConv_Batch16_Dim7x7_Ch8){
  const float eps = 0.001;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "half";
  const std::string partialsTypeStr = "float";

  const unsigned tilesPerIPU = 32;
  auto matchesModel = BatchNormConv({16, 7, 7, 8}, eps, learningRate,
                                    tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormConv_Batch4_DataFloat_PartialsFloat){
  const float eps = 0.0001;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "float";
  const std::string partialsTypeStr = "float";

  const unsigned tilesPerIPU = 64;
  auto matchesModel = BatchNormConv({1, 56, 56, 8}, eps, learningRate,
                                    tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch4_Acts2048) {
  const float eps = 0.001;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "half";
  const std::string partialsTypeStr = "float";
  const unsigned tilesPerIPU = 1216;
  auto matchesModel = BatchNormFc({4, 2048}, eps, learningRate,
                                  tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch16_Acts256_SmallEps) {
  const float eps = 0.00001;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "half";
  const std::string partialsTypeStr = "float";
  const unsigned tilesPerIPU = 64;
  auto matchesModel = BatchNormFc({16, 256}, eps, learningRate,
                                  tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch8_Acts512_LargeEps) {
  const float eps = 0.01;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "half";
  const std::string partialsTypeStr = "float";
  const unsigned tilesPerIPU = 64;
  auto matchesModel = BatchNormFc({16, 256}, eps, learningRate,
                                  tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(BatchNormFc_Batch8_Acts512_DataFloat_PartialsFloat) {
  const float eps = 0.001;
  const float learningRate = 0.1;
  const std::string dataTypeStr = "float";
  const std::string partialsTypeStr = "float";
  const unsigned tilesPerIPU = 64;
  auto matchesModel = BatchNormFc({16, 256}, eps, learningRate,
                                  tilesPerIPU, dataTypeStr, partialsTypeStr);
  BOOST_TEST(matchesModel == true);
}

