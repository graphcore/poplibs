#include "TestDevice.hpp"
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
#include <popnn/Norms.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/exceptions.hpp>
#include <poplibs_test/Norms.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popnn/LayerNorm.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popnn/Norms.hpp>
#include <iostream>
#include <functional>
#include <tuple>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>

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

const OptionFlags options {
  {"target.workerStackSizeInBytes", "0x180"},
};

static std::tuple<poplibs_test::norm::NormType, bool, unsigned>
parseTestType(const std::string &testTypeString,
              const std::vector<unsigned int> &dims,
              unsigned numGroups) {
  if (testTypeString == "BN-Conv") {
    return std::make_tuple(poplibs_test::norm::NormType::BatchNorm, true,
                           numGroups);
  } else if (testTypeString == "BN-Fc") {
    return std::make_tuple(poplibs_test::norm::NormType::BatchNorm, false,
                          numGroups);
  } else if (testTypeString == "GN-Conv") {
    return std::make_tuple(poplibs_test::norm::NormType::GroupNorm, true,
                          numGroups);
  } else if (testTypeString == "GN-Fc") {
    return std::make_tuple(poplibs_test::norm::NormType::GroupNorm, false,
                           numGroups);
  } else if (testTypeString == "IN-Conv") {
    return std::make_tuple(poplibs_test::norm::NormType::InstanceNorm, true,
                           dims.back());
  } else if (testTypeString == "LN-Conv") {
    return std::make_tuple(poplibs_test::norm::NormType::LayerNorm, true, 1);
  } else if (testTypeString == "LN-Fc") {
    return std::make_tuple(poplibs_test::norm::NormType::LayerNorm, false, 1);
  } else {
    throw poplibs_test::poplibs_test_error("Invalid test type");
  }
}

static std::pair<Tensor, Tensor>
normStatistics(Graph &graph, const Tensor acts,
               float eps,
               Sequence &prog,
               bool unbiasedVarEstimate,
               const Type &partialsType,
               const std::string &debugPrefix,
               unsigned numGroups,
               poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
      return bn::batchNormStatistics(graph, acts, eps, prog,
                                     unbiasedVarEstimate, partialsType,
                                     debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
      return gn::groupNormStatistics(graph, acts, eps, prog,
                                     numGroups, unbiasedVarEstimate,
                                     partialsType, debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
      return ln::layerNormStatistics(graph, acts, eps, prog,
                                     unbiasedVarEstimate, partialsType,
                                     debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
      return in::instanceNormStatistics(graph, acts, eps, prog,
                                        unbiasedVarEstimate, partialsType,
                                        debugPrefix);
  }
}

std::pair<Tensor, Tensor>
normalise(Graph &graph,
          const Tensor &acts,
          const Tensor &gamma,
          const Tensor &beta,
          const Tensor &mean,
          const Tensor &iStdDev,
          Sequence &prog,
          const std::string &debugPrefix,
          poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
      return bn::batchNormalise(graph, acts, gamma, beta, mean, iStdDev,
                                prog, debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
      return gn::groupNormalise(graph, acts, gamma, beta, mean, iStdDev,
                                prog, debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
      return ln::layerNormalise(graph, acts, gamma, beta, mean, iStdDev,
                                prog, debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
      return in::instanceNormalise(graph, acts, gamma, beta, mean, iStdDev,
                                   prog, debugPrefix);
  }
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph,
                   const Tensor &actsWhitened,
                   const Tensor &gradsIn,
                   Sequence &prog,
                   const Type &partialsType,
                   const std::string &debugPrefix,
                   poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
      return bn::batchNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                         partialsType, debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
      return gn::groupNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                         partialsType, debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
      return ln::layerNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                         partialsType, debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
      return in::instanceNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                            partialsType, debugPrefix);
  }
}

static Tensor
normGradients(Graph &graph,
              const Tensor &actsWhitened,
              const Tensor &gradsIn,
              const Tensor &iStdDev,
              const Tensor &gamma,
              Sequence &prog,
              const Type &partialsType,
              const std::string &debugPrefix,
              poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
      return bn::batchNormGradients(graph, actsWhitened, gradsIn, iStdDev,
                                    gamma, prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
      return gn::groupNormGradients(graph, actsWhitened, gradsIn, iStdDev,
                                    gamma, prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
      return ln::layerNormGradients(graph, actsWhitened, gradsIn, iStdDev,
                                    gamma, prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
      return in::instanceNormGradients(graph, actsWhitened, gradsIn, iStdDev,
                                       gamma, prog, partialsType, debugPrefix);
  }
}

static void
normParamUpdate(poplar::Graph &graph,
                const poplar::Tensor &gammaDelta,
                const poplar::Tensor &betaDelta,
                float learningRate,
                poplar::Tensor &gamma,
                poplar::Tensor &beta,
                poplar::program::Sequence &prog,
                const std::string &debugPrefix,
                poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
      bn::batchNormParamUpdate(graph, gammaDelta, betaDelta, learningRate,
                               gamma, beta, prog, debugPrefix);
      return;
  case poplibs_test::norm::NormType::GroupNorm:
      gn::groupNormParamUpdate(graph, gammaDelta, betaDelta, learningRate,
                               gamma, beta, prog, debugPrefix);
      return;
  case poplibs_test::norm::NormType::LayerNorm:
      ln::layerNormParamUpdate(graph, gammaDelta, betaDelta, learningRate,
                               gamma, beta, prog, debugPrefix);
      return;
  case poplibs_test::norm::NormType::InstanceNorm:
      in::instanceNormParamUpdate(graph, gammaDelta, betaDelta, learningRate,
                                  gamma, beta, prog, debugPrefix);
      return;
  }
}

static bool normConv(const DeviceType &deviceType,
                     const std::vector<unsigned> dims,
                     float eps,
                     float learningRate,
                     unsigned tilesPerIPU,
                     const Type &dataType,
                     bool unbiasedVarEstimate,
                     const Type &partialsType,
                     poplibs_test::norm::NormType normType,
                     unsigned numGroups,
                     bool dumpProfile) {
  assert(dims.size() == 4);
  const auto batchSize = dims[0];
  const auto dimY = dims[1];
  const auto dimX = dims[2];
  const auto numChannels = dims[3];

  auto device = createTestDevice(deviceType, 1, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  poplin::addCodelets(graph);

  auto acts = graph.addVariable(dataType, {batchSize, dimY, dimX, numChannels},
                                "act");
  poputil::mapTensorLinearly(graph, acts);
  acts = acts.dimShufflePartial({3}, {1});

  auto prog = Sequence();

  Tensor mean, invStdDev;
  const bool isBatchNorm = normType == poplibs_test::norm::NormType::BatchNorm;

  std::tie(mean, invStdDev) =
      normStatistics(graph, acts, eps, prog, unbiasedVarEstimate, partialsType,
                     "", numGroups, normType);
  Tensor gamma, beta;
  std::tie(gamma, beta) = popnn::createNormParams(graph, acts);

  Tensor actsWhitened, actsBN;
  std::tie(actsBN, actsWhitened) =
      normalise(graph, acts, gamma, beta, mean, invStdDev, prog, "", normType);
  Tensor actsBNInf;
  if (isBatchNorm) {
    // create combined parameters for inference
    const auto combinedScale = mul(graph, gamma, invStdDev, prog);
    const auto addendPart = mul(graph, mean, combinedScale, prog);
    const auto addend = sub(graph, beta, addendPart, prog);
    actsBNInf = bn::batchNormalise(graph, acts, combinedScale, addend, prog);
  }
  else
    actsBNInf = actsBN;

  auto gradsIn = graph.clone(actsWhitened);

  Tensor gammaDelta, betaDelta;
  std::tie(gammaDelta, betaDelta) =
      normParamGradients(graph, actsWhitened, gradsIn, prog, partialsType, "",
                         normType);
  auto gradsOut =
      normGradients(graph, actsWhitened, gradsIn, invStdDev, gamma, prog,
                    partialsType, "", normType);

  normParamUpdate(graph, gammaDelta, betaDelta, learningRate, gamma, beta,
                  prog, "", normType);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  auto rawHostActs =
      allocateHostMemoryForTensor(acts, "acts", graph, uploadProg, downloadProg,
                                  tmap);
  auto rawHostActsBN =
          allocateHostMemoryForTensor(actsBN, "actsBN", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostActsBNInf =
          allocateHostMemoryForTensor(actsBNInf, "actsBNInf", graph, uploadProg,
                                      downloadProg,tmap);
  auto rawHostGradsIn =
          allocateHostMemoryForTensor(gradsIn, "gradsIn", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostGradsOut =
          allocateHostMemoryForTensor(gradsOut, "gradsOut", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostActsWhitened =
          allocateHostMemoryForTensor(actsWhitened, "actsWhitened", graph,
                                      uploadProg, downloadProg, tmap);
  auto rawHostMean =
          allocateHostMemoryForTensor(mean, "mean", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostInvStdDev =
          allocateHostMemoryForTensor(invStdDev, "invStdDev", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostGamma =
          allocateHostMemoryForTensor(gamma, "gamma", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostBeta =
          allocateHostMemoryForTensor(beta, "beta", graph, uploadProg,
                                      downloadProg, tmap);
  unsigned numStatsElems = isBatchNorm ? numChannels :
                                         numGroups * batchSize;
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
      hostMean(boost::extents[numStatsElems]);
  boost::multi_array<double, 1>
      hostInvStdDev(boost::extents[numStatsElems]);
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

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0); // Run.
  });

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
  boost::multi_array<double, 1> modelMean(boost::extents[numStatsElems]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numStatsElems]);

  poplibs_test::norm::normStatistics(hostActs, eps, unbiasedVarEstimate,
                                     modelMean, modelInvStdDev, normType);

  boost::multi_array<double, 4>
      modelActsBN(boost::extents[batchSize][numChannels][dimY][dimX]);
  poplibs_test::norm::normalise(hostActs, modelGamma, modelBeta,
                                modelMean, modelInvStdDev,
                                modelActsBN, modelActsWhitened, normType);
  boost::multi_array<double, 4>
      modelGradsOut(boost::extents[batchSize][numChannels][dimY][dimX]);

  poplibs_test::norm::normGradients(modelActsWhitened, modelGradsIn,
                                    modelInvStdDev, modelGamma,
                                    modelGradsOut, normType);
  poplibs_test::norm::normParamUpdate(modelActsWhitened,
                                      modelGradsIn, learningRate,
                                      modelGamma, modelBeta, normType);

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

  if (deviceType != DeviceType::Cpu && dumpProfile) {
    engine.printSummary(std::cout, OptionFlags{
      { "doLayerWiseBreakdown", "true" }
    });
  }
  return matchesModel;
}

static bool normFc(const DeviceType &deviceType,
                   const std::vector<unsigned> dims,
                   float eps,
                   float learningRate,
                   unsigned tilesPerIPU,
                   const Type &dataType,
                   bool unbiasedVarEstimate,
                   const Type &partialsType,
                   poplibs_test::norm::NormType normType,
                   unsigned numGroups,
                   bool dumpProfile) {
  auto device = createTestDevice(deviceType, 1, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  poplin::addCodelets(graph);

  assert(dims.size() == 2);
  const unsigned batchSize = dims[0];
  const unsigned numActs = dims[1];

  auto acts = graph.addVariable(dataType, {batchSize, numActs}, "act");
  poputil::mapTensorLinearly(graph, acts);
  auto gradsIn = graph.addVariable(dataType, {batchSize, numActs}, "gradsIn");
  poputil::mapTensorLinearly(graph, gradsIn);
  auto prog = Sequence();

  const bool isBatchNorm = normType == poplibs_test::norm::NormType::BatchNorm;
  Tensor mean, invStdDev;
  std::tie(mean, invStdDev) =
      normStatistics(graph, acts, eps, prog, unbiasedVarEstimate, partialsType,
                     "", numGroups, normType);
  Tensor gamma, beta;
  std::tie(gamma, beta) = popnn::createNormParams(graph, acts);

  Tensor actsWhitened, actsBN;
  std::tie(actsBN, actsWhitened) =
      normalise(graph, acts, gamma, beta, mean, invStdDev, prog, "", normType);

  // create combined parameters for inference
  Tensor actsBNInf;
  if (isBatchNorm) {
    // The calculations here to obtain combinedScale and addendPart have to be
    // done at a higher precision as would be typical in an actual
    // implementation. Here we keep them at the data precision inorder not
    // to increase the graph size. The alternate approach would be to do the
    // calculation on the host and run a second graph to do inference alone.
    const auto combinedScale = mul(graph, gamma, invStdDev, prog);
    const auto addendPart = mul(graph, mean, combinedScale, prog);
    const auto addend = sub(graph, beta, addendPart, prog);
    actsBNInf = bn::batchNormalise(graph, acts, combinedScale, addend, prog);
  } else
    actsBNInf = actsBN;

  Tensor gammaDelta, betaDelta;
  std::tie(gammaDelta, betaDelta) =
      normParamGradients(graph, actsWhitened, gradsIn, prog, partialsType, "",
                         normType);

  auto gradsOut =
    normGradients(graph, actsWhitened, gradsIn, invStdDev, gamma, prog,
                  partialsType, "", normType);

  normParamUpdate(graph, gammaDelta, betaDelta, learningRate, gamma, beta, prog,
                  "", normType);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActs = allocateHostMemoryForTensor(acts, "acts", graph,
                                                 uploadProg, downloadProg,
                                                 tmap);
  auto rawHostActsBN = allocateHostMemoryForTensor(actsBN, "actsBN", graph,
                                                   uploadProg, downloadProg,
                                                   tmap);
  auto rawHostActsBNInf = allocateHostMemoryForTensor(actsBNInf, "actsBNInf",
                                                      graph, uploadProg,
                                                      downloadProg, tmap);
  auto rawHostActsWhitened =
          allocateHostMemoryForTensor(actsWhitened, "actsWhitened",
                                      graph, uploadProg, downloadProg, tmap);
  auto rawHostGradsIn =
          allocateHostMemoryForTensor(gradsIn, "gradsIn", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostGradsOut =
          allocateHostMemoryForTensor(gradsOut, "gradsOut", graph, uploadProg,
                                      downloadProg, tmap);
  auto rawHostMean = allocateHostMemoryForTensor(mean, "mean", graph,
                                                 uploadProg, downloadProg,
                                                 tmap);
  auto rawHostInvStdDev = allocateHostMemoryForTensor(invStdDev, "invStdDev",
                                                      graph, uploadProg,
                                                      downloadProg, tmap);
  auto rawHostGamma = allocateHostMemoryForTensor(gamma, "gamma", graph,
                                                  uploadProg, downloadProg,
                                                  tmap);
  auto rawHostBeta = allocateHostMemoryForTensor(beta, "beta", graph,
                                                 uploadProg, downloadProg,
                                                 tmap);

  const unsigned numStatElems = isBatchNorm ? numActs : batchSize * numGroups;
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
  boost::multi_array<double, 1> hostMean(boost::extents[numStatElems]);
  boost::multi_array<double, 1> hostInvStdDev(boost::extents[numStatElems]);
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

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0); // Run.
  });

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
  boost::multi_array<double, 1> modelMean(boost::extents[numStatElems]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numStatElems]);

  poplibs_test::norm::normStatistics(hostActs, eps, unbiasedVarEstimate,
                                     modelMean, modelInvStdDev, normType);

  boost::multi_array<double, 2> modelActsBN(boost::extents[batchSize][numActs]);
  poplibs_test::norm::normalise(hostActs, modelGamma, modelBeta,
                                modelMean, modelInvStdDev, modelActsBN,
                                modelActsWhitened, normType);

  boost::multi_array<double, 2> modelGradsOut(boost::extents[batchSize]
                                                            [numActs]);
  poplibs_test::norm::normGradients(hostActsWhitened, hostGradsIn,
                                    modelInvStdDev, modelGamma,
                                    modelGradsOut, normType);
  poplibs_test::norm::normParamUpdate(modelActsWhitened, hostGradsIn,
                                      learningRate, modelGamma,
                                      modelBeta, normType);

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

  if (deviceType != DeviceType::Cpu && dumpProfile) {
    engine.printSummary(std::cout, OptionFlags{
      { "doLayerWiseBreakdown", "true" }
    });
  }
  return matchesModel;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  float eps;
  float learningRate;
  Type dataType;
  Type partialsType;
  unsigned tilesPerIPU;
  ShapeOption<unsigned> dims;
  std::string test;
  bool unbiasedVarEstimate = false;
  unsigned numGroups = 1;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("eps",
     po::value<float>(&eps)->required(),
     "eps")
    ("learning-rate",
     po::value<float>(&learningRate)->required(),
     "Learning Rate")
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
    ("dims",
     po::value<ShapeOption<unsigned>>(&dims)->required(),
     "Dimensions : {batch,height,width,channels} for conv, {batch,channels} "
     " for fc")
    ("unbiased-var-estimate",
     po::value<bool>(&unbiasedVarEstimate)->default_value(unbiasedVarEstimate),
     "Use unbiased variance estimate")
    ("num-groups",
     po::value<unsigned>(&numGroups)->default_value(numGroups),
     "Number of groups in group norm. Ignored for BN, LN and IN")
    ("test",
     po::value<std::string>(&test)->required(),
     "Test: BN-Conv | BN-Fc | GN-Conv | GN-Fc | IN-Conv | LN-Conv | "
     "LN-Fc ");
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
  bool isConv;
  unsigned groups;
  poplibs_test::norm::NormType normType;
  std::tie(normType, isConv, groups) = parseTestType(test, dims.val, numGroups);

  std::cerr << "\n Test " << test << " isConv " << isConv;
  std::cerr << " groups " << groups << " channels " << dims.val.back();

  if (isConv) {
    if (dims.val.size() != 4) {
      std::cerr << "error: convolution test must have tensor dimensions of 4";
      return 1;
    }

    auto matchesModel =
        normConv(deviceType, dims.val, eps, learningRate, tilesPerIPU, dataType,
                 unbiasedVarEstimate, partialsType, normType, groups,
                 dumpProfile);
    return matchesModel ? 0 : 1;
  } else {
    if (dims.val.size() != 2) {
      std::cerr << "error: fc test must have tensor dimensions of 4";
      return 1;
    }

    auto matchesModel =
        normFc(deviceType, dims.val, eps, learningRate, tilesPerIPU, dataType,
               unbiasedVarEstimate, partialsType, normType, groups,
               dumpProfile);
    return matchesModel ? 0 : 1;
  }
}
