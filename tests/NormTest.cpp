#include "TestDevice.hpp"
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <functional>
#include <iostream>
#include <poplar/Engine.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/Norms.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_test/exceptions.hpp>
#include <poplin/codelets.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popnn/LayerNorm.hpp>
#include <popnn/Norms.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
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

const OptionFlags options{
    {"target.workerStackSizeInBytes", "0x180"},
};

static std::pair<poplibs_test::norm::NormType, unsigned>
parseTestType(const std::string &testTypeString,
              const std::vector<std::size_t> &dims, unsigned numGroups) {
  if (testTypeString == "BN") {
    return std::make_pair(poplibs_test::norm::NormType::BatchNorm, numGroups);
  } else if (testTypeString == "GN") {
    return std::make_pair(poplibs_test::norm::NormType::GroupNorm, numGroups);
  } else if (testTypeString == "IN") {
    return std::make_pair(poplibs_test::norm::NormType::InstanceNorm, dims[1]);
  } else if (testTypeString == "LN") {
    return std::make_pair(poplibs_test::norm::NormType::LayerNorm, 1);
  } else {
    throw poplibs_test::poplibs_test_error("Invalid test type");
  }
}

static std::pair<Tensor, Tensor>
normStatistics(Graph &graph, const Tensor acts, float eps, Sequence &prog,
               bool unbiasedVarEstimate, const Type &partialsType,
               const std::string &debugPrefix, unsigned numGroups,
               poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
    return bn::batchNormStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                                   partialsType, debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
    return gn::groupNormStatistics(graph, acts, eps, prog, numGroups,
                                   unbiasedVarEstimate, partialsType,
                                   debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
    return ln::layerNormStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                                   partialsType, debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
    return in::instanceNormStatistics(
        graph, acts, eps, prog, unbiasedVarEstimate, partialsType, debugPrefix);
  }
  throw poplibs_test::poplibs_test_error("Invalid normType");
}

std::pair<Tensor, Tensor> normalise(Graph &graph, const Tensor &acts,
                                    const Tensor &gamma, const Tensor &beta,
                                    const Tensor &mean, const Tensor &iStdDev,
                                    Sequence &prog,
                                    const std::string &debugPrefix,
                                    poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
    return bn::batchNormalise(graph, acts, gamma, beta, mean, iStdDev, prog,
                              debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
    return gn::groupNormalise(graph, acts, gamma, beta, mean, iStdDev, prog,
                              debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
    return ln::layerNormalise(graph, acts, gamma, beta, mean, iStdDev, prog,
                              debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
    return in::instanceNormalise(graph, acts, gamma, beta, mean, iStdDev, prog,
                                 debugPrefix);
  }
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &acts, const Tensor &gradsIn,
                   const Tensor &mean, const Tensor &iStdDev, Sequence &prog,
                   const Type &partialsType, const std::string &debugPrefix,
                   poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
    return bn::batchNormParamGradients(graph, acts, gradsIn, mean, iStdDev,
                                       prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
    return gn::groupNormParamGradients(graph, acts, gradsIn, mean, iStdDev,
                                       prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
    return ln::layerNormParamGradients(graph, acts, gradsIn, mean, iStdDev,
                                       prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
    return in::instanceNormParamGradients(graph, acts, gradsIn, mean, iStdDev,
                                          prog, partialsType, debugPrefix);
  }
  throw poplibs_test::poplibs_test_error("Invalid normType");
}

static Tensor normGradients(Graph &graph, const Tensor &acts,
                            const Tensor &gradsIn, const Tensor &mean,
                            const Tensor &iStdDev, const Tensor &gamma,
                            Sequence &prog, const Type &partialsType,
                            const std::string &debugPrefix,
                            poplibs_test::norm::NormType normType) {
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
    return bn::batchNormGradients(graph, acts, gradsIn, mean, iStdDev, gamma,
                                  prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::GroupNorm:
    return gn::groupNormGradients(graph, acts, gradsIn, mean, iStdDev, gamma,
                                  prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::LayerNorm:
    return ln::layerNormGradients(graph, acts, gradsIn, mean, iStdDev, gamma,
                                  prog, partialsType, debugPrefix);
  case poplibs_test::norm::NormType::InstanceNorm:
    return in::instanceNormGradients(graph, acts, gradsIn, mean, iStdDev, gamma,
                                     prog, partialsType, debugPrefix);
  }
  throw poplibs_test::poplibs_test_error("Invalid normType");
}

static void normParamUpdate(poplar::Graph &graph,
                            const poplar::Tensor &gammaDelta,
                            const poplar::Tensor &betaDelta, float learningRate,
                            poplar::Tensor &gamma, poplar::Tensor &beta,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix,
                            poplibs_test::norm::NormType normType) {
  auto scale = graph.addConstant(beta.elementType(), {}, -learningRate);
  graph.setTileMapping(scale, 0);
  switch (normType) {
  case poplibs_test::norm::NormType::BatchNorm:
    bn::batchNormParamUpdate(graph, gammaDelta, betaDelta, scale, gamma, beta,
                             prog, debugPrefix);
    return;
  case poplibs_test::norm::NormType::GroupNorm:
    gn::groupNormParamUpdate(graph, gammaDelta, betaDelta, scale, gamma, beta,
                             prog, debugPrefix);
    return;
  case poplibs_test::norm::NormType::LayerNorm:
    ln::layerNormParamUpdate(graph, gammaDelta, betaDelta, scale, gamma, beta,
                             prog, debugPrefix);
    return;
  case poplibs_test::norm::NormType::InstanceNorm:
    in::instanceNormParamUpdate(graph, gammaDelta, betaDelta, scale, gamma,
                                beta, prog, debugPrefix);
    return;
  }
}

static bool normTest(const DeviceType &deviceType,
                     const std::vector<std::size_t> dims, float eps,
                     float learningRate, unsigned tilesPerIPU,
                     const Type &dataType, bool unbiasedVarEstimate,
                     const Type &partialsType,
                     poplibs_test::norm::NormType normType, unsigned numGroups,
                     bool dumpProfile) {
  assert(dims.size() >= 2);
  const auto batchSize = dims[0];
  const auto numChannels = dims[1];

  auto device = createTestDevice(deviceType, 1, tilesPerIPU);
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

  auto acts = graph.addVariable(dataType, actDims, "act");
  poputil::mapTensorLinearly(graph, acts);
  acts = acts.dimShufflePartial({acts.rank() - 1}, {1});

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

  auto gradsIn = graph.clone(actsBN);

  Tensor gammaDelta, betaDelta;
  std::tie(gammaDelta, betaDelta) = normParamGradients(
      graph, acts, gradsIn, mean, invStdDev, prog, partialsType, "", normType);
  auto gradsOut = normGradients(graph, acts, gradsIn, mean, invStdDev, gamma,
                                prog, partialsType, "", normType);

  normParamUpdate(graph, gammaDelta, betaDelta, learningRate, gamma, beta, prog,
                  "", normType);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  auto rawHostActs = allocateHostMemoryForTensor(
      acts, "acts", graph, uploadProg, downloadProg, tmap);
  auto rawHostActsBN = allocateHostMemoryForTensor(
      actsBN, "actsBN", graph, uploadProg, downloadProg, tmap);
  auto rawHostActsBNInf = allocateHostMemoryForTensor(
      actsBNInf, "actsBNInf", graph, uploadProg, downloadProg, tmap);
  auto rawHostGradsIn = allocateHostMemoryForTensor(
      gradsIn, "gradsIn", graph, uploadProg, downloadProg, tmap);
  auto rawHostGradsOut = allocateHostMemoryForTensor(
      gradsOut, "gradsOut", graph, uploadProg, downloadProg, tmap);
  auto rawHostMean = allocateHostMemoryForTensor(
      mean, "mean", graph, uploadProg, downloadProg, tmap);
  auto rawHostInvStdDev = allocateHostMemoryForTensor(
      invStdDev, "invStdDev", graph, uploadProg, downloadProg, tmap);
  auto rawHostGamma = allocateHostMemoryForTensor(
      gamma, "gamma", graph, uploadProg, downloadProg, tmap);
  auto rawHostBeta = allocateHostMemoryForTensor(
      beta, "beta", graph, uploadProg, downloadProg, tmap);
  unsigned numStatsElems = isBatchNorm ? numChannels : numGroups * batchSize;

  boost::multi_array<double, 3> hostActs(
      boost::extents[batchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostActsBN(
      boost::extents[batchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostActsBNInf(
      boost::extents[batchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostGradsIn(
      boost::extents[batchSize][numChannels][fieldSize]);
  boost::multi_array<double, 3> hostGradsOut(
      boost::extents[batchSize][numChannels][fieldSize]);
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
  copy(target, hostGamma, dataType, rawHostGamma.get());
  copy(target, hostBeta, dataType, rawHostBeta.get());
  copy(target, hostGradsIn, dataType, rawHostGradsIn.get());

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0); // Run.
  });

  copy(target, dataType, rawHostMean.get(), hostMean);
  copy(target, dataType, rawHostInvStdDev.get(), hostInvStdDev);
  copy(target, dataType, rawHostActsBN.get(), hostActsBN);
  copy(target, dataType, rawHostActsBNInf.get(), hostActsBNInf);
  copy(target, dataType, rawHostGradsOut.get(), hostGradsOut);
  copy(target, dataType, rawHostBeta.get(), hostBeta);
  copy(target, dataType, rawHostGamma.get(), hostGamma);

  bool matchesModel = true;

  boost::multi_array<double, 3> modelActsWhitened(
      boost::extents[batchSize][numChannels][fieldSize]);
  boost::multi_array<double, 1> modelMean(boost::extents[numStatsElems]);
  boost::multi_array<double, 1> modelInvStdDev(boost::extents[numStatsElems]);

  poplibs_test::norm::normStatistics(hostActs, eps, unbiasedVarEstimate,
                                     modelMean, modelInvStdDev, normType);

  boost::multi_array<double, 3> modelActsBN(
      boost::extents[batchSize][numChannels][fieldSize]);
  poplibs_test::norm::normalise(hostActs, modelGamma, modelBeta, modelMean,
                                modelInvStdDev, modelActsBN, modelActsWhitened,
                                normType);
  boost::multi_array<double, 3> modelGradsOut(
      boost::extents[batchSize][numChannels][fieldSize]);

  poplibs_test::norm::normGradients(modelActsWhitened, modelGradsIn,
                                    modelInvStdDev, modelGamma, modelGradsOut,
                                    normType);
  poplibs_test::norm::normParamUpdate(modelActsWhitened, modelGradsIn,
                                      learningRate, modelGamma, modelBeta,
                                      normType);

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
  matchesModel &= checkIsClose("actsBNInf", hostActsBNInf, modelActsBN,
                               relativeTolerance, absoluteTolerance);
  matchesModel &= checkIsClose("gradsOut", hostGradsOut, modelGradsOut,
                               relativeTolerance, absoluteTolerance);
  matchesModel &= checkIsClose("beta", hostBeta, modelBeta, relativeTolerance,
                               absoluteTolerance);
  matchesModel &= checkIsClose("gamma", hostGamma, modelGamma,
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
  float learningRate;
  Type dataType;
  Type partialsType;
  unsigned tilesPerIPU;
  ShapeOption<std::size_t> dims;
  std::string test;
  bool unbiasedVarEstimate = false;
  unsigned numGroups = 1;

  po::options_description desc("Options");
  // clang-format off
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
     po::value<ShapeOption<std::size_t>>(&dims)->required(),
     "Dimensions : {batch,channels, ....field....}, where field could be "
     "empty or have any dimension")
    ("unbiased-var-estimate",
     po::value<bool>(&unbiasedVarEstimate)->default_value(unbiasedVarEstimate),
     "Use unbiased variance estimate")
    ("num-groups",
     po::value<unsigned>(&numGroups)->default_value(numGroups),
     "Number of groups in group norm. Ignored for BN, LN and IN")
    ("norm-type",
     po::value<std::string>(&test)->required(),
     "Normalisation type: BN | GN | IN | LN");
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
  unsigned groups;
  poplibs_test::norm::NormType normType;
  std::tie(normType, groups) = parseTestType(test, dims.val, numGroups);

  std::cerr << "\n Test ";
  std::cerr << test << " groups " << groups << " channels " << dims.val[1];

  if (dims.val.size() < 2) {
    std::cerr << "error: norm test must have tensor dimensions of at least 2";
    return 1;
  }
  auto matchesModel = normTest(deviceType, dims.val, eps, learningRate,
                               tilesPerIPU, dataType, unbiasedVarEstimate,
                               partialsType, normType, groups, dumpProfile);
  return matchesModel ? 0 : 1;
}
