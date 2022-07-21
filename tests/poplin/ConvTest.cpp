// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvTest
#include "ConvUtilInternal.hpp"
#include "poputil/TileMapping.hpp"
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <random>

using namespace poplar;
using namespace poplin;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

BOOST_AUTO_TEST_CASE(MappingSplitOutChansSerially) {
  constexpr std::size_t numTiles = 16;
  constexpr std::size_t split = 4;
  constexpr std::size_t numOutChans = 16;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  static_assert(numOutChans % split == 0, "output channels must be evenly "
                                          "divisible by the serial split");
  const auto params =
      ConvParams(HALF, 4 /* batchSize */, {8, 8} /* inputFieldShape */,
                 {3, 3} /* kernelShape */, 16 /* inputChannels */,
                 numOutChans /* outputChannels */, 1 /* numConvGroups */);

  const OptionFlags options{
      {
          "planConstraints",
          R"({"0":{"partition":{"outChanSplit":{"serial":)" +
              std::to_string(split) + R"(}}}})",
      },
  };

  PlanningCache cache;
  const auto weights = createWeights(graph, params, "weights", options, &cache);

  std::stringstream ss;
  reportPlanInfo(ss, graph, params, options, &cache);
  BOOST_TEST_MESSAGE(ss.str());

  // Check each of the splits in the serial partition has the same mapping
  // for efficient dynamic slicing.
  const auto outChansPerSplit = numOutChans / split;
  const auto referenceMapping =
      graph.getTileMapping(weights.slice(0, outChansPerSplit, 1));
  for (std::size_t s = 1; s < split; ++s) {
    const auto slice =
        weights.slice(s * outChansPerSplit, (s + 1) * outChansPerSplit, 1);
    BOOST_CHECK(graph.getTileMapping(slice) == referenceMapping);
  }
}

void convolve(const boost::const_multi_array_ref<double, 3> &in,
              const boost::const_multi_array_ref<double, 4> &weights,
              const boost::const_multi_array_ref<double, 1> &biases,
              const boost::multi_array_ref<double, 3> &out,
              const ConvParams &params) {
  poplibs_test::conv::convolution(
      vectorConvert<unsigned>(params.inputFieldShape),
      params.inputTransform.truncationLower,
      params.inputTransform.truncationUpper, params.inputTransform.dilation,
      params.inputTransform.paddingLower, params.inputTransform.paddingUpper,
      params.inputTransform.flip, vectorConvert<unsigned>(params.kernelShape),
      params.kernelTransform.truncationLower,
      params.kernelTransform.truncationUpper, params.kernelTransform.dilation,
      params.kernelTransform.paddingLower, params.kernelTransform.paddingUpper,
      params.kernelTransform.flip, params.outputTransform.truncationLower,
      params.outputTransform.truncationUpper, params.outputTransform.stride,
      params.outputTransform.paddingLower, params.outputTransform.paddingUpper,
      in, weights, biases, out);
}

boost::multi_array<double, 3> createOut(const ConvParams &params) {
  boost::multi_array<double, 3> out(
      boost::extents[params.batchSize]
                    [params.numConvGroups * params.outputChannelsPerConvGroup]
                    [product(params.getOutputFieldShape())]);
  return out;
}

boost::multi_array<double, 1> createDummyBiases(const ConvParams &params) {
  boost::multi_array<double, 1> biases(
      boost::extents[params.numConvGroups * params.outputChannelsPerConvGroup]);
  std::fill(biases.data(), biases.data() + biases.num_elements(), 0.0);
  return biases;
}

ConvParams createParams() {
  ConvParams p;

  // Random sensible convolution parameters
  p.numConvGroups = 1;
  p.batchSize = 2;
  p.inputChannelsPerConvGroup = 2;
  p.outputChannelsPerConvGroup = 3;
  p.inputType = HALF;
  p.outputType = FLOAT;

  p.inputFieldShape = {4, 4};
  p.inputTransform.truncationLower = {0, 0};
  p.inputTransform.truncationUpper = {0, 0};
  p.inputTransform.dilation = {1, 1};
  p.inputTransform.paddingLower = {0, 0};
  p.inputTransform.paddingUpper = {0, 0};
  p.inputTransform.flip = {false, false};

  p.kernelShape = {1, 1};
  p.kernelTransform.truncationLower = {0, 0};
  p.kernelTransform.truncationUpper = {0, 0};
  p.kernelTransform.dilation = {1, 1};
  p.kernelTransform.paddingLower = {0, 0};
  p.kernelTransform.paddingUpper = {0, 0};
  p.kernelTransform.flip = {false, false};

  p.outputTransform.truncationLower = {0, 0};
  p.outputTransform.truncationUpper = {0, 0};
  p.outputTransform.stride = {1, 1};
  p.outputTransform.paddingLower = {0, 0};
  p.outputTransform.paddingUpper = {0, 0};

  return p;
}

multiconv::ConvolutionArgs
createConvolutionArgs(Graph &graph, const CanonicalConvParams &cp) {
  multiconv::ConvolutionArgs ca;
  ca.params = *cp;
  poplin::PlanningCache cache;
  OptionFlags convOptions;
  ca.inputs =
      poplin::createInput(graph, ca.params, "inputs", convOptions, &cache);
  ca.weights =
      poplin::createWeights(graph, ca.params, "weights", convOptions, &cache);
  return ca;
}

std::vector<boost::multi_array<double, 3>> combineAndRunConvolution(
    TestDevice &device, Graph &graph,
    const std::vector<CanonicalConvParams> &convParams,
    const std::vector<boost::multi_array<double, 3>> &hostIns,
    const std::vector<boost::multi_array<double, 4>> &hostWeights) {
  assert(convParams.size() == hostIns.size());
  assert(convParams.size() == hostWeights.size());

  // Allocate host memory for input and weight tensors
  std::vector<multiconv::ConvolutionArgs> convolutionArgs;
  poplar::program::Sequence uploadProg, downloadProg;
  static std::vector<std::pair<std::string, HostMemory>> tmap;
  std::vector<std::unique_ptr<char[]>> rawHostIns, rawHostWeights;
  auto target = device.getTarget();
  for (unsigned i(0); i < convParams.size(); ++i) {
    const auto ca = createConvolutionArgs(graph, convParams[i]);
    convolutionArgs.push_back(ca);
    rawHostIns.push_back(
        allocateHostMemoryForTensor(ca.inputs, "inputs" + std::to_string(i),
                                    graph, uploadProg, downloadProg, tmap));
    rawHostWeights.push_back(
        allocateHostMemoryForTensor(ca.weights, "weights" + std::to_string(i),
                                    graph, uploadProg, downloadProg, tmap));
    copy(target, hostIns[i], convParams[i]->inputType, rawHostIns.back().get());
    copy(target, hostWeights[i], convParams[i]->inputType,
         rawHostWeights.back().get());
  }

  // Combine convolution parameters and tensors
  auto argsWithConvOptions = convertToConvOptions(graph, convolutionArgs);
  BOOST_CHECK(poplin::canBeCombined(argsWithConvOptions));
  auto ca = poplin::combine(argsWithConvOptions);

  // Create a single convolution
  poplar::program::Sequence prog;
  poplin::PlanningCache cache;
  auto out = poplin::convolution(graph, ca.inputs, ca.weights, *ca.params,
                                 false, prog, "conv", {}, &cache);
  // Split the result tensor
  auto outs = poplin::splitOutput(convParams, out);
  assert(outs.size() == convParams.size());
  // Allocate host memory for split output tensors
  std::vector<std::unique_ptr<char[]>> rawHostOuts;
  for (unsigned i(0); i < outs.size(); ++i) {
    rawHostOuts.push_back(
        allocateHostMemoryForTensor(outs[i], "out" + std::to_string(i), graph,
                                    uploadProg, downloadProg, tmap));
  }

  Engine e(graph, poplar::program::Sequence{uploadProg, prog, downloadProg});
  attachStreams(e, tmap);

  // Run convolution and extract result
  device.bind([&](const Device &d) { e.loadAndRun(d); });
  std::vector<boost::multi_array<double, 3>> hostOuts;
  for (unsigned i(0); i < convParams.size(); ++i) {
    const auto cp = convParams[i];
    hostOuts.emplace_back(
        boost::extents[cp->batchSize]
                      [cp->numConvGroups * cp->outputChannelsPerConvGroup]
                      [product(cp->getOutputFieldShape())]);
    copy(target, cp->outputType, rawHostOuts[i].get(), hostOuts.back());
  }
  return hostOuts;
}

void runAndCheckConvolution(
    TestDevice &device, Graph &graph,
    const std::vector<CanonicalConvParams> &convParams) {
  // Write random values to input tensors
  std::vector<boost::multi_array<double, 3>> hostIns;
  std::vector<boost::multi_array<double, 4>> hostWeights;
  std::mt19937 randomEngine;
  auto target = device.getTarget();
  for (const auto &ccp : convParams) {
    const auto &p = *ccp;
    hostIns.emplace_back(
        boost::extents[p.batchSize]
                      [p.numConvGroups * p.inputChannelsPerConvGroup]
                      [product(p.inputFieldShape)]);
    hostWeights.emplace_back(
        boost::extents[p.numConvGroups][p.outputChannelsPerConvGroup]
                      [p.inputChannelsPerConvGroup][product(p.kernelShape)]);
    writeRandomValues(target, p.inputType, hostIns.back(), -1.0, +5.0,
                      randomEngine);
    writeRandomValues(target, p.inputType, hostWeights.back(), -1.0, +7.0,
                      randomEngine);
  }

  auto hostOuts =
      combineAndRunConvolution(device, graph, convParams, hostIns, hostWeights);
  assert(hostOuts.size() == convParams.size());

  // Compare results to model
  for (unsigned i(0); i < convParams.size(); ++i) {
    auto params = convParams[i];
    auto hostBiases = createDummyBiases(*params);
    auto modelOut = createOut(*params);
    convolve(hostIns[i], hostWeights[i], hostBiases, modelOut, *params);
    double absoluteTolerance = params->outputType == FLOAT ? 1e-6 : 1e-5;
    double relativeTolerance = params->outputType == FLOAT ? 0.01 : 0.1;
    bool isClose = checkIsClose("combined_conv", hostOuts[i], modelOut,
                                relativeTolerance, absoluteTolerance);
    BOOST_CHECK(isClose);
  }
}

BOOST_AUTO_TEST_CASE(MultiConvCanBeCombined) {
  auto createArgsPair = [](Graph &graph, const auto &modify) {
    auto cp1 = createParams();
    auto cp2 = createParams();
    modify(cp1, cp2);

    auto ca1 = createConvolutionArgs(graph, cp1);
    auto ca2 = createConvolutionArgs(graph, cp2);
    return convertToConvOptions(graph, {ca1, ca2});
  };
  auto device = createTestDevice(TEST_TARGET, 1, 16);
  Graph graph(device.getTarget());
  {
    auto args =
        createArgsPair(graph, [](ConvParams &first, ConvParams &second) {
          first.batchSize = 1;
          second.batchSize = 2;
        });
    BOOST_CHECK(!poplin::canBeCombined(args));
  }
  {
    auto args =
        createArgsPair(graph, [](ConvParams &first, ConvParams &second) {
          first.inputFieldShape = {1, 1};
          second.inputFieldShape = {2, 2};
        });
    BOOST_CHECK(!poplin::canBeCombined(args));
  }
  {
    auto args =
        createArgsPair(graph, [](ConvParams &first, ConvParams &second) {
          first.kernelTransform.paddingLower = {0, 0};
          second.kernelTransform.paddingLower = {1, 1};
        });
    BOOST_CHECK(!poplin::canBeCombined(args));
  }
  {
    auto args = createArgsPair(graph, [](const auto &, const auto &) {});
    args[0].options.interIpuPartialsType = poplar::FLOAT;
    args[1].options.interIpuPartialsType = poplar::HALF;
    BOOST_CHECK(!poplin::canBeCombined(args));
  }
}

BOOST_AUTO_TEST_CASE(MultiConvCombination) {
  auto device = createTestDevice(TEST_TARGET, 1, 16);
  Graph graph(device.getTarget());
  poplin::addCodelets(graph);

  auto cp1 = createParams();
  auto cp2 = createParams();
  auto cp3 = createParams();
  cp2.numConvGroups = 2;
  cp3.numConvGroups = 3;

  runAndCheckConvolution(device, graph, {cp1, cp2, cp3});
}

const auto printMapping = [](const auto &mapping) {
  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    std::stringstream ss;
    ss << "tile = " << tile << ", intervals=[ ";
    for (const auto &i : mapping[tile])
      ss << i << " ";
    ss << "]";

    BOOST_TEST_MESSAGE(ss.str());
  }
};

const auto checkMappingEntirelyOneTile = [](const Graph &graph, const Tensor &t,
                                            const unsigned i) {
  auto mapping = graph.getTileMapping(t);

  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    if (tile == i) {
      BOOST_TEST(!mapping[tile].empty());
    } else {
      BOOST_TEST(mapping[tile].empty());
    }
  }
};

const auto checkSplitTensorsMappedToSeparateTiles =
    [](const Graph &graph, const std::vector<Tensor> &result) {
      BOOST_TEST(result.size() == 3);

      for (unsigned i = 0; i < result.size(); ++i) {
        std::stringstream ss0;
        ss0 << "result << " << i << " ";
        result[i].outputRegions(ss0);
        BOOST_TEST_MESSAGE(ss0.str());

        // the first conv params should be entirely mapped on tile 0, etc.
        checkMappingEntirelyOneTile(graph, result[i], i);
      }
    };

BOOST_AUTO_TEST_CASE(SplitOutput) {
  constexpr static auto batch = 2;
  constexpr static auto outChans = 3;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, batch, {1}, {1}, 1, outChans, convGroups);
  };

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  // out shape should [B][G * Co]...
  //   where G*Co = (4+5+6)*3 = 45
  const std::vector<CanonicalConvParams> params{
      makeConvParams(4), makeConvParams(5), makeConvParams(6)};

  const auto uut =
      graph.addVariable(poplar::HALF, {batch, (4 + 5 + 6) * outChans, 1});

  // map each ConvParam to a separate tile.
  const unsigned gDim = 1;
  graph.setTileMapping(uut.slice(0, 4 * outChans, gDim), 0);
  graph.setTileMapping(uut.slice(4 * outChans, (4 + 5) * outChans, gDim), 1);
  graph.setTileMapping(
      uut.slice((4 + 5) * outChans, (4 + 5 + 6) * outChans, gDim), 2);
  printMapping(graph.getTileMapping(uut));

  // split the output and verify that each one is entirely mapped to the
  // expected tile.
  auto result = splitOutput(params, uut);
  checkSplitTensorsMappedToSeparateTiles(graph, result);
}

BOOST_AUTO_TEST_CASE(SplitInput) {
  constexpr static auto batch = 2;
  constexpr static auto inChans = 3;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, batch, {1}, {1}, inChans, 1, convGroups);
  };

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  // out shape should [B][G * Ci]...
  //   where G*Co = (4+5+6)*3 = 45
  const std::vector<CanonicalConvParams> params{
      makeConvParams(4), makeConvParams(5), makeConvParams(6)};

  const auto uut =
      graph.addVariable(poplar::HALF, {batch, (4 + 5 + 6) * inChans, 1});

  // map each ConvParam to a separate tile.
  const unsigned gDim = 1;
  graph.setTileMapping(uut.slice(0, 4 * inChans, gDim), 0);
  graph.setTileMapping(uut.slice(4 * inChans, (4 + 5) * inChans, gDim), 1);
  graph.setTileMapping(
      uut.slice((4 + 5) * inChans, (4 + 5 + 6) * inChans, gDim), 2);
  printMapping(graph.getTileMapping(uut));

  // split the input and verify that each one is entirely mapped to the
  // expected tile.
  auto result = splitInput(params, uut);
  checkSplitTensorsMappedToSeparateTiles(graph, result);
}

BOOST_AUTO_TEST_CASE(SplitWeights) {
  constexpr static auto inChans = 3;
  constexpr static auto outChans = 4;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, 1, {1}, {1}, inChans, outChans, convGroups);
  };

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  // weights shape should [G][Ci][Co]...
  const std::vector<CanonicalConvParams> params{
      makeConvParams(4), makeConvParams(5), makeConvParams(6)};

  const auto uut =
      graph.addVariable(poplar::HALF, {4 + 5 + 6, inChans, outChans, 1});

  // map each ConvParam to a separate tile.
  const unsigned gDim = 0;
  graph.setTileMapping(uut.slice(0, 4, gDim), 0);
  graph.setTileMapping(uut.slice(4, 4 + 5, gDim), 1);
  graph.setTileMapping(uut.slice(4 + 5, 4 + 5 + 6, gDim), 2);
  printMapping(graph.getTileMapping(uut));

  // split the input and verify that each one is entirely mapped to the
  // expected tile.
  auto result = splitWeights(params, uut);
  checkSplitTensorsMappedToSeparateTiles(graph, result);
}

BOOST_AUTO_TEST_CASE(CombineCreateTensorArgs) {
  constexpr static auto batch = 2;
  constexpr static auto inChans = 3;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, batch, {1}, {1}, inChans, 1, convGroups);
  };

  auto device = createTestDevice(TEST_TARGET);
  ConvOptions options{};
  options.pass = Pass::FC_TRAINING_WU;

  std::vector<multiconv::internal::CreateTensorArgs> args{
      {makeConvParams(4), options, "four"},
      {makeConvParams(5), options, "five"},
      {makeConvParams(6), options, "six"}};
  auto result = combine(args);

  BOOST_TEST(*result.params == makeConvParams(4 + 5 + 6));
  BOOST_TEST(result.options == options);
  // for now only the first name is preserved.
  BOOST_TEST(result.name == "four");
}

BOOST_AUTO_TEST_CASE(CombineConvolutionArgs) {
  constexpr static auto batch = 2;
  constexpr static auto inChans = 3;
  constexpr static auto outChans = 4;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, batch, {1}, {1}, inChans, outChans,
                      convGroups);
  };

  auto device = createTestDevice(TEST_TARGET, 1, 8);
  Graph graph(device.getTarget());

  const auto makeInput = [&graph](unsigned convGroups) {
    // acts external shape: [N][G * C]...
    auto t = graph.addVariable(poplar::HALF, {batch, convGroups * inChans, 1});
    graph.setTileMapping(t, convGroups);
    return t;
  };

  const auto makeWeights = [&graph](unsigned convGroups) {
    // weights external shape: [G][Co][Ci]...
    auto t =
        graph.addVariable(poplar::HALF, {convGroups, outChans, inChans, 1});
    graph.setTileMapping(t, convGroups);
    return t;
  };

  ConvOptions options{};
  options.pass = Pass::FC_TRAINING_WU;

  std::vector<multiconv::internal::ConvolutionArgs> args{
      {makeInput(5), makeWeights(5), makeConvParams(5), options},
      {makeInput(6), makeWeights(6), makeConvParams(6), options},
      {makeInput(7), makeWeights(7), makeConvParams(7), options},
  };

  auto result = combine(args);
  BOOST_TEST(*result.params == makeConvParams(5 + 6 + 7));
  BOOST_TEST(result.options == options);

  const unsigned iDim = 1;
  checkMappingEntirelyOneTile(graph, result.inputs.slice(0, 5 * inChans, iDim),
                              5);
  checkMappingEntirelyOneTile(
      graph, result.inputs.slice(5 * inChans, (5 + 6) * inChans, iDim), 6);
  checkMappingEntirelyOneTile(
      graph,
      result.inputs.slice((5 + 6) * inChans, (5 + 6 + 7) * inChans, iDim), 7);

  const unsigned wDim = 0;
  checkMappingEntirelyOneTile(graph, result.weights.slice(0, 5, wDim), 5);
  checkMappingEntirelyOneTile(graph, result.weights.slice(5, 5 + 6, wDim), 6);
  checkMappingEntirelyOneTile(graph,
                              result.weights.slice(5 + 6, 5 + 6 + 7, wDim), 7);
}

BOOST_AUTO_TEST_CASE(CombineCalculateWeightDeltasArgs) {
  constexpr static auto batch = 2;
  constexpr static auto inChans = 3;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, batch, {1}, {1}, inChans, 1, convGroups);
  };

  auto device = createTestDevice(TEST_TARGET, 1, 8);
  Graph graph(device.getTarget());

  const auto makeInput = [&graph](unsigned convGroups) {
    // acts external shape: [N][G * C]...
    auto t = graph.addVariable(poplar::HALF, {batch, convGroups * inChans, 1});
    graph.setTileMapping(t, convGroups);
    return t;
  };

  ConvOptions options{};
  options.pass = Pass::FC_TRAINING_WU;

  std::vector<multiconv::internal::CalculateWeightDeltasArgs> args{
      {makeInput(5), makeInput(5), makeConvParams(5), options},
      {makeInput(6), makeInput(6), makeConvParams(6), options},
      {makeInput(7), makeInput(7), makeConvParams(7), options},
  };

  auto result = combine(args);
  BOOST_TEST(*result.params == makeConvParams(5 + 6 + 7));
  BOOST_TEST(result.options == options);

  const unsigned d = 1;
  checkMappingEntirelyOneTile(graph, result.zDeltas.slice(0, 5 * inChans, d),
                              5);
  checkMappingEntirelyOneTile(
      graph, result.zDeltas.slice(5 * inChans, (5 + 6) * inChans, d), 6);
  checkMappingEntirelyOneTile(
      graph, result.zDeltas.slice((5 + 6) * inChans, (5 + 6 + 7) * inChans, d),
      7);

  checkMappingEntirelyOneTile(graph,
                              result.activations.slice(0, 5 * inChans, d), 5);
  checkMappingEntirelyOneTile(
      graph, result.activations.slice(5 * inChans, (5 + 6) * inChans, d), 6);
  checkMappingEntirelyOneTile(
      graph,
      result.activations.slice((5 + 6) * inChans, (5 + 6 + 7) * inChans, d), 7);
}

BOOST_AUTO_TEST_CASE(CombineConvWeightUpdateArgs) {
  constexpr static auto batch = 2;
  constexpr static auto inChans = 3;
  constexpr static auto outChans = 4;

  const auto makeConvParams = [](unsigned convGroups) {
    return ConvParams(poplar::HALF, batch, {1}, {1}, inChans, outChans,
                      convGroups);
  };

  auto device = createTestDevice(TEST_TARGET, 1, 8);
  Graph graph(device.getTarget());

  const auto makeInput = [&graph](unsigned convGroups) {
    // acts external shape: [N][G * C]...
    auto t = graph.addVariable(poplar::HALF, {batch, convGroups * inChans, 1});
    graph.setTileMapping(t, convGroups);
    return t;
  };

  const auto makeWeights = [&graph](unsigned convGroups) {
    // weights external shape: [G][Co][Ci]...
    auto t =
        graph.addVariable(poplar::HALF, {convGroups, outChans, inChans, 1});
    graph.setTileMapping(t, convGroups);
    return t;
  };

  ConvOptions options{};
  options.pass = Pass::FC_TRAINING_WU;

  std::vector<multiconv::internal::ConvWeightUpdateArgs<float>> args{
      {makeInput(5), makeWeights(5), makeInput(5), 15, makeConvParams(5),
       options},
      {makeInput(6), makeWeights(6), makeInput(6), 16, makeConvParams(6),
       options},
      {makeInput(7), makeWeights(7), makeInput(7), 17, makeConvParams(7),
       options},
  };

  auto result = combine(args);
  // for now only the first scale is preserved... obviously needs fixing before
  // multi-convs are complete.
  BOOST_TEST(result.scale == 15);
  BOOST_TEST(*result.params == makeConvParams(5 + 6 + 7));
  BOOST_TEST(result.options == options);

  const unsigned iDim = 1;
  checkMappingEntirelyOneTile(graph, result.zDeltas.slice(0, 5 * inChans, iDim),
                              5);
  checkMappingEntirelyOneTile(
      graph, result.zDeltas.slice(5 * inChans, (5 + 6) * inChans, iDim), 6);
  checkMappingEntirelyOneTile(
      graph,
      result.zDeltas.slice((5 + 6) * inChans, (5 + 6 + 7) * inChans, iDim), 7);

  checkMappingEntirelyOneTile(
      graph, result.activations.slice(0, 5 * inChans, iDim), 5);
  checkMappingEntirelyOneTile(
      graph, result.activations.slice(5 * inChans, (5 + 6) * inChans, iDim), 6);
  checkMappingEntirelyOneTile(
      graph,
      result.activations.slice((5 + 6) * inChans, (5 + 6 + 7) * inChans, iDim),
      7);

  const unsigned wDim = 0;
  checkMappingEntirelyOneTile(graph, result.weights.slice(0, 5, wDim), 5);
  checkMappingEntirelyOneTile(graph, result.weights.slice(5, 5 + 6, wDim), 6);
  checkMappingEntirelyOneTile(graph,
                              result.weights.slice(5 + 6, 5 + 6 + 7, wDim), 7);
}

BOOST_AUTO_TEST_CASE(ReadConvParamsFromJSON) {
  std::stringstream is;
  is << R"(
    {
      "dataType": "float",
      "batchSize": 1,
      "numConvGroups": 2,
      "inputChannelsPerConvGroup": 3,
      "outputChannelsPerConvGroup": 4,
      "kernelShape": [50, 60],
      "inputFieldShape": [70, 80],
      "inputTransform": {
        "truncationLower": [1, 2],
        "paddingUpper": [3, 4],
        "flip": [true, false]
      },
      "outputTransform": {
        "stride": [5, 6]
      }
    }
  )";

  ConvParams uut{};
  is >> uut;

  ConvParams expected(FLOAT, 1, {70, 80}, {50, 60}, 3, 4, 2);
  expected.inputTransform.truncationLower = {1, 2};
  expected.inputTransform.paddingUpper = {3, 4};
  expected.inputTransform.flip = {true, false};
  expected.outputTransform.stride = {5, 6};
  BOOST_TEST(uut == expected);
}

void validateOptions(void (*optionsValidationFn)(const poplar::OptionFlags &)) {
  OptionFlags optionsValid;
  optionsValid.set({{"partialsType", "half"}});
  BOOST_CHECK_NO_THROW(optionsValidationFn(optionsValid));

  OptionFlags optionsInvalidValue;
  optionsInvalidValue.set({{"partialsType", "invalid_value"}});
  BOOST_CHECK_THROW(optionsValidationFn(optionsInvalidValue),
                    poplar::invalid_option);

  OptionFlags optionsInvalidKey;
  optionsInvalidKey.set({{"invalid_key", "half"}});
  BOOST_CHECK_THROW(optionsValidationFn(optionsInvalidKey),
                    poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ValidateConvOptions) {
  validateOptions(convolutionValidateOptions);
}

BOOST_AUTO_TEST_CASE(ValidateMatMulOptions) {
  validateOptions(matmulValidateOptions);
}
