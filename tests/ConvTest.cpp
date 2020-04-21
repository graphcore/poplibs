// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvTest
#include "ConvUtilInternal.hpp"
#include "TestDevice.hpp"
#include "poputil/TileMapping.hpp"
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/codelets.hpp>
#include <random>

using namespace poplar;
using namespace poplin;
using namespace poplibs_test::util;

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

  p.kernelShape = {2, 2};
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

multiconv::ConvolutionArgs createConvolutionArgs(Graph &graph,
                                                 const ConvParams &cp) {
  multiconv::ConvolutionArgs ca;
  ca.params = cp;
  poplin::PlanningCache cache;
  OptionFlags convOptions;
  ca.inputs =
      poplin::createInput(graph, ca.params, "inputs", convOptions, &cache);
  ca.weights =
      poplin::createWeights(graph, ca.params, "weights", convOptions, &cache);
  return ca;
}

std::vector<boost::multi_array<double, 3>> combineAndRunConvolution(
    TestDevice &device, Graph &graph, const std::vector<ConvParams> &convParams,
    const std::vector<boost::multi_array<double, 3>> &hostIns,
    const std::vector<boost::multi_array<double, 4>> &hostWeights) {
  assert(convParams.size() == hostIns.size());
  assert(convParams.size() == hostWeights.size());

  // Allocate host memory for input and weight tensors
  std::vector<multiconv::ConvolutionArgs> convolutionArgs;
  poplar::program::Sequence uploadProg, downloadProg;
  static std::vector<std::pair<std::string, char *>> tmap;
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
    copy(target, hostIns[i], convParams[i].inputType, rawHostIns.back().get());
    copy(target, hostWeights[i], convParams[i].inputType,
         rawHostWeights.back().get());
  }

  // Combine convolution parameters and tensors
  BOOST_CHECK(poplin::canBeCombined(convolutionArgs));
  auto ca = poplin::combine(convolutionArgs);

  // Create a single convolution
  poplar::program::Sequence prog;
  poplin::PlanningCache cache;
  auto out = poplin::convolution(graph, ca.inputs, ca.weights, ca.params, false,
                                 prog, "conv", {}, &cache);
  // Split the result tensor
  auto outs = poplin::split(convParams, out);
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
        boost::extents[cp.batchSize]
                      [cp.numConvGroups * cp.outputChannelsPerConvGroup]
                      [product(cp.getOutputFieldShape())]);
    copy(target, cp.outputType, rawHostOuts[i].get(), hostOuts.back());
  }
  return hostOuts;
}

void runAndCheckConvolution(TestDevice &device, Graph &graph,
                            const std::vector<ConvParams> &convParams) {
  // Write random values to input tensors
  std::vector<boost::multi_array<double, 3>> hostIns;
  std::vector<boost::multi_array<double, 4>> hostWeights;
  std::mt19937 randomEngine;
  auto target = device.getTarget();
  for (const auto p : convParams) {
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
    auto hostBiases = createDummyBiases(params);
    auto modelOut = createOut(params);
    convolve(hostIns[i], hostWeights[i], hostBiases, modelOut, params);
    double absoluteTolerance = params.outputType == FLOAT ? 1e-6 : 1e-5;
    double relativeTolerance = params.outputType == FLOAT ? 0.01 : 0.1;
    bool isClose = checkIsClose("combined_conv", hostOuts[i], modelOut,
                                relativeTolerance, absoluteTolerance);
    BOOST_CHECK(isClose);
  }
}

BOOST_AUTO_TEST_CASE(MultiConvCanBeCombined) {
  auto device = createTestDevice(TEST_TARGET, 1, 16);
  Graph graph(device.getTarget());
  {
    auto cp1 = createParams();
    auto cp2 = createParams();
    auto ca1 = createConvolutionArgs(graph, cp1);
    auto ca2 = createConvolutionArgs(graph, cp2);
    ca1.params.batchSize = 1;
    ca2.params.batchSize = 2;
    BOOST_CHECK(!poplin::canBeCombined({ca1, ca2}));
  }
  {
    auto cp1 = createParams();
    auto cp2 = createParams();
    auto ca1 = createConvolutionArgs(graph, cp1);
    auto ca2 = createConvolutionArgs(graph, cp2);
    ca1.params.inputFieldShape = {1, 1};
    ca2.params.inputFieldShape = {2, 2};
    BOOST_CHECK(!poplin::canBeCombined({ca1, ca2}));
  }
  {
    auto cp1 = createParams();
    auto cp2 = createParams();
    auto ca1 = createConvolutionArgs(graph, cp1);
    auto ca2 = createConvolutionArgs(graph, cp2);
    ca1.params.kernelTransform.paddingLower = {0, 0};
    ca2.params.kernelTransform.paddingLower = {1, 1};
    BOOST_CHECK(!poplin::canBeCombined({ca1, ca2}));
  }
  {
    auto cp1 = createParams();
    auto cp2 = createParams();
    auto ca1 = createConvolutionArgs(graph, cp1);
    auto ca2 = createConvolutionArgs(graph, cp2);
    ca1.options.set("opt1", "v1");
    ca2.options.set("opt1", "v2");
    BOOST_CHECK(!poplin::canBeCombined({ca1, ca2}));
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
