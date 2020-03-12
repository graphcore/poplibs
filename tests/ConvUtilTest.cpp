// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvUtilTest
#include "ConvUtilInternal.hpp"
#include "TestDevice.hpp"
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/test/unit_test.hpp>

using namespace poplar;

BOOST_AUTO_TEST_CASE(getInputRangeFlipActsAndWeights) {
  poplin::ConvParams params{
      poplar::FLOAT, // data type,
      1,             // batch size
      {3, 1},        // input size
      {5, 1},        // kernel size
      1,             // input channels
      1,             // output channels
      1              // conv groups
  };
  params.inputTransform.paddingLower = {2, 0};
  params.inputTransform.paddingUpper = {2, 0};
  params.inputTransform.flip = {true, false};
  params.kernelTransform.flip = {true, false};

  auto inRange = poplin::getInputRange(0, {0, 2}, 1, params);
  BOOST_CHECK_EQUAL(inRange.first, 0);
  BOOST_CHECK_EQUAL(inRange.second, 2);
}

BOOST_AUTO_TEST_CASE(getKernelRangeTruncateInput) {
  poplin::ConvParams params{
      poplar::FLOAT, // data type,
      1,             // batch size
      {3},           // input size
      {4},           // kernel size
      1,             // input channels
      1,             // output channels
      1              // conv groups
  };
  params.inputTransform.truncationUpper = {1};
  params.inputTransform.dilation = {2};
  params.inputTransform.paddingLower = {1};

  auto kernelRange = poplin::getKernelRange(0, {0, 1}, {0, 3}, params);
  BOOST_CHECK_EQUAL(kernelRange.first, 1);
  BOOST_CHECK_EQUAL(kernelRange.second, 4);
}

BOOST_AUTO_TEST_CASE(SplitActivationIntoGroups) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());

  const auto g = 4;
  const auto n = 100;
  const auto h = 101;
  const auto w = 102;
  const auto c = 16;
  const auto uut = graph.addVariable(HALF, {g, n, h, w, c});

  const auto cgpg = 4;
  const auto cpg = 8;

  BOOST_TEST(g % cgpg == 0);
  BOOST_TEST(c % cpg == 0);

  const auto result = poplin::splitActivationIntoGroups(uut, cgpg, cpg);
  const auto &resultShape = result.shape();
  BOOST_TEST(resultShape.size() == uut.shape().size() + 2);

  std::vector<std::size_t> expected{g / cgpg, c / cpg, n, h, w, cgpg, cpg};
  BOOST_TEST(resultShape == expected, boost::test_tools::per_element());

  const auto t = poplin::unsplitActivationFromGroups(result);
  BOOST_TEST(t.shape() == uut.shape(), boost::test_tools::per_element());
  BOOST_TEST(t.getVarRegions() == uut.getVarRegions());
}

BOOST_AUTO_TEST_CASE(SplitWeightsIntoGroups) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());

  const auto g = 4;
  const auto h = 101;
  const auto w = 102;
  const auto co = 15;
  const auto ci = 16;
  const auto uut = graph.addVariable(HALF, {g, h, w, co, ci});

  const auto cgpg = 4;
  const auto copg = 3;
  const auto cipg = 8;

  BOOST_TEST(g % cgpg == 0);
  BOOST_TEST(co % copg == 0);
  BOOST_TEST(ci % cipg == 0);

  const auto result = poplin::splitWeightsIntoGroups(uut, cgpg, cipg, copg);
  const auto &resultShape = result.shape();
  BOOST_TEST(resultShape.size() == uut.shape().size() + 3);

  std::vector<std::size_t> expected{g / cgpg, co / copg, ci / cipg, h,
                                    w,        cgpg,      copg,      cipg};
  BOOST_TEST(resultShape == expected, boost::test_tools::per_element());

  const auto t = poplin::unsplitWeightsFromGroups(result);
  BOOST_TEST(t.shape() == uut.shape(), boost::test_tools::per_element());
  BOOST_TEST(t.getVarRegions() == uut.getVarRegions());
}

BOOST_AUTO_TEST_CASE(DetectActivationChannelGrouping) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());

  const auto numConvGroupGroups = 5;
  const auto numInChanGroups = 7;

  const auto n = 100;
  const auto h = 101;
  const auto w = 102;

  const auto convGroupsPerGroup = 4;
  const auto inChansPerGroup = 16;

  // how the weights are laid out in memory.
  auto t = graph.addVariable(HALF, {numConvGroupGroups, numInChanGroups, n, h,
                                    w, convGroupsPerGroup, inChansPerGroup});
  graph.setTileMapping(t, 0);

  // reshape to internal shape.
  t = poplin::unsplitActivationFromGroups(t);

  auto detectedGrouping = poplin::detectChannelGrouping(graph, t);
  BOOST_CHECK_EQUAL(convGroupsPerGroup, detectedGrouping.convGroupsPerGroup);
  BOOST_CHECK_EQUAL(inChansPerGroup, detectedGrouping.chansPerGroup);
}

BOOST_AUTO_TEST_CASE(DetectWeightsChannelGrouping) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());

  const auto numConvGroupGroups = 5;
  const auto numOutChanGroups = 6;
  const auto numInChanGroups = 7;

  const auto kh = 10;
  const auto kw = 11;

  const auto convGroupsPerGroup = 4;
  const auto outChansPerGroup = 8;
  const auto inChansPerGroup = 16;

  // how the weights are laid out in memory.
  auto t = graph.addVariable(HALF, {numConvGroupGroups, numOutChanGroups,
                                    numInChanGroups, kh, kw, convGroupsPerGroup,
                                    outChansPerGroup, inChansPerGroup});
  graph.setTileMapping(t, 0);

  // reshape to internal shape.
  t = poplin::unsplitWeightsFromGroups(t);

  auto detectedGrouping = poplin::detectWeightsChannelGrouping(graph, t);
  BOOST_CHECK_EQUAL(convGroupsPerGroup, detectedGrouping.convGroupsPerGroup);
  BOOST_CHECK_EQUAL(outChansPerGroup, detectedGrouping.outChansPerGroup);
  BOOST_CHECK_EQUAL(inChansPerGroup, detectedGrouping.inChansPerGroup);
}
