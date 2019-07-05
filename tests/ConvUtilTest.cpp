#define BOOST_TEST_MODULE ConvUtilTest
#include "ConvUtilInternal.hpp"
#include <poplin/Convolution.hpp>
#include <poplin/ConvUtil.hpp>
#include "TestDevice.hpp"
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

BOOST_AUTO_TEST_CASE(DetectWeightsChannelGrouping) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  const auto outChansPerGroup = 8;
  const auto inChansPerGroup = 16;
  auto t =
      graph.addVariable(HALF, {2, 14, 7, outChansPerGroup, inChansPerGroup});
  poputil::mapTensorLinearly(graph, t);
  unsigned detectedOutChansPerGroup, detectedInChansPerGroup;
  std::tie(detectedOutChansPerGroup, detectedInChansPerGroup) =
    poplin::detectWeightsChannelGrouping(graph, t);
  BOOST_CHECK_EQUAL(outChansPerGroup, detectedOutChansPerGroup);
  BOOST_CHECK_EQUAL(inChansPerGroup, detectedInChansPerGroup);
}
