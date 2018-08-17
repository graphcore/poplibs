#define BOOST_TEST_MODULE ConvUtilTest
#include "ConvUtilInternal.hpp"
#include <poplin/Convolution.hpp>
#include <poplin/ConvUtil.hpp>
#include "TestDevice.hpp"

#include <boost/test/unit_test.hpp>

using namespace poplar;

BOOST_AUTO_TEST_CASE(getInputRangeFlipActsAndWeights) {
  auto params = poplin::ConvParams(poplar::FLOAT, // type,
                                    1, // batch size
                                    {3, 1}, // input size
                                    {5, 1}, // kernel size
                                    1, // input channels
                                    1, // output channels
                                    1, // conv groups
                                    {0, 0}, {0, 0}, // input truncation
                                    {1, 1}, // input dilation
                                    {2, 0}, {2, 0}, // input padding
                                    {true, false}, // flip input
                                    {0, 0}, {0, 0}, // kernel truncation
                                    {1, 1}, // kernel dilation
                                    {0, 0}, {0, 0}, // kernel padding
                                    {true, false}, // flip kernel
                                    {0, 0}, {0, 0}, // output truncation
                                    {1, 1}, // stride
                                    {0, 0}, {0, 0} // output padding
                                    );
  auto inRange = poplin::getInputRange(0, {0, 2}, 1, params);
  BOOST_CHECK_EQUAL(inRange.first, 0);
  BOOST_CHECK_EQUAL(inRange.second, 2);
}

BOOST_AUTO_TEST_CASE(getKernelRangeTruncateInput) {
  auto params = poplin::ConvParams(poplar::FLOAT, // type,
                                    1, // batch size
                                    {3}, // input size
                                    {4}, // kernel size
                                    1, // input channels
                                    1, // output channels
                                    1, // conv groups
                                    {0}, {1}, // input truncation
                                    {2}, // input dilation
                                    {1}, {0}, // input padding
                                    {false}, // flip input
                                    {0}, {0}, // kernel truncation
                                    {1}, // kernel dilation
                                    {0}, {0}, // kernel padding
                                    {false}, // flip kernel
                                    {0}, {0}, // output truncation
                                    {1}, // stride
                                    {0}, {0} // output padding
                                    );
  auto kernelRange = poplin::getKernelRange(0, {0, 1}, {0, 3}, params);
  BOOST_CHECK_EQUAL(kernelRange.first, 1);
  BOOST_CHECK_EQUAL(kernelRange.second, 4);
}

BOOST_AUTO_TEST_CASE(DetectWeightsChannelGrouping) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  const auto outChansPerGroup = 8;
  const auto inChansPerGroup = 16;
  auto t =
      graph.addVariable(HALF, {2, 14, 7, outChansPerGroup, inChansPerGroup});
  unsigned detectedOutChansPerGroup, detectedInChansPerGroup;
  std::tie(detectedOutChansPerGroup, detectedInChansPerGroup) =
    poplin::detectWeightsChannelGrouping(t);
  BOOST_CHECK_EQUAL(outChansPerGroup, detectedOutChansPerGroup);
  BOOST_CHECK_EQUAL(inChansPerGroup, detectedInChansPerGroup);
}
