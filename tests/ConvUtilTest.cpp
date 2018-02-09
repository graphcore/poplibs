#define BOOST_TEST_MODULE ConvUtilTest
#include <boost/test/unit_test.hpp>
#include <popconv/ConvUtil.hpp>

BOOST_AUTO_TEST_CASE(getInputRangeFlipActsAndWeights) {
  auto params = popconv::ConvParams(poplar::FLOAT, // type,
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
  auto inRange = popconv::getInputRange(0, {0, 2}, 1, params);
  BOOST_CHECK_EQUAL(inRange.first, 0);
  BOOST_CHECK_EQUAL(inRange.second, 2);
}

BOOST_AUTO_TEST_CASE(getKernelRangeTruncateInput) {
  auto params = popconv::ConvParams(poplar::FLOAT, // type,
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
  auto kernelRange = popconv::getKernelRange(0, {0, 1}, {0, 3}, params);
  BOOST_CHECK_EQUAL(kernelRange.first, 1);
  BOOST_CHECK_EQUAL(kernelRange.second, 4);
}
