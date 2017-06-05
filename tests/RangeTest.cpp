#define BOOST_TEST_MODULE RangeTest
#include <boost/test/unit_test.hpp>
#include <popconv/ConvUtil.hpp>
#include <vector>

namespace std
{
  ostream& operator<<(ostream& s, const pair<unsigned,unsigned>& p) {
    s << '<' << p.first << ',' << p.second << '>';
    return s;
  }
}

static popconv::ConvParams
makeParams(unsigned stride, unsigned kernelSize, int paddingLower,
           int paddingUpper, unsigned inputSize) {
  return {"float",
          {1, inputSize, inputSize, 1},
          {kernelSize, kernelSize, 1, 1},
          {stride, stride},
          {paddingLower, paddingLower},
          {paddingUpper, paddingUpper},
          {1, 1}};
}

BOOST_AUTO_TEST_CASE(inputRangeTest){
  // No stride, no padding
  const auto params1 = makeParams(1, 3, 0, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params1), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 0, params1), 1U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params1), 1U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 7, 2, params1), 9U);

  // Stride = 2, no padding
  const auto params2 = makeParams(2, 3, 0, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 0, params2), 2U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 3, 2, params2), 8U);

  // Stride 1, padding lower=1, padding upper=0
  const auto params3 = makeParams(1, 3, 1, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params3), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params3), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params3), 1U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 8, 1, params3), 8U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 8, 2, params3), 9U);

  // Stride 1, padding lower=2, padding upper=0
  const auto params4 = makeParams(1, 3, 2, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params4), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params4), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params4), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 9, 1, params4), 8U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 9, 2, params4), 9U);

  // Stride 2, padding lower=1, padding upper=0
  const auto params5 = makeParams(2, 3, 1, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params5), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params5), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params5), 1U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 4, 1, params5), 8U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 4, 2, params5), 9U);

  // Stride 2, padding lower=2, padding upper=0
  const auto params6 = makeParams(2, 3, 2, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params6), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params6), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params6), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 4, 1, params6), 7U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 4, 2, params6), 8U);

  // Stride 2, padding lower=4, padding upper=0
  const auto params7 = makeParams(2, 3, 4, 0, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params7), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params7), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params7), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 0, params7), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 1, params7), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 2, params7), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 5, 1, params7), 7U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 5, 2, params7), 8U);

  // Stride 1, padding lower=1, padding upper=1
  const auto params8 = makeParams(1, 3, 1, 1, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params8), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params8), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params8), 1U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 9, 1, params8), 9U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 9, 2, params8), ~0U);

  // Stride 2, padding lower=1, padding upper=2
  const auto params9 = makeParams(2, 3, 1, 2, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params9), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params9), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params9), 1U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 4, 1, params9), 8U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 4, 2, params9), 9U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 5, 0, params9), 9U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 5, 1, params9), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 5, 2, params9), ~0U);

  // Stride 3, padding lower=2, padding upper=2
  const auto params10 = makeParams(3, 3, 2, 2, 10);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 0, params10), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 1, params10), ~0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 0, 2, params10), 0U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 3, 1, params10), 8U);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 3, 2, params10), 9U);
}

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX,
             unsigned kernelSizeY, unsigned kernelSizeX,
             const std::vector<unsigned> &stride,
             const std::vector<int> &paddingLower,
             const std::vector<int> &paddingUpper) {
  auto params =
      popconv::ConvParams("",
                          {1, inDimY, inDimX, 1},
                          {kernelSizeY, kernelSizeX, 1, 1},
                          stride, paddingLower, paddingUpper, {1, 1});
  return {params.getOutputHeight(), params.getOutputWidth()};
}

BOOST_AUTO_TEST_CASE(outputDimTest){

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 3,3, {1, 1},
                                       {0, 0}, {0, 0}),
        std::make_pair(8u, 8u));

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 2,2, {1, 1},
                                       {0, 0}, {0, 0}),
        std::make_pair(9u, 9u));

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 3,3, {1, 1},
                                       {1, 1}, {0, 0}),
        std::make_pair(9u, 9u));

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 3,3, {1, 1},
                                       {0, 0}, {2, 2}),
        std::make_pair(10u, 10u));

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 3,3, {1, 1},
                                       {3, 3}, {2, 2}),
        std::make_pair(13u, 13u));

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 3,3, {2, 2},
                                       {3, 3}, {2, 2}),
        std::make_pair(7u, 7u));

        BOOST_CHECK_EQUAL(getOutputDim(10,10, 3,3, {1, 2},
                                       {3, 3}, {2, 2}),
        std::make_pair(13u, 7u));

        BOOST_CHECK_EQUAL(getOutputDim(10,12, 3,5, {1, 1},
                                       {0, 1}, {0, 2}),
        std::make_pair(8u, 11u));

        BOOST_CHECK_EQUAL(getOutputDim(4,4, 3,3, {2, 1},
                                       {0, 0}, {1, 0}),
        std::make_pair(2u, 2u));

        BOOST_CHECK_EQUAL(getOutputDim(4,4, 3,3, {3, 3},
                                       {0, 1}, {1, 0}),
        std::make_pair(1u, 1u));
}
