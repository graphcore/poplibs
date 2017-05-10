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

BOOST_AUTO_TEST_CASE(inputRangeTest){
  // No stride, no padding
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 0, 0, 10, 0, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(1, 1, 3, 0, 0, 10, 0, false), 1);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 0, 0, 10, 1, false), 1);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(7, 1, 3, 0, 0, 10, 2, false), 9);

  // Stride = 2, no padding
  BOOST_CHECK_EQUAL(popconv::getInputIndex(1, 2, 3, 0, 0, 10, 0, false), 2);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(3, 2, 3, 0, 0, 10, 2, false), 8);

  // Stride 1, padding lower=1, padding upper=0
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 1, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 1, 0, 10, 1, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 1, 0, 10, 2, false), 1);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(8, 1, 3, 1, 0, 10, 1, false), 8);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(8, 1, 3, 1, 0, 10, 2, false), 9);

  // Stride 1, padding lower=2, padding upper=0
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 2, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 2, 0, 10, 1, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 2, 0, 10, 2, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(9, 1, 3, 2, 0, 10, 1, false), 8);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(9, 1, 3, 2, 0, 10, 2, false), 9);

  // Stride 2, padding lower=1, padding upper=0
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 1, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 1, 0, 10, 1, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 1, 0, 10, 2, false), 1);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(4, 2, 3, 1, 0, 10, 1, false), 8);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(4, 2, 3, 1, 0, 10, 2, false), 9);

  // Stride 2, padding lower=2, padding upper=0
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 2, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 2, 0, 10, 1, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 2, 0, 10, 2, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(4, 2, 3, 2, 0, 10, 1, false), 7);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(4, 2, 3, 2, 0, 10, 2, false), 8);

  // Stride 2, padding lower=4, padding upper=0
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 4, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 4, 0, 10, 1, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 4, 0, 10, 2, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(1, 2, 3, 4, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(1, 2, 3, 4, 0, 10, 1, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(1, 2, 3, 4, 0, 10, 2, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(5, 2, 3, 4, 0, 10, 1, false), 7);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(5, 2, 3, 4, 0, 10, 2, false), 8);

  // Stride 1, padding lower=1, padding upper=1
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 1, 0, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 1, 0, 10, 1, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 1, 3, 1, 0, 10, 2, false), 1);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(9, 1, 3, 1, 0, 10, 1, false), 9);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(9, 1, 3, 1, 0, 10, 2, false), ~0u);

  // Stride 2, padding lower=1, padding upper=2
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 1, 2, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 1, 2, 10, 1, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 2, 3, 1, 2, 10, 2, false), 1);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(4, 2, 3, 1, 0, 10, 1, false), 8);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(4, 2, 3, 1, 0, 10, 2, false), 9);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(5, 2, 3, 1, 2, 10, 0, false), 9);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(5, 2, 3, 1, 2, 10, 1, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(5, 2, 3, 1, 2, 10, 2, false), ~0u);

  // Stride 3, padding lower=2, padding upper=2
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 3, 3, 2, 2, 10, 0, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 3, 3, 2, 2, 10, 1, false), ~0u);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(0, 3, 3, 2, 2, 10, 2, false), 0);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(3, 3, 3, 2, 2, 10, 1, false), 8);
  BOOST_CHECK_EQUAL(popconv::getInputIndex(3, 3, 3, 2, 2, 10, 2, false), 9);

}

BOOST_AUTO_TEST_CASE(outputDimTest){

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 3,3, {1, 1},
                                                {0, 0}, {0, 0}, false),
        std::make_pair(8u, 8u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 2,2, {1, 1},
                                                {0, 0}, {0, 0}, false),
        std::make_pair(9u, 9u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 3,3, {1, 1},
                                                {1, 1}, {0, 0}, false),
        std::make_pair(9u, 9u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 3,3, {1, 1},
                                                {0, 0}, {2, 2}, false),
        std::make_pair(10u, 10u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 3,3, {1, 1},
                                                {3, 3}, {2, 2}, false),
        std::make_pair(13u, 13u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 3,3, {2, 2},
                                                {3, 3}, {2, 2}, false),
        std::make_pair(7u, 7u));


        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,10, 3,3, {1, 2},
                                                {3, 3}, {2, 2}, false),
        std::make_pair(13u, 7u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(10,12, 3,5, {1, 1},
                                                {0, 1}, {0, 2}, false),
        std::make_pair(8u, 11u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(4,4, 3,3, {2, 1},
                                                {0, 0}, {1, 0}, false),
        std::make_pair(2u, 2u));

        BOOST_CHECK_EQUAL(popconv::getOutputDim(4,4, 3,3, {3, 3},
                                                {0, 1}, {1, 0}, false),
        std::make_pair(1u, 1u));
}
