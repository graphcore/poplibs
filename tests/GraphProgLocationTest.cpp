#define BOOST_TEST_MODULE GraphProgLocationTest
#include <boost/test/unit_test.hpp>
#include <popnn/Net.hpp>
#include <fstream>

using namespace poplar;
using namespace poplar::program;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(GraphProgLocation) {
  BOOST_CHECK(std::ifstream(popnn::findGraphProg()).good());
}
