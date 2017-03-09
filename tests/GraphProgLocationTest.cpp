#define BOOST_TEST_MODULE GraphProgLocationTest
#include <boost/test/unit_test.hpp>
#include <popnn/codelets.hpp>
#include <fstream>

using namespace poplar;
using namespace poplar::program;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(GraphProgLocation) {
  Graph graph(createCPUDevice());
  BOOST_CHECK_NO_THROW(popnn::addCodelets(graph));
}
