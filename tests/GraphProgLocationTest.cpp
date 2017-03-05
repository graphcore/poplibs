#define BOOST_TEST_MODULE GraphProgLocationTest
#include <boost/test/unit_test.hpp>
#include <poplar/Graph.hpp>
#include <popnn/codelets.hpp>
#include <fstream>
namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(GraphProgLocation) {
  poplar::Graph graph(poplar::createCPUDevice());
  BOOST_CHECK_NO_THROW(popnn::addCodelets(graph));
}
