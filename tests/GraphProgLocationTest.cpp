// Copyright (c) 2017 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE GraphProgLocationTest
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <poplar/Graph.hpp>
#include <popnn/codelets.hpp>
namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(GraphProgLocation) {
  poplar::Graph graph(poplar::Target::createCPUTarget());
  BOOST_CHECK_NO_THROW(popnn::addCodelets(graph));
}
