// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE CloneNTest
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(CloneNTest) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5, 20});
  mapTensorLinearly(graph, x1);
  auto N = 20u;
  Tensor y1 = cloneN(graph, x1, N);
  BOOST_CHECK_EQUAL(y1.rank(), 3);
  BOOST_CHECK_EQUAL(y1.dim(0), N);
  BOOST_CHECK_EQUAL(y1.dim(1), x1.dim(0));
  BOOST_CHECK_EQUAL(y1.dim(2), x1.dim(1));
  for (unsigned i = 0; i < N; ++i) {
    auto mapping1 = graph.getTileMapping(y1[i]);
    auto mapping2 = graph.getTileMapping(x1);
    for (unsigned i = 0; i < mapping1.size(); ++i) {
      BOOST_CHECK_EQUAL_COLLECTIONS(mapping1[i].begin(), mapping1[i].end(),
                                    mapping2[i].begin(), mapping2[i].end());
    }
  }
}
