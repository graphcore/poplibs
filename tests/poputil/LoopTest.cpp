// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LoopTests
#include "poputil/Loop.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/ElementWise.hpp"
#include "popops/codelets.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include <boost/test/unit_test.hpp>
#include <poplibs_support/TestDevice.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;

static std::size_t loopTest(std::size_t begin, std::size_t end,
                            std::size_t step) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto acc = graph.addVariable(UNSIGNED_INT, {1}, "acc");
  poputil::mapTensorLinearly(graph, acc);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostAcc = allocateHostMemoryForTensor(acc, "acc", graph, uploadProg,
                                                downloadProg, tmap);

  auto loop = Sequence();
  loop.add(poputil::countedLoop(
      graph, begin, end, step, "/countedLoop", [&](Tensor idx) {
        Sequence prog;
        popops::addInPlace(graph, acc, idx, prog, "/accumulate");
        return prog;
      }));

  Engine engine(graph, Sequence(uploadProg, loop, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  return *reinterpret_cast<unsigned *>(rawHostAcc.get());
}

BOOST_AUTO_TEST_CASE(LoopTests) {
  BOOST_CHECK_EQUAL(loopTest(1, 2, 1), 1);
  BOOST_CHECK_EQUAL(loopTest(1, 2, 100), 1);
  BOOST_CHECK_EQUAL(loopTest(1, 10, 1), 45); // 1 + 2 + ... + 9
  BOOST_CHECK_EQUAL(loopTest(1, 10, 2), 25); // 1 + 3 + 5 + 7 + 9

  BOOST_CHECK_THROW(loopTest(0, 0, 0), poputil::poplibs_error);
  BOOST_CHECK_THROW(loopTest(0, 1, 0), poputil::poplibs_error);
  BOOST_CHECK_THROW(loopTest(10, 10, 10), poputil::poplibs_error);
}
