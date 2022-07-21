// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LoopTests
#include "popops/Loop.hpp"
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
using namespace poplar_test;
using namespace popops;
using namespace poplibs_support;
using namespace popops::expr;

static std::size_t loopTest(std::size_t begin, std::size_t end,
                            std::size_t step) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto acc = graph.addVariable(UNSIGNED_INT, {1}, "acc");
  poputil::mapTensorLinearly(graph, acc);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostAcc = allocateHostMemoryForTensor(acc, "acc", graph, uploadProg,
                                                downloadProg, tmap);

  auto loop = Sequence();
  loop.add(popops::countedLoop(
      graph, begin, end, step,
      [&](Tensor idx) {
        Sequence prog;
        popops::addInPlace(graph, acc, idx, prog, "/accumulate");
        return prog;
      },
      "/countedLoop"));

  Engine engine(graph, Sequence{uploadProg, loop, downloadProg});
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

static std::size_t forLoopTest(int begin, int end, int step,
                               const Type &tensorType,
                               bool createCountTensor = false) {

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto limit = graph.addConstant<unsigned>(tensorType, {1}, end, "limit");
  graph.setTileMapping(limit, 0);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;

  auto prog = Sequence();
  auto loop = Sequence();

  // Make a loop body which always counts 0.... which we use to check the
  // result
  auto bodyVar = graph.addVariable(UNSIGNED_INT, {1}, "bodyVar");
  graph.setTileMapping(bodyVar, 0);
  auto zero = graph.addConstant<unsigned>(UNSIGNED_INT, {1}, 0u, "bodyVar");
  graph.setTileMapping(zero, 0);

  prog.add(Copy(zero, bodyVar));

  popops::mapInPlace(graph, Add(_1, Const(1u)), {bodyVar}, loop,
                     "/bodyFunction");

  auto rawHostbodyVar = allocateHostMemoryForTensor(
      bodyVar, "bodyVar", graph, uploadProg, downloadProg, tmap);
  if (createCountTensor) {
    auto count = graph.addVariable(tensorType, {1}, "loopCount");
    graph.setTileMapping(count, 0);
    prog.add(popops::countedForLoop(graph, count, begin, limit, step, loop,
                                    "/countedForLoop"));
  } else {
    prog.add(popops::countedForLoop(graph, begin, limit, step, loop,
                                    "/countedForLoop"));
  }

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg});
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  return *reinterpret_cast<unsigned *>(rawHostbodyVar.get());
}

BOOST_AUTO_TEST_CASE(ForLoopTests) {
  BOOST_CHECK_EQUAL(forLoopTest(0, 2, 1, UNSIGNED_INT), 2);
  BOOST_CHECK_EQUAL(forLoopTest(0, 2, 1, UNSIGNED_INT, true), 2);
  BOOST_CHECK_EQUAL(forLoopTest(0, 10, 1, UNSIGNED_INT), 10);
  BOOST_CHECK_EQUAL(forLoopTest(2, 12, 1, UNSIGNED_INT), 10);
  BOOST_CHECK_EQUAL(forLoopTest(0, 20, 2, UNSIGNED_INT), 10);
  BOOST_CHECK_EQUAL(forLoopTest(7, 23, 4, UNSIGNED_INT), 4);

  BOOST_CHECK_EQUAL(forLoopTest(10, 0, -2, UNSIGNED_INT), 5);
  BOOST_CHECK_EQUAL(forLoopTest(10, 0, -2, UNSIGNED_INT, true), 5);

  BOOST_CHECK_EQUAL(forLoopTest(-3, 11, 2, INT), 7);
  BOOST_CHECK_EQUAL(forLoopTest(-3, -11, -1, INT), 8);
}
