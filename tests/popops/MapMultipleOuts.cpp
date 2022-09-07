// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

/** Provides some targeted tests to cover map API for map
 *  expressions with multiple outputs
 */

#define BOOST_TEST_MODULE MapMultipleOutputs

#include "poplibs_test/Util.hpp"
#include <poplibs_support/TestDevice.hpp>

#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poplibs_support;
using namespace poplar_test;
using namespace poplibs_test::util;

void testExecutor(bool inPlace, bool withOutput) {
  const unsigned dataSize = 10;
  auto device = createTestDevice(TEST_TARGET);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 =
      graph.addVariable(INT, {dataSize}, VariableMappingMethod::LINEAR, "t1");
  auto t2 =
      graph.addVariable(INT, {dataSize}, VariableMappingMethod::LINEAR, "t2");

  Sequence uploadProg, downloadProg, prog;
  Tensor t3, t4;
  std::vector<Tensor> outs;

  if (!inPlace && !withOutput) {
    outs = popops::map(graph, {expr::_1 + expr::_2, expr::_1 * expr::_2},
                       {t1, t2}, prog, "MapMultipleOutsTest");
    t3 = outs[0];
    t4 = outs[1];
  } else if (inPlace && !withOutput) {
    popops::mapInPlace(graph, {expr::_1 + expr::_2, expr::_1 * expr::_2},
                       {t1, t2}, prog, "MapInPlaceMultipleOutsTest");
    t3 = t1;
    t4 = t2;
  } else if (!inPlace && withOutput) {
    t3 =
        graph.addVariable(INT, {dataSize}, VariableMappingMethod::LINEAR, "t3");
    t4 =
        graph.addVariable(INT, {dataSize}, VariableMappingMethod::LINEAR, "t4");
    outs.emplace_back(t3);
    outs.emplace_back(t4);
    popops::mapWithOutput(graph, {expr::_1 + expr::_2, expr::_1 * expr::_2},
                          {t1, t2}, outs, prog,
                          "MapWithOutputMultipleOutsTest");
  } else {
    throw std::logic_error(
        "Unsupported combination of inPlace and withOutput flags");
  }

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostT1 = allocateHostMemoryForTensor(t1, "t1", graph, uploadProg,
                                               boost::none, tmap);
  auto rawHostT2 = allocateHostMemoryForTensor(t2, "t2", graph, uploadProg,
                                               boost::none, tmap);
  auto rawHostT3 = allocateHostMemoryForTensor(t3, "t3", graph, boost::none,
                                               downloadProg, tmap);
  auto rawHostT4 = allocateHostMemoryForTensor(t4, "t4", graph, boost::none,
                                               downloadProg, tmap);

  Engine e(graph, Sequence({uploadProg, prog, downloadProg}));
  attachStreams(e, tmap);

  std::vector<int> t1Host(t1.numElements());
  std::vector<int> t2Host(t2.numElements());
  std::iota(t1Host.begin(), t1Host.end(), 1u);
  std::iota(t2Host.begin(), t2Host.end(), 1u);
  copy(target, t1Host.data(), t1.numElements(), INT, rawHostT1.get());
  copy(target, t2Host.data(), t2.numElements(), INT, rawHostT2.get());

  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<int> resultT3(t1.numElements());
  std::vector<int> resultT4(t2.numElements());
  copy(target, INT, rawHostT3.get(), resultT3.data(), resultT3.size());
  copy(target, INT, rawHostT4.get(), resultT4.data(), resultT4.size());

  std::vector<int> expectedT3(t1.numElements());
  std::vector<int> expectedT4(t2.numElements());

  for (unsigned i = 0; i < dataSize; i++) {
    expectedT3[i] = t1Host[i] + t2Host[i];
    expectedT4[i] = t1Host[i] * t2Host[i];
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(resultT3.begin(), resultT3.end(),
                                expectedT3.begin(), expectedT3.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(resultT4.begin(), resultT4.end(),
                                expectedT4.begin(), expectedT4.end());
}

BOOST_AUTO_TEST_CASE(MapInPlace) { testExecutor(true, false); }

BOOST_AUTO_TEST_CASE(Map) { testExecutor(false, false); }

BOOST_AUTO_TEST_CASE(MapWithOutputs) { testExecutor(false, true); }

BOOST_AUTO_TEST_CASE(MapWithOutputsCheckExceptions) {
  const unsigned dataSize = 10;
  auto device = createTestDevice(TEST_TARGET);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 =
      graph.addVariable(INT, {dataSize}, VariableMappingMethod::LINEAR, "t1");
  auto t2 =
      graph.addVariable(INT, {dataSize}, VariableMappingMethod::LINEAR, "t2");

  Sequence prog;
  std::vector<Tensor> outs;
  BOOST_CHECK_THROW(popops::mapWithOutput(
                        graph, {expr::_1 + expr::_2, expr::_1 * expr::_2},
                        {t1, t2}, outs, prog,
                        "mapWithOutput: check for expr.size() == outs.size()"),
                    poputil::poplibs_error);

  BOOST_CHECK_THROW(
      popops::mapInPlace(
          graph,
          {expr::_1 + expr::_2, expr::_1 * expr::_2, expr::_1 - expr::_2},
          {t1, t2}, prog, "mapInPlace: check for expr.size() <= ts.size()"),
      poputil::poplibs_error);
}
