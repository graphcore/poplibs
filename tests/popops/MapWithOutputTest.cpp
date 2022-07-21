// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

/** Provides some targeted tests to cover a few combinations of types of map
 * expression in a mapWithOutput call. As the mapWithOutput API ultimately
 * utilises a particular incantation of mapInPlace, other tests more robustly
 * cover the different Unary, Binary, Ternary etc. op types.
 */

#define BOOST_TEST_MODULE MapWithOutputTest

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

// Cover a Const expression
// TODO: Enable, due to limitations of type inference with constants
// in the expression an expression that is just a constant is not
// valid.
BOOST_AUTO_TEST_CASE(MapWithOutputConst, *boost::unit_test::disabled()) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t1");

  Sequence uploadProg, downloadProg, prog;

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostT1 = allocateHostMemoryForTensor(t1, "t1", graph, uploadProg,
                                               downloadProg, tmap);

  popops::mapWithOutput(graph, expr::Const(2.5f), {}, t1, prog, "setFromConst");

  Engine e(graph, Sequence({uploadProg, prog, downloadProg}));
  attachStreams(e, tmap);

  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<float> result(t1.numElements());
  copy(target, FLOAT, rawHostT1.get(), result.data(), result.size());

  std::vector<float> expected(t1.numElements(), 2.5f);
  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}

// Cover a Cast expression
BOOST_AUTO_TEST_CASE(MapWithOutputCast) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 = graph.addVariable(UNSIGNED_INT, {100},
                              VariableMappingMethod::LINEAR, "t1");
  auto t2 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t2");

  Sequence uploadProg, downloadProg, prog;

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostT1 = allocateHostMemoryForTensor(t1, "t1", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT2 = allocateHostMemoryForTensor(t2, "t2", graph, uploadProg,
                                               downloadProg, tmap);

  popops::mapWithOutput(graph, expr::Cast(expr::_1, FLOAT), {t1}, t2, prog,
                        "cast");

  Engine e(graph, Sequence({uploadProg, prog, downloadProg}));
  attachStreams(e, tmap);

  std::vector<unsigned> t1Host(t1.numElements());
  std::iota(t1Host.begin(), t1Host.end(), 1u);
  copy(target, t1Host.data(), t1.numElements(), UNSIGNED_INT, rawHostT1.get());
  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<float> result(t1.numElements());
  copy(target, FLOAT, rawHostT2.get(), result.data(), result.size());

  std::vector<float> expected(t1Host.size());
  std::copy(t1Host.begin(), t1Host.end(), expected.begin());
  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}

// Cover a Unary expression
BOOST_AUTO_TEST_CASE(MapWithOutputUnary) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t1");
  auto t2 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t2");

  Sequence uploadProg, downloadProg, prog;

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostT1 = allocateHostMemoryForTensor(t1, "t1", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT2 = allocateHostMemoryForTensor(t2, "t2", graph, uploadProg,
                                               downloadProg, tmap);

  popops::mapWithOutput(graph,
                        expr::UnaryOp(expr::UnaryOpType::FLOOR, expr::_1), {t1},
                        t2, prog, "floor");

  Engine e(graph, Sequence({uploadProg, prog, downloadProg}));
  attachStreams(e, tmap);

  std::vector<float> t1Host(t1.numElements());
  for (std::size_t i = 0; i != t1.numElements(); ++i) {
    t1Host[i] = float(i) + 1.5f;
  }
  copy(target, t1Host.data(), t1.numElements(), FLOAT, rawHostT1.get());
  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<float> result(t1.numElements());
  copy(target, FLOAT, rawHostT2.get(), result.data(), result.size());

  std::vector<float> expected(t1Host.size());
  std::iota(expected.begin(), expected.end(), 1.0f);
  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}

// Cover a Binary expression
BOOST_AUTO_TEST_CASE(MapWithOutputBinary) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t1");
  auto t2 = graph.addVariable(FLOAT, {1}, VariableMappingMethod::LINEAR, "t2");
  auto t3 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t3");

  Sequence uploadProg, downloadProg, prog;

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostT1 = allocateHostMemoryForTensor(t1, "t1", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT2 = allocateHostMemoryForTensor(t2, "t2", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT3 = allocateHostMemoryForTensor(t3, "t3", graph, uploadProg,
                                               downloadProg, tmap);

  popops::mapWithOutput(
      graph, expr::BinaryOp(expr::BinaryOpType::ADD, expr::_1, expr::_2),
      {t1, t2}, t3, prog, "floor");

  Engine e(graph, Sequence({uploadProg, prog, downloadProg}));
  attachStreams(e, tmap);

  std::vector<float> t1Host(t1.numElements());
  std::vector<float> t2Host(t2.numElements());
  std::iota(t1Host.begin(), t1Host.end(), 1.0f);
  t2Host.front() = 10.0f;
  copy(target, t1Host.data(), t1.numElements(), FLOAT, rawHostT1.get());
  copy(target, t2Host.data(), t2.numElements(), FLOAT, rawHostT2.get());

  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<float> result(t1.numElements());
  copy(target, FLOAT, rawHostT3.get(), result.data(), result.size());

  std::vector<float> expected(t1Host.size());
  std::iota(expected.begin(), expected.end(), 11.0f);
  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}

// Cover a Ternary expression
BOOST_AUTO_TEST_CASE(MapWithOutputTernary) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto t1 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t1");
  auto t2 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t2");
  auto t3 = graph.addVariable(BOOL, {100}, VariableMappingMethod::LINEAR, "t3");
  auto t4 =
      graph.addVariable(FLOAT, {100}, VariableMappingMethod::LINEAR, "t4");

  Sequence uploadProg, downloadProg, prog;

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostT1 = allocateHostMemoryForTensor(t1, "t1", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT2 = allocateHostMemoryForTensor(t2, "t2", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT3 = allocateHostMemoryForTensor(t3, "t3", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostT4 = allocateHostMemoryForTensor(t4, "t4", graph, uploadProg,
                                               downloadProg, tmap);

  popops::mapWithOutput(graph,
                        expr::TernaryOp(expr::TernaryOpType::SELECT, expr::_1,
                                        expr::_2, expr::_3),
                        {t1, t2, t3}, t4, prog, "select");

  Engine e(graph, Sequence({uploadProg, prog, downloadProg}));
  attachStreams(e, tmap);

  std::vector<float> t1Host(t1.numElements());
  std::vector<float> t2Host(t2.numElements());
  std::vector<std::uint8_t> t3Host(t3.numElements());
  for (std::size_t i = 0; i != t1.numElements(); ++i) {
    t1Host[i] = 10.0f + float(i);
    t2Host[i] = 5.0f;
    t3Host[i] = (i % 2 == 0);
  }
  copy(target, t1Host.data(), t1.numElements(), FLOAT, rawHostT1.get());
  copy(target, t2Host.data(), t2.numElements(), FLOAT, rawHostT2.get());
  copy(target, t3Host.data(), t3.numElements(), BOOL, rawHostT3.get());

  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<float> result(t1.numElements());
  copy(target, FLOAT, rawHostT4.get(), result.data(), result.size());

  std::vector<float> expected(t1Host.size());
  for (std::size_t i = 0; i != t1.numElements(); ++i) {
    if (i % 2 == 0) {
      expected[i] = 10.0f + float(i);
    } else {
      expected[i] = 5.0f;
    }
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}
