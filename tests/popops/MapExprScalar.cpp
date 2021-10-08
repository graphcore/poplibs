// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MapExprScalar

#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <pva/pva.hpp>

#include <sstream>
#include <utility>
#include <vector>

using namespace poplar;
using namespace poplibs_support;
using namespace popops;
using namespace popops::expr;

static unsigned numMapVerticesCreated(const Expr &expression,
                                      const poplar::Type &dType) {
  auto device = createTestDevice(DeviceType::IpuModel2);
  poplar::Graph g(device.getTarget());
  popops::addCodelets(g);
  poplar::program::Sequence prog;

  auto a = g.addVariable(dType, {1}, "inputa");
  g.setTileMapping(a, 0);
  auto b = g.addVariable(dType, {1}, "inputb");
  g.setTileMapping(b, 0);

  popops::map(g, expression, {a, b}, prog, "ScalarOp");
  Engine engine(g, prog);
  const auto report = engine.getReport(false);

  unsigned numMapVertices = 0;
  for (const auto &t : report.compilation().tiles()) {
    for (const auto &v : t.memory().vertices()) {
      const auto name = v.type().name();
      std::cout << v.type().name() << "\n";

      if (name.find("popops::map::") != std::string::npos) {
        numMapVertices++;
      }
    }
  }
  return numMapVertices;
}

BOOST_AUTO_TEST_CASE(MapExprScalarFloatConsts) {
  const auto type = poplar::FLOAT;
  // These generate a map expression
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(1.0f)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(0.5f)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(-2.0f)), type), 1);
  // These do not
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(4.0f)), type), 0);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(100.0f)), type), 0);
}

BOOST_AUTO_TEST_CASE(MapExprScalarHalfConsts) {
  const auto type = poplar::HALF;
  // These generate a map expression
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(1.0f)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(0.5f)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(-2.0f)), type), 1);
  // These do not
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(4.0f)), type), 0);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(100.0f)), type), 0);
}

BOOST_AUTO_TEST_CASE(MapExprScalarUnsignedIntegerConsts) {
  const auto type = poplar::UNSIGNED_INT;
  // These generate a map expression
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(0u)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(1u)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(2u)), type), 1);
  // These do not
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(4u)), type), 0);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(100u)), type), 0);
}

BOOST_AUTO_TEST_CASE(MapExprScalarSignedIntegerConsts) {
  const auto type = poplar::INT;
  // These generate a map expression
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(0)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(-1)), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(2)), type), 1);
  // These do not
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(4)), type), 0);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, Const(100)), type), 0);
}

BOOST_AUTO_TEST_CASE(MapExprScalarSupportedOps) {
  const auto type = poplar::FLOAT;
  // These generate a map expression
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Add(_1, _1), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Sub(_1, _1), type), 1);
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Mul(_1, _1), type), 1);
  // These do not
  BOOST_CHECK_EQUAL(numMapVerticesCreated(Divide(_1, _1), type), 0);
}
