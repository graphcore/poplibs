// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MapExprOptimisations

#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

// Access to internal functions to directly check expression optimisations
#include "popops/ElementWiseInternal.hpp"

#include <sstream>
#include <utility>
#include <vector>

using namespace poplibs_support;
using namespace popops;
using namespace popops::expr;

#define HALF_TOLERANCE .1
#define FLOAT_TOLERANCE .001

std::pair<std::vector<float>, std::vector<float>>
executeExpr(const Expr &expression, const poplar::Type &dType, bool inPlace,
            bool enableFusedCodelets = true) {
  auto device = createTestDevice(TEST_TARGET);
  const auto &target = device.getTarget();
  poplar::Graph g(device.getTarget());
  popops::addCodelets(g);
  unsigned size = 16;

  auto a = g.addVariable(dType, {size}, "inputa");
  g.setTileMapping(a, 0);
  auto b = g.addVariable(dType, {size}, "inputb");
  g.setTileMapping(b, 0);
  auto c = g.addVariable(dType, {size}, "inputc");
  g.setTileMapping(c, 0);

  auto rawBufSize = target.getTypeSize(dType) * size;

  std::vector<char> rawInA(rawBufSize);
  std::vector<char> rawInB(rawBufSize);
  std::vector<char> rawInC(rawBufSize);
  std::vector<char> rawOutOpt(rawBufSize);
  std::vector<char> rawOutNoOpt(rawBufSize);

  const std::vector<float> hostInA = {1,    2,    3,   4,   5,   6,    7,    8,
                                      1.25, 2.25, 3.5, 4.5, 5.5, 6.75, 6.75, 8};

  const std::vector<float> hostInC = hostInA;

  const std::vector<float> hostInB = {
      8.25, 7.25, 6.5, 5.5, 4.5, 3.5, 2.75, 1.75, 8, 7, 6, 5, 4, 3, 2, 1};

  poplibs_test::util::copy(target, hostInA.data(), size, dType,
                           reinterpret_cast<void *>(rawInA.data()));
  poplibs_test::util::copy(target, hostInC.data(), size, dType,
                           reinterpret_cast<void *>(rawInC.data()));

  poplar::OptionFlags enableOptims = {
      {"enableExpressionOptimizations", "true"}};
  poplar::OptionFlags disableOptims = {
      {"enableExpressionOptimizations", "false"}};

  enableOptims.set("enableGenerateCodelet",
                   enableFusedCodelets ? "true" : "false");

  poplar::program::Sequence progOpt;
  poplar::Tensor tOpt;
  if (!inPlace) {
    tOpt =
        popops::map(g, expression, {a, b}, progOpt, "test1Opt", enableOptims);
  } else {
    popops::mapInPlace(g, expression, {a, b}, progOpt, "test1Opt",
                       enableOptims);
  }

  poplar::program::Sequence progNoOpt;
  poplar::Tensor tNoOpt;
  if (!inPlace) {
    tNoOpt = popops::map(g, expression, {c, b}, progNoOpt, "test1NoOpt",
                         disableOptims);
  } else {
    popops::mapInPlace(g, expression, {c, b}, progNoOpt, "test1NoOpt",
                       disableOptims);
  }
  g.createHostWrite("inA", a);
  g.createHostWrite("inC", c);
  g.createHostWrite("inB", b);
  if (!inPlace) {
    g.createHostRead("tOpt", tOpt);
    g.createHostRead("tNoOpt", tNoOpt);
  } else {
    g.createHostRead("tOpt", a);
    g.createHostRead("tNoOpt", c);
  }

  poplar::program::Sequence controlProg(
      {std::move(progOpt), std::move(progNoOpt)});

  poplar::Engine e(g, controlProg);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.writeTensor("inA", rawInA.data(), rawInA.data() + rawInA.size());
    e.writeTensor("inC", rawInC.data(), rawInC.data() + rawInC.size());
    e.writeTensor("inB", rawInB.data(), rawInB.data() + rawInB.size());
    e.run();
    e.readTensor("tOpt", rawOutOpt.data(), rawOutOpt.data() + rawOutOpt.size());
    e.readTensor("tNoOpt", rawOutNoOpt.data(),
                 rawOutNoOpt.data() + rawOutNoOpt.size());
  });

  std::vector<float> hostOutOpt(size);
  poplibs_test::util::copy(target, dType,
                           reinterpret_cast<void *>(rawOutOpt.data()),
                           hostOutOpt.data(), size);
  std::vector<float> hostOutNoOpt(size);
  poplibs_test::util::copy(target, dType,
                           reinterpret_cast<void *>(rawOutNoOpt.data()),
                           hostOutNoOpt.data(), size);

  return std::make_pair(hostOutOpt, hostOutNoOpt);
}

static const auto exampleBinaryExpr =
    Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1)));

BOOST_AUTO_TEST_CASE(Pow_cast_half) {
  const auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(1.0f));
  const auto dType = poplar::HALF;
  const auto expectedOptimisedExpr =
      popops::expr::Cast(exampleBinaryExpr, dType);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_cast_float) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(1.0f));
  const auto dType = poplar::FLOAT;
  const auto expectedOptimisedExpr =
      popops::expr::Cast(exampleBinaryExpr, dType);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, poplar::FLOAT, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(0.5f));
  const auto dType = poplar::HALF;
  const auto expectedOptimisedExpr = popops::expr::Sqrt(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_float) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(0.5f));
  const auto dType = poplar::FLOAT;
  const auto expectedOptimisedExpr = popops::expr::Sqrt(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_isqrt_half) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(-0.5f));
  const auto dType = poplar::HALF;
  const auto expectedOptimisedExpr = popops::expr::Rsqrt(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_isqrt_float) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(-0.5f));
  const auto dType = poplar::FLOAT;
  const auto expectedOptimisedExpr = popops::expr::Rsqrt(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_inv_half) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(-1.0f));
  const auto dType = poplar::HALF;
  const auto expectedOptimisedExpr = popops::expr::Inv(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_inv_float) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(-1.0f));
  const auto dType = poplar::FLOAT;
  const auto expectedOptimisedExpr = popops::expr::Inv(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sq_half) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(2.0f));
  const auto dType = poplar::HALF;
  const auto expectedOptimisedExpr = popops::expr::Square(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sq_float) {
  auto e1 = popops::expr::Pow(exampleBinaryExpr, Const(2.0f));
  const auto dType = poplar::FLOAT;
  const auto expectedOptimisedExpr = popops::expr::Square(exampleBinaryExpr);
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sq_float_1) {
  const auto buildExpr = [](const auto &innerExpr) {
    return Add(Add(Add(innerExpr, _2), Sub(_2, Const(-1))), _2);
  };
  auto e1 = buildExpr(Pow(_1, Const(2)));
  const auto dType = poplar::FLOAT;
  const auto expectedOptimisedExpr = buildExpr(Square(_1));
  const auto actualOptimisedExpr = optimise(e1, {dType, dType}).expression;
  BOOST_CHECK(actualOptimisedExpr->deepEquals(expectedOptimisedExpr));
  auto p = executeExpr(e1, dType, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sq_float_1_inplace) {
  auto e1 = popops::expr::Add(
      Add(Add(Pow(_1, Const(2)), _2), Sub(_2, Const(-1))), _2);
  auto p = executeExpr(e1, poplar::FLOAT, true);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_cast_float_inplace) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(1.0f));
  auto p = executeExpr(e1, poplar::FLOAT, true);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half_inplace) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, true);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half_disable_fusedcodelets) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, false, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half_inplace_disable_fusedcodelets) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, true, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Ternary_pred_false) {
  auto e1 = expr::TernaryOp(expr::TernaryOpType::SELECT, expr::_1, expr::_2,
                            expr::Const(false));
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Ternary_pred_true) {
  auto e1 = expr::TernaryOp(expr::TernaryOpType::SELECT, expr::_1, expr::_2,
                            expr::Const(true));
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Add_float_const_lhs) {
  auto e1 = popops::expr::Add(Const(0.f), _1);
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Add_float_const_rhs) {
  auto e1 = popops::expr::Add(_1, Const(0.f));
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Sub_float_const_lhs) {
  auto e1 = popops::expr::Sub(Const(0.f), _1);
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Sub_float_const_rhs) {
  auto e1 = popops::expr::Sub(_1, Const(0.f));
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Mul_float_const_lhs) {
  auto e1 = popops::expr::Mul(Const(1.f), _1);
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Mul_float_const_rhs) {
  auto e1 = popops::expr::Mul(_1, Const(1.f));
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Div_float_const_rhs) {
  auto e1 = popops::expr::Divide(_1, Const(1.f));
  auto p = executeExpr(e1, poplar::FLOAT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(ShiftLeft_int_const_rhs) {
  auto e1 = popops::expr::Shl(_1, Const(0));
  auto p = executeExpr(e1, poplar::INT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(ShiftRight_int_const_rhs) {
  auto e1 = popops::expr::Shr(_1, Const(0));
  auto p = executeExpr(e1, poplar::INT, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Add_half_const_lhs) {
  auto e1 = popops::expr::Add(Const(0.f), _1);
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Add_half_const_rhs) {
  auto e1 = popops::expr::Add(_1, Const(0.f));
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Sub_half_const_lhs) {
  auto e1 = popops::expr::Sub(Const(0.f), _1);
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Sub_half_const_rhs) {
  auto e1 = popops::expr::Sub(_1, Const(0.f));
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Mul_half_const_lhs) {
  auto e1 = popops::expr::Mul(Const(1.f), _1);
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Mul_half_const_rhs) {
  auto e1 = popops::expr::Mul(_1, Const(1.f));
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Div_half_const_rhs) {
  auto e1 = popops::expr::Divide(_1, Const(1.f));
  auto p = executeExpr(e1, poplar::HALF, true, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}
