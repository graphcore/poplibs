// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE ExprAPI

#include <boost/test/unit_test.hpp>

#include <popops/Expr.hpp>

using namespace poplar;
using namespace popops;

BOOST_AUTO_TEST_CASE(DeepEqualsConst) {
  expr::Const c(1.0f);
  BOOST_CHECK(c.deepEquals(expr::Const(1.0f)));
  BOOST_CHECK(!c.deepEquals(expr::Const(2.0f)));
  BOOST_CHECK(!c.deepEquals(expr::Const(1u)));
  BOOST_CHECK(!c.deepEquals(expr::ConstHalf(1.0f)));
  BOOST_CHECK(!c.deepEquals(expr::PlaceHolder(1u)));

  expr::ConstHalf ch(2.0f);
  BOOST_CHECK(ch.deepEquals(expr::ConstHalf(2.0f)));
  BOOST_CHECK(!ch.deepEquals(expr::ConstHalf(1.0f)));
  BOOST_CHECK(!ch.deepEquals(expr::PlaceHolder(1u)));
}

BOOST_AUTO_TEST_CASE(DeepEqualsPlaceholder) {
  expr::PlaceHolder p(1u);
  BOOST_CHECK(p.deepEquals(expr::PlaceHolder(1u)));
  BOOST_CHECK(!p.deepEquals(expr::PlaceHolder(2u)));
  BOOST_CHECK(!p.deepEquals(expr::Const(1.0f)));
}

BOOST_AUTO_TEST_CASE(DeepEqualsCast) {
  expr::Cast c(expr::Const(1.0f), poplar::UNSIGNED_INT);
  BOOST_CHECK(
      c.deepEquals(expr::Cast(expr::Const(1.0f), poplar::UNSIGNED_INT)));
  BOOST_CHECK(!c.deepEquals(expr::Cast(expr::Const(1.0f), poplar::INT)));
  BOOST_CHECK(!c.deepEquals(expr::Cast(expr::Const(1u), poplar::UNSIGNED_INT)));
  BOOST_CHECK(!c.deepEquals(expr::PlaceHolder(1u)));
}

BOOST_AUTO_TEST_CASE(DeepEqualsUnaryOp) {
  expr::UnaryOp u(expr::UnaryOpType::ABSOLUTE, expr::Const(1.0f));
  BOOST_CHECK(u.deepEquals(
      expr::UnaryOp(expr::UnaryOpType::ABSOLUTE, expr::Const(1.0f))));
  BOOST_CHECK(
      !u.deepEquals(expr::UnaryOp(expr::UnaryOpType::ASIN, expr::Const(1.0f))));
  BOOST_CHECK(!u.deepEquals(
      expr::UnaryOp(expr::UnaryOpType::ABSOLUTE, expr::PlaceHolder(1u))));
}

BOOST_AUTO_TEST_CASE(DeepEqualsBinaryOp) {
  expr::BinaryOp b(expr::BinaryOpType::ADD, expr::Const(1.0f),
                   expr::PlaceHolder(1u));
  BOOST_CHECK(b.deepEquals(expr::BinaryOp(
      expr::BinaryOpType::ADD, expr::Const(1.0f), expr::PlaceHolder(1u))));
  BOOST_CHECK(!b.deepEquals(expr::BinaryOp(
      expr::BinaryOpType::MAXIMUM, expr::Const(1.0f), expr::PlaceHolder(1u))));
  BOOST_CHECK(!b.deepEquals(expr::BinaryOp(
      expr::BinaryOpType::ADD, expr::Const(1u), expr::PlaceHolder(1u))));
  BOOST_CHECK(!b.deepEquals(expr::BinaryOp(
      expr::BinaryOpType::ADD, expr::Const(1.0f), expr::PlaceHolder(2u))));
}

BOOST_AUTO_TEST_CASE(DeepEqualsTernaryOp) {
  expr::TernaryOp t(expr::TernaryOpType::CLAMP, expr::Const(1.0f),
                    expr::PlaceHolder(1u),
                    expr::Cast(expr::Const(1u), poplar::FLOAT));
  BOOST_CHECK(t.deepEquals(expr::TernaryOp(
      expr::TernaryOpType::CLAMP, expr::Const(1.0f), expr::PlaceHolder(1u),
      expr::Cast(expr::Const(1u), poplar::FLOAT))));
  BOOST_CHECK(!t.deepEquals(expr::TernaryOp(
      expr::TernaryOpType::SELECT, expr::Const(1.0f), expr::PlaceHolder(1u),
      expr::Cast(expr::Const(1u), poplar::FLOAT))));
  BOOST_CHECK(!t.deepEquals(expr::TernaryOp(
      expr::TernaryOpType::CLAMP, expr::Const(1u), expr::PlaceHolder(1u),
      expr::Cast(expr::Const(1u), poplar::FLOAT))));
  BOOST_CHECK(!t.deepEquals(expr::TernaryOp(
      expr::TernaryOpType::CLAMP, expr::Const(1.0f), expr::PlaceHolder(2u),
      expr::Cast(expr::Const(1u), poplar::FLOAT))));
  BOOST_CHECK(!t.deepEquals(expr::TernaryOp(
      expr::TernaryOpType::CLAMP, expr::Const(1.0f), expr::PlaceHolder(1u),
      expr::Cast(expr::Const(2u), poplar::FLOAT))));
}

BOOST_AUTO_TEST_CASE(ConstructAndAssignConst) {
  expr::Const c(1.0f);
  // Copy construct
  expr::Const c2(c);
  BOOST_CHECK(c2.deepEquals(c));
  // Move construct
  expr::Const c3(std::move(c2));
  BOOST_CHECK(c3.deepEquals(c));
  // Copy assign
  expr::Const c4 = c;
  BOOST_CHECK(c4.deepEquals(c));
  // Move assign
  expr::Const c5 = std::move(c4);
  BOOST_CHECK(c5.deepEquals(c));
}

BOOST_AUTO_TEST_CASE(ConstructAndAssignPlaceHolder) {
  expr::PlaceHolder p(1u);
  // Copy construct
  expr::PlaceHolder p2(p);
  BOOST_CHECK(p2.deepEquals(p));
  // Move construct
  expr::PlaceHolder p3(std::move(p2));
  BOOST_CHECK(p3.deepEquals(p));
  // Copy assign
  expr::PlaceHolder p4 = p;
  BOOST_CHECK(p4.deepEquals(p));
  // Move assign
  expr::PlaceHolder p5 = std::move(p4);
  BOOST_CHECK(p5.deepEquals(p));
}

BOOST_AUTO_TEST_CASE(ConstructAndAssignCast) {
  expr::Cast c(expr::Const(1u), poplar::FLOAT);
  // Copy construct
  expr::Cast c2(c);
  BOOST_CHECK(c2.deepEquals(c));
  // Move construct
  expr::Cast c3(std::move(c2));
  BOOST_CHECK(c3.deepEquals(c));
  // Copy assign
  expr::Cast c4 = c;
  BOOST_CHECK(c4.deepEquals(c));
  // Move assign
  expr::Cast c5 = std::move(c4);
  BOOST_CHECK(c5.deepEquals(c));
}

BOOST_AUTO_TEST_CASE(ConstructAndAssignUnaryOp) {
  expr::UnaryOp u(expr::UnaryOpType::ABSOLUTE, expr::PlaceHolder(1u));
  // Copy construct
  expr::UnaryOp u2(u);
  BOOST_CHECK(u2.deepEquals(u));
  // Move construct
  expr::UnaryOp u3(std::move(u2));
  BOOST_CHECK(u3.deepEquals(u));
  // Copy assign
  expr::UnaryOp u4 = u;
  BOOST_CHECK(u4.deepEquals(u));
  // Move assign
  expr::UnaryOp u5 = std::move(u4);
  BOOST_CHECK(u5.deepEquals(u));
}

BOOST_AUTO_TEST_CASE(ConstructAndAssignBinaryOp) {
  expr::BinaryOp b(expr::BinaryOpType::ADD, expr::PlaceHolder(1u),
                   expr::Const(1u));
  // Copy construct
  expr::BinaryOp b2(b);
  BOOST_CHECK(b2.deepEquals(b));
  // Move construct
  expr::BinaryOp b3(std::move(b2));
  BOOST_CHECK(b3.deepEquals(b));
  // Copy assign
  expr::BinaryOp b4 = b;
  BOOST_CHECK(b4.deepEquals(b));
  // Move assign
  expr::BinaryOp b5 = std::move(b4);
  BOOST_CHECK(b5.deepEquals(b));
}

BOOST_AUTO_TEST_CASE(ConstructAndAssignTernaryOp) {
  expr::TernaryOp t(expr::TernaryOpType::CLAMP, expr::PlaceHolder(1u),
                    expr::Const(1u),
                    expr::Cast(expr::PlaceHolder(2u), poplar::FLOAT));
  // Copy construct
  expr::TernaryOp t2(t);
  BOOST_CHECK(t2.deepEquals(t));
  // Move construct
  expr::TernaryOp t3(std::move(t2));
  BOOST_CHECK(t3.deepEquals(t));
  // Copy assign
  expr::TernaryOp t4 = t;
  BOOST_CHECK(t4.deepEquals(t));
  // Move assign
  expr::TernaryOp t5 = std::move(t4);
  BOOST_CHECK(t5.deepEquals(t));
}
