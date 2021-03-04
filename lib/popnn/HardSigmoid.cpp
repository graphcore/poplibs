// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "HardSigmoid.hpp"
#include "popops/ElementWise.hpp"

#include <cassert>

namespace {
float slopeFromDiscontinuity(float discontinuity) {
  return (0.5f * 1 / discontinuity);
}
} // namespace

namespace popnn {

std::unique_ptr<popops::expr::Expr> HardSigmoidExpr::activation() {
  using namespace popops;

  const auto slope = slopeFromDiscontinuity(discontinuity_);
  auto scaledAdd =
      expr::Add(expr::Mul(expr::_1, expr::Const(slope)), expr::Const(0.5f));
  auto clamp = expr::Clamp(scaledAdd, expr::Const(0.0f), expr::Const(1.0f));

  return clamp.clone();
}

std::unique_ptr<popops::expr::Expr>
HardSigmoidExpr::gradient(const poplar::Type &elementType) {
  using namespace popops;

  auto mask = expr::Cast(
      expr::Lte(expr::Abs(expr::_1), expr::Const(discontinuity_)), elementType);

  const auto slope = slopeFromDiscontinuity(discontinuity_);
  auto derivative = expr::Mul(mask, expr::Const(slope));
  auto apply = expr::Mul(derivative, expr::_2);

  return apply.clone();
}

} // end namespace popnn
