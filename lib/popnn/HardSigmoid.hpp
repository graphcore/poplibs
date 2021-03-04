// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popnn_HardSigmoid_hpp
#define popnn_HardSigmoid_hpp

#include "popops/Expr.hpp"

namespace popnn {

class HardSigmoidExpr {
public:
  // Return a unary argument hard sigmoid expression, equivalent to max(0,
  // min(1, 0.2*x + 0.5)) The argument represents the input tensor.
  static std::unique_ptr<popops::expr::Expr> activation();

  // Return a binary argument hard sigmoid gradient expression,
  // equivalent to 0.2 if -2.5 <= x <= 2.5 and 0 otherwise.
  // The arguments represent the output and output gradient tensors.
  static std::unique_ptr<popops::expr::Expr>
  gradient(const poplar::Type &elementType);

private:
  static constexpr float discontinuity_ = 2.5f;
};

} // end namespace popnn

#endif // popnn_HardSigmoid_hpp