// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/ExprOp.hpp"

namespace popops {

// Define function templates to do add or a multiply (based on a
// 'expr::BinaryOpType' parameter) with float and half
template <expr::BinaryOpType op, typename T> struct ElementOp {};

template <typename T> struct ElementOp<expr::BinaryOpType::ADD, T> {
  static T fn(T a, T b) { return a + b; }
};

template <typename T> struct ElementOp<expr::BinaryOpType::SUBTRACT, T> {
  static T fn(T a, T b) { return a - b; }
};

template <typename T> struct ElementOp<expr::BinaryOpType::MULTIPLY, T> {
  static T fn(T a, T b) { return a * b; }
};

} // namespace popops
