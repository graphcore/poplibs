// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "popops/ExprOp.hpp"

namespace popops {

// Define function templates to do add or a multiply (based on a
// 'expr::BroadcastOpType' parameter) with float and half
template <expr::BroadcastOpType op, typename T> struct ElementOp {};

template <typename T> struct ElementOp<expr::BroadcastOpType::ADD, T> {
  static T fn(T a, T b) { return a + b; }
};

template <typename T> struct ElementOp<expr::BroadcastOpType::MULTIPLY, T> {
  static T fn(T a, T b) { return a * b; }
};

} // namespace popops
