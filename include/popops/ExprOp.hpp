// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef _popops_ExprOp_hpp_
#define _popops_ExprOp_hpp_

namespace popops { namespace expr {

// Enum classes uses for expressions.

enum class TernaryOpType {
  CLAMP,
  SELECT
};

enum class BinaryOpType {
  ADD,
  ATAN2,
  BITWISE_AND,
  BITWISE_OR,
  DIVIDE,
  EQUAL,
  GREATER_THAN_EQUAL,
  GREATER_THAN,
  INV_STD_DEV_TO_VARIANCE,
  LESS_THAN_EQUAL,
  LOGICAL_AND,
  LOGICAL_OR,
  LESS_THAN,
  MAXIMUM,
  MINIMUM,
  MULTIPLY,
  NOT_EQUAL,
  POWER,
  REMAINDER,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  SHIFT_RIGHT_SIGN_EXTEND,
  SUBTRACT,
  VARIANCE_TO_INV_STD_DEV
};

enum class UnaryOpType {
  ABSOLUTE,
  BITWISE_NOT,
  CEIL,
  COS,
  COUNT_LEADING_ZEROS,
  EXPONENT,
  EXPONENT_MINUS_ONE,
  FLOOR,
  IS_FINITE,
  LOGARITHM,
  LOGARITHM_ONE_PLUS,
  LOGICAL_NOT,
  NEGATE,
  POPCOUNT,
  SIGNUM,
  SIN,
  TANH,
  ROUND,
  SQRT,
  SQUARE,
  SIGMOID,
  RSQRT
};

enum class BroadcastOpType {
  ADD,
  INV_STD_DEV_TO_VARIANCE,
  MULTIPLY,
  SUBTRACT,
  VARIANCE_TO_INV_STD_DEV
};

}} // end namespace popops::expr

#endif // _popops_ExprOp_hpp_
