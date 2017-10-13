#ifndef __operationsdef_hpp__
#define __operationsdef_hpp__

namespace popstd {

enum TernaryOp {
  CLAMP,
  SELECT
};

enum BinaryOp {
  ADD,
  BITWISE_AND,
  BITWISE_OR,
  DIVIDE,
  EQUAL,
  GREATER_THAN_EQUAL,
  GREATER_THAN,
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
  SUBTRACT
};

enum UnaryOp {
  ABSOLUTE,
  BITWISE_NOT,
  CEIL,
  COS,
  EXPONENT,
  FLOOR,
  IS_FINITE,
  LOGARITHM,
  LOGICAL_NOT,
  NEGATE,
  SIGNUM,
  SIN,
  TANH,
  ROUND,
  SQRT,
  SQUARE
};

} // namespace popstd

#endif // __operationsdef_hpp__
