#ifndef __operationsdef_hpp__
#define __operationsdef_hpp__

namespace popstd {

enum BinaryOp {
  ADD,
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
  CEIL,
  EXPONENT,
  FLOOR,
  LOGARITHM,
  LOGICAL_NOT,
  NEGATE,
  SIGNUM,
  TANH
};

} // namespace popstd

#endif // __operationsdef_hpp__
