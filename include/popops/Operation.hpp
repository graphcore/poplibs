// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popops_Operation_hpp
#define popops_Operation_hpp

#include <iosfwd>

namespace popops {

/// Type of operation to use in a reduction.
/// See Reduce.hpp for example usage.
enum class Operation {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND, // Only supports boolean operands.
  LOGICAL_OR,  // Only supports boolean operands.
  SQUARE_ADD,  // Squares each element before applying ADD reduction.
};

/// Parse token from istream to Operation, valid input values are the
/// stringified enum, e.g. "ADD", "MUL", ...
std::istream &operator>>(std::istream &is, Operation &op);

/// Write Operation to ostream, value written is the stringified enum,
/// e.g. "ADD", "MUL", ...
std::ostream &operator<<(std::ostream &os, const Operation &op);

} // End namespace popops

#endif // popops_Operation_hpp
