// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Operation_hpp
#define popops_Operation_hpp

#include <iosfwd>

namespace popops {

/// Type of operation in a reduction
enum class Operation {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND,
  LOGICAL_OR,
  SQUARE_ADD,
};

std::istream &operator>>(std::istream &is, Operation &op);

std::ostream &operator<<(std::ostream &os, const Operation &op);

} // End namespace popops

#endif // popops_Operation_hpp
