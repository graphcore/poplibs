// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Define types of operations used in a reduce.
 *
 */

#ifndef popops_Operation_hpp
#define popops_Operation_hpp

#include <iosfwd>

namespace popops {

/// Type of operation to use in a reduction.
/// See reduce() for example use.
enum class Operation {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND, ///< Only supports boolean operands.
  LOGICAL_OR,  ///< Only supports boolean operands.
  SQUARE_ADD,  ///< Squares each element before applying ADD reduction.
};

/// Parse token from input stream \is to \op. Valid input values are the
/// stringified enumerations, for example "ADD" or "MUL".
/// \return The original input stream.
std::istream &operator>>(std::istream &is, Operation &op);

/// Write \op to output stream \os. The value written is the stringified
/// enumeration, for example "ADD" or "MUL".
/// \return The original output stream.
std::ostream &operator<<(std::ostream &os, const Operation &op);

} // End namespace popops

#endif // popops_Operation_hpp
