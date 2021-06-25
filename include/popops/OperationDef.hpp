// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Define types of operations used in Reduce/MultiUpdate
 *
 */

#ifndef popops_OperationDef_hpp
#define popops_OperationDef_hpp

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
  LOG_ADD,     ///< Reduce using acc = a+log(1+exp(b-a))
};

} // End namespace popops

#endif // popops_OperationDef_hpp
