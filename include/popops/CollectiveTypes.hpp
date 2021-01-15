// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support types for replicated and non-replicated collectives.
 *
 */

#ifndef popops_CollectiveTypes_hpp
#define popops_CollectiveTypes_hpp

#include <poplar/Tensor.hpp>
#include <popops/Operation.hpp>

namespace popops {
/**
 * Supported collective operators.
 */
enum class CollectiveOperator {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND, ///< Only supports boolean operands.
  LOGICAL_OR,  ///< Only supports boolean operands.
  SQUARE_ADD,  ///< Squares each element before applying ADD reduction.
  LOCAL,       ///< Do nothing and keep the local value.
};

/**
 * Parse token from input stream \is to \op. Valid input values are the
 * stringified enumerations, for example "ADD" or "MUL".
 * \return The original input stream.
 */
std::istream &operator>>(std::istream &is, CollectiveOperator &op);

/**
 * Write \op to output stream \os. The value written is the stringified
 * enumeration, for example "ADD" or "MUL".
 * \return The original output stream.
 */
std::ostream &operator<<(std::ostream &os, const CollectiveOperator &op);

/** Convert from popops::Operation to popops::CollectiveOperator */
CollectiveOperator operationToCollectiveOperator(const Operation &col);

/// Represents a section of a tensor mapped to an IPU.
struct Chunk {
  poplar::Tensor tensor;
  /// Ring index (data parallel index).
  unsigned index;
  /// Offset within rank (model parallel index).
  unsigned offset;
  Chunk() = default;
  Chunk(poplar::Tensor tensor, unsigned index, unsigned offset)
      : tensor(tensor), index(index), offset(offset) {}
};

/// A vector of Chunk data.
struct Chunks {
  /// Used to undo shuffles introduced by scatter.
  poplar::Tensor originalInput;
  /// Chunks produced by the scatter step.
  std::vector<Chunk> chunks;
  Chunks() = default;
  Chunks(unsigned size) : chunks(std::vector<Chunk>(size)) {}
};
} // namespace popops

#endif // popops_CollectiveTypes_hpp