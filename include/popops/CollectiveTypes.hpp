// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Supported types for replicated and non-replicated collectives.
 *
 */

#ifndef popops_CollectiveTypes_hpp
#define popops_CollectiveTypes_hpp

#include <gccs/CompilerFeatures.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Operation.hpp>

namespace popops {
/**
 *
 * Supported collective operators.
 * \deprecated Use gcl::CollectiveOperator instead.
 */
enum class GC_DEPRECATED_MSG("Use gcl::CollectiveOperator instead")
    CollectiveOperator {
      ADD,
      MEAN,
      MUL,
      MIN,
      MAX,
      LOGICAL_AND, ///< Only supports boolean operands.
      LOGICAL_OR,  ///< Only supports boolean operands.
      SQUARE_ADD,  ///< Squares each element before applying ADD reduction.
      LOCAL,       ///< Do nothing and keep the local value.
    };

/**
 * Parse token from input stream \p is to \p op. Valid input values are the
 * stringified enumerations, for example "ADD" or "MUL".
 * \return The original input stream.
 * \deprecated This operator overload has been deprecated and will be removed
 * in a future release.
 */
GC_DEPRECATED std::istream &operator>>(std::istream &is,
                                       CollectiveOperator &op);

/**
 * Write \p op to output stream \p os. The value written is the stringified
 * enumeration, for example "ADD" or "MUL".
 * \return The original output stream.
 * \deprecated This operator overload has been deprecated and will be removed
 * in a future release.
 */
GC_DEPRECATED std::ostream &operator<<(std::ostream &os,
                                       const CollectiveOperator &op);

/**
 *
 * Convert from popops::Operation to popops::CollectiveOperator
 * \deprecated Use gcl::operationToCollectiveOperator instead.
 */
GC_DEPRECATED_MSG("Use gcl::operationToCollectiveOperator instead")
CollectiveOperator operationToCollectiveOperator(const Operation &col);

/**
 * Represents a section of a tensor mapped to an IPU.
 * \deprecated Use gcl::Chunk instead.
 */
struct GC_DEPRECATED_MSG("Use gcl::Chunk instead") Chunk {
  poplar::Tensor tensor;
  /// Ring index (data parallel index).
  unsigned index;
  /// Offset within rank (model parallel index).
  unsigned offset;
  Chunk() = default;
  Chunk(poplar::Tensor tensor, unsigned index, unsigned offset)
      : tensor(tensor), index(index), offset(offset) {}
};

/**
 * A vector of Chunk data.
 * \deprecated Use gcl::Chunks instead.
 */
struct GC_DEPRECATED_MSG("Use gcl::Chunks instead") Chunks {
  /// Used to undo shuffles introduced by scatter.
  poplar::Tensor originalInput;
  /// Chunks produced by the scatter step.
  std::vector<Chunk> chunks;
  Chunks() = default;
  Chunks(unsigned size) : chunks(std::vector<Chunk>(size)) {}
};
} // namespace popops

#endif // popops_CollectiveTypes_hpp