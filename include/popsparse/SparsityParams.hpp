// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Parameters used for sparse tensors.
 */

#ifndef popsparse_SparsityParams_hpp
#define popsparse_SparsityParams_hpp

#include <array>
#include <ostream>
#include <tuple>
#include <utility>

namespace popsparse {
namespace dynamic {

/// Sparsity type.
enum class SparsityType {
  /// Sparsity is defined at an element level.
  Element,

  /// Sparsity is defined at a block level. The matrix is made up of blocks
  /// with each of these block are either all zero or not.
  Block,
};

std::ostream &operator<<(std::ostream &os, const SparsityType &t);

/// Sparsity structure.
enum class SparsityStructure { Unstructured };

std::ostream &operator<<(std::ostream &os, const SparsityStructure &s);

struct SparsityParams {

  /// sparsity type.
  SparsityType type;

  /// sparsity structure.
  SparsityStructure structure;

  /// Block dimensions
  std::array<std::size_t, 2> blockDimensions;

  SparsityParams(SparsityType type_ = SparsityType::Element,
                 SparsityStructure structure_ = SparsityStructure::Unstructured,
                 std::array<std::size_t, 2> blockDimensions_ = {1, 1}) {
    // This parameter is redundant and should be removed from the constructor.
    (void)type_;
    type = blockDimensions_[0] * blockDimensions_[1] == 1
               ? SparsityType::Element
               : SparsityType::Block;
    structure = structure_;
    blockDimensions = std::move(blockDimensions_);
  }

  SparsityParams(const SparsityParams &) = default;

  friend bool operator<(const SparsityParams &a, const SparsityParams &b);
  friend bool operator==(const SparsityParams &a, const SparsityParams &b);
  friend bool operator!=(const SparsityParams &a, const SparsityParams &b);

  friend std::ostream &operator<<(std::ostream &os, const SparsityParams &p);
};

} // namespace dynamic
} // namespace popsparse

#endif // popsparse_SparsityParams_hpp
