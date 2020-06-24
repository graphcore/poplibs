// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_SparsityParams_hpp
#define popsparse_SparsityParams_hpp

#include <ostream>
#include <tuple>

namespace popsparse {
namespace dynamic {

/// Sparsity type
enum class SparsityType {
  /// Sparsity is defined at an element level
  Element
};

std::ostream &operator<<(std::ostream &os, const SparsityType &t);

/// Sparsity structure
enum class SparsityStructure { Unstructured };

std::ostream &operator<<(std::ostream &os, const SparsityStructure &s);

struct SparsityParams {

  /// sparsity type
  SparsityType type;

  /// sparsity structure
  SparsityStructure structure;

  SparsityParams(SparsityType type = SparsityType::Element,
                 SparsityStructure structure = SparsityStructure::Unstructured)
      : type(type), structure(structure){};

  SparsityParams(const SparsityParams &) = default;

  friend bool operator<(const SparsityParams &a, const SparsityParams &b);

  friend std::ostream &operator<<(std::ostream &os, const SparsityParams &p);
};

} // namespace dynamic
} // namespace popsparse

#endif // popsparse_SparsityParams_hpp
