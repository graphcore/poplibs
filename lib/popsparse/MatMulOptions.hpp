// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_MatMulOptions_hpp
#define popsparse_MatMulOptions_hpp

#include <ostream>

#include <poplar/OptionFlags.hpp>
#include <poplar/Type.hpp>

#include "SparsePartitionerOptions.hpp"

namespace popsparse {
namespace dynamic {

struct MatMulOptions {
  double availableMemoryProportion = 0.6;
  double metaInfoBucketOversizeProportion = 0.3;
  poplar::Type partialsType = poplar::FLOAT;
  bool sharedBuckets = true;
  PartitionerOptions partitioner;

  friend bool operator<(const MatMulOptions &a, const MatMulOptions &b);
  friend bool operator!=(const MatMulOptions &a, const MatMulOptions &b);
  friend bool operator==(const MatMulOptions &a, const MatMulOptions &b);
  friend std::ostream &operator<<(std::ostream &os, const MatMulOptions &o);
};

MatMulOptions parseMatMulOptionFlags(const poplar::OptionFlags &flags);

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_MatMulOptions_hpp
