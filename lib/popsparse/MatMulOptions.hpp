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

namespace static_ {
struct MatMulOptions {
  // This can eventually be a plan constraint
  unsigned numBands = 0;
  // This can eventually be a plan constraint
  unsigned nSplit = 0;
  // Enable pre-processing of matrix in the partitioner. Permutes rows and
  // columns so that sparsity used in planning excludes some rows and/or columns
  // The plan selected could change and thus changes overall memory/cycle
  // performance.
  // Internal option
  bool enablePreprocessing = true;
  // partials type. Internal option not exposed because of restriction that
  // partial type = data type
  poplar::Type partialsType = poplar::FLOAT;
  bool verboseLogging = false;
  double availableMemoryProportion = 0.6;

  friend bool operator<(const MatMulOptions &a, const MatMulOptions &b);
  friend bool operator!=(const MatMulOptions &a, const MatMulOptions &b);
  friend bool operator==(const MatMulOptions &a, const MatMulOptions &b);
  friend std::ostream &operator<<(std::ostream &os, const MatMulOptions &o);
};

MatMulOptions parseMatMulOptionFlags(const poplar::OptionFlags &flags);

} // end namespace static_

} // end namespace popsparse

#endif // popsparse_MatMulOptions_hpp
