// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_SparsePartitionerOptions_hpp
#define popsparse_SparsePartitionerOptions_hpp

namespace popsparse {

struct PartitionerOptions {
  // Optimise bucket overflow allocation for speed. Overflow allocation would
  // attempt to allocate buckets that have the shortest distance to travel
  bool optimiseForSpeed = true;

  // If set uses actual worker split every time costs for a partition are
  // evaluated. This will give exact cost as the final "real" allocation, but
  // is expensive to compute. If not set, then all workers are assumed to be
  // used and the final allocation will actually be lower.
  bool useActualWorkerSplitCosts = false;

  // Test mode to force bucket spills
  bool forceBucketSpills = false;

  friend bool operator<(const PartitionerOptions &a,
                        const PartitionerOptions &b);
  friend bool operator==(const PartitionerOptions &a,
                         const PartitionerOptions &b);
  friend bool operator!=(const PartitionerOptions &a,
                         const PartitionerOptions &b);
};

} // end namespace popsparse

#endif // popsparse_SparsePartitionerOptions_hpp
