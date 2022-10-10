// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "SparsePartitionerOptions.hpp"

#include <gccs/StructHelper.hpp>

namespace popsparse {

static constexpr auto partitionerHelper =
    gccs::makeStructHelper(&PartitionerOptions::optimiseForSpeed,
                           &PartitionerOptions::useActualWorkerSplitCosts,
                           &PartitionerOptions::forceBucketSpills);

bool operator<(const PartitionerOptions &a, const PartitionerOptions &b) {
  return partitionerHelper.lt(a, b);
}

bool operator==(const PartitionerOptions &a, const PartitionerOptions &b) {
  return partitionerHelper.eq(a, b);
}

bool operator!=(const PartitionerOptions &a, const PartitionerOptions &b) {
  return !(a == b);
}

} // end namespace popsparse
