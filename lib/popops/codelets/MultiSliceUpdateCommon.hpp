// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef _MultiSliceUpdateCommon_hpp_
#define _MultiSliceUpdateCommon_hpp_

namespace popops {

// Search for thr lowest index in `indices` such that the entries including and
// above that index is >= `targetVal`.
// `indices` must be sorted in increasing order and can have repeated entries.
// If all entries are smaller than target value `numIndices` is returned.
int lowerBinarySearch(const int *indices, int numIndices, int targetVal);

// Search for thr largest index in `indices` such that the entries including and
// above that index is < `targetVal`.
// `indices` must be sorted in increasing order and can have repeated entries.
// If all entries are greater than target value of 0 is returned.
int upperBinarySearch(const int *indices, int numIndices, int targetVal);

} // namespace popops

#endif // #ifndef _MultiSliceUpdateCommon_hpp_
