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

struct MultiGenericWorkSplit {
  unsigned offsetBegin; // start offset processed by worker
  unsigned offsetEnd;   // end offset processed by worker
  unsigned regionBegin; // Begin region offset processed by worker
  unsigned regionEnd;   // End of region offset processed by worker
};

// Work division without region split
struct MultiUpdateOpWorkSplit {
  unsigned offsetBegin; // start offset processed by worker
  unsigned offsetEnd;   // end offset processed by worker
};

// return amount of work to be done by worker
MultiGenericWorkSplit multiGenericWorkerDivision(
    bool hasAtomicWriteGranularity, bool indicesAreSorted,
    bool splitSingleRegion, unsigned offsetIndexBegin, unsigned offsetIndexEnd,
    unsigned regionSize, unsigned wid, unsigned maxElementsPerWorker);

MultiUpdateOpWorkSplit
multiUpdateOpWorkerDivision(bool hasAtomicWriteGranularity,
                            bool indicesAreSorted, unsigned offsetIndexBegin,
                            unsigned offsetIndexEnd, unsigned wid,
                            unsigned maxElementsPerWorker);
} // namespace popops

#endif // #ifndef _MultiSliceUpdateCommon_hpp_
