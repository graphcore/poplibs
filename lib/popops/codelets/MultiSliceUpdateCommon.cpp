// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MultiSliceUpdateCommon.hpp"

namespace popops {

namespace {
template <typename T> constexpr static inline T min(const T &a, const T &b) {
  return a < b ? a : b;
}
} // namespace

int lowerBinarySearch(const int *indices, int numIndices, int targetVal) {
  int low = 0, high = numIndices - 1;
  int startIndex = -1;
  while (low <= high) {
    int mid = (high - low) / 2 + low;
    if (indices[mid] >= targetVal) {
      high = mid - 1;
      startIndex = mid;
    } else {
      low = mid + 1;
    }
  }

  if (startIndex < 0)
    return numIndices;

  while (startIndex > 0) {
    if (indices[startIndex - 1] < targetVal) {
      break;
    }
    --startIndex;
  }
  return startIndex;
}

int upperBinarySearch(const int *indices, int numIndices, int targetVal) {
  int low = 0, high = numIndices - 1;
  int endIndex = -1;
  while (low <= high) {
    int mid = (high - low) / 2 + low;
    if (indices[mid] >= targetVal) {
      high = mid - 1;
    } else {
      low = mid + 1;
      endIndex = mid;
    }
  }

  if (endIndex < 0)
    return 0;

  while (endIndex + 1 < numIndices) {
    if (indices[endIndex + 1] > targetVal) {
      break;
    }
    ++endIndex;
  }
  return endIndex + 1;
}

MultiUpdateOpWorkSplit
multiUpdateOpWorkerDivision(bool hasAtomicWriteGranularity,
                            bool indicesAreSorted, unsigned offsetIndexBegin,
                            unsigned offsetIndexEnd, unsigned wid,
                            unsigned maxElementsPerWorker) {
  MultiUpdateOpWorkSplit s;
  const bool canSplit = !indicesAreSorted || hasAtomicWriteGranularity;
  if (canSplit) {
    s.offsetBegin =
        min(offsetIndexBegin + wid * maxElementsPerWorker, offsetIndexEnd);
    s.offsetEnd = min(offsetIndexBegin + (wid + 1) * maxElementsPerWorker,
                      offsetIndexEnd);
  } else {
    if (wid == 0) {
      s.offsetBegin = offsetIndexBegin;
      s.offsetEnd = offsetIndexEnd;
    } else {
      s.offsetBegin = 0;
      s.offsetEnd = 0;
    }
  }
  return s;
}

MultiGenericWorkSplit multiGenericWorkerDivision(
    bool hasAtomicWriteGranularity, bool indicesAreSorted,
    bool splitSingleRegion, unsigned offsetIndexBegin, unsigned offsetIndexEnd,
    unsigned regionSize, unsigned wid, unsigned maxElementsPerWorker) {
  MultiGenericWorkSplit s;
  // All work is performed by a single worker if indices are sorted and the
  // region size is not a multiple of 4 bytes to avoid read/write hazards.
  // Ideally we want to divide the number of elements after pruning the indices
  // list.
  const bool canSplit = !indicesAreSorted || hasAtomicWriteGranularity;
  if (canSplit) {
    if (splitSingleRegion) {
      s.offsetBegin = offsetIndexBegin;
      s.offsetEnd = offsetIndexEnd;
      s.regionBegin = min(regionSize, maxElementsPerWorker * wid);
      s.regionEnd = min(regionSize, maxElementsPerWorker * (wid + 1));
    } else {
      s.offsetBegin =
          min(offsetIndexBegin + wid * maxElementsPerWorker, offsetIndexEnd);
      s.offsetEnd = min(offsetIndexBegin + (wid + 1) * maxElementsPerWorker,
                        offsetIndexEnd);
      s.regionBegin = 0;
      s.regionEnd = regionSize;
    }
  } else {
    if (wid == 0) {
      s.offsetBegin = offsetIndexBegin;
      s.offsetEnd = offsetIndexEnd;
      s.regionBegin = 0;
      s.regionEnd = regionSize;
    } else {
      s.offsetBegin = 0;
      s.offsetEnd = 0;
      s.regionBegin = 0;
      s.regionEnd = 0;
    }
  }
  return s;
}

} // namespace popops
