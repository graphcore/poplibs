// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

namespace popops {

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

} // namespace popops
