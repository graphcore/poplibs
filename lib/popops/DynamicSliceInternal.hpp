// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef DYNAMIC_SLICE_INTERNAL_HPP
#define DYNAMIC_SLICE_INTERNAL_HPP
#include <iostream>
#include <memory>
#include <vector>

#include <boost/functional/hash.hpp>

namespace popops {
namespace sliceInternal {
// How to partition work across tiles.
template <typename T> struct Partition {
  // When the base tensor should be copied to tiles ahead of a serialised
  // slicing loop.
  T precopyBaseWhenSerialSlice;
  // How much to serialise processing of indices.
  T lookupSerialSplit;
  // How much to split processing of lookup indices between tiles.
  T lookupParallelSplit;
  // How much to split the sliced/updated dimension of the
  // tensor to be sliced/updated between tiles.
  T slicedDimSplit;
  // How much to split the product of dimensions that are not
  // sliced/updated between tiles.
  T unslicedDimSplit;
  // How much to split the group dimension
  T groupSplit;
  // Grain size for no. of elements in the product of dimensions that
  // are not sliced/updated on each tile.
  T unslicedGrainSize;
};

template <typename T>
bool operator<(const Partition<T> &a, const Partition<T> &b) {
  return std::tie(a.precopyBaseWhenSerialSlice, a.lookupSerialSplit,
                  a.lookupParallelSplit, a.slicedDimSplit, a.unslicedDimSplit,
                  a.groupSplit, a.unslicedGrainSize) <
         std::tie(b.precopyBaseWhenSerialSlice, b.lookupSerialSplit,
                  b.lookupParallelSplit, b.slicedDimSplit, b.unslicedDimSplit,
                  b.groupSplit, b.unslicedGrainSize);
}

template <typename T>
bool operator==(const Partition<T> &a, const Partition<T> &b) {
  return std::tie(a.precopyBaseWhenSerialSlice, a.lookupSerialSplit,
                  a.lookupParallelSplit, a.slicedDimSplit, a.unslicedDimSplit,
                  a.groupSplit, a.unslicedGrainSize) ==
         std::tie(b.precopyBaseWhenSerialSlice, b.lookupSerialSplit,
                  b.lookupParallelSplit, b.slicedDimSplit, b.unslicedDimSplit,
                  b.groupSplit, b.unslicedGrainSize);
}

} // namespace sliceInternal

class SlicePlanInternal {
public:
  SlicePlanInternal() : isNull(true), groupSize(1) {}

public:
  bool isNull;
  bool useIndicesOrderingInfo;
  bool validateIndices;
  sliceInternal::Partition<std::size_t> partition;

  // For validation, to identify the restrictions on what this
  // plan can be used to implement,
  std::size_t rank;
  unsigned groupSize;
  std::vector<std::size_t> slicedDims;
  std::vector<std::size_t> slicedDimSizes;

  // Excluded from the hash and comparison operators intentionally as it
  // is a function of the other plan parameters.
  std::size_t startTile;

  std::unique_ptr<SlicePlanInternal> clone() const {
    return std::make_unique<SlicePlanInternal>(*this);
  };
};

bool operator<(const SlicePlanInternal &a,
               const SlicePlanInternal &b) noexcept {
  return std::tie(a.isNull, a.useIndicesOrderingInfo, a.partition, a.rank,
                  a.slicedDims, a.slicedDimSizes, a.groupSize) <
         std::tie(b.isNull, b.useIndicesOrderingInfo, b.partition, b.rank,
                  b.slicedDims, b.slicedDimSizes, b.groupSize);
}

bool operator==(const SlicePlanInternal &a,
                const SlicePlanInternal &b) noexcept {
  return std::tie(a.isNull, a.useIndicesOrderingInfo, a.partition, a.rank,
                  a.slicedDims, a.slicedDimSizes, a.groupSize) ==
         std::tie(b.isNull, b.useIndicesOrderingInfo, b.partition, b.rank,
                  b.slicedDims, b.slicedDimSizes, b.groupSize);
}

std::ostream &operator<<(std::ostream &o, const SlicePlanInternal &p);

} // namespace popops
#endif // DYNAMIC_SLICE_INTERNAL_HPP
