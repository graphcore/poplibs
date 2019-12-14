// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef DYNAMIC_SLICE_INTERNAL_HPP
#define DYNAMIC_SLICE_INTERNAL_HPP
#include <iostream>
#include <memory>
#include <vector>

namespace popops {
namespace sliceInternal {
// How to partition work across tiles.
struct Partition {
  // How much to split processing of lookup indices between tiles.
  std::size_t lookupSplit;
  // How much to split the sliced/updated dimension of the
  // tensor to be sliced/updated between tiles.
  std::size_t slicedDimSplit;
  // How much to split the product of dimensions that are not
  // sliced/updated between tiles.
  std::size_t unslicedDimSplit;
  // Grain size for no. of elements in the product of dimensions that
  // are not sliced/updated on each tile.
  std::size_t unslicedGrainSize;
};
} // namespace sliceInternal

class SlicePlanInternal {
public:
  SlicePlanInternal() : isNull(true) {}

public:
  bool isNull;
  sliceInternal::Partition partition;

  // For validation, to identify the restrictions on what this
  // plan can be used to implement,
  std::size_t rank;
  std::vector<std::size_t> slicedDims;
  std::vector<std::size_t> slicedDimSizes;

  std::unique_ptr<SlicePlanInternal> clone() const {
    return std::make_unique<SlicePlanInternal>(*this);
  };
};
std::ostream &operator<<(std::ostream &o, const SlicePlanInternal &p);

} // namespace popops
#endif // DYNAMIC_SLICE_INTERNAL_HPP
