// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "poplibs_support/StridedRegions.hpp"
#include "poplibs_support/VectorUtils.hpp"

#include "poplibs_support/logging.hpp"

#include <numeric>

using namespace poplar;

namespace poplibs_support {

static inline std::int64_t strideBetween(std::size_t a, std::size_t b) {
  return std::minus<std::int64_t>{}(b, a);
}

static bool tryMergeStridedRegionPair(StridedRegion &lhs,
                                      const StridedRegion &rhs) {
  const auto numLHSLevels = lhs.stride.size();
  const auto numRHSLevels = rhs.stride.size();

  if (numLHSLevels == numRHSLevels) {
    const bool allExceptLastCountMatch =
        lhs.stride.empty() ||
        (std::equal(lhs.stride.begin(), std::prev(lhs.stride.end()),
                    rhs.stride.begin(), std::prev(rhs.stride.end())) &&
         lhs.stride.back().stride == rhs.stride.back().stride);
    const bool canAddOuterStride =
        lhs.stride.empty() ||
        (allExceptLastCountMatch &&
         lhs.stride.back().count == rhs.stride.back().count);

    if (lhs.getFinalOffset() == rhs.offset && allExceptLastCountMatch) {
      lhs.stride.back().count += rhs.stride.back().count;
      return true;
    } else if (canAddOuterStride) {
      const auto offset = strideBetween(lhs.offset, rhs.offset);
      // For the timebeing we only support unsigned strides.
      if (offset >= 0) {
        lhs.stride.emplace_back(offset, 2);
        return true;
      }
    }
  } else if (numLHSLevels == numRHSLevels + 1) {
    const bool allInnerMatch =
        std::equal(rhs.stride.begin(), rhs.stride.end(), lhs.stride.begin());

    if (lhs.getFinalOffset() == rhs.offset && allInnerMatch) {
      ++lhs.stride.back().count;
      return true;
    }
  } else if (numRHSLevels == numLHSLevels + 1) {
    const bool allInnerMatch =
        std::equal(lhs.stride.begin(), lhs.stride.end(), rhs.stride.begin());
    if (lhs.getFinalOffset() == rhs.offset && allInnerMatch) {
      lhs.stride.emplace_back(rhs.stride.back().stride,
                              rhs.stride.back().count + 1);
      return true;
    }
  }
  return false;
}

static bool mergeStridedRegionsOnce(StridedRegionList &regions) {
  bool madeAnyChange = false;

  assert(!regions.empty());
  auto outIt = regions.begin();
  auto inIt = std::next(regions.cbegin());

  while (inIt != regions.end()) {
    bool merged = tryMergeStridedRegionPair(*outIt, *inIt);
    if (!merged) {
      ++outIt;
      *outIt = *inIt;
    }
    ++inIt;
    madeAnyChange |= merged;
  }

  ++outIt;

  regions.erase(outIt, regions.end());
  return madeAnyChange;
}

template <typename F>
static void iterateIntervals(const StridedRegion &r, const F &f) {
  const auto numDims = r.stride.size();
  if (numDims == 0) {
    f(r.offset, r.offset + 1);
    return;
  }
  std::vector<std::size_t> indices(numDims);
  std::size_t offsetDueToStride = 0;
  bool innermostIsContiguous = r.stride.front().stride == 1;
  const std::size_t sliceElems =
      innermostIsContiguous ? r.stride.front().count : 1;
  while (true) {
    const auto lower = r.offset + offsetDueToStride;
    const auto upper = lower + sliceElems;
    f(lower, upper);
    // The following increments the index of the inner-most level of
    // the stride, and advances the offset due to the stride. Because
    // this is multi-level, it wraps indices around for each dimension.
    unsigned carryDim = 0;
    while (carryDim < numDims) {
      const bool isContiguousDim = carryDim == 0 && innermostIsContiguous;
      if (isContiguousDim) {
        indices[carryDim] += sliceElems;
        offsetDueToStride += sliceElems;
      } else {
        ++indices[carryDim];
        offsetDueToStride += r.stride[carryDim].stride;
      }
      // If the index is within the limit of this level of the stride
      // we are dne
      if (indices[carryDim] < r.stride[carryDim].count) {
        break;
      }
      indices[carryDim] = 0;
      offsetDueToStride -= r.stride[carryDim].stride * r.stride[carryDim].count;
      ++carryDim;
    }
    if (carryDim == indices.size()) {
      break;
    }
  }
}

std::ostream &operator<<(std::ostream &os, const StrideAndCount &s) {
  os << std::showpos << s.stride << "x" << s.count;
  return os;
}

std::size_t getTotalCount(const StrideDesc &s) {
  if (s.empty()) {
    return 1;
  }
  return std::accumulate(s.begin(), s.end(), std::size_t(1),
                         [](const std::size_t total, const StrideAndCount &s) {
                           return total * s.count;
                         });
}

std::size_t getContiguousRegionSize(const StrideDesc &s) {
  if (s.empty()) {
    return 1;
  }
  if (s.front().stride != 1) {
    return 1;
  }
  return s.front().count;
}

std::size_t getTotalContiguousRegions(const StrideDesc &s) {
  return getTotalCount(s) / getContiguousRegionSize(s);
}

std::size_t getTotalOffset(const StrideDesc &s) {
  if (s.empty()) {
    return 1;
  }
  return s.back().stride * s.back().count;
}

std::ostream &operator<<(std::ostream &os, const StrideDesc &s) {
  bool needsComma = false;

  os << "[";
  for (const auto &entry : s) {
    if (needsComma) {
      os << ",";
    }
    os << entry;
    needsComma = true;
  }
  os << "]";

  return os;
}

std::ostream &operator<<(std::ostream &os, const StridedRegion &r) {
  os << "{offset=" << r.offset << ",stride=" << r.stride << "}";
  return os;
}

bool mergeStridedRegions(StridedRegionList &regions) {
  bool madeChange, madeAnyChange = false;
  do {
    madeChange = mergeStridedRegionsOnce(regions);
    madeAnyChange |= madeChange;
  } while (madeChange);
  return madeAnyChange;
}

static void sliceStridedRegionByInterval(const Tensor &t,
                                         const StridedRegion &r,
                                         std::vector<Tensor> &slices) {
  assert(t.rank() == 1);
  slices.reserve(slices.size() + r.numContiguousRegions());
  iterateIntervals(r, [&](std::size_t lower, std::size_t upper) {
    slices.emplace_back(t.slice(lower, upper));
  });
}

poplar::Tensor sliceStridedRegions(const Tensor &t,
                                   const StridedRegionList &regions) {
  assert(t.rank() == 1);

  std::vector<Tensor> toConcat;
  toConcat.reserve(regions.size());
  for (const auto &r : regions) {
    // Given an offset and a set of strides, we work backwards to find:
    //
    //  1) a shape to apply to the input tensor.
    //  2) a slice of that shape.
    //  3) a dim-shuffle of the sliced regions.
    //
    // applied in that order.
    //
    // A single strided region can represent more than the above
    // allows. In the case that the strided region cannot be represented
    // using the above we currently fall back to slicing each region
    // as described by the strided region.

    auto numDims = r.stride.size();
    if (numDims == 0) {
      sliceStridedRegionByInterval(t, r, toConcat);
      continue;
    }

    // First find a permutation from the post-shuffle order to
    // the original order by sorting the indices of each dimension
    // by their stride.
    std::vector<unsigned> permutation(numDims);
    // NOTE: Strides are specified from innermost to outermost but
    // tensor shapes are specified from outermost to innermost
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(),
              [&](const auto a, const auto b) {
                return r.stride[a].stride < r.stride[b].stride;
              });

    // Next find the original shape we apply to the input
    // out of which we will slice the elements we want
    // (preSliceShape).

    // If at any point we discover we can't represent this strided
    // region as shape, slice, and shuffle, we fallback to slices.
    bool hasSimpleRepresentation = true;

    const auto innermostDimIsContiguous =
        r.stride[permutation.front()].stride == 1;

    std::vector<std::size_t> preSliceShape;
    std::vector<std::size_t> shape;
    preSliceShape.reserve(numDims);
    shape.reserve(numDims);

    std::size_t totalElems = 1;
    for (std::size_t i = 0; i < permutation.size(); ++i) {
      const auto d = permutation[i];
      const auto stride = r.stride[d].stride;
      // We can't represent broadcasting/aliasing in our shape +
      // slice so don't try.
      if (stride == 0 ||
          (i > 0 && stride == r.stride[permutation[i - 1]].stride) ||
          stride % totalElems != 0) {
        hasSimpleRepresentation = false;
        break;
      }
      // A StridedRegion implicitly represents a single element
      if (i == 0 && !innermostDimIsContiguous) {
        shape.emplace_back(1);
      }
      if (!(i == 0 && innermostDimIsContiguous)) {
        preSliceShape.emplace_back(stride / totalElems);
        totalElems *= preSliceShape.back();
      }
      shape.emplace_back(r.stride[d].count);
    }
    if (!hasSimpleRepresentation) {
      sliceStridedRegionByInterval(t, r, toConcat);
      continue;
    }
    preSliceShape.emplace_back(r.stride[permutation.back()].count);
    totalElems *= preSliceShape.back();

    // If the innermost dimension after sorting does not have a stride
    // of 1 then we will end up with one more dimensions than levels
    // in the strided region.
    if (!innermostDimIsContiguous) {
      ++numDims;
      permutation.resize(numDims);
      for (std::size_t i = numDims - 1; i > 0; --i) {
        permutation[i] = permutation[i - 1] + 1;
      }
      permutation[0] = 0;
    }

    // Try and factor the starting offset for this region into the pre-slice
    // shape.
    std::size_t flatOffset = r.offset;
    std::vector<std::size_t> begins, ends;
    begins.resize(numDims);
    ends.resize(numDims);
    std::size_t innerElems = 1;
    for (std::size_t dim = 0; dim < preSliceShape.size(); ++dim) {
      const auto dimSize = preSliceShape[dim];
      const auto begin = (flatOffset / innerElems) % dimSize;
      if (begin + shape[dim] <= dimSize) {
        begins[dim] = begin;
        flatOffset -= begin * innerElems;
      }
      ends[dim] = begins[dim] + shape[dim];
      innerElems *= dimSize;
    }

    // Unfortunately it is possible to construct arbitrary examples where
    // the given series of shape, slice + shuffle are valid but the
    // intermediate shape references more elements than are in the tensor.
    // Just fallback if this is the case.
    if (flatOffset + totalElems > t.numElements()) {
      sliceStridedRegionByInterval(t, r, toConcat);
      continue;
    }

    // Because our strided representation has innermost dimension first
    // and tensors have innermost dimension last, reverse all
    // the shapes etc. we built.
    std::reverse(preSliceShape.begin(), preSliceShape.end());
    std::reverse(shape.begin(), shape.end());
    std::reverse(begins.begin(), begins.end());
    std::reverse(ends.begin(), ends.end());
    // The permutation also needs its contents modifying as it refers
    // to indices of dimensions.
    std::reverse(permutation.begin(), permutation.end());
    for (auto &entry : permutation) {
      entry = permutation.size() - 1 - entry;
    }
    // Get the inverse permutation as the original permutation
    // gives the transformation from final order to pre-shuffle order
    // and we want to go the other way.
    permutation = inversePermutation(permutation);

    Tensor slice = t.flatten().slice(flatOffset, flatOffset + totalElems);
    slice = slice.reshape(preSliceShape).slice(begins, ends);
    slice = slice.dimShuffle(permutation);
    toConcat.emplace_back(slice.flatten());
  }
  return concat(toConcat);
}

void appendIntervalsForStridedRegion(const StridedRegion &r,
                                     std::vector<Interval> &intervals) {
  iterateIntervals(r, [&](std::size_t lower, std::size_t upper) {
    intervals.emplace_back(lower, upper);
  });
}

std::vector<Interval> intervalsForStridedRegion(const StridedRegion &r) {
  std::vector<Interval> intervals;
  appendIntervalsForStridedRegion(r, intervals);
  return intervals;
}

std::vector<Interval>
intervalsForStridedRegions(const StridedRegionList &regions) {
  std::vector<Interval> intervals;
  for (const auto &r : regions) {
    appendIntervalsForStridedRegion(r, intervals);
  }
  return intervals;
}

} // end namespace poplibs_support
