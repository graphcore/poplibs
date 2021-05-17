// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_StridedRegions_hpp
#define poplibs_support_StridedRegions_hpp

#include <boost/container/small_vector.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Tensor.hpp>

#include <ostream>
#include <vector>

namespace poplibs_support {

struct StrideAndCount {
  std::size_t stride;
  std::size_t count;
  StrideAndCount(std::size_t stride, std::size_t count) noexcept
      : stride(stride), count(count) {}
};

inline bool operator==(const StrideAndCount &a, const StrideAndCount &b) {
  return std::tie(a.stride, a.count) == std::tie(b.stride, b.count);
}

inline bool operator!=(const StrideAndCount &a, const StrideAndCount &b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const StrideAndCount &s);

using StrideDesc = boost::container::small_vector<StrideAndCount, 1>;

std::size_t getTotalCount(const StrideDesc &s);
std::size_t getContiguousRegionSize(const StrideDesc &s);
std::size_t getTotalContiguousRegions(const StrideDesc &s);
std::size_t getTotalOffset(const StrideDesc &s);
std::ostream &operator<<(std::ostream &os, const StrideDesc &s);

/** StridedRegion is a structure to describe a series of intervals
 *  in a more compressed way.
 *
 *   - The strides are always positive (a choice due to its intended
 *     use rather than a hard requirement).
 *   - The strides are not strides applied one after another when iterating
 *     the region but rather like multipliers for an (multi-dimensional)
 *     index. This is again a choice due to the intended usage.
 *   - A StridedRegion with an empty StrideDesc represents a single element.
 *     This is an intentional choice to make a StrideDesc behave similarly
 *     to a tensor shape (where empty = scalar), and that also simplifies
 *     merging logic by assuming there are no strides with count = 1.
 */
struct StridedRegion {
  std::size_t offset;
  StrideDesc stride;
  StridedRegion(std::size_t offset, StrideDesc stride) noexcept
      : offset(offset), stride(std::move(stride)) {}
  StridedRegion(const poplar::Interval &i) noexcept : offset(i.begin()) {
    const auto size = i.size();
    if (size > 1) {
      stride = {StrideAndCount(1, size)};
    }
  }
  std::size_t numContiguousRegions() const {
    return getTotalContiguousRegions(stride);
  }
  std::size_t numElements() const { return getTotalCount(stride); }
  std::size_t getFinalOffset() const { return offset + getTotalOffset(stride); }
};

inline bool operator==(const StridedRegion &a, const StridedRegion &b) {
  return std::tie(a.offset, a.stride) == std::tie(b.offset, b.stride);
}

inline bool operator!=(const StridedRegion &a, const StridedRegion &b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const StridedRegion &r);

using StridedRegionList = std::vector<StridedRegion>;

/// Compress a list of strided regions as much as possible.
bool mergeStridedRegions(StridedRegionList &regions);

/// Slice and concatenate together the given regions of a tensor.
poplar::Tensor sliceStridedRegions(const poplar::Tensor &t,
                                   const StridedRegionList &regions);

/// These convert strided region descriptions into intervals.
/// This is mostly for round-trip testing.
void appendIntervalsForStridedRegion(const StridedRegion &region,
                                     std::vector<poplar::Interval> &intervals);
std::vector<poplar::Interval>
intervalsForStridedRegion(const StridedRegion &region);

std::vector<poplar::Interval>
intervalsForStridedRegions(const StridedRegionList &regions);

} // end namespace poplibs_support

#endif // poplibs_support_StridedRegions_hpp
