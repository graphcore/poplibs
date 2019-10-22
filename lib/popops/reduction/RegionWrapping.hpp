#ifndef RegionWrapping_hpp
#define RegionWrapping_hpp

#include <cstddef>
#include <numeric>
#include <vector>

#include <boost/icl/interval_map.hpp>
#include <boost/icl/split_interval_set.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Tensor.hpp>

#include "popops/Reduce.hpp"

namespace popops {

/// Wrap the given intervals by wrapSize, and then calculate the set of
/// wrapped indices that they cover, preserving splits.
///
/// For example if we had these input intervals, with a wrapSize of 10
///
/// [0,        3) [3, 6) [6,
///           13)        [16, 20)
///      [22, 23)
///
/// The result would be
///
/// [0, 2) [2, 3) [3, 6) [6,  10)
///
boost::icl::split_interval_set<std::size_t>
getSplitWrappedRegions(const std::vector<poplar::Interval> &ivals,
                       std::size_t wrapSize);

/// Similar to getSplitWrappedRegions but the splits aren't preserved. For
/// example if we had these input intervals, with a wrapSize of 10
///
/// [0,      3)    [5,   7) [7,
///         13)    [15, 17)
///    [22, 23)
///
/// The result would be
///
/// [0,      3)    [5,         10)
///
boost::icl::interval_set<std::size_t>
getWrappedRegions(const std::vector<poplar::Interval> &ivals,
                  std::size_t wrapSize);

/// Extra convenience versions for vectors of vectors, which happens a fair bit.
boost::icl::split_interval_set<std::size_t>
getSplitWrappedRegions(const std::vector<std::vector<poplar::Interval>> &ivals,
                       std::size_t wrapSize);
boost::icl::interval_set<std::size_t>
getWrappedRegions(const std::vector<std::vector<poplar::Interval>> &ivals,
                  std::size_t wrapSize);

/// Split the given regions so that there are enough to distribute to
/// the workers.
///
/// This never merges regions but it might split them up so that there are
/// at least numPartitions partitions. The minimum split region size is
/// calculated based on the target data path width for the given operation.
///
/// \param target  The target to split for. This is used to work out the
///                minimum split size.
/// \param numPartitions  The number of partitions to aim for.
/// \param operation      The reduction operation. This is used to work out the
///                       minimum split size.
/// \param partialType    The type used for the reduction. This is used to
///                       work out the minimum split size.
/// \param regions        The regions to split.
///
/// \returns The split regions. Regions are never merged.
std::vector<poplar::Interval> splitOutputRegionsForWorkers(
    const poplar::Target &target, unsigned numPartitions,
    popops::Operation operation, poplar::Type partialType,
    const std::vector<poplar::Interval> &regions);

/// Take a set of regions and wrap them to [0, wrapSize). For each of the
/// wrapped spans, call f(span_begin, span_end). If a single region wraps the
/// full [0, wrapSize) span multiple times, f(0, wrapSize) is only called once.
///
/// begin and end must be iterators with a value_type of poplar::Interval.
template <typename RegionFunction, typename InputIterator>
void wrapRegions(InputIterator begin, InputIterator end, std::size_t wrapSize,
                 RegionFunction f) {
  // Loop through the intervals.
  std::for_each(begin, end, [&](const poplar::Interval &ival) {
    if (ival.size() == 0)
      return;

    // Index in the row of the first element.
    auto x1 = ival.begin() % wrapSize;
    // Index in the row of the first element without wrapping.
    auto x2 = x1 + ival.size();

    // Add the first wrapped region, from the first element
    // to the last or the end of the line - whichever is first.
    f(x1, std::min(x2, wrapSize));

    // If it wraps an entire line, add that (as long as we haven't already).
    if (x1 != 0 && x2 >= wrapSize * 2)
      f(0, wrapSize);

    // Add the tail, if there is one.
    if (x2 > wrapSize) {
      x2 %= wrapSize;
      if (x2 != 0)
        f(0, x2);
    }
  });
}

/// Take a set of regions and wrap them to [0, wrapSize). For each of the
/// wrapped spans, call f(span_begin, span_end, row).
///
/// begin and end must be iterators with a value_type of poplar::Interval.
template <typename RegionFunction, typename InputIterator>
void wrapRegionsToRows(InputIterator begin, InputIterator end,
                       std::size_t wrapSize, RegionFunction f) {
  // Loop through the intervals.
  std::for_each(begin, end, [&](const poplar::Interval &ival) {
    if (ival.size() == 0)
      return;

    for (auto i = ival.begin(); i < ival.end();) {

      auto x = i % wrapSize;

      // Length of this section.
      auto len = ival.end() - i;

      // If it goes past the end of the row truncate it.
      if (x + len > wrapSize) {
        len = wrapSize - x;
      }

      f(x, x + len, i / wrapSize);

      i += len;
    }
  });
}

} // namespace popops

#endif // RegionWrapping_hpp
