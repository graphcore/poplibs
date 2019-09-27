// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_support_Algorithms_hpp
#define poplibs_support_Algorithms_hpp

#include <poplar/Interval.hpp>

#include <vector>
#include <cassert>
#include <cstddef>
#include <functional>

#include <boost/icl/interval.hpp>

namespace poplibs {

// Flatten a vector of vectors into one vector.
template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
  std::size_t total = 0;
  for (const auto &sub : v)
    total += sub.size();

  std::vector<T> flat;
  flat.reserve(total);

  for (const auto &sub : v)
    flat.insert(flat.end(), sub.begin(), sub.end());

  return flat;
}

// Get the bounds of an interval in a generic way, so that it works
// for boost::icl::interval, and also poplar::Interval.
template<typename T>
std::size_t ival_begin(const T& ival) {
  return ival.begin();
}
template<typename T>
std::size_t ival_end(const T& ival) {
  return ival.end();
}

template<>
std::size_t ival_begin<boost::icl::interval<std::size_t>::type>(
    const boost::icl::right_open_interval<std::size_t>& ival);
template<>
std::size_t ival_end<boost::icl::interval<std::size_t>::type>(
    const boost::icl::right_open_interval<std::size_t>& ival);

/// Iterate over two region lists. There must be no gaps in the lists and
/// they must cover the same span.
///
/// For example if you have the following region lists
///
///     [0, 10) -> A
///     [10, 15) -> B
///     [15, 30) -> C
///
/// and
///
///     [0, 5) -> Alpha
///     [5, 20) -> Beta
///     [20, 25) -> Gamma
///     [25, 30) -> Delta
///
/// f would be called with the following parameters
///
///      f(0, 5, A, Alpha)
///      f(5, 10, A, Beta)
///      f(10, 15, B, Beta)
///      f(15, 20, C, Beta)
///      f(20, 25, C, Gamma)
///      f(25, 30, C, Delta)
///
template<typename RegionFunction,
         typename InputIteratorA,
         typename InputIteratorB>
void for_each_zipped_region(
    InputIteratorA beginA, InputIteratorA endA,
    InputIteratorB beginB, InputIteratorB endB,
    RegionFunction f) {

  assert(ival_begin(beginA->first) == ival_begin(beginB->first));

  auto itA = beginA;
  auto itB = beginB;

  while (itA != endA && itB != endB) {

    // Catch up the other region if this region is past its end.
    if (ival_begin(itA->first) >= ival_end(itB->first))
      ++itB;
    if (ival_begin(itB->first) >= ival_end(itA->first))
      ++itA;

    // Get the intersection of the two regions.
    auto isecBegin = std::max(ival_begin(itA->first), ival_begin(itB->first));
    auto isecEnd = std::min(ival_end(itA->first), ival_end(itB->first));

    // They should intersect!
    assert(isecEnd > isecBegin);

    // Call the relevant function!
    f(isecBegin, isecEnd, itA->second, itB->second);

    // Advance the furthest-back region.
    if (ival_end(itA->first) < ival_end(itB->first))
      ++itA;
    else
      ++itB;
  }
}

// From the given interval set, construct a new interval set that forms
// the inverse mapping.
std::vector<poplar::Interval>
getInverseMapping(const std::vector<std::vector<poplar::Interval>> &mapping);

} // namespace poplibs

#endif // poplibs_support_Algorithms_hpp
