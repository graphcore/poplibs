#include "poplibs_support/Algorithms.hpp"

namespace poplibs {

template <>
std::size_t ival_begin<boost::icl::interval<std::size_t>::type>(
    const boost::icl::interval<std::size_t>::type &ival) {
  return ival.lower();
}
template <>
std::size_t ival_end<boost::icl::interval<std::size_t>::type>(
    const boost::icl::interval<std::size_t>::type &ival) {
  return ival.upper();
}

// From the given interval set, construct a new interval set that forms
// the inverse mapping.
std::vector<poplar::Interval>
getInverseMapping(const std::vector<std::vector<poplar::Interval>> &mapping) {
  std::map<poplar::Interval, poplar::Interval> inverseMap;

  std::size_t offset = 0;
  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    for (const auto &i : mapping[tile]) {
      inverseMap.emplace(i, poplar::Interval(offset, offset + i.size()));
      offset += i.size();
    }
  }
  std::vector<poplar::Interval> result;
  result.reserve(inverseMap.size());
  for (const auto &entry : inverseMap) {
    result.push_back(entry.second);
  }
  return result;
}

} // namespace poplibs
