// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#include "poplibs_support/IclUtil.hpp"

namespace poplibs {

std::vector<poplar::Interval> splitIntervalSetToPoplar(
    const boost::icl::split_interval_set<std::size_t> &intervals) {

  std::vector<poplar::Interval> ret;
  ret.reserve(intervals.iterative_size());

  for (const auto &ival : intervals)
    ret.emplace_back(ival.lower(), ival.upper());

  return ret;
}

boost::icl::split_interval_set<std::size_t>
poplarToSplitIntervalSet(const std::vector<poplar::Interval> &intervals) {

  boost::icl::split_interval_set<std::size_t> ret;

  for (const auto &ival : intervals) {
    ret.insert(
        boost::icl::interval<size_t>::right_open(ival.begin(), ival.end()));
  }

  return ret;
}

boost::icl::interval_map<std::size_t, unsigned, boost::icl::partial_enricher>
tileMappingToIntervalMap(const poplar::Graph::TileToTensorMapping &mapping) {
  // Convert the tile mapping into an interval_map from variable index to tile
  // id. partial_enricher is used otherwise tile 0 is not stored.
  boost::icl::interval_map<std::size_t, unsigned, boost::icl::partial_enricher>
      mappingIcl;

  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    for (const auto &re : mapping[tile]) {
      auto reIcl =
          boost::icl::interval<std::size_t>::right_open(re.begin(), re.end());
      mappingIcl.set(std::make_pair(reIcl, tile));
    }
  }

  return mappingIcl;
}

} // namespace poplibs
