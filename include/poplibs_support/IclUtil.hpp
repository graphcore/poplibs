// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_IclUtil_hpp
#define poplibs_support_IclUtil_hpp

#include <boost/icl/interval_map.hpp>
#include <boost/icl/split_interval_set.hpp>
#include <cstddef>
#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <vector>

// Utility functions for converting to/from boost::icl classes.

namespace poplibs {

/// Convert an ICL split_interval_set to a vector of poplar::Interval's.
std::vector<poplar::Interval> splitIntervalSetToPoplar(
    const boost::icl::split_interval_set<std::size_t> &intervals);

/// Convert a vector of poplar::Interval's to an ICL split_interval_set.
boost::icl::split_interval_set<std::size_t>
poplarToSplitIntervalSet(const std::vector<poplar::Interval> &intervals);

/// Convert a tile mapping from poplar's vector<vector<Interval>> format
/// to an ICL interval_map. partial_enricher ensures tile 0 is stored.
boost::icl::interval_map<std::size_t, unsigned, boost::icl::partial_enricher>
tileMappingToIntervalMap(const poplar::Graph::TileToTensorMapping &mapping);

} // namespace poplibs

#endif // poplibs_support_IclUtil_hpp
