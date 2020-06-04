// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_TileHierarchy_hpp
#define poplibs_support_TileHierarchy_hpp

#include <poplar/Target.hpp>

namespace poplibs {

// TODO: Consider merging following two functions
// TODO: Consider allowing these functions to take a target instead of
// separate configured numIPUs and tilesPerIPU

std::vector<unsigned> getTileHierarchy(unsigned numIPUs, unsigned tilesPerIPU);

unsigned numIPUs(const std::vector<unsigned> &hierarchy);

std::vector<double>
getPerLevelExchangeBytesPerCycle(const poplar::Target &target,
                                 unsigned numIPUs);

} // end namespace poplibs

#endif // poplibs_support_TileHierarchy_hpp
