// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_support_TileHierarchy_hpp
#define poplibs_support_TileHierarchy_hpp

#include <poplar/Target.hpp>

namespace poplibs {

std::vector<unsigned>
getTileHierarchy(const poplar::Target &target,
                 unsigned numIPUs,
                 unsigned numTiles,
                 std::vector<double> &perLevelExchangeBytesPerCycle);

} // end namespace poplibs

#endif // poplibs_support_TileHierarchy_hpp
