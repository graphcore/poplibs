// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_TileHierarchy_hpp
#define poplibs_support_TileHierarchy_hpp

#include <poplar/Target.hpp>

namespace poplibs {

// TODO: Consider merging following two functions

std::vector<unsigned> getTileHierarchy(const poplar::Target &target);

std::vector<double>
getPerLevelExchangeBytesPerCycle(const poplar::Target &target);

} // end namespace poplibs

#endif // poplibs_support_TileHierarchy_hpp
