#ifndef _Util_hpp_
#define _Util_hpp_

#include <poplar/Device.hpp>
#include <vector>
#include <utility>

void mergeAdjacentRegions(
    std::vector<std::pair<unsigned, unsigned>> &regions);

void mergeAdjacentRegions(
    std::vector<std::vector<std::pair<unsigned, unsigned>>> &mapping);

// Given a set of contiguous regions per tile, partition these regions
// between vertices on that tile, respecting the specified grain size.
// Regions may be split to balance the work across vertices.
void splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<std::pair<unsigned, unsigned>> &regions,
    std::vector<std::vector<std::pair<unsigned, unsigned>>> &vertexRegions,
    unsigned grainSize);

#endif // _Util_hpp_
