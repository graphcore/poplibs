// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "poplibs_support/TileHierarchy.hpp"

#include <algorithm>
#include <map>

namespace poplibs {

std::vector<unsigned> getTileHierarchy(unsigned numIPUs, unsigned tilesPerIPU) {
  std::vector<unsigned> hierarchy;
  if (numIPUs > 1) {
    hierarchy.push_back(numIPUs);
  }
  hierarchy.push_back(tilesPerIPU);
  return hierarchy;
}

unsigned numIPUs(const std::vector<unsigned> &hierarchy) {
  if (hierarchy.size() == 1) { // single IPU case
    return 1;
  } else {
    return hierarchy[0];
  }
}

std::vector<double>
getPerLevelExchangeBytesPerCycle(const poplar::Target &target,
                                 unsigned numIPUs) {
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto clockFrequency = target.getTileClockFrequency();
  if (numIPUs > 1) {
    auto ipuExchangeBytesPerCycle =
        static_cast<double>(std::numeric_limits<double>::infinity());
    // Compute the maximum number of bytes per cycle for a traffic pattern
    // where every IPU sends an equal amount of data to every other IPU.
    for (const auto &constraint : target.getGlobalExchangeConstraints()) {
      std::map<unsigned, unsigned> ipuSrcCount;
      std::map<unsigned, unsigned> ipuDstCount;
      for (const auto &flow : constraint.flows) {
        ++ipuSrcCount[flow.src];
        ++ipuDstCount[flow.dst];
      }
      auto secondLess = [](const std::pair<unsigned, unsigned> &a,
                           const std::pair<unsigned, unsigned> &b) {
        return a.second < b.second;
      };
      const auto maxSrcCount =
          std::max_element(ipuSrcCount.begin(), ipuSrcCount.end(), secondLess)
              ->second;
      const auto maxDstCount =
          std::max_element(ipuDstCount.begin(), ipuDstCount.end(), secondLess)
              ->second;
      const auto maxCount = std::max(maxSrcCount, maxDstCount);
      const auto constraintBytesPerCycle =
          (constraint.bandwidth / clockFrequency) / 8;
      ipuExchangeBytesPerCycle = std::min(ipuExchangeBytesPerCycle,
                                          constraintBytesPerCycle / maxCount);
    }
    perLevelExchangeBytesPerCycle.push_back(ipuExchangeBytesPerCycle);
  }
  perLevelExchangeBytesPerCycle.push_back(target.getExchangeBytesPerCycle());
  return perLevelExchangeBytesPerCycle;
}

} // namespace poplibs
