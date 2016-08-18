#ifndef _FullyConnectedPlan_hpp_
#define _FullyConnectedPlan_hpp_

#include <string>
#include <vector>

class DeviceInfo;

namespace fc {

struct Partition {
  unsigned tilesPerColumn;
  unsigned tilesPerRow;
  Partition(unsigned tilesPerColumn, unsigned tilesPerRow) :
    tilesPerColumn(tilesPerColumn), tilesPerRow(tilesPerRow) {}
};

struct Plan {
  Partition ipuPartition;
  std::vector<unsigned> outputMapping;
  Plan(const Partition &ipuPartition, std::vector<unsigned> outputMapping) :
    ipuPartition(ipuPartition), outputMapping(std::move(outputMapping)) {}
};

Plan
createPlan(const DeviceInfo &deviceInfo,
           const std::string &dType, unsigned numIn,
           std::vector<unsigned> outputMapping,
           bool forwardOnly);

} // End namespace fc.

#endif // _FullyConnectedPlan_hpp_
