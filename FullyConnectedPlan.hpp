#ifndef _FullyConnectedPlan_hpp_
#define _FullyConnectedPlan_hpp_

#include <string>

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
  Plan(const Partition &ipuPartition) : ipuPartition(ipuPartition) {}
};

Plan
createPlan(const DeviceInfo &deviceInfo,
           const std::string &dType, unsigned numRows,
           unsigned numCols);

} // End namespace fc.

#endif // _FullyConnectedPlan_hpp_
