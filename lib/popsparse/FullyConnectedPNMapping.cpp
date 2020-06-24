// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "FullyConnectedPNMapping.hpp"

#include "poputil/exceptions.hpp"

namespace popsparse {
namespace fullyconnected {

std::ostream &operator<<(std::ostream &os,
                         const PartitionToPNMappingOrder &order) {
  switch (order) {
  case PartitionToPNMappingOrder::FwdLinearGYZX:
    os << "FwdLinearGYZX";
    break;
  default:
    throw poputil::poplibs_error("Unrecognised mapping order");
  }
  return os;
}

unsigned getPNIdForPartition(const PartitionToPNMappingOrder &order,
                             const Vector<unsigned> &partitions,
                             const Vector<unsigned> &index) {
  switch (order) {
  case PartitionToPNMappingOrder::FwdLinearGYZX: {
    unsigned id = index.groups;
    id = id * partitions.y + index.y;
    id = id * partitions.z + index.z;
    id = id * partitions.x + index.x;
    return id;
  }
  default:
    throw poputil::poplibs_error("Unrecognised mapping order");
  }
}

} // end namespace fullyconnected
} // end namespace popsparse
