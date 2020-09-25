// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popsparse_FullyConnectedPNMapping_hpp
#define popsparse_FullyConnectedPNMapping_hpp

#include "FullyConnectedVector.hpp"

#include <ostream>
#include <vector>

namespace popsparse {
namespace fullyconnected {

/** The order in which to map partitions of different dimensions
 *  of a sparse fully connected layer operation to tiles.
 */
class PartitionToPNMapping {
  Vector<unsigned> linearisationOrder;

public:
  PartitionToPNMapping() = default;
  PartitionToPNMapping(const PartitionToPNMapping &other) = default;
  PartitionToPNMapping(PartitionToPNMapping &&other) = default;
  PartitionToPNMapping &operator=(const PartitionToPNMapping &other) = default;
  PartitionToPNMapping &operator=(PartitionToPNMapping &&other) = default;

  PartitionToPNMapping(const Vector<unsigned> &linearisationOrder);
  unsigned getPNIdForPartition(const Vector<unsigned> &partitions,
                               const Vector<unsigned> &index) const;
  const Vector<unsigned> &getLinearisationOrder() const {
    return linearisationOrder;
  }
  friend std::ostream &operator<<(std::ostream &os,
                                  const PartitionToPNMapping &m);
};

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedPNMapping_hpp
