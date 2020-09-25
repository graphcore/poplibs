// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "FullyConnectedPNMapping.hpp"

#include "poputil/exceptions.hpp"

#include "poplibs_support/VectorUtils.hpp"

namespace popsparse {
namespace fullyconnected {

PartitionToPNMapping::PartitionToPNMapping(
    const Vector<unsigned> &linearisationOrder)
    : linearisationOrder(linearisationOrder) {}

unsigned PartitionToPNMapping::getPNIdForPartition(
    const Vector<unsigned> &partitions_, const Vector<unsigned> &index_) const {
  unsigned id = 0;
  const auto inverseOrder =
      inversePermutation(linearisationOrder.asStdVector());
  const auto &partitions = partitions_.asStdVector();
  const auto &index = index_.asStdVector();
  for (const auto dim : inverseOrder) {
    id = id * partitions[dim] + index[dim];
  }
  return id;
}

std::ostream &operator<<(std::ostream &os, const PartitionToPNMapping &m) {
  os << m.linearisationOrder;
  return os;
}

} // end namespace fullyconnected
} // end namespace popsparse
