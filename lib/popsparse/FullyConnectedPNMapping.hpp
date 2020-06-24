// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popsparse_FullyConnectedPNMapping_hpp
#define popsparse_FullyConnectedPNMapping_hpp

#include "FullyConnectedVector.hpp"

#include <ostream>

namespace popsparse {
namespace fullyconnected {

/** The order in which to map partitions of different dimensions
 *  of a sparse fully connected layer operation to tiles.
 */
enum class PartitionToPNMappingOrder {
  /// Partitions are mapped to PNs linearly, where the PN no.
  /// is given by flattening the 4-dimensional index into a
  /// shape given by no. of partitions of Groups, X, Y, and Z,
  /// respectively: {Groups,Y,Z,X}.
  FwdLinearGYZX,
};

std::ostream &operator<<(std::ostream &os,
                         const PartitionToPNMappingOrder &order);

unsigned getPNIdForPartition(const PartitionToPNMappingOrder &order,
                             const Vector<unsigned> &partitions,
                             const Vector<unsigned> &index);

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedPNMapping_hpp
