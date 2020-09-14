// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Create tensor layouts that are optimised for host transfers.
 *
 */

#ifndef popops_HostSliceTensor_hpp_
#define popops_HostSliceTensor_hpp_

#include <poplar/Tensor.hpp>

namespace poplar {

class Graph;

}

namespace popops {

/// The pair of values returned by createHostSliceableTensor().
struct IndicesAndTensor {
  poplar::Tensor indices;
  poplar::Tensor tensor;
};

/// Create a Tensor that is well laid out for a host exchange copy
/// and at the same time create the index tensor for the copy.
/// The shape must be size 2, dim(1) must be the size of the datastream
/// or remote buffer, if using a copy from a remote buffer with multiple
/// slice indices dim(0) must be num slice indices, other wise dim(0) is 1.
///
/// \param graph  The Poplar graph to add the tensor to.
/// \param type   The element type of the tensor created.
/// \param shape  The hape of created tensor.
/// \param isRead If true, the tensor will be read by the host. If false,
///               the tensor data will be written to the host. If \p isRead is
///               true, tile imbalance is likely to be greater.
/// \return Two tensors: the indices, which will have size shape[0] and the
///         tensor that will be written to.
IndicesAndTensor createHostSliceableTensor(poplar::Graph &graph,
                                           const poplar::Type &type,
                                           const std::vector<size_t> &shape,
                                           const bool isRead,
                                           const std::string &debugPrefix = "");

/// Create a tensor that is well laid out for a host exchange copy.
/// \param graph  The graph to add the tensor to.
/// \param type   The element type of the tensor created.
/// \param shape  The shape of the tensor created.
/// \param isRead If true, the tensor will be read by the host. If false,
///               the tensor data will be written to the host. Setting \p isRead
///               to true is likely to make the read operation faster without
///               affecting the write, but is also likely to cause greater tile
///               imbalance.
/// \return       The tensor created.
poplar::Tensor
createHostTransferableTensor(poplar::Graph &graph, const poplar::Type &type,
                             const std::vector<size_t> &shape, bool isRead,
                             const std::string &debugPrefix = "");

} // namespace popops
#endif
