// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_HostSliceTensor_hpp_
#define popops_HostSliceTensor_hpp_

#include <poplar/Tensor.hpp>

namespace poplar {

class Graph;

}

namespace popops {

// Create a Tensor that is well layed out for a host exchange copy
// and at the same time create the index tensor for the copy.
// The shape must be size 2, dim(1) must be the size of the datastream
// or remote buffer, if using a copy from a remote buffer with multiple
// slice indices dim(0) must be num slice indices, other wise dim(0) is 1
// param graph  - the poplar graph to add the tensor to
// param type   - element type of the tensor created
// param shape  - shape of created tensor
// param isRead - whether the tensor is goig to be read from the host
//                or have it's data written to the host (with isRead true
//                tile imbalance is likely to be greater)
// return - 2 tensor, the indices which will have size sjape[0] and the
//         tensor that will be written to
struct IndicesAndTensor {
  poplar::Tensor indices;
  poplar::Tensor tensor;
};

IndicesAndTensor createHostSliceableTensor(poplar::Graph &graph,
                                           const poplar::Type &type,
                                           const std::vector<size_t> &shape,
                                           const bool isRead,
                                           const std::string &debugPrefix = "");

} // namespace popops
#endif
