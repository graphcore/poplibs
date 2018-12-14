// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_PoolPlan_hpp
#define popnn_PoolPlan_hpp

#include "poplin/Convolution.hpp"
#include <poplar/Graph.hpp>
#include <vector>
#include <ostream>

namespace popnn {
namespace pooling {

// Gives the pass the pooling operation is performed for
enum class PoolPass {
  // Forward pooling operation
  POOL_FWD,
  // Backward pooling operation
  POOL_BWD
};

// Partition represents an actual partition (or split) of the constituent
// variables. Each variable in the partition gives the number of tiles over
// which that variable is spread.
struct Partition {
  // For each spatial dimension gives the split over tiles
  std::vector<std::size_t> field;
  // For each spatial dimension of the kernel gives the split over tiles
  std::vector<std::size_t> kernel;
  // The number of tiles batch dimension is split over
  std::size_t batch;
  // The number of tiles over which the channel group dimension is split over
  std::size_t chanGroups;
  // The number of channels per group. Each group is always mapped on a tile
  std::size_t chansPerGroup;
  Partition() = default;


  // Transforms the partition(split) into the number of elements per tile
  Partition getPerTile(const poplin::ConvParams &params) const {
    Partition perTile;
    perTile.chansPerGroup = chansPerGroup;
    perTile.batch = (params.batchSize + batch - 1) / batch;
    auto numChanGroups =
        (params.getNumOutputChans() + chansPerGroup - 1) / chansPerGroup;
    perTile.chanGroups = (numChanGroups + chanGroups - 1) / chanGroups;
    const auto outputShape = params.getOutputFieldShape();
    const auto kernelShape = params.kernelShape;
    for (std::size_t i = 0; i != field.size(); ++i) {
      perTile.field.push_back((outputShape[i] + field[i] - 1) / field[i]);
      perTile.kernel.push_back((kernelShape[i] + kernel[i] - 1) / kernel[i]);
    }
    return perTile;
  }
};


std::ostream& operator<<(std::ostream &os, const Partition &p);


// Get plan based on compute and exchange cost. As a further improvement, the
// plan could incorporate introspection. For now, keep it simple.
// Fwd and Bwd plans are kept separate as there is possibly no benefit for
// doing a joint one.
Partition
getPlan(const poplar::Graph &graph, const poplin::ConvParams &params,
        const std::vector<std::size_t> &inShape, unsigned chansPerGroupDet,
        PoolPass pass);

} //namespace pooling
} // namespace popnn


#endif // #ifndef popnn_PoolPlan_hpp
