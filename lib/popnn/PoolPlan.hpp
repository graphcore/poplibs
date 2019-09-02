// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_PoolPlan_hpp
#define popnn_PoolPlan_hpp

#include "popnn/Pooling.hpp"
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

struct PoolConfig {
  PoolingType type;
  PoolPass pass;
  // only valid for MAX pool
  bool scaledGradient;

  PoolConfig(PoolingType type, PoolPass pass, bool scaledGradient) :
      type(type), pass(pass), scaledGradient(scaledGradient) {}
};


struct Transform {
  // Flatten independent spatial dimensions into channels.
  // Batch size is dimension 0.
  //
  // Ordering of listed dimensions indicates order in which
  // they will be flattened. Second item of the pair indicates the
  // number of elements of this dimension that will be flattened
  std::vector<std::pair<std::size_t, std::size_t>> flattenDims;
};

std::ostream &operator<<(std::ostream &o, const Transform &t);

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

struct Plan {
  Transform transform;
  Partition partition;
};

std::ostream& operator<<(std::ostream &os, const Plan &p);

/** Apply pooling plan transform to ConvParams and any number of
 *  activation shaped tensors given as a list of pointers to tensors.
 *
 * \param params    Convolutional parameters for the pooling operation
 *                  to which the transform will be applied.
 * \param transform Transform applied to produce `a`
 * \param as        List of pointers to activation shaped tensors to which
 *                  the transform will be applied in-place.
 *
 * \returns ConvParams with given transform applied.
 */
poplin::ConvParams
applyTransform(poplin::ConvParams params,
               const Transform &transform,
               const std::vector<poplar::Tensor *> &as = {});

/** Apply the inverse transform to an activation shaped tensor given
 *  the original parameters and the transform applied to them.
 *
 * \param params    Convolutional parameters for the pooling operation
 *                  prior to any transforms.
 * \param transform Transform applied to produce each tensor in `as`.
 * \param as        List of pointers to activation shaped tensors post-
 *                  transform to which the inverse transform will be
 *                  applied to produce the result.
 */
void
applyTransformInverse(const poplin::ConvParams &params,
                      const Transform &transform,
                      const std::vector<poplar::Tensor *> &as);

// Get plan based on compute and exchange cost. As a further improvement, the
// plan could incorporate introspection. For now, keep it simple.
// Fwd and Bwd plans are kept separate as there is possibly no benefit for
// doing a joint one.
Plan
getPlan(const poplar::Graph &graph,
        const PoolConfig &poolCfg,
        const poplin::ConvParams &params,
        const poplar::Tensor &in);

} //namespace pooling
} // namespace popnn


#endif // #ifndef popnn_PoolPlan_hpp
