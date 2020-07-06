// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef popnn_PoolVertices_hpp
#define popnn_PoolVertices_hpp

#include "PoolOptions.hpp"
#include "PoolPlan.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "poplin/ConvParams.hpp"
#include "popnn/PoolingDef.hpp"
#include <vector>

namespace popnn {
namespace pooling {

struct PoolIndices {
  std::size_t batch;
  std::vector<std::size_t> out;
  std::size_t chan;
  std::vector<std::size_t> kernel;
};

struct PoolSlice {
  std::size_t batchBegin, batchEnd;
  std::vector<std::size_t> fieldBegin, fieldEnd;
  std::size_t chanBegin, chanEnd;
  std::vector<std::size_t> kernelBegin, kernelEnd;

  std::size_t getNumFieldDims() const { return fieldBegin.size(); }
  std::size_t getBatchSize() const { return batchEnd - batchBegin; }
  std::size_t getNumChans() const { return chanEnd - chanBegin; }
  std::size_t getFieldSize(std::size_t dim) const {
    return fieldEnd[dim] - fieldBegin[dim];
  }
  std::size_t getKernelSize(std::size_t dim) const {
    return kernelEnd[dim] - kernelBegin[dim];
  }
};

// Partition work given a plan and generated vertices to do the pooling
// operation
// graph            Graph in which the vertices are created
// poolCfg          Gives pooling type and pass information
// in               Input tensor of shape [CG][B][...][CPG]
// out              Input tensor of shape [CG][B][...][CPG]
// fwdInputActs     input activations in the forward pass used in max pool
//                  backward pass. Must be a nullptr otherwise
// fwdOutputActs    output activations of pooling operation in the forward pass
//                  used in max pool backward pass. Must be a nullptr unless
//                  poolingType is POOL_BWD.
// params           Parameters for the pooling operation
// prog             Sequence program created for the pooling operation
// plan             Plan for this pooling operation.
// tile             Tile on which vertices are generated
// indices          indices of planning parameter splits assigned to this tile
// slice            parameters for slicing channels, batch, field and kernel
// debugPrefix      Debug prefix for operations and tensors for this operation
// poolOptions      Pooling options
void tilePartitions(poplar::Graph &graph, const PoolConfig &poolConfig,
                    const poplar::Tensor &in, const poplar::Tensor &out,
                    const poplar::Tensor *fwdInputActs,
                    const poplar::Tensor *fwdOutputActs,
                    const poplin::ConvParams &params,
                    poplar::program::Sequence &prog, const Plan &plan,
                    const std::string &debugPrefix,
                    const PoolOptions &poolOptions);

} // namespace pooling
} // namespace popnn

#endif // #define popnn_PoolVertices_hpp
