// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphBlockZoltan_hpp
#define popsparse_HyperGraphBlockZoltan_hpp

#include "HyperGraphBlock.hpp"
#include "ZoltanPartitioner.hpp"

namespace popsparse {
namespace experimental {

/*
This class uses Zoltan library for partitioning.
*/
class HyperGraphBlockZoltan : public HyperGraphBlock {

public:
  HyperGraphBlockZoltan(BlockMatrix &A, BlockMatrix &B,
                        poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                        poplar::Type partialDataTypeIn, int nTileIn,
                        float memoryCycleRatioIn,
                        int nTargetNodesVPerTileIn = TARGET_V_NODES_PER_TILE);

  virtual ~HyperGraphBlockZoltan() = default;

protected:
  // Set up weights for a graph
  virtual void setupWeights(const poplar::Graph &graph) override;

  virtual void partitionGraph() override;

private:
  // Used to tune partitioning algorithm
  float memoryCycleRatio;

  // Represents hypergraph in a Zoltan format
  HyperGraphData getDataForPartitioner();
};

} // namespace experimental
} // namespace popsparse

#endif