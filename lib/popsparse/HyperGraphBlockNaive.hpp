// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphBlockNaive_hpp
#define popsparse_HyperGraphBlockNaive_hpp

#include "HyperGraphBlock.hpp"

namespace popsparse {
namespace experimental {

/*
This class uses simple tiles mapping scheme
without any graph partitioning
*/
class HyperGraphBlockNaive : public HyperGraphBlock {
public:
  HyperGraphBlockNaive(BlockMatrix &A, BlockMatrix &B,
                       poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                       poplar::Type partialDataTypeIn, int nTileIn,
                       int nTargetNodesVPerTileIn = TARGET_V_NODES_PER_TILE);

  virtual ~HyperGraphBlockNaive() = default;

protected:
  virtual void partitionGraph() override;
};

} // namespace experimental
} // namespace popsparse

#endif