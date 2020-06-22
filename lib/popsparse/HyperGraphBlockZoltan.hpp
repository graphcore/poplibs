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
  HyperGraphBlockZoltan(const BlockMatrix &A, const BlockMatrix &B,
                        poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                        poplar::Type partialDataTypeIn, int nTileIn,
                        float memoryCycleRatioIn,
                        int nMulNodesSplitFactorIn = MUL_ON_NODE_V);

  virtual ~HyperGraphBlockZoltan() = default;

protected:
  virtual void partitionGraph() override;

private:
  // Represents hypergraph in a Zoltan format
  HyperGraphData getDataForPartitioner();
};

} // namespace experimental
} // namespace popsparse

#endif