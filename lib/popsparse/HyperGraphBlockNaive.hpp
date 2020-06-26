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
  HyperGraphBlockNaive(const BlockMatrix &A, const BlockMatrix &B,
                       poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                       poplar::Type partialDataTypeIn, int nTileIn,
                       float memoryCycleRatioIn,
                       int nMulsOnVNodeIn = MUL_ON_NODE_V);

  virtual ~HyperGraphBlockNaive() = default;

protected:
  virtual void partitionGraph() override;
};

} // namespace experimental
} // namespace popsparse

#endif