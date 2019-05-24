// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Pooling_hpp
#define poplibs_test_Pooling_hpp
#include "popnn/PoolingDef.hpp"
#include "poplibs_support/MultiArray.hpp"

namespace poplibs_test {
namespace pooling {

void pooling(popnn::PoolingType pType, unsigned strideHeight,
             unsigned strideWidth, unsigned kernelHeight, unsigned kernelWidth,
             int paddingHeightL, int paddingWidthL,
             int paddingHeightU, int paddingWidthU,
             const poplibs_support::MultiArray<double> &in,
             poplibs_support::MultiArray<double> &out);

void poolingBackward(popnn::PoolingType pType, bool useScaledGradForMaxPool,
                     unsigned strideHeight,
                     unsigned strideWidth, unsigned kernelHeight,
                     unsigned kernelWidth, int paddingHeightL,
                     int paddingWidthL, int paddingHeightU, int paddingWidthH,
                     const poplibs_support::MultiArray<double> &prevAct,
                     const poplibs_support::MultiArray<double> &nextAct,
                     const poplibs_support::MultiArray<double> &in,
                     poplibs_support::MultiArray<double> &out);

} // namespace pooling
} // namespace poplibs_test

#endif // poplibs_test_Pooling_hpp
