// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Pooling_hpp
#define poplibs_test_Pooling_hpp

#include "poplibs_support/MultiArray.hpp"
#include "popnn/PoolingDef.hpp"

#include <vector>

namespace poplibs_test {
namespace pooling {

void pooling(popnn::PoolingType pType, const std::vector<unsigned> &stride,
             const std::vector<std::size_t> &kernel,
             const std::vector<int> &paddingLower,
             const std::vector<int> &paddingUpper,
             const poplibs_support::MultiArray<double> &in,
             poplibs_support::MultiArray<double> &out);

void poolingBackward(popnn::PoolingType pType, bool useScaledGradForMaxPool,
                     const std::vector<unsigned> &stride,
                     const std::vector<std::size_t> &kernel,
                     const std::vector<int> &paddingLower,
                     const std::vector<int> &paddingUpper,
                     const poplibs_support::MultiArray<double> &prevAct,
                     const poplibs_support::MultiArray<double> &nextAct,
                     const poplibs_support::MultiArray<double> &in,
                     poplibs_support::MultiArray<double> &out);

} // namespace pooling
} // namespace poplibs_test

#endif // poplibs_test_Pooling_hpp
