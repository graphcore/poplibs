#ifndef _poplib_test_Pooling_hpp_
#define _poplib_test_Pooling_hpp_
#include "popnn/PoolingDef.hpp"
#include <boost/multi_array.hpp>

namespace poplib_test {
namespace pooling {

void pooling(popnn::PoolingType pType, unsigned strideHeight,
             unsigned strideWidth, unsigned kernelHeight, unsigned kernelWidth,
             int paddingHeightL, int paddingWidthL,
             int paddingHeightU, int paddingWidthU,
             const boost::multi_array<double, 4> &in,
             boost::multi_array<double, 4> &out);

void poolingBackward(popnn::PoolingType pType, unsigned strideHeight,
                     unsigned strideWidth, unsigned kernelHeight,
                     unsigned kernelWidth, int paddingHeightL,
                     int paddingWidthL, int paddingHeightU, int paddingWidthH,
                     const boost::multi_array<double, 4> &prevAct,
                     const boost::multi_array<double, 4> &nextAct,
                     const boost::multi_array<double, 4> &in,
                     boost::multi_array<double, 4> &out);


} // End namespace pooling.
} // End namespace poplib_test.

#endif  // _poplib_test_Pooling__
