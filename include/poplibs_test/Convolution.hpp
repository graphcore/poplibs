// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Convolution_hpp
#define poplibs_test_Convolution_hpp

#include<boost/multi_array.hpp>

namespace poplibs_test {
namespace conv {

void convolution(const std::vector<unsigned> &inputFieldSize,
                 const std::vector<unsigned> &truncationLower,
                 const std::vector<unsigned> &truncationUpper,
                 const std::vector<unsigned> &inputDilation,
                 const std::vector<unsigned> &paddingLower,
                 const std::vector<unsigned> &paddingUpper,
                 const std::vector<bool> &flipInput,
                 const std::vector<unsigned> &kernelSize,
                 const std::vector<unsigned> &kernelTruncationLower,
                 const std::vector<unsigned> &kernelTruncationUpper,
                 const std::vector<unsigned> &kernelDilation,
                 const std::vector<unsigned> &kernelPaddingLower,
                 const std::vector<unsigned> &kernelPaddingUpper,
                 const std::vector<bool> &flipKernel,
                 const std::vector<unsigned> &outputTruncationLower,
                 const std::vector<unsigned> &outputTruncationUpper,
                 const std::vector<unsigned> &stride,
                 const std::vector<unsigned> &outputPaddingLower,
                 const std::vector<unsigned> &outputPaddingUpper,
                 boost::const_multi_array_ref<double, 3> in,
                 boost::const_multi_array_ref<double, 4> weights,
                 boost::const_multi_array_ref<double, 1> biases,
                 boost::multi_array_ref<double, 3> out);

void convolutionBackward(const std::vector<unsigned> &inputFieldSize,
                         const std::vector<unsigned> &truncationLower,
                         const std::vector<unsigned> &truncationUpper,
                         const std::vector<unsigned> &inputDilation,
                         const std::vector<unsigned> &paddingLower,
                         const std::vector<unsigned> &paddingUpper,
                         const std::vector<bool> &flipInput,
                         const std::vector<unsigned> &kernelSize,
                         const std::vector<unsigned> &kernelTruncationLower,
                         const std::vector<unsigned> &kernelTruncationUpper,
                         const std::vector<unsigned> &kernelDilation,
                         const std::vector<unsigned> &kernelPaddingLower,
                         const std::vector<unsigned> &kernelPaddingUpper,
                         const std::vector<bool> &flipKernel,
                         const std::vector<unsigned> &outputTruncationLower,
                         const std::vector<unsigned> &outputTruncationUpper,
                         const std::vector<unsigned> &stride,
                         const std::vector<unsigned> &outputPaddingLower,
                         const std::vector<unsigned> &outputPaddingUpper,
                         boost::const_multi_array_ref<double, 3> in,
                         boost::const_multi_array_ref<double, 4> weights,
                         boost::multi_array_ref<double, 3> out);

void weightUpdate(const std::vector<unsigned> &inputFieldSize,
                  const std::vector<unsigned> &truncationLower,
                  const std::vector<unsigned> &truncationUpper,
                  const std::vector<unsigned> &inputDilation,
                  const std::vector<unsigned> &paddingLower,
                  const std::vector<unsigned> &paddingUpper,
                  const std::vector<bool> &flipInput,
                  const std::vector<unsigned> &kernelSize,
                  const std::vector<unsigned> &kernelTruncationLower,
                  const std::vector<unsigned> &kernelTruncationUpper,
                  const std::vector<unsigned> &kernelDilation,
                  const std::vector<unsigned> &kernelPaddingLower,
                  const std::vector<unsigned> &kernelPaddingUpper,
                  const std::vector<bool> &flipKernel,
                  const std::vector<unsigned> &outputTruncationLower,
                  const std::vector<unsigned> &outputTruncationUpper,
                  const std::vector<unsigned> &stride,
                  const std::vector<unsigned> &outputPaddingLower,
                  const std::vector<unsigned> &outputPaddingUpper,
                  double learningRate,
                  boost::const_multi_array_ref<double, 3> activations,
                  boost::const_multi_array_ref<double, 3> deltas,
                  boost::multi_array_ref<double, 4> weights,
                  boost::multi_array_ref<double, 1> biases);

} // End namespace poplibs_test.
} // End namespace conv.

#endif // poplibs_test_Convolution_hpp
