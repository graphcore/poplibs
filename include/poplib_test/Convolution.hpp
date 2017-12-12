#ifndef _poplib_test_Convolution_hpp_
#define _poplib_test_Convolution_hpp_

#include<boost/multi_array.hpp>

namespace poplib_test {
namespace conv {

void convolution(const std::vector<unsigned> &inputFieldSize,
                 const std::vector<unsigned> &inputDilation,
                 const std::vector<int> &paddingLower,
                 const std::vector<int> &paddingUpper,
                 const std::vector<bool> &flipInput,
                 const std::vector<unsigned> &kernelSize,
                 const std::vector<unsigned> &kernelDilation,
                 const std::vector<int> &kernelPaddingLower,
                 const std::vector<int> &kernelPaddingUpper,
                 const std::vector<bool> &flipKernel,
                 const std::vector<unsigned> &stride,
                 boost::const_multi_array_ref<double, 3> in,
                 boost::const_multi_array_ref<double, 4> weights,
                 boost::const_multi_array_ref<double, 1> biases,
                 boost::multi_array_ref<double, 3> out);

void convolutionBackward(const std::vector<unsigned> &inputFieldSize,
                         const std::vector<unsigned> &inputDilation,
                         const std::vector<int> &paddingLower,
                         const std::vector<int> &paddingUpper,
                         const std::vector<bool> &flipInput,
                         const std::vector<unsigned> &kernelSize,
                         const std::vector<unsigned> &kernelDilation,
                         const std::vector<int> &kernelPaddingLower,
                         const std::vector<int> &kernelPaddingUpper,
                         const std::vector<bool> &flipKernel,
                         const std::vector<unsigned> &stride,
                         boost::const_multi_array_ref<double, 3> in,
                         boost::const_multi_array_ref<double, 4> weights,
                         boost::multi_array_ref<double, 3> out);

void weightUpdate(const std::vector<unsigned> &inputFieldSize,
                  const std::vector<unsigned> &inputDilation,
                  const std::vector<int> &paddingLower,
                  const std::vector<int> &paddingUpper,
                  const std::vector<bool> &flipInput,
                  const std::vector<unsigned> &kernelSize,
                  const std::vector<unsigned> &kernelDilation,
                  const std::vector<int> &kernelPaddingLower,
                  const std::vector<int> &kernelPaddingUpper,
                  const std::vector<bool> &flipKernel,
                  const std::vector<unsigned> &stride,
                  double learningRate,
                  boost::const_multi_array_ref<double, 3> activations,
                  boost::const_multi_array_ref<double, 3> deltas,
                  boost::multi_array_ref<double, 4> weights,
                  boost::multi_array_ref<double, 1> biases);

// Compute estimates of mean and standard deviation for a batch of activations
// and return
// 1) mean
// 2) 1/sqrt(stdDev * stdDev + eps)
void batchNormEstimates(const boost::multi_array_ref<double, 4> actsIn,
                        double eps,
                        boost::multi_array_ref<double, 1> mean,
                        boost::multi_array_ref<double, 1> invStdDev);

// Batch normalise activations given whiteinputned activations and parameters
// gamma and beta and estimates mean and inverse standard deviation
void batchNormalise(const boost::multi_array_ref<double, 4> acts,
                    const boost::multi_array_ref<double, 1> gamma,
                    const boost::multi_array_ref<double, 1> beta,
                    const boost::multi_array_ref<double, 1> mean,
                    const boost::multi_array_ref<double, 1> invStdDev,
                    boost::multi_array_ref<double, 4> actsOut,
                    boost::multi_array_ref<double, 4> actsWhitened);

// Compute gradients for batch normalisation given whitened activations,
// input gradients, inverse of standard deviation and gamma
void batchNormGradients(const boost::multi_array_ref<double, 4> actsWhitened,
                        const boost::multi_array_ref<double, 4> gradsIn,
                        const boost::multi_array_ref<double, 1> invStdDev,
                        const boost::multi_array_ref<double, 1> gamma,
                        boost::multi_array_ref<double, 4> gradsOut);

// Update parameters gamma and beta given whitened activations
void batchNormParamUpdate(const boost::multi_array_ref<double, 4> actsWhitened,
                          const boost::multi_array_ref<double, 4> gradsIn,
                          double learningRate,
                          boost::multi_array_ref<double, 1> gamma,
                          boost::multi_array_ref<double, 1> beta);

} // End namespace poplib_test.
} // End namespace conv.

#endif  // _poplib_test_Convolution_hpp_
