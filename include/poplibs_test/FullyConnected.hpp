#ifndef _poplibs_test_FullyConnected_hpp_
#define _poplibs_test_FullyConnected_hpp_

#include<boost/multi_array.hpp>

namespace poplibs_test {
namespace fc {

void fullyConnected(const boost::multi_array<double, 3> &in,
                    const boost::multi_array<double, 3> &weights,
                    const boost::multi_array<double, 2> &biases,
                    boost::multi_array<double, 3> &out);

void fullyConnectedBackward(const boost::multi_array<double, 3> &in,
                            const boost::multi_array<double, 3> &weights,
                            boost::multi_array<double, 3> &out);

void fullyConnectedWeightUpdate(
                  double learningRate,
                  const boost::multi_array<double, 3> &activations,
                  const boost::multi_array<double, 3> &deltas,
                  boost::multi_array<double, 3> &weights,
                  boost::multi_array<double, 2> &biases);

// Compute estimates of mean and standard deviation for a batch of activations
// and return
// 1) mean
// 2) 1.0 / sqrt(stdDev * stdDev + eps)
void batchNormEstimates(
                  const boost::multi_array_ref<double, 2> actsIn,
                  double eps,
                  boost::multi_array_ref<double, 1> mean,
                  boost::multi_array_ref<double, 1> invStdDev);

// Batch normalise activations given input activations and parameters
// gamma and beta and estimates mean and inverse of standard deviation
void batchNormalise(
                  const boost::multi_array_ref<double, 2> acts,
                  const boost::multi_array_ref<double, 1> gamma,
                  const boost::multi_array_ref<double, 1> beta,
                  const boost::multi_array_ref<double, 1> mean,
                  const boost::multi_array_ref<double, 1> invStdDev,
                  boost::multi_array_ref<double, 2> actsOut,
                  boost::multi_array_ref<double, 2> actsWhitened
                  );

// Compute gradients for batch normalisation given whitened activations,
// input gradients, inverse of standard deviation and gamma
void batchNormGradients(
                  const boost::multi_array_ref<double, 2> actsWhitened,
                  const boost::multi_array_ref<double, 2> gradsIn,
                  const boost::multi_array_ref<double, 1> invStdDev,
                  const boost::multi_array_ref<double, 1> gamma,
                  boost::multi_array_ref<double, 2> gradsOut);


// Update parameters gamma and beta given whitened activations
void batchNormParamUpdate(
                  const boost::multi_array_ref<double, 2> actsWhitened,
                  const boost::multi_array_ref<double, 2> gradsIn,
                  double learningRate,
                  boost::multi_array_ref<double, 1> gamma,
                  boost::multi_array_ref<double, 1> beta);

} // End namespace poplibs_test.
} // End namespace conv.

#endif  // _poplibs_test_FullyConnected_hpp_
