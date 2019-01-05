// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Norms_hpp
#define poplibs_test_Norms_hpp

#include<boost/multi_array.hpp>

namespace poplibs_test {
namespace norm {

// Norms are supported for activations of shape [B][C] and [B][C][H][W]
// where B is the number of batches
//       C is the number of channels
//       H, W are field dimensions

// Type of normalisations
enum class NormType {
  BatchNorm,
  GroupNorm,
  InstanceNorm,
  LayerNorm
};

// Compute estimates of mean and standard deviation for a batch of
// activations and return
// 1) mean
// 2) 1.0 / sqrt(stdDev * stdDev + eps)
// inverse standard deviation is returned because both whitening and gradient
// calculation requires an inverse of the standard deviation.
void normStatistics(const boost::multi_array_ref<double, 2> actsIn,
                    double eps,
                    bool unbiasedVarEstimate,
                    boost::multi_array_ref<double, 1> mean,
                    boost::multi_array_ref<double, 1> invStdDev,
                    NormType normType);

// Normalise activations given input statistics (mean and invStdDev) and
// parameters gamma and beta and estimates mean and inverse of standard
// deviation
void normalise(const boost::multi_array_ref<double, 2> actsIn,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> invStdDev,
               boost::multi_array_ref<double, 2> actsOut,
               boost::multi_array_ref<double, 2> actsWhitened,
               NormType normType);

// Compute gradients for normalisation given whitened activations,
// input gradients, inverse of standard deviation and gamma
void normGradients(const boost::multi_array_ref<double, 2> actsWhitened,
                   const boost::multi_array_ref<double, 2> gradsIn,
                   const boost::multi_array_ref<double, 1> invStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 2> gradsOut,
                   NormType normType);

// Update parameters gamma and beta given whitened activations
void normParamUpdate(const boost::multi_array_ref<double, 2> actsWhitened,
                     const boost::multi_array_ref<double, 2> gradsIn,
                     double learningRate,
                     boost::multi_array_ref<double, 1> gamma,
                     boost::multi_array_ref<double, 1> beta,
                     NormType normType);

// Compute estimates of mean and standard deviation for a batch of activations
// and return
// 1) mean
// 2) 1/sqrt(stdDev * stdDev + eps)
void normStatistics(const boost::multi_array_ref<double, 4> actsIn,
                    double eps, bool unbiasedVarEstimate,
                    boost::multi_array_ref<double, 1> mean,
                    boost::multi_array_ref<double, 1> invStdDev,
                    NormType normType);

// Normalise activations togiven whitening statistics and parameters
// gamma and beta and estimates mean and inverse standard deviation
void normalise(const boost::multi_array_ref<double, 4> actsIn,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> invStdDev,
               boost::multi_array_ref<double, 4> actsOut,
               boost::multi_array_ref<double, 4> actsWhitened,
               NormType normType);

// Compute gradients for normalisation given whitened activations,
// input gradients, inverse of standard deviation and gamma
void normGradients(const boost::multi_array_ref<double, 4> actsWhitened,
                   const boost::multi_array_ref<double, 4> gradsIn,
                   const boost::multi_array_ref<double, 1> invStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 4> gradsOut,
                   NormType normType);

// Update parameters gamma and beta given whitened activations
void normParamUpdate(const boost::multi_array_ref<double, 4> actsWhitened,
                     const boost::multi_array_ref<double, 4> gradsIn,
                     double learningRate,
                     boost::multi_array_ref<double, 1> gamma,
                     boost::multi_array_ref<double, 1> beta,
                     NormType normType);
} // End namespace norm.
} // End namespace poplibs_test

#endif // poplibs_test_Norms_hpp
