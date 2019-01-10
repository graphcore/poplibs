// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Norms_hpp
#define poplibs_test_Norms_hpp

#include<boost/multi_array.hpp>

namespace poplibs_test {
namespace norm {

// Norms are supported for activations of shape [B][C][FIELD]
// where B is the number of batches
//       C is the number of channels
//       FIELD is the field size
// Fields of dimension > 1 are supported by flattening the field dimension

// Type of normalisations
enum class NormType {
  BatchNorm,
  GroupNorm,
  InstanceNorm,
  LayerNorm
};

// Compute estimates of mean and standard deviation for a batch of activations
// and return
// 1) mean
// 2) 1/sqrt(stdDev * stdDev + eps)
void normStatistics(const boost::multi_array_ref<double, 3> actsIn,
                    double eps, bool unbiasedVarEstimate,
                    boost::multi_array_ref<double, 1> mean,
                    boost::multi_array_ref<double, 1> invStdDev,
                    NormType normType);

// Normalise activations togiven whitening statistics and parameters
// gamma and beta and estimates mean and inverse standard deviation
void normalise(const boost::multi_array_ref<double, 3> actsIn,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> invStdDev,
               boost::multi_array_ref<double, 3> actsOut,
               boost::multi_array_ref<double, 3> actsWhitened,
               NormType normType);

// Compute gradients for normalisation given whitened activations,
// input gradients, inverse of standard deviation and gamma
void normGradients(const boost::multi_array_ref<double, 3> actsWhitened,
                   const boost::multi_array_ref<double, 3> gradsIn,
                   const boost::multi_array_ref<double, 1> invStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 3> gradsOut,
                   NormType normType);

// Update parameters gamma and beta given whitened activations
void normParamUpdate(const boost::multi_array_ref<double, 3> actsWhitened,
                     const boost::multi_array_ref<double, 3> gradsIn,
                     double learningRate,
                     boost::multi_array_ref<double, 1> gamma,
                     boost::multi_array_ref<double, 1> beta,
                     NormType normType);
} // End namespace norm.
} // End namespace poplibs_test

#endif // poplibs_test_Norms_hpp
