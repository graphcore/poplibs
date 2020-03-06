// Copyright (c) 2017 Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_NonLinearity_hpp
#define poplibs_test_NonLinearity_hpp

#include "poplibs_support/Compiler.hpp"
#include "poplibs_test/exceptions.hpp"
#include "popnn/NonLinearity.hpp"
#include <boost/multi_array.hpp>

namespace poplibs_test {

// input/output can be pointers to same memory
void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  const double *inputData, double *outputData, std::size_t n);

void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  boost::multi_array_ref<double, 2> array);

void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  boost::multi_array<double, 4> &array);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const double *activations, double *deltas, std::size_t n);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 4> &activations,
                     boost::multi_array<double, 4> &deltas);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 2> &activations,
                     boost::multi_array<double, 2> &deltas);

} // End namespace poplibs_test.

#endif // poplibs_test_NonLinearity_hpp
