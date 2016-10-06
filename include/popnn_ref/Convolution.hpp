#ifndef popnn_ref_Convolution_hpp_
#define popnn_ref_Convolution_hpp_

#include "popnn/NonLinearityDef.hpp"
#include<boost/multi_array.hpp>

namespace ref {
namespace conv {

void convolution(unsigned strideHeight, unsigned strideWidth,
                 unsigned paddingHeight, unsigned paddingWidth,
                 NonLinearityType nonLinearityType,
                 const boost::multi_array<double, 4> &in,
                 const boost::multi_array<double, 4> &weights,
                 const boost::multi_array<double, 1> &biases,
                 boost::multi_array<double, 4> &out);

void convolutionBackward(unsigned strideHeight, unsigned strideWidth,
                         unsigned paddingHeight, unsigned paddingWidth,
                         const boost::multi_array<double, 4> &in,
                         const boost::multi_array<double, 4> &weights,
                         boost::multi_array<double, 4> &out);

void weightUpdate(unsigned strideHeight, unsigned strideWidth,
                  unsigned paddingHeight, unsigned paddingWidth,
                  double learningRate,
                  const boost::multi_array<double, 4> &activations,
                  const boost::multi_array<double, 4> &deltas,
                  boost::multi_array<double, 4> &weights,
                  boost::multi_array<double, 1> &biases);

} // End namespace ref.
} // End namespace conv.

#endif  // popnn_ref_Convolution_hpp_
