#ifndef popnn_ref_Convolution_hpp_
#define popnn_ref_Convolution_hpp_

#include "popnn/NonLinearityDef.hpp"
#include<boost/multi_array.hpp>

namespace ref {
namespace conv {

void convolution(unsigned stride, unsigned padding,
                 NonLinearityType nonLinearityType,
                 const boost::multi_array<double, 3> &in,
                 const boost::multi_array<double, 4> &weights,
                 const boost::multi_array<double, 1> &biases,
                 boost::multi_array<double, 3> &out);

} // End namespace ref.
} // End namespace conv.

#endif  // popnn_ref_Convolution_hpp_
