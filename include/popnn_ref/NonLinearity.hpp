#ifndef popnn_ref_NonLinearity_hpp_
#define popnn_ref_NonLinearity_hpp_

#include "popnn/NonLinearityDef.hpp"
#include <boost/multi_array.hpp>

namespace ref {

void nonLinearity(NonLinearityType nonLinearityType,
                  boost::multi_array<double, 4> &array);

void bwdNonLinearity(NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 4> &activations,
                     boost::multi_array<double, 4> &deltas);

} // End namespace ref.

#endif  // popnn_ref_NonLinearity_hpp_
