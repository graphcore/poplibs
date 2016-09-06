#ifndef popnn_ref_NonLinearity_hpp_
#define popnn_ref_NonLinearity_hpp_

#include "popnn/NonLinearityDef.hpp"
#include <boost/multi_array.hpp>

namespace ref {

void fwdNonLinearity(NonLinearityType nonLinearityType,
                     boost::multi_array<double, 3> &array);

} // End namespace ref.

#endif  // popnn_ref_NonLinearity_hpp_
