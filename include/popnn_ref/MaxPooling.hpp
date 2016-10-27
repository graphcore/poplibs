#ifndef popnn_ref_MaxPooling_hpp_
#define popnn_ref_MaxPooling_hpp_

#include <boost/multi_array.hpp>

namespace ref {
namespace maxpool {

void maxPooling(unsigned stride, unsigned kernelSize, unsigned padding,
                const boost::multi_array<double, 4> &in,
                boost::multi_array<double, 4> &out);

void maxPoolingBackward(unsigned stride, unsigned kernelSize, unsigned padding,
                        const boost::multi_array<double, 4> &prevAct,
                        const boost::multi_array<double, 4> &nextAct,
                        const boost::multi_array<double, 4> &in,
                        boost::multi_array<double, 4> &out);


} // End namespace maxpool.
} // End namespace ref.

#endif  // popnn_ref_MaxPooling__
