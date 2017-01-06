#ifndef popnn_ref_MaxPooling_hpp_
#define popnn_ref_MaxPooling_hpp_

#include <boost/multi_array.hpp>

namespace ref {
namespace maxpool {

void maxPooling(unsigned strideHeight, unsigned strideWidth,
                unsigned kernelHeight, unsigned kernelWidth,
                unsigned paddingHeight, unsigned paddingWidth,
                const boost::multi_array<double, 4> &in,
                boost::multi_array<double, 4> &out);

void maxPoolingBackward(unsigned strideHeight, unsigned strideWidth,
                        unsigned kernelHeight, unsigned kernelWidth,
                        unsigned paddingHeight, unsigned paddingWidth,
                        const boost::multi_array<double, 4> &prevAct,
                        const boost::multi_array<double, 4> &nextAct,
                        const boost::multi_array<double, 4> &in,
                        boost::multi_array<double, 4> &out);


} // End namespace maxpool.
} // End namespace ref.

#endif  // popnn_ref_MaxPooling__
