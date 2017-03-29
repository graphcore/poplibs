#ifndef _poplib_test_FullyConnected_hpp_
#define _poplib_test_FullyConnected_hpp_

#include<boost/multi_array.hpp>

namespace poplib_test {
namespace fc {

void fullyConnected(const boost::multi_array<double, 2> &in,
                    const boost::multi_array<double, 2> &weights,
                    const boost::multi_array<double, 1> &biases,
                    boost::multi_array<double, 2> &out);

void fullyConnectedBackward(const boost::multi_array<double, 2> &in,
                            const boost::multi_array<double, 2> &weights,
                            boost::multi_array<double, 2> &out);

void fullyConnectedWeightUpdate(
                  double learningRate,
                  const boost::multi_array<double, 2> &activations,
                  const boost::multi_array<double, 2> &deltas,
                  boost::multi_array<double, 2> &weights,
                  boost::multi_array<double, 1> &biases);

} // End namespace poplib_test.
} // End namespace conv.

#endif  // _poplib_test_FullyConnected_hpp_
