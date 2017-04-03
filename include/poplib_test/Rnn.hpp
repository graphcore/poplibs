#ifndef _poplib_test_rnn_hpp_
#define _poplib_test_rnn_hpp_

#include <boost/multi_array.hpp>
#include <popnn/NonLinearityDef.hpp>

namespace poplib_test {
namespace rnn {

/**
 * Computes the forward non-recursive part of the RNN
 */
void forwardWeightInput(
            const boost::multi_array_ref<double, 3> x,
            const boost::multi_array_ref<double, 2> weights,
            const boost::multi_array_ref<double, 3> y);

/**
 * Computes the recursive part of a RNN. The sequence length is derived from the
 * input x. The initial value of the output isis in yInit.
 *
 * sequence_length = x.shape()[0]
 *
 * for (s = 0; s != sequence_length; ++s) {
 *   yPrev = s == 0 ? yInit : y(s - 1, :);
 *   y(s, :) = NonLinearity(weights * yPrev + x(s, :) + bias)
 * }
 *
 */
void forwardIterate(
            const boost::multi_array_ref<double, 3> x,
            const boost::multi_array_ref<double, 2> yInit,
            const boost::multi_array_ref<double, 2> weights,
            const boost::multi_array_ref<double, 1> bias,
            boost::multi_array_ref<double, 3> y,
            popnn::NonLinearityType nonLinearityType);

} // End namespace rnn.
} // End namespace poplib_test.

#endif  // _poplib_test_rnn_hpp_
