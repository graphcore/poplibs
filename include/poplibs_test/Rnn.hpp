// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Rnn_hpp
#define poplibs_test_Rnn_hpp

/**
 * Functions to compute forward, backward and weight update phases of a Vanilla
 * RNN
 */

#include <boost/multi_array.hpp>
#include <popnn/NonLinearityDef.hpp>

namespace poplibs_test {
namespace rnn {

/**
 * Computes the forward non-recursive part of the RNN
 *
 * Dimensions:
 *  x:
 *  [sequence][batch][input channel]
 * weights:
 *  [input channel][output channel]
 * y:
 *  [sequence][batch][output channel]
 *
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
 * Dimensions:
 *  x:
 *  [sequence][batch][output channel]
 *  yInit:
 *  [batch][output channel]
 *  weights:
 *  [input channel][output channel]
 *  bias:
 *  [output channel]
 *  y:[sequence][batch][output channel]
 */
void forwardIterate(
            const boost::multi_array_ref<double, 3> x,
            const boost::multi_array_ref<double, 2> yInit,
            const boost::multi_array_ref<double, 2> weights,
            const boost::multi_array_ref<double, 1> bias,
            boost::multi_array_ref<double, 3> y,
            popnn::NonLinearityType nonLinearityType);


/**
 * Computes the backward pass of a RNN sequence.
 *
 * Loss gradients are computed at the summer and at the input in case these need
 * to be backpropagated to previous layers.
 *
 * Dimensions:
 * acts:
 * [sequence][batch][output channel]
 * nextLayerGrads:
 * [sequence][batch][output channel]
 * weightsInput:
 * [input channel][output channel]
 * weightsOutput:
 * [output channel][output channel]
 * prevLayerGrads:
 * [sequence][batch][input channel]
 * gradientSum:
 * [sequence][batch][output channel]
 */
void backward(
            const boost::multi_array_ref<double, 3> acts,
            const boost::multi_array_ref<double, 3> nextLayerGrads,
            const boost::multi_array_ref<double, 2> weightsInput,
            const boost::multi_array_ref<double, 2> weightsFeedback,
            boost::multi_array_ref<double, 3> prevLayerGrads,
            boost::multi_array_ref<double, 3> gradientSum,
            popnn::NonLinearityType nonLinearityType);

/**
 * Compute the parameter deltas for the whole sequence of an RNN
 *
 * The deltas are computed for all the parameters across sequence steps and
 * batch elements.
 *
 * Dimensions:
 * actsIn:
 * [sequence][batch][input channel]
 * initState:
 * [batch][output channel]
 * actsOut:
 * [sequence][batch][output channel]
 * gradientSum:
 * [sequence][batch][output channel]
 * weightsInputDeltas
 * [input channel][output channel]
 * weightsFeedbackDeltas
 * [output channel][output channel]
 * biasesDeltas
 * [output channel]
 */
void paramUpdate(
            const boost::multi_array_ref<double, 3> actsIn,
            const boost::multi_array_ref<double, 2> initState,
            const boost::multi_array_ref<double, 3> actsOut,
            const boost::multi_array_ref<double, 3> gradientSum,
            boost::multi_array_ref<double, 2> weightsInputDeltas,
            boost::multi_array_ref<double, 2> weightsFeedbackDeltas,
            boost::multi_array_ref<double, 1> biasesDeltas);

} // End namespace rnn.
} // End namespace poplibs_test.

#endif // poplibs_test_Rnn_hpp
