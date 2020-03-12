// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Rnn.hpp>
#include <poplibs_test/exceptions.hpp>

void poplibs_test::rnn::forwardWeightInput(
    const boost::multi_array_ref<double, 3> x,
    const boost::multi_array_ref<double, 2> weights,
    boost::multi_array_ref<double, 3> y) {

  const auto sequenceSize = x.shape()[0];
#ifndef NDEBUG
  const auto batchSize = x.shape()[1];
  const auto inputSize = x.shape()[2];
  const auto outputSize = y.shape()[2];
#endif
  assert(y.shape()[1] == batchSize);
  assert(y.shape()[0] == sequenceSize);
  assert(weights.shape()[0] == inputSize);
  assert(weights.shape()[1] == outputSize);

  for (unsigned s = 0; s != sequenceSize; ++s) {
    boost::multi_array<double, 2> inp = x[s];
    boost::multi_array<double, 2> out = y[s];
    poplibs_test::gemm::generalMatrixMultiply(inp, weights, out, out, 1.0, 0,
                                              false, false);
    y[s] = out;
  }
}

void poplibs_test::rnn::forwardIterate(
    const boost::multi_array_ref<double, 3> x,
    const boost::multi_array_ref<double, 2> yInit,
    const boost::multi_array_ref<double, 2> weights,
    const boost::multi_array_ref<double, 1> bias,
    boost::multi_array_ref<double, 3> y,
    popnn::NonLinearityType nonLinearityType) {

  const auto sequenceSize = x.shape()[0];
  const auto batchSize = x.shape()[1];
  const auto outputSize = x.shape()[2];

  assert(yInit.shape()[0] == batchSize);
  assert(yInit.shape()[1] == outputSize);
  assert(weights.shape()[0] == outputSize);
  assert(weights.shape()[1] == outputSize);
  assert(bias.shape()[0] == outputSize);

  for (auto s = 0U; s != sequenceSize; ++s) {
    boost::multi_array<double, 2> ySm1 = (s == 0) ? y[s] : y[s - 1];
    boost::multi_array<double, 2> yS = y[s];
    boost::multi_array<double, 2> xS = x[s];
    const boost::multi_array_ref<double, 2> yPrev = s == 0 ? yInit : ySm1;
    poplibs_test::gemm::generalMatrixMultiply(yPrev, weights, xS, yS, 1.0, 1.0,
                                              false, false);
    /* apply bias */
    for (unsigned b = 0U; b != batchSize; ++b) {
      for (unsigned i = 0U; i != outputSize; ++i) {
        yS[b][i] = yS[b][i] + bias[i];
      }
    }
    poplibs_test::nonLinearity(nonLinearityType, yS);
    y[s] = yS;
  }
}

void poplibs_test::rnn::backward(
    const boost::multi_array_ref<double, 3> acts,
    const boost::multi_array_ref<double, 3> nextLayerGrads,
    const boost::multi_array_ref<double, 2> weightsInput,
    const boost::multi_array_ref<double, 2> weightsFeedback,
    boost::multi_array_ref<double, 3> prevLayerGrads,
    boost::multi_array_ref<double, 3> gradientSum,
    popnn::NonLinearityType nonLinearityType) {
  const auto sequenceSize = nextLayerGrads.shape()[0];

#ifndef NDEBUG
  const auto batchSize = nextLayerGrads.shape()[1];
  const auto outputSize = nextLayerGrads.shape()[2];
  const auto inputSize = weightsInput.shape()[0];
#endif

  assert(weightsInput.shape()[1] == outputSize);
  assert(weightsFeedback.shape()[0] == outputSize);
  assert(weightsFeedback.shape()[1] == outputSize);
  assert(gradientSum.shape()[0] == sequenceSize);
  assert(gradientSum.shape()[1] == batchSize);
  assert(gradientSum.shape()[2] == outputSize);
  assert(prevLayerGrads.shape()[0] == sequenceSize);
  assert(prevLayerGrads.shape()[1] == batchSize);
  assert(prevLayerGrads.shape()[2] == inputSize);

  boost::multi_array<double, 2> gradSumThisStep =
      nextLayerGrads[sequenceSize - 1];
  for (auto i = sequenceSize; i != 0; --i) {
    const auto s = i - 1;
    if (s != sequenceSize - 1) {
      const boost::multi_array<double, 2> &gradIn = nextLayerGrads[s];
      const boost::multi_array<double, 2> &prevGradSum = gradientSum[s + 1];
      poplibs_test::gemm::generalMatrixMultiply(prevGradSum, weightsFeedback,
                                                gradIn, gradSumThisStep, 1.0,
                                                1.0, false, true);
    }
    const boost::multi_array<double, 2> &actsThisStep = acts[s];

    bwdNonLinearity(nonLinearityType, actsThisStep, gradSumThisStep);
    gradientSum[s] = gradSumThisStep;

    boost::multi_array<double, 2> feedfwdGrad = prevLayerGrads[s];
    poplibs_test::gemm::generalMatrixMultiply(gradSumThisStep, weightsInput,
                                              feedfwdGrad, feedfwdGrad, 1.0, 0,
                                              false, true);
    prevLayerGrads[s] = feedfwdGrad;
  }
}

void poplibs_test::rnn::paramUpdate(
    const boost::multi_array_ref<double, 3> actsIn,
    const boost::multi_array_ref<double, 2> initState,
    const boost::multi_array_ref<double, 3> actsOut,
    const boost::multi_array_ref<double, 3> gradientSum,
    boost::multi_array_ref<double, 2> weightsInputDeltas,
    boost::multi_array_ref<double, 2> weightsFeedbackDeltas,
    boost::multi_array_ref<double, 1> biasesDeltas) {
  const auto sequenceSize = actsIn.shape()[0];
  const auto batchSize = actsIn.shape()[1];
  const auto outputSize = actsOut.shape()[2];

#ifndef NDEBUG
  const auto inputSize = actsIn.shape()[2];
#endif

  assert(actsOut.shape()[0] == sequenceSize);
  assert(actsOut.shape()[1] == batchSize);
  assert(gradientSum.shape()[0] == sequenceSize);
  assert(gradientSum.shape()[1] == batchSize);
  assert(gradientSum.shape()[2] == outputSize);
  assert(weightsInputDeltas.shape()[0] == inputSize);
  assert(weightsInputDeltas.shape()[1] == outputSize);
  assert(weightsFeedbackDeltas.shape()[0] == outputSize);
  assert(weightsFeedbackDeltas.shape()[1] == outputSize);
  assert(biasesDeltas.shape()[0] == outputSize);
  assert(initState.shape()[0] == batchSize);
  assert(initState.shape()[1] == outputSize);

  // zero
  for (auto it = weightsInputDeltas.data(),
            end = weightsInputDeltas.data() + weightsInputDeltas.num_elements();
       it != end; ++it) {
    *it = 0;
  }
  for (auto it = weightsFeedbackDeltas.data(),
            end = weightsFeedbackDeltas.data() +
                  weightsFeedbackDeltas.num_elements();
       it != end; ++it) {
    *it = 0;
  }
  for (auto it = biasesDeltas.data(),
            end = biasesDeltas.data() + biasesDeltas.num_elements();
       it != end; ++it) {
    *it = 0;
  }

  for (auto i = sequenceSize; i != 0; --i) {
    const auto s = i - 1;
    const boost::multi_array<double, 2> &in1 = actsIn[s];
    const boost::multi_array<double, 2> &in2 = gradientSum[s];
    poplibs_test::gemm::generalMatrixMultiply(in1, in2, weightsInputDeltas,
                                              weightsInputDeltas, 1.0, 1.0,
                                              true, false);
    boost::multi_array<double, 2> in3(boost::extents[batchSize][outputSize]);
    if (s == 0) {
      in3 = initState;
    } else {
      in3 = actsOut[s - 1];
    }
    poplibs_test::gemm::generalMatrixMultiply(in3, in2, weightsFeedbackDeltas,
                                              weightsFeedbackDeltas, 1.0, 1.0,
                                              true, false);
    for (auto o = 0U; o != outputSize; ++o) {
      for (auto b = 0U; b != batchSize; ++b) {
        biasesDeltas[o] += in2[b][o];
      }
    }
  }
}
