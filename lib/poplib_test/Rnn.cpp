#include <poplib_test/GeneralMatrixMultiply.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Rnn.hpp>
#include <poplib_test/exceptions.hpp>
#include <cassert>

void poplib_test::rnn::forwardWeightInput(
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
  assert(weights.shape()[0] == outputSize);
  assert(weights.shape()[1] == inputSize);

  for (unsigned s = 0; s != sequenceSize; ++s) {
    boost::multi_array<double, 2> inp = x[s];
    boost::multi_array<double, 2> out = y[s];
    poplib_test::gemm::generalMatrixMultiply(inp, weights, out, out,
                                             1.0, 0, false, true);
    y[s] = out;
  }
}

void poplib_test::rnn::forwardIterate(
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
    poplib_test::gemm::generalMatrixMultiply(yPrev, weights, xS, yS, 1.0, 1.0,
                                             false, true);
    /* apply bias */
    for (unsigned b = 0U; b != batchSize; ++b) {
      for (unsigned i = 0U; i != outputSize; ++i) {
        yS[b][i] = yS[b][i] + bias[i];
      }
    }
    poplib_test::nonLinearity(nonLinearityType, yS);
    y[s] = yS;
  }
}
