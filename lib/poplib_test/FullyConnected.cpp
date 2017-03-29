#include <poplib_test/FullyConnected.hpp>
#include <poplib_test/exceptions.hpp>

void poplib_test::fc::fullyConnected(
            const boost::multi_array<double, 2> &in,
            const boost::multi_array<double, 2> &weights,
            const boost::multi_array<double, 1> &biases,
            boost::multi_array<double, 2> &out) {
  const auto batchSize = in.shape()[0];
  const auto inputSize = in.shape()[1];
  const auto outputSize = out.shape()[1];
  assert(batchSize == out.shape()[0]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < outputSize; ++i) {
      double sum = 0;
      for (unsigned j = 0; j < inputSize; ++j) {
        sum += weights[i][j] * in[b][j];
      }
      out[b][i] = sum + biases[i];
    }
  }
}

void poplib_test::fc::fullyConnectedBackward(
    const boost::multi_array<double, 2> &in,
    const boost::multi_array<double, 2> &weights,
    boost::multi_array<double, 2> &out) {
  const auto batchSize = in.shape()[0];
  const auto inputSize = in.shape()[1];
  const auto outputSize = out.shape()[1];
  assert(batchSize == out.shape()[0]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < outputSize; ++i) {
      double sum = 0;
      for (unsigned j = 0; j < inputSize; ++j) {
        sum += weights[j][i] * in[b][j];
      }
      out[b][i] = sum;
    }
  }
}

void poplib_test::fc::fullyConnectedWeightUpdate(
                  double learningRate,
                  const boost::multi_array<double, 2> &activations,
                  const boost::multi_array<double, 2> &deltas,
                  boost::multi_array<double, 2> &weights,
                  boost::multi_array<double, 1> &biases) {
  const auto batchSize = activations.shape()[0];
  const auto inputSize = activations.shape()[1];
  const auto outputSize = deltas.shape()[1];
  assert(batchSize == deltas.shape()[0]);

  boost::multi_array<double, 2>
      weightDeltas(boost::extents[outputSize][inputSize]);
  std::fill(weightDeltas.data(),
            weightDeltas.data() + weightDeltas.num_elements(), 0.0);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < outputSize; ++i) {
      for (unsigned j = 0; j < inputSize; ++j) {
        weightDeltas[i][j] += activations[b][j] * deltas[b][i];
      }
    }
  }

  for (unsigned i = 0; i < outputSize; ++i) {
    for (unsigned j = 0; j < inputSize; ++j) {
      weights[i][j] += learningRate * -weightDeltas[i][j];
    }
  }

  boost::multi_array<double, 1> biasDeltas(boost::extents[outputSize]);
  std::fill(biasDeltas.data(),
            biasDeltas.data() + biasDeltas.num_elements(), 0.0);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < outputSize; ++i) {
      biasDeltas[i] += deltas[b][i];
    }
  }

  for (unsigned i = 0; i < outputSize; ++i) {
    biases[i] += learningRate * -biasDeltas[i];
  }
}
