#include <poplib_test/FullyConnected.hpp>
#include <poplib_test/exceptions.hpp>

void poplib_test::fc::fullyConnected(
            const boost::multi_array<double, 2> &in,
            const boost::multi_array<double, 2> &weights,
            const boost::multi_array<double, 1> &biases,
            boost::multi_array<double, 2> &out) {
  const auto batchSize = in.shape()[0];
  const auto inputSize = in.shape()[1];
  const auto outputSize = weights.shape()[1];
  assert(weights.shape()[0] == inputSize);
  assert(out.shape()[0] == batchSize);
  assert(out.shape()[1] == outputSize);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < outputSize; ++i) {
      double sum = 0;
      for (unsigned j = 0; j < inputSize; ++j) {
        sum += in[b][j] * weights[j][i];
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
  const auto outputSize = weights.shape()[0];
  assert(weights.shape()[1] == inputSize);
  assert(out.shape()[0] == batchSize);
  assert(out.shape()[1] == outputSize);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < outputSize; ++i) {
      double sum = 0;
      for (unsigned j = 0; j < inputSize; ++j) {
        sum += in[b][j] * weights[i][j];
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
  assert(weights.shape()[0] == inputSize);
  assert(weights.shape()[1] == outputSize);

  boost::multi_array<double, 2>
      weightDeltas(boost::extents[inputSize][outputSize]);
  std::fill(weightDeltas.data(),
            weightDeltas.data() + weightDeltas.num_elements(), 0.0);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned i = 0; i < inputSize; ++i) {
      for (unsigned j = 0; j < outputSize; ++j) {
        weightDeltas[i][j] += activations[b][i] * deltas[b][j];
      }
    }
  }

  for (unsigned i = 0; i < inputSize; ++i) {
    for (unsigned j = 0; j < outputSize; ++j) {
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

void poplib_test::fc::batchNormEstimates(
                  const boost::multi_array_ref<double, 2> actsIn,
                  double eps,
                  boost::multi_array_ref<double, 1> mean,
                  boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize = actsIn.shape()[0];
  const unsigned numActs = actsIn.shape()[1];

  assert(mean.shape()[0] == numActs);
  assert(iStdDev.shape()[0] == numActs);

  for (unsigned a = 0; a != numActs; ++a) {
    double rSum = 0;
    double rSumOfSquares = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      rSum += actsIn[b][a];
      rSumOfSquares += actsIn[b][a] * actsIn[b][a];
    }
    mean[a] = batchSize == 1 ? 0 : rSum / batchSize;
    iStdDev[a] = batchSize == 1 ? 1.0 :
         1.0 / std::sqrt(rSumOfSquares / batchSize - mean[a] * mean[a] + eps);
  }
}


void poplib_test::fc::
batchNormalise(const boost::multi_array_ref<double, 2> acts,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> iStdDev,
               boost::multi_array_ref<double, 2> actsOut,
               boost::multi_array_ref<double, 2> actsWhitened) {

  const unsigned batchSize = acts.shape()[0];
  const unsigned numActs = acts.shape()[1];

  assert(gamma.shape()[0] == numActs);
  assert(beta.shape()[0] == numActs);
  assert(mean.shape()[0] == numActs);
  assert(iStdDev.shape()[0] == numActs);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numActs);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numActs);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned a = 0; a != numActs; ++a) {
      actsWhitened[b][a] = (acts[b][a] - mean[a]) *  iStdDev[a];
      actsOut[b][a] = actsWhitened[b][a] * gamma[a] + beta[a];
    }
  }
}


void poplib_test::fc::
batchNormGradients(const boost::multi_array_ref<double, 2> actsWhitened,
                   const boost::multi_array_ref<double, 2> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 2> gradsOut) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numActs = actsWhitened.shape()[1];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numActs);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numActs);
  assert(iStdDev.shape()[0] == numActs);
  assert(gamma.shape()[0] == numActs);

  for (unsigned a = 0; a != numActs; ++a) {
    double sumGradsIn = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      sumGradsIn += gradsIn[b][a];
    }
    double sumGradsInAndxMu = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      sumGradsInAndxMu += actsWhitened[b][a] * gradsIn[b][a];
    }

    for (unsigned b = 0; b != batchSize; ++b) {
      double out =
        gradsIn[b][a]
        - actsWhitened[b][a] * sumGradsInAndxMu / batchSize
        - sumGradsIn / batchSize;

      gradsOut[b][a] = out * gamma[a] * iStdDev[a];
    }
  }
}

void poplib_test::fc::
batchNormParamUpdate(const boost::multi_array_ref<double, 2> actsWhitened,
                     const boost::multi_array_ref<double, 2> gradsIn,
                     double learningRate,
                     boost::multi_array_ref<double, 1> gamma,
                     boost::multi_array_ref<double, 1> beta) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numActs = actsWhitened.shape()[1];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numActs);
  assert(gamma.shape()[0] == numActs);
  assert(beta.shape()[0] == numActs);

  for (unsigned a = 0; a != numActs; ++a) {
    double dBeta = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      dBeta += gradsIn[b][a];
    }
    beta[a] -= learningRate * dBeta;

    double dGamma = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      dGamma += actsWhitened[b][a] * gradsIn[b][a];
    }

    gamma[a] -= learningRate * dGamma;
  }
}
