#include <poplibs_test/FullyConnected.hpp>
#include <poplibs_test/exceptions.hpp>

void poplibs_test::fc::fullyConnected(
            const boost::multi_array<double, 3> &in,
            const boost::multi_array<double, 3> &weights,
            const boost::multi_array<double, 2> &biases,
            boost::multi_array<double, 3> &out) {
  const auto numGroups = in.shape()[0];
  const auto batchSize = in.shape()[1];
  const auto inputSize = in.shape()[2];
  const auto outputSize = weights.shape()[2];
  assert(weights.shape()[0] == numGroups);
  assert(out.shape()[0] == numGroups);
  assert(biases.shape()[0] == numGroups);
  assert(weights.shape()[1] == inputSize);
  assert(out.shape()[1] == batchSize);
  assert(out.shape()[2] == outputSize);

  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned i = 0; i < outputSize; ++i) {
        double sum = 0;
        for (unsigned j = 0; j < inputSize; ++j) {
          sum += in[g][b][j] * weights[g][j][i];
        }
        out[g][b][i] = sum + biases[g][i];
      }
    }
  }
}

void poplibs_test::fc::fullyConnectedBackward(
    const boost::multi_array<double, 3> &in,
    const boost::multi_array<double, 3> &weights,
    boost::multi_array<double, 3> &out) {
  const auto numGroups = in.shape()[0];
  const auto batchSize = in.shape()[1];
  const auto inputSize = in.shape()[2];
  const auto outputSize = weights.shape()[1];
  assert(weights.shape()[0] == numGroups);
  assert(out.shape()[0] == numGroups);
  assert(weights.shape()[2] == inputSize);
  assert(out.shape()[1] == batchSize);
  assert(out.shape()[2] == outputSize);

  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned i = 0; i < outputSize; ++i) {
        double sum = 0;
        for (unsigned j = 0; j < inputSize; ++j) {
          sum += in[g][b][j] * weights[g][i][j];
        }
        out[g][b][i] = sum;
      }
    }
  }
}

void poplibs_test::fc::fullyConnectedWeightUpdate(
                  double learningRate,
                  const boost::multi_array<double, 3> &activations,
                  const boost::multi_array<double, 3> &deltas,
                  boost::multi_array<double, 3> &weights,
                  boost::multi_array<double, 2> &biases) {
  const auto numGroups = activations.shape()[0];
  const auto batchSize = activations.shape()[1];
  const auto inputSize = activations.shape()[2];
  const auto outputSize = deltas.shape()[2];
  assert(deltas.shape()[0] == numGroups);
  assert(weights.shape()[0] == numGroups);
  assert(biases.shape()[0] == numGroups);
  assert(batchSize == deltas.shape()[1]);
  assert(weights.shape()[1] == inputSize);
  assert(weights.shape()[2] == outputSize);

  boost::multi_array<double, 3>
      weightDeltas(boost::extents[numGroups][inputSize][outputSize]);
  std::fill(weightDeltas.data(),
            weightDeltas.data() + weightDeltas.num_elements(), 0.0);

  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned i = 0; i < inputSize; ++i) {
        for (unsigned j = 0; j < outputSize; ++j) {
          weightDeltas[g][i][j] += activations[g][b][i] * deltas[g][b][j];
        }
      }
    }
  }

  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned i = 0; i < inputSize; ++i) {
      for (unsigned j = 0; j < outputSize; ++j) {
        weights[g][i][j] += learningRate * -weightDeltas[g][i][j];
      }
    }
  }

  boost::multi_array<double, 2> biasDeltas(boost::extents[numGroups]
                                                         [outputSize]);
  std::fill(biasDeltas.data(),
            biasDeltas.data() + biasDeltas.num_elements(), 0.0);

  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned i = 0; i < outputSize; ++i) {
        biasDeltas[g][i] += deltas[g][b][i];
      }
    }
  }

  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned i = 0; i < outputSize; ++i) {
      biases[g][i] += learningRate * -biasDeltas[g][i];
    }
  }
}

void poplibs_test::fc::batchNormEstimates(
                  const boost::multi_array_ref<double, 2> actsIn,
                  double eps,
                  bool unbiasedVarEstimate,
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
    const auto biasedVar = rSumOfSquares / batchSize - mean[a] * mean[a];
    const auto correctedVar = batchSize == 1 ?  1.0 :
      (unbiasedVarEstimate ? biasedVar * batchSize / (batchSize - 1) :
                             biasedVar);
    iStdDev[a] = 1.0 / std::sqrt(correctedVar + eps);
  }
}


void poplibs_test::fc::
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


void poplibs_test::fc::
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

void poplibs_test::fc::
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
