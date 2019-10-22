#include <cassert>
#include <iostream>
#include <poplibs_test/Norms.hpp>
#include <poplibs_test/exceptions.hpp>

// Number of groups in group norm
static unsigned getNumGroups(std::size_t batchSize, std::size_t numChans,
                             std::size_t statsSize) {
  assert(statsSize % batchSize == 0);
  return statsSize / batchSize;
}

// Number of channels per group in group norm
static unsigned getNumChansPerGroups(std::size_t batchSize,
                                     std::size_t numChans,
                                     std::size_t statsSize) {
  const auto numGroups = getNumGroups(batchSize, numChans, statsSize);
  assert(numChans % numGroups == 0);
  return numChans / numGroups;
}

// Of group norm class
static bool isOfGroupNormType(poplibs_test::norm::NormType normType) {
  return normType == poplibs_test::norm::NormType::GroupNorm ||
         normType == poplibs_test::norm::NormType::LayerNorm ||
         normType == poplibs_test::norm::NormType::InstanceNorm;
}

// valid norm test
static bool isSupportedNormType(poplibs_test::norm::NormType normType) {
  return normType == poplibs_test::norm::NormType::BatchNorm ||
         normType == poplibs_test::norm::NormType::GroupNorm ||
         normType == poplibs_test::norm::NormType::LayerNorm ||
         normType == poplibs_test::norm::NormType::InstanceNorm;
}

static void batchNormEstimates(const boost::multi_array_ref<double, 3> actsIn,
                               double eps, bool unbiasedVarEstimate,
                               boost::multi_array_ref<double, 1> mean,
                               boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize = actsIn.shape()[0];
  const unsigned numChannels = actsIn.shape()[1];
  const unsigned numFieldElems = actsIn.shape()[2];
  const auto numElems = batchSize * numFieldElems;

  assert(iStdDev.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double sum = 0;
    double sumSquares = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned f = 0; f != numFieldElems; ++f) {
        sum += actsIn[b][c][f];
        sumSquares += actsIn[b][c][f] * actsIn[b][c][f];
      }
    }

    // unbiased sample mean
    mean[c] = sum / numElems;
    const auto biasedVar = sumSquares / numElems - mean[c] * mean[c];
    const auto correctedVar =
        numElems == 1
            ? 1.0
            : (unbiasedVarEstimate ? biasedVar * numElems / (numElems - 1)
                                   : biasedVar);
    iStdDev[c] = 1.0 / std::sqrt(correctedVar + eps);
  }
}

void static groupNormEstimates(const boost::multi_array_ref<double, 3> actsIn,
                               double eps, bool unbiasedVarEstimate,
                               boost::multi_array_ref<double, 1> mean,
                               boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize = actsIn.shape()[0];
  const unsigned numChannels = actsIn.shape()[1];
  const auto numGroups = getNumGroups(batchSize, numChannels, mean.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChannels, mean.shape()[0]);

  const unsigned numFieldElems = actsIn.shape()[2];

  assert(iStdDev.shape()[0] == mean.shape()[0]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      double sum = 0;
      double sumSquares = 0;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const auto c = cpg * numGroups + g;
        for (unsigned f = 0; f != numFieldElems; ++f) {
          sum += actsIn[b][c][f];
          sumSquares += actsIn[b][c][f] * actsIn[b][c][f];
        }
      }
      const auto statIndex = b * numGroups + g;
      const auto numElems = numFieldElems * chansPerGroup;
      mean[statIndex] = sum / numElems;
      const auto biasedVar =
          sumSquares / numElems - mean[statIndex] * mean[statIndex];
      const auto correctedVar =
          numElems == 1
              ? 1.0
              : (unbiasedVarEstimate ? biasedVar * numElems / (numElems - 1)
                                     : biasedVar);
      iStdDev[statIndex] = 1.0 / std::sqrt(correctedVar + eps);
    }
  }
}

static void batchNormalise(const boost::multi_array_ref<double, 3> acts,
                           const boost::multi_array_ref<double, 1> gamma,
                           const boost::multi_array_ref<double, 1> beta,
                           const boost::multi_array_ref<double, 1> mean,
                           const boost::multi_array_ref<double, 1> iStdDev,
                           boost::multi_array_ref<double, 3> actsOut,
                           boost::multi_array_ref<double, 3> actsWhitened) {

  const unsigned batchSize = acts.shape()[0];
  const unsigned numChannels = acts.shape()[1];
  const unsigned numFieldElems = acts.shape()[2];

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);
  assert(iStdDev.shape()[0] == mean.shape()[0]);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numChannels);
  assert(actsOut.shape()[2] == numFieldElems);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numChannels);
  assert(actsWhitened.shape()[2] == numFieldElems);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned f = 0; f != numFieldElems; ++f) {
      for (unsigned c = 0; c != numChannels; ++c) {
        actsWhitened[b][c][f] = (acts[b][c][f] - mean[c]) * iStdDev[c];
        actsOut[b][c][f] = gamma[c] * actsWhitened[b][c][f] + beta[c];
      }
    }
  }
}

static void groupNormalise(const boost::multi_array_ref<double, 3> acts,
                           const boost::multi_array_ref<double, 1> gamma,
                           const boost::multi_array_ref<double, 1> beta,
                           const boost::multi_array_ref<double, 1> mean,
                           const boost::multi_array_ref<double, 1> iStdDev,
                           boost::multi_array_ref<double, 3> actsOut,
                           boost::multi_array_ref<double, 3> actsWhitened) {

  const auto batchSize = acts.shape()[0];
  const auto numChannels = acts.shape()[1];
  const auto numFieldElems = acts.shape()[2];

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);
  assert(iStdDev.shape()[0] == mean.shape()[0]);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numChannels);
  assert(actsOut.shape()[2] == numFieldElems);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numChannels);
  assert(actsWhitened.shape()[2] == numFieldElems);

  const auto numGroups = getNumGroups(batchSize, numChannels, mean.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChannels, mean.shape()[0]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      const auto statIndex = b * numGroups + g;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const auto c = cpg * numGroups + g;
        for (unsigned f = 0; f != numFieldElems; ++f) {
          actsWhitened[b][c][f] =
              (acts[b][c][f] - mean[statIndex]) * iStdDev[statIndex];
        }
      }
    }
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned f = 0; f != numFieldElems; ++f) {
      for (unsigned c = 0; c != numChannels; ++c) {
        actsOut[b][c][f] = gamma[c] * actsWhitened[b][c][f] + beta[c];
      }
    }
  }
}

static void
batchNormGradients(const boost::multi_array_ref<double, 3> actsWhitened,
                   const boost::multi_array_ref<double, 3> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 3> gradsOut) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChannels = actsWhitened.shape()[1];
  const unsigned numFieldElems = actsWhitened.shape()[2];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == numFieldElems);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numChannels);
  assert(gradsOut.shape()[2] == numFieldElems);

  assert(iStdDev.shape()[0] == numChannels);
  assert(gamma.shape()[0] == numChannels);

  const auto numElements = batchSize * numFieldElems;

  for (unsigned c = 0; c != numChannels; ++c) {
    double sumGradsIn = 0;
    double sumGradsInAndxMu = 0;

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned f = 0; f != numFieldElems; ++f) {
        sumGradsIn += gradsIn[b][c][f];
        sumGradsInAndxMu += actsWhitened[b][c][f] * gradsIn[b][c][f];
      }
    }

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned f = 0; f != numFieldElems; ++f) {
        double out = gradsIn[b][c][f] -
                     actsWhitened[b][c][f] * sumGradsInAndxMu / numElements -
                     sumGradsIn / numElements;

        gradsOut[b][c][f] = out * gamma[c] * iStdDev[c];
      }
    }
  }
}

static void
groupNormGradients(const boost::multi_array_ref<double, 3> actsWhitened,
                   const boost::multi_array_ref<double, 3> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 3> gradsOut) {
  const auto batchSize = actsWhitened.shape()[0];
  const auto numChannels = actsWhitened.shape()[1];
  const auto numFieldElems = actsWhitened.shape()[2];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == numFieldElems);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numChannels);
  assert(gradsOut.shape()[2] == numFieldElems);
  assert(gamma.shape()[0] == numChannels);

  const auto numGroups =
      getNumGroups(batchSize, numChannels, iStdDev.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChannels, iStdDev.shape()[0]);

  boost::multi_array<double, 3> gradsNorm(
      boost::extents[batchSize][numChannels][numFieldElems]);
  for (unsigned c = 0; c != numChannels; ++c) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned f = 0; f != numFieldElems; ++f) {
        gradsNorm[b][c][f] = gradsIn[b][c][f] * gamma[c];
      }
    }
  }

  boost::multi_array<double, 1> varGrad(boost::extents[batchSize * numGroups]);
  boost::multi_array<double, 1> meanGrad(boost::extents[batchSize * numGroups]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      double varGradAcc = 0;
      double meanGradAcc = 0;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = cpg * numGroups + g;
        for (unsigned f = 0; f != numFieldElems; ++f) {
          varGradAcc += actsWhitened[b][c][f] * gradsNorm[b][c][f];
          meanGradAcc += gradsNorm[b][c][f];
        }
      }
      const unsigned statIndex = b * numGroups + g;
      varGrad[statIndex] = varGradAcc;
      meanGrad[statIndex] = meanGradAcc;
    }
  }

  double scale = 1.0 / (chansPerGroup * numFieldElems);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = cpg * numGroups + g;
        const unsigned statIndex = b * numGroups + g;
        for (unsigned f = 0; f != numFieldElems; ++f) {
          gradsOut[b][c][f] =
              (gradsNorm[b][c][f] -
               scale * actsWhitened[b][c][f] * varGrad[statIndex] -
               scale * meanGrad[statIndex]) *
              iStdDev[statIndex];
        }
      }
    }
  }
}

static void paramUpdate(const boost::multi_array_ref<double, 3> actsWhitened,
                        const boost::multi_array_ref<double, 3> gradsIn,
                        double learningRate,
                        boost::multi_array_ref<double, 1> gamma,
                        boost::multi_array_ref<double, 1> beta) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChannels = actsWhitened.shape()[1];
  const unsigned numFieldElems = actsWhitened.shape()[2];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == numFieldElems);

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double dBeta = 0;
    double dGamma = 0;

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned f = 0; f != numFieldElems; ++f) {
        dBeta += gradsIn[b][c][f];
        dGamma += actsWhitened[b][c][f] * gradsIn[b][c][f];
      }
    }
    beta[c] -= learningRate * dBeta;
    gamma[c] -= learningRate * dGamma;
  }
}

void poplibs_test::norm::normStatistics(
    const boost::multi_array_ref<double, 3> actsIn, double eps,
    bool unbiasedVarEstimate, boost::multi_array_ref<double, 1> mean,
    boost::multi_array_ref<double, 1> iStdDev, NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormEstimates(actsIn, eps, unbiasedVarEstimate, mean, iStdDev);
  } else if (isOfGroupNormType(normType)) {
    groupNormEstimates(actsIn, eps, unbiasedVarEstimate, mean, iStdDev);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::normalise(
    const boost::multi_array_ref<double, 3> actsIn,
    const boost::multi_array_ref<double, 1> gamma,
    const boost::multi_array_ref<double, 1> beta,
    const boost::multi_array_ref<double, 1> mean,
    const boost::multi_array_ref<double, 1> iStdDev,
    boost::multi_array_ref<double, 3> actsOut,
    boost::multi_array_ref<double, 3> actsWhitened, NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormalise(actsIn, gamma, beta, mean, iStdDev, actsOut, actsWhitened);
  } else if (isOfGroupNormType(normType)) {
    groupNormalise(actsIn, gamma, beta, mean, iStdDev, actsOut, actsWhitened);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::normGradients(
    const boost::multi_array_ref<double, 3> actsWhitened,
    const boost::multi_array_ref<double, 3> gradsIn,
    const boost::multi_array_ref<double, 1> iStdDev,
    const boost::multi_array_ref<double, 1> gamma,
    boost::multi_array_ref<double, 3> gradsOut, NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormGradients(actsWhitened, gradsIn, iStdDev, gamma, gradsOut);
  } else if (isOfGroupNormType(normType)) {
    groupNormGradients(actsWhitened, gradsIn, iStdDev, gamma, gradsOut);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::normParamUpdate(
    const boost::multi_array_ref<double, 3> actsWhitened,
    const boost::multi_array_ref<double, 3> gradsIn, double learningRate,
    boost::multi_array_ref<double, 1> gamma,
    boost::multi_array_ref<double, 1> beta, NormType normType) {
  if (!isSupportedNormType(normType)) {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
  paramUpdate(actsWhitened, gradsIn, learningRate, gamma, beta);
}
