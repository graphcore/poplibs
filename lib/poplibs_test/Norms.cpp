#include <poplibs_test/Norms.hpp>
#include <poplibs_test/exceptions.hpp>
#include <iostream>

// Number of groups in group norm
static unsigned getNumGroups(std::size_t batchSize,
                             std::size_t numChans,
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


static void batchNormEstimates(
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

static void groupNormEstimates(
                  const boost::multi_array_ref<double, 2> actsIn,
                  double eps,
                  bool unbiasedVarEstimate,
                  boost::multi_array_ref<double, 1> mean,
                  boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize = actsIn.shape()[0];
  const unsigned numChans = actsIn.shape()[1];
  const unsigned numGroups = getNumGroups(batchSize, numChans, mean.shape()[0]);
  assert(iStdDev.shape()[0] == mean.shape()[0]);
  const unsigned chansPerGroup =
      getNumChansPerGroups(batchSize, numChans, mean.shape()[0]);
  const auto numElems = chansPerGroup;
  for (unsigned g = 0; g != numGroups; ++g) {
    for (unsigned b = 0; b != batchSize; ++b) {
      double rSum = 0;
      double rSumOfSquares = 0;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        unsigned chan = g * chansPerGroup + cpg;
        rSum += actsIn[b][chan];
        rSumOfSquares += actsIn[b][chan] * actsIn[b][chan];
      }
      const unsigned statIndex = b * numGroups + g;
      mean[statIndex] = numElems == 1 ? 0 : rSum / numElems;
      const auto biasedVar =
          rSumOfSquares / numElems - mean[statIndex] * mean[statIndex];
      const auto correctedVar = numElems == 1 ?  1.0 :
        (unbiasedVarEstimate ? biasedVar * numElems / (numElems - 1) :
                               biasedVar);
      iStdDev[statIndex] = 1.0 / std::sqrt(correctedVar + eps);
    }
  }
}

static void
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

static void
groupNormalise(const boost::multi_array_ref<double, 2> acts,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> iStdDev,
               boost::multi_array_ref<double, 2> actsOut,
               boost::multi_array_ref<double, 2> actsWhitened) {

  const unsigned batchSize = acts.shape()[0];
  const unsigned numChans = acts.shape()[1];
  const auto numGroups = getNumGroups(batchSize, numChans, mean.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChans,  mean.shape()[0]);
  assert(iStdDev.shape()[0] == mean.shape()[0]);
  assert(gamma.shape()[0] == numChans);
  assert(beta.shape()[0] == numChans);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numChans);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numChans);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = g * chansPerGroup + cpg;
        const unsigned statIndex = b * numGroups + g;
        actsWhitened[b][c] =
            (acts[b][c] - mean[statIndex]) *  iStdDev[statIndex];
      }
    }
  }

  for (unsigned c = 0; c != numChans; ++c) {
    for (unsigned b = 0; b != batchSize; ++b) {
      actsOut[b][c] = actsWhitened[b][c] * gamma[c] + beta[c];
    }
  }
}


static void
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

static void
groupNormGradients(const boost::multi_array_ref<double, 2> actsWhitened,
                   const boost::multi_array_ref<double, 2> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 2> gradsOut) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChans = actsWhitened.shape()[1];
  const auto numGroups = getNumGroups(batchSize, numChans, iStdDev.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChans, iStdDev.shape()[0]);

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChans);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numChans);
  assert(gamma.shape()[0] == numChans);

  boost::multi_array<double, 2>
      gradsNorm(boost::extents[batchSize]
                              [numChans]);
  for (unsigned c = 0; c != numChans; ++c) {
    for (unsigned b = 0; b != batchSize; ++b) {
      gradsNorm[b][c] = gradsIn[b][c] * gamma[c];
    }
  }
  boost::multi_array<double, 1>
      varGrad(boost::extents[batchSize * numGroups]);
  boost::multi_array<double, 1>
      meanGrad(boost::extents[batchSize * numGroups]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      double varGradAcc = 0;
      double meanGradAcc = 0;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = g * chansPerGroup + cpg;
        varGradAcc += actsWhitened[b][c] * gradsNorm[b][c];
        meanGradAcc += gradsNorm[b][c];
      }
      const unsigned statIndex = b * numGroups + g;
      varGrad[statIndex] = varGradAcc;
      meanGrad[statIndex] = meanGradAcc;
    }
  }

  double scale = 1.0 / chansPerGroup;
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = g * chansPerGroup + cpg;
        const unsigned statIndex = b * numGroups + g;
        gradsOut[b][c] =
            (gradsNorm[b][c] - scale * actsWhitened[b][c] * varGrad[statIndex]
             - scale * meanGrad[statIndex]) * iStdDev[statIndex];
      }
    }
  }
}


static void
paramUpdate(const boost::multi_array_ref<double, 2> actsWhitened,
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

static void
batchNormEstimates(const boost::multi_array_ref<double, 4> actsIn,
                   double eps, bool unbiasedVarEstimate,
                   boost::multi_array_ref<double, 1> mean,
                   boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize= actsIn.shape()[0];
  const unsigned numChannels = actsIn.shape()[1];
  const unsigned dimY = actsIn.shape()[2];
  const unsigned dimX = actsIn.shape()[3];
  const auto numElems = batchSize * dimX * dimY;

  assert(iStdDev.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double sum =  0;
    double sumSquares = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != dimY; ++h) {
        for (unsigned w = 0; w != dimX; ++w) {
          sum += actsIn[b][c][h][w];
          sumSquares += actsIn[b][c][h][w] * actsIn[b][c][h][w];
        }
      }
    }

    // unbiased sample mean
    mean[c] = sum / numElems;
    const auto biasedVar = sumSquares / numElems - mean[c] * mean[c];
    const auto correctedVar = numElems == 1 ?  1.0 :
      (unbiasedVarEstimate ? biasedVar * numElems / (numElems - 1) :
                             biasedVar);
    iStdDev[c] = 1.0 / std::sqrt(correctedVar + eps);
  }
}


void static
groupNormEstimates(const boost::multi_array_ref<double, 4> actsIn,
                   double eps, bool unbiasedVarEstimate,
                   boost::multi_array_ref<double, 1> mean,
                   boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize = actsIn.shape()[0];
  const unsigned numChannels = actsIn.shape()[1];
  const auto numGroups = getNumGroups(batchSize, numChannels, mean.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChannels, mean.shape()[0]);

  const unsigned dimY = actsIn.shape()[2];
  const unsigned dimX = actsIn.shape()[3];

  assert(iStdDev.shape()[0] == mean.shape()[0]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      double sum =  0;
      double sumSquares = 0;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const auto c = g *chansPerGroup + cpg;
        for (unsigned h = 0; h != dimY; ++h) {
          for (unsigned w = 0; w != dimX; ++w) {
            sum += actsIn[b][c][h][w];
            sumSquares += actsIn[b][c][h][w] * actsIn[b][c][h][w];
          }
        }
      }
      const auto statIndex = b * numGroups + g;
      const auto numElems = dimX * dimY * chansPerGroup;
      mean[statIndex] = sum / numElems;
      const auto biasedVar =
          sumSquares / numElems - mean[statIndex] * mean[statIndex];
      const auto correctedVar = numElems == 1 ? 1.0 :
              (unbiasedVarEstimate ? biasedVar * numElems / (numElems - 1) :
                                     biasedVar);
      iStdDev[statIndex] = 1.0 / std::sqrt(correctedVar + eps);
    }
  }
}

static void
batchNormalise(const boost::multi_array_ref<double, 4> acts,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> iStdDev,
               boost::multi_array_ref<double, 4> actsOut,
               boost::multi_array_ref<double, 4> actsWhitened) {

  const unsigned batchSize = acts.shape()[0];
  const unsigned numChannels = acts.shape()[1];
  const unsigned dimY = acts.shape()[2];
  const unsigned dimX = acts.shape()[3];


  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);
  assert(iStdDev.shape()[0] == mean.shape()[0]);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numChannels);
  assert(actsOut.shape()[2] == dimY);
  assert(actsOut.shape()[3] == dimX);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numChannels);
  assert(actsWhitened.shape()[2] == dimY);
  assert(actsWhitened.shape()[3] == dimX);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned h = 0; h != dimY; ++h) {
      for (unsigned w = 0; w != dimX; ++w) {
        for (unsigned c = 0; c != numChannels; ++c) {
          actsWhitened[b][c][h][w] = (acts[b][c][h][w] - mean[c]) * iStdDev[c];
          actsOut[b][c][h][w] = gamma[c] * actsWhitened[b][c][h][w] + beta[c];
        }
      }
    }
  }
}

static void
groupNormalise(const boost::multi_array_ref<double, 4> acts,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> iStdDev,
               boost::multi_array_ref<double, 4> actsOut,
               boost::multi_array_ref<double, 4> actsWhitened) {

  const auto batchSize = acts.shape()[0];
  const auto numChannels = acts.shape()[1];
  const auto dimY = acts.shape()[2];
  const auto dimX = acts.shape()[3];

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);
  assert(iStdDev.shape()[0] == mean.shape()[0]);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numChannels);
  assert(actsOut.shape()[2] == dimY);
  assert(actsOut.shape()[3] == dimX);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numChannels);
  assert(actsWhitened.shape()[2] == dimY);
  assert(actsWhitened.shape()[3] == dimX);

  const auto numGroups = getNumGroups(batchSize, numChannels, mean.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChannels, mean.shape()[0]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      const auto statIndex = b * numGroups + g;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const auto c = g *chansPerGroup + cpg;
        for (unsigned h = 0; h != dimY; ++h) {
          for (unsigned w = 0; w != dimX; ++w) {
            actsWhitened[b][c][h][w] =
                (acts[b][c][h][w] - mean[statIndex]) * iStdDev[statIndex];
          }
        }
      }
    }
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned h = 0; h != dimY; ++h) {
      for (unsigned w = 0; w != dimX; ++w) {
        for (unsigned c = 0; c != numChannels; ++c) {
          actsOut[b][c][h][w] = gamma[c] * actsWhitened[b][c][h][w] + beta[c];
        }
      }
    }
  }
}


static void
batchNormGradients(const boost::multi_array_ref<double, 4> actsWhitened,
                   const boost::multi_array_ref<double, 4> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 4> gradsOut) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChannels = actsWhitened.shape()[1];
  const unsigned height = actsWhitened.shape()[2];
  const unsigned width = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == height);
  assert(gradsIn.shape()[3] == width);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numChannels);
  assert(gradsOut.shape()[2] == height);
  assert(gradsOut.shape()[3] == width);

  assert(iStdDev.shape()[0] == numChannels);
  assert(gamma.shape()[0] == numChannels);

  const auto numElements = batchSize * height * width;

  for (unsigned c = 0; c != numChannels; ++c) {
    double sumGradsIn = 0;
    double sumGradsInAndxMu = 0;

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          sumGradsIn += gradsIn[b][c][h][w];
          sumGradsInAndxMu += actsWhitened[b][c][h][w] * gradsIn[b][c][h][w];
        }
      }
    }

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          double out =
            gradsIn[b][c][h][w]
            - actsWhitened[b][c][h][w] * sumGradsInAndxMu / numElements
            - sumGradsIn / numElements;

          gradsOut[b][c][h][w] = out * gamma[c] * iStdDev[c];
        }
      }
    }
  }
}

static void
groupNormGradients(const boost::multi_array_ref<double, 4> actsWhitened,
                   const boost::multi_array_ref<double, 4> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 4> gradsOut) {
  const auto batchSize = actsWhitened.shape()[0];
  const auto numChannels = actsWhitened.shape()[1];
  const auto height = actsWhitened.shape()[2];
  const auto width = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == height);
  assert(gradsIn.shape()[3] == width);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numChannels);
  assert(gradsOut.shape()[2] == height);
  assert(gradsOut.shape()[3] == width);
  assert(gamma.shape()[0] == numChannels);

  const auto numGroups =
      getNumGroups(batchSize, numChannels, iStdDev.shape()[0]);
  const auto chansPerGroup =
      getNumChansPerGroups(batchSize, numChannels, iStdDev.shape()[0]);

  boost::multi_array<double, 4>
      gradsNorm(boost::extents[batchSize]
                              [numChannels]
                              [height]
                              [width]);
  for (unsigned c = 0; c != numChannels; ++c) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          gradsNorm[b][c][h][w] = gradsIn[b][c][h][w] * gamma[c];
        }
      }
    }
  }

  boost::multi_array<double, 1>
      varGrad(boost::extents[batchSize * numGroups]);
  boost::multi_array<double, 1>
      meanGrad(boost::extents[batchSize * numGroups]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      double varGradAcc = 0;
      double meanGradAcc = 0;
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = g * chansPerGroup + cpg;
        for (unsigned h = 0; h != height; ++h) {
          for (unsigned w = 0; w != width; ++w) {
            varGradAcc += actsWhitened[b][c][h][w] * gradsNorm[b][c][h][w];
            meanGradAcc += gradsNorm[b][c][h][w];
          }
        }
      }
      const unsigned statIndex = b * numGroups + g;
      varGrad[statIndex] = varGradAcc;
      meanGrad[statIndex] = meanGradAcc;
    }
  }

  double scale = 1.0 / (chansPerGroup * height * width);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned g = 0; g != numGroups; ++g) {
      for (unsigned cpg = 0; cpg != chansPerGroup; ++cpg) {
        const unsigned c = g * chansPerGroup + cpg;
        const unsigned statIndex = b * numGroups + g;
        for (unsigned h = 0; h != height; ++h) {
          for (unsigned w = 0; w != width; ++w) {
            gradsOut[b][c][h][w] =
                (gradsNorm[b][c][h][w] -
                 scale * actsWhitened[b][c][h][w] * varGrad[statIndex]
                 - scale * meanGrad[statIndex]) * iStdDev[statIndex];
          }
        }
      }
    }
  }
}

static void
paramUpdate(const boost::multi_array_ref<double, 4> actsWhitened,
                const boost::multi_array_ref<double, 4> gradsIn,
                double learningRate,
                boost::multi_array_ref<double, 1> gamma,
                boost::multi_array_ref<double, 1> beta) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChannels = actsWhitened.shape()[1];
  const unsigned height = actsWhitened.shape()[2];
  const unsigned width = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == height);
  assert(gradsIn.shape()[3] == width);

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double dBeta = 0;
    double dGamma = 0;

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          dBeta += gradsIn[b][c][h][w];
          dGamma += actsWhitened[b][c][h][w] * gradsIn[b][c][h][w];
        }
      }
    }
    beta[c] -= learningRate * dBeta;
    gamma[c] -= learningRate * dGamma;
  }
}

void poplibs_test::norm::
normStatistics(const boost::multi_array_ref<double, 2> actsIn,
               double eps,
               bool unbiasedVarEstimate,
               boost::multi_array_ref<double, 1> mean,
               boost::multi_array_ref<double, 1> iStdDev,
               NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormEstimates(actsIn, eps, unbiasedVarEstimate, mean, iStdDev);
  } else if (isOfGroupNormType(normType)) {
    groupNormEstimates(actsIn, eps, unbiasedVarEstimate, mean, iStdDev);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::
normalise(const boost::multi_array_ref<double, 2> acts,
          const boost::multi_array_ref<double, 1> gamma,
          const boost::multi_array_ref<double, 1> beta,
          const boost::multi_array_ref<double, 1> mean,
          const boost::multi_array_ref<double, 1> iStdDev,
          boost::multi_array_ref<double, 2> actsOut,
          boost::multi_array_ref<double, 2> actsWhitened,
          NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormalise(acts, gamma, beta, mean, iStdDev, actsOut, actsWhitened);
  } else if (isOfGroupNormType(normType)) {
    groupNormalise(acts, gamma, beta, mean, iStdDev, actsOut, actsWhitened);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::
normGradients(const boost::multi_array_ref<double, 2> actsWhitened,
              const boost::multi_array_ref<double, 2> gradsIn,
              const boost::multi_array_ref<double, 1> iStdDev,
              const boost::multi_array_ref<double, 1> gamma,
              boost::multi_array_ref<double, 2> gradsOut,
              NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormGradients(actsWhitened, gradsIn, iStdDev, gamma, gradsOut);
  } else if (isOfGroupNormType(normType)) {
    groupNormGradients(actsWhitened, gradsIn, iStdDev, gamma, gradsOut);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::
normParamUpdate(const boost::multi_array_ref<double, 2> actsWhitened,
                const boost::multi_array_ref<double, 2> gradsIn,
                double learningRate,
                boost::multi_array_ref<double, 1> gamma,
                boost::multi_array_ref<double, 1> beta,
                NormType normType) {
  if (!isSupportedNormType(normType)) {
     throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
  paramUpdate(actsWhitened, gradsIn, learningRate, gamma, beta);
}

void poplibs_test::norm::
normStatistics(const boost::multi_array_ref<double, 4> actsIn,
               double eps, bool unbiasedVarEstimate,
               boost::multi_array_ref<double, 1> mean,
               boost::multi_array_ref<double, 1> iStdDev,
               NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormEstimates(actsIn, eps, unbiasedVarEstimate, mean, iStdDev);
  } else if (isOfGroupNormType(normType)) {
    groupNormEstimates(actsIn, eps, unbiasedVarEstimate, mean, iStdDev);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::
normalise(const boost::multi_array_ref<double, 4> actsIn,
          const boost::multi_array_ref<double, 1> gamma,
          const boost::multi_array_ref<double, 1> beta,
          const boost::multi_array_ref<double, 1> mean,
          const boost::multi_array_ref<double, 1> iStdDev,
          boost::multi_array_ref<double, 4> actsOut,
          boost::multi_array_ref<double, 4> actsWhitened,
          NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormalise(actsIn, gamma, beta, mean, iStdDev, actsOut, actsWhitened);
  } else if (isOfGroupNormType(normType))  {
    groupNormalise(actsIn, gamma, beta, mean, iStdDev, actsOut, actsWhitened);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::
normGradients(const boost::multi_array_ref<double, 4> actsWhitened,
              const boost::multi_array_ref<double, 4> gradsIn,
              const boost::multi_array_ref<double, 1> iStdDev,
              const boost::multi_array_ref<double, 1> gamma,
              boost::multi_array_ref<double, 4> gradsOut,
              NormType normType) {
  if (normType == NormType::BatchNorm) {
    batchNormGradients(actsWhitened, gradsIn, iStdDev, gamma, gradsOut);
  } else if (isOfGroupNormType(normType)) {
    groupNormGradients(actsWhitened, gradsIn, iStdDev, gamma, gradsOut);
  } else {
    throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
}

void poplibs_test::norm::
normParamUpdate(const boost::multi_array_ref<double, 4> actsWhitened,
                const boost::multi_array_ref<double, 4> gradsIn,
                double learningRate,
                boost::multi_array_ref<double, 1> gamma,
                boost::multi_array_ref<double, 1> beta,
                NormType normType) {
  if (!isSupportedNormType(normType)) {
     throw poplibs_test::poplibs_test_error("Normalisation type not supported");
  }
  paramUpdate(actsWhitened, gradsIn, learningRate, gamma, beta);
}
