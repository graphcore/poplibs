#include <poplib_test/Convolution.hpp>
#include <poplib_test/exceptions.hpp>

static unsigned absdiff(unsigned a, unsigned b) {
  return a < b ? b - a : a - b;
}

static boost::multi_array<double, 4>
dilateAndPadActivations(const boost::multi_array<double, 4> &in,
                        const std::vector<unsigned> inputDilation,
                        const std::vector<int> &paddingLower,
                        const std::vector<int> &paddingUpper) {
  // Pad input.
  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[3];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  std::vector<unsigned> dilatedShape(2);
  std::vector<unsigned> paddedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    dilatedShape[dim] = (in.shape()[dim + 1] - 1) * inputDilation[dim] + 1;
    paddedShape[dim] = paddingLower[dim] + dilatedShape[dim] +
                       paddingUpper[dim];
  }

  boost::multi_array<double, 4>
      dilatedIn(boost::extents[batchSize][dilatedShape[0]][dilatedShape[1]]
                              [inputChannels]);
  std::fill(dilatedIn.data(), dilatedIn.data() + dilatedIn.num_elements(),
            0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          dilatedIn[b][y * inputDilation[0]][x * inputDilation[1]][c] =
               in[b][y][x][c];
        }
      }
    }
  }
  boost::multi_array<double, 4>
      paddedIn(boost::extents[batchSize][paddedShape[0]][paddedShape[1]]
                             [inputChannels]);
  std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(), 0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != dilatedShape[0]; ++y) {
        auto paddedY = static_cast<int>(y) + paddingLower[0];
        if (paddedY < 0 || static_cast<unsigned>(paddedY) >= paddedShape[0])
          continue;
        for (unsigned x = 0; x != dilatedShape[1]; ++x) {
          auto paddedX = static_cast<int>(x) + paddingLower[1];
          if (paddedX < 0 || static_cast<unsigned>(paddedX) >= paddedShape[1])
            continue;
          paddedIn[b][paddedY][paddedX][c] = dilatedIn[b][y][x][c];
        }
      }
    }
  }
  return paddedIn;
}

static boost::multi_array<double, 4>
dilateAndPadActivationsInverse(const boost::multi_array<double, 4> &paddedActs,
                               const std::vector<unsigned> dilation,
                               const std::vector<int> &paddingLower,
                               const std::vector<int> &paddingUpper) {
  // Pad input.
  const auto batchSize = paddedActs.shape()[0];
  const auto channels = paddedActs.shape()[3];

  std::vector<unsigned> paddedShape(2);
  std::vector<unsigned> dilatedShape(2);
  std::vector<unsigned> inShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedShape[dim] = paddedActs.shape()[dim + 1];
    dilatedShape[dim] = paddedShape[dim] - paddingLower[dim] -
                        paddingUpper[dim];
    inShape[dim] = (dilatedShape[dim] + dilation[dim] - 1) /
                   dilation[dim];
  }

  // Truncate.
  boost::multi_array<double, 4>
      dilatedActs(boost::extents[batchSize][dilatedShape[0]][dilatedShape[1]]
                                [channels]);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != dilatedShape[0]; ++y) {
        auto paddedY = static_cast<int>(y) + paddingLower[0];
        if (paddedY < 0 || static_cast<unsigned>(paddedY) >= paddedShape[0])
          continue;
        for (unsigned x = 0; x != dilatedShape[1]; ++x) {
          auto paddedX = static_cast<int>(x) + paddingLower[1];
          if (paddedX < 0 || static_cast<unsigned>(paddedX) >= paddedShape[1])
            continue;
          dilatedActs[b][y][x][c] = paddedActs[b][paddedY][paddedX][c];
        }
      }
    }
  }

  // Downsample.
  boost::multi_array<double, 4>
      acts(boost::extents[batchSize][inShape[0]][inShape[1]][channels]);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != inShape[0]; ++y) {
        for (unsigned x = 0; x != inShape[1]; ++x) {
          acts[b][y][x][c] = dilatedActs[b][y * dilation[0]]
                                        [x * dilation[1]][c];
        }
      }
    }
  }
  return acts;
}

static boost::multi_array<double, 4>
dilateAndPadKernel(const boost::multi_array<double, 4> &kernel,
                   const std::vector<unsigned> kernelDilation,
                   const std::vector<int> &kernelPaddingLower,
                   const std::vector<int> &kernelPaddingUpper) {
  const auto outputChannels = kernel.shape()[2];
  const auto inputChannels = kernel.shape()[3];
  std::vector<unsigned> kernelShape(2);
  std::vector<unsigned> dilatedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    kernelShape[dim] = kernel.shape()[dim];
    dilatedShape[dim] = (kernelShape[dim] - 1) * kernelDilation[dim] + 1;
  }
  boost::multi_array<double, 4>
      dilated(boost::extents[dilatedShape[0]][dilatedShape[1]]
                            [outputChannels][inputChannels]);
  std::fill(dilated.data(), dilated.data() + dilated.num_elements(), 0.0);
  for (unsigned y = 0; y != kernelShape[0]; ++y) {
    for (unsigned x = 0; x != kernelShape[1]; ++x) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          dilated[y * kernelDilation[0]][x * kernelDilation[1]][oc][ic] =
              kernel[y][x][oc][ic];
        }
      }
    }
  }
  std::vector<unsigned> paddedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedShape[dim] = dilatedShape[dim] + kernelPaddingLower[dim] +
                       kernelPaddingUpper[dim];
  }
  boost::multi_array<double, 4>
      padded(boost::extents[paddedShape[0]][paddedShape[1]]
                           [outputChannels][inputChannels]);
  std::fill(padded.data(), padded.data() + padded.num_elements(), 0.0);
  for (unsigned y = 0; y != dilatedShape[0]; ++y) {
    auto paddedY = static_cast<int>(y) + kernelPaddingLower[0];
    if (paddedY < 0 || static_cast<unsigned>(paddedY) >= paddedShape[0])
      continue;
    for (unsigned x = 0; x != dilatedShape[1]; ++x) {
      auto paddedX = static_cast<int>(x) + kernelPaddingLower[1];
      if (paddedX < 0 || static_cast<unsigned>(paddedX) >= paddedShape[1])
        continue;
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          padded[paddedY][paddedX][oc][ic] = dilated[y][x][oc][ic];
        }
      }
    }
  }
  return padded;
}

static boost::multi_array<double, 4>
dilateAndPadKernelInverse(const boost::multi_array<double, 4> &padded,
                          const std::vector<unsigned> kernelDilation,
                          const std::vector<int> &kernelPaddingLower,
                          const std::vector<int> &kernelPaddingUpper) {
  const auto outputChannels = padded.shape()[2];
  const auto inputChannels = padded.shape()[3];
  std::vector<unsigned> paddedShape(2);
  std::vector<unsigned> dilatedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedShape[dim] = padded.shape()[dim];
    dilatedShape[dim] = paddedShape[dim] - kernelPaddingLower[dim] -
                        kernelPaddingUpper[dim];
  }
  boost::multi_array<double, 4>
      dilated(boost::extents[dilatedShape[0]][dilatedShape[1]]
                            [outputChannels][inputChannels]);
  for (unsigned y = 0; y != dilatedShape[0]; ++y) {
    auto paddedY = static_cast<int>(y) + kernelPaddingLower[0];
    if (paddedY < 0 || static_cast<unsigned>(paddedY) >= paddedShape[0])
      continue;
    for (unsigned x = 0; x != dilatedShape[1]; ++x) {
      auto paddedX = static_cast<int>(x) + kernelPaddingLower[1];
      if (paddedX < 0 || static_cast<unsigned>(paddedX) >= paddedShape[1])
        continue;
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          dilated[y][x][oc][ic] =
              padded[paddedY][paddedX][oc][ic];
        }
      }
    }
  }
  std::vector<unsigned> kernelShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    kernelShape[dim] = (dilatedShape[dim] - 1) / kernelDilation[dim] + 1;
  }
  boost::multi_array<double, 4>
      kernel(boost::extents[kernelShape[0]][kernelShape[1]]
                           [outputChannels][inputChannels]);
  for (unsigned y = 0; y != kernelShape[0]; ++y) {
    for (unsigned x = 0; x != kernelShape[1]; ++x) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          kernel[y][x][oc][ic] =
              dilated[y * kernelDilation[0]][x * kernelDilation[1]][oc][ic];
        }
      }
    }
  }
  return kernel;
}

void poplib_test::conv::
convolution(const std::vector<unsigned> &stride,
            const std::vector<unsigned> &inputDilation,
            const std::vector<int> &paddingLower,
            const std::vector<int> &paddingUpper,
            const std::vector<unsigned> kernelDilation,
            const std::vector<int> &kernelPaddingLower,
            const std::vector<int> &kernelPaddingUpper,
            const boost::multi_array<double, 4> &in,
            const boost::multi_array<double, 4> &kernel,
            const boost::multi_array<double, 1> &biases,
            boost::multi_array<double, 4> &out) {
  if (paddingLower.size() != paddingUpper.size() ||
      paddingLower.size() != stride.size() ||
      paddingLower.size() != inputDilation.size()) {
    throw poplib_test::poplib_test_error("Padding, dilation and stride vectors "
                                         "must be equal sizes.");
  }
  if (stride.size() != 2) {
    throw poplib_test::poplib_test_error("Convolution of >2 spatial dimensions "
                                         "not supported.");
  }

  auto paddedKernel = dilateAndPadKernel(kernel, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper);
  auto paddedIn = dilateAndPadActivations(in, inputDilation, paddingLower,
                                          paddingUpper);

  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[3];
  std::vector<unsigned> paddedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedShape[dim] = paddedIn.shape()[dim + 1];
  }
  const auto outputChannels = kernel.shape()[2];
  std::vector<unsigned> paddedKernelShape(2);
  std::vector<unsigned> convOutShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedKernelShape[dim] = paddedKernel.shape()[dim];
    convOutShape[dim] = absdiff(paddedShape[dim],
                                paddedKernelShape[dim]) + 1;
  }
  boost::multi_array<double, 4>
      convOut(boost::extents[batchSize]
                            [convOutShape[0]]
                            [convOutShape[1]]
                            [outputChannels]);
  std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    // Perform convolution.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutShape[0]; ++y) {
        for (unsigned x = 0; x != convOutShape[1]; ++x) {
          for (unsigned ky = 0;
               ky != std::min(paddedKernelShape[0], paddedShape[0]); ++ky) {
            for (unsigned kx = 0;
                 kx != std::min(paddedKernelShape[1], paddedShape[1]); ++kx) {
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                convOut[b][y][x][oc] +=
                    paddedKernel[ky +
                                 (paddedKernelShape[0] < paddedShape[0] ? 0 :
                                                                          y)]
                                [kx +
                                 (paddedKernelShape[1] < paddedShape[1] ? 0 :
                                                                          x)]
                                [oc][ic] *
                    paddedIn[b]
                            [ky +
                             (paddedKernelShape[0] < paddedShape[0] ? y : 0)]
                            [kx +
                             (paddedKernelShape[1] < paddedShape[1]  ? x : 0)]
                            [ic];
              }
            }
          }
          convOut[b][y][x][oc] += biases[oc];
        }
      }
    }
  }

  out = dilateAndPadActivationsInverse(convOut, stride, {0, 0}, {0, 0});
}

void poplib_test::conv::
convolutionBackward(const std::vector<unsigned> &stride,
                    const std::vector<unsigned> &inputDilation,
                    const std::vector<int> &paddingLower,
                    const std::vector<int> &paddingUpper,
                    const std::vector<unsigned> kernelDilation,
                    const std::vector<int> &kernelPaddingLower,
                    const std::vector<int> &kernelPaddingUpper,
                    const boost::multi_array<double, 4> &in,
                    const boost::multi_array<double, 4> &kernel,
                    boost::multi_array<double, 4> &out) {
  if (paddingLower.size() != paddingUpper.size() ||
      paddingLower.size() != stride.size() ||
      paddingLower.size() != inputDilation.size()) {
    throw poplib_test::poplib_test_error("Padding, dilation and stride vectors "
                                         "must be equal sizes.");
  }
  if (stride.size() != 2) {
    throw poplib_test::poplib_test_error("Convolution of >2 spatial dimensions "
                                         "not supported.");
  }

  auto paddedKernel = dilateAndPadKernel(kernel, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper);

  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[3];

  // Pad input.
  std::vector<unsigned> convOutShape(2);
  std::vector<unsigned> paddedKernelShape(2);
  std::vector<unsigned> paddedInShape(2);
  std::vector<int> inPaddingUpper(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    convOutShape[dim] =
        (out.shape()[dim + 1] - 1) * inputDilation[dim] + 1 +
        paddingLower[dim] + paddingUpper[dim];
    paddedKernelShape[dim] = paddedKernel.shape()[dim];
    paddedInShape[dim] =
        absdiff(convOutShape[dim], paddedKernelShape[dim]) + 1;
    if ((paddedInShape[dim] + stride[dim] - 1)/ stride[dim] !=
        in.shape()[dim + 1]) {
      throw poplib_test::poplib_test_error("Output and input tensor "
                                           "dimensions do not match");
    }
    inPaddingUpper[dim] = paddedInShape[dim] -
                          ((in.shape()[dim + 1] - 1) * stride[dim] + 1);
    assert(inPaddingUpper[dim] >= 0 &&
           static_cast<unsigned>(inPaddingUpper[dim]) < stride[dim]);
  }
  auto paddedIn = dilateAndPadActivations(in, stride, {0, 0}, inPaddingUpper);

  const auto outputChannels = out.shape()[3];
  boost::multi_array<double, 4>
      convOut(boost::extents[batchSize]
                            [convOutShape[0]]
                            [convOutShape[1]]
                            [outputChannels]);
  std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    // Perform a full convolution with flipped weights.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutShape[0]; ++y) {
        for (unsigned x = 0; x != convOutShape[1]; ++x) {
          for (unsigned ky = 0; ky != paddedKernelShape[0]; ++ky) {
            const auto kyFlipped = paddedKernelShape[0] - 1 - ky;
            for (unsigned kx = 0; kx != paddedKernelShape[1]; ++kx) {
              const auto kxFlipped = paddedKernelShape[1] - 1 - kx;
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                if (y + ky >= (paddedKernelShape[0] - 1) &&
                    y + ky < (paddedKernelShape[0] - 1) + paddedInShape[0] &&
                    x + kx >= (paddedKernelShape[1] - 1) &&
                    x + kx < (paddedKernelShape[1] - 1) + paddedInShape[1]) {
                  convOut[b][y][x][oc] +=
                      paddedKernel[kyFlipped][kxFlipped][ic][oc] *
                      paddedIn[b]
                              [y - (paddedKernelShape[0] - 1) + ky]
                              [x - (paddedKernelShape[1] - 1) + kx][ic];
                }
              }
            }
          }
        }
      }
    }
  }

  out = dilateAndPadActivationsInverse(convOut, inputDilation, paddingLower,
                                       paddingUpper);
}

void poplib_test::conv::
weightUpdate(const std::vector<unsigned> &stride,
             const std::vector<unsigned> &inputDilation,
             const std::vector<int> &paddingLower,
             const std::vector<int> &paddingUpper,
             const std::vector<unsigned> kernelDilation,
             const std::vector<int> &kernelPaddingLower,
             const std::vector<int> &kernelPaddingUpper,
             double learningRate,
             const boost::multi_array<double, 4> &activations,
             const boost::multi_array<double, 4> &deltas,
             boost::multi_array<double, 4> &kernel,
             boost::multi_array<double, 1> &biases) {
  if (paddingLower.size() != paddingUpper.size() ||
      paddingLower.size() != stride.size() ||
      paddingLower.size() != inputDilation.size()) {
    throw poplib_test::poplib_test_error("Padding, dilation and stride vectors "
                                         "must be equal sizes.");
  }
  if (inputDilation[0] != 1 || inputDilation[1] != 1) {
   throw poplib_test::poplib_test_error("Weight update or input dilation != 1 "
                                        "not implemented.");
  }
  if (stride.size() != 2) {
    throw poplib_test::poplib_test_error("Convolution of >2 spatial dimensions "
                                         "not supported.");
  }

  auto paddedKernel = dilateAndPadKernel(kernel, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper);

  // Pad activations.
  auto paddedActivations =
      dilateAndPadActivations(activations, inputDilation, paddingLower,
                              paddingUpper);
  std::vector<unsigned> paddedDeltasShape(2);
  std::vector<unsigned> paddedActivationsShape(2);
  std::vector<unsigned> paddedKernelShape(2);
  std::vector<int> deltasPaddingUpper(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedKernelShape[dim] = paddedKernel.shape()[dim];
    paddedActivationsShape[dim] = paddedActivations.shape()[dim + 1];
    paddedDeltasShape[dim] = absdiff(paddedActivationsShape[dim],
                                     paddedKernelShape[dim]) + 1;
    if ((paddedDeltasShape[dim] + stride[dim] - 1) / stride[dim] !=
        deltas.shape()[dim + 1]) {
      throw poplib_test::poplib_test_error("Output and input tensor "
                                           "dimensions do not match");
    }
    deltasPaddingUpper[dim] = paddedDeltasShape[dim] -
                              ((deltas.shape()[dim + 1] - 1) * stride[dim] + 1);
    assert(deltasPaddingUpper[dim] >= 0 &&
           static_cast<unsigned>(deltasPaddingUpper[dim]) < stride[dim]);
  }
  // Pad deltas.
  auto paddedDeltas =
      dilateAndPadActivations(deltas, stride, {0, 0}, deltasPaddingUpper);
  const auto batchSize = paddedActivations.shape()[0];
  const auto inputChannels = paddedActivations.shape()[3];
  const auto outputChannels = paddedDeltas.shape()[3];

  boost::multi_array<double, 4>
      paddedWeightDeltas(boost::extents[paddedKernelShape[0]]
                                       [paddedKernelShape[1]]
                                       [outputChannels]
                                       [inputChannels]);
  std::fill(paddedWeightDeltas.data(),
            paddedWeightDeltas.data() + paddedWeightDeltas.num_elements(), 0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    // Compute the weight deltas.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned ic = 0; ic != inputChannels; ++ic) {
        for (unsigned ky = 0; ky != paddedKernelShape[0]; ++ky) {
          for (unsigned kx = 0; kx != paddedKernelShape[1]; ++kx) {
            for (unsigned y = 0; y != paddedDeltasShape[0]; ++y) {
              for (unsigned x = 0; x != paddedDeltasShape[1]; ++x) {
                paddedWeightDeltas[ky][kx][oc][ic] +=
                    paddedActivations[b][y + ky][x + kx][ic] *
                    paddedDeltas[b][y][x][oc];
              }
            }
          }
        }
      }
    }
  }
  auto weightDeltas =
      dilateAndPadKernelInverse(paddedWeightDeltas, kernelDilation,
                                kernelPaddingLower,
                                kernelPaddingUpper);

  // Add the weight deltas.
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    for (unsigned ic = 0; ic != inputChannels; ++ic) {
      for (unsigned ky = 0; ky != kernel.shape()[0]; ++ky) {
        for (unsigned kx = 0; kx != kernel.shape()[1]; ++kx) {
          kernel[ky][kx][oc][ic] +=
              learningRate * -weightDeltas[ky][kx][oc][ic];
        }
      }
    }
  }

  boost::multi_array<double, 1> biasDeltas(boost::extents[outputChannels]);
  std::fill(biasDeltas.data(),
            biasDeltas.data() + biasDeltas.num_elements(), 0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    // Compute the bias deltas.
    for (unsigned y = 0; y != paddedDeltasShape[0]; ++y) {
      for (unsigned x = 0; x != paddedDeltasShape[1]; ++x) {
        for (unsigned oc = 0; oc != outputChannels; ++oc) {
          biasDeltas[oc] += paddedDeltas[b][y][x][oc];
        }
      }
    }
  }

  // Add the bias deltas.
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    biases[oc] += learningRate * -biasDeltas[oc];
  }
}

void poplib_test::conv::
batchNormEstimates(const boost::multi_array_ref<double, 4> actsIn,
                   double eps,
                   boost::multi_array_ref<double, 1> mean,
                   boost::multi_array_ref<double, 1> stdDev) {
  const unsigned batchSize= actsIn.shape()[0];
  const unsigned dimY = actsIn.shape()[1];
  const unsigned dimX = actsIn.shape()[2];
  const unsigned numChannels = actsIn.shape()[3];
  const auto numElems = batchSize * dimX * dimY;

  assert(stdDev.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double sum =  0;
    double sumSquares = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != dimY; ++h) {
        for (unsigned w = 0; w != dimX; ++w) {
          sum += actsIn[b][h][w][c];
          sumSquares += actsIn[b][h][w][c] * actsIn[b][h][w][c];
        }
      }
    }

    // unbiased sample mean
    mean[c] = sum / numElems;
    stdDev[c] = std::sqrt(sumSquares / numElems - mean[c] * mean[c] + eps);
  }
}

void poplib_test::conv::
batchNormalise(const boost::multi_array_ref<double, 4> acts,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> stdDev,
               boost::multi_array_ref<double, 4> actsOut,
               boost::multi_array_ref<double, 4> actsWhitened) {

  const unsigned batchSize = acts.shape()[0];
  const unsigned dimY = acts.shape()[1];
  const unsigned dimX = acts.shape()[2];
  const unsigned numChannels = acts.shape()[3];

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);
  assert(stdDev.shape()[0] == numChannels);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == dimY);
  assert(actsOut.shape()[2] == dimX);
  assert(actsOut.shape()[3] == numChannels);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == dimY);
  assert(actsWhitened.shape()[2] == dimX);
  assert(actsWhitened.shape()[3] == numChannels);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned h = 0; h != dimY; ++h) {
      for (unsigned w = 0; w != dimX; ++w) {
        for (unsigned c = 0; c != numChannels; ++c) {
          actsWhitened[b][h][w][c] = (acts[b][h][w][c] - mean[c]) / stdDev[c];
          actsOut[b][h][w][c] = gamma[c] * actsWhitened[b][h][w][c] + beta[c];
        }
      }
    }
  }
}

void poplib_test::conv::
batchNormGradients(const boost::multi_array_ref<double, 4> actsWhitened,
                   const boost::multi_array_ref<double, 4> gradsIn,
                   const boost::multi_array_ref<double, 1> stdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 4> gradsOut) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned height = actsWhitened.shape()[1];
  const unsigned width = actsWhitened.shape()[2];
  const unsigned numChannels = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == height);
  assert(gradsIn.shape()[2] == width);
  assert(gradsIn.shape()[3] == numChannels);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == height);
  assert(gradsOut.shape()[2] == width);
  assert(gradsOut.shape()[3] == numChannels);

  assert(stdDev.shape()[0] == numChannels);
  assert(gamma.shape()[0] == numChannels);

  const auto numElements = batchSize * height * width;

  for (unsigned c = 0; c != numChannels; ++c) {
    double sumGradsIn = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          sumGradsIn += gradsIn[b][h][w][c];
        }
      }
    }
    double sumGradsInAndxMu = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          sumGradsInAndxMu += actsWhitened[b][h][w][c] * gradsIn[b][h][w][c];
        }
      }
    }

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          double out =
            gradsIn[b][h][w][c]
            - actsWhitened[b][h][w][c] * sumGradsInAndxMu / numElements
            - sumGradsIn / numElements;

          gradsOut[b][h][w][c] = out * gamma[c] / stdDev[c];
        }
      }
    }
  }
}

void poplib_test::conv::
batchNormParamUpdate(const boost::multi_array_ref<double, 4> actsWhitened,
                     const boost::multi_array_ref<double, 4> gradsIn,
                     double learningRate,
                     boost::multi_array_ref<double, 1> gamma,
                     boost::multi_array_ref<double, 1> beta) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned height = actsWhitened.shape()[1];
  const unsigned width = actsWhitened.shape()[2];
  const unsigned numChannels = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == height);
  assert(gradsIn.shape()[2] == width);
  assert(gradsIn.shape()[3] == numChannels);

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double dBeta = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          dBeta += gradsIn[b][h][w][c];
        }
      }
    }
    beta[c] -= learningRate * dBeta;

    double dGamma = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          dGamma += actsWhitened[b][h][w][c] * gradsIn[b][h][w][c];
        }
      }
    }
    gamma[c] -= learningRate * dGamma;
  }
}
