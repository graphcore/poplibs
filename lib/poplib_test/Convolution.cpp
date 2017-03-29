#include <poplib_test/Convolution.hpp>
#include <poplib_test/exceptions.hpp>

static unsigned absdiff(unsigned a, unsigned b) {
  return a < b ? b - a : a - b;
}

void poplib_test::conv::
convolution(unsigned strideH, unsigned strideW,
            unsigned paddingHeight, unsigned paddingWidth,
            const boost::multi_array<double, 4> &in,
            const boost::multi_array<double, 4> &weights,
            const boost::multi_array<double, 1> &biases,
            boost::multi_array<double, 4> &out) {
  // Pad input.
  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[1];
  const auto inputHeight = in.shape()[2];
  const auto inputWidth = in.shape()[3];
  const auto paddedHeight = inputHeight + 2 * paddingHeight;
  const auto paddedWidth = inputWidth + 2 * paddingWidth;

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedIn(boost::extents[inputChannels][paddedHeight][paddedWidth]);
    std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(), 0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          paddedIn[c][y + paddingHeight][x + paddingWidth] = in[b][c][y][x];
        }
      }
    }

    // Perform convolution.
    const auto outputChannels = out.shape()[1];
    const auto kernelHeight = weights.shape()[2];
    const auto kernelWidth = weights.shape()[3];
    const auto convOutHeight = absdiff(paddedHeight, kernelHeight) + 1;
    const auto convOutWidth = absdiff(paddedWidth, kernelWidth) + 1;
    boost::multi_array<double, 3>
        convOut(boost::extents[outputChannels]
                              [convOutHeight]
                              [convOutWidth]);
    std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutHeight; ++y) {
        for (unsigned x = 0; x != convOutWidth; ++x) {
          for (unsigned ky = 0;
               ky != std::min(kernelHeight, paddedHeight); ++ky) {
            for (unsigned kx = 0;
                 kx != std::min(kernelWidth, paddedWidth); ++kx) {
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                convOut[oc][y][x] +=
                    weights[oc][ic]
                           [ky + (kernelHeight < paddedHeight ? 0 : y)]
                           [kx + (kernelWidth  < paddedWidth  ? 0 : x)] *
                    paddedIn[ic]
                            [ky + (kernelHeight < paddedHeight ? y : 0)]
                            [kx + (kernelWidth  < paddedWidth  ? x : 0)];
              }
            }
          }
          convOut[oc][y][x] += biases[oc];
        }
      }
    }

    // Downsample.
    const auto outHeight = (convOutHeight + strideH - 1) / strideH;
    const auto outWidth = (convOutWidth + strideW - 1) / strideW;
    if (outHeight != out.shape()[2] ||
        outWidth != out.shape()[3]) {
      throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != outHeight; ++y) {
        for (unsigned x = 0; x != outWidth; ++x) {
          out[b][oc][y][x] = convOut[oc][y * strideH][x * strideW];
        }
      }
    }
  }
}

void poplib_test::conv::
convolutionBackward(unsigned strideH, unsigned strideW,
                    unsigned paddingHeight, unsigned paddingWidth,
                    const boost::multi_array<double, 4> &in,
                    const boost::multi_array<double, 4> &weights,
                    boost::multi_array<double, 4> &out) {
  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[1];
  const auto inputHeight = in.shape()[2];
  const auto inputWidth = in.shape()[3];
  const auto outputHeight = out.shape()[2];
  const auto outputWidth = out.shape()[3];
  const auto kernelHeight = weights.shape()[2];
  const auto kernelWidth = weights.shape()[3];

  for (unsigned b = 0; b != batchSize; ++b) {
    // Upsample.
    const auto upsampledHeight =
        outputHeight + 2 * paddingHeight - (kernelHeight - 1) ;
    const auto upsampledWidth = outputWidth + 2 * paddingWidth - (kernelWidth - 1);
    if ((upsampledHeight + strideH - 1)/ strideH != inputHeight ||
        (upsampledWidth + strideW - 1)/ strideW != inputWidth) {
      throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                       "do not match");
    }
    boost::multi_array<double, 3>
        upsampledIn(boost::extents[inputChannels][upsampledHeight]
                                  [upsampledWidth]);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != upsampledHeight; ++y) {
        for (unsigned x = 0; x != upsampledWidth; ++x) {
          if (y % strideH == 0 &&
              x % strideW == 0) {
            upsampledIn[c][y][x] = in[b][c][y / strideH][x / strideW];
          } else {
            upsampledIn[c][y][x] = 0;
          }
        }
      }
    }

    // Perform a full convolution with flipped weights.
    const auto outputChannels = out.shape()[1];
    const auto convOutHeight = upsampledHeight + kernelHeight - 1;
    const auto convOutWidth = upsampledWidth + kernelWidth - 1;
    boost::multi_array<double, 3>
        convOut(boost::extents[outputChannels]
                             [convOutHeight]
                             [convOutWidth]);
    std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutHeight; ++y) {
        for (unsigned x = 0; x != convOutWidth; ++x) {
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            const auto kyFlipped = kernelHeight - 1 - ky;
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              const auto kxFlipped = kernelWidth - 1 - kx;
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                if (y + ky >= (kernelHeight - 1) &&
                    y + ky < (kernelHeight - 1) + upsampledHeight &&
                    x + kx >= (kernelWidth - 1) &&
                    x + kx < (kernelWidth - 1) + upsampledWidth) {
                  convOut[oc][y][x] += weights[ic][oc]
                                             [kyFlipped]
                                             [kxFlipped] *
                                      upsampledIn[ic]
                                                 [y - (kernelHeight - 1) + ky]
                                                 [x - (kernelWidth - 1) + kx];
                }
              }
            }
          }
        }
      }
    }

    // Truncate.
    for (unsigned c = 0; c != outputChannels; ++c) {
      for (unsigned y = 0; y != outputHeight; ++y) {
        for (unsigned x = 0; x != outputWidth; ++x) {
          out[b][c][y][x] = convOut[c][y + paddingHeight][x + paddingWidth];
        }
      }
    }
  }
}

void poplib_test::conv::
weightUpdate(unsigned strideH, unsigned strideW,
             unsigned paddingHeight, unsigned paddingWidth,
             double learningRate,
             const boost::multi_array<double, 4> &activations,
             const boost::multi_array<double, 4> &deltas,
             boost::multi_array<double, 4> &weights,
             boost::multi_array<double, 1> &biases) {
  // Pad activations.
  const auto batchSize = activations.shape()[0];
  const auto inputChannels = activations.shape()[1];
  const auto inputHeight = activations.shape()[2];
  const auto inputWidth = activations.shape()[3];
  const auto paddedHeight = inputHeight + 2 * paddingHeight;
  const auto paddedWidth = inputWidth + 2 * paddingWidth;

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedActivations(boost::extents[inputChannels][paddedHeight]
                                        [paddedWidth]);
    std::fill(paddedActivations.data(),
              paddedActivations.data() + paddedActivations.num_elements(),
              0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          paddedActivations[c][y + paddingHeight][x + paddingWidth] =
              activations[b][c][y][x];
        }
      }
    }

    // Upsample deltas.
    const auto outputChannels = deltas.shape()[1];
    const auto outputHeight = deltas.shape()[2];
    const auto outputWidth = deltas.shape()[3];
    const auto kernelHeight = weights.shape()[2];
    const auto kernelWidth = weights.shape()[3];
    const auto upsampledDeltasHeight =
        inputHeight + 2 * paddingHeight - (kernelHeight - 1);
    const auto upsampledDeltasWidth =
        inputWidth + 2 * paddingWidth - (kernelWidth - 1);
    if ((upsampledDeltasHeight + strideH - 1) / strideH != outputHeight ||
        (upsampledDeltasWidth + strideW - 1) / strideW != outputWidth) {
      throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                       "do not match");
    }
    boost::multi_array<double, 3>
        upsampledDeltas(boost::extents[outputChannels][upsampledDeltasHeight]
                                  [upsampledDeltasWidth]);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
        for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
          if (y % strideH == 0 &&
              x % strideW == 0) {
            upsampledDeltas[oc][y][x] = deltas[b][oc][y / strideH]
                                                 [x / strideW];
          } else {
            upsampledDeltas[oc][y][x] = 0;
          }
        }
      }
    }

    // Compute the weight deltas.
    boost::multi_array<double, 4>
        weightDeltas(boost::extents[outputChannels][inputChannels]
                                   [kernelHeight][kernelWidth]);
    std::fill(weightDeltas.data(),
              weightDeltas.data() + weightDeltas.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned ic = 0; ic != inputChannels; ++ic) {
        for (unsigned ky = 0; ky != kernelHeight; ++ky) {
          for (unsigned kx = 0; kx != kernelWidth; ++kx) {
            for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
              for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
                weightDeltas[oc][ic][ky][kx] +=
                    paddedActivations[ic][y + ky][x + kx] *
                    upsampledDeltas[oc][y][x];
              }
            }
          }
        }
      }
    }

    // Add the weight deltas.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned ic = 0; ic != inputChannels; ++ic) {
        for (unsigned ky = 0; ky != kernelHeight; ++ky) {
          for (unsigned kx = 0; kx != kernelWidth; ++kx) {
            weights[oc][ic][ky][kx] +=
                learningRate * -weightDeltas[oc][ic][ky][kx];
          }
        }
      }
    }

    // Compute the bias deltas.
    boost::multi_array<double, 1> biasDeltas(boost::extents[outputChannels]);
    std::fill(biasDeltas.data(),
              biasDeltas.data() + biasDeltas.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
        for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
          biasDeltas[oc] += upsampledDeltas[oc][y][x];
        }
      }
    }

    // Add the bias deltas.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      biases[oc] += learningRate * -biasDeltas[oc];
    }
  }
}
