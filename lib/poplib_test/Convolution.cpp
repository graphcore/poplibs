#include <poplib_test/Convolution.hpp>
#include <poplib_test/exceptions.hpp>

static unsigned absdiff(unsigned a, unsigned b) {
  return a < b ? b - a : a - b;
}

static boost::multi_array<double, 4>
dilateAndPadKernel(const boost::multi_array<double, 4> &kernel,
                   const std::vector<unsigned> kernelDilation,
                   const std::vector<int> &kernelPaddingLower,
                   const std::vector<int> &kernelPaddingUpper) {
  const auto kernelHeight = kernel.shape()[0];
  const auto kernelWidth = kernel.shape()[1];
  const auto outputChannels = kernel.shape()[2];
  const auto inputChannels = kernel.shape()[3];
  const auto dilatedHeight = (kernelHeight - 1) * kernelDilation[0] + 1;
  const auto dilatedWidth = (kernelWidth - 1) * kernelDilation[1] + 1;
  boost::multi_array<double, 4>
      dilated(boost::extents[dilatedHeight][dilatedWidth]
                            [outputChannels][inputChannels]);
  std::fill(dilated.data(), dilated.data() + dilated.num_elements(), 0.0);
  for (unsigned y = 0; y != kernelHeight; ++y) {
    for (unsigned x = 0; x != kernelWidth; ++x) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          dilated[y * kernelDilation[0]][x * kernelDilation[1]][oc][ic] =
              kernel[y][x][oc][ic];
        }
      }
    }
  }
  const auto paddedHeight = dilatedHeight + kernelPaddingLower[0] +
                            kernelPaddingUpper[0];
  const auto paddedWidth = dilatedWidth + kernelPaddingLower[1] +
                           kernelPaddingUpper[1];
  boost::multi_array<double, 4>
      padded(boost::extents[paddedHeight][paddedWidth]
                           [outputChannels][inputChannels]);
  std::fill(padded.data(), padded.data() + padded.num_elements(), 0.0);
  for (unsigned y = 0; y != dilatedHeight; ++y) {
    for (unsigned x = 0; x != dilatedWidth; ++x) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          padded[y + kernelPaddingLower[0]][x + kernelPaddingLower[1]][oc][ic] =
              dilated[y][x][oc][ic];
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
  const auto paddedHeight = padded.shape()[0];
  const auto paddedWidth = padded.shape()[1];
  const auto outputChannels = padded.shape()[2];
  const auto inputChannels = padded.shape()[3];
  const auto dilatedHeight = paddedHeight - kernelPaddingLower[0] -
                             kernelPaddingUpper[0];
  const auto dilatedWidth = paddedWidth - kernelPaddingLower[1] -
                            kernelPaddingUpper[1];
  boost::multi_array<double, 4>
      dilated(boost::extents[dilatedHeight][dilatedWidth]
                            [outputChannels][inputChannels]);
  for (unsigned y = 0; y != dilatedHeight; ++y) {
    for (unsigned x = 0; x != dilatedWidth; ++x) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          dilated[y][x][oc][ic] =
              padded[y + kernelPaddingLower[0]][x + kernelPaddingLower[1]]
                    [oc][ic];
        }
      }
    }
  }
  const auto kernelHeight = (dilatedHeight - 1) / kernelDilation[0] + 1;
  const auto kernelWidth = (dilatedWidth - 1) / kernelDilation[1] + 1;
  boost::multi_array<double, 4>
      kernel(boost::extents[kernelHeight][kernelWidth]
                           [outputChannels][inputChannels]);
  for (unsigned y = 0; y != kernelHeight; ++y) {
    for (unsigned x = 0; x != kernelWidth; ++x) {
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

  unsigned strideH = stride[0];
  unsigned strideW = stride[1];
  unsigned paddingHeightL = paddingLower[0];
  unsigned paddingWidthL = paddingLower[1];
  unsigned paddingHeightU = paddingUpper[0];
  unsigned paddingWidthU = paddingUpper[1];

  auto paddedKernel = dilateAndPadKernel(kernel, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper);

  // Pad input.
  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[3];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto dilatedHeight = (inputHeight - 1) * inputDilation[0] + 1;
  const auto paddedHeight = dilatedHeight + paddingHeightL + paddingHeightU;
  const auto dilatedWidth = (inputWidth - 1) * inputDilation[1] + 1;
  const auto paddedWidth = dilatedWidth + paddingWidthL + paddingWidthU;

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        dilatedIn(boost::extents[dilatedHeight][dilatedWidth][inputChannels]);
    std::fill(dilatedIn.data(), dilatedIn.data() + dilatedIn.num_elements(),
              0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          dilatedIn[y * inputDilation[0]][x * inputDilation[1]][c] =
               in[b][y][x][c];
        }
      }
    }

    boost::multi_array<double, 3>
        paddedIn(boost::extents[paddedHeight][paddedWidth][inputChannels]);
    std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(), 0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != dilatedHeight; ++y) {
        for (unsigned x = 0; x != dilatedWidth; ++x) {
          paddedIn[y + paddingHeightL][x + paddingWidthL][c] =
              dilatedIn[y][x][c];
        }
      }
    }

    // Perform convolution.
    const auto outputChannels = out.shape()[3];
    const auto kernelHeight = paddedKernel.shape()[0];
    const auto kernelWidth = paddedKernel.shape()[1];
    const auto convOutHeight = absdiff(paddedHeight, kernelHeight) + 1;
    const auto convOutWidth = absdiff(paddedWidth, kernelWidth) + 1;
    boost::multi_array<double, 3>
        convOut(boost::extents[convOutHeight]
                              [convOutWidth]
                              [outputChannels]);
    std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutHeight; ++y) {
        for (unsigned x = 0; x != convOutWidth; ++x) {
          for (unsigned ky = 0;
               ky != std::min(kernelHeight, paddedHeight); ++ky) {
            for (unsigned kx = 0;
                 kx != std::min(kernelWidth, paddedWidth); ++kx) {
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                convOut[y][x][oc] +=
                    paddedKernel[ky + (kernelHeight < paddedHeight ? 0 : y)]
                                [kx + (kernelWidth  < paddedWidth  ? 0 : x)]
                                [oc][ic] *
                    paddedIn[ky + (kernelHeight < paddedHeight ? y : 0)]
                            [kx + (kernelWidth  < paddedWidth  ? x : 0)][ic];
              }
            }
          }
          convOut[y][x][oc] += biases[oc];
        }
      }
    }

    // Downsample.
    const auto outHeight = (convOutHeight + strideH - 1) / strideH;
    const auto outWidth = (convOutWidth + strideW - 1) / strideW;
    if (outHeight != out.shape()[1] ||
        outWidth != out.shape()[2]) {
      throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != outHeight; ++y) {
        for (unsigned x = 0; x != outWidth; ++x) {
          out[b][y][x][oc] = convOut[y * strideH][x * strideW][oc];
        }
      }
    }
  }
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

  unsigned strideH = stride[0];
  unsigned strideW = stride[1];
  unsigned paddingHeightL = paddingLower[0];
  unsigned paddingWidthL = paddingLower[1];
  unsigned paddingHeightU = paddingUpper[0];
  unsigned paddingWidthU = paddingUpper[1];

  auto paddedKernel = dilateAndPadKernel(kernel, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper);

  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[3];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto outputHeight = out.shape()[1];
  const auto outputWidth = out.shape()[2];
  const auto kernelHeight = paddedKernel.shape()[0];
  const auto kernelWidth = paddedKernel.shape()[1];

  for (unsigned b = 0; b != batchSize; ++b) {
    // Upsample.
    const auto upsampledHeight =
        outputHeight + paddingHeightL + paddingHeightU - (kernelHeight - 1) ;
    const auto upsampledWidth =
            outputWidth + paddingWidthL + paddingWidthU - (kernelWidth - 1);
    if ((upsampledHeight + strideH - 1)/ strideH != inputHeight ||
        (upsampledWidth + strideW - 1)/ strideW != inputWidth) {
      throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                           "do not match");
    }
    boost::multi_array<double, 3>
        upsampledIn(boost::extents[upsampledHeight]
                                  [upsampledWidth][inputChannels]);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != upsampledHeight; ++y) {
        for (unsigned x = 0; x != upsampledWidth; ++x) {
          if (y % strideH == 0 &&
              x % strideW == 0) {
            upsampledIn[y][x][c] = in[b][y / strideH][x / strideW][c];
          } else {
            upsampledIn[y][x][c] = 0;
          }
        }
      }
    }

    // Perform a full convolution with flipped weights.
    const auto outputChannels = out.shape()[3];
    const auto convOutHeight = upsampledHeight + kernelHeight - 1;
    const auto convOutWidth = upsampledWidth + kernelWidth - 1;
    boost::multi_array<double, 3>
        convOut(boost::extents[convOutHeight]
                              [convOutWidth]
                              [outputChannels]);
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
                  convOut[y][x][oc] += paddedKernel[kyFlipped]
                                                   [kxFlipped][ic][oc] *
                                      upsampledIn[y - (kernelHeight - 1) + ky]
                                                 [x - (kernelWidth - 1) + kx]
                                                 [ic];
                }
              }
            }
          }
        }
      }
    }

    const auto truncOutHeight = convOutHeight - paddingHeightL - paddingHeightU;
    const auto truncOutWidth = convOutWidth - paddingWidthL - paddingWidthU;
    boost::multi_array<double, 3>
        truncOut(boost::extents[truncOutHeight]
                               [truncOutWidth]
                               [outputChannels]);
    // Truncate.
    for (unsigned c = 0; c != outputChannels; ++c) {
      for (unsigned y = 0; y != truncOutHeight; ++y) {
        for (unsigned x = 0; x != truncOutWidth; ++x) {
          truncOut[y][x][c] =
              convOut[y + paddingHeightL][x + paddingWidthL][c];
        }
      }
    }

    // Downsample.
    const auto inDilationH = inputDilation[0];
    const auto inDilationW = inputDilation[1];
    const auto outHeight = (truncOutHeight + inDilationH - 1) / inDilationH;
    const auto outWidth = (truncOutWidth + inDilationW - 1) / inDilationW;
    if (outHeight != out.shape()[1] ||
        outWidth != out.shape()[2]) {
      throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != outHeight; ++y) {
        for (unsigned x = 0; x != outWidth; ++x) {
          out[b][y][x][oc] = truncOut[y * inDilationH][x * inDilationW][oc];
        }
      }
    }
  }
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

  unsigned strideH = stride[0];
  unsigned strideW = stride[1];
  unsigned paddingHeightL = paddingLower[0];
  unsigned paddingWidthL = paddingLower[1];
  unsigned paddingHeightU = paddingUpper[0];
  unsigned paddingWidthU = paddingUpper[1];

  // Pad activations.
  const auto batchSize = activations.shape()[0];
  const auto inputChannels = activations.shape()[3];
  const auto inputHeight = activations.shape()[1];
  const auto inputWidth = activations.shape()[2];
  const auto paddedHeight = inputHeight + paddingHeightL + paddingHeightU;
  const auto paddedWidth = inputWidth + paddingWidthL + paddingWidthU;

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedActivations(boost::extents[paddedHeight]
                                        [paddedWidth][inputChannels]);
    std::fill(paddedActivations.data(),
              paddedActivations.data() + paddedActivations.num_elements(),
              0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          paddedActivations[y + paddingHeightL][x + paddingWidthL][c] =
              activations[b][y][x][c];
        }
      }
    }

    // Upsample deltas.
    const auto outputChannels = deltas.shape()[3];
    const auto outputHeight = deltas.shape()[1];
    const auto outputWidth = deltas.shape()[2];
    const auto paddedKernelHeight = paddedKernel.shape()[0];
    const auto paddedKernelWidth = paddedKernel.shape()[1];
    const auto upsampledDeltasHeight =
        inputHeight + paddingHeightL + paddingHeightU -
        (paddedKernelHeight - 1);
    const auto upsampledDeltasWidth =
        inputWidth + paddingWidthL + paddingWidthU - (paddedKernelWidth - 1);
    if ((upsampledDeltasHeight + strideH - 1) / strideH != outputHeight ||
        (upsampledDeltasWidth + strideW - 1) / strideW != outputWidth) {
      throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                       "do not match");
    }
    boost::multi_array<double, 3>
        upsampledDeltas(boost::extents[upsampledDeltasHeight]
                                      [upsampledDeltasWidth][outputChannels]);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
        for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
          if (y % strideH == 0 &&
              x % strideW == 0) {
            upsampledDeltas[y][x][oc] = deltas[b][y / strideH]
                                                 [x / strideW][oc];
          } else {
            upsampledDeltas[y][x][oc] = 0;
          }
        }
      }
    }

    // Compute the weight deltas.
    boost::multi_array<double, 4>
        weightDeltas(boost::extents[paddedKernelHeight][paddedKernelWidth]
                                   [outputChannels][inputChannels]);
    std::fill(weightDeltas.data(),
              weightDeltas.data() + weightDeltas.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned ic = 0; ic != inputChannels; ++ic) {
        for (unsigned ky = 0; ky != paddedKernelHeight; ++ky) {
          for (unsigned kx = 0; kx != paddedKernelWidth; ++kx) {
            for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
              for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
                weightDeltas[ky][kx][oc][ic] +=
                    paddedActivations[y + ky][x + kx][ic] *
                    upsampledDeltas[y][x][oc];
              }
            }
          }
        }
      }
    }

    // Add the weight deltas.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned ic = 0; ic != inputChannels; ++ic) {
        for (unsigned ky = 0; ky != paddedKernelHeight; ++ky) {
          for (unsigned kx = 0; kx != paddedKernelWidth; ++kx) {
            paddedKernel[ky][kx][oc][ic] +=
                learningRate * -weightDeltas[ky][kx][oc][ic];
          }
        }
      }
    }

    // Compute the bias deltas.
    boost::multi_array<double, 1> biasDeltas(boost::extents[outputChannels]);
    std::fill(biasDeltas.data(),
              biasDeltas.data() + biasDeltas.num_elements(), 0.0);
    for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
      for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
        for (unsigned oc = 0; oc != outputChannels; ++oc) {
          biasDeltas[oc] += upsampledDeltas[y][x][oc];
        }
      }
    }

    // Add the bias deltas.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      biases[oc] += learningRate * -biasDeltas[oc];
    }
  }
  auto newKernel = dilateAndPadKernelInverse(paddedKernel, kernelDilation,
                                             kernelPaddingLower,
                                             kernelPaddingUpper);
  assert(newKernel.num_elements() == kernel.num_elements());
  kernel = newKernel;
}
