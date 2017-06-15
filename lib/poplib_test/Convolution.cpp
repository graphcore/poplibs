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
    for (unsigned x = 0; x != dilatedShape[1]; ++x) {
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
  const auto outputChannels = padded.shape()[2];
  const auto inputChannels = padded.shape()[3];
  std::vector<unsigned> dilatedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    dilatedShape[dim] = padded.shape()[dim] - kernelPaddingLower[dim] -
                        kernelPaddingUpper[dim];
  }
  boost::multi_array<double, 4>
      dilated(boost::extents[dilatedShape[0]][dilatedShape[1]]
                            [outputChannels][inputChannels]);
  for (unsigned y = 0; y != dilatedShape[0]; ++y) {
    for (unsigned x = 0; x != dilatedShape[1]; ++x) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        for (unsigned ic = 0; ic != inputChannels; ++ic) {
          dilated[y][x][oc][ic] =
              padded[y + kernelPaddingLower[0]][x + kernelPaddingLower[1]]
                    [oc][ic];
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

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        dilatedIn(boost::extents[dilatedShape[0]][dilatedShape[1]]
                                [inputChannels]);
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
        paddedIn(boost::extents[paddedShape[0]][paddedShape[1]][inputChannels]);
    std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(), 0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != dilatedShape[0]; ++y) {
        for (unsigned x = 0; x != dilatedShape[1]; ++x) {
          paddedIn[y + paddingLower[0]][x + paddingLower[1]][c] =
              dilatedIn[y][x][c];
        }
      }
    }

    // Perform convolution.
    const auto outputChannels = out.shape()[3];
    std::vector<unsigned> paddedKernelShape(2);
    std::vector<unsigned> convOutShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      paddedKernelShape[dim] = paddedKernel.shape()[dim];
      convOutShape[dim] = absdiff(paddedShape[dim],
                                  paddedKernelShape[dim]) + 1;
    }
    boost::multi_array<double, 3>
        convOut(boost::extents[convOutShape[0]]
                              [convOutShape[1]]
                              [outputChannels]);
    std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutShape[0]; ++y) {
        for (unsigned x = 0; x != convOutShape[1]; ++x) {
          for (unsigned ky = 0;
               ky != std::min(paddedKernelShape[0], paddedShape[0]); ++ky) {
            for (unsigned kx = 0;
                 kx != std::min(paddedKernelShape[1], paddedShape[1]); ++kx) {
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                convOut[y][x][oc] +=
                    paddedKernel[ky +
                                 (paddedKernelShape[0] < paddedShape[0] ? 0 :
                                                                          y)]
                                [kx +
                                 (paddedKernelShape[1] < paddedShape[1] ? 0 :
                                                                          x)]
                                [oc][ic] *
                    paddedIn[ky +
                             (paddedKernelShape[0] < paddedShape[0] ? y : 0)]
                            [kx +
                             (paddedKernelShape[1] < paddedShape[1]  ? x : 0)]
                            [ic];
              }
            }
          }
          convOut[y][x][oc] += biases[oc];
        }
      }
    }

    // Downsample.
    std::vector<unsigned> outShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      outShape[dim] = (convOutShape[dim] + stride[dim] - 1) / stride[dim];
    }
    if (outShape[0] != out.shape()[1] ||
        outShape[1] != out.shape()[2]) {
      throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != outShape[0]; ++y) {
        for (unsigned x = 0; x != outShape[1]; ++x) {
          out[b][y][x][oc] = convOut[y * stride[0]][x * stride[1]][oc];
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

  auto paddedKernel = dilateAndPadKernel(kernel, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper);

  const auto batchSize = in.shape()[0];
  const auto inputChannels = in.shape()[3];

  for (unsigned b = 0; b != batchSize; ++b) {
    // Upsample.
    std::vector<unsigned> paddedKernelShape(2);
    std::vector<unsigned> upsampledShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      paddedKernelShape[dim] = paddedKernel.shape()[dim];
      upsampledShape[dim] =
          out.shape()[dim + 1] +
          paddingLower[dim] +
          paddingUpper[dim] -
          (paddedKernelShape[dim] - 1);
      if ((upsampledShape[dim] + stride[dim] - 1)/ stride[dim] !=
          in.shape()[dim + 1]) {
        throw poplib_test::poplib_test_error("Output and input tensor "
                                             "dimensions do not match");
      }
    }
    boost::multi_array<double, 3>
        upsampledIn(boost::extents[upsampledShape[0]]
                                  [upsampledShape[1]][inputChannels]);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != upsampledShape[0]; ++y) {
        for (unsigned x = 0; x != upsampledShape[1]; ++x) {
          if (y % stride[0] == 0 &&
              x % stride[1] == 0) {
            upsampledIn[y][x][c] = in[b][y / stride[0]][x / stride[1]][c];
          } else {
            upsampledIn[y][x][c] = 0;
          }
        }
      }
    }

    // Perform a full convolution with flipped weights.
    const auto outputChannels = out.shape()[3];
    std::vector<unsigned> convOutShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      convOutShape[dim] = upsampledShape[dim] + paddedKernelShape[dim] - 1;
    }
    boost::multi_array<double, 3>
        convOut(boost::extents[convOutShape[0]]
                              [convOutShape[1]]
                              [outputChannels]);
    std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != convOutShape[0]; ++y) {
        for (unsigned x = 0; x != convOutShape[1]; ++x) {
          for (unsigned ky = 0; ky != paddedKernelShape[0]; ++ky) {
            const auto kyFlipped = paddedKernelShape[0] - 1 - ky;
            for (unsigned kx = 0; kx != paddedKernelShape[1]; ++kx) {
              const auto kxFlipped = paddedKernelShape[1] - 1 - kx;
              for (unsigned ic = 0; ic != inputChannels; ++ic) {
                if (y + ky >= (paddedKernelShape[0] - 1) &&
                    y + ky < (paddedKernelShape[0] - 1) + upsampledShape[0] &&
                    x + kx >= (paddedKernelShape[1] - 1) &&
                    x + kx < (paddedKernelShape[1] - 1) + upsampledShape[1]) {
                  convOut[y][x][oc] +=
                      paddedKernel[kyFlipped][kxFlipped][ic][oc] *
                      upsampledIn[y - (paddedKernelShape[0] - 1) + ky]
                                 [x - (paddedKernelShape[1] - 1) + kx][ic];
                }
              }
            }
          }
        }
      }
    }

    std::vector<unsigned> truncOutShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      truncOutShape[dim] = convOutShape[dim] - paddingLower[dim] -
                           paddingUpper[dim];
    }
    boost::multi_array<double, 3>
        truncOut(boost::extents[truncOutShape[0]]
                               [truncOutShape[1]]
                               [outputChannels]);
    // Truncate.
    for (unsigned c = 0; c != outputChannels; ++c) {
      for (unsigned y = 0; y != truncOutShape[0]; ++y) {
        for (unsigned x = 0; x != truncOutShape[1]; ++x) {
          truncOut[y][x][c] =
              convOut[y + paddingLower[0]][x + paddingLower[1]][c];
        }
      }
    }

    // Downsample.
    std::vector<unsigned> outShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      outShape[dim] = (truncOutShape[dim] + inputDilation[dim] - 1) /
                      inputDilation[dim];
      if (outShape[dim] != out.shape()[dim + 1]) {
        throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                             "match expected dimensions");
      }
    }
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != outShape[0]; ++y) {
        for (unsigned x = 0; x != outShape[1]; ++x) {
          out[b][y][x][oc] = truncOut[y * inputDilation[0]]
                                     [x * inputDilation[1]][oc];
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

  // Pad activations.
  const auto batchSize = activations.shape()[0];
  const auto inputChannels = activations.shape()[3];
  std::vector<unsigned> inputShape(2);
  std::vector<unsigned> paddedShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    inputShape[dim] = activations.shape()[dim + 1];
    paddedShape[dim] = inputShape[dim] + paddingLower[dim] + paddingUpper[dim];
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedActivations(boost::extents[paddedShape[0]]
                                        [paddedShape[1]][inputChannels]);
    std::fill(paddedActivations.data(),
              paddedActivations.data() + paddedActivations.num_elements(),
              0.0);
    for (unsigned c = 0; c != inputChannels; ++c) {
      for (unsigned y = 0; y != inputShape[0]; ++y) {
        for (unsigned x = 0; x != inputShape[1]; ++x) {
          paddedActivations[y + paddingLower[0]][x + paddingLower[1]][c] =
              activations[b][y][x][c];
        }
      }
    }

    // Upsample deltas.
    const auto outputChannels = deltas.shape()[3];
    std::vector<unsigned> paddedKernelShape(2);
    std::vector<unsigned> upsampledDeltasShape(2);
    for (unsigned dim = 0; dim != 2; ++dim) {
      paddedKernelShape[dim] = paddedKernel.shape()[dim];
      upsampledDeltasShape[dim] = inputShape[dim] + paddingLower[dim] +
                                  paddingUpper[dim] -
                                  (paddedKernelShape[dim] - 1);
      if ((upsampledDeltasShape[dim] + stride[dim] - 1) / stride[dim] !=
          deltas.shape()[dim + 1]) {
        throw poplib_test::poplib_test_error("Output and input tensor "
                                             "dimensions do not match");
      }
    }
    boost::multi_array<double, 3>
        upsampledDeltas(boost::extents[upsampledDeltasShape[0]]
                                      [upsampledDeltasShape[1]]
                                      [outputChannels]);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned y = 0; y != upsampledDeltasShape[0]; ++y) {
        for (unsigned x = 0; x != upsampledDeltasShape[1]; ++x) {
          if (y % stride[0] == 0 &&
              x % stride[1] == 0) {
            upsampledDeltas[y][x][oc] = deltas[b][y / stride[0]]
                                                 [x / stride[1]][oc];
          } else {
            upsampledDeltas[y][x][oc] = 0;
          }
        }
      }
    }

    // Compute the weight deltas.
    boost::multi_array<double, 4>
        weightDeltas(boost::extents[paddedKernelShape[0]][paddedKernelShape[1]]
                                   [outputChannels][inputChannels]);
    std::fill(weightDeltas.data(),
              weightDeltas.data() + weightDeltas.num_elements(), 0.0);
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (unsigned ic = 0; ic != inputChannels; ++ic) {
        for (unsigned ky = 0; ky != paddedKernelShape[0]; ++ky) {
          for (unsigned kx = 0; kx != paddedKernelShape[1]; ++kx) {
            for (unsigned y = 0; y != upsampledDeltasShape[0]; ++y) {
              for (unsigned x = 0; x != upsampledDeltasShape[1]; ++x) {
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
        for (unsigned ky = 0; ky != paddedKernelShape[0]; ++ky) {
          for (unsigned kx = 0; kx != paddedKernelShape[1]; ++kx) {
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
    for (unsigned y = 0; y != upsampledDeltasShape[0]; ++y) {
      for (unsigned x = 0; x != upsampledDeltasShape[1]; ++x) {
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
