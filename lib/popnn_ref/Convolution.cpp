#include <popnn_ref/Convolution.hpp>

#include <popnn_ref/NonLinearity.hpp>

void ref::conv::
convolution(unsigned stride, unsigned padding,
            NonLinearityType nonLinearityType,
            const boost::multi_array<double, 3> &in,
            const boost::multi_array<double, 4> &weights,
            const boost::multi_array<double, 1> &biases,
            boost::multi_array<double, 3> &out) {
  // Pad input.
  const auto inputChannels = in.shape()[0];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto paddedHeight = inputHeight + 2 * padding;
  const auto paddedWidth = inputWidth + 2 * padding;

  boost::multi_array<double, 3>
      paddedIn(boost::extents[inputChannels][paddedHeight][paddedWidth]);
  std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(), 0.0);
  for (unsigned c = 0; c != inputChannels; ++c) {
    for (unsigned y = 0; y != inputHeight; ++y) {
      for (unsigned x = 0; x != inputWidth; ++x) {
        paddedIn[c][y + padding][x + padding] = in[c][y][x];
      }
    }
  }

  // Perform convolution.
  const auto outputChannels = out.shape()[0];
  const auto kernelHeight = weights.shape()[2];
  const auto kernelWidth = weights.shape()[3];
  if (paddedHeight < kernelHeight ||
      paddedWidth < kernelWidth) {
    std::abort();
  }
  const auto convOutHeight = paddedHeight - (kernelHeight - 1);
  const auto convOutWidth = paddedWidth - (kernelWidth - 1);
  boost::multi_array<double, 3>
      convOut(boost::extents[outputChannels]
                            [convOutHeight]
                            [convOutWidth]);
  std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    for (unsigned y = 0; y != convOutHeight; ++y) {
      for (unsigned x = 0; x != convOutWidth; ++x) {
        for (unsigned ky = 0; ky != kernelHeight; ++ky) {
          for (unsigned kx = 0; kx != kernelWidth; ++kx) {
            for (unsigned ic = 0; ic != inputChannels; ++ic) {
              convOut[oc][y][x] += weights[oc][ic][ky][kx] *
                                 paddedIn[ic][y + ky][x + kx];
            }
          }
        }
        convOut[oc][y][x] += biases[oc];
      }
    }
  }

  // Downsample.
  const auto outHeight = (convOutHeight + stride - 1) / stride;
  const auto outWidth = (convOutWidth + stride - 1) / stride;
  if (outHeight != out.shape()[1] ||
      outWidth != out.shape()[2]) {
    std::abort();
  }
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    for (unsigned y = 0; y != outHeight; ++y) {
      for (unsigned x = 0; x != outWidth; ++x) {
        out[oc][y][x] = convOut[oc][y * stride][x * stride];
      }
    }
  }

  // Apply nonlinearity.
  ref::nonLinearity(nonLinearityType, out);
}

void ref::conv::
convolutionBackward(unsigned stride, unsigned padding,
                    const boost::multi_array<double, 3> &in,
                    const boost::multi_array<double, 4> &weights,
                    boost::multi_array<double, 3> &out) {
  const auto inputChannels = in.shape()[0];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto outputHeight = out.shape()[1];
  const auto outputWidth = out.shape()[2];
  const auto kernelHeight = weights.shape()[2];
  const auto kernelWidth = weights.shape()[3];

  // Upsample.
  const auto upsampledHeight = outputHeight + 2 * padding - (kernelHeight - 1) ;
  const auto upsampledWidth = outputWidth + 2 * padding - (kernelWidth - 1);
  if ((upsampledHeight + stride - 1)/ stride != inputHeight ||
      (upsampledWidth + stride - 1)/ stride != inputWidth) {
    std::abort();
  }
  boost::multi_array<double, 3>
      upsampledIn(boost::extents[inputChannels][upsampledHeight]
                                [upsampledWidth]);
  for (unsigned c = 0; c != inputChannels; ++c) {
    for (unsigned y = 0; y != upsampledHeight; ++y) {
      for (unsigned x = 0; x != upsampledWidth; ++x) {
        if (y % stride == 0 &&
            x % stride == 0) {
          upsampledIn[c][y][x] = in[c][y / stride][x / stride];
        } else {
          upsampledIn[c][y][x] = 0;
        }
      }
    }
  }

  // Perform a full convolution with flipped weights.
  const auto outputChannels = out.shape()[0];
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
  for (unsigned c = 0; c != inputChannels; ++c) {
    for (unsigned y = 0; y != outputHeight; ++y) {
      for (unsigned x = 0; x != outputWidth; ++x) {
        out[c][y][x] = convOut[c][y + padding][x + padding];
      }
    }
  }
}

void ref::conv::weightUpdate(unsigned stride, unsigned padding,
                             double learningRate,
                             const boost::multi_array<double, 3> &activations,
                             const boost::multi_array<double, 3> &deltas,
                             boost::multi_array<double, 4> &weights,
                             boost::multi_array<double, 1> &biases) {
  // Pad activations.
  const auto inputChannels = activations.shape()[0];
  const auto inputHeight = activations.shape()[1];
  const auto inputWidth = activations.shape()[2];
  const auto paddedHeight = inputHeight + 2 * padding;
  const auto paddedWidth = inputWidth + 2 * padding;

  boost::multi_array<double, 3>
      paddedActivations(boost::extents[inputChannels][paddedHeight]
                                      [paddedWidth]);
  std::fill(paddedActivations.data(),
            paddedActivations.data() + paddedActivations.num_elements(),
            0.0);
  for (unsigned c = 0; c != inputChannels; ++c) {
    for (unsigned y = 0; y != inputHeight; ++y) {
      for (unsigned x = 0; x != inputWidth; ++x) {
        paddedActivations[c][y + padding][x + padding] = activations[c][y][x];
      }
    }
  }

  // Upsample deltas.
  const auto outputChannels = deltas.shape()[0];
  const auto outputHeight = deltas.shape()[1];
  const auto outputWidth = deltas.shape()[2];
  const auto kernelHeight = weights.shape()[2];
  const auto kernelWidth = weights.shape()[3];
  const auto upsampledDeltasHeight =
      inputHeight + 2 * padding - (kernelHeight - 1);
  const auto upsampledDeltasWidth =
      inputWidth + 2 * padding - (kernelWidth - 1);
  if ((upsampledDeltasHeight + stride - 1) / stride != outputHeight ||
      (upsampledDeltasWidth + stride - 1) / stride != outputWidth) {
    std::abort();
  }
  boost::multi_array<double, 3>
      upsampledDeltas(boost::extents[inputChannels][upsampledDeltasHeight]
                                [upsampledDeltasWidth]);
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    for (unsigned y = 0; y != upsampledDeltasHeight; ++y) {
      for (unsigned x = 0; x != upsampledDeltasWidth; ++x) {
        if (y % stride == 0 &&
            x % stride == 0) {
          upsampledDeltas[oc][y][x] = deltas[oc][y / stride][x / stride];
        } else {
          upsampledDeltas[oc][y][x] = 0;
        }
      }
    }
  }

  // Compute the weight deltas.
  boost::multi_array<double, 4>
      weightDeltas(boost::extents[outputChannels][inputChannels][kernelHeight]
                                 [kernelWidth]);
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
