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
  const auto rawOutHeight = paddedHeight - (kernelHeight - 1);
  const auto rawOutWidth = paddedWidth - (kernelWidth - 1);
  boost::multi_array<double, 3>
      rawOut(boost::extents[outputChannels]
                           [rawOutHeight]
                           [rawOutWidth]);
  std::fill(rawOut.data(), rawOut.data() + rawOut.num_elements(), 0.0);
  for (unsigned c = 0; c != outputChannels; ++c) {
    for (unsigned y = 0; y != rawOutHeight; ++y) {
      for (unsigned x = 0; x != rawOutWidth; ++x) {
        for (unsigned ky = 0; ky != kernelHeight; ++ky) {
          for (unsigned kx = 0; kx != kernelWidth; ++kx) {
            for (unsigned ic = 0; ic != inputChannels; ++ic) {
              rawOut[c][y][x] += weights[c][ic][ky][kx] *
                                 paddedIn[ic][y + ky][x + kx];
            }
          }
        }
        rawOut[c][y][x] += biases[c];
      }
    }
  }

  // Downsample.
  const auto outHeight = (rawOutHeight + stride - 1) / stride;
  const auto outWidth = (rawOutWidth + stride - 1) / stride;
  if (outHeight != out.shape()[1] ||
      outWidth != out.shape()[2]) {
    std::abort();
  }
  for (unsigned c = 0; c != outputChannels; ++c) {
    for (unsigned y = 0; y != outHeight; ++y) {
      for (unsigned x = 0; x != outWidth; ++x) {
        out[c][y][x] = rawOut[c][y * stride][x * stride];
      }
    }
  }

  // Apply nonlinearity.
  ref::fwdNonLinearity(nonLinearityType, out);
}
