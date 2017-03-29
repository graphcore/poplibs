#include <poplib_test/MaxPooling.hpp>
#include <poplib_test/exceptions.hpp>
#include <iostream>

void poplib_test::maxpool::
maxPooling(unsigned strideHeight, unsigned strideWidth,
           unsigned kernelHeight, unsigned kernelWidth,
           unsigned paddingHeight, unsigned paddingWidth,
           const boost::multi_array<double, 4> &in,
           boost::multi_array<double, 4> &out) {
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[1];
  const auto inputHeight = in.shape()[2];
  const auto inputWidth = in.shape()[3];
  const auto paddedHeight = inputHeight + 2 * paddingHeight;
  const auto paddedWidth = inputWidth + 2 * paddingWidth;

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedIn(boost::extents[channels][paddedHeight][paddedWidth]);
    std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(),
              std::numeric_limits<double>::lowest());
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          paddedIn[c][y + paddingHeight][x + paddingWidth] = in[b][c][y][x];
        }
      }
    }

    // Perform pooling.
    if (paddedHeight < kernelHeight ||
        paddedWidth < kernelWidth) {
      throw poplib_test::poplib_test_error("Kernels larger than (padded) input "
                                           "not supported");
    }
    const auto poolOutHeight = paddedHeight - (kernelHeight - 1);
    const auto poolOutWidth = paddedWidth - (kernelWidth - 1);
    boost::multi_array<double, 3>
        poolOut(boost::extents[channels]
                              [poolOutHeight]
                              [poolOutWidth]);
    std::fill(poolOut.data(), poolOut.data() + poolOut.num_elements(), 0.0);
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != poolOutHeight; ++y) {
        for (unsigned x = 0; x != poolOutWidth; ++x) {
          double v = std::numeric_limits<double>::lowest();
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              v = std::max(v, paddedIn[c][y + ky][x + kx]);
            }
          }
          poolOut[c][y][x] = v;
        }
      }
    }

    // Downsample.
    const auto outHeight = (poolOutHeight + strideHeight - 1) / strideHeight;
    const auto outWidth = (poolOutWidth + strideWidth - 1) / strideWidth;
    if (outHeight != out.shape()[2] ||
        outWidth != out.shape()[3]) {
      throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned oc = 0; oc != channels; ++oc) {
      for (unsigned y = 0; y != outHeight; ++y) {
        for (unsigned x = 0; x != outWidth; ++x) {
          out[b][oc][y][x] = poolOut[oc][y * strideHeight][x * strideWidth];
        }
      }
    }

  }
}


void poplib_test::maxpool::maxPoolingBackward(
    unsigned strideHeight, unsigned strideWidth,
    unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth,
    const boost::multi_array<double, 4> &prevAct,
    const boost::multi_array<double, 4> &nextAct,
    const boost::multi_array<double, 4> &in,
    boost::multi_array<double, 4> &out) {
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[1];
  const auto inputHeight = in.shape()[2];
  const auto inputWidth = in.shape()[3];
  const auto outputHeight = out.shape()[2];
  const auto outputWidth = out.shape()[3];

  for (unsigned b = 0; b != batchSize; ++b) {
    // Pad activations.
    const auto actHeight = prevAct.shape()[2];
    const auto actWidth = prevAct.shape()[3];
    const auto paddedHeight = actHeight + 2 * paddingHeight;
    const auto paddedWidth = actWidth + 2 * paddingWidth;
    boost::multi_array<double, 3>
        paddedActivations(boost::extents[channels][paddedHeight]
                                        [paddedWidth]);
    std::fill(paddedActivations.data(),
              paddedActivations.data() + paddedActivations.num_elements(),
              0.0);
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != actHeight; ++y) {
        for (unsigned x = 0; x != actWidth; ++x) {
          paddedActivations[c][y + paddingHeight][x + paddingWidth] =
              prevAct[b][c][y][x];
        }
      }
    }

    // Upsample.
    const auto upsampledHeight =
        outputHeight + 2 * paddingHeight - (kernelHeight - 1) ;
    const auto upsampledWidth =
        outputWidth + 2 * paddingWidth - (kernelWidth - 1);
    if ((upsampledHeight + strideHeight - 1)/ strideHeight != inputHeight ||
        (upsampledWidth + strideWidth - 1)/ strideWidth != inputWidth) {
      throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                           "do not match");
    }
    boost::multi_array<double, 3>
        upsampledIn(boost::extents[channels][upsampledHeight]
                                  [upsampledWidth]);
    boost::multi_array<double, 3>
        upsampledNextAct(boost::extents[channels][upsampledHeight]
                                       [upsampledWidth]);
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != upsampledHeight; ++y) {
        for (unsigned x = 0; x != upsampledWidth; ++x) {
          if (y % strideHeight == 0 &&
              x % strideWidth == 0) {
            upsampledIn[c][y][x] = in[b][c][y / strideHeight][x / strideWidth];
            upsampledNextAct[c][y][x] =
                nextAct[b][c][y / strideHeight][x / strideWidth];
          } else {
            upsampledIn[c][y][x] = 0;
            upsampledNextAct[c][y][x] =
                std::numeric_limits<double>::quiet_NaN();
          }
        }
      }
    }

    // Perform a full convolution with flipped weights.
    const auto outputChannels = out.shape()[1];
    const auto poolOutHeight = upsampledHeight + kernelHeight - 1;
    const auto poolOutWidth = upsampledWidth + kernelWidth - 1;
    if (poolOutHeight != paddedHeight ||
        poolOutWidth  != paddedWidth) {
      throw poplib_test::poplib_test_error("Deltas and activation tensor "
                                       "dimensions do not match");
    }
    boost::multi_array<double, 3>
        poolOut(boost::extents[outputChannels]
                              [poolOutHeight]
                              [poolOutWidth]);
    std::fill(poolOut.data(), poolOut.data() + poolOut.num_elements(), 0.0);
    for (unsigned c = 0; c != outputChannels; ++c) {
      for (unsigned y = 0; y != poolOutHeight; ++y) {
        for (unsigned x = 0; x != poolOutWidth; ++x) {
          double v = 0;
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            if (ky > y || (y - ky) >= upsampledHeight)
              continue;
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              if (kx > x || (x - kx) >= upsampledWidth)
                continue;
              if (paddedActivations[c][y][x] ==
                  upsampledNextAct[c][y - ky][x - kx]) {
                v += upsampledIn[c][y - ky][x - kx];
              }
            }
          }
          poolOut[c][y][x] = v;
        }
      }
    }

    // Truncate.
    for (unsigned c = 0; c != outputChannels; ++c) {
      for (unsigned y = 0; y != outputHeight; ++y) {
        for (unsigned x = 0; x != outputWidth; ++x) {
          out[b][c][y][x] = poolOut[c][y + paddingHeight][x + paddingWidth];
        }
      }
    }
  }
}
