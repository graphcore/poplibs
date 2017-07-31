#include <poplib_test/Pooling.hpp>
#include <poplib_test/exceptions.hpp>
#include <iostream>


static inline double
getScale(PoolingType pType, unsigned kernelHeight, unsigned kernelWidth) {
  double scale = 1.0;
  if (pType == PoolingType::AVG) {
    scale = 1.0 / (kernelHeight * kernelWidth);
  }
  return scale;
}

void poplib_test::pooling::
pooling(PoolingType pType, unsigned strideHeight, unsigned strideWidth,
        unsigned kernelHeight, unsigned kernelWidth,
        int paddingHeightL, int paddingWidthL,
        int paddingHeightU, int paddingWidthU,
        const boost::multi_array<double, 4> &in,
        boost::multi_array<double, 4> &out) {
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[3];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto paddedHeight = inputHeight + paddingHeightL + paddingHeightU;
  const auto paddedWidth = inputWidth + paddingWidthL + paddingWidthU;

  double scale = getScale(pType, kernelHeight, kernelWidth);

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedIn(boost::extents[paddedHeight][paddedWidth][channels]);
    std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(),
              pType == PoolingType::MAX ?
                  std::numeric_limits<double>::lowest() : 0);

    for (int y = 0; y != paddedHeight; ++y) {
      for (int x = 0; x != paddedWidth; ++x) {
        if ((y - paddingHeightL) < 0 ||
            (y - paddingHeightL) >= inputHeight ||
            (x - paddingWidthL) < 0 ||
            (x - paddingWidthL) >= inputWidth) {
          continue;
        }
        for (unsigned c = 0; c != channels; ++c) {
          paddedIn[y][x][c] = in[b][y - paddingHeightL][x - paddingWidthL][c];
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
        poolOut(boost::extents[poolOutHeight]
                              [poolOutWidth]
                              [channels]);
    std::fill(poolOut.data(), poolOut.data() + poolOut.num_elements(), 0.0);
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != poolOutHeight; ++y) {
        for (unsigned x = 0; x != poolOutWidth; ++x) {
          double v = pType == PoolingType::MAX ?
                              std::numeric_limits<double>::lowest() :
                              0;
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              if (pType == PoolingType::MAX)
                v = std::max(v, paddedIn[y + ky][x + kx][c]);
              else if (pType ==  PoolingType::AVG || pType ==  PoolingType::SUM)
                v += paddedIn[y + ky][x + kx][c];
            }
          }
          poolOut[y][x][c] = v * scale;
        }
      }
    }

    // Downsample.
    const auto outHeight = (poolOutHeight + strideHeight - 1) / strideHeight;
    const auto outWidth = (poolOutWidth + strideWidth - 1) / strideWidth;
    if (outHeight != out.shape()[1] ||
        outWidth != out.shape()[2]) {
      throw poplib_test::poplib_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned y = 0; y != outHeight; ++y) {
      for (unsigned x = 0; x != outWidth; ++x) {
        for (unsigned oc = 0; oc != channels; ++oc) {
          out[b][y][x][oc] = poolOut[y * strideHeight][x * strideWidth][oc];
        }
      }
    }
  }
}


static void
maxPoolingBackward(unsigned strideHeight, unsigned strideWidth,
                   unsigned kernelHeight, unsigned kernelWidth,
                   int paddingHeightL, int paddingWidthL,
                   int paddingHeightU, int paddingWidthU,
                   const boost::multi_array<double, 4> &prevAct,
                   const boost::multi_array<double, 4> &nextAct,
                   const boost::multi_array<double, 4> &in,
                   boost::multi_array<double, 4> &out) {
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[3];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto outputHeight = out.shape()[1];
  const auto outputWidth = out.shape()[2];

  for (unsigned b = 0; b != batchSize; ++b) {
    // Pad activations.
    const auto actHeight = prevAct.shape()[1];
    const auto actWidth = prevAct.shape()[2];
    const auto paddedHeight = actHeight + paddingHeightL + paddingHeightU;
    const auto paddedWidth = actWidth + paddingWidthL + paddingWidthU;
    boost::multi_array<double, 3>
        paddedActivations(boost::extents[paddedHeight]
                                        [paddedWidth][channels]);
    std::fill(paddedActivations.data(),
              paddedActivations.data() + paddedActivations.num_elements(),
              0.0);
    for (int y = 0; y != paddedHeight; ++y) {
      for (int x = 0; x != paddedWidth; ++x) {
        for (unsigned c = 0; c != channels; ++c) {
          if ((y - paddingHeightL) < 0 ||
              (y - paddingHeightL) >= actHeight ||
              (x - paddingWidthL) < 0 ||
              (x - paddingWidthL) >= actWidth) {
            continue;
          }
          paddedActivations[y][x][c] =
              prevAct[b][y - paddingHeightL][x - paddingWidthL][c];
        }
      }
    }

    // Upsample.
    const auto upsampledHeight =
        outputHeight + paddingHeightL + paddingHeightU - (kernelHeight - 1) ;
    const auto upsampledWidth =
        outputWidth + paddingWidthL + paddingWidthU - (kernelWidth - 1);
    if ((upsampledHeight + strideHeight - 1)/ strideHeight != inputHeight ||
        (upsampledWidth + strideWidth - 1)/ strideWidth != inputWidth) {
      throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                           "do not match");
    }
    boost::multi_array<double, 3>
        upsampledIn(boost::extents[upsampledHeight]
                                  [upsampledWidth][channels]);
    boost::multi_array<double, 3>
        upsampledNextAct(boost::extents[upsampledHeight]
                                       [upsampledWidth][channels]);
    for (unsigned y = 0; y != upsampledHeight; ++y) {
      for (unsigned x = 0; x != upsampledWidth; ++x) {
        for (unsigned c = 0; c != channels; ++c) {
          if (y % strideHeight == 0 &&
              x % strideWidth == 0) {
            upsampledIn[y][x][c] = in[b][y / strideHeight][x / strideWidth][c];
            upsampledNextAct[y][x][c] =
                nextAct[b][y / strideHeight][x / strideWidth][c];
          } else {
            upsampledIn[y][x][c] = 0;
            upsampledNextAct[y][x][c] =
                std::numeric_limits<double>::quiet_NaN();
          }
        }
      }
    }

    // Perform a full convolution with flipped weights.
    const auto outputChannels = out.shape()[3];
    const auto poolOutHeight = upsampledHeight + kernelHeight - 1;
    const auto poolOutWidth = upsampledWidth + kernelWidth - 1;
    if (poolOutHeight != paddedHeight ||
        poolOutWidth  != paddedWidth) {
      throw poplib_test::poplib_test_error("Deltas and activation tensor "
                                       "dimensions do not match");
    }
    boost::multi_array<double, 3>
        poolOut(boost::extents[poolOutHeight]
                              [poolOutWidth]
                              [outputChannels]);
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
              if (paddedActivations[y][x][c] ==
                  upsampledNextAct[y - ky][x - kx][c]) {
                v += upsampledIn[y - ky][x - kx][c];
              }
            }
          }
          poolOut[y][x][c] = v;
        }
      }
    }

    // Truncate.
    for (int y = 0; y != outputHeight; ++y) {
      for (int x = 0; x != outputWidth; ++x) {

        for (unsigned c = 0; c != outputChannels; ++c) {
          if ((y + paddingHeightL) < 0 ||
              (y + paddingHeightL) >= poolOutHeight ||
              (x + paddingWidthL) < 0 ||
              (x + paddingWidthL) >= poolOutWidth) {
            continue;
          }
          out[b][y][x][c] = poolOut[y + paddingHeightL][x + paddingWidthL][c];
        }
      }
    }
  }
}

static void
sumPoolingBackward(PoolingType pType,
                   unsigned strideHeight, unsigned strideWidth,
                   unsigned kernelHeight, unsigned kernelWidth,
                   int paddingHeightL, int paddingWidthL,
                   int paddingHeightU, int paddingWidthU,
                   const boost::multi_array<double, 4> &prevAct,
                   const boost::multi_array<double, 4> &nextAct,
                   const boost::multi_array<double, 4> &in,
                   boost::multi_array<double, 4> &out) {
  assert(pType == PoolingType::AVG || pType == PoolingType::SUM);
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[3];
  const auto inputHeight = in.shape()[1];
  const auto inputWidth = in.shape()[2];
  const auto outputHeight = out.shape()[1];
  const auto outputWidth = out.shape()[2];

  const auto actHeight = prevAct.shape()[1];
  const auto actWidth = prevAct.shape()[2];
  const auto paddedHeight = actHeight + paddingHeightL + paddingHeightU;
  const auto paddedWidth = actWidth + paddingWidthL + paddingWidthU;
  const auto upsampledHeight =
      outputHeight + paddingHeightL + paddingHeightU - (kernelHeight - 1) ;
  const auto upsampledWidth =
      outputWidth + paddingWidthL + paddingWidthU - (kernelWidth - 1);
  if ((upsampledHeight + strideHeight - 1)/ strideHeight != inputHeight ||
      (upsampledWidth + strideWidth - 1)/ strideWidth != inputWidth) {
    throw poplib_test::poplib_test_error("Output and input tensor dimensions "
                                           "do not match");
  }
  const auto poolOutHeight = upsampledHeight + kernelHeight - 1;
  const auto poolOutWidth = upsampledWidth + kernelWidth - 1;
  if (poolOutHeight != paddedHeight ||
      poolOutWidth  != paddedWidth) {
    throw poplib_test::poplib_test_error("Deltas and activation tensor "
                                         "dimensions do not match");
  }
  const auto scale = getScale(pType, kernelHeight, kernelWidth);

  for (unsigned b = 0; b != batchSize; ++b) {
    const auto outputChannels = out.shape()[3];
    boost::multi_array<double, 3>
        poolOut(boost::extents[poolOutHeight]
                              [poolOutWidth]
                              [outputChannels]);

    std::fill(poolOut.data(), poolOut.data() + poolOut.num_elements(), 0.0);

    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != inputHeight; ++y) {
        for (unsigned x = 0; x != inputWidth; ++x) {
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            if (y * strideHeight + ky >= poolOutHeight) {
              continue;
            }
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              if (x * strideWidth + kx >= poolOutWidth) {
                continue;
              }
              poolOut[y * strideHeight + ky][x * strideWidth + kx][c] +=
                                                         scale * in[b][y][x][c];
            }
          }
        }
      }
    }
    // Truncate.
    for (int y = 0; y != outputHeight; ++y) {
      for (int x = 0; x != outputWidth; ++x) {
        for (unsigned c = 0; c != outputChannels; ++c) {
          if ((y + paddingHeightL) < 0 ||
              (y + paddingHeightL) >= poolOutHeight ||
              (x + paddingWidthL) < 0 ||
              (x + paddingWidthL) >= poolOutWidth) {
            continue;
          }
          out[b][y][x][c] = poolOut[y + paddingHeightL][x + paddingWidthL][c];
        }
      }
    }
  }
}

void poplib_test::pooling::poolingBackward(
    PoolingType pType,
    unsigned strideHeight, unsigned strideWidth,
    unsigned kernelHeight, unsigned kernelWidth,
    int paddingHeightL, int paddingWidthL,
    int paddingHeightU, int paddingWidthU,
    const boost::multi_array<double, 4> &prevAct,
    const boost::multi_array<double, 4> &nextAct,
    const boost::multi_array<double, 4> &in,
    boost::multi_array<double, 4> &out) {
  if (pType == PoolingType::MAX) {
    maxPoolingBackward(strideHeight, strideWidth, kernelHeight,  kernelWidth,
                       paddingHeightL,  paddingWidthL,
                       paddingHeightU,  paddingWidthU,
                       prevAct, nextAct, in, out);
  } else if (pType == PoolingType::AVG || pType == PoolingType::SUM) {
    sumPoolingBackward(pType, strideHeight, strideWidth, kernelHeight,
                       kernelWidth, paddingHeightL,  paddingWidthL,
                       paddingHeightU,  paddingWidthU,
                       prevAct, nextAct, in, out);
  }
}
