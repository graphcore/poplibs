#include <poplibs_test/Pooling.hpp>
#include <poplibs_test/exceptions.hpp>
#include <cassert>
#include <iostream>

using popnn::PoolingType;

static void
pooling(PoolingType pType, unsigned strideHeight, unsigned strideWidth,
        unsigned kernelHeight, unsigned kernelWidth,
        int paddingHeightL, int paddingWidthL,
        int paddingHeightU, int paddingWidthU,
        const boost::multi_array<double, 4> &in,
        boost::multi_array<double, 4> &out,
        boost::multi_array_ref<double, 2> &scale) {
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[1];
  const int inputHeight = in.shape()[2];
  const int inputWidth = in.shape()[3];
  const auto paddedHeight = inputHeight + paddingHeightL + paddingHeightU;
  const auto paddedWidth = inputWidth + paddingWidthL + paddingWidthU;
  const double lowestValue = std::numeric_limits<double>::lowest();

  for (unsigned b = 0; b != batchSize; ++b) {
    boost::multi_array<double, 3>
        paddedIn(boost::extents[channels][paddedHeight][paddedWidth]);
    std::fill(paddedIn.data(), paddedIn.data() + paddedIn.num_elements(),
              lowestValue);

    for (int y = 0; y != paddedHeight; ++y) {
      for (int x = 0; x != paddedWidth; ++x) {
        if ((y - paddingHeightL) < 0 ||
            (y - paddingHeightL) >= inputHeight ||
            (x - paddingWidthL) < 0 ||
            (x - paddingWidthL) >= inputWidth) {
          continue;
        }
        for (unsigned c = 0; c != channels; ++c) {
          paddedIn[c][y][x] = in[b][c][y - paddingHeightL][x - paddingWidthL];
        }
      }
    }

    // Perform pooling.
    if (paddedHeight < static_cast<int>(kernelHeight) ||
        paddedWidth < static_cast<int>(kernelWidth)) {
      throw poplibs_test::poplibs_test_error("Kernels larger than (padded) "
                                             "input not supported");
    }
    const auto poolOutHeight = paddedHeight - (kernelHeight - 1);
    const auto poolOutWidth = paddedWidth - (kernelWidth - 1);
    boost::multi_array<double, 3>
        poolOut(boost::extents[channels]
                              [poolOutHeight]
                              [poolOutWidth]);
    boost::multi_array<double, 2>
        scaleOut(boost::extents[poolOutHeight]
                               [poolOutWidth]);

    std::fill(poolOut.data(), poolOut.data() + poolOut.num_elements(), 0.0);
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != poolOutHeight; ++y) {
        for (unsigned x = 0; x != poolOutWidth; ++x) {
          double v = pType == PoolingType::MAX ? lowestValue : 0;
          unsigned usedKernelElems = 0;
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              if (pType == PoolingType::MAX)
                v = std::max(v, paddedIn[c][y + ky][x + kx]);
              else if ((pType ==  PoolingType::AVG
                        || pType ==  PoolingType::SUM)
                       && (paddedIn[c][y + ky][x + kx] != lowestValue))  {
                v += paddedIn[c][y + ky][x + kx];
                if (pType ==  PoolingType::AVG) {
                  ++usedKernelElems;
                }
              }
            }
          }
          const double elScale = usedKernelElems != 0
                                 ? 1.0 / usedKernelElems : 1.0;
          poolOut[c][y][x] = elScale * v;
          scaleOut[y][x] = elScale;
        }
      }
    }

    // Downsample.
    const auto outHeight = (poolOutHeight + strideHeight - 1) / strideHeight;
    const auto outWidth = (poolOutWidth + strideWidth - 1) / strideWidth;
    if (outHeight != out.shape()[2] ||
        outWidth != out.shape()[3]) {
      throw poplibs_test::poplibs_test_error("Output tensor dimensions do not "
                                           "match expected dimensions");
    }
    for (unsigned y = 0; y != outHeight; ++y) {
      for (unsigned x = 0; x != outWidth; ++x) {
        for (unsigned oc = 0; oc != channels; ++oc) {
          out[b][oc][y][x] = poolOut[oc][y * strideHeight][x * strideWidth];
          scale[y][x] = scaleOut[y * strideHeight][x * strideWidth];
        }
      }
    }
  }
}

void poplibs_test::pooling::
pooling(PoolingType pType, unsigned strideHeight, unsigned strideWidth,
        unsigned kernelHeight, unsigned kernelWidth,
        int paddingHeightL, int paddingWidthL,
        int paddingHeightU, int paddingWidthU,
        const boost::multi_array<double, 4> &in,
        boost::multi_array<double, 4> &out) {
  boost::multi_array<double, 2> scale(boost::extents[out.shape()[2]]
                                                    [out.shape()[3]]);
  ::pooling(pType, strideHeight, strideWidth, kernelHeight, kernelWidth,
            paddingHeightL, paddingWidthL, paddingHeightU, paddingWidthU,
            in, out, scale);
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
  const auto channels = in.shape()[1];
  const int inputHeight = in.shape()[2];
  const int inputWidth = in.shape()[3];
  const int outputHeight = out.shape()[2];
  const int outputWidth = out.shape()[3];

  for (unsigned b = 0; b != batchSize; ++b) {
    // Pad activations.
    const int actHeight = prevAct.shape()[2];
    const int actWidth = prevAct.shape()[3];
    const auto paddedHeight = actHeight + paddingHeightL + paddingHeightU;
    const auto paddedWidth = actWidth + paddingWidthL + paddingWidthU;
    boost::multi_array<double, 3>
        paddedActivations(boost::extents[channels]
                                        [paddedHeight]
                                        [paddedWidth]);
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
          paddedActivations[c][y][x] =
              prevAct[b][c][y - paddingHeightL][x - paddingWidthL];
        }
      }
    }

    // Upsample.
    const int upsampledHeight =
        outputHeight + paddingHeightL + paddingHeightU - (kernelHeight - 1) ;
    const int upsampledWidth =
        outputWidth + paddingWidthL + paddingWidthU - (kernelWidth - 1);
    if ((upsampledHeight + static_cast<int>(strideHeight) - 1)
        / static_cast<int>(strideHeight) != inputHeight
        ||
        (upsampledWidth + static_cast<int>(strideWidth) - 1)
        / static_cast<int>(strideWidth) != inputWidth) {
      throw poplibs_test::poplibs_test_error("Output and input tensor "
                                             "dimensions do not match");
    }
    boost::multi_array<double, 3>
        upsampledIn(boost::extents[channels]
                                  [upsampledHeight]
                                  [upsampledWidth]);
    boost::multi_array<double, 3>
        upsampledNextAct(boost::extents[channels]
                                       [upsampledHeight]
                                       [upsampledWidth]);
    for (int y = 0; y != upsampledHeight; ++y) {
      for (int x = 0; x != upsampledWidth; ++x) {
        for (unsigned c = 0; c != channels; ++c) {
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
    const int poolOutHeight = upsampledHeight + kernelHeight - 1;
    const int poolOutWidth = upsampledWidth + kernelWidth - 1;
    if (poolOutHeight != paddedHeight ||
        poolOutWidth  != paddedWidth) {
      throw poplibs_test::poplibs_test_error("Deltas and activation tensor "
                                       "dimensions do not match");
    }
    boost::multi_array<double, 3>
        poolOut(boost::extents[outputChannels]
                              [poolOutHeight]
                              [poolOutWidth]);
    std::fill(poolOut.data(), poolOut.data() + poolOut.num_elements(), 0.0);
    for (unsigned c = 0; c != outputChannels; ++c) {
      for (int y = 0; y != poolOutHeight; ++y) {
        for (int x = 0; x != poolOutWidth; ++x) {
          double v = 0;
          for (int ky = 0; ky != static_cast<int>(kernelHeight); ++ky) {
            if (ky > y || (y - ky) >= upsampledHeight)
              continue;
            for (int kx = 0; kx != static_cast<int>(kernelWidth); ++kx) {
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
    for (int y = 0; y != outputHeight; ++y) {
      for (int x = 0; x != outputWidth; ++x) {
        for (unsigned c = 0; c != outputChannels; ++c) {
          if ((y + paddingHeightL) < 0 ||
              (y + paddingHeightL) >= poolOutHeight ||
              (x + paddingWidthL) < 0 ||
              (x + paddingWidthL) >= poolOutWidth) {
            continue;
          }
          out[b][c][y][x] = poolOut[c][y + paddingHeightL][x + paddingWidthL];
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
  const auto channels = in.shape()[1];
  const auto inputHeight = in.shape()[2];
  const auto inputWidth = in.shape()[3];
  const int outputHeight = out.shape()[2];
  const int outputWidth = out.shape()[3];

  const auto actHeight = prevAct.shape()[2];
  const auto actWidth = prevAct.shape()[3];
  const auto paddedHeight = actHeight + paddingHeightL + paddingHeightU;
  const auto paddedWidth = actWidth + paddingWidthL + paddingWidthU;
  const auto upsampledHeight =
      outputHeight + paddingHeightL + paddingHeightU - (kernelHeight - 1) ;
  const auto upsampledWidth =
      outputWidth + paddingWidthL + paddingWidthU - (kernelWidth - 1);
  if ((upsampledHeight + strideHeight - 1)/ strideHeight != inputHeight ||
      (upsampledWidth + strideWidth - 1)/ strideWidth != inputWidth) {
    throw poplibs_test::poplibs_test_error("Output and input tensor dimensions "
                                           "do not match");
  }
  const auto poolOutHeight = upsampledHeight + kernelHeight - 1;
  const auto poolOutWidth = upsampledWidth + kernelWidth - 1;
  if (poolOutHeight != paddedHeight ||
      poolOutWidth  != paddedWidth) {
    throw poplibs_test::poplibs_test_error("Deltas and activation tensor "
                                         "dimensions do not match");
  }

  // Run forward pooling to get scale factors. The output of the pooling
  // is not used
  boost::multi_array<double, 2> scale(boost::extents[inputHeight][inputWidth]);
  boost::multi_array<double, 4> fwdAct(boost::extents[nextAct.shape()[0]]
                                                     [nextAct.shape()[1]]
                                                     [nextAct.shape()[2]]
                                                     [nextAct.shape()[3]]);
  pooling(pType, strideHeight, strideWidth, kernelHeight, kernelWidth,
          paddingHeightL, paddingWidthL, paddingHeightU, paddingWidthU,
          prevAct, fwdAct, scale);
  boost::multi_array<double, 4> scaledIn(boost::extents[batchSize]
                                                       [channels]
                                                       [inputHeight]
                                                       [inputWidth]);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned h = 0; h != inputHeight; ++h) {
      for (unsigned w = 0; w != inputWidth; ++w) {
        for (unsigned c = 0; c != channels; ++c) {
          scaledIn[b][c][h][w] = in[b][c][h][w] * scale[h][w];
        }
      }
    }
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    const auto outputChannels = out.shape()[1];
    boost::multi_array<double, 3>
        poolOut(boost::extents[outputChannels]
                              [poolOutHeight]
                              [poolOutWidth]);

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
              poolOut[c][y * strideHeight + ky][x * strideWidth + kx] +=
                                                         scaledIn[b][c][y][x];
            }
          }
        }
      }
    }
    // Truncate and scale
    for (int y = 0; y != outputHeight; ++y) {
      for (int x = 0; x != outputWidth; ++x) {
        for (unsigned c = 0; c != outputChannels; ++c) {
          if ((y + paddingHeightL) < 0 ||
              (y + paddingHeightL) >= static_cast<int>(poolOutHeight) ||
              (x + paddingWidthL) < 0 ||
              (x + paddingWidthL) >= static_cast<int>(poolOutWidth)) {
            continue;
          }
          out[b][c][y][x] = poolOut[c][y + paddingHeightL][x + paddingWidthL];
        }
      }
    }
  }
}

void poplibs_test::pooling::poolingBackward(
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
