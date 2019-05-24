#include <poplibs_test/Pooling.hpp>
#include <poplibs_test/exceptions.hpp>
#include <cassert>
#include <iostream>

using popnn::PoolingType;

using namespace poplibs_support;

static void
pooling(PoolingType pType, unsigned strideHeight, unsigned strideWidth,
        unsigned kernelHeight, unsigned kernelWidth,
        int paddingHeightL, int paddingWidthL,
        int paddingHeightU, int paddingWidthU,
        const MultiArray<double> &in,
        MultiArray<double> &out,
        MultiArray<double> &scale,
        MultiArray<double> *maxCount = nullptr) {
  // for now only support 4D pooling (batch x channels x height x width)
  if (in.numDimensions() != 4 ||
      out.numDimensions() != 4 ||
      scale.numDimensions() != 2) {
    throw poplibs_test::poplibs_test_error(
      "Model pooling only supports 4D pooling.");
  }

  // maxCount should not be null pointer only when pooling type is
  // MAX_POOL. It is assumed that maxCount is set to 0.
  const bool doMaxCount = maxCount != nullptr;
  if (pType != PoolingType::MAX) {
    assert(!doMaxCount);
  }

  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[1];
  const int inputHeight = in.shape()[2];
  const int inputWidth = in.shape()[3];
  const auto paddedHeight = inputHeight + paddingHeightL + paddingHeightU;
  const auto paddedWidth = inputWidth + paddingWidthL + paddingWidthU;
  const double lowestValue = std::numeric_limits<double>::lowest();

  assert(paddedHeight >= 0);
  assert(paddedWidth >= 0);

  for (unsigned b = 0; b != batchSize; ++b) {
    MultiArray<double> paddedIn{channels,
                                std::size_t(paddedHeight),
                                std::size_t(paddedWidth)};
    std::fill_n(paddedIn.data(), paddedIn.numElements(), lowestValue);

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
    MultiArray<double> poolOut{channels, poolOutHeight, poolOutWidth};
    MultiArray<double> scaleOut{poolOutHeight, poolOutWidth};

    MultiArray<double> countUnstrided{channels, poolOutHeight, poolOutWidth};
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != poolOutHeight; ++y) {
        for (unsigned x = 0; x != poolOutWidth; ++x) {
          double v = pType == PoolingType::MAX ? lowestValue : 0;
          unsigned usedKernelElems = 0;
          for (unsigned ky = 0; ky != kernelHeight; ++ky) {
            for (unsigned kx = 0; kx != kernelWidth; ++kx) {
              if (pType == PoolingType::MAX) {
                double nv = paddedIn[c][y + ky][x + kx];
                v = std::max(v, nv);
              } else if ((pType ==  PoolingType::AVG
                        || pType ==  PoolingType::SUM)
                       && (paddedIn[c][y + ky][x + kx] != lowestValue))  {
                v += paddedIn[c][y + ky][x + kx];
                if (pType ==  PoolingType::AVG) {
                  ++usedKernelElems;
                }
              }
            }
          }

          if (doMaxCount) {
            // Now that max is computed, we can count the number of maxima
            // in the input kernel for a given output point in the spatial
            // dimension
            for (unsigned ky = 0; ky != kernelHeight; ++ky) {
              for (unsigned kx = 0; kx != kernelWidth; ++kx) {
                if (paddedIn[c][y + ky][x + kx] == v &&
                    paddedIn[c][y + ky][x + kx] != lowestValue)
                  countUnstrided[c][y][x] += 1.0;
              }
            }
          }

          const double elScale = usedKernelElems != 0
                                 ? 1.0 / usedKernelElems : 1.0;

          // lowestValue must be set to zero if output is only padding;
          poolOut[c][y][x] =
              pType == PoolingType::MAX && v == lowestValue ? 0 : elScale * v;
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
    if (doMaxCount) {
      if (outHeight != maxCount->shape()[2] ||
          outWidth != maxCount->shape()[3]) {
        throw poplibs_test::poplibs_test_error("Count tensor dimensions do "
                                               "not match expected dimensions");
      }
    }
    for (unsigned y = 0; y != outHeight; ++y) {
      for (unsigned x = 0; x != outWidth; ++x) {
        for (unsigned oc = 0; oc != channels; ++oc) {
          out[b][oc][y][x] = poolOut[oc][y * strideHeight][x * strideWidth];
          scale[y][x] = scaleOut[y * strideHeight][x * strideWidth];
          if (doMaxCount) {
            (*maxCount)[b][oc][y][x] = countUnstrided[oc][y * strideHeight]
                                                         [x * strideWidth];
          }
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
        const MultiArray<double> &in,
        MultiArray<double> &out) {
  MultiArray<double> scale{out.shape()[2], out.shape()[3]};
  ::pooling(pType, strideHeight, strideWidth, kernelHeight, kernelWidth,
            paddingHeightL, paddingWidthL, paddingHeightU, paddingWidthU,
            in, out, scale);
}

static void computeGradientScale(unsigned strideHeight, unsigned strideWidth,
                                 unsigned kernelHeight, unsigned kernelWidth,
                                 int paddingHeightL, int paddingWidthL,
                                 int paddingHeightU, int paddingWidthU,
                                 const MultiArray<double> &actsIn,
                                 MultiArray<double> &gradScale) {
  const auto batchSize = actsIn.shape()[0];
  const auto channels = actsIn.shape()[1];
  const auto outputHeight = gradScale.shape()[2];
  const auto outputWidth = gradScale.shape()[3];
  assert(gradScale.shape()[0] == batchSize);
  assert(gradScale.shape()[1] == channels);

  MultiArray<double> actsOut{batchSize, channels, outputHeight, outputWidth};
  MultiArray<double> scale{outputHeight, outputWidth};
  MultiArray<double> maxCount{batchSize, channels, outputHeight, outputWidth};

  ::pooling(PoolingType::MAX, strideHeight, strideWidth, kernelHeight,
            kernelWidth, paddingHeightL, paddingWidthL, paddingHeightU,
            paddingWidthU, actsIn, actsOut, scale, &maxCount);

  for (std::size_t b = 0; b != batchSize; ++b) {
    for (std::size_t c = 0; c != channels; ++c) {
      for (std::size_t h = 0; h != outputHeight; ++h) {
        for (std::size_t w = 0; w != outputWidth; ++w) {
          gradScale[b][c][h][w] = 1.0 / maxCount[b][c][h][w];
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
                   const MultiArray<double> &prevAct,
                   const MultiArray<double> &nextAct,
                   const MultiArray<double> &in,
                   MultiArray<double> &out,
                   bool useScaledGradient) {
  const auto batchSize = in.shape()[0];
  const auto channels = in.shape()[1];
  const int inputHeight = in.shape()[2];
  const int inputWidth = in.shape()[3];
  const int outputHeight = out.shape()[2];
  const int outputWidth = out.shape()[3];

  assert(inputHeight >= 0);
  assert(inputWidth >= 0);

  MultiArray<double> scaledIn{batchSize,
                              channels,
                              std::size_t(inputHeight),
                              std::size_t(inputWidth)};

  if (useScaledGradient) {
    MultiArray<double> gradScale{batchSize,
                                 channels,
                                 std::size_t(inputHeight),
                                 std::size_t(inputWidth)};

    computeGradientScale(strideHeight, strideWidth, kernelHeight, kernelWidth,
                         paddingHeightL, paddingWidthL,
                         paddingHeightU, paddingWidthU, prevAct, gradScale);
    // scale gradients
    for (std::size_t b = 0; b != batchSize; ++b) {
      for (std::size_t c = 0; c != channels; ++c) {
        for (int h = 0; h != inputHeight; ++h) {
          for (int w = 0; w != inputWidth; ++w) {
            scaledIn[b][c][h][w] = in[b][c][h][w] * gradScale[b][c][h][w];
          }
        }
      }
    }
  } else {
    assert(in.shape() == scaledIn.shape());
    std::copy_n(in.data(), in.numElements(), scaledIn.data());
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    // Pad activations.
    const int actHeight = prevAct.shape()[2];
    const int actWidth = prevAct.shape()[3];
    const auto paddedHeight = actHeight + paddingHeightL + paddingHeightU;
    const auto paddedWidth = actWidth + paddingWidthL + paddingWidthU;

    assert(paddedHeight >= 0);
    assert(paddedWidth >= 0);

    MultiArray<double> paddedActivations{channels,
                                         std::size_t(paddedHeight),
                                         std::size_t(paddedWidth)};
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

    assert(upsampledHeight >= 0);
    assert(upsampledWidth >= 0);

    MultiArray<double> upsampledIn{channels,
                                   std::size_t(upsampledHeight),
                                   std::size_t(upsampledWidth)};
    MultiArray<double> upsampledNextAct{channels,
                                        std::size_t(upsampledHeight),
                                        std::size_t(upsampledWidth)};
    for (int y = 0; y != upsampledHeight; ++y) {
      for (int x = 0; x != upsampledWidth; ++x) {
        for (unsigned c = 0; c != channels; ++c) {
          if (y % strideHeight == 0 &&
              x % strideWidth == 0) {
            upsampledIn[c][y][x] =
                scaledIn[b][c][y / strideHeight][x / strideWidth];
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

    assert(poolOutHeight >= 0);
    assert(poolOutWidth >= 0);

    MultiArray<double> poolOut{outputChannels,
                               std::size_t(poolOutHeight),
                               std::size_t(poolOutWidth)};
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
                   const MultiArray<double> &prevAct,
                   const MultiArray<double> &nextAct,
                   const MultiArray<double> &in,
                   MultiArray<double> &out) {
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
  MultiArray<double> scale{inputHeight, inputWidth};
  MultiArray<double> fwdAct{nextAct.shape()[0],
                            nextAct.shape()[1],
                            nextAct.shape()[2],
                            nextAct.shape()[3]};
  pooling(pType, strideHeight, strideWidth, kernelHeight, kernelWidth,
          paddingHeightL, paddingWidthL, paddingHeightU, paddingWidthU,
          prevAct, fwdAct, scale);
  MultiArray<double> scaledIn{batchSize, channels, inputHeight, inputWidth};

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
    MultiArray<double> poolOut{outputChannels, poolOutHeight, poolOutWidth};

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
    bool useScaledGradientForMaxPool,
    unsigned strideHeight, unsigned strideWidth,
    unsigned kernelHeight, unsigned kernelWidth,
    int paddingHeightL, int paddingWidthL,
    int paddingHeightU, int paddingWidthU,
    const MultiArray<double> &prevAct,
    const MultiArray<double> &nextAct,
    const MultiArray<double> &in,
    MultiArray<double> &out) {
  if (pType == PoolingType::MAX) {
    maxPoolingBackward(strideHeight, strideWidth, kernelHeight,  kernelWidth,
                       paddingHeightL,  paddingWidthL,
                       paddingHeightU,  paddingWidthU,
                       prevAct, nextAct, in, out,
                       useScaledGradientForMaxPool);
  } else if (pType == PoolingType::AVG || pType == PoolingType::SUM) {
    sumPoolingBackward(pType, strideHeight, strideWidth, kernelHeight,
                       kernelWidth, paddingHeightL,  paddingWidthL,
                       paddingHeightU,  paddingWidthU,
                       prevAct, nextAct, in, out);
  }
}
