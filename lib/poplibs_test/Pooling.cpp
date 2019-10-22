#include <cassert>
#include <iostream>
#include <poplibs_test/Pooling.hpp>
#include <poplibs_test/exceptions.hpp>

using popnn::PoolingType;

using namespace poplibs_support;

static void pooling(popnn::PoolingType pType,
                    const std::vector<unsigned> &stride,
                    const std::vector<std::size_t> &kernel,
                    const std::vector<int> &paddingLower,
                    const std::vector<int> &paddingUpper,
                    const MultiArray<double> &in, MultiArray<double> &out,
                    MultiArray<double> &scale,
                    MultiArray<double> *maxCount = nullptr) {
  // maxCount should not be null pointer only when pooling type is
  // MAX_POOL. It is assumed that maxCount is set to 0.
  const bool doMaxCount = maxCount != nullptr;
  if (pType != PoolingType::MAX) {
    assert(!doMaxCount);
  }

  const auto batchSize = in.shape()[0];
  const MultiArrayShape kernelShape{std::begin(kernel), std::end(kernel)};

  const double lowestValue = std::numeric_limits<double>::lowest();

  const auto inShape = in.shape();
  MultiArrayShape paddedInShape{std::begin(inShape) + 1, std::end(inShape)};
  for (unsigned i = 1; i < paddedInShape.size(); ++i) {
    paddedInShape[i] += paddingLower[i - 1] + paddingUpper[i - 1];
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    MultiArray<double> paddedIn{paddedInShape};
    std::fill_n(paddedIn.data(), paddedIn.numElements(), lowestValue);

    MultiArrayShape inIndices;
    forEachIndex(paddedInShape, [&](const MultiArrayShapeRange indices) {
      // the index into the in array is [b][c][field - padding...]
      inIndices.clear();
      inIndices.push_back(b);
      inIndices.push_back(indices[0]);

      for (unsigned i = 1; i < indices.size(); ++i) {
        const int dim = indices[i] - paddingLower[i - 1];
        if (dim < 0 || dim >= static_cast<int>(inShape[i + 1])) {
          return;
        }

        inIndices.push_back(dim);
      }

      paddedIn[indices] = in[inIndices];
    });

    // Perform pooling.
    MultiArrayShape poolOutShape;
    poolOutShape.push_back(paddedInShape[0]);
    for (unsigned i = 1; i < paddedInShape.size(); ++i) {
      if (paddedInShape[i] < kernel[i - 1]) {
        throw poplibs_test::poplibs_test_error("Kernels larger than (padded) "
                                               "input not supported");
      }

      poolOutShape.push_back(paddedInShape[i] - (kernel[i - 1] - 1));
    }

    MultiArray<double> poolOut{poolOutShape};

    MultiArrayShapeRange scaleOutShape = poolOutShape;
    scaleOutShape.advance_begin(1);
    MultiArray<double> scaleOut{scaleOutShape};

    MultiArray<double> countUnstrided{poolOutShape};

    MultiArrayShape paddedInIndices;
    forEachIndex(poolOutShape, [&](const MultiArrayShapeRange indices) {
      double v = pType == PoolingType::MAX ? lowestValue : 0;
      unsigned usedKernelElems = 0;
      forEachIndex(kernelShape, [&](const MultiArrayShapeRange kernelIndices) {
        paddedInIndices.clear();
        paddedInIndices.push_back(indices[0]);
        for (unsigned i = 1; i < indices.size(); ++i) {
          paddedInIndices.push_back(indices[i] + kernelIndices[i - 1]);
        }

        if (pType == PoolingType::MAX) {
          double nv = paddedIn[paddedInIndices];
          v = std::max(v, nv);
        } else if ((pType == PoolingType::AVG || pType == PoolingType::SUM) &&
                   (paddedIn[paddedInIndices] != lowestValue)) {
          v += paddedIn[paddedInIndices];
          if (pType == PoolingType::AVG) {
            ++usedKernelElems;
          }
        }
      });

      if (doMaxCount) {
        // Now that max is computed, we can count the number of maxima
        // in the input kernel for a given output point in the spatial
        // dimension
        forEachIndex(
            kernelShape, [&](const MultiArrayShapeRange kernelIndices) {
              paddedInIndices.clear();
              paddedInIndices.push_back(indices[0]);
              for (unsigned i = 1; i < indices.size(); ++i) {
                paddedInIndices.push_back(indices[i] + kernelIndices[i - 1]);
              }

              if (paddedIn[paddedInIndices] == v &&
                  paddedIn[paddedInIndices] != lowestValue)
                countUnstrided[indices] += 1.0;
            });
      }

      const double elScale = usedKernelElems != 0 ? 1.0 / usedKernelElems : 1.0;

      // lowestValue must be set to zero if output is only padding;
      poolOut[indices] =
          pType == PoolingType::MAX && v == lowestValue ? 0 : elScale * v;

      auto scaleOutIndices = indices;
      scaleOutIndices.advance_begin(1);
      scaleOut[scaleOutIndices] = elScale;
    });

    // Downsample.
    MultiArrayShape outShape;
    outShape.push_back(poolOutShape[0]);
    for (unsigned i = 1; i < poolOutShape.size(); ++i) {
      const auto outDim = (poolOutShape[i] + stride[i - 1] - 1) / stride[i - 1];
      if (outDim != out.shape()[i + 1]) {
        throw poplibs_test::poplibs_test_error("Output tensor dimensions do not"
                                               " match expected dimensions");
      }

      if (doMaxCount) {
        if (outDim != maxCount->shape()[i + 1]) {
          throw poplibs_test::poplibs_test_error("Output tensor dimensions do "
                                                 "not match expected");
        }
      }

      outShape.push_back(outDim);
    }

    MultiArrayShape outIndices;
    MultiArrayShape poolOutIndices;
    forEachIndex(outShape, [&](const MultiArrayShapeRange indices) {
      outIndices.clear();
      outIndices.push_back(b);
      outIndices.insert(std::end(outIndices), std::begin(indices),
                        std::end(indices));

      poolOutIndices.clear();
      poolOutIndices.push_back(indices[0]);
      for (unsigned i = 1; i < indices.size(); ++i) {
        poolOutIndices.push_back(indices[i] * stride[i - 1]);
      }

      MultiArrayShapeRange scaleIndices = outIndices;
      scaleIndices.advance_begin(2);

      MultiArrayShapeRange scaleOutIndices = poolOutIndices;
      scaleOutIndices.advance_begin(1);

      out[outIndices] = poolOut[poolOutIndices];
      scale[scaleIndices] = scaleOut[scaleOutIndices];
      if (doMaxCount) {
        (*maxCount)[outIndices] = countUnstrided[poolOutIndices];
      }
    });
  }
}

void poplibs_test::pooling::pooling(popnn::PoolingType pType,
                                    const std::vector<unsigned> &stride,
                                    const std::vector<std::size_t> &kernel,
                                    const std::vector<int> &paddingLower,
                                    const std::vector<int> &paddingUpper,
                                    const MultiArray<double> &in,
                                    MultiArray<double> &out) {
  auto scaleShape = out.shape();
  scaleShape.advance_begin(2);
  MultiArray<double> scale{scaleShape};
  ::pooling(pType, stride, kernel, paddingLower, paddingUpper, in, out, scale);
}

static void computeGradientScale(
    const std::vector<unsigned> &stride, const std::vector<std::size_t> &kernel,
    const std::vector<int> &paddingLower, const std::vector<int> &paddingUpper,
    const poplibs_support::MultiArray<double> &actsIn,
    poplibs_support::MultiArray<double> &gradScale) {
  const auto gradShape = gradScale.shape();

  // batch size and channels
  assert(gradShape.size() >= 2);
  assert(gradShape[0] == actsIn.shape()[0]);
  assert(gradShape[1] == actsIn.shape()[1]);

  MultiArray<double> actsOut{gradShape};

  auto scaleShape = gradShape;
  scaleShape.advance_begin(2);
  MultiArray<double> scale{scaleShape};
  MultiArray<double> maxCount{gradShape};

  ::pooling(PoolingType::MAX, stride, kernel, paddingLower, paddingUpper,
            actsIn, actsOut, scale, &maxCount);

  forEachIndex(gradShape, [&](const MultiArrayShapeRange indices) {
    gradScale[indices] = 1.0 / maxCount[indices];
  });
}

static void maxPoolingBackward(
    const std::vector<unsigned> &stride, const std::vector<std::size_t> &kernel,
    const std::vector<int> &paddingLower, const std::vector<int> &paddingUpper,
    const poplibs_support::MultiArray<double> &prevAct,
    const poplibs_support::MultiArray<double> &nextAct,
    const poplibs_support::MultiArray<double> &in,
    poplibs_support::MultiArray<double> &out, const bool useScaledGradient) {
  const auto batchSize = in.shape()[0];

  const auto inShape = in.shape();
  MultiArray<double> scaledIn{inShape};

  if (useScaledGradient) {
    MultiArray<double> gradScale{inShape};
    computeGradientScale(stride, kernel, paddingLower, paddingUpper, prevAct,
                         gradScale);

    // scale gradients
    forEachIndex(inShape, [&](const MultiArrayShapeRange indices) {
      scaledIn[indices] = in[indices] * gradScale[indices];
    });
  } else {
    assert(inShape == scaledIn.shape());
    std::copy_n(in.data(), in.numElements(), scaledIn.data());
  }

  for (unsigned b = 0; b != batchSize; ++b) {
    const auto prevActShape = prevAct.shape();

    // Pad activations.
    assert(prevActShape[0] == inShape[0]);
    assert(prevActShape[1] == inShape[1]);

    MultiArrayShape paddedActShape{std::begin(prevActShape) + 1,
                                   std::end(prevActShape)};
    assert(paddedActShape.size() - 1 == paddingLower.size());
    assert(paddedActShape.size() - 1 == paddingUpper.size());

    for (unsigned i = 1; i < paddedActShape.size(); ++i) {
      paddedActShape[i] += paddingLower[i - 1] + paddingUpper[i - 1];
      assert(paddedActShape[i] >= 0);
    }

    MultiArray<double> paddedActivations{paddedActShape};

    MultiArrayShape prevActIndices;
    forEachIndex(paddedActShape, [&](const MultiArrayShapeRange indices) {
      // the index into the prevAct array is [b][c][field - padding...]
      prevActIndices.clear();
      prevActIndices.push_back(b);
      prevActIndices.push_back(indices[0]);

      // first dim is channels, check that the remaining dims are in the range
      //   0 <= D < prevAct dim
      for (unsigned i = 1; i < indices.size(); ++i) {
        const int dim = static_cast<int>(indices[i]) - paddingLower[i - 1];
        if (dim < 0 || dim >= static_cast<int>(prevActShape[i + 1])) {
          return;
        }

        prevActIndices.push_back(indices[i] - paddingLower[i - 1]);
      }

      paddedActivations[indices] = prevAct[prevActIndices];
    });

    // Upsample.
    auto outShape = out.shape();
    outShape.advance_begin(1);

    MultiArrayShape upsampledShape{std::begin(outShape), std::end(outShape)};
    for (unsigned i = 1; i < upsampledShape.size(); ++i) {
      upsampledShape[i] +=
          paddingLower[i - 1] + paddingUpper[i - 1] - (kernel[i - 1] - 1);
    }

    // first dim is channels.
    for (unsigned i = 1; i < upsampledShape.size(); ++i) {
      if ((upsampledShape[i] + static_cast<int>(stride[i - 1]) - 1) /
              static_cast<int>(stride[i - 1]) !=
          inShape[i + 1]) {
        throw poplibs_test::poplibs_test_error("Output and input tensor "
                                               "dimensions do not match");
      }
    }

    MultiArray<double> upsampledIn{upsampledShape};
    MultiArray<double> upsampledNextAct{upsampledShape};

    MultiArrayShape stridedIndices;
    forEachIndex(upsampledShape, [&](const MultiArrayShapeRange indices) {
      // if all of the fields are exact multiples of the stride
      const auto multiplesOfStride = [&] {
        for (unsigned i = 1; i < indices.size(); ++i) {
          if (indices[i] % stride[i - 1] != 0) {
            return false;
          }
        }

        return true;
      };

      if (multiplesOfStride()) {
        // the index into the inputs is [b][c][field / stride...]
        stridedIndices.clear();
        stridedIndices.push_back(b);
        stridedIndices.push_back(indices[0]);
        for (unsigned i = 1; i < indices.size(); ++i) {
          stridedIndices.push_back(indices[i] / stride[i - 1]);
        }

        upsampledIn[indices] = scaledIn[stridedIndices];
        upsampledNextAct[indices] = nextAct[stridedIndices];
      } else {
        upsampledIn[indices] = 0;
        upsampledNextAct[indices] = std::numeric_limits<double>::quiet_NaN();
      }
    });

    assert(kernel.size() + 1 == upsampledShape.size());
    for (unsigned i = 1; i < kernel.size(); ++i) {
      const auto poolOut = upsampledShape[i] + kernel[i - 1] - 1;
      if (poolOut != paddedActShape[i]) {
        throw poplibs_test::poplibs_test_error("Deltas and activation tensor "
                                               "dimensions do not match");
      }
    }

    MultiArrayShape kernelShape{std::begin(kernel), std::end(kernel)};
    MultiArray<double> poolOut{paddedActShape};

    MultiArrayShape upsampledIndices;
    forEachIndex(paddedActShape, [&](const MultiArrayShapeRange indices) {
      double v = 0;

      forEachIndex(kernelShape, [&](const MultiArrayShapeRange kernelIndices) {
        // the index into the upsampled arrays is [c][y - ky][...] where y is
        // the dim of the poolOut shape and ky is the dim of the kernel shape.
        upsampledIndices.clear();
        upsampledIndices.push_back(indices[0]);

        for (unsigned i = 1; i < indices.size(); ++i) {
          const int k = kernelIndices[i - 1];

          if (k > static_cast<int>(indices[i]) ||
              (indices[i] - k) >= upsampledShape[i]) {
            return;
          }

          upsampledIndices.push_back(indices[i] - k);
        }

        if (paddedActivations[indices] == upsampledNextAct[upsampledIndices]) {
          v += upsampledIn[upsampledIndices];
        }
      });

      poolOut[indices] = v;
    });

    // Truncate.
    MultiArrayShape outIndices;
    MultiArrayShape poolOutIndices;
    forEachIndex(outShape, [&](const MultiArrayShapeRange indices) {
      // the index into the out array is [b][c][field...] and for the
      // poolOut array it is [c][field + padding...]
      outIndices.clear();
      outIndices.push_back(b);
      outIndices.push_back(indices[0]);

      poolOutIndices.clear();
      poolOutIndices.push_back(indices[0]);

      // first dim is channels, check that the remaining dims are in the range
      //   0 <= D < prevAct dim
      for (unsigned i = 1; i < indices.size(); ++i) {
        const int dim = static_cast<int>(indices[i]) + paddingLower[i - 1];
        if (dim < 0 || dim >= static_cast<int>(paddedActShape[i])) {
          return;
        }

        outIndices.push_back(indices[i]);
        poolOutIndices.push_back(indices[i] + paddingLower[i - 1]);
      }

      out[outIndices] = poolOut[poolOutIndices];
    });
  }
}

static void
sumPoolingBackward(PoolingType pType, const std::vector<unsigned> &stride,
                   const std::vector<std::size_t> &kernel,
                   const std::vector<int> &paddingLower,
                   const std::vector<int> &paddingUpper,
                   const poplibs_support::MultiArray<double> &prevAct,
                   const poplibs_support::MultiArray<double> &nextAct,
                   const poplibs_support::MultiArray<double> &in,
                   poplibs_support::MultiArray<double> &out) {
  assert(pType == PoolingType::AVG || pType == PoolingType::SUM);
  const auto batchSize = in.shape()[0];

  MultiArrayShape poolOutShape;
  poolOutShape.push_back(out.shape()[1]);
  for (unsigned i = 0; i < kernel.size(); ++i) {
    const auto inDim = in.shape()[i + 2];
    const auto outDim = out.shape()[i + 2];
    const auto actDim = prevAct.shape()[i + 2];

    const auto paddedDim = actDim + paddingLower[i] + paddingUpper[i];
    const auto upsampledDim =
        outDim + paddingLower[i] + paddingUpper[i] - (kernel[i] - 1);

    if ((upsampledDim + stride[i] - 1) / stride[i] != inDim) {
      throw poplibs_test::poplibs_test_error("Output and input tensor "
                                             "dimensions do not match");
    }

    const auto poolOutDim = upsampledDim + kernel[i] - 1;
    if (poolOutDim != paddedDim) {
      throw poplibs_test::poplibs_test_error("Deltas and activation tensor "
                                             "dimensions do not match");
    }

    poolOutShape.push_back(poolOutDim);
  }

  // Run forward pooling to get scale factors. The output of the pooling
  // is not used
  auto scaleShape = in.shape();
  scaleShape.advance_begin(2);
  MultiArray<double> scale{scaleShape};
  MultiArray<double> fwdAct{nextAct.shape()};
  pooling(pType, stride, kernel, paddingLower, paddingUpper, prevAct, fwdAct,
          scale);

  MultiArray<double> scaledIn{in.shape()};
  forEachIndex(in.shape(), [&](const MultiArrayShapeRange indices) {
    auto scaleIndices = indices;
    scaleIndices.advance_begin(2);

    scaledIn[indices] = in[indices] * scale[scaleIndices];
  });

  const MultiArrayShape kernelShape{std::begin(kernel), std::end(kernel)};

  for (unsigned b = 0; b != batchSize; ++b) {
    MultiArray<double> poolOut{poolOutShape};

    auto inShape = in.shape();
    inShape.advance_begin(1);

    MultiArrayShape scaledInIndices;
    MultiArrayShape poolOutIndices;
    forEachIndex(inShape, [&](const MultiArrayShapeRange indices) {
      // the index into the scaledIn array is [b][c][fields...]
      scaledInIndices.clear();
      scaledInIndices.push_back(b);
      scaledInIndices.insert(std::end(scaledInIndices), std::begin(indices),
                             std::end(indices));

      forEachIndex(kernelShape, [&](const MultiArrayShapeRange kernelIndices) {
        // the index into the poolOut array is [c][y + ky + stride...]
        poolOutIndices.clear();
        poolOutIndices.push_back(indices[0]);

        for (unsigned i = 0; i < kernelIndices.size(); ++i) {
          const auto dim = indices[i + 1] * stride[i] + kernelIndices[i];
          if (dim >= poolOutShape[i + 1]) {
            return;
          }

          poolOutIndices.push_back(dim);
        }

        poolOut[poolOutIndices] += scaledIn[scaledInIndices];
      });
    });

    // Truncate and scale
    auto outShape = out.shape();
    outShape.advance_begin(1);

    MultiArrayShape outIndices;
    forEachIndex(outShape, [&](const MultiArrayShapeRange indices) {
      // the index into the out array is [b][c][fields...]
      outIndices.clear();
      outIndices.push_back(b);
      outIndices.insert(std::end(outIndices), std::begin(indices),
                        std::end(indices));

      // the index into the poolOut array is [c][fields + padding...]
      poolOutIndices.clear();
      poolOutIndices.push_back(indices[0]);

      for (unsigned i = 1; i < indices.size(); ++i) {
        const int dim = indices[i] + paddingLower[i - 1];
        if (dim < 0 || dim >= static_cast<int>(poolOutShape[i])) {
          return;
        }

        poolOutIndices.push_back(dim);
      }

      out[outIndices] = poolOut[poolOutIndices];
    });
  }
}

void poplibs_test::pooling::poolingBackward(
    popnn::PoolingType pType, bool useScaledGradForMaxPool,
    const std::vector<unsigned> &stride, const std::vector<std::size_t> &kernel,
    const std::vector<int> &paddingLower, const std::vector<int> &paddingUpper,
    const MultiArray<double> &prevAct, const MultiArray<double> &nextAct,
    const MultiArray<double> &in, MultiArray<double> &out) {
  if (pType == PoolingType::MAX) {
    maxPoolingBackward(stride, kernel, paddingLower, paddingUpper, prevAct,
                       nextAct, in, out, useScaledGradForMaxPool);
  } else if (pType == PoolingType::AVG || pType == PoolingType::SUM) {
    sumPoolingBackward(pType, stride, kernel, paddingLower, paddingUpper,
                       prevAct, nextAct, in, out);
  }
}
