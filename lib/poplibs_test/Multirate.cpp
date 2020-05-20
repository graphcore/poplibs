// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <iostream>
#include <poplibs_test/Multirate.hpp>
#include <poplibs_test/Pooling.hpp>

using popnn::PoolingType;

using namespace poplibs_support;

void poplibs_test::upsample(const std::vector<std::size_t> &inFieldShape,
                            const unsigned samplingRate,
                            const boost::multi_array<double, 3> &input,
                            boost::multi_array<double, 3> &output) {
  auto numFieldDims = inFieldShape.size();
  std::vector<std::size_t> inCumDimLen(numFieldDims);
  std::vector<std::size_t> outCumDimLen(numFieldDims);
  auto outShape = output.shape();
  inCumDimLen[numFieldDims - 1] = 1;
  outCumDimLen[numFieldDims - 1] = 1;
  std::size_t cumProd = inFieldShape[numFieldDims - 1];
  for (int i = numFieldDims - 2; i >= 0; --i) {
    inCumDimLen[i] = cumProd;
    outCumDimLen[i] = cumProd * std::pow(samplingRate, numFieldDims - i - 1);
    cumProd *= inFieldShape[i];
  }
  // Iterate over flattened field dimension
  for (unsigned outIndex = 0; outIndex < outShape[2]; ++outIndex) {
    unsigned inIndex = 0;
    unsigned remainder = outIndex;
    for (unsigned j = 0; j < numFieldDims; ++j) {
      auto outDimIndex = remainder / outCumDimLen[j];
      remainder %= outCumDimLen[j];
      inIndex += (outDimIndex / samplingRate) * inCumDimLen[j];
    }
    for (unsigned i = 0; i < outShape[0]; ++i) {
      for (unsigned j = 0; j < outShape[1]; ++j) {
        output[i][j][outIndex] = input[i][j][inIndex];
      }
    }
  }
}

void poplibs_test::downsample(const std::vector<std::size_t> &outFieldShape,
                              const unsigned samplingRate,
                              const boost::multi_array<double, 3> &input,
                              boost::multi_array<double, 3> &output) {
  popnn::PoolingType poolingType = popnn::PoolingType::AVG;
  std::vector<std::size_t> kernelShape;
  std::vector<unsigned> stride;
  std::vector<int> paddingLower;
  std::vector<int> paddingUpper;
  for (unsigned i = 0; i < outFieldShape.size(); ++i) {
    kernelShape.push_back(samplingRate);
    stride.push_back(samplingRate);
    paddingLower.push_back(0);
    paddingUpper.push_back(0);
  }
  std::vector<std::size_t> inFieldShape(outFieldShape);
  std::for_each(inFieldShape.begin(), inFieldShape.end(),
                [samplingRate](std::size_t &v) { v *= samplingRate; });
  MultiArrayShape inputShape = {input.shape()[0], input.shape()[1]};
  inputShape.insert(inputShape.end(), inFieldShape.begin(), inFieldShape.end());
  MultiArrayShape outputShape = {output.shape()[0], output.shape()[1]};
  outputShape.insert(outputShape.end(), outFieldShape.begin(),
                     outFieldShape.end());
  MultiArray<double> inputMultiArray{inputShape};
  MultiArray<double> outputMultiArray{outputShape};
  std::memcpy(inputMultiArray.data(), input.data(),
              inputMultiArray.numElements() * sizeof(double));
  poplibs_test::pooling::pooling(poolingType, stride, kernelShape, paddingLower,
                                 paddingUpper, inputMultiArray,
                                 outputMultiArray);
  std::memcpy(output.data(), outputMultiArray.data(),
              outputMultiArray.numElements() * sizeof(double));
}
