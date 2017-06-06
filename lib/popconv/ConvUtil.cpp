#include "popconv/ConvUtil.hpp"
#include <cassert>
#include <popstd/exceptions.hpp>

namespace popconv {

unsigned
getInputIndex(unsigned dim, unsigned outputIndex,
              unsigned kernelIndex, const ConvParams &params) {
  const auto paddingLower = params.paddingLower[dim];
  const auto paddingUpper = params.paddingUpper[dim];
  const auto kernelSize = params.kernelShape[dim];
  const auto stride = params.stride[dim];
  const auto inputSize = params.inputShape[dim + 1];
  const auto inputDilation = params.inputDilation[dim];
  const auto upsampledInputSize = (inputSize - 1) * inputDilation + 1;
  const auto upsampledOutputIndex = outputIndex * stride;
  const auto paddedInputSize = upsampledInputSize + paddingLower + paddingUpper;
  int paddedInputIndex;
  if (kernelSize > paddedInputSize) {
    if (kernelIndex < upsampledOutputIndex) {
      return ~0U;
    }
    paddedInputIndex = kernelIndex - upsampledOutputIndex;
  } else {
    paddedInputIndex = kernelIndex + upsampledOutputIndex;
  }
  if (paddedInputIndex < paddingLower)
    return ~0U;
  const auto inputIndex = paddedInputIndex - paddingLower;
  if (inputIndex >= upsampledInputSize)
    return ~0U;
  if (inputIndex % inputDilation != 0)
    return ~0U;
  return inputIndex / inputDilation;
}

std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              unsigned kernelIndex, const ConvParams &params) {

  unsigned inputBegin = 0, inputEnd = 0;
  for (unsigned x = outputRange.first; x != outputRange.second; ++x) {
    auto inputIndex = getInputIndex(dim, x, kernelIndex, params);
    if (inputIndex != ~0U) {
      inputBegin = inputIndex;
      break;
    }
  }
  for (unsigned x = outputRange.second; x != outputRange.first; --x) {
    auto inputIndex = getInputIndex(dim, x - 1, kernelIndex, params);
    if (inputIndex != ~0U) {
      inputEnd = inputIndex + 1;
      break;
    }
  }
  return {inputBegin, inputEnd};
}

std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              std::pair<unsigned, unsigned> kernelRange,
              const ConvParams &params) {
  unsigned inputBegin = 0, inputEnd = 0;
  for (unsigned k = kernelRange.first; k != kernelRange.second; ++k) {
    auto inputRange = getInputRange(dim, outputRange, k, params);
    if (inputRange.first != inputRange.second) {
      inputBegin = inputRange.first;
      break;
    }
  }
  for (unsigned k = kernelRange.second; k != kernelRange.first; --k) {
    auto inputRange = getInputRange(dim, outputRange, k - 1, params);
    if (inputRange.first != inputRange.second) {
      inputEnd = inputRange.second;
      break;
    }
  }
  return {inputBegin, inputEnd};
}

std::pair<unsigned, unsigned>
getOutputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               unsigned kernelIndex, const ConvParams &params) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  unsigned outputBegin = 0, outputEnd = 0;
  for (unsigned i = outputRange.first; i != outputRange.second; ++i) {
    if (getInputIndex(dim, i, kernelIndex, params) == ~0U) {
      continue;
    }
    outputBegin = i;
    break;
  }
  for (unsigned i = outputRange.second; i != outputRange.first; --i) {
    if (getInputIndex(dim, i - 1, kernelIndex, params) == ~0U) {
      continue;
    }
    outputEnd = i;
    break;
  }
  return {outputBegin, outputEnd};
}

std::pair<unsigned, unsigned>
getOutputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               std::pair<unsigned, unsigned> kernelIndexRange,
               const ConvParams &params) {
  assert(kernelIndexRange.second >= kernelIndexRange.first);
  unsigned outputBegin = 0, outputEnd = 0;
  bool first = true;
  for (unsigned kernelIndex = kernelIndexRange.first;
       kernelIndex != kernelIndexRange.second; ++kernelIndex) {
    const auto trimmedOutputRange =
        getOutputRange(dim, outputRange, kernelIndex, params);
    if (trimmedOutputRange.first != trimmedOutputRange.second) {
      if (first) {
        outputBegin = trimmedOutputRange.first;
        outputEnd = trimmedOutputRange.second;
        first = false;
      } else {
        outputBegin = std::min(outputBegin, trimmedOutputRange.first);
        outputEnd = std::max(outputEnd, trimmedOutputRange.second);
      }
    }
  }
  return {outputBegin, outputEnd};
}

std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(unsigned numConvolutions, unsigned convSize,
                             unsigned numContexts, unsigned stride) {
  std::vector<std::vector<PartialRow>> partitionByWorker;
  partitionByWorker.reserve(numContexts);
  const auto elementsPerRow = (convSize + stride - 1) / stride;
  const auto activeRows = (numConvolutions + stride - 1) / stride;
  const auto numElements = activeRows * elementsPerRow;
  for (unsigned i = 0; i != numContexts; ++i) {
    partitionByWorker.emplace_back();
    const auto beginElement = (i * numElements) / numContexts;
    const auto endElement = ((i + 1) * numElements) / numContexts;
    if (beginElement == endElement)
      continue;
    const auto beginRow = beginElement / elementsPerRow;
    const auto endRow = 1 + (endElement - 1) / elementsPerRow;
    for (unsigned j = beginRow; j != endRow; ++j) {
      unsigned beginIndex;
      if (j == beginRow) {
        beginIndex = (beginElement % elementsPerRow * stride);
      } else {
        beginIndex = 0;
      }
      unsigned endIndex;
      if (j + 1 == endRow) {
        endIndex = 1 + ((endElement - 1) % elementsPerRow) * stride;
      } else {
        endIndex = ((elementsPerRow - 1) * stride) + 1;
      }
      unsigned rowIndex = j * stride;
      partitionByWorker.back().emplace_back(rowIndex, beginIndex, endIndex);
    }
  }
  return partitionByWorker;
}

std::vector<std::size_t>
getOutputShape(const ConvParams &params) {
  return {params.getBatchSize(), params.getOutputHeight(),
          params.getOutputWidth(), params.getOutputDepth()};
}

ConvParams getGradientParams(const ConvParams &params) {
  std::vector<int> bwdPaddingLower, bwdPaddingUpper;
  std::vector<unsigned> bwdStride, bwdInputDilation;
  bwdStride = params.inputDilation;
  bwdInputDilation = params.stride;
  for (const auto dim : {0, 1}) {
    const auto kernelSize = params.kernelShape[dim];
    const auto paddingLower = params.paddingLower[dim];
    const auto paddingUpper = params.paddingUpper[dim];
    bwdPaddingLower.push_back(
      static_cast<int>(kernelSize) - 1 - paddingLower
    );
    auto paddedInputSize =
        params.inputShape[1 + dim] + paddingLower + paddingUpper;
    int inputSizeIgnored =
        (paddedInputSize - kernelSize) % params.stride[dim];
    bwdPaddingUpper.push_back(
      static_cast<int>(kernelSize) - 1 - paddingUpper + inputSizeIgnored
    );
  }
  auto bwdKernelShape = params.kernelShape;
  std::swap(bwdKernelShape[2], bwdKernelShape[3]);
  return popconv::ConvParams(params.dType, params.getOutputShape(),
                             bwdKernelShape, bwdStride, bwdPaddingLower,
                             bwdPaddingUpper, bwdInputDilation);
}

unsigned detectChannelGrouping(const poplar::Tensor &t0) {
  if (t0.rank() == 0)
    throw popstd::poplib_error("Cannot detect channel grouping of "
                               "0-rank tensor");
  // Sample the first point in the inner dimension
  auto t = t0;
  while (t.rank() != 1)
    t = t[0];

  // Perform a binary search to find the largest contiguous slice in
  // the inner dimension.
  auto lower = 1U;
  auto upper = t.numElements();
  while (lower != upper) {
    // Find a mid-point such that lower < mid <= upper
    auto mid = upper - (upper - lower) / 2;
    if (t.slice(0, mid).isContiguous()) {
      lower = mid;
    } else {
      upper = mid - 1;
    }
  }

  // The channel grouping must divide the number of channels
  if (t.numElements() % upper != 0)
    upper = 1;
  return upper;
}

} // namespace convutil
