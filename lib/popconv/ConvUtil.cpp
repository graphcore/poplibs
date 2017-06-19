#include "popconv/ConvUtil.hpp"
#include <cassert>
#include <popstd/exceptions.hpp>

namespace popconv {

/// Given an index in a volume return the corresponding index the volume after
/// applying the specified dilation and padding.
static unsigned
applyDilationAndPadding(unsigned index, unsigned dilation, int paddingLower) {
  return index * dilation + paddingLower;
}

/// Given a index in a dilated and padded volume return the index in the
/// original volume. Return ~0U if the index doesn't correspond to any
/// index in the original volume.
static unsigned
reverseDilationAndPadding(unsigned dilatedPaddedIndex, unsigned inputSize,
                          unsigned dilation, int paddingLower) {
  if (inputSize == 0)
    return ~0U;
  int dilatedSize = (inputSize - 1) * dilation + 1;
  int dilatedIndex = static_cast<int>(dilatedPaddedIndex) - paddingLower;
  if (dilatedIndex < 0 ||
      dilatedIndex >= dilatedSize)
    return ~0U;
  if (dilatedIndex % dilation != 0)
    return ~0U;
  return dilatedIndex / dilation;
}

unsigned
getInputIndex(unsigned dim, unsigned outputIndex, unsigned kernelIndex,
              const ConvParams &params) {
  assert(outputIndex < params.getOutputShape()[dim + 1]);
  const auto paddedKernelIndex =
      applyDilationAndPadding(kernelIndex, params.kernelDilation[dim],
                              params.kernelPaddingLower[dim]);
  const auto upsampledOutputIndex = outputIndex * params.stride[dim];
  const auto paddedKernelSize = params.getPaddedDilatedKernelSize(dim);
  const auto paddedInputSize = params.getPaddedDilatedInputSize(dim);
  int paddedInputIndex;
  if (paddedKernelSize > paddedInputSize) {
    paddedInputIndex = paddedKernelIndex - upsampledOutputIndex;
    if (paddedInputIndex < 0 || paddedInputIndex >= paddedInputSize)
      return ~0U;
  } else {
    paddedInputIndex = paddedKernelIndex + upsampledOutputIndex;
  }
  return reverseDilationAndPadding(paddedInputIndex, params.inputShape[dim + 1],
                                   params.inputDilation[dim],
                                   params.inputPaddingLower[dim]);
}

std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              unsigned kernelIndex, const ConvParams &params) {
  unsigned inputBegin = 0, inputEnd = 0;
  auto trimmedOutputRange = getOutputRange(dim, outputRange, kernelIndex,
                                           params);
  if (trimmedOutputRange.first != trimmedOutputRange.second) {
    if (params.getPaddedDilatedKernelSize(dim) <
        params.getPaddedDilatedInputSize(dim)) {
      inputBegin = getInputIndex(dim, trimmedOutputRange.first, kernelIndex,
                                 params);
      inputEnd = getInputIndex(dim, trimmedOutputRange.second - 1, kernelIndex,
                               params) + 1;
    } else {
      inputBegin = getInputIndex(dim, trimmedOutputRange.second - 1,
                                 kernelIndex, params);
      inputEnd = getInputIndex(dim, trimmedOutputRange.first, kernelIndex,
                               params) + 1;
    }
  }
  assert(inputBegin <= inputEnd);
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
partitionConvPartialByWorker(unsigned convHeight, unsigned convWidth,
                             unsigned numContexts,
                             const std::vector<unsigned> &inputDilation) {
  std::vector<std::vector<PartialRow>> partitionByWorker;
  partitionByWorker.reserve(numContexts);
  const auto elementsPerRow =
      (convWidth + inputDilation[1] - 1) / inputDilation[1];
  const auto activeRows =
      (convHeight + inputDilation[0] - 1) / inputDilation[0];
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
        beginIndex = (beginElement % elementsPerRow * inputDilation[1]);
      } else {
        beginIndex = 0;
      }
      unsigned endIndex;
      if (j + 1 == endRow) {
        endIndex = 1 + ((endElement - 1) % elementsPerRow) * inputDilation[1];
      } else {
        endIndex = ((elementsPerRow - 1) * inputDilation[1]) + 1;
      }
      unsigned rowIndex = j * inputDilation[0];
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
  std::vector<int> bwdInputPaddingLower, bwdInputPaddingUpper;
  std::vector<unsigned> bwdStride, bwdInputDilation;
  bwdStride = params.inputDilation;
  bwdInputDilation = params.stride;
  for (const auto dim : {0, 1}) {
    const auto kernelSize = params.getPaddedDilatedKernelSize(dim);
    const auto inputPaddingLower = params.inputPaddingLower[dim];
    const auto inputPaddingUpper = params.inputPaddingUpper[dim];
    bwdInputPaddingLower.push_back(
      static_cast<int>(kernelSize) - 1 - inputPaddingLower
    );
    auto paddedInputSize =
        params.inputShape[1 + dim] + inputPaddingLower + inputPaddingUpper;
    int inputSizeIgnored =
        (paddedInputSize - kernelSize) % params.stride[dim];
    bwdInputPaddingUpper.push_back(
      static_cast<int>(kernelSize) - 1 - inputPaddingUpper + inputSizeIgnored
    );
  }
  auto bwdKernelShape = params.kernelShape;
  std::swap(bwdKernelShape[2], bwdKernelShape[3]);
  // Going backwards the weights are flipped in each axis and so we must flip
  // the upper and lower padding.
  auto bwdKernelPaddingLower = params.kernelPaddingUpper;
  auto bwdKernelPaddingUpper = params.kernelPaddingLower;
  return popconv::ConvParams(params.dType, params.getOutputShape(),
                             bwdKernelShape, bwdStride, bwdInputPaddingLower,
                             bwdInputPaddingUpper, bwdInputDilation,
                             bwdKernelPaddingLower, bwdKernelPaddingUpper,
                             params.kernelDilation);
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
