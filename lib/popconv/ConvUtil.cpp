#include "popconv/ConvUtil.hpp"
#include <cassert>

namespace popconv {

unsigned
getInputIndex(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned paddingLower, unsigned paddingUpper, unsigned inputSize,
              unsigned kernelIndex, bool isFractionallyStrided) {
  if (isFractionallyStrided) {
    // Stride represents a upsampling of the input.
    // Padding represents a truncation of the output.
    int adjusted = static_cast<int>(outputIndex + paddingLower) -
                   static_cast<int>(kernelSize - 1 - kernelIndex);
    if (adjusted < 0 || adjusted % stride != 0)
      return ~0U;
    auto inputIndex = (static_cast<unsigned>(adjusted) / stride);
    if (inputIndex >= inputSize)
      return ~0U;
    return inputIndex;
  }
  const auto upsampledOutputIndex = outputIndex * stride;
  const auto paddedInputSize = inputSize + paddingLower + paddingUpper;
  unsigned paddedInputIndex;
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
  if (inputIndex >= inputSize)
    return ~0U;
  return inputIndex;
}

std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned paddingLower, unsigned paddingUpper,
              unsigned inputSize, unsigned kernelIndex,
              bool isFractionallyStrided) {

  unsigned inputBegin = 0, inputEnd = 0;
  for (unsigned x = outputRange.first; x != outputRange.second; ++x) {
    auto inputIndex = getInputIndex(x, stride, kernelSize, paddingLower,
                                    paddingUpper, inputSize, kernelIndex,
                                    isFractionallyStrided);
    if (inputIndex != ~0U) {
      inputBegin = inputIndex;
      break;
    }
  }
  for (unsigned x = outputRange.second; x != outputRange.first; --x) {
    auto inputIndex = getInputIndex(x - 1, stride, kernelSize, paddingLower,
                                    paddingUpper, inputSize, kernelIndex,
                                    isFractionallyStrided);
    if (inputIndex != ~0U) {
      inputEnd = inputIndex + 1;
      break;
    }
  }
  return {inputBegin, inputEnd};
}

std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned paddingLower, unsigned paddingUpper,
              unsigned inputSize, std::pair<unsigned, unsigned> kernelRange,
              bool isFractionallyStrided) {
  unsigned inputBegin = 0, inputEnd = 0;
  for (unsigned k = kernelRange.first; k != kernelRange.second; ++k) {
    auto inputRange = getInputRange(outputRange, stride, kernelSize,
                                    paddingLower, paddingUpper, inputSize, k,
                                    isFractionallyStrided);
    if (inputRange.first != inputRange.second) {
      inputBegin = inputRange.first;
      break;
    }
  }
  for (unsigned k = kernelRange.second; k != kernelRange.first; --k) {
    auto inputRange = getInputRange(outputRange, stride, kernelSize,
                                    paddingLower, paddingUpper, inputSize,
                                    k - 1, isFractionallyStrided);
    if (inputRange.first != inputRange.second) {
      inputEnd = inputRange.second;
      break;
    }
  }
  return {inputBegin, inputEnd};
}

std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned paddingLower,
               unsigned paddingUpper, unsigned inputSize, unsigned kernelIndex,
               bool isFractionallyStrided) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  unsigned outputBegin = 0, outputEnd = 0;
  for (unsigned i = outputRange.first; i != outputRange.second; ++i) {
    if (getInputIndex(i, stride, kernelSize, paddingLower, paddingUpper,
                      inputSize, kernelIndex, isFractionallyStrided) == ~0U) {
      continue;
    }
    outputBegin = i;
    break;
  }
  for (unsigned i = outputRange.second; i != outputRange.first; --i) {
    if (getInputIndex(i - 1, stride, kernelSize, paddingLower, paddingUpper,
                      inputSize, kernelIndex, isFractionallyStrided) == ~0U) {
      continue;
    }
    outputEnd = i;
    break;
  }
  return {outputBegin, outputEnd};
}

std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned paddingLower,
               unsigned paddingUpper, unsigned inputSize,
               std::pair<unsigned, unsigned> kernelIndexRange,
               bool isFractionallyStrided) {
  assert(kernelIndexRange.second >= kernelIndexRange.first);
  unsigned outputBegin = 0, outputEnd = 0;
  bool first = true;
  for (unsigned kernelIndex = kernelIndexRange.first;
       kernelIndex != kernelIndexRange.second; ++kernelIndex) {
    const auto trimmedOutputRange =
        getOutputRange(outputRange, stride, kernelSize, paddingLower,
                       paddingUpper, inputSize, kernelIndex,
                       isFractionallyStrided);
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

std::pair<unsigned, unsigned>
getKernelRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
               unsigned paddingLower, unsigned paddingUpper, unsigned inputSize,
               bool isFractionallyStrided) {
  if (isFractionallyStrided)
    assert(0 && "non implemented");
  unsigned kernelBegin = 0, kernelEnd = 0;
  for (unsigned i = 0; i != kernelSize; ++i) {
    if (getInputIndex(outputIndex, stride, kernelSize, paddingLower,
                      paddingUpper, inputSize, i,
                      isFractionallyStrided) == ~0U) {
      continue;
    }
    kernelBegin = i;
    break;
  }
  for (unsigned i = kernelSize; i != 0; --i) {
    if (getInputIndex(outputIndex, stride, kernelSize, paddingLower,
                      paddingUpper, inputSize, i - 1,
                      isFractionallyStrided) == ~0U) {
      continue;
    }
    kernelEnd = i;
    break;
  }
  return {kernelBegin, kernelEnd};
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

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX,
             unsigned kernelSizeY, unsigned kernelSizeX,
             const std::vector<unsigned> &stride,
             const std::vector<unsigned> &paddingLower,
             const std::vector<unsigned> &paddingUpper,
             bool isFractional) {
  if (isFractional) {
    unsigned outDimX =
        (inDimX * stride[1] + kernelSizeX - 1) -
        (paddingLower[1] + paddingUpper[1]);
    unsigned outDimY =
        (inDimY * stride[0] + kernelSizeY - 1) -
        (paddingLower[0] + paddingUpper[1]);
    return {outDimY, outDimX};
  } else {
    unsigned outDimX =
        absdiff(inDimX + (paddingLower[1] + paddingUpper[1]), kernelSizeX) /
        stride[1] + 1;
    unsigned outDimY =
        absdiff(inDimY + (paddingLower[0] + paddingUpper[0]), kernelSizeY) /
        stride[0] + 1;
    return {outDimY, outDimX};
  }
}

} // namespace convutil
