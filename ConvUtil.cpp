#include "ConvUtil.hpp"
#include <cassert>

unsigned
getInputIndex(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize, unsigned kernelIndex,
              bool forward) {
  if (forward) {
    const auto start = static_cast<int>(outputIndex * stride) -
                       static_cast<int>(padding);
    auto inputIndex = start + static_cast<int>(kernelIndex);
    if (inputIndex < 0 || inputIndex >= inputSize)
      return ~0U;
    return inputIndex;
  } else {
    int adjusted = static_cast<int>(outputIndex + padding) -
                static_cast<int>(kernelIndex);
    if (adjusted < 0 || adjusted % stride != 0)
      return ~0U;
    auto inputIndex = (static_cast<unsigned>(adjusted) / stride);
    if (inputIndex >= inputSize)
      return ~0U;
    return inputIndex;
  }
}

std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned padding,
               unsigned inputSize, unsigned kernelIndex,
               bool forward) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  unsigned outputBegin = 0, outputEnd = 0;
  for (unsigned i = outputRange.first; i != outputRange.second; ++i) {
    if (getInputIndex(i, stride, kernelSize, padding,
                      inputSize, kernelIndex, forward) == ~0U) {
      continue;
    }
    outputBegin = i;
    break;
  }
  for (unsigned i = outputRange.second; i != outputRange.first; --i) {
    if (getInputIndex(i - 1, stride, kernelSize, padding, inputSize,
                      kernelIndex, forward) == ~0U) {
      continue;
    }
    outputEnd = i;
    break;
  }
  return {outputBegin, outputEnd};
}

std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding,
              unsigned inputSize, unsigned kernelIndex,
              bool forward) {
  auto truncatedOutputRange =
    getOutputRange(outputRange, stride, kernelSize, padding, inputSize,
                   kernelIndex, forward);
  if (truncatedOutputRange.first == truncatedOutputRange.second) {
    return {0, 0};
  }
  return {
    getInputIndex(truncatedOutputRange.first, stride, kernelSize,
                  padding, inputSize, kernelIndex, forward),
    getInputIndex(truncatedOutputRange.second - 1, stride, kernelSize,
                  padding, inputSize, kernelIndex, forward) + 1
  };
}

std::pair<unsigned, unsigned>
getInputRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize, bool forward) {
  if (forward) {
    int begin = static_cast<int>(outputIndex * stride) -
        static_cast<int>(padding);
    unsigned underflow = 0;
    if (begin < 0) {
      underflow = -begin;
      begin = 0;
    }
    assert(underflow <= kernelSize);
    unsigned end = std::min(begin + kernelSize - underflow, inputSize);
    return {begin, end};
  } else {
    unsigned end = outputIndex + padding + 1;
    unsigned begin =
        std::max(static_cast<int>(end) - static_cast<int>(kernelSize), 0);
    begin = begin / stride;
    end = (end + stride - 1) / stride;
    end = std::min(end, inputSize);
    return {begin, end};
  }
}

std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding, unsigned inputSize,
              bool forward) {

  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  const auto begin =
      getInputRange(outputRange.first, stride, kernelSize,
                    padding, inputSize, forward).first;
  const auto end =
      getInputRange(outputRange.second - 1, stride, kernelSize, padding,
                    inputSize, forward).second;
  return {begin, end};
}

std::pair<unsigned, unsigned>
getKernelRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
               unsigned padding, unsigned inputSize, bool forward) {
  if (!forward)
    assert(0 && "non implemented");
  int begin = static_cast<int>(outputIndex * stride) -
              static_cast<int>(padding);
  unsigned inputBegin, inputEnd;
  std::tie(inputBegin, inputEnd) = getInputRange(outputIndex, stride,
                                                 kernelSize, padding,
                                                 inputSize, forward);
  const auto kernelBegin = inputBegin - begin;
  const auto kernelEnd = inputEnd - begin;
  return { kernelBegin, kernelEnd };
}

std::pair<unsigned, unsigned>
getKernelRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned padding,
               unsigned inputSize, bool forward) {
  if (!forward)
    assert(0 && "non implemented");
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  const auto begin =
      getKernelRange(outputRange.first, stride, kernelSize,
                     padding, inputSize, forward).first;
  const auto end =
      getKernelRange(outputRange.second - 1, stride, kernelSize, padding,
                     inputSize, forward).second;
  return {begin, end};
}

std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(unsigned numConvolutions, unsigned convSize,
                             unsigned numContexts, unsigned stride) {
  std::vector<std::vector<PartialRow>> partitionByWorker;
  partitionByWorker.reserve(numContexts);
  const auto elementsPerRow = convSize / stride;
  const auto activeRows = numConvolutions / stride;
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
        endIndex = 1 + ((endElement- 1) % elementsPerRow) * stride;
      } else {
        endIndex = elementsPerRow * stride;
      }
      unsigned rowIndex = j * stride;
      partitionByWorker.back().emplace_back(rowIndex, beginIndex, endIndex);
    }
  }
  return partitionByWorker;
}
