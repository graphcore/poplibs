#include <ConvUtil.hpp>
#include <cassert>

unsigned
getInputIndex(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize, unsigned kernelIndex) {
  const auto start  = static_cast<int>(outputIndex * stride) -
                      static_cast<int>(padding);
  auto inputIndex = start + static_cast<int>(kernelIndex);
  if (inputIndex < 0 ||
      static_cast<unsigned>(inputIndex) >= inputSize)
    return ~0U;
  return inputIndex;
}

std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned padding,
               unsigned inputSize, unsigned kernelIndex) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  unsigned outputBegin = 0, outputEnd = 0;
  for (unsigned i = outputRange.first; i != outputRange.second; ++i) {
    if (getInputIndex(i, stride, kernelSize, padding,
                      inputSize, kernelIndex) == ~0U) {
      continue;
    }
    outputBegin = i;
    break;
  }
  for (unsigned i = outputRange.second; i != outputRange.first; --i) {
    if (getInputIndex(i - 1, stride, kernelSize, padding, inputSize,
                      kernelIndex) == ~0U) {
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
              unsigned inputSize, unsigned kernelIndex) {
  auto truncatedOutputRange =
    getOutputRange(outputRange, stride, kernelSize, padding, inputSize,
                   kernelIndex);
  if (truncatedOutputRange.first == truncatedOutputRange.second) {
    return {0, 0};
  }
  return {
    getInputIndex(truncatedOutputRange.first, stride, kernelSize,
                  padding, inputSize, kernelIndex),
    getInputIndex(truncatedOutputRange.second - 1, stride, kernelSize,
                  padding, inputSize, kernelIndex) + 1
  };
}

std::pair<unsigned, unsigned>
getInputRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize) {
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
}

std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding, unsigned inputSize) {

  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  const auto begin =
      getInputRange(outputRange.first, stride, kernelSize,
                    padding, inputSize).first;
  const auto end =
      getInputRange(outputRange.second - 1, stride, kernelSize, padding,
                    inputSize).second;
  return {begin, end};
}

std::pair<unsigned, unsigned>
getKernelRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
               unsigned padding, unsigned inputSize) {
  int begin = static_cast<int>(outputIndex * stride) -
              static_cast<int>(padding);
  unsigned inputBegin, inputEnd;
  std::tie(inputBegin, inputEnd) = getInputRange(outputIndex, stride,
                                                 kernelSize, padding,
                                                 inputSize);
  const auto kernelBegin = inputBegin - begin;
  const auto kernelEnd = inputEnd - begin;
  return { kernelBegin, kernelEnd };
}

std::pair<unsigned, unsigned>
getKernelRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned padding,
               unsigned inputSize) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  const auto begin =
      getKernelRange(outputRange.first, stride, kernelSize,
                     padding, inputSize).first;
  const auto end =
      getKernelRange(outputRange.second - 1, stride, kernelSize, padding,
                     inputSize).second;
  return {begin, end};
}

std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(
    unsigned numConvolutions,
    unsigned convSize,
    unsigned numContexts) {
  std::vector<std::vector<PartialRow>> partitionByWorker;
  partitionByWorker.reserve(numContexts);
  const auto numElements = numConvolutions * convSize;
  for (unsigned i = 0; i != numContexts; ++i) {
    partitionByWorker.emplace_back();
    const auto beginElement = (i * numElements) / numContexts;
    const auto endElement = ((i + 1) * numElements) / numContexts;
    if (beginElement == endElement)
      continue;
    const auto beginRow = beginElement / convSize;
    const auto endRow = 1 + (endElement - 1) / convSize;
    for (unsigned j = beginRow; j != endRow; ++j) {
      unsigned beginIndex = j == beginRow ? beginElement % convSize :
                                            0;
      unsigned endIndex = j + 1 == endRow ? 1 + (endElement - 1) % convSize :
                                            convSize;
      partitionByWorker.back().emplace_back(j, beginIndex, endIndex);
    }
  }
  return partitionByWorker;
}
