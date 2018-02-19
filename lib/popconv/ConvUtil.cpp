#include "popconv/ConvUtil.hpp"

#include <boost/range/irange.hpp>
#include <cassert>
#include <poputil/exceptions.hpp>
#include <poputil/Util.hpp>
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/VectorUtils.hpp"

namespace popconv {

unsigned getDilatedSize(unsigned size, unsigned dilation) {
  if (size == 0)
    return 0;
  return 1 + (size - 1) * dilation;
}

/// Given an index in a volume return the corresponding index the volume after
/// applying the specified truncation, dilation, padding and flipping. Return
/// ~0U if truncation means the element is ignored.
static unsigned
applyTruncateDilatePadAndFlip(unsigned index,
                              unsigned inputSize,
                              unsigned truncationLower,
                              unsigned truncationUpper,
                              unsigned dilation,
                              unsigned paddingLower,
                              unsigned paddingUpper,
                              bool flip) {
  assert(index < inputSize);
  if (index < truncationLower || index >= inputSize - truncationUpper)
    return ~0U;
  const auto truncatedIndex = index - truncationLower;
  const auto truncatedSize = inputSize - (truncationLower + truncationUpper);
  const auto truncatedDilatedIndex = truncatedIndex * dilation;
  const auto truncatedDilatedSize = getDilatedSize(truncatedSize, dilation);
  const auto truncatedDilatedPaddedIndex = truncatedDilatedIndex + paddingLower;
  const auto truncatedDilatedPaddedSize = paddingLower + truncatedDilatedSize +
                                          paddingUpper;
  return flip ? truncatedDilatedPaddedSize - 1 - truncatedDilatedPaddedIndex :
                truncatedDilatedPaddedIndex;
}

/// Given a index in a dilated and padded volume return the index in the
/// original volume. Return ~0U if the index doesn't correspond to any
/// index in the original volume.
static unsigned
reverseTruncateDilatePadAndFlip(unsigned truncatedDilatedPaddedFlippedIndex,
                                unsigned inputSize,
                                unsigned truncationLower,
                                unsigned truncationUpper,
                                unsigned dilation,
                                unsigned paddingLower,
                                unsigned paddingUpper,
                                bool flip) {
  const auto truncatedSize = inputSize - (truncationLower + truncationUpper);
  const auto truncatedDilatedSize = getDilatedSize(truncatedSize, dilation);
  const auto truncatedDilatedPaddedSize = paddingLower + truncatedDilatedSize +
                                          paddingUpper;
  assert(truncatedDilatedPaddedFlippedIndex < truncatedDilatedPaddedSize);
  const auto truncatedDilatedPaddedIndex =
      flip ? truncatedDilatedPaddedSize - 1 -
             truncatedDilatedPaddedFlippedIndex:
             truncatedDilatedPaddedFlippedIndex;
  if (truncatedDilatedPaddedIndex < paddingLower ||
      truncatedDilatedPaddedIndex >= truncatedDilatedPaddedSize - paddingUpper)
    return ~0U;
  const auto truncatedDilatedIndex = truncatedDilatedPaddedIndex - paddingLower;
  if (truncatedDilatedIndex % dilation != 0)
    return ~0U;
  const auto truncatedIndex = truncatedDilatedIndex / dilation;
  return truncatedIndex + truncationLower;
}

unsigned
getInputIndex(unsigned dim, unsigned truncatedStridedPaddedOutputIndex,
              unsigned kernelIndex, const ConvParams &params) {
  const auto outputSize = params.getOutputSize(dim);
  assert(truncatedStridedPaddedOutputIndex < outputSize);
  const auto paddedKernelIndex =
      applyTruncateDilatePadAndFlip(kernelIndex,
                                    params.kernelShape[dim],
                                    params.kernelTransform.truncationLower[dim],
                                    params.kernelTransform.truncationUpper[dim],
                                    params.kernelTransform.dilation[dim],
                                    params.kernelTransform.paddingLower[dim],
                                    params.kernelTransform.paddingUpper[dim],
                                    params.kernelTransform.flip[dim]);
  if (paddedKernelIndex == ~0U)
    return ~0U;
  if (truncatedStridedPaddedOutputIndex <
      params.outputTransform.paddingLower[dim] ||
      truncatedStridedPaddedOutputIndex >=
      outputSize - params.outputTransform.paddingUpper[dim])
    return ~0U;
  const auto truncatedStridedOutputIndex =
      truncatedStridedPaddedOutputIndex -
      params.outputTransform.paddingLower[dim];
  const auto truncatedOutputIndex =
      truncatedStridedOutputIndex * params.outputTransform.stride[dim];
  const auto outputIndex = truncatedOutputIndex +
                           params.outputTransform.truncationLower[dim];
  int paddedInputIndex = paddedKernelIndex + outputIndex;
  return reverseTruncateDilatePadAndFlip(
    paddedInputIndex,
    params.inputFieldShape[dim],
    params.inputTransform.truncationLower[dim],
    params.inputTransform.truncationUpper[dim],
    params.inputTransform.dilation[dim],
    params.inputTransform.paddingLower[dim],
    params.inputTransform.paddingUpper[dim],
    params.inputTransform.flip[dim]
  );
}

unsigned
getKernelIndex(unsigned dim, unsigned truncatedStridedPaddedOutputIndex,
               unsigned inputIndex, const ConvParams &params) {
  const auto outputSize = params.getOutputSize(dim);
  assert(truncatedStridedPaddedOutputIndex < outputSize);
  const auto paddedInputIndex =
      applyTruncateDilatePadAndFlip(inputIndex,
                                    params.inputFieldShape[dim],
                                    params.inputTransform.truncationLower[dim],
                                    params.inputTransform.truncationUpper[dim],
                                    params.inputTransform.dilation[dim],
                                    params.inputTransform.paddingLower[dim],
                                    params.inputTransform.paddingUpper[dim],
                                    params.inputTransform.flip[dim]);
  if (paddedInputIndex == ~0U)
    return ~0U;
  if (truncatedStridedPaddedOutputIndex <
        params.outputTransform.paddingLower[dim] ||
      truncatedStridedPaddedOutputIndex >=
        outputSize - params.outputTransform.paddingUpper[dim])
    return ~0U;
  const auto truncatedStridedOutputIndex =
      truncatedStridedPaddedOutputIndex -
      params.outputTransform.paddingLower[dim];
  const auto truncatedOutputIndex =
      truncatedStridedOutputIndex * params.outputTransform.stride[dim];
  const auto outputIndex = truncatedOutputIndex +
                           params.outputTransform.truncationLower[dim];
  if (outputIndex > paddedInputIndex)
    return ~0U;
  const auto paddedKernelIndex = paddedInputIndex - outputIndex;
  if (paddedKernelIndex >= params.getTransformedKernelSize(dim))
    return ~0U;
  return reverseTruncateDilatePadAndFlip(
    paddedKernelIndex,
    params.kernelShape[dim],
    params.kernelTransform.truncationLower[dim],
    params.kernelTransform.truncationUpper[dim],
    params.kernelTransform.dilation[dim],
    params.kernelTransform.paddingLower[dim],
    params.kernelTransform.paddingUpper[dim],
    params.kernelTransform.flip[dim]
  );
}

std::pair<unsigned, unsigned>
getOutputRangeForKernelIndex(unsigned dim,
                             std::pair<unsigned, unsigned> outputRange,
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
getOutputRangeForInputIndex(unsigned dim,
                            std::pair<unsigned, unsigned> outputRange,
                            unsigned inputIndex, const ConvParams &params) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  unsigned outputBegin = 0, outputEnd = 0;
  for (unsigned i = outputRange.first; i != outputRange.second; ++i) {
    if (getKernelIndex(dim, i, inputIndex, params) == ~0U) {
      continue;
    }
    outputBegin = i;
    break;
  }
  for (unsigned i = outputRange.second; i != outputRange.first; --i) {
    if (getKernelIndex(dim, i - 1, inputIndex, params) == ~0U) {
      continue;
    }
    outputEnd = i;
    break;
  }
  return {outputBegin, outputEnd};
}

std::pair<unsigned, unsigned>
getOutputRangeForKernelRange(
    unsigned dim, std::pair<unsigned, unsigned> outputRange,
    std::pair<unsigned, unsigned> kernelIndexRange,
    const ConvParams &params) {
  assert(kernelIndexRange.second >= kernelIndexRange.first);
  unsigned outputBegin = 0, outputEnd = 0;
  bool first = true;
  for (unsigned kernelIndex = kernelIndexRange.first;
       kernelIndex != kernelIndexRange.second; ++kernelIndex) {
    const auto trimmedOutputRange =
        getOutputRangeForKernelIndex(dim, outputRange, kernelIndex, params);
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
getOutputRangeForInputRange(unsigned dim,
                            std::pair<unsigned, unsigned> outputRange,
                            std::pair<unsigned, unsigned> inputRange,
               const ConvParams &params) {
  assert(inputRange.second >= inputRange.first);
  unsigned outputBegin = 0, outputEnd = 0;
  bool first = true;
  for (unsigned inputIndex = inputRange.first;
       inputIndex != inputRange.second; ++inputIndex) {
    const auto trimmedOutputRange =
        getOutputRangeForInputIndex(dim, outputRange, inputIndex, params);
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
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              unsigned kernelIndex, const ConvParams &params) {
  unsigned inputBegin = 0, inputEnd = 0;
  auto trimmedOutputRange = getOutputRangeForKernelIndex(dim, outputRange,
                                                         kernelIndex, params);
  if (trimmedOutputRange.first != trimmedOutputRange.second) {
    inputBegin = getInputIndex(dim, trimmedOutputRange.first, kernelIndex,
                               params);
    auto inputLast = getInputIndex(dim, trimmedOutputRange.second - 1,
                                   kernelIndex, params);
    if (params.inputTransform.flip[dim]) {
      std::swap(inputBegin, inputLast);
    }
    inputEnd = inputLast + 1;
  }
  assert(inputBegin <= inputEnd);
  return {inputBegin, inputEnd};
}

std::pair<unsigned, unsigned>
getKernelRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               unsigned inputIndex, const ConvParams &params) {
  unsigned kernelBegin = 0, kernelEnd = 0;
  auto trimmedOutputRange =
      getOutputRangeForInputIndex(dim, outputRange, inputIndex, params);
  if (trimmedOutputRange.first != trimmedOutputRange.second) {
    kernelBegin = getKernelIndex(dim, trimmedOutputRange.second - 1,
                                 inputIndex, params);
    auto kernelLast = getKernelIndex(dim, trimmedOutputRange.first, inputIndex,
                                     params);
    if (params.kernelTransform.flip[dim]) {
      std::swap(kernelBegin, kernelLast);
    }
    kernelEnd = kernelLast + 1;
  }
  assert(kernelBegin <= kernelEnd);
  return {kernelBegin, kernelEnd};
}

std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              std::pair<unsigned, unsigned> kernelRange,
              const ConvParams &params) {
  assert(kernelRange.second >= kernelRange.first);
  // If the kernel range is large try to narrow it down by calculating the
  // kernel range corresponding to the output range.
  if (kernelRange.second - kernelRange.first > params.inputFieldShape[dim]) {
    auto kernelRangeForOutputRange =
        getKernelRange(dim, outputRange, {0, params.inputFieldShape[dim]},
                       params);
    kernelRange.first = std::max(kernelRange.first,
                                 kernelRangeForOutputRange.first);
    kernelRange.second = std::min(kernelRange.second,
                                  kernelRangeForOutputRange.second);
  }
  if (kernelRange.first >= kernelRange.second)
    return {0, 0};
  unsigned inputEnd = 0;
  unsigned minBegin = 0, maxEnd = params.inputFieldShape[dim];
  const auto kernelRangeFwd =
      boost::irange(static_cast<int>(kernelRange.first),
                    static_cast<int>(kernelRange.second),
                    1);
  const auto kernelRangeBwd =
      boost::irange(static_cast<int>(kernelRange.second) - 1,
                    static_cast<int>(kernelRange.first) - 1,
                    -1);
  bool flip = params.inputTransform.flip[dim];
  for (unsigned k : flip ? kernelRangeFwd : kernelRangeBwd) {
    auto inputRange = getInputRange(dim, outputRange, k, params);
    if (inputRange.first == inputRange.second)
      continue;
    inputEnd = std::max(inputEnd, inputRange.second);
    if (inputEnd == maxEnd)
      break;
  }
  unsigned inputBegin = inputEnd;
  for (unsigned k : flip ? kernelRangeBwd : kernelRangeFwd) {
    auto inputRange = getInputRange(dim, outputRange, k, params);
    if (inputRange.first == inputRange.second)
      continue;
    inputBegin = std::min(inputBegin, inputRange.first);
    if (inputBegin == minBegin)
      break;
  }
  return {inputBegin, inputEnd};
}

std::pair<unsigned, unsigned>
getKernelRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               std::pair<unsigned, unsigned> inputRange,
               const ConvParams &params) {
  assert(inputRange.second >= inputRange.first);
  // If the input range is large try to narrow it down by calculating the
  // input range corresponding to the output range.
  if (inputRange.second - inputRange.first > params.kernelShape[dim]) {
    auto inputRangeForOutputRange =
        getInputRange(dim, outputRange, {0, params.kernelShape[dim]},
                      params);
    inputRange.first = std::max(inputRange.first,
                                inputRangeForOutputRange.first);
    inputRange.second = std::min(inputRange.second,
                                 inputRangeForOutputRange.second);
  }
  if (inputRange.first >= inputRange.second)
    return {0, 0};
  unsigned kernelEnd = 0;
  unsigned minBegin = 0, maxEnd = params.kernelShape[dim];
  const auto inputRangeFwd =
      boost::irange(static_cast<int>(inputRange.first),
                    static_cast<int>(inputRange.second),
                    1);
  const auto inputRangeBwd =
      boost::irange(static_cast<int>(inputRange.second) - 1,
                    static_cast<int>(inputRange.first) - 1,
                    -1);
  bool flip = params.inputTransform.flip[dim];
  for (unsigned i : flip ? inputRangeBwd : inputRangeFwd) {
    auto kernelRange = getKernelRange(dim, outputRange, i, params);
    if (kernelRange.first == kernelRange.second)
      continue;
    kernelEnd = std::max(kernelEnd, kernelRange.second);
    if (kernelEnd == maxEnd)
      break;
  }
  unsigned kernelBegin = kernelEnd;
  for (unsigned i : flip ? inputRangeFwd : inputRangeBwd) {
    auto kernelRange = getKernelRange(dim, outputRange, i, params);
    if (kernelRange.first == kernelRange.second)
      continue;
    kernelBegin = std::min(kernelBegin, kernelRange.first);
    if (kernelBegin == minBegin)
      break;
  }
  return {kernelBegin, kernelEnd};
}

std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(unsigned batchElements,
                             const std::vector<unsigned> &tileConvOutSize,
                             unsigned numContexts,
                             const std::vector<unsigned> &inputDilation,
                             const std::vector<unsigned> &stride) {
  const auto numFieldDims = tileConvOutSize.size();
  assert(inputDilation.size() == numFieldDims);
  assert(stride.size() == numFieldDims);
  std::vector<unsigned> outputStride = inputDilation;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outputStride[dim] /= gcd(outputStride[dim], stride[dim]);
  }
  std::vector<std::vector<PartialRow>> partitionByWorker;
  partitionByWorker.reserve(numContexts);
  const auto elementsPerRow =
      (tileConvOutSize.back() + outputStride.back() - 1) /
      outputStride.back();
  unsigned activeRows = 1;
  std::vector<unsigned> activeRowShape;
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    auto dimActiveRows = (tileConvOutSize[dim] + outputStride[dim] - 1) /
                         outputStride[dim];
    activeRowShape.push_back(dimActiveRows);
    activeRows *= dimActiveRows;
  }
  const auto numElements = batchElements * activeRows * elementsPerRow;
  for (unsigned i = 0; i != numContexts; ++i) {
    partitionByWorker.emplace_back();
    const auto beginElement = (i * numElements) / numContexts;
    const auto endElement = ((i + 1) * numElements) / numContexts;
    if (beginElement == endElement)
      continue;
    const auto lastElement = endElement - 1;
    auto beginIndices =
        poputil::unflattenIndex<std::size_t>({batchElements, activeRows,
                                             elementsPerRow}, beginElement);
    auto lastIndices =
        poputil::unflattenIndex<std::size_t>({batchElements, activeRows,
                                             elementsPerRow}, lastElement);
    for (unsigned b = beginIndices[0]; b != lastIndices[0] + 1; ++b) {
      unsigned activeRowBegin = b == beginIndices[0] ?
                                beginIndices[1] :
                                0;
      unsigned activeRowLast = b == lastIndices[0] ?
                               lastIndices[1] :
                               activeRows - 1;
      for (unsigned activeRow = activeRowBegin; activeRow != activeRowLast + 1;
           ++activeRow) {
        unsigned activeXBegin =
            b == beginIndices[0] && activeRow == beginIndices[1] ?
              beginIndices[2] : 0;
        unsigned activeXLast =
            b == lastIndices[0] && activeRow == lastIndices[1] ?
              lastIndices[2] : elementsPerRow - 1;
        auto outerFieldIndices = poputil::unflattenIndex(activeRowShape,
                                                        activeRow);
        for (unsigned dim = 0; dim != outerFieldIndices.size(); ++dim) {
          outerFieldIndices[dim] *= outputStride[dim];
          assert(outerFieldIndices[dim] < tileConvOutSize[dim]);
        }
        const auto xBegin = activeXBegin * outputStride.back();
        const auto xEnd = activeXLast * outputStride.back() + 1;
        assert(b < batchElements);
        assert(xBegin < tileConvOutSize.back());
        assert(xEnd <= tileConvOutSize.back());
        partitionByWorker.back().emplace_back(b, outerFieldIndices, xBegin,
                                              xEnd);
      }
    }
  }
  return partitionByWorker;
}

// Return a convolution where the same input, kernel and output size match the
// specified convolution and where the output is all zero.
static ConvParams getZeroConv(const ConvParams &params) {
  // We represent the zero convolution as follows:
  // - truncate the input and the kernel to size zero.
  // - zero pad the input and the kernel to size one.
  // - convolve the input and kernel resulting in an output of size one.
  // - truncate the output to size zero.
  // - pad the output to match the expected output size.
  ConvParams zeroConv = params;
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> allZeros(numFieldDims, 0);
  std::vector<unsigned> allOnes(numFieldDims, 1);
  std::vector<bool> allFalse(numFieldDims, false);
  zeroConv.inputTransform.truncationLower = allZeros;
  zeroConv.inputTransform.truncationUpper =
      vectorConvert<unsigned>(params.inputFieldShape);
  zeroConv.inputTransform.dilation = allOnes;
  zeroConv.inputTransform.paddingLower = allOnes;
  zeroConv.inputTransform.paddingUpper = allZeros;
  zeroConv.inputTransform.flip = allFalse;
  zeroConv.kernelTransform.truncationLower = allZeros;
  zeroConv.kernelTransform.truncationUpper =
      vectorConvert<unsigned>(params.kernelShape);
  zeroConv.kernelTransform.dilation = allOnes;
  zeroConv.kernelTransform.paddingLower = allOnes;
  zeroConv.kernelTransform.paddingUpper = allZeros;
  zeroConv.kernelTransform.flip = allFalse;
  zeroConv.outputTransform.truncationLower = allZeros;
  zeroConv.outputTransform.truncationUpper = allOnes;
  zeroConv.outputTransform.stride = allOnes;
  zeroConv.outputTransform.paddingLower = allZeros;
  zeroConv.outputTransform.paddingUpper =
      vectorConvert<unsigned>(params.getOutputFieldShape());
  assert(zeroConv.getOutputFieldShape() == params.getOutputFieldShape());
  return zeroConv;
}

static ConvParams canonicalizeParamsImpl(const ConvParams &params) {
  ConvParams newParams = params;
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto outSize = newParams.getOutputSize(dim);
    auto &inputTruncationLower = newParams.inputTransform.truncationLower[dim];
    auto &inputTruncationUpper = newParams.inputTransform.truncationUpper[dim];
    auto &inputPaddingLower = newParams.inputTransform.paddingLower[dim];
    auto &inputPaddingUpper = newParams.inputTransform.paddingUpper[dim];
    auto &kernelTruncationLower =
        newParams.kernelTransform.truncationLower[dim];
    auto &kernelTruncationUpper =
        newParams.kernelTransform.truncationUpper[dim];
    auto &kernelPaddingLower = newParams.kernelTransform.paddingLower[dim];
    auto &kernelPaddingUpper = newParams.kernelTransform.paddingUpper[dim];
    auto &outputTruncationLower =
        newParams.outputTransform.truncationLower[dim];
    auto &outputTruncationUpper =
        newParams.outputTransform.truncationUpper[dim];
    auto &outputPaddingLower = newParams.outputTransform.paddingLower[dim];
    auto &outputPaddingUpper = newParams.outputTransform.paddingUpper[dim];

    // Compute output elements that are known to be zero.
    auto nonZeroRange =
        getOutputRangeForKernelRange(dim, {0, newParams.getOutputSize(dim)},
                                     {0, newParams.kernelShape[dim]},
                                     newParams);
    // Truncate and pad the output so the number zero elements can be
    // determined directly from the output padding.
    if (nonZeroRange.first == nonZeroRange.second) {
      return getZeroConv(newParams);
    }
    const auto outputZerosLower = nonZeroRange.first;
    const auto outputZerosUpper = outSize - nonZeroRange.second;
    if (outputZerosLower > outputPaddingLower) {
      outputTruncationLower += (outputZerosLower - outputPaddingLower) *
                               newParams.outputTransform.stride[dim];
      outputPaddingLower = outputZerosLower;
    }
    if (outputZerosUpper > outputPaddingUpper) {
      outputTruncationUpper += (outputZerosUpper - outputPaddingUpper) *
                               newParams.outputTransform.stride[dim];
      outputPaddingUpper = outputZerosUpper;
    }
    // Truncate the output of the convolution so there are no excess elements
    // at the end that are ignored. If there are no ignored elements backprop
    // of the striding operation is input dilation with no padding.
    auto truncatedConvOutSize =
        newParams.getUntransformedOutputSize(dim) - (outputTruncationLower +
                                                     outputTruncationUpper);
    const auto ignored = (truncatedConvOutSize - 1) %
                         newParams.outputTransform.stride[dim];
    outputTruncationUpper += ignored;
    truncatedConvOutSize -= ignored;
    // Avoid unnecessary striding.
    if (truncatedConvOutSize == 1) {
      newParams.outputTransform.stride[dim] = 1;
    }
    // Compute input elements that are ignored.
    auto inputUsedRange =
        getInputRange(dim, {0, outSize},
                      {0, newParams.kernelShape[dim]}, newParams);
    // Truncate and pad the input so the number of ignored elements can
    // be determined directly from the input truncation.
    assert(inputUsedRange.first != inputUsedRange.second);
    const auto inputIgnoredLower = inputUsedRange.first;
    const auto inputIgnoredUpper = newParams.getInputSize(dim) -
                                   inputUsedRange.second;
    if (inputIgnoredLower > inputTruncationLower) {
      inputPaddingLower += (inputIgnoredLower - inputTruncationLower) *
                           newParams.inputTransform.dilation[dim];
      inputTruncationLower = inputIgnoredLower;
    }
    if (inputIgnoredUpper > inputTruncationUpper) {
      inputPaddingUpper += (inputIgnoredUpper - inputTruncationUpper) *
                           newParams.inputTransform.dilation[dim];
      inputTruncationUpper = inputIgnoredUpper;
    }

    // Compute kernel elements that are ignored.
    auto kernelUsedRange =
        getKernelRange(dim, {0, outSize},
                       {0, newParams.getInputSize(dim)}, newParams);
    // Truncate and pad the kernel so the number of ignored elements can
    // be determined directly from the kernel truncation.
    assert(kernelUsedRange.first != kernelUsedRange.second);
    const auto kernelIgnoredLower = kernelUsedRange.first;
    const auto kernelIgnoredUpper = newParams.kernelShape[dim] -
                                   kernelUsedRange.second;
    if (kernelIgnoredLower > kernelTruncationLower) {
      kernelPaddingLower += (kernelIgnoredLower - kernelTruncationLower) *
                           newParams.kernelTransform.dilation[dim];
      kernelTruncationLower = kernelIgnoredLower;
    }
    if (kernelIgnoredUpper > kernelTruncationUpper) {
      kernelPaddingUpper += (kernelIgnoredUpper - kernelTruncationUpper) *
                           newParams.kernelTransform.dilation[dim];
      kernelTruncationUpper = kernelIgnoredUpper;
    }

    // Remove padding if both the input and the kernel are padded.
    auto &flippedKernelPaddingLower =
        newParams.kernelTransform.flip[dim] ?
          newParams.kernelTransform.paddingUpper[dim] :
          newParams.kernelTransform.paddingLower[dim];
    auto &flippedKernelPaddingUpper =
        newParams.kernelTransform.flip[dim] ?
          newParams.kernelTransform.paddingLower[dim] :
          newParams.kernelTransform.paddingUpper[dim];
    auto &flippedPaddingLower =
        newParams.inputTransform.flip[dim] ?
          newParams.inputTransform.paddingUpper[dim] :
          newParams.inputTransform.paddingLower[dim];
    auto &flippedPaddingUpper =
        newParams.inputTransform.flip[dim] ?
          newParams.inputTransform.paddingLower[dim] :
          newParams.inputTransform.paddingUpper[dim];
    auto excessPaddingLower =
        std::min({flippedPaddingLower, flippedKernelPaddingLower,
                  newParams.getTransformedKernelSize(dim) - 1});
    flippedPaddingLower -= excessPaddingLower;
    flippedKernelPaddingLower -= excessPaddingLower;
    auto excessPaddingUpper =
        std::min({flippedPaddingUpper, flippedKernelPaddingUpper,
                  newParams.getTransformedKernelSize(dim) - 1});
    flippedPaddingUpper -= excessPaddingUpper;
    flippedKernelPaddingUpper -= excessPaddingUpper;

    // Remove padding if the input is padded and the output is truncated.
    excessPaddingLower =
        std::min({flippedPaddingLower, outputTruncationLower,
                  static_cast<unsigned>(
                    newParams.getUntransformedOutputSize(dim) - 1
                  )});
    flippedPaddingLower -= excessPaddingLower;
    outputTruncationLower -= excessPaddingLower;
    excessPaddingUpper =
        std::min({flippedPaddingUpper, outputTruncationUpper,
                  static_cast<unsigned>(
                    newParams.getUntransformedOutputSize(dim) - 1
                  )});
    flippedPaddingUpper -= excessPaddingUpper;
    outputTruncationUpper -= excessPaddingUpper;

    // Avoid unnecessary flipping / dilation.
    if (newParams.inputFieldShape[dim] <=
        newParams.inputTransform.truncationLower[dim] +
        1 + newParams.inputTransform.truncationUpper[dim]) {
      newParams.inputTransform.dilation[dim] = 1;
      if (newParams.inputTransform.flip[dim]) {
        newParams.inputTransform.flip[dim] = false;
        std::swap(newParams.inputTransform.paddingLower[dim],
                  newParams.inputTransform.paddingUpper[dim]);
      }
    }
    if (newParams.kernelShape[dim] <=
        newParams.kernelTransform.truncationLower[dim] + 1 +
        newParams.kernelTransform.truncationUpper[dim]) {
      newParams.kernelTransform.dilation[dim] = 1;
      if (newParams.kernelTransform.flip[dim]) {
        newParams.kernelTransform.flip[dim] = false;
        std::swap(newParams.kernelTransform.paddingLower[dim],
                  newParams.kernelTransform.paddingUpper[dim]);
      }
    }
    assert(newParams.getOutputSize(dim) == outSize);
  }
  return newParams;
}

ConvParams canonicalizeParams(const ConvParams &params) {
  ConvParams newParams = canonicalizeParamsImpl(params);
  assert(newParams == canonicalizeParamsImpl(newParams) &&
         "canonicalizeParams is not idempotent");
  return newParams;
}

ConvParams getGradientParams(const ConvParams &params) {
  // Note we assume the caller explicitly flips the weights in each spatial
  // axis before the convolution. TODO it may be more efficient to fold the
  // flipping of the weights into the convolution by setting the flipKernel
  // parameter appropriately.
  auto canonicalParams = canonicalizeParams(params);
  auto bwdInputTruncationLower = canonicalParams.outputTransform.paddingLower;
  auto bwdInputTruncationUpper = canonicalParams.outputTransform.paddingUpper;
  auto bwdInputDilation = canonicalParams.outputTransform.stride;
  auto bwdInputPaddingLower = canonicalParams.outputTransform.truncationLower;
  auto bwdInputPaddingUpper = canonicalParams.outputTransform.truncationUpper;
  auto bwdOutputTruncationLower = canonicalParams.inputTransform.paddingLower;
  auto bwdOutputTruncationUpper = canonicalParams.inputTransform.paddingUpper;
  auto bwdStride = canonicalParams.inputTransform.dilation;
  auto bwdOutputPaddingLower = canonicalParams.inputTransform.truncationLower;
  auto bwdOutputPaddingUpper = canonicalParams.inputTransform.truncationUpper;
  const auto numFieldDims = params.getNumFieldDims();
  // The "valid" convolution in the forward pass becomes a "full" convolution
  // in the backward pass. We can express this as a "valid" convolution with
  // (kernelSize - 1) padding.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto kernelSize =
        canonicalParams.getTransformedKernelSize(dim);
    bwdInputPaddingLower[dim] += kernelSize - 1;
    bwdInputPaddingUpper[dim] += kernelSize - 1;
  }
  // Going backwards the weights are flipped in each axis so we must flip
  // the upper and lower truncation / padding.
  auto bwdKernelTruncationLower =
      canonicalParams.kernelTransform.truncationUpper;
  auto bwdKernelTruncationUpper =
      canonicalParams.kernelTransform.truncationLower;
  auto bwdKernelPaddingLower = canonicalParams.kernelTransform.paddingUpper;
  auto bwdKernelPaddingUpper = canonicalParams.kernelTransform.paddingLower;
  auto bwdFlipInput = std::vector<bool>(numFieldDims);
  auto bwdFlipKernel = canonicalParams.kernelTransform.flip;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (canonicalParams.inputTransform.flip[dim]) {
      // If the input is flipped in the forward pass we must flip the output
      // in the backward pass. This is equivalent to flipping both the
      // input and the kernel in the backward pass.
      bwdFlipKernel[dim] = !bwdFlipKernel[dim];
      bwdFlipInput[dim] = !bwdFlipInput[dim];
    }
  }
  popconv::ConvParams bwdParams(canonicalParams.dType,
                                canonicalParams.batchSize,
                                canonicalParams.getOutputFieldShape(),
                                canonicalParams.kernelShape,
                                canonicalParams.getNumOutputChansPerConvGroup(),
                                canonicalParams.getNumInputChansPerConvGroup(),
                                canonicalParams.getNumConvGroups(),
                                bwdInputTruncationLower,
                                bwdInputTruncationUpper,
                                bwdInputDilation,
                                bwdInputPaddingLower,
                                bwdInputPaddingUpper,
                                bwdFlipInput,
                                bwdKernelTruncationLower,
                                bwdKernelTruncationUpper,
                                canonicalParams.kernelTransform.dilation,
                                bwdKernelPaddingLower,
                                bwdKernelPaddingUpper,
                                bwdFlipKernel,
                                bwdOutputTruncationLower,
                                bwdOutputTruncationUpper,
                                bwdStride,
                                bwdOutputPaddingLower,
                                bwdOutputPaddingUpper);
  return canonicalizeParams(bwdParams);
}

unsigned detectChannelGrouping(const poplar::Tensor &t0) {
  if (t0.rank() == 0)
    throw poputil::poplib_error("Cannot detect channel grouping of "
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

std::pair<unsigned, unsigned>
getTileSplitForGroup(unsigned group, unsigned numGroups, unsigned numTiles) {
  if (numTiles < numGroups) {
    return std::make_pair(group % numTiles, 1);
  } else {
    const auto tilesPerGroup = numTiles / numGroups;
    const auto g1 = numGroups * (tilesPerGroup + 1) - numTiles;
    if (group < g1) {
      return std::make_pair(group * tilesPerGroup, tilesPerGroup);
    } else {
      return std::make_pair(group * (tilesPerGroup + 1) -  g1,
                            tilesPerGroup + 1);
    }
  }
}

} // namespace convutil
