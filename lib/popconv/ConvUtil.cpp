#include "popconv/ConvUtil.hpp"
#include <cassert>
#include <popstd/exceptions.hpp>
#include <popstd/Util.hpp>
#include "util/gcd.hpp"

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
  assert(outputIndex < params.getOutputSize(dim));
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
  return reverseDilationAndPadding(paddedInputIndex,
                                   params.inputFieldShape[dim],
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
  unsigned inputEnd = 0;
  unsigned minBegin = 0, maxEnd = params.inputFieldShape[dim];
  for (unsigned k = kernelRange.second; k != kernelRange.first; --k) {
    auto inputRange = getInputRange(dim, outputRange, k - 1, params);
    inputEnd = std::max(inputEnd, inputRange.second);
    if (inputEnd == maxEnd)
      break;
  }
  unsigned inputBegin = inputEnd;
  for (unsigned k = kernelRange.first; k != kernelRange.second; ++k) {
    auto inputRange = getInputRange(dim, outputRange, k, params);
    inputBegin = std::min(inputBegin, inputRange.first);
    if (inputBegin == minBegin)
      break;
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
        popstd::unflattenIndex<std::size_t>({batchElements, activeRows,
                                             elementsPerRow}, beginElement);
    auto lastIndices =
        popstd::unflattenIndex<std::size_t>({batchElements, activeRows,
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
        auto outerFieldIndices = popstd::unflattenIndex(activeRowShape,
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

ConvParams canonicalizeParams(const ConvParams &params) {
  ConvParams newParams = params;
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto paddedInputSize = newParams.getPaddedDilatedInputSize(dim);
    const auto paddedKernelSize = newParams.getPaddedDilatedKernelSize(dim);
    const auto postConvolveSize =
        absdiff(paddedInputSize, paddedKernelSize) + 1;
    // Truncate the input or the kernel (whichever is larger) so there are no
    // excess elements at the end that are ignored. If there are no ignored
    // elements backprop of the striding operation is input dilation with no
    // padding.
    const auto ignored = (postConvolveSize - 1) % newParams.stride[dim];
    if (paddedInputSize > paddedKernelSize)
      newParams.inputPaddingUpper[dim] -= ignored;
    else
      newParams.kernelPaddingUpper[dim] -= ignored;
  }
  return newParams;
}

ConvParams getGradientParams(const ConvParams &params) {
  auto canonicalParams = canonicalizeParams(params);
  std::vector<int> bwdInputPaddingLower, bwdInputPaddingUpper;
  std::vector<unsigned> bwdStride, bwdInputDilation;
  bwdStride = canonicalParams.inputDilation;
  bwdInputDilation = canonicalParams.stride;
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto kernelSize = canonicalParams.getPaddedDilatedKernelSize(dim);
    const auto inputPaddingLower = canonicalParams.inputPaddingLower[dim];
    const auto inputPaddingUpper = canonicalParams.inputPaddingUpper[dim];
    bwdInputPaddingLower.push_back(
      static_cast<int>(kernelSize) - 1 - inputPaddingLower
    );
    bwdInputPaddingUpper.push_back(
      static_cast<int>(kernelSize) - 1 - inputPaddingUpper
    );
  }
  // Going backwards the weights are flipped in each axis and so we must flip
  // the upper and lower padding.
  auto bwdKernelPaddingLower = canonicalParams.kernelPaddingUpper;
  auto bwdKernelPaddingUpper = canonicalParams.kernelPaddingLower;
  return popconv::ConvParams(canonicalParams.dType,
                             canonicalParams.batchSize,
                             canonicalParams.getOutputFieldShape(),
                             canonicalParams.kernelShape,
                             canonicalParams.getNumOutputChansPerConvGroup(),
                             canonicalParams.getNumInputChansPerConvGroup(),
                             bwdStride, bwdInputPaddingLower,
                             bwdInputPaddingUpper, bwdInputDilation,
                             bwdKernelPaddingLower, bwdKernelPaddingUpper,
                             canonicalParams.kernelDilation,
                             canonicalParams.getNumConvGroups());
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
