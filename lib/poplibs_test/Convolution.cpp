#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/exceptions.hpp>

#include <algorithm>
#include <functional>

// TODO move unflattenIndex / flattenIndex to poplibs_support once T1724
// is fixed.
static std::vector<unsigned>
unflattenIndex(const std::vector<unsigned> &shape,
               std::size_t index) {
  std::vector<unsigned> coord;
  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    const auto dim = *it;
    coord.push_back(index % dim);
    index /= dim;
  }
  std::reverse(coord.begin(), coord.end());
  return coord;
}

static unsigned flattenIndex(const std::vector<unsigned> &shape,
                             const std::vector<unsigned> &indices) {
  auto rank = shape.size();
  assert(indices.size() == rank);
  std::size_t index = 0;
  for (unsigned i = 0; i != rank; ++i) {
    index = index * shape[i] + indices[i];
  }
  return index;
}

static unsigned absdiff(unsigned a, unsigned b) {
  return a < b ? b - a : a - b;
}

template <class T>
static T product(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), T(1), std::multiplies<T>());
}

static unsigned getDilatedSize(unsigned size, unsigned dilation) {
  if (size == 0)
    return 0;
  return 1 + (size - 1) * dilation;
}

static unsigned getTransformedSize(unsigned size,
                                   unsigned truncationLower,
                                   unsigned truncationUpper,
                                   unsigned dilation,
                                   unsigned paddingLower,
                                   unsigned paddingUpper) {
  unsigned truncatedSize = size - (truncationLower + truncationUpper);
  unsigned dilatedSize = getDilatedSize(truncatedSize, dilation);
  unsigned paddedSize = paddingLower + dilatedSize + paddingUpper;
  return paddedSize;
}

static std::vector<unsigned>
getTransformedSize(const std::vector<unsigned> &inputSize,
                   const std::vector<unsigned> &truncationLower,
                   const std::vector<unsigned> &truncationUpper,
                   const std::vector<unsigned> &dilation,
                   const std::vector<unsigned> &paddingLower,
                   const std::vector<unsigned> &paddingUpper) {
  const auto numFieldDims = inputSize.size();
  std::vector<unsigned> outputSize;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outputSize.push_back(getTransformedSize(inputSize[dim],
                                            truncationLower[dim],
                                            truncationUpper[dim],
                                            dilation[dim],
                                            paddingLower[dim],
                                            paddingUpper[dim]));
  }
  return outputSize;
}

static unsigned
getOutputFieldSize(unsigned inputSize,
                   unsigned kernelSize,
                   unsigned inputTruncationLower,
                   unsigned inputTruncationUpper,
                   unsigned inputDilation,
                   unsigned inputPaddingLower,
                   unsigned inputPaddingUpper,
                   unsigned kernelTruncationLower,
                   unsigned kernelTruncationUpper,
                   unsigned kernelDilation,
                   unsigned kernelPaddingLower,
                   unsigned kernelPaddingUpper,
                   unsigned outputTruncationLower,
                   unsigned outputTruncationUpper,
                   unsigned stride,
                   unsigned outputPaddingLower,
                   unsigned outputPaddingUpper) {
  auto transformedInputSize =
      getTransformedSize(inputSize, inputTruncationLower,
                         inputTruncationUpper, inputDilation,
                         inputPaddingLower, inputPaddingUpper);
  auto transformedKernelSize =
      getTransformedSize(kernelSize, kernelTruncationLower,
                         kernelTruncationUpper, kernelDilation,
                         kernelPaddingLower, kernelPaddingUpper);
  auto convOutSize = transformedInputSize - transformedKernelSize + 1;
  auto truncatedOutputSize =
      convOutSize - (outputTruncationLower + outputTruncationUpper);
  auto truncatedStridedOutputSize = (truncatedOutputSize + stride - 1) / stride;
  return outputPaddingLower + truncatedStridedOutputSize + outputPaddingUpper;
}

static std::vector<unsigned>
getOutputFieldSize(const std::vector<unsigned> &inputSize,
                   const std::vector<unsigned> &kernelSize,
                   const std::vector<unsigned> &inputTruncationLower,
                   const std::vector<unsigned> &inputTruncationUpper,
                   const std::vector<unsigned> &inputDilation,
                   const std::vector<unsigned> &inputPaddingLower,
                   const std::vector<unsigned> &inputPaddingUpper,
                   const std::vector<unsigned> &kernelTruncationLower,
                   const std::vector<unsigned> &kernelTruncationUpper,
                   const std::vector<unsigned> &kernelDilation,
                   const std::vector<unsigned> &kernelPaddingLower,
                   const std::vector<unsigned> &kernelPaddingUpper,
                   const std::vector<unsigned> &outputTruncationLower,
                   const std::vector<unsigned> &outputTruncationUpper,
                   const std::vector<unsigned> &stride,
                   const std::vector<unsigned> &outputPaddingLower,
                   const std::vector<unsigned> &outputPaddingUpper) {
  const auto numFieldDims = inputSize.size();
  std::vector<unsigned> outputSize;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outputSize.push_back(getOutputFieldSize(inputSize[dim],
                                            kernelSize[dim],
                                            inputTruncationLower[dim],
                                            inputTruncationUpper[dim],
                                            inputDilation[dim],
                                            inputPaddingLower[dim],
                                            inputPaddingUpper[dim],
                                            kernelTruncationLower[dim],
                                            kernelTruncationUpper[dim],
                                            kernelDilation[dim],
                                            kernelPaddingLower[dim],
                                            kernelPaddingUpper[dim],
                                            outputTruncationLower[dim],
                                            outputTruncationUpper[dim],
                                            stride[dim],
                                            outputPaddingLower[dim],
                                            outputPaddingUpper[dim]));
  }
  return outputSize;
}

static std::vector<unsigned>
dilateIndices(const std::vector<unsigned> &indices,
              const std::vector<unsigned> &dilation) {
  const auto numFieldDims = indices.size();
  std::vector<unsigned> dilatedIndices;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto index = indices[dim];
    dilatedIndices.push_back(index * dilation[dim]);
  }
  return dilatedIndices;
}

static boost::multi_array<double, 2>
truncateDilatePadAndFlip(boost::const_multi_array_ref<double, 2> in,
                         const std::vector<unsigned> &size,
                         const std::vector<unsigned> &truncationLower,
                         const std::vector<unsigned> &truncationUpper,
                         const std::vector<unsigned> &dilation,
                         const std::vector<unsigned> &paddingLower,
                         const std::vector<unsigned> &paddingUpper,
                         const std::vector<bool> &flip,
                         std::vector<unsigned> &paddedSize) {
#ifndef NDEBUG
  const auto numFieldElements = product(size);
  assert(in.shape()[1] == numFieldElements);
#endif
  const auto numFieldDims = size.size();
  std::vector<unsigned> truncatedSize(numFieldDims);
  std::vector<unsigned> dilatedSize(numFieldDims);
  paddedSize.resize(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    truncatedSize[dim] = size[dim] -
                         (truncationLower[dim] + truncationUpper[dim]);
    dilatedSize[dim] = getDilatedSize(truncatedSize[dim], dilation[dim]);
    paddedSize[dim] = dilatedSize[dim] + paddingLower[dim] + paddingUpper[dim];
  }
  const auto truncatedElements = product(truncatedSize);
  boost::multi_array<double, 2>
      truncated(boost::extents[in.shape()[0]][truncatedElements]);
  for (unsigned i = 0; i != in.shape()[0]; ++i) {
    for (unsigned e = 0; e != truncatedElements; ++e) {
      auto truncatedIndices = unflattenIndex(truncatedSize, e);
      std::vector<unsigned> indices;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto truncatedIndex = truncatedIndices[dim];
        const auto index = truncatedIndex + truncationLower[dim];
        indices.push_back(index);
      }
      truncated[i][e] = in[i][flattenIndex(size, indices)];
    }
  }
  const auto dilatedElements = product(dilatedSize);
  boost::multi_array<double, 2>
      dilated(boost::extents[in.shape()[0]][dilatedElements]);
  std::fill(dilated.data(), dilated.data() + dilated.num_elements(), 0.0);
  for (unsigned i = 0; i != in.shape()[0]; ++i) {
    for (unsigned e = 0; e != truncatedElements; ++e) {
      auto indices = unflattenIndex(truncatedSize, e);
      auto dilatedIndices = dilateIndices(indices, dilation);
      dilated[i][flattenIndex(dilatedSize, dilatedIndices)] = truncated[i][e];
    }
  }
  boost::multi_array<double, 2>
      padded(boost::extents[in.shape()[0]]
                           [product(paddedSize)]);
  std::fill(padded.data(), padded.data() + padded.num_elements(), 0.0);
  for (unsigned i = 0; i != in.shape()[0]; ++i) {
    for (unsigned e = 0; e != dilatedElements; ++e) {
      auto indices = unflattenIndex(dilatedSize, e);
      std::vector<unsigned> paddedIndices;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto index = indices[dim];
        auto paddedIndex = index + paddingLower[dim];
        if (flip[dim])
          paddedIndex = paddedSize[dim] - 1 - paddedIndex;
        paddedIndices.push_back(paddedIndex);
      }
      padded[i][flattenIndex(paddedSize, paddedIndices)] =
          dilated[i][e];
    }
  }
  return padded;
}

static boost::multi_array<double, 3>
truncateDilatePadAndFlipActivations(
    boost::const_multi_array_ref<double, 3> in,
    const std::vector<unsigned> &fieldSize,
    const std::vector<unsigned> &truncationLower,
    const std::vector<unsigned> &truncationUpper,
    const std::vector<unsigned> inputDilation,
    const std::vector<unsigned> &paddingLower,
    const std::vector<unsigned> &paddingUpper,
    const std::vector<bool> &flipInput,
    std::vector<unsigned> &paddedFieldSize) {
  const auto numFieldElements = in.shape()[2];
  assert(numFieldElements == product(fieldSize));
  boost::const_multi_array_ref<double, 2>
      kernelFlattened(in.data(),
                      boost::extents[in.num_elements() / numFieldElements]
                                    [numFieldElements]);
  auto paddedFlattened =
      truncateDilatePadAndFlip(kernelFlattened, fieldSize,
                               truncationLower, truncationUpper, inputDilation,
                               paddingLower, paddingUpper, flipInput,
                               paddedFieldSize);
  boost::multi_array<double, 3>
      padded(boost::extents[in.shape()[0]]
                           [in.shape()[1]]
                           [paddedFlattened.shape()[1]]);
  boost::multi_array_ref<double, 2>(
    padded.data(),
    boost::extents[paddedFlattened.shape()[0]][paddedFlattened.shape()[1]]
  ) = paddedFlattened;
  return padded;
}

static boost::multi_array<double, 3>
truncateDilatePadAndFlipActivations(
    boost::const_multi_array_ref<double, 3> in,
    const std::vector<unsigned> &fieldSize,
    const std::vector<unsigned> &truncationLower,
    const std::vector<unsigned> &truncationUpper,
    const std::vector<unsigned> inputDilation,
    const std::vector<unsigned> &paddingLower,
    const std::vector<unsigned> &paddingUpper,
    const std::vector<bool> &flipInput) {
  std::vector<unsigned> dummy;
  return truncateDilatePadAndFlipActivations(in, fieldSize,
                                             truncationLower,
                                             truncationUpper,
                                             inputDilation,
                                             paddingLower, paddingUpper,
                                             flipInput, dummy);
}

static boost::multi_array<double, 4>
truncateDilatePadAndFlipKernel(
    boost::const_multi_array_ref<double, 4> kernel,
    const std::vector<unsigned> &kernelSize,
    const std::vector<unsigned> &kernelTruncationLower,
    const std::vector<unsigned> &kernelTruncationUpper,
    const std::vector<unsigned> &kernelDilation,
    const std::vector<unsigned> &kernelPaddingLower,
    const std::vector<unsigned> &kernelPaddingUpper,
    const std::vector<bool> &flipKernel,
    std::vector<unsigned> &paddedSize) {
  const auto numFieldElements = kernel.shape()[3];
  assert(numFieldElements == product(kernelSize));
  boost::const_multi_array_ref<double, 2>
      kernelFlattened(kernel.data(),
                      boost::extents[kernel.num_elements() / numFieldElements]
                                    [numFieldElements]);
  auto paddedFlattened =
      truncateDilatePadAndFlip(kernelFlattened, kernelSize,
                               kernelTruncationLower, kernelTruncationUpper,
                               kernelDilation,
                               kernelPaddingLower, kernelPaddingUpper,
                               flipKernel, paddedSize);
  boost::multi_array<double, 4>
      padded(boost::extents[kernel.shape()[0]]
                           [kernel.shape()[1]]
                           [kernel.shape()[2]]
                           [paddedFlattened.shape()[1]]);
  boost::multi_array_ref<double, 2>(
    padded.data(),
    boost::extents[paddedFlattened.shape()[0]][paddedFlattened.shape()[1]]
  ) = paddedFlattened;
  return padded;
}

static boost::multi_array<double, 2>
truncateDilatePadAndFlipInverse(boost::const_multi_array_ref<double, 2> padded,
                                const std::vector<unsigned> &paddedSize,
                                const std::vector<unsigned> &truncationLower,
                                const std::vector<unsigned> &truncationUpper,
                                const std::vector<unsigned> &dilation,
                                const std::vector<unsigned> &paddingLower,
                                const std::vector<unsigned> &paddingUpper,
                                const std::vector<bool> &flip,
                                std::vector<unsigned> &size) {
  assert(padded.shape()[1] == product(paddedSize));
  const auto numFieldDims = paddedSize.size();
  std::vector<unsigned> truncatedSize(numFieldDims);
  std::vector<unsigned> dilatedSize(numFieldDims);
  size.resize(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    dilatedSize[dim] = paddedSize[dim] -
                       (paddingLower[dim] + paddingUpper[dim]);
    truncatedSize[dim] = (dilatedSize[dim] + dilation[dim] - 1) / dilation[dim];
    size[dim] = truncationLower[dim] + truncatedSize[dim] +
                truncationUpper[dim];
  }
  const auto dilatedElements = product(dilatedSize);
  boost::multi_array<double, 2>
      dilated(boost::extents[padded.shape()[0]][dilatedElements]);
  for (unsigned i = 0; i != padded.shape()[0]; ++i) {
    for (unsigned e = 0; e != dilatedElements; ++e) {
      auto indices = unflattenIndex(dilatedSize, e);
      std::vector<unsigned> paddedIndices;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto index = indices[dim];
        auto paddedIndex = index + paddingLower[dim];
        if (flip[dim])
          paddedIndex = paddedSize[dim] - 1 - paddedIndex;
        paddedIndices.push_back(paddedIndex);
      }
      dilated[i][e] = padded[i][flattenIndex(paddedSize, paddedIndices)];
    }
  }

  const auto truncatedElements = product(truncatedSize);
  boost::multi_array<double, 2>
      truncated(boost::extents[padded.shape()[0]][truncatedElements]);
  std::fill(truncated.data(), truncated.data() + truncated.num_elements(), 0.0);
  for (unsigned i = 0; i != padded.shape()[0]; ++i) {
    for (unsigned e = 0; e != truncatedElements; ++e) {
      auto indices = unflattenIndex(truncatedSize, e);
      auto dilatedIndices = dilateIndices(indices, dilation);
      truncated[i][e] = dilated[i][flattenIndex(dilatedSize, dilatedIndices)];
    }
  }
  const auto numElements = product(size);
  boost::multi_array<double, 2>
      out(boost::extents[padded.shape()[0]][numElements]);
  for (unsigned i = 0; i != padded.shape()[0]; ++i) {
    for (unsigned e = 0; e != truncatedElements; ++e) {
      auto truncatedIndices = unflattenIndex(truncatedSize, e);
      std::vector<unsigned> indices;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto truncatedIndex = truncatedIndices[dim];
        const auto index = truncatedIndex + truncationLower[dim];
        indices.push_back(index);
      }
      out[i][flattenIndex(size, indices)] = truncated[i][e];
    }
  }
  return out;
}

static boost::multi_array<double, 3>
truncateDilatePadAndFlipActivationsInverse(
    boost::const_multi_array_ref<double, 3> paddedActs,
    const std::vector<unsigned> &paddedFieldSize,
    const std::vector<unsigned> &truncationLower,
    const std::vector<unsigned> &truncationUpper,
    const std::vector<unsigned> dilation,
    const std::vector<unsigned> &paddingLower,
    const std::vector<unsigned> &paddingUpper,
    const std::vector<bool> &flip,
    std::vector<unsigned> &fieldSize) {
  const auto numFieldElements = paddedActs.shape()[2];
  assert(numFieldElements == product(paddedFieldSize));
  boost::const_multi_array_ref<double, 2>
      paddedFlattened(
        paddedActs.data(),
        boost::extents[paddedActs.num_elements() / numFieldElements]
                      [numFieldElements]
      );
  auto actsFlattened =
      truncateDilatePadAndFlipInverse(paddedFlattened, paddedFieldSize,
                                      truncationLower, truncationUpper,
                                      dilation, paddingLower, paddingUpper,
                                      flip, fieldSize);
  boost::multi_array<double, 3>
      acts(boost::extents[paddedActs.shape()[0]]
                         [paddedActs.shape()[1]]
                         [actsFlattened.shape()[1]]);
  boost::multi_array_ref<double, 2>(
    acts.data(),
    boost::extents[actsFlattened.shape()[0]][actsFlattened.shape()[1]]
  ) = actsFlattened;
  return acts;
}

static boost::multi_array<double, 3>
truncateDilatePadAndFlipActivationsInverse(
    boost::const_multi_array_ref<double, 3> paddedActs,
    const std::vector<unsigned> &paddedFieldSize,
    const std::vector<unsigned> &truncationLower,
    const std::vector<unsigned> &truncationUpper,
    const std::vector<unsigned> &dilation,
    const std::vector<unsigned> &paddingLower,
    const std::vector<unsigned> &paddingUpper,
    const std::vector<bool> &flip) {
  std::vector<unsigned> dummy;
  return truncateDilatePadAndFlipActivationsInverse(
        paddedActs, paddedFieldSize, truncationLower, truncationUpper,
        dilation, paddingLower, paddingUpper, flip, dummy);
}

static boost::multi_array<double, 4>
truncateDilatePadAndFlipKernelInverse(
    boost::const_multi_array_ref<double, 4> padded,
    const std::vector<unsigned> &paddedSize,
    const std::vector<unsigned> &kernelTruncationLower,
    const std::vector<unsigned> &kernelTruncationUpper,
    const std::vector<unsigned> &kernelDilation,
    const std::vector<unsigned> &kernelPaddingLower,
    const std::vector<unsigned> &kernelPaddingUpper,
    const std::vector<bool> &flipKernel,
    std::vector<unsigned> &kernelSize) {
  const auto numFieldElements = padded.shape()[3];
  assert(numFieldElements == product(paddedSize));
  boost::const_multi_array_ref<double, 2>
      paddedFlattened(padded.data(),
                      boost::extents[padded.num_elements() / numFieldElements]
                                    [numFieldElements]);
  auto kernelFlattened =
      truncateDilatePadAndFlipInverse(paddedFlattened, paddedSize,
                                      kernelTruncationLower,
                                      kernelTruncationUpper,
                                      kernelDilation,
                                      kernelPaddingLower, kernelPaddingUpper,
                                      flipKernel, kernelSize);
  boost::multi_array<double, 4>
      kernel(boost::extents[padded.shape()[0]]
                           [padded.shape()[1]]
                           [padded.shape()[2]]
                           [kernelFlattened.shape()[1]]);
  boost::multi_array_ref<double, 2>(
    kernel.data(),
    boost::extents[kernelFlattened.shape()[0]][kernelFlattened.shape()[1]]
  ) = kernelFlattened;
  return kernel;
}

static boost::multi_array<double, 4>
truncateDilatePadAndFlipKernelInverse(
    boost::const_multi_array_ref<double, 4> padded,
    const std::vector<unsigned> &paddedSize,
    const std::vector<unsigned> &kernelTruncationLower,
    const std::vector<unsigned> &kernelTruncationUpper,
    const std::vector<unsigned> &kernelDilation,
    const std::vector<unsigned> &kernelPaddingLower,
    const std::vector<unsigned> &kernelPaddingUpper,
    const std::vector<bool> &flipKernel) {
  std::vector<unsigned> dummy;
  return truncateDilatePadAndFlipKernelInverse(padded, paddedSize,
                                               kernelTruncationLower,
                                               kernelTruncationUpper,
                                               kernelDilation,
                                               kernelPaddingLower,
                                               kernelPaddingUpper,
                                               flipKernel,
                                               dummy);
}

static unsigned
getInputIndex(unsigned inputSize, unsigned kernelSize,
              unsigned outputIndex, unsigned kernelIndex) {
  if (kernelSize > inputSize) {
    int inputIndex = static_cast<int>(kernelIndex) -
                     static_cast<int>(outputIndex);
    if (inputIndex < 0 || static_cast<unsigned>(inputIndex) >= inputSize)
      return ~0U;
    return inputIndex;
  }
  return kernelIndex + outputIndex;
}

static bool
getInputIndices(const std::vector<unsigned> &inputSize,
                const std::vector<unsigned> &kernelSize,
                const std::vector<unsigned> &outputIndices,
                const std::vector<unsigned> &kernelIndices,
                std::vector<unsigned> &inputIndices) {
  const auto numFieldDims = inputSize.size();
  inputIndices.clear();
  inputIndices.reserve(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto inputIndex = getInputIndex(inputSize[dim],
                                          kernelSize[dim],
                                          outputIndices[dim],
                                          kernelIndices[dim]);
    if (inputIndex == ~0U) {
      return false;
    }
    inputIndices.push_back(inputIndex);
  }
  return true;
}

void poplibs_test::conv::
convolution(const std::vector<unsigned> &inputFieldSize,
            const std::vector<unsigned> &truncationLower,
            const std::vector<unsigned> &truncationUpper,
            const std::vector<unsigned> &inputDilation,
            const std::vector<unsigned> &paddingLower,
            const std::vector<unsigned> &paddingUpper,
            const std::vector<bool> &flipInput,
            const std::vector<unsigned> &kernelSize,
            const std::vector<unsigned> &kernelTruncationLower,
            const std::vector<unsigned> &kernelTruncationUpper,
            const std::vector<unsigned> &kernelDilation,
            const std::vector<unsigned> &kernelPaddingLower,
            const std::vector<unsigned> &kernelPaddingUpper,
            const std::vector<bool> &flipKernel,
            const std::vector<unsigned> &outputTruncationLower,
            const std::vector<unsigned> &outputTruncationUpper,
            const std::vector<unsigned> &stride,
            const std::vector<unsigned> &outputPaddingLower,
            const std::vector<unsigned> &outputPaddingUpper,
            boost::const_multi_array_ref<double, 3> in,
            boost::const_multi_array_ref<double, 4> kernel,
            boost::const_multi_array_ref<double, 1> biases,
            boost::multi_array_ref<double, 3> out) {
  if (inputFieldSize.size() != truncationLower.size() ||
      inputFieldSize.size() != truncationUpper.size() ||
      inputFieldSize.size() != inputDilation.size() ||
      inputFieldSize.size() != paddingLower.size() ||
      inputFieldSize.size() != paddingUpper.size() ||
      inputFieldSize.size() != flipInput.size() ||
      inputFieldSize.size() != kernelSize.size() ||
      inputFieldSize.size() != kernelTruncationLower.size() ||
      inputFieldSize.size() != kernelTruncationUpper.size() ||
      inputFieldSize.size() != kernelDilation.size() ||
      inputFieldSize.size() != kernelPaddingLower.size() ||
      inputFieldSize.size() != kernelPaddingUpper.size() ||
      inputFieldSize.size() != flipKernel.size() ||
      inputFieldSize.size() != outputTruncationLower.size() ||
      inputFieldSize.size() != outputTruncationUpper.size() ||
      inputFieldSize.size() != stride.size() ||
      inputFieldSize.size() != outputPaddingLower.size() ||
      inputFieldSize.size() != outputPaddingUpper.size()) {
    throw poplibs_test::poplibs_test_error(
      "Mismatch in number of spatial dimensions."
    );
  }
  if (product(inputFieldSize) != in.shape()[2] ||
      product(kernelSize) != kernel.shape()[3]) {
    throw poplibs_test::poplibs_test_error(
      "Mismatch between tensor size and spatial field size."
    );
  }
  const auto batchSize = in.shape()[0];
  const auto numConvGroups = kernel.shape()[0];
  const auto inputChannelsPerConvGroup = kernel.shape()[2];
  const auto inputChannels = in.shape()[1];
  if (inputChannels != inputChannelsPerConvGroup * numConvGroups) {
    throw poplibs_test::poplibs_test_error("Input channels in kernel do not "
                                         "match activations for grouped conv");
  }
  const auto numFieldDims = inputFieldSize.size();
  std::vector<unsigned> paddedKernelSize;
  auto paddedKernel =
      truncateDilatePadAndFlipKernel(kernel,
                                     kernelSize,
                                     kernelTruncationLower,
                                     kernelTruncationUpper,
                                     kernelDilation,
                                     kernelPaddingLower,
                                     kernelPaddingUpper,
                                     flipKernel,
                                     paddedKernelSize);
  std::vector<unsigned> paddedFieldSize;
  auto paddedIn =
      truncateDilatePadAndFlipActivations(in,
                                          inputFieldSize,
                                          truncationLower,
                                          truncationUpper,
                                          inputDilation,
                                          paddingLower,
                                          paddingUpper,
                                          flipInput,
                                          paddedFieldSize);

  const auto outputChannelsPerConvGroup = kernel.shape()[1];
  const auto outputChannels = out.shape()[1];
  assert(outputChannels == outputChannelsPerConvGroup * numConvGroups);
  std::vector<unsigned> convOutSize(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    convOutSize[dim] = absdiff(paddedFieldSize[dim],
                               paddedKernelSize[dim]) + 1;
  }
  const auto convOutElements = product(convOutSize);
  boost::multi_array<double, 3>
      convOut(boost::extents[batchSize]
                            [outputChannels]
                            [convOutElements]);
  std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
  const auto paddedKernelElements = product(paddedKernelSize);
  for (unsigned gc = 0; gc != numConvGroups; ++gc) {
    for (unsigned b = 0; b != batchSize; ++b) {
      // Perform convolution.
      for (unsigned oc = 0; oc != outputChannelsPerConvGroup; ++oc) {
        unsigned ocAct = gc * outputChannelsPerConvGroup + oc;
        for (unsigned oe = 0; oe != convOutElements; ++oe) {
          auto outputIndices = unflattenIndex(convOutSize, oe);
          for (unsigned ke = 0; ke != paddedKernelElements; ++ke) {
            auto kernelIndices = unflattenIndex(paddedKernelSize, ke);
            std::vector<unsigned> inputIndices;
            if (getInputIndices(paddedFieldSize, paddedKernelSize,
                                outputIndices, kernelIndices, inputIndices)) {
              const auto ie = flattenIndex(paddedFieldSize, inputIndices);
              for (unsigned ic = 0; ic != inputChannelsPerConvGroup; ++ic) {
                unsigned icAct = gc * inputChannelsPerConvGroup + ic;
                convOut[b][ocAct][oe] +=
                    paddedKernel[gc][oc][ic][ke] *
                    paddedIn[b][icAct][ie];
              }
            }
          }
        }
      }
    }
  }

  std::vector<bool> noFlipping(numFieldDims);
  out = truncateDilatePadAndFlipActivationsInverse(convOut, convOutSize,
                                                   outputPaddingLower,
                                                   outputPaddingUpper,
                                                   stride,
                                                   outputTruncationLower,
                                                   outputTruncationUpper,
                                                   noFlipping);
  for (unsigned gc = 0; gc != numConvGroups; ++gc) {
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned oc = 0; oc != outputChannelsPerConvGroup; ++oc) {
        unsigned ocAct = gc * outputChannelsPerConvGroup + oc;
        for (auto &e : out[b][ocAct]) {
          e += biases[ocAct];
        }
      }
    }
  }
}

void poplibs_test::conv::
convolutionBackward(const std::vector<unsigned> &fwdInputFieldSize,
                    const std::vector<unsigned> &truncationLower,
                    const std::vector<unsigned> &truncationUpper,
                    const std::vector<unsigned> &inputDilation,
                    const std::vector<unsigned> &paddingLower,
                    const std::vector<unsigned> &paddingUpper,
                    const std::vector<bool> &flipInput,
                    const std::vector<unsigned> &kernelSize,
                    const std::vector<unsigned> &kernelTruncationLower,
                    const std::vector<unsigned> &kernelTruncationUpper,
                    const std::vector<unsigned> &kernelDilation,
                    const std::vector<unsigned> &kernelPaddingLower,
                    const std::vector<unsigned> &kernelPaddingUpper,
                    const std::vector<bool> &flipKernel,
                    const std::vector<unsigned> &outputTruncationLower,
                    const std::vector<unsigned> &outputTruncationUpper,
                    const std::vector<unsigned> &stride,
                    const std::vector<unsigned> &outputPaddingLower,
                    const std::vector<unsigned> &outputPaddingUpper,
                    boost::const_multi_array_ref<double, 3> deltasIn,
                    boost::const_multi_array_ref<double, 4> kernel,
                    boost::multi_array_ref<double, 3> deltasOut) {
  if (fwdInputFieldSize.size() != truncationLower.size() ||
      fwdInputFieldSize.size() != truncationUpper.size() ||
      fwdInputFieldSize.size() != inputDilation.size() ||
      fwdInputFieldSize.size() != paddingLower.size() ||
      fwdInputFieldSize.size() != paddingUpper.size() ||
      fwdInputFieldSize.size() != flipInput.size() ||
      fwdInputFieldSize.size() != kernelSize.size() ||
      fwdInputFieldSize.size() != kernelTruncationLower.size() ||
      fwdInputFieldSize.size() != kernelTruncationUpper.size() ||
      fwdInputFieldSize.size() != kernelDilation.size() ||
      fwdInputFieldSize.size() != kernelPaddingLower.size() ||
      fwdInputFieldSize.size() != kernelPaddingUpper.size() ||
      fwdInputFieldSize.size() != flipKernel.size() ||
      fwdInputFieldSize.size() != outputTruncationLower.size() ||
      fwdInputFieldSize.size() != outputTruncationUpper.size() ||
      fwdInputFieldSize.size() != stride.size() ||
      fwdInputFieldSize.size() != outputPaddingLower.size() ||
      fwdInputFieldSize.size() != outputPaddingUpper.size()) {
    throw poplibs_test::poplibs_test_error(
      "Mismatch in number of spatial dimensions."
    );
  }
  auto fwdOutputFieldSize = getOutputFieldSize(fwdInputFieldSize,
                                               kernelSize,
                                               truncationLower,
                                               truncationUpper,
                                               inputDilation,
                                               paddingLower,
                                               paddingUpper,
                                               kernelTruncationLower,
                                               kernelTruncationUpper,
                                               kernelDilation,
                                               kernelPaddingLower,
                                               kernelPaddingUpper,
                                               outputTruncationLower,
                                               outputTruncationUpper,
                                               stride,
                                               outputPaddingLower,
                                               outputPaddingUpper);
  if (product(fwdOutputFieldSize) != deltasIn.shape()[2] ||
      product(fwdInputFieldSize) != deltasOut.shape()[2] ||
      product(kernelSize) != kernel.shape()[3]) {
    throw poplibs_test::poplibs_test_error(
      "Mismatch between tensor size and spatial field size."
    );
  }
  const auto batchSize = deltasIn.shape()[0];
  const auto fwdOutputChannels = deltasIn.shape()[1];
  const auto numConvGroups = kernel.shape()[0];
  const auto fwdOutputChannelsPerConvGroup = kernel.shape()[1];
  if (fwdOutputChannels != fwdOutputChannelsPerConvGroup * numConvGroups) {
    throw poplibs_test::poplibs_test_error("Input channels in kernel do not "
                                          "match activations for grouped conv");
  }

  std::vector<unsigned> paddedKernelSize;
  auto paddedKernel = truncateDilatePadAndFlipKernel(kernel, kernelSize,
                                                     kernelTruncationLower,
                                                     kernelTruncationUpper,
                                                     kernelDilation,
                                                     kernelPaddingLower,
                                                     kernelPaddingUpper,
                                                     flipKernel,
                                                     paddedKernelSize);

  // Pad input.
  const auto numFieldDims = fwdInputFieldSize.size();
  std::vector<unsigned> fwdPaddedInSize(numFieldDims);
  std::vector<unsigned> fwdConvOutSize(numFieldDims);
  const auto deltasInPaddingLower = outputTruncationLower;
  auto deltasInPaddingUpper = outputTruncationUpper;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    fwdPaddedInSize[dim] = fwdInputFieldSize[dim];
    fwdPaddedInSize[dim] -= truncationLower[dim] + truncationUpper[dim];
    fwdPaddedInSize[dim] = getDilatedSize(fwdPaddedInSize[dim],
                                          inputDilation[dim]);
    fwdPaddedInSize[dim] += paddingLower[dim] + paddingUpper[dim];
    fwdConvOutSize[dim] = fwdPaddedInSize[dim] - paddedKernelSize[dim] + 1;
    const auto fwdTruncatedConvOutSize = fwdConvOutSize[dim] -
                                         (outputTruncationLower[dim] +
                                          outputTruncationUpper[dim]);
    if (outputPaddingLower[dim] +
        (fwdTruncatedConvOutSize + stride[dim] - 1) / stride[dim] +
        outputPaddingUpper[dim] != fwdOutputFieldSize[dim]) {
      throw poplibs_test::poplibs_test_error("Output and input tensor "
                                           "dimensions do not match");
    }
    const auto fwdStridingIgnored =
        fwdTruncatedConvOutSize == 0 ? 0 :
                                       (fwdTruncatedConvOutSize - 1) %
                                       stride[dim];
    deltasInPaddingUpper[dim] += fwdStridingIgnored;
  }
  std::vector<bool> noFlipping(numFieldDims);
  auto paddedDeltasIn =
      truncateDilatePadAndFlipActivations(deltasIn, fwdOutputFieldSize,
                                          outputPaddingLower,
                                          outputPaddingUpper,
                                          stride,
                                          deltasInPaddingLower,
                                          deltasInPaddingUpper,
                                          noFlipping);

  const auto fwdInputChannels = deltasOut.shape()[1];
  const auto fwdInputChannelsPerConvGroup = kernel.shape()[2];
  if (fwdInputChannels != fwdInputChannelsPerConvGroup * numConvGroups) {
    throw poplibs_test::poplibs_test_error("Output channels in kernel do not "
                                          "match activations for grouped conv");
  }
  const auto fwdConvOutElements = product(fwdConvOutSize);
  const auto fwdPaddedInElements = product(fwdPaddedInSize);
  boost::multi_array<double, 3>
      convOut(boost::extents[batchSize]
                            [fwdInputChannels]
                            [fwdPaddedInElements]);
  std::fill(convOut.data(), convOut.data() + convOut.num_elements(), 0.0);
  const auto paddedKernelElements = product(paddedKernelSize);
  for (unsigned gc = 0; gc != numConvGroups; ++gc) {
    for (unsigned b = 0; b != batchSize; ++b) {
      // Perform convolution.
      for (unsigned oc = 0; oc != fwdOutputChannelsPerConvGroup; ++oc) {
        unsigned ocAct = gc * fwdOutputChannelsPerConvGroup + oc;
        for (unsigned oe = 0; oe != fwdConvOutElements; ++oe) {
          auto outputIndices = unflattenIndex(fwdConvOutSize, oe);
          for (unsigned ke = 0; ke != paddedKernelElements; ++ke) {
            auto kernelIndices = unflattenIndex(paddedKernelSize, ke);
            std::vector<unsigned> inputIndices;
            if (getInputIndices(fwdPaddedInSize, paddedKernelSize,
                                outputIndices, kernelIndices, inputIndices)) {
              const auto ie = flattenIndex(fwdPaddedInSize, inputIndices);
              for (unsigned ic = 0; ic != fwdInputChannelsPerConvGroup; ++ic) {
                unsigned icAct = gc * fwdInputChannelsPerConvGroup + ic;
                convOut[b][icAct][ie] +=
                    paddedKernel[gc][oc][ic][ke] *
                    paddedDeltasIn[b][ocAct][oe];
              }
            }
          }
        }
      }
    }
  }
  deltasOut =
      truncateDilatePadAndFlipActivationsInverse(convOut, fwdPaddedInSize,
                                                 truncationLower,
                                                 truncationUpper,
                                                 inputDilation,
                                                 paddingLower,
                                                 paddingUpper,
                                                 flipInput);
}

void poplibs_test::conv::
weightUpdate(const std::vector<unsigned> &inputFieldSize,
             const std::vector<unsigned> &truncationLower,
             const std::vector<unsigned> &truncationUpper,
             const std::vector<unsigned> &inputDilation,
             const std::vector<unsigned> &paddingLower,
             const std::vector<unsigned> &paddingUpper,
             const std::vector<bool> &flipInput,
             const std::vector<unsigned> &kernelSize,
             const std::vector<unsigned> &kernelTruncationLower,
             const std::vector<unsigned> &kernelTruncationUpper,
             const std::vector<unsigned> &kernelDilation,
             const std::vector<unsigned> &kernelPaddingLower,
             const std::vector<unsigned> &kernelPaddingUpper,
             const std::vector<bool> &flipKernel,
             const std::vector<unsigned> &outputTruncationLower,
             const std::vector<unsigned> &outputTruncationUpper,
             const std::vector<unsigned> &stride,
             const std::vector<unsigned> &outputPaddingLower,
             const std::vector<unsigned> &outputPaddingUpper,
             double learningRate,
             boost::const_multi_array_ref<double, 3> activations,
             boost::const_multi_array_ref<double, 3> deltas,
             boost::multi_array_ref<double, 4> kernel,
             boost::multi_array_ref<double, 1> biases) {
  if (inputFieldSize.size() != truncationLower.size() ||
      inputFieldSize.size() != truncationUpper.size() ||
      inputFieldSize.size() != inputDilation.size() ||
      inputFieldSize.size() != paddingLower.size() ||
      inputFieldSize.size() != paddingUpper.size() ||
      inputFieldSize.size() != flipInput.size() ||
      inputFieldSize.size() != kernelSize.size() ||
      inputFieldSize.size() != kernelTruncationLower.size() ||
      inputFieldSize.size() != kernelTruncationUpper.size() ||
      inputFieldSize.size() != kernelDilation.size() ||
      inputFieldSize.size() != kernelPaddingLower.size() ||
      inputFieldSize.size() != kernelPaddingUpper.size() ||
      inputFieldSize.size() != flipKernel.size() ||
      inputFieldSize.size() != outputTruncationLower.size() ||
      inputFieldSize.size() != outputTruncationUpper.size() ||
      inputFieldSize.size() != stride.size() ||
      inputFieldSize.size() != outputPaddingLower.size() ||
      inputFieldSize.size() != outputPaddingUpper.size()) {
    throw poplibs_test::poplibs_test_error(
      "Mismatch in number of spatial dimensions."
    );
  }
  auto outputFieldSize = getOutputFieldSize(inputFieldSize,
                                            kernelSize,
                                            truncationLower,
                                            truncationUpper,
                                            inputDilation,
                                            paddingLower,
                                            paddingUpper,
                                            kernelTruncationLower,
                                            kernelTruncationUpper,
                                            kernelDilation,
                                            kernelPaddingLower,
                                            kernelPaddingUpper,
                                            outputTruncationLower,
                                            outputTruncationUpper,
                                            stride,
                                            outputPaddingLower,
                                            outputPaddingUpper);
  if (product(outputFieldSize) != deltas.shape()[2] ||
      product(inputFieldSize) != activations.shape()[2] ||
      product(kernelSize) != kernel.shape()[3]) {
    throw poplibs_test::poplibs_test_error(
      "Mismatch between tensor size and spatial field size."
    );
  }
  auto paddedKernelSize =
      getTransformedSize(kernelSize,
                         kernelTruncationLower,
                         kernelTruncationUpper,
                         kernelDilation,
                         kernelPaddingLower,
                         kernelPaddingUpper);

  // Pad activations.
  std::vector<unsigned> paddedActivationsSize;
  auto paddedActivations =
      truncateDilatePadAndFlipActivations(activations, inputFieldSize,
                                          truncationLower, truncationUpper,
                                          inputDilation,
                                          paddingLower, paddingUpper,
                                          flipInput,
                                          paddedActivationsSize);
  const auto numFieldDims = inputFieldSize.size();
  std::vector<unsigned> fwdConvOutSize(numFieldDims);
  const auto deltasPaddingLower = outputTruncationLower;
  auto deltasPaddingUpper = outputTruncationUpper;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    fwdConvOutSize[dim] = absdiff(paddedActivationsSize[dim],
                                  paddedKernelSize[dim]) + 1;
    const auto fwdTruncatedConvOutSize = fwdConvOutSize[dim] +
                                         outputTruncationLower[dim] +
                                         outputTruncationUpper[dim];
    assert(outputPaddingLower[dim] +
           (fwdTruncatedConvOutSize + stride[dim] - 1) / stride[dim] +
           outputPaddingUpper[dim] == outputFieldSize[dim]);
    const auto fwdStridingIgnored =
        fwdTruncatedConvOutSize == 0 ? 0 :
                                       (fwdTruncatedConvOutSize - 1) %
                                       stride[dim];
    deltasPaddingUpper[dim] += fwdStridingIgnored;
  }
  // Pad deltas.
  std::vector<bool> flipDeltas(numFieldDims);
  auto paddedDeltas =
      truncateDilatePadAndFlipActivations(deltas, outputFieldSize,
                                          outputPaddingLower,
                                          outputPaddingUpper,
                                          stride,
                                          deltasPaddingLower,
                                          deltasPaddingUpper,
                                          flipDeltas);
  const auto batchSize = paddedActivations.shape()[0];
  const auto inputChannels = paddedActivations.shape()[1];
  const auto outputChannels = paddedDeltas.shape()[1];
  const auto numConvGroups = kernel.shape()[0];
  const auto inputChannelsPerConvGroup = kernel.shape()[2];
  const auto outputChannelsPerConvGroup = kernel.shape()[1];
  if (inputChannels != inputChannelsPerConvGroup * numConvGroups) {
    throw poplibs_test::poplibs_test_error("Input channels in kernel do not "
                                         "match channels in activations");
  }
  if (outputChannels != outputChannelsPerConvGroup * numConvGroups) {
    throw poplibs_test::poplibs_test_error("Output channels in kernel do not "
                                         "match channels in activations");
  }

  const auto paddedKernelElements = product(paddedKernelSize);
  boost::multi_array<double, 4>
      paddedWeightDeltas(boost::extents[numConvGroups]
                                       [outputChannelsPerConvGroup]
                                       [inputChannelsPerConvGroup]
                                       [paddedKernelElements]);
  std::fill(paddedWeightDeltas.data(),
            paddedWeightDeltas.data() + paddedWeightDeltas.num_elements(), 0.0);
  const auto paddedDeltasElements = product(fwdConvOutSize);
  for (unsigned gc = 0; gc != numConvGroups; ++gc) {
    for (unsigned b = 0; b != batchSize; ++b) {
      // Perform convolution.
      for (unsigned oc = 0; oc != outputChannelsPerConvGroup; ++oc) {
        unsigned ocAct = gc * outputChannelsPerConvGroup + oc;
        for (unsigned oe = 0; oe != paddedDeltasElements; ++oe) {
          auto outputIndices = unflattenIndex(fwdConvOutSize, oe);
          for (unsigned ke = 0; ke != paddedKernelElements; ++ke) {
            auto kernelIndices = unflattenIndex(paddedKernelSize, ke);
            std::vector<unsigned> inputIndices;
            if (getInputIndices(paddedActivationsSize, paddedKernelSize,
                                outputIndices, kernelIndices, inputIndices)) {
              const auto ie = flattenIndex(paddedActivationsSize, inputIndices);
              for (unsigned ic = 0; ic != inputChannelsPerConvGroup; ++ic) {
                unsigned icAct = gc * inputChannelsPerConvGroup + ic;
                paddedWeightDeltas[gc][oc][ic][ke] +=
                    paddedActivations[b][icAct][ie] *
                    paddedDeltas[b][ocAct][oe];
              }
            }
          }
        }
      }
    }
  }

  auto weightDeltas =
      truncateDilatePadAndFlipKernelInverse(paddedWeightDeltas,
                                            paddedKernelSize,
                                            kernelTruncationLower,
                                            kernelTruncationUpper,
                                            kernelDilation, kernelPaddingLower,
                                            kernelPaddingUpper, flipKernel);

  // Add the weight deltas.
  for (unsigned gc = 0; gc != numConvGroups; ++gc) {
    for (unsigned oc = 0; oc != outputChannelsPerConvGroup; ++oc) {
      for (unsigned ic = 0; ic != inputChannelsPerConvGroup; ++ic) {
        for (unsigned e = 0; e != kernel.shape()[3]; ++e) {
          kernel[gc][oc][ic][e] +=
              learningRate * -weightDeltas[gc][oc][ic][e];
        }
      }
    }
  }

  boost::multi_array<double, 1> biasDeltas(boost::extents[outputChannels]);
  std::fill(biasDeltas.data(),
            biasDeltas.data() + biasDeltas.num_elements(), 0.0);
  for (unsigned b = 0; b != batchSize; ++b) {
    // Compute the bias deltas.
    for (unsigned oc = 0; oc != outputChannels; ++oc) {
      for (const auto &e :deltas[b][oc]) {
        biasDeltas[oc] += e;
      }
    }
  }

  // Add the bias deltas.
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    biases[oc] += learningRate * -biasDeltas[oc];
  }
}

void poplibs_test::conv::
batchNormEstimates(const boost::multi_array_ref<double, 4> actsIn,
                   double eps, bool unbiasedVarEstimate,
                   boost::multi_array_ref<double, 1> mean,
                   boost::multi_array_ref<double, 1> iStdDev) {
  const unsigned batchSize= actsIn.shape()[0];
  const unsigned numChannels = actsIn.shape()[1];
  const unsigned dimY = actsIn.shape()[2];
  const unsigned dimX = actsIn.shape()[3];
  const auto numElems = batchSize * dimX * dimY;

  assert(iStdDev.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double sum =  0;
    double sumSquares = 0;
    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != dimY; ++h) {
        for (unsigned w = 0; w != dimX; ++w) {
          sum += actsIn[b][c][h][w];
          sumSquares += actsIn[b][c][h][w] * actsIn[b][c][h][w];
        }
      }
    }

    // unbiased sample mean
    mean[c] = sum / numElems;
    const auto biasedVar = sumSquares / numElems - mean[c] * mean[c];
    const auto correctedVar = numElems == 1 ?  1.0 :
      (unbiasedVarEstimate ? biasedVar * numElems / (numElems - 1) :
                             biasedVar);
    iStdDev[c] = 1.0 / std::sqrt(correctedVar + eps);
  }
}

void poplibs_test::conv::
batchNormalise(const boost::multi_array_ref<double, 4> acts,
               const boost::multi_array_ref<double, 1> gamma,
               const boost::multi_array_ref<double, 1> beta,
               const boost::multi_array_ref<double, 1> mean,
               const boost::multi_array_ref<double, 1> iStdDev,
               boost::multi_array_ref<double, 4> actsOut,
               boost::multi_array_ref<double, 4> actsWhitened) {

  const unsigned batchSize = acts.shape()[0];
  const unsigned numChannels = acts.shape()[1];
  const unsigned dimY = acts.shape()[2];
  const unsigned dimX = acts.shape()[3];

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);
  assert(mean.shape()[0] == numChannels);
  assert(iStdDev.shape()[0] == numChannels);
  assert(actsOut.shape()[0] == batchSize);
  assert(actsOut.shape()[1] == numChannels);
  assert(actsOut.shape()[2] == dimY);
  assert(actsOut.shape()[3] == dimX);
  assert(actsWhitened.shape()[0] == batchSize);
  assert(actsWhitened.shape()[1] == numChannels);
  assert(actsWhitened.shape()[2] == dimY);
  assert(actsWhitened.shape()[3] == dimX);

  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned h = 0; h != dimY; ++h) {
      for (unsigned w = 0; w != dimX; ++w) {
        for (unsigned c = 0; c != numChannels; ++c) {
          actsWhitened[b][c][h][w] = (acts[b][c][h][w] - mean[c]) * iStdDev[c];
          actsOut[b][c][h][w] = gamma[c] * actsWhitened[b][c][h][w] + beta[c];
        }
      }
    }
  }
}

void poplibs_test::conv::
batchNormGradients(const boost::multi_array_ref<double, 4> actsWhitened,
                   const boost::multi_array_ref<double, 4> gradsIn,
                   const boost::multi_array_ref<double, 1> iStdDev,
                   const boost::multi_array_ref<double, 1> gamma,
                   boost::multi_array_ref<double, 4> gradsOut) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChannels = actsWhitened.shape()[1];
  const unsigned height = actsWhitened.shape()[2];
  const unsigned width = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == height);
  assert(gradsIn.shape()[3] == width);
  assert(gradsOut.shape()[0] == batchSize);
  assert(gradsOut.shape()[1] == numChannels);
  assert(gradsOut.shape()[2] == height);
  assert(gradsOut.shape()[3] == width);

  assert(iStdDev.shape()[0] == numChannels);
  assert(gamma.shape()[0] == numChannels);

  const auto numElements = batchSize * height * width;

  for (unsigned c = 0; c != numChannels; ++c) {
    double sumGradsIn = 0;
    double sumGradsInAndxMu = 0;

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          sumGradsIn += gradsIn[b][c][h][w];
          sumGradsInAndxMu += actsWhitened[b][c][h][w] * gradsIn[b][c][h][w];
        }
      }
    }

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          double out =
            gradsIn[b][c][h][w]
            - actsWhitened[b][c][h][w] * sumGradsInAndxMu / numElements
            - sumGradsIn / numElements;

          gradsOut[b][c][h][w] = out * gamma[c] * iStdDev[c];
        }
      }
    }
  }
}

void poplibs_test::conv::
batchNormParamUpdate(const boost::multi_array_ref<double, 4> actsWhitened,
                     const boost::multi_array_ref<double, 4> gradsIn,
                     double learningRate,
                     boost::multi_array_ref<double, 1> gamma,
                     boost::multi_array_ref<double, 1> beta) {
  const unsigned batchSize = actsWhitened.shape()[0];
  const unsigned numChannels = actsWhitened.shape()[1];
  const unsigned height = actsWhitened.shape()[2];
  const unsigned width = actsWhitened.shape()[3];

  assert(gradsIn.shape()[0] == batchSize);
  assert(gradsIn.shape()[1] == numChannels);
  assert(gradsIn.shape()[2] == height);
  assert(gradsIn.shape()[3] == width);

  assert(gamma.shape()[0] == numChannels);
  assert(beta.shape()[0] == numChannels);

  for (unsigned c = 0; c != numChannels; ++c) {
    double dBeta = 0;
    double dGamma = 0;

    for (unsigned b = 0; b != batchSize; ++b) {
      for (unsigned h = 0; h != height; ++h) {
        for (unsigned w = 0; w != width; ++w) {
          dBeta += gradsIn[b][c][h][w];
          dGamma += actsWhitened[b][c][h][w] * gradsIn[b][c][h][w];
        }
      }
    }
    beta[c] -= learningRate * dBeta;
    gamma[c] -= learningRate * dGamma;
  }
}
