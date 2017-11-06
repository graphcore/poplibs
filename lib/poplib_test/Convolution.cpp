#include <poplib_test/Convolution.hpp>
#include <poplib_test/exceptions.hpp>

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

static unsigned getPaddedDilatedSize(unsigned size, unsigned dilation,
                                     int paddingLower, int paddingUpper) {
  unsigned dilatedSize = (size - 1) * dilation + 1;
  unsigned paddedSize = paddingLower + dilatedSize + paddingUpper;
  return paddedSize;
}

static std::vector<unsigned>
getPaddedDilatedSize(std::vector<unsigned> inputSize,
                     std::vector<unsigned> dilation,
                     std::vector<int> paddingLower,
                     std::vector<int> paddingUpper) {
  const auto numFieldDims = inputSize.size();
  std::vector<unsigned> outputSize;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outputSize.push_back(getPaddedDilatedSize(inputSize[dim],
                                              dilation[dim],
                                              paddingLower[dim],
                                              paddingUpper[dim]));
  }
  return outputSize;
}

static unsigned
getOutputFieldSize(unsigned inputSize, unsigned inputDilation,
                   int inputPaddingLower, int inputPaddingUpper,
                   unsigned kernelSize, unsigned kernelDilation,
                   int kernelPaddingLower, int kernelPaddingUpper,
                   unsigned stride) {
  auto paddedDilatedInputSize =
      getPaddedDilatedSize(inputSize, inputDilation, inputPaddingLower,
                           inputPaddingUpper);
  auto paddedDilatedKernelSize =
      getPaddedDilatedSize(kernelSize, kernelDilation, kernelPaddingLower,
                           kernelPaddingUpper);
  return absdiff(paddedDilatedInputSize, paddedDilatedKernelSize) / stride + 1;
}

static std::vector<unsigned>
getOutputFieldSize(const std::vector<unsigned> &inputSize,
                   const std::vector<unsigned> &inputDilation,
                   const std::vector<int> &inputPaddingLower,
                   const std::vector<int> &inputPaddingUpper,
                   const std::vector<unsigned> &kernelSize,
                   const std::vector<unsigned> &kernelDilation,
                   const std::vector<int> &kernelPaddingLower,
                   const std::vector<int> &kernelPaddingUpper,
                   const std::vector<unsigned> &stride) {
  const auto numFieldDims = inputSize.size();
  std::vector<unsigned> outputSize;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outputSize.push_back(getOutputFieldSize(inputSize[dim],
                                            inputDilation[dim],
                                            inputPaddingLower[dim],
                                            inputPaddingUpper[dim],
                                            kernelSize[dim],
                                            kernelDilation[dim],
                                            kernelPaddingLower[dim],
                                            kernelPaddingUpper[dim],
                                            stride[dim]));
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
dilateAndPad(boost::const_multi_array_ref<double, 2> in,
             const std::vector<unsigned> &size,
             const std::vector<unsigned> &dilation,
             const std::vector<int> &paddingLower,
             const std::vector<int> &paddingUpper,
             std::vector<unsigned> &paddedSize) {
  const auto numFieldElements = product(size);
  assert(in.shape()[1] == numFieldElements);
  const auto numFieldDims = size.size();
  std::vector<unsigned> dilatedSize(numFieldDims);
  paddedSize.resize(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    dilatedSize[dim] = (size[dim] - 1) * dilation[dim] + 1;
    paddedSize[dim] = dilatedSize[dim] + paddingLower[dim] + paddingUpper[dim];
  }
  const auto dilatedElements = product(dilatedSize);
  boost::multi_array<double, 2>
      dilated(boost::extents[in.shape()[0]][dilatedElements]);
  std::fill(dilated.data(), dilated.data() + dilated.num_elements(), 0.0);
  for (unsigned i = 0; i != in.shape()[0]; ++i) {
    for (unsigned e = 0; e != numFieldElements; ++e) {
      auto indices = unflattenIndex(size, e);
      auto dilatedIndices = dilateIndices(indices, dilation);
      dilated[i][flattenIndex(dilatedSize, dilatedIndices)] = in[i][e];
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
      bool hasPaddedIndex = true;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto index = indices[dim];
        const auto paddedIndex =
            static_cast<int>(index) + paddingLower[dim];
        if (paddedIndex < 0 ||
            static_cast<unsigned>(paddedIndex) >= paddedSize[dim]) {
          hasPaddedIndex = false;
          break;
        }
        paddedIndices.push_back(paddedIndex);
      }
      if (hasPaddedIndex) {
        padded[i][flattenIndex(paddedSize, paddedIndices)] =
            dilated[i][e];
      }
    }
  }
  return padded;
}

static boost::multi_array<double, 3>
dilateAndPadActivations(boost::const_multi_array_ref<double, 3> in,
                        const std::vector<unsigned> &fieldSize,
                        const std::vector<unsigned> inputDilation,
                        const std::vector<int> &paddingLower,
                        const std::vector<int> &paddingUpper,
                        std::vector<unsigned> &paddedFieldSize) {
  const auto numFieldElements = in.shape()[2];
  assert(numFieldElements == product(fieldSize));
  boost::const_multi_array_ref<double, 2>
      kernelFlattened(in.data(),
                      boost::extents[in.num_elements() / numFieldElements]
                                    [numFieldElements]);
  auto paddedFlattened =
      dilateAndPad(kernelFlattened, fieldSize, inputDilation,
                   paddingLower, paddingUpper, paddedFieldSize);
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
dilateAndPadActivations(boost::const_multi_array_ref<double, 3> in,
                        const std::vector<unsigned> &fieldSize,
                        const std::vector<unsigned> inputDilation,
                        const std::vector<int> &paddingLower,
                        const std::vector<int> &paddingUpper) {
  std::vector<unsigned> dummy;
  return dilateAndPadActivations(in, fieldSize, inputDilation, paddingLower,
                                 paddingUpper, dummy);
}

static boost::multi_array<double, 4>
dilateAndPadKernel(boost::const_multi_array_ref<double, 4> kernel,
                   const std::vector<unsigned> &kernelSize,
                   const std::vector<unsigned> &kernelDilation,
                   const std::vector<int> &kernelPaddingLower,
                   const std::vector<int> &kernelPaddingUpper,
                   std::vector<unsigned> &paddedSize) {
  const auto numFieldElements = kernel.shape()[3];
  assert(numFieldElements == product(kernelSize));
  boost::const_multi_array_ref<double, 2>
      kernelFlattened(kernel.data(),
                      boost::extents[kernel.num_elements() / numFieldElements]
                                    [numFieldElements]);
  auto paddedFlattened =
      dilateAndPad(kernelFlattened, kernelSize, kernelDilation,
                   kernelPaddingLower, kernelPaddingUpper, paddedSize);
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
dilateAndPadInverse(boost::const_multi_array_ref<double, 2> padded,
                    const std::vector<unsigned> &paddedSize,
                    const std::vector<unsigned> &dilation,
                    const std::vector<int> &paddingLower,
                    const std::vector<int> &paddingUpper,
                    std::vector<unsigned> &size) {
  assert(padded.shape()[1] == product(paddedSize));
  const auto numFieldDims = paddedSize.size();
  std::vector<unsigned> dilatedSize(numFieldDims);
  size.resize(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    dilatedSize[dim] = paddedSize[dim] - paddingLower[dim] -
                       paddingUpper[dim];
    size[dim] = (dilatedSize[dim] - 1) / dilation[dim] + 1;
  }
  const auto dilatedFieldElements = product(dilatedSize);
  boost::multi_array<double, 2>
      dilated(boost::extents[padded.shape()[0]]
                            [dilatedFieldElements]);
  for (unsigned i = 0; i != padded.shape()[0]; ++i) {
    for (unsigned e = 0; e != dilatedFieldElements; ++e) {
      auto indices = unflattenIndex(dilatedSize, e);
      std::vector<unsigned> paddedIndices;
      bool hasPaddedIndex = true;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto index = indices[dim];
        const auto paddedIndex =
            static_cast<int>(index) + paddingLower[dim];
        if (paddedIndex < 0 ||
            static_cast<unsigned>(paddedIndex) >= paddedSize[dim]) {
          hasPaddedIndex = false;
          break;
        }
        paddedIndices.push_back(paddedIndex);
      }
      if (hasPaddedIndex) {
        dilated[i][e] = padded[i][flattenIndex(paddedSize, paddedIndices)];
      }
    }
  }
  const auto numElements = product(size);
  boost::multi_array<double, 2>
      out(boost::extents[padded.shape()[0]]
                             [numElements]);
  for (unsigned i = 0; i != padded.shape()[0]; ++i) {
    for (unsigned e = 0; e != numElements; ++e) {
      auto indices = unflattenIndex(size, e);
      auto dilatedIndices = dilateIndices(indices, dilation);
      out[i][e] = dilated[i][flattenIndex(dilatedSize, dilatedIndices)];
    }
  }
  return out;
}

static boost::multi_array<double, 3>
dilateAndPadActivationsInverse(
    boost::const_multi_array_ref<double, 3> paddedActs,
    const std::vector<unsigned> &paddedFieldSize,
    const std::vector<unsigned> dilation,
    const std::vector<int> &paddingLower,
    const std::vector<int> &paddingUpper,
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
      dilateAndPadInverse(paddedFlattened, paddedFieldSize, dilation,
                          paddingLower, paddingUpper, fieldSize);
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
dilateAndPadActivationsInverse(
    boost::const_multi_array_ref<double, 3> paddedActs,
    const std::vector<unsigned> &paddedFieldSize,
    const std::vector<unsigned> dilation,
    const std::vector<int> &paddingLower,
    const std::vector<int> &paddingUpper) {
  std::vector<unsigned> dummy;
  return dilateAndPadActivationsInverse(paddedActs, paddedFieldSize,
                                        dilation, paddingLower, paddingUpper,
                                        dummy);
}

static boost::multi_array<double, 4>
dilateAndPadKernelInverse(boost::const_multi_array_ref<double, 4> padded,
                          const std::vector<unsigned> &paddedSize,
                          const std::vector<unsigned> &kernelDilation,
                          const std::vector<int> &kernelPaddingLower,
                          const std::vector<int> &kernelPaddingUpper,
                          std::vector<unsigned> &kernelSize) {
  const auto numFieldElements = padded.shape()[3];
  assert(numFieldElements == product(paddedSize));
  boost::const_multi_array_ref<double, 2>
      paddedFlattened(padded.data(),
                      boost::extents[padded.num_elements() / numFieldElements]
                                    [numFieldElements]);
  auto kernelFlattened =
      dilateAndPadInverse(paddedFlattened, paddedSize, kernelDilation,
                          kernelPaddingLower, kernelPaddingUpper, kernelSize);
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
dilateAndPadKernelInverse(boost::const_multi_array_ref<double, 4> padded,
                          const std::vector<unsigned> &paddedSize,
                          const std::vector<unsigned> &kernelDilation,
                          const std::vector<int> &kernelPaddingLower,
                          const std::vector<int> &kernelPaddingUpper) {
  std::vector<unsigned> dummy;
  return dilateAndPadKernelInverse(padded, paddedSize, kernelDilation,
                                   kernelPaddingLower, kernelPaddingUpper,
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

void poplib_test::conv::
convolution(const std::vector<unsigned> &inputFieldSize,
            const std::vector<unsigned> &inputDilation,
            const std::vector<int> &paddingLower,
            const std::vector<int> &paddingUpper,
            const std::vector<unsigned> &kernelSize,
            const std::vector<unsigned> &kernelDilation,
            const std::vector<int> &kernelPaddingLower,
            const std::vector<int> &kernelPaddingUpper,
            const std::vector<unsigned> &stride,
            boost::const_multi_array_ref<double, 3> in,
            boost::const_multi_array_ref<double, 4> kernel,
            boost::const_multi_array_ref<double, 1> biases,
            boost::multi_array_ref<double, 3> out) {
  if (inputFieldSize.size() != inputDilation.size() ||
      inputFieldSize.size() != paddingLower.size() ||
      inputFieldSize.size() != paddingUpper.size() ||
      inputFieldSize.size() != kernelSize.size() ||
      inputFieldSize.size() != kernelDilation.size() ||
      inputFieldSize.size() != kernelPaddingLower.size() ||
      inputFieldSize.size() != kernelPaddingUpper.size() ||
      inputFieldSize.size() != stride.size()) {
    throw poplib_test::poplib_test_error(
      "Mismatch in number of spatial dimensions."
    );
  }
  if (product(inputFieldSize) != in.shape()[2] ||
      product(kernelSize) != kernel.shape()[3]) {
    throw poplib_test::poplib_test_error(
      "Mismatch between tensor size and spatial field size."
    );
  }
  const auto batchSize = in.shape()[0];
  const auto numConvGroups = kernel.shape()[0];
  const auto inputChannelsPerConvGroup = kernel.shape()[2];
  const auto inputChannels = in.shape()[1];
  if (inputChannels != inputChannelsPerConvGroup * numConvGroups) {
    throw poplib_test::poplib_test_error("Input channels in kernel do not "
                                         "match activations for grouped conv");
  }
  const auto numFieldDims = inputFieldSize.size();
  std::vector<unsigned> paddedKernelSize;
  auto paddedKernel = dilateAndPadKernel(kernel, kernelSize, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper,
                                         paddedKernelSize);
  std::vector<unsigned> paddedFieldSize;
  auto paddedIn = dilateAndPadActivations(in, inputFieldSize, inputDilation,
                                          paddingLower, paddingUpper,
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
          convOut[b][ocAct][oe] += biases[ocAct];
        }
      }
    }
  }

  std::vector<int> noPadding(numFieldDims);
  out = dilateAndPadActivationsInverse(convOut, convOutSize, stride, noPadding,
                                       noPadding);
}

void poplib_test::conv::
convolutionBackward(const std::vector<unsigned> &fwdInputFieldSize,
                    const std::vector<unsigned> &inputDilation,
                    const std::vector<int> &paddingLower,
                    const std::vector<int> &paddingUpper,
                    const std::vector<unsigned> &kernelSize,
                    const std::vector<unsigned> &kernelDilation,
                    const std::vector<int> &kernelPaddingLower,
                    const std::vector<int> &kernelPaddingUpper,
                    const std::vector<unsigned> &stride,
                    boost::const_multi_array_ref<double, 3> deltasIn,
                    boost::const_multi_array_ref<double, 4> kernel,
                    boost::multi_array_ref<double, 3> deltasOut) {
  if (fwdInputFieldSize.size() != inputDilation.size() ||
      fwdInputFieldSize.size() != paddingLower.size() ||
      fwdInputFieldSize.size() != paddingUpper.size() ||
      fwdInputFieldSize.size() != kernelSize.size() ||
      fwdInputFieldSize.size() != kernelDilation.size() ||
      fwdInputFieldSize.size() != kernelPaddingLower.size() ||
      fwdInputFieldSize.size() != kernelPaddingUpper.size() ||
      fwdInputFieldSize.size() != stride.size()) {
    throw poplib_test::poplib_test_error(
      "Mismatch in number of spatial dimensions."
    );
  }
  auto fwdOutputFieldSize = getOutputFieldSize(fwdInputFieldSize,
                                               inputDilation,
                                               paddingLower,
                                               paddingUpper,
                                               kernelSize,
                                               kernelDilation,
                                               kernelPaddingLower,
                                               kernelPaddingUpper,
                                               stride);
  if (product(fwdOutputFieldSize) != deltasIn.shape()[2] ||
      product(fwdInputFieldSize) != deltasOut.shape()[2] ||
      product(kernelSize) != kernel.shape()[3]) {
    throw poplib_test::poplib_test_error(
      "Mismatch between tensor size and spatial field size."
    );
  }
  const auto batchSize = deltasIn.shape()[0];
  const auto fwdOutputChannels = deltasIn.shape()[1];
  const auto numConvGroups = kernel.shape()[0];
  const auto fwdOutputChannelsPerConvGroup = kernel.shape()[1];
  if (fwdOutputChannels != fwdOutputChannelsPerConvGroup * numConvGroups) {
    throw poplib_test::poplib_test_error("Input channels in kernel do not "
                                          "match activations for grouped conv");
  }

  std::vector<unsigned> paddedKernelSize;
  auto paddedKernel = dilateAndPadKernel(kernel, kernelSize, kernelDilation,
                                         kernelPaddingLower,
                                         kernelPaddingUpper,
                                         paddedKernelSize);

  // Pad input.
  const auto numFieldDims = fwdInputFieldSize.size();
  std::vector<unsigned> fwdPaddedInSize(numFieldDims);
  std::vector<unsigned> fwdConvOutSize(numFieldDims);
  std::vector<int> deltasInPaddingUpper(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    fwdPaddedInSize[dim] =
        (fwdInputFieldSize[dim] - 1) * inputDilation[dim] + 1 +
        paddingLower[dim] + paddingUpper[dim];
    fwdConvOutSize[dim] =
        absdiff(fwdPaddedInSize[dim], paddedKernelSize[dim]) + 1;
    if ((fwdConvOutSize[dim] + stride[dim] - 1)/ stride[dim] !=
        fwdOutputFieldSize[dim]) {
      throw poplib_test::poplib_test_error("Output and input tensor "
                                           "dimensions do not match");
    }
    deltasInPaddingUpper[dim] = fwdConvOutSize[dim] -
                          ((fwdOutputFieldSize[dim] - 1) * stride[dim] + 1);
    assert(deltasInPaddingUpper[dim] >= 0 &&
           static_cast<unsigned>(deltasInPaddingUpper[dim]) < stride[dim]);
  }
  std::vector<int> deltasInPaddingLower(numFieldDims);
  auto paddedDeltasIn = dilateAndPadActivations(deltasIn, fwdOutputFieldSize,
                                                stride, deltasInPaddingLower,
                                                deltasInPaddingUpper);

  const auto fwdInputChannels = deltasOut.shape()[1];
  const auto fwdInputChannelsPerConvGroup = kernel.shape()[2];
  if (fwdInputChannels != fwdInputChannelsPerConvGroup * numConvGroups) {
    throw poplib_test::poplib_test_error("Output channels in kernel do not "
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
  deltasOut = dilateAndPadActivationsInverse(convOut, fwdPaddedInSize,
                                             inputDilation, paddingLower,
                                             paddingUpper);
}

void poplib_test::conv::
weightUpdate(const std::vector<unsigned> &inputFieldSize,
             const std::vector<unsigned> &inputDilation,
             const std::vector<int> &paddingLower,
             const std::vector<int> &paddingUpper,
             const std::vector<unsigned> &kernelSize,
             const std::vector<unsigned> &kernelDilation,
             const std::vector<int> &kernelPaddingLower,
             const std::vector<int> &kernelPaddingUpper,
             const std::vector<unsigned> &stride,
             double learningRate,
             boost::const_multi_array_ref<double, 3> activations,
             boost::const_multi_array_ref<double, 3> deltas,
             boost::multi_array_ref<double, 4> kernel,
             boost::multi_array_ref<double, 1> biases) {
  if (inputFieldSize.size() != inputDilation.size() ||
      inputFieldSize.size() != paddingLower.size() ||
      inputFieldSize.size() != paddingUpper.size() ||
      inputFieldSize.size() != kernelSize.size() ||
      inputFieldSize.size() != kernelDilation.size() ||
      inputFieldSize.size() != kernelPaddingLower.size() ||
      inputFieldSize.size() != kernelPaddingUpper.size() ||
      inputFieldSize.size() != stride.size()) {
    throw poplib_test::poplib_test_error("Mismatch in size of spatial field.");
  }
  auto outputFieldSize =
      getOutputFieldSize(inputFieldSize, inputDilation, paddingLower,
                         paddingUpper, kernelSize, kernelDilation,
                         kernelPaddingLower, kernelPaddingUpper,
                         stride);
  if (product(outputFieldSize) != deltas.shape()[2] ||
      product(inputFieldSize) != activations.shape()[2] ||
      product(kernelSize) != kernel.shape()[3]) {
    throw poplib_test::poplib_test_error(
      "Mismatch between tensor size and spatial field size."
    );
  }
  auto paddedKernelSize = getPaddedDilatedSize(kernelSize, kernelDilation,
                                               kernelPaddingLower,
                                               kernelPaddingUpper);

  // Pad activations.
  std::vector<unsigned> paddedActivationsSize;
  auto paddedActivations =
      dilateAndPadActivations(activations, inputFieldSize, inputDilation,
                              paddingLower, paddingUpper,
                              paddedActivationsSize);
  const auto numFieldDims = inputFieldSize.size();
  std::vector<unsigned> paddedDeltasSize(numFieldDims);
  std::vector<int> deltasPaddingUpper(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    paddedDeltasSize[dim] = absdiff(paddedActivationsSize[dim],
                                     paddedKernelSize[dim]) + 1;
    if ((paddedDeltasSize[dim] + stride[dim] - 1) / stride[dim] !=
        outputFieldSize[dim]) {
      throw poplib_test::poplib_test_error("Output and input tensor "
                                           "dimensions do not match");
    }
    deltasPaddingUpper[dim] = paddedDeltasSize[dim] -
                              ((outputFieldSize[dim] - 1) * stride[dim] + 1);
    assert(deltasPaddingUpper[dim] >= 0 &&
           static_cast<unsigned>(deltasPaddingUpper[dim]) < stride[dim]);
  }
  // Pad deltas.
  std::vector<int> deltasPaddingLower(numFieldDims);
  auto paddedDeltas =
      dilateAndPadActivations(deltas, outputFieldSize, stride,
                              deltasPaddingLower, deltasPaddingUpper);
  const auto batchSize = paddedActivations.shape()[0];
  const auto inputChannels = paddedActivations.shape()[1];
  const auto outputChannels = paddedDeltas.shape()[1];
  const auto numConvGroups = kernel.shape()[0];
  const auto inputChannelsPerConvGroup = kernel.shape()[2];
  const auto outputChannelsPerConvGroup = kernel.shape()[1];
  if (inputChannels != inputChannelsPerConvGroup * numConvGroups) {
    throw poplib_test::poplib_test_error("Input channels in kernel do not "
                                         "match channels in activations");
  }
  if (outputChannels != outputChannelsPerConvGroup * numConvGroups) {
    throw poplib_test::poplib_test_error("Output channels in kernel do not "
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
  const auto paddedDeltasElements = product(paddedDeltasSize);
  for (unsigned gc = 0; gc != numConvGroups; ++gc) {
    for (unsigned b = 0; b != batchSize; ++b) {
      // Perform convolution.
      for (unsigned oc = 0; oc != outputChannelsPerConvGroup; ++oc) {
        unsigned ocAct = gc * outputChannelsPerConvGroup + oc;
        for (unsigned oe = 0; oe != paddedDeltasElements; ++oe) {
          auto outputIndices = unflattenIndex(paddedDeltasSize, oe);
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
      dilateAndPadKernelInverse(paddedWeightDeltas, paddedKernelSize,
                                kernelDilation, kernelPaddingLower,
                                kernelPaddingUpper);

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
    for (unsigned e = 0; e != paddedDeltasElements; ++e) {
      for (unsigned oc = 0; oc != outputChannels; ++oc) {
        biasDeltas[oc] += paddedDeltas[b][oc][e];
      }
    }
  }

  // Add the bias deltas.
  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    biases[oc] += learningRate * -biasDeltas[oc];
  }
}

void poplib_test::conv::
batchNormEstimates(const boost::multi_array_ref<double, 4> actsIn,
                   double eps,
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
    iStdDev[c] =
        1.0 / std::sqrt(sumSquares / numElems - mean[c] * mean[c] + eps);
  }
}

void poplib_test::conv::
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

void poplib_test::conv::
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

void poplib_test::conv::
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
