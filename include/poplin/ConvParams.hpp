// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Data types for convolution parameters.
 *
 */

#ifndef poplin_ConvParams_hpp
#define poplin_ConvParams_hpp
#include "poplar/Type.hpp"
#include <tuple>
#include <vector>

namespace poplin {

struct ConvParams {
  struct InputTransform {
    /// The position where the lower end of each spatial dimension is truncated
    ///  before dilation.
    std::vector<unsigned> truncationLower;
    /// The position where the upper end of each spatial dimension is truncated
    /// before dilation.
    std::vector<unsigned> truncationUpper;
    /// Dilation applied to each spatial dimensions after truncation and before
    /// padding. Dilation is performed by placing a number of zeroed elements
    /// between the elements of the field.
    std::vector<unsigned> dilation;
    /// Padding applied to each spatial dimension after dilation and before
    /// flipping.
    std::vector<unsigned> paddingLower;
    /// Padding applied to each spatial dimension after dilation and before
    /// flipping.
    std::vector<unsigned> paddingUpper;
    /// If true, each spatial dimension is flipped after being padded.
    std::vector<bool> flip;

    InputTransform() = default;
    InputTransform(const std::size_t size);
    /**
     * \param truncationLower Where to truncate the lower end of each
     *                        dimension.
     * \param truncationUpper Where to truncate the upper end of each
     *                        dimension.
     * \param dilation        Dilation to apply to each dimension.
     * \param paddingLower    How much to pad the lower end of each dimension.
     * \param paddingUpper    How much to pad the upper end of each dimension.
     * \param flip            If true, each spatial dimension is flipped after
     *                        being padded.
     */
    InputTransform(std::vector<unsigned> truncationLower,
                   std::vector<unsigned> truncationUpper,
                   std::vector<unsigned> dilation,
                   std::vector<unsigned> paddingLower,
                   std::vector<unsigned> paddingUpper, std::vector<bool> flip);

    friend bool operator<(const InputTransform &a, const InputTransform &b);
    friend bool operator==(const InputTransform &a, const InputTransform &b);
    friend bool operator!=(const InputTransform &a, const InputTransform &b);
  };

  struct OutputTransform {
    /// The position where the lower end of each spatial dimension is truncated
    ///  before dilation.
    std::vector<unsigned> truncationLower;
    /// The position where the upper end of each spatial dimension is truncated
    ///  before dilation.
    std::vector<unsigned> truncationUpper;
    /// Striding applied to each spatial dimension after truncation and before
    /// padding.
    std::vector<unsigned> stride;
    /// Padding applied to lower end of each spatial dimension after striding.
    std::vector<unsigned> paddingLower;
    /// Padding applied to upper end of each spatial dimension after striding.
    std::vector<unsigned> paddingUpper;

    OutputTransform() = default;
    OutputTransform(const std::size_t size);
    /**
     * \param truncationLower Where to truncate the lower end of each dimension.
     * \param truncationUpper Where to truncate the upper end of each dimension.
     * \param striding        Stride to use in convolution.
     * \param paddingLower    How much to pad the lower end of each dimension.
     * \param paddingUpper    How much to pad the upper end of each dimension.
     */
    OutputTransform(std::vector<unsigned> truncationLower,
                    std::vector<unsigned> truncationUpper,
                    std::vector<unsigned> striding,
                    std::vector<unsigned> paddingLower,
                    std::vector<unsigned> paddingUpper);

    friend bool operator<(const OutputTransform &a, const OutputTransform &b);
    friend bool operator==(const OutputTransform &a, const OutputTransform &b);
    friend bool operator!=(const OutputTransform &a, const OutputTransform &b);
  };

  poplar::Type inputType;
  poplar::Type outputType;
  /// Batch size (B).
  std::size_t batchSize;
  /// Input field shape for each channel in a batch.
  std::vector<std::size_t> inputFieldShape;
  /// Kernel shape for each channel.
  std::vector<std::size_t> kernelShape;
  /// Input channels per conv group (Ci).
  std::size_t inputChannelsPerConvGroup;
  /// Output channels per group (Co).
  std::size_t outputChannelsPerConvGroup;
  /// Number of groups in a grouped convolution (G). The input and output
  /// channels are divided by G such that G kernels are applied to an input
  /// tensors of size [B, O{dims}, Ci/G] to produce output tensors of size
  /// [B, O{dims}, Co/G]. O{dims} is the output field dimensions.
  std::size_t numConvGroups;

  /// The transform applied to the input.
  InputTransform inputTransform;
  /// The transform applied to the kernel.
  InputTransform kernelTransform;
  /// The transform applied to the output.
  OutputTransform outputTransform;

  ConvParams() = default;
  /*
   * \param dataType
   * \param batchSize
   * \param inputFieldShape
   * \param kernelShape
   * \param inputChannels
   * \param outputChannels
   * \param numConvGroups
   */
  ConvParams(poplar::Type dataType, std::size_t batchSize,
             std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape, std::size_t inputChannels,
             std::size_t outputChannels, std::size_t numConvGroups);
  /*
   * \param inputType
   * \param outputType
   * \param batchSize
   * \param inputFieldShape
   * \param kernelShape
   * \param inputChannels
   * \param outputChannels
   * \param numConvGroups
   */
  ConvParams(poplar::Type inputType, poplar::Type outputType,
             std::size_t batchSize, std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape, std::size_t inputChannels,
             std::size_t outputChannels, std::size_t numConvGroups);
  /*
   * \param inputType
   * \param outputType
   * \param batchSize
   * \param inputFieldShape
   * \param kernelShape
   * \param inputChannels
   * \param outputChannels
   * \param numConvGroups
   * \param inputTransform
   * \param kernelTransform
   * \param outputTransform
   */
  ConvParams(poplar::Type inputType, poplar::Type outputType,
             std::size_t batchSize, std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape, std::size_t inputChannels,
             std::size_t outputChannels, std::size_t numConvGroups,
             InputTransform inputTransform, InputTransform kernelTransform,
             OutputTransform outputTransform);

  /// Return the size of the output of the convolution operation, before
  /// output transformations are applied.
  std::size_t getUntransformedOutputSize(unsigned dim) const;
  /// Return the size of the output.
  std::size_t getOutputSize(unsigned dim) const;
  /// Return the number of output channels per group.
  std::size_t getNumOutputChansPerConvGroup() const {
    return outputChannelsPerConvGroup;
  }
  /// Return the number of output channels.
  std::size_t getNumOutputChans() const {
    return outputChannelsPerConvGroup * numConvGroups;
  }
  /// Return the input size.
  std::size_t getInputSize(unsigned dim) const { return inputFieldShape[dim]; }
  /// Return the number of input channels per group.
  std::size_t getNumInputChansPerConvGroup() const {
    return inputChannelsPerConvGroup;
  }
  /// Return the number of input channels per group.
  std::size_t getNumInputChans() const {
    return inputChannelsPerConvGroup * numConvGroups;
  }
  /// Return the number of convolution groups.
  std::size_t getNumConvGroups() const { return numConvGroups; }
  /// Return the number of dimensions of the input field.
  std::size_t getNumFieldDims() const { return inputFieldShape.size(); }
  /// Return the shape of the input field.
  std::vector<std::size_t> getInputFieldShape() const {
    return inputFieldShape;
  }
  /// Return the shape of the kernel.
  std::vector<std::size_t> getKernelShape() const { return kernelShape; }
  /// Return the batch size.
  std::size_t getBatchSize() const { return batchSize; }

  /// Return the size of input in the specified dimension after truncation.
  unsigned getTruncatedInputSize(unsigned dim) const;
  /// Return the size of kernel in the specified dimension after truncation.
  unsigned getTruncatedKernelSize(unsigned dim) const;
  /// Return the size of input in the specified dimension after applying the
  /// input transforms.
  unsigned getTransformedInputSize(unsigned dim) const;
  /// Return the size of kernel in the specified dimension after applying the
  /// kernel transforms.
  unsigned getTransformedKernelSize(unsigned dim) const;
  /// Returns the shape of the output field
  std::vector<size_t> getOutputFieldShape() const;

  void validate() const;
  ConvParams canonicalize() const;

  friend bool operator<(const ConvParams &a, const ConvParams &b);
  friend bool operator==(const ConvParams &a, const ConvParams &b);
  friend bool operator!=(const ConvParams &a, const ConvParams &b);
};

std::ostream &operator<<(std::ostream &os, const ConvParams &p);
std::istream &operator>>(std::istream &is, ConvParams &p);

std::size_t hash_value(const ConvParams::InputTransform &it);
std::size_t hash_value(const ConvParams::OutputTransform &ot);

} // namespace poplin

namespace std {

template <> struct hash<poplin::ConvParams::InputTransform> {
  std::size_t operator()(const poplin::ConvParams::InputTransform &it) const;
};

template <> struct hash<poplin::ConvParams::OutputTransform> {
  std::size_t operator()(const poplin::ConvParams::OutputTransform &ot) const;
};

template <> struct hash<poplin::ConvParams> {
  std::size_t operator()(const poplin::ConvParams &params) const;
};

} // namespace std

#endif // poplin_ConvParams_hpp
