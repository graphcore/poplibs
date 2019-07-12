// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef poplin_ConvParams_hpp
#define poplin_ConvParams_hpp
#include <vector>
#include <tuple>
#include "poplar/Type.hpp"

namespace poplin {

struct ConvParams {
  struct InputTransform {
    // Amount each spatial dimension is truncated before dilation.
    std::vector<unsigned> truncationLower;
    std::vector<unsigned> truncationUpper;
    // Dilation applied to each spatial dimensions after truncation and before
    // padding. Dilation is peformed by placing zeroed elements between the
    // elements of the field.
    std::vector<unsigned> dilation;
    // Padding applied to each spatial dimension after dilation and before
    // flipping.
    std::vector<unsigned> paddingLower;
    std::vector<unsigned> paddingUpper;
    // Whether to flip each spatial dimension. Flipping is applied after
    // padding.
    std::vector<bool> flip;

    InputTransform() = default;
    InputTransform(const std::size_t size);
    InputTransform(std::vector<unsigned> truncationLower,
                   std::vector<unsigned> truncationUpper,
                   std::vector<unsigned> dilation,
                   std::vector<unsigned> paddingLower,
                   std::vector<unsigned> paddingUpper,
                   std::vector<bool> flip);

    friend bool operator<(const ConvParams::InputTransform &a,
                          const ConvParams::InputTransform &b) {
      return std::tie(a.truncationLower,
                      a.truncationUpper,
                      a.dilation,
                      a.paddingLower,
                      a.paddingUpper,
                      a.flip) <
            std::tie(b.truncationLower,
                      b.truncationUpper,
                      b.dilation,
                      b.paddingLower,
                      b.paddingUpper,
                      b.flip);
    }

    friend bool operator==(const ConvParams::InputTransform &a,
                           const ConvParams::InputTransform &b) {
      return std::tie(a.truncationLower,
                      a.truncationUpper,
                      a.dilation,
                      a.paddingLower,
                      a.paddingUpper,
                      a.flip) ==
            std::tie(b.truncationLower,
                      b.truncationUpper,
                      b.dilation,
                      b.paddingLower,
                      b.paddingUpper,
                      b.flip);
    }

    friend bool operator!=(const ConvParams::InputTransform &a,
                           const ConvParams::InputTransform &b) {
      return !(a == b);
    }
  };
  struct OutputTransform {
    // Amount each spatial dimension is truncated before striding.
    std::vector<unsigned> truncationLower;
    std::vector<unsigned> truncationUpper;
    // Striding applied to each spatial dimension after truncation and before
    // padding.
    std::vector<unsigned> stride;
    // Padding applied to each spatial dimension after striding.
    std::vector<unsigned> paddingLower;
    std::vector<unsigned> paddingUpper;

    OutputTransform() = default;
    OutputTransform(const std::size_t size);
    OutputTransform(std::vector<unsigned> truncationLower,
                    std::vector<unsigned> truncationUpper,
                    std::vector<unsigned> striding,
                    std::vector<unsigned> paddingLower,
                    std::vector<unsigned> paddingUpper);

    friend bool operator<(const ConvParams::OutputTransform &a,
                          const ConvParams::OutputTransform &b) {
      return std::tie(a.truncationLower,
                      a.truncationUpper,
                      a.stride,
                      a.paddingLower,
                      a.paddingUpper) <
            std::tie(b.truncationLower,
                     b.truncationUpper,
                     b.stride,
                     b.paddingLower,
                     b.paddingUpper);
    }

    friend bool operator==(const ConvParams::OutputTransform &a,
                          const ConvParams::OutputTransform &b) {
      return std::tie(a.truncationLower,
                      a.truncationUpper,
                      a.stride,
                      a.paddingLower,
                      a.paddingUpper) ==
            std::tie(b.truncationLower,
                     b.truncationUpper,
                     b.stride,
                     b.paddingLower,
                     b.paddingUpper);
    }

    friend bool operator!=(const ConvParams::OutputTransform &a,
                          const ConvParams::OutputTransform &b) {
      return !(a == b);
    }
  };
  poplar::Type inputType;
  poplar::Type outputType;
  // batch size (B)
  std::size_t batchSize;
  // Input field shape for each channel in a batch
  std::vector<std::size_t> inputFieldShape;
  // kernel shape for each channel
  std::vector<std::size_t> kernelShape;
  // input channels (Ci)
  std::size_t inputChannels;
  // output channels (Co)
  std::size_t outputChannels;
  // number of groups in a grouped convolution (G). The input and output
  // channels are divided by G such that G kernels are applied to an input
  // tensors of size {B, {dims}, Ci/G} to produce output tensors of size
  // {B, O{dims}, Co/G}. O{dims} is the output field dimensions
  std::size_t numConvGroups;

  // The transform applied to the input.
  InputTransform inputTransform;
  // The transform applied to the kernel.
  InputTransform kernelTransform;
  // The transform applied to the output.
  OutputTransform outputTransform;

  ConvParams() = default;
  ConvParams(poplar::Type dataType,
             std::size_t batchSize,
             std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape,
             std::size_t inputChannels,
             std::size_t outputChannels,
             std::size_t numConvGroups);
  ConvParams(poplar::Type inputType,
             poplar::Type outputType,
             std::size_t batchSize,
             std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape,
             std::size_t inputChannels,
             std::size_t outputChannels,
             std::size_t numConvGroups);
  ConvParams(poplar::Type inputType,
             poplar::Type outputType,
             std::size_t batchSize,
             std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape,
             std::size_t inputChannels,
             std::size_t outputChannels,
             std::size_t numConvGroups,
             InputTransform inputTransform,
             InputTransform kernelTransform,
             OutputTransform outputTransform);

  /// Return the size of the output of the convolution operation, before
  /// output transformations are applied.
  std::size_t getUntransformedOutputSize(unsigned dim) const;
  /// Return the size of the output.
  std::size_t getOutputSize(unsigned dim) const;
  std::size_t getNumOutputChansPerConvGroup() const { return outputChannels;}
  std::size_t getNumOutputChans() const {
    return outputChannels * numConvGroups;
  }
  std::size_t getInputSize(unsigned dim) const { return inputFieldShape[dim]; }
  std::size_t getNumInputChansPerConvGroup() const { return inputChannels; }
  std::size_t getNumInputChans() const {
    return inputChannels * numConvGroups;
  }
  std::size_t getNumConvGroups() const { return numConvGroups; }
  std::size_t getNumFieldDims() const { return inputFieldShape.size(); }
  std::size_t getBatchSize() const { return batchSize; }

  // Return the size of input in the specified dimension after truncation.
  unsigned getTruncatedInputSize(unsigned dim) const;
  // Return the size of kernel in the specified dimension after truncation.
  unsigned getTruncatedKernelSize(unsigned dim) const;
  // Return the size of input in the specified dimension after applying the
  // input transforms.
  unsigned getTransformedInputSize(unsigned dim) const;
  // Return the size of kernel in the specified dimension after applying the
  // kernel transforms.
  unsigned getTransformedKernelSize(unsigned dim) const;
  // Returns the shape of the output field
  std::vector<size_t> getOutputFieldShape() const;

  void validate() const;
  ConvParams canonicalize() const;

  friend bool operator<(const ConvParams &a, const ConvParams &b) {
  return std::tie(a.inputType,
                  a.outputType,
                  a.batchSize,
                  a.inputFieldShape,
                  a.kernelShape,
                  a.inputChannels,
                  a.outputChannels,
                  a.numConvGroups,
                  a.inputTransform,
                  a.kernelTransform,
                  a.outputTransform) <
         std::tie(b.inputType,
                  b.outputType,
                  b.batchSize,
                  b.inputFieldShape,
                  b.kernelShape,
                  b.inputChannels,
                  b.outputChannels,
                  b.numConvGroups,
                  b.inputTransform,
                  b.kernelTransform,
                  b.outputTransform);
  }

  friend bool operator==(const ConvParams &a, const ConvParams &b) {
    return std::tie(a.inputType,
                    a.outputType,
                    a.batchSize,
                    a.inputFieldShape,
                    a.kernelShape,
                    a.inputChannels,
                    a.outputChannels,
                    a.numConvGroups,
                    a.inputTransform,
                    a.kernelTransform,
                    a.outputTransform) ==
          std::tie(b.inputType,
                    b.outputType,
                    b.batchSize,
                    b.inputFieldShape,
                    b.kernelShape,
                    b.inputChannels,
                    b.outputChannels,
                    b.numConvGroups,
                    b.inputTransform,
                    b.kernelTransform,
                    b.outputTransform);
  }

  friend bool operator!=(const ConvParams &a, const ConvParams &b) {
    return !(a == b);
  }

};

std::ostream& operator<<(std::ostream &os, const ConvParams &p);

std::size_t hash_value(const ConvParams::InputTransform &it);
std::size_t hash_value(const ConvParams::OutputTransform &ot);

} // namespace poplin

namespace std {

template <>
struct hash<poplin::ConvParams::InputTransform> {
  std::size_t operator()(const poplin::ConvParams::InputTransform &it) const;
};

template <>
struct hash<poplin::ConvParams::OutputTransform> {
  std::size_t operator()(const poplin::ConvParams::OutputTransform &ot) const;
};

template <>
struct hash<poplin::ConvParams> {
  std::size_t operator()(const poplin::ConvParams &params) const;
};

} // namespace std

#endif // poplin_ConvParams_hpp
