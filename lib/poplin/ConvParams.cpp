// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplin/ConvParams.hpp"
#include "ConvUtilInternal.hpp"
#include "poplibs_support/StructHelper.hpp"
#include "poplibs_support/print.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <unordered_map>

namespace poplin {

using InputTransform = ConvParams::InputTransform;
using OutputTransform = ConvParams::OutputTransform;

static constexpr auto inputTransformHelper = poplibs_support::makeStructHelper(
    &InputTransform::truncationLower, &InputTransform::truncationUpper,
    &InputTransform::dilation, &InputTransform::paddingLower,
    &InputTransform::paddingUpper, &InputTransform::flip);

bool operator<(const InputTransform &a, const InputTransform &b) {
  return inputTransformHelper.lt(a, b);
}

bool operator==(const InputTransform &a, const InputTransform &b) {
  return inputTransformHelper.eq(a, b);
}

bool operator!=(const InputTransform &a, const InputTransform &b) {
  return !(a == b);
}

static constexpr auto outputTransformHelper = poplibs_support::makeStructHelper(
    &OutputTransform::truncationLower, &OutputTransform::truncationUpper,
    &OutputTransform::stride, &OutputTransform::paddingLower,
    &OutputTransform::paddingUpper);

bool operator<(const OutputTransform &a, const OutputTransform &b) {
  return outputTransformHelper.lt(a, b);
}

bool operator==(const OutputTransform &a, const OutputTransform &b) {
  return outputTransformHelper.eq(a, b);
}

bool operator!=(const OutputTransform &a, const OutputTransform &b) {
  return !(a == b);
}

static constexpr auto convParamsHelper = poplibs_support::makeStructHelper(
    &ConvParams::inputType, &ConvParams::outputType, &ConvParams::batchSize,
    &ConvParams::inputFieldShape, &ConvParams::kernelShape,
    &ConvParams::inputChannelsPerConvGroup,
    &ConvParams::outputChannelsPerConvGroup, &ConvParams::numConvGroups,
    &ConvParams::inputTransform, &ConvParams::kernelTransform,
    &ConvParams::outputTransform);

bool operator<(const ConvParams &a, const ConvParams &b) {
  return convParamsHelper.lt(a, b);
}

bool operator==(const ConvParams &a, const ConvParams &b) {
  return convParamsHelper.eq(a, b);
}

bool operator!=(const ConvParams &a, const ConvParams &b) { return !(a == b); }

InputTransform::InputTransform(std::vector<unsigned> truncationLower_,
                               std::vector<unsigned> truncationUpper_,
                               std::vector<unsigned> dilation_,
                               std::vector<unsigned> paddingLower_,
                               std::vector<unsigned> paddingUpper_,
                               std::vector<bool> flip_)
    : truncationLower(std::move(truncationLower_)),
      truncationUpper(std::move(truncationUpper_)),
      dilation(std::move(dilation_)), paddingLower(std::move(paddingLower_)),
      paddingUpper(std::move(paddingUpper_)), flip(flip_) {}

InputTransform::InputTransform(const std::size_t size)
    : InputTransform(
          std::vector<unsigned>(size, 0), std::vector<unsigned>(size, 0),
          std::vector<unsigned>(size, 1), std::vector<unsigned>(size, 0),
          std::vector<unsigned>(size, 0), std::vector<bool>(size, false)) {}

OutputTransform::OutputTransform(std::vector<unsigned> truncationLower_,
                                 std::vector<unsigned> truncationUpper_,
                                 std::vector<unsigned> stride_,
                                 std::vector<unsigned> paddingLower_,
                                 std::vector<unsigned> paddingUpper_)
    : truncationLower(std::move(truncationLower_)),
      truncationUpper(std::move(truncationUpper_)), stride(std::move(stride_)),
      paddingLower(std::move(paddingLower_)),
      paddingUpper(std::move(paddingUpper_)) {}

OutputTransform::OutputTransform(const std::size_t size)
    : OutputTransform(
          std::vector<unsigned>(size, 0), std::vector<unsigned>(size, 0),
          std::vector<unsigned>(size, 1), std::vector<unsigned>(size, 0),
          std::vector<unsigned>(size, 0)) {}

ConvParams::ConvParams(poplar::Type inputType_, poplar::Type outputType_,
                       std::size_t batchSize_,
                       std::vector<std::size_t> inputFieldShape_,
                       std::vector<std::size_t> kernelShape_,
                       std::size_t inputChannelsPerConvGroup_,
                       std::size_t outputChannelsPerConvGroup_,
                       std::size_t numConvGroups_,
                       InputTransform inputTransform_,
                       InputTransform kernelTransform_,
                       OutputTransform outputTransform_)
    : inputType(std::move(inputType_)), outputType(std::move(outputType_)),
      batchSize(batchSize_), inputFieldShape(std::move(inputFieldShape_)),
      kernelShape(std::move(kernelShape_)),
      inputChannelsPerConvGroup(inputChannelsPerConvGroup_),
      outputChannelsPerConvGroup(outputChannelsPerConvGroup_),
      numConvGroups(numConvGroups_), inputTransform(inputTransform_),
      kernelTransform(kernelTransform_), outputTransform(outputTransform_) {}

void ConvParams::validate() const {
  const auto numFieldDims = inputFieldShape.size();
  if (kernelShape.size() != numFieldDims) {
    throw poputil::poplibs_error("Number of kernel field dimensions does not"
                                 "match the number of input field dimensions");
  }
  for (const auto stride : outputTransform.stride) {
    if (stride == 0) {
      throw poputil::poplibs_error("Stride must be non zero");
    }
  }
  for (const auto dilation : inputTransform.dilation) {
    if (dilation == 0) {
      throw poputil::poplibs_error("Input dilation must be non zero."
                                   " Dilation = 1 results in no dilation");
    }
  }
  for (const auto dilation : kernelTransform.dilation) {
    if (dilation == 0) {
      throw poputil::poplibs_error("Kernel dilation must be non zero."
                                   " Dilation = 1 results in no dilation");
    }
  }
  const std::pair<std::size_t, const char *> sizes[] = {
      {inputTransform.truncationLower.size(), "input truncation (lower)"},
      {inputTransform.truncationUpper.size(), "input truncation (upper)"},
      {inputTransform.dilation.size(), "input dilation"},
      {inputTransform.paddingLower.size(), "input padding (lower)"},
      {inputTransform.paddingUpper.size(), "input padding (upper)"},
      {inputTransform.flip.size(), "input flip"},
      {kernelTransform.truncationLower.size(), "kernel truncation (lower)"},
      {kernelTransform.truncationUpper.size(), "kernel truncation (upper)"},
      {kernelTransform.dilation.size(), "kernel dilation"},
      {kernelTransform.paddingLower.size(), "kernel padding (lower)"},
      {kernelTransform.paddingUpper.size(), "kernel padding (upper)"},
      {kernelTransform.flip.size(), "kernel flip"},
      {outputTransform.truncationLower.size(), "output truncation (lower)"},
      {outputTransform.truncationUpper.size(), "output truncation (upper)"},
      {outputTransform.stride.size(), "stride"},
      {outputTransform.paddingLower.size(), "output padding (lower)"},
      {outputTransform.paddingUpper.size(), "output padding (upper)"},
  };
  for (const auto &entry : sizes) {
    if (entry.first != numFieldDims) {
      throw poputil::poplibs_error(std::string("Number of ") + entry.second +
                                   " dimensions does not match the number of "
                                   "field dimensions");
    }
  }
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (inputTransform.truncationLower[dim] +
            inputTransform.truncationUpper[dim] >
        inputFieldShape[dim]) {
      throw poputil::poplibs_error("Truncation for dimension " +
                                   std::to_string(dim) +
                                   " truncates by more than the size of the "
                                   "field");
    }
    if (kernelTransform.truncationLower[dim] +
            kernelTransform.truncationUpper[dim] >
        kernelShape[dim]) {
      throw poputil::poplibs_error("Truncation for dimension " +
                                   std::to_string(dim) +
                                   " truncates by more than the size of the "
                                   "kernel");
    }
    const auto transformedInputSize = getTransformedInputSize(dim);
    const auto transformedKernelSize = getTransformedKernelSize(dim);
    if (transformedKernelSize == 0) {
      throw poputil::poplibs_error("Transformed kernel for dimension " +
                                   std::to_string(dim) + " has zero size");
    }

    if (transformedInputSize < transformedKernelSize) {
      throw poputil::poplibs_error("Transformed input size for dimension " +
                                   std::to_string(dim) +
                                   " is less than the transformed kernel size");
    }
    const auto convOutSize = getUntransformedOutputSize(dim);
    if (outputTransform.truncationLower[dim] +
            outputTransform.truncationUpper[dim] >
        convOutSize) {
      throw poputil::poplibs_error("Output truncation for dimension " +
                                   std::to_string(dim) +
                                   " truncates by more than the size of the "
                                   "convolution output");
    }
  }
}

ConvParams::ConvParams(poplar::Type inputType_, poplar::Type outputType_,
                       std::size_t batchSize_,
                       std::vector<std::size_t> inputFieldShape_,
                       std::vector<std::size_t> kernelShape_,
                       std::size_t inputChannelsPerConvGroup_,
                       std::size_t outputChannelsPerConvGroup_,
                       std::size_t numConvGroups_)
    : ConvParams(inputType_, outputType_, batchSize_, inputFieldShape_,
                 kernelShape_, inputChannelsPerConvGroup_,
                 outputChannelsPerConvGroup_, numConvGroups_,
                 InputTransform(inputFieldShape_.size()),
                 InputTransform(inputFieldShape_.size()),
                 OutputTransform(inputFieldShape_.size())) {}

ConvParams::ConvParams(poplar::Type dataType_, std::size_t batchSize_,
                       std::vector<std::size_t> inputFieldShape_,
                       std::vector<std::size_t> kernelShape_,
                       std::size_t inputChannelsPerConvGroup_,
                       std::size_t outputChannelsPerConvGroup_,
                       std::size_t numConvGroups_)
    : ConvParams(dataType_, dataType_, batchSize_, inputFieldShape_,
                 kernelShape_, inputChannelsPerConvGroup_,
                 outputChannelsPerConvGroup_, numConvGroups_) {}

std::ostream &operator<<(std::ostream &os, const ConvParams &p) {
  os << "Params: inputType                  " << p.inputType << "\n";
  os << "        outputType                 " << p.outputType << "\n";
  os << "        batchSize                  " << p.batchSize << "\n";
  os << "        numConvGroups              " << p.numConvGroups << "\n";
  os << "        inputFieldShape            ";
  printContainer(p.inputFieldShape, os);
  os << "\n";
  os << "        kernelShape                ";
  printContainer(p.kernelShape, os);
  os << "\n";
  os << "        inputChannelsPerConvGroup  ";
  os << p.getNumInputChansPerConvGroup() << "\n";
  os << "        outputChannelsPerConvGroup ";
  os << p.getNumOutputChansPerConvGroup() << "\n";
  os << "        inputTruncationLower       ";
  printContainer(p.inputTransform.truncationLower, os);
  os << "\n";
  os << "        inputTruncationUpper       ";
  printContainer(p.inputTransform.truncationUpper, os);
  os << "\n";
  os << "        inputDilation              ";
  printContainer(p.inputTransform.dilation, os);
  os << "\n";
  os << "        inputPaddingLower          ";
  printContainer(p.inputTransform.paddingLower, os);
  os << "\n";
  os << "        inputPaddingUpper          ";
  printContainer(p.inputTransform.paddingUpper, os);
  os << "\n";
  os << "        flipInput                  ";
  printContainer(p.inputTransform.flip, os);
  os << "\n";
  os << "        kernelTruncationLower      ";
  printContainer(p.kernelTransform.truncationLower, os);
  os << "\n";
  os << "        kernelTruncationUpper      ";
  printContainer(p.kernelTransform.truncationUpper, os);
  os << "\n";
  os << "        kernelDilation             ";
  printContainer(p.kernelTransform.dilation, os);
  os << "\n";
  os << "        kernelPaddingLower         ";
  printContainer(p.kernelTransform.paddingLower, os);
  os << "\n";
  os << "        kernelPaddingUpper         ";
  printContainer(p.kernelTransform.paddingUpper, os);
  os << "\n";
  os << "        flipKernel                 ";
  printContainer(p.kernelTransform.flip, os);
  os << "\n";
  os << "        outputTruncationLower      ";
  printContainer(p.outputTransform.truncationLower, os);
  os << "\n";
  os << "        outputTruncationUpper      ";
  printContainer(p.outputTransform.truncationUpper, os);
  os << "\n";
  os << "        stride                     ";
  printContainer(p.outputTransform.stride, os);
  os << "\n";
  os << "        outputPaddingLower         ";
  printContainer(p.outputTransform.paddingLower, os);
  os << "\n";
  os << "        outputPaddingUpper         ";
  printContainer(p.outputTransform.paddingUpper, os);
  os << "\n";
  os << "        outputFieldShape           ";
  printContainer(p.getOutputFieldShape(), os);
  os << "\n";
  return os;
}

static const std::unordered_map<std::string, poplar::Type> typeMap{
    {"half", poplar::HALF}, {"float", poplar::FLOAT}};

std::istream &operator>>(std::istream &is, ConvParams &p) {
  namespace pt = boost::property_tree;

  pt::ptree root;
  pt::json_parser::read_json(is, root);

  if (const auto dataType = root.get_optional<std::string>("dataType")) {
    p.inputType = typeMap.at(*dataType);
    p.outputType = typeMap.at(*dataType);
  }

  if (const auto inputType = root.get_optional<std::string>("inputType")) {
    p.inputType = typeMap.at(*inputType);
  }

  if (const auto outputType = root.get_optional<std::string>("outputType")) {
    p.outputType = typeMap.at(*outputType);
  }

  const auto getScalar = [](const pt::ptree &node, const char *name,
                            auto &field) {
    using T = std::remove_reference_t<decltype(field)>;
    if (const auto child = node.get_optional<T>(name)) {
      field = *child;
    }
  };

  getScalar(root, "batchSize", p.batchSize);
  getScalar(root, "numConvGroups", p.numConvGroups);
  getScalar(root, "inputChannelsPerConvGroup", p.inputChannelsPerConvGroup);
  getScalar(root, "outputChannelsPerConvGroup", p.outputChannelsPerConvGroup);

  const auto getVector = [](const pt::ptree &node, const char *name,
                            auto &field) {
    using T = std::remove_reference_t<decltype(field)>;
    if (const auto child = node.get_child_optional(name)) {
      T values;
      for (const auto &item : *child) {
        values.push_back(item.second.get_value<typename T::value_type>());
      }

      field = std::move(values);
    }
  };

  getVector(root, "inputFieldShape", p.inputFieldShape);
  getVector(root, "kernelShape", p.kernelShape);
  if (p.inputFieldShape.size() != p.kernelShape.size()) {
    throw poputil::poplibs_error(
        "Input and kernel must have the same number of dimensions");
  }

  p.inputTransform = InputTransform(p.inputFieldShape.size());
  if (const auto inputTransform = root.get_child_optional("inputTransform")) {
    getVector(*inputTransform, "truncationLower",
              p.inputTransform.truncationLower);
    getVector(*inputTransform, "truncationUpper",
              p.inputTransform.truncationUpper);
    getVector(*inputTransform, "dilation", p.inputTransform.dilation);
    getVector(*inputTransform, "paddingLower", p.inputTransform.paddingLower);
    getVector(*inputTransform, "paddingUpper", p.inputTransform.paddingUpper);
    getVector(*inputTransform, "flip", p.inputTransform.flip);
  }

  p.kernelTransform = InputTransform(p.inputFieldShape.size());
  if (const auto kernelTransform = root.get_child_optional("kernelTransform")) {
    getVector(*kernelTransform, "truncationLower",
              p.kernelTransform.truncationLower);
    getVector(*kernelTransform, "truncationUpper",
              p.kernelTransform.truncationUpper);
    getVector(*kernelTransform, "dilation", p.kernelTransform.dilation);
    getVector(*kernelTransform, "paddingLower", p.kernelTransform.paddingLower);
    getVector(*kernelTransform, "paddingUpper", p.kernelTransform.paddingUpper);
    getVector(*kernelTransform, "flip", p.kernelTransform.flip);
  }

  p.outputTransform = OutputTransform(p.inputFieldShape.size());
  if (const auto outputTransform = root.get_child_optional("outputTransform")) {
    getVector(*outputTransform, "truncationLower",
              p.outputTransform.truncationLower);
    getVector(*outputTransform, "truncationUpper",
              p.outputTransform.truncationUpper);
    getVector(*outputTransform, "stride", p.outputTransform.stride);
    getVector(*outputTransform, "paddingLower", p.outputTransform.paddingLower);
    getVector(*outputTransform, "paddingUpper", p.outputTransform.paddingUpper);
  }

  return is;
}

std::size_t hash_value(const ConvParams::InputTransform &it) {
  return std::hash<ConvParams::InputTransform>()(it);
}

std::size_t hash_value(const ConvParams::OutputTransform &ot) {
  return std::hash<ConvParams::OutputTransform>()(ot);
}

ConvParams ConvParams::canonicalize() const {
  validate();
  ConvParams newParams = *this;
  const auto numFieldDims = getNumFieldDims();
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
    auto nonZeroRange = getOutputRangeForKernelRange(
        dim, {0, newParams.getOutputSize(dim)}, {0, newParams.kernelShape[dim]},
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
    auto truncatedConvOutSize = newParams.getUntransformedOutputSize(dim) -
                                (outputTruncationLower + outputTruncationUpper);
    const auto ignored =
        (truncatedConvOutSize - 1) % newParams.outputTransform.stride[dim];
    outputTruncationUpper += ignored;
    truncatedConvOutSize -= ignored;
    // Avoid unnecessary striding.
    if (truncatedConvOutSize == 1) {
      newParams.outputTransform.stride[dim] = 1;
    }
    // Compute input elements that are ignored.
    auto inputUsedRange = getInputRange(
        dim, {0, outSize}, {0, newParams.kernelShape[dim]}, newParams);
    // Truncate and pad the input so the number of ignored elements can
    // be determined directly from the input truncation.
    assert(inputUsedRange.first != inputUsedRange.second);
    const auto inputIgnoredLower = inputUsedRange.first;
    const auto inputIgnoredUpper =
        newParams.getInputSize(dim) - inputUsedRange.second;
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
    auto kernelUsedRange = getKernelRange(
        dim, {0, outSize}, {0, newParams.getInputSize(dim)}, newParams);
    // Truncate and pad the kernel so the number of ignored elements can
    // be determined directly from the kernel truncation.
    assert(kernelUsedRange.first != kernelUsedRange.second);
    const auto kernelIgnoredLower = kernelUsedRange.first;
    const auto kernelIgnoredUpper =
        newParams.kernelShape[dim] - kernelUsedRange.second;
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
        newParams.kernelTransform.flip[dim]
            ? newParams.kernelTransform.paddingUpper[dim]
            : newParams.kernelTransform.paddingLower[dim];
    auto &flippedKernelPaddingUpper =
        newParams.kernelTransform.flip[dim]
            ? newParams.kernelTransform.paddingLower[dim]
            : newParams.kernelTransform.paddingUpper[dim];
    auto &flippedPaddingLower =
        newParams.inputTransform.flip[dim]
            ? newParams.inputTransform.paddingUpper[dim]
            : newParams.inputTransform.paddingLower[dim];
    auto &flippedPaddingUpper =
        newParams.inputTransform.flip[dim]
            ? newParams.inputTransform.paddingLower[dim]
            : newParams.inputTransform.paddingUpper[dim];
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
    excessPaddingLower = std::min(
        {flippedPaddingLower, outputTruncationLower,
         static_cast<unsigned>(newParams.getUntransformedOutputSize(dim) - 1)});
    flippedPaddingLower -= excessPaddingLower;
    outputTruncationLower -= excessPaddingLower;
    excessPaddingUpper = std::min(
        {flippedPaddingUpper, outputTruncationUpper,
         static_cast<unsigned>(newParams.getUntransformedOutputSize(dim) - 1)});
    flippedPaddingUpper -= excessPaddingUpper;
    outputTruncationUpper -= excessPaddingUpper;

    // Avoid unnecessary flipping / dilation.
    if (newParams.inputFieldShape[dim] <=
        newParams.inputTransform.truncationLower[dim] + 1 +
            newParams.inputTransform.truncationUpper[dim]) {
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

static unsigned getTruncatedSize(std::size_t size, unsigned truncationLower,
                                 unsigned truncationUpper) {
  assert(size >= truncationLower + truncationUpper);
  return size - (truncationLower + truncationUpper);
}

static unsigned getTransformedSize(const std::vector<std::size_t> &size,
                                   const ConvParams::InputTransform &transform,
                                   unsigned dim) {
  const auto truncatedSize =
      getTruncatedSize(size[dim], transform.truncationLower[dim],
                       transform.truncationUpper[dim]);
  const auto truncatedDilatedSize =
      getDilatedSize(truncatedSize, transform.dilation[dim]);
  int truncatedDilatedPaddedSize = transform.paddingLower[dim] +
                                   truncatedDilatedSize +
                                   transform.paddingUpper[dim];
  return truncatedDilatedPaddedSize;
}

unsigned ConvParams::getTruncatedInputSize(unsigned dim) const {
  return getTruncatedSize(inputFieldShape[dim],
                          inputTransform.truncationLower[dim],
                          inputTransform.truncationUpper[dim]);
}

unsigned ConvParams::getTruncatedKernelSize(unsigned dim) const {
  return getTruncatedSize(kernelShape[dim],
                          kernelTransform.truncationLower[dim],
                          kernelTransform.truncationUpper[dim]);
}

unsigned ConvParams::getTransformedInputSize(unsigned dim) const {
  return getTransformedSize(inputFieldShape, inputTransform, dim);
}
unsigned ConvParams::getTransformedKernelSize(unsigned dim) const {
  return getTransformedSize(kernelShape, kernelTransform, dim);
}

std::size_t ConvParams::getUntransformedOutputSize(unsigned dim) const {
  auto transformedInputSize = getTransformedInputSize(dim);
  auto transformedKernelSize = getTransformedKernelSize(dim);
  assert(transformedInputSize >= transformedKernelSize);
  return transformedInputSize + 1 - transformedKernelSize;
}

std::size_t ConvParams::getOutputSize(unsigned dim) const {
  auto convOutSize = getUntransformedOutputSize(dim);
  assert(convOutSize >= outputTransform.truncationLower[dim] +
                            outputTransform.truncationUpper[dim]);
  auto truncatedSize = convOutSize - (outputTransform.truncationLower[dim] +
                                      outputTransform.truncationUpper[dim]);
  auto stride = outputTransform.stride[dim];
  auto truncatedStridedSize = (truncatedSize + stride - 1) / stride;
  auto truncatedStridedPaddedSize = outputTransform.paddingLower[dim] +
                                    truncatedStridedSize +
                                    outputTransform.paddingUpper[dim];
  return truncatedStridedPaddedSize;
}

std::vector<std::size_t> ConvParams::getOutputFieldShape() const {
  std::vector<std::size_t> outputFieldShape;
  outputFieldShape.reserve(inputFieldShape.size());
  for (auto dim = 0U; dim != inputFieldShape.size(); ++dim) {
    outputFieldShape.push_back(getOutputSize(dim));
  }
  return outputFieldShape;
}

} // namespace poplin

namespace std {

std::size_t hash<poplin::ConvParams::InputTransform>::operator()(
    const poplin::ConvParams::InputTransform &it) const {
  std::size_t seed = 0;
  boost::hash_range(seed, std::begin(it.truncationLower),
                    std::end(it.truncationLower));
  boost::hash_range(seed, std::begin(it.truncationUpper),
                    std::end(it.truncationUpper));
  boost::hash_range(seed, std::begin(it.dilation), std::end(it.dilation));
  boost::hash_range(seed, std::begin(it.paddingLower),
                    std::end(it.paddingLower));
  boost::hash_range(seed, std::begin(it.paddingUpper),
                    std::end(it.paddingUpper));
  return seed;
}

std::size_t hash<poplin::ConvParams::OutputTransform>::operator()(
    const poplin::ConvParams::OutputTransform &ot) const {
  std::size_t seed = 0;
  boost::hash_range(seed, std::begin(ot.truncationLower),
                    std::end(ot.truncationLower));
  boost::hash_range(seed, std::begin(ot.truncationUpper),
                    std::end(ot.truncationUpper));
  boost::hash_range(seed, std::begin(ot.stride), std::end(ot.stride));
  boost::hash_range(seed, std::begin(ot.paddingLower),
                    std::end(ot.paddingLower));
  boost::hash_range(seed, std::begin(ot.paddingUpper),
                    std::end(ot.paddingUpper));
  return seed;
}

std::size_t
hash<poplin::ConvParams>::operator()(const poplin::ConvParams &p) const {
  std::size_t seed = 0;
  // TODO: T12874 Specialise std::hash for poplar::Type.
  boost::hash_combine(seed, std::string(p.inputType.toString()));
  boost::hash_combine(seed, std::string(p.outputType.toString()));
  boost::hash_combine(seed, p.batchSize);
  boost::hash_range(seed, std::begin(p.inputFieldShape),
                    std::end(p.inputFieldShape));
  boost::hash_range(seed, std::begin(p.kernelShape), std::end(p.kernelShape));
  boost::hash_combine(seed, p.inputChannelsPerConvGroup);
  boost::hash_combine(seed, p.outputChannelsPerConvGroup);
  boost::hash_combine(seed, p.numConvGroups);
  boost::hash_combine(seed, p.inputTransform);
  boost::hash_combine(seed, p.kernelTransform);
  boost::hash_combine(seed, p.outputTransform);
  return seed;
}
} // namespace std
