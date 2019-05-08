#include "poplin/Convolution.hpp"
#include "ConvUtilInternal.hpp"
#include "poplin/internal/ConvPlan.hpp"
#include "poplin/internal/ConvOptions.hpp"
#include <limits>
#include <algorithm>
#include <boost/optional.hpp>
#include <boost/functional/hash.hpp>
#include <boost/integer/common_factor_rt.hpp>
#include <boost/icl/interval_map.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include "poplin/ConvUtil.hpp"
#include "popops/Pad.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/TileMapping.hpp"
#include "popops/Reduce.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "poputil/exceptions.hpp"
#include "popops/Cast.hpp"
#include "poputil/Util.hpp"
#include "poputil/TileMapping.hpp"
#include "Winograd.hpp"
#include "popops/Zero.hpp"
#include "popops/ElementWise.hpp"
#include <unordered_set>
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "poplibs_support/print.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "ConvReduce.hpp"
#include "ChannelOps.hpp"
#include "popops/DynamicSlice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace poplin {

namespace {

std::map<std::string, WeightUpdateMethod> updateMethodMap {
  { "AMP", WeightUpdateMethod::AMP },
  { "AUTO", WeightUpdateMethod::AUTO }
};

std::map<std::string, Pass> passMap {
  { "NONE", Pass::NONE },
  { "INFERENCE_FWD", Pass::INFERENCE_FWD },
  { "TRAINING_FWD", Pass::TRAINING_FWD },
  { "TRAINING_BWD", Pass::TRAINING_BWD },
  { "TRAINING_WU", Pass::TRAINING_WU },
  { "FC_INFERENCE_FWD", Pass::FC_INFERENCE_FWD },
  { "FC_TRAINING_FWD", Pass::FC_TRAINING_FWD },
  { "FC_TRAINING_BWD", Pass::FC_TRAINING_BWD },
  { "FC_TRAINING_WU", Pass::FC_TRAINING_WU }
};

std::map<std::string, poplar::Type> partialsTypeMap {
  { "half", poplar::HALF },
  { "float", poplar::FLOAT }
};


// parse the passed options, taking default numIPUs and tilesPerIPU from the
// target
ConvOptions parseConvOptions(const Target &target,
                             const poplar::OptionFlags &options) {
  ConvOptions convOptions(target.getNumIPUs(), target.getTilesPerIPU());

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec convSpec{
    { "weightUpdateMethod", OptionHandler::createWithEnum(
      convOptions.weightUpdateMethod, updateMethodMap) },
    { "useWinograd", OptionHandler::createWithBool(
      convOptions.useWinograd) },
    { "winogradPatchSize", OptionHandler::createWithUnsignedInt(
      convOptions.winogradPatchSize) },
    { "tempMemoryBudget",
      OptionHandler::createWithUnsignedInt(
        convOptions.tempMemoryBudget) },
    { "cycleBackoffPercent",
      OptionHandler::createWithUnsignedInt(convOptions.cycleBackoffPercent)},
    { "pass", OptionHandler::createWithEnum(
      convOptions.pass, passMap) },
    { "partialsType", OptionHandler::createWithEnum(
      convOptions.partialsType, partialsTypeMap) },
    { "partialsType.interTile", OptionHandler::createWithEnum(
      convOptions.interTilePartialsType, partialsTypeMap) },
    { "partialsType.interIPU", OptionHandler::createWithEnum(
      convOptions.interIpuPartialsType, partialsTypeMap) },
    { "use128BitConvUnitLoad", OptionHandler::createWithBool(
      convOptions.use128BitConvUnitLoad) },
    { "useAggressiveRegrouping", OptionHandler::createWithBool(
      convOptions.useAggressiveRegrouping) },
    { "startTileMultiplier", OptionHandler::createWithUnsignedInt(
      convOptions.startTileMultiplier) },
    { "numIPUs", OptionHandler::createWithUnsignedInt(
      convOptions.numIPUs) },
    { "tilesPerIPU", OptionHandler::createWithUnsignedInt(
      convOptions.tilesPerIPU) },
    { "maxOutputMemoryProportion", OptionHandler::createWithDouble(
      convOptions.maxOutputMemoryProportion) }
  };
  for (const auto &entry : options) {
    convSpec.parse(entry.first, entry.second);
  }
  return convOptions;
}

template <typename T>
T roundToMultiple(T a, T to) {
  return ((a + to - 1) / to) * to;
}

}

ConvParams::InputTransform::
InputTransform(std::vector<unsigned> truncationLower_,
               std::vector<unsigned> truncationUpper_,
               std::vector<unsigned> dilation_,
               std::vector<unsigned> paddingLower_,
               std::vector<unsigned> paddingUpper_,
               std::vector<bool> flip_) :
    truncationLower(std::move(truncationLower_)),
    truncationUpper(std::move(truncationUpper_)),
    dilation(std::move(dilation_)),
    paddingLower(std::move(paddingLower_)),
    paddingUpper(std::move(paddingUpper_)),
    flip(flip_) {}

ConvParams::OutputTransform::
OutputTransform(std::vector<unsigned> truncationLower_,
                std::vector<unsigned> truncationUpper_,
                std::vector<unsigned> stride_,
                std::vector<unsigned> paddingLower_,
                std::vector<unsigned> paddingUpper_) :
    truncationLower(std::move(truncationLower_)),
    truncationUpper(std::move(truncationUpper_)),
    stride(std::move(stride_)),
    paddingLower(std::move(paddingLower_)),
    paddingUpper(std::move(paddingUpper_))
{}

ConvParams::
ConvParams(poplar::Type dType_,
           std::size_t batchSize_,
           std::vector<std::size_t> inputFieldShape_,
           std::vector<std::size_t> kernelShape_,
           std::size_t inputChannels_,
           std::size_t outputChannels_,
           std::size_t numConvGroups_,

           std::vector<unsigned> inputTruncationLower_,
           std::vector<unsigned> inputTruncationUpper_,
           std::vector<unsigned> inputDilation_,
           std::vector<unsigned> inputPaddingLower_,
           std::vector<unsigned> inputPaddingUpper_,
           std::vector<bool> flipInput_,

           std::vector<unsigned> kernelTruncationLower_,
           std::vector<unsigned> kernelTruncationUpper_,
           std::vector<unsigned> kernelDilation_,
           std::vector<unsigned> kernelPaddingLower_,
           std::vector<unsigned> kernelPaddingUpper_,
           std::vector<bool> flipKernel_,

           std::vector<unsigned> outputTruncationLower_,
           std::vector<unsigned> outputTruncationUpper_,
           std::vector<unsigned> stride_,
           std::vector<unsigned> outputPaddingLower_,
           std::vector<unsigned> outputPaddingUpper_) :
    dType(std::move(dType_)),
    batchSize(batchSize_),
    inputFieldShape(std::move(inputFieldShape_)),
    kernelShape(std::move(kernelShape_)),
    inputChannels(inputChannels_),
    outputChannels(outputChannels_),
    numConvGroups(numConvGroups_),
    inputTransform(std::move(inputTruncationLower_),
                   std::move(inputTruncationUpper_),
                   std::move(inputDilation_),
                   std::move(inputPaddingLower_),
                   std::move(inputPaddingUpper_),
                   std::move(flipInput_)),
    kernelTransform(std::move(kernelTruncationLower_),
                    std::move(kernelTruncationUpper_),
                    std::move(kernelDilation_),
                    std::move(kernelPaddingLower_),
                    std::move(kernelPaddingUpper_),
                    std::move(flipKernel_)),
    outputTransform(std::move(outputTruncationLower_),
                    std::move(outputTruncationUpper_),
                    std::move(stride_),
                    std::move(outputPaddingLower_),
                    std::move(outputPaddingUpper_)) {
  const auto numFieldDims = inputFieldShape.size();
  if (kernelShape.size() != numFieldDims) {
    throw poputil::poplibs_error("Number of kernel field dimensions does not"
                               "match the number of input field dimensions");
  }
  for(const auto stride : outputTransform.stride) {
    if(stride == 0) {
      throw poputil::poplibs_error("Stride must be non zero");
    }
  }
  for(const auto dilation : inputTransform.dilation) {
    if(dilation == 0) {
      throw poputil::poplibs_error("Input dilation must be non zero."
                                   " Dilation = 1 results in no dilation");
    }
  }
  for(const auto dilation : kernelTransform.dilation) {
    if(dilation == 0) {
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
                                  std::to_string(dim) +
                                  " has zero size");
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

std::ostream& operator<<(std::ostream &os, const ConvParams &p) {
  os << "Params: dType                      " << p.dType << "\n";
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

std::size_t hash_value(const ConvParams::InputTransform &it) {
  return std::hash<ConvParams::InputTransform>()(it);
}

std::size_t hash_value(const ConvParams::OutputTransform &ot) {
  return std::hash<ConvParams::OutputTransform>()(ot);
}

namespace {
  struct ConvIndices {
    unsigned cg;
    unsigned b;
    std::vector<unsigned> out;
    unsigned oc;
    unsigned ic;
    std::vector<unsigned> kernel;
  };

  struct ConvSlice {
    unsigned cgBegin, cgEnd;
    unsigned batchBegin, batchEnd;
    std::vector<unsigned> outFieldBegin, outFieldEnd;
    unsigned outChanBegin, outChanEnd;
    unsigned inChanBegin, inChanEnd;
    std::vector<unsigned> kernelBegin, kernelEnd;

    unsigned getNumFieldDims() const {
      return outFieldBegin.size();
    }
    unsigned getNumConvGroups() const { return cgEnd - cgBegin; }
    unsigned getBatchSize() const { return batchEnd - batchBegin; }
    unsigned getNumOutputChans() const { return outChanEnd - outChanBegin; }
    unsigned getNumInputChans() const { return inChanEnd - inChanBegin; }
    unsigned getOutputSize(unsigned dim) const {
      return outFieldEnd[dim] - outFieldBegin[dim];
    }
    unsigned getKernelSize(unsigned dim) const {
      return kernelEnd[dim] - kernelBegin[dim];
    }
  };
} // End anonymous namespace

static Tensor
createInputImpl(Graph &graph, const ConvParams &params,
                unsigned level, bool postPreprocess,
                const std::vector<ConvIndices> &indices,
                const std::string &name,
                const Plan &plan, const ConvOptions &options);
static Tensor
createWeightsImpl(Graph &graph, const ConvParams &params, unsigned level,
                  bool postPreprocess,
                  const std::vector<ConvIndices> &indices,
                  const std::string &name, const Plan &plan,
                  const ConvOptions &options);

static unsigned
getNumElementsInSlice(const std::vector<unsigned> &sliceBegin,
                      const std::vector<unsigned> &sliceEnd) {
  const auto rank = sliceBegin.size();
  assert(sliceEnd.size() == rank);
  unsigned numElements = 1;
  for (unsigned dim = 0; dim != rank; ++dim) {
    numElements *= sliceEnd[dim] - sliceBegin[dim];
  }
  return numElements;
}

static unsigned
getTruncatedSize(std::size_t size,
                 unsigned truncationLower,
                 unsigned truncationUpper) {
  assert(size >= truncationLower + truncationUpper);
  return size - (truncationLower + truncationUpper);
}


static unsigned
getTransformedSize(const std::vector<std::size_t> &size,
                   const ConvParams::InputTransform &transform,
                   unsigned dim) {
  const auto truncatedSize = getTruncatedSize(size[dim],
                                              transform.truncationLower[dim],
                                              transform.truncationUpper[dim]);
  const auto truncatedDilatedSize =
      getDilatedSize(truncatedSize, transform.dilation[dim]);
  int truncatedDilatedPaddedSize =
      transform.paddingLower[dim] + truncatedDilatedSize +
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
  auto truncatedSize =
      convOutSize - (outputTransform.truncationLower[dim] +
                     outputTransform.truncationUpper[dim]);
  auto stride = outputTransform.stride[dim];
  auto truncatedStridedSize = (truncatedSize + stride - 1) / stride;
  auto truncatedStridedPaddedSize =
      outputTransform.paddingLower[dim] + truncatedStridedSize +
      outputTransform.paddingUpper[dim];
  return truncatedStridedPaddedSize;
}

std::vector<std::size_t> ConvParams::getOutputFieldShape() const {
  std::vector<std::size_t> outputFieldShape;
  for (auto dim = 0U; dim != inputFieldShape.size(); ++dim) {
    outputFieldShape.push_back(getOutputSize(dim));
  }
  return outputFieldShape;
}

static std::string
getCapitalizedFieldDimName(unsigned dim, unsigned numFieldDims) {
  assert(dim < numFieldDims);
  if (numFieldDims > 3) {
    return "Field dimension " + std::to_string(dim);
  }
  // Dimensions are named from the innermost dimension outwards.
  switch (numFieldDims - dim) {
  case 1: return "Width";
  case 2: return "Height";
  case 3: return "Depth";
  }
  POPLIB_UNREACHABLE();
}

static void verifyInputShapes(const ConvParams &params,
                              const Tensor &in,
                              const Tensor &weights) {
  const auto numFieldDims = params.getNumFieldDims();
  if (in.rank() != 3 + numFieldDims) {
    throw poputil::poplibs_error(
          "Input tensor does not have the expected rank");
  }
  if (weights.rank() != 3 + numFieldDims) {
    throw poputil::poplibs_error("Weight tensor does not have the expected "
                                "rank");
  }
  for (unsigned i = 0; i != numFieldDims; ++i) {
    if (params.inputFieldShape[i] != in.dim(2 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw poputil::poplibs_error(dimName + " of input tensor does not match "
                                  "convolution parameters");
    }
    if (params.kernelShape[i] != weights.dim(1 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw poputil::poplibs_error(dimName + " of kernel does not match "
                                  "convolution parameters");
    }
  }
  if (params.numConvGroups != in.dim(0)) {
    throw poputil::poplibs_error("Number of convolution groups of input tensor "
                                "does not match convolution parameters");
  }
  if (params.getBatchSize() != in.dim(1)) {
    throw poputil::poplibs_error("Batchsize of input tensor does not match "
                                "convolution parameters");
  }
  if (in.dim(1) == 0) {
    throw poputil::poplibs_error("Batch size of input tensor equal to zero "
                                 "is not supported");
  }
  if (params.getNumInputChansPerConvGroup() != in.dim(in.rank() - 1)) {
    throw poputil::poplibs_error("Number of channels per convolution group of "
                                "input tensor does not match convolution "
                                "parameters");
  }
  if (params.numConvGroups != weights.dim(0)) {
    throw poputil::poplibs_error("Number of convolution groups of weights "
                                "tensor does not match convolution parameters");
  }
  if (params.getNumOutputChansPerConvGroup() !=
      weights.dim(weights.rank() - 2)) {
    throw poputil::poplibs_error("Kernel output channel size does not match "
                                "convolution parameters");
  }
  if (params.getNumInputChansPerConvGroup() !=
      weights.dim(weights.rank() - 1)) {
    throw poputil::poplibs_error("Kernel input channel size does not match "
                                "convolution parameters");
  }
}

static unsigned
getInChansPerGroup(const Plan &plan, unsigned numInChans) {
  return gcd(plan.inChansPerGroup, numInChans);
}

static unsigned
getWeightInChansPerGroup(const Plan &plan, unsigned numInChans) {
  return gcd(plan.inChansPerGroup, numInChans);
}

static unsigned
getWeightOutChansPerGroup(const Plan &plan, unsigned numOutChans) {
  return gcd(plan.partialChansPerGroup, numOutChans);
}

static unsigned
linearizeConvIndices(const std::vector<unsigned> &outIndices,
                     const std::vector<unsigned> &kernelIndices,
                     unsigned ic, unsigned b, unsigned oc, unsigned cg,
                     const std::vector<unsigned> &fieldSplit,
                     const std::vector<unsigned> &kernelSplit,
                     unsigned inChanSplit, unsigned batchSplit,
                     unsigned outChanSplit) {
  const auto numFieldDims = outIndices.size();
  // Use ozg as the innermost dimension to increase the chance that
  // tiles in a supertile both read the same activations. This reduces
  // exchange time when supertile send / receive is used.
  auto tile = cg;
  tile = tile * inChanSplit + ic;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    tile = tile * kernelSplit[dim] + kernelIndices[dim];
  }
  tile = tile * batchSplit + b;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    tile = tile * fieldSplit[dim] + outIndices[dim];
  }
  tile = tile * outChanSplit + oc;
  return tile;
}

static unsigned
linearizeTileIndices(const Target &target,
                     const ConvOptions &options,
                     const std::vector<ConvIndices> &indices,
                     const Plan &plan) {
  const auto hierarchy = getTileHierarchy(target, options);
  const auto numLevels = hierarchy.size();
  assert(indices.size() == numLevels);
  assert(plan.partitions.size() == numLevels);
  unsigned tile = 0;
  for (unsigned i = 0; i != numLevels; ++i) {
    auto fwdOutIndices = indices[i].out;
    const auto &fwdKernelIndices = indices[i].kernel;
    auto fwdic = indices[i].ic;
    const auto fwdb = indices[i].b;
    auto fwdoc = indices[i].oc;
    const auto fwdcg = indices[i].cg;
    auto fwdFieldSplit = plan.partitions[i].fieldSplit;
    const auto &fwdKernelSplit = plan.partitions[i].kernelSplit;
    auto fwdInChanSplit = plan.partitions[i].inChanSplit;
    const auto &fwdBatchSplit = plan.partitions[i].batchSplit;
    auto fwdOutChanSplit = plan.partitions[i].outChanSplit;
    switch (plan.linearizeTileOrder) {
    case Plan::LinearizeTileOrder::FC_WU:
      // For the fully connected weight update the in group and out group are
      // swapped compared to the forward pass.
      std::swap(fwdInChanSplit, fwdOutChanSplit);
      std::swap(fwdic, fwdoc);
      break;
    case Plan::LinearizeTileOrder::FC_BWD_AS_CONV:
      // For the fully connected backward pass the width and the input channels
      // are swapped compared to the forward pass.
      {
        std::swap(fwdFieldSplit.back(), fwdInChanSplit);
        std::swap(fwdOutIndices.back(), fwdic);
      }
      break;
    case Plan::LinearizeTileOrder::STANDARD:
      break;
    }
    const auto linearizedIndex =
        linearizeConvIndices(fwdOutIndices, fwdKernelIndices, fwdic, fwdb,
                             fwdoc, fwdcg, fwdFieldSplit,
                             fwdKernelSplit, fwdInChanSplit,
                             fwdBatchSplit, fwdOutChanSplit);
    tile = tile * hierarchy[i] + linearizedIndex;
  }
  assert(tile < target.getNumTiles());
  const auto tilesPerIpu = target.getTilesPerIPU();
  const auto ipu = tile / tilesPerIpu;
  return ipu * tilesPerIpu + (plan.startTile + tile) % tilesPerIpu;
}

static std::pair<unsigned, unsigned>
getTileOutRange(const ConvParams &params, const Partition &partition,
                unsigned tileIndex, unsigned dim) {
  const auto outSize = params.getOutputSize(dim);
  const auto grainSize = partition.fieldAxisGrainSize[dim];
  const auto numGrains = (outSize + grainSize - 1) / grainSize;
  const auto split = partition.fieldSplit[dim];
  const auto outGrainBegin = (tileIndex * numGrains) / split;
  const auto outGrainEnd = ((tileIndex + 1) * numGrains) / split;
  const auto outBegin = outGrainBegin * grainSize;
  const auto outEnd = std::min(outGrainEnd * grainSize, outSize);
  return {outBegin, outEnd};
}

/// Compute the sub-convolution corresponding to the specified slice of a larger
/// convolution. The parameters and tensors are updated in place to
/// the parameters and tensors for the sub-convolution.
static void
getSubConvolution(const ConvSlice &slice,
                  ConvParams &params,
                  Tensor *in, Tensor *weights) {
  const auto numFieldDims = params.getNumFieldDims();
  // Explicitly truncate the convGroup, channel and batch axes.
  params.numConvGroups = slice.getNumConvGroups();
  params.batchSize = slice.getBatchSize();
  params.inputChannels = slice.getNumInputChans();
  params.outputChannels = slice.getNumOutputChans();
  if (in) {
    *in = in->slice({slice.cgBegin, slice.batchBegin},
                  {slice.cgEnd, slice.batchEnd})
            .slice(slice.inChanBegin, slice.inChanEnd, in->rank() - 1);
  }
  if (weights) {
    *weights =
        weights->slice(slice.cgBegin, slice.cgEnd)
               .slice(slice.outChanBegin, slice.outChanEnd, weights->rank() - 2)
               .slice(slice.inChanBegin, slice.inChanEnd, weights->rank() - 1);
  }
  // Explicitly truncate the spatial dimensions.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto extraTruncationLower = slice.outFieldBegin[dim];
    auto extraTruncationUpper =
        static_cast<unsigned>(params.getOutputSize(dim))
        - slice.outFieldEnd[dim];
    // Ensure the truncation at either end is less than or equal to the padding
    // at that end plus the size of the downsampled convolution output. If the
    // truncation exceeds this amount the final output is zero and it is
    // therefore equivalent to any other convolution with inputs of the same
    // size that results in a zero output of the same size. We choose to
    // transform it into a convolution where the truncation at one end equals
    // the padding at that end plus the size of the downsampled convolution
    // output (ensuring the output remains zero) and the truncation at the other
    // end is adjusted to keep the same output size.
    if (extraTruncationLower >
        params.getOutputSize(dim) - params.outputTransform.paddingUpper[dim]) {
      auto excess = extraTruncationLower -
                    (params.getOutputSize(dim) -
                     params.outputTransform.paddingUpper[dim]);
      extraTruncationUpper += excess;
      extraTruncationLower -= excess;
    }
    if (extraTruncationUpper >
        params.getOutputSize(dim) - params.outputTransform.paddingLower[dim]) {
      auto excess = extraTruncationUpper -
                    (params.getOutputSize(dim) -
                     params.outputTransform.paddingLower[dim]);
      extraTruncationLower += excess;
      extraTruncationUpper -= excess;
    }
    auto &outputPaddingLower = params.outputTransform.paddingLower[dim];
    auto &outputPaddingUpper = params.outputTransform.paddingUpper[dim];
    const auto &stride = params.outputTransform.stride[dim];
    auto &outputTruncationLower = params.outputTransform.truncationLower[dim];
    auto &outputTruncationUpper = params.outputTransform.truncationUpper[dim];
    const auto excessPaddingLower = std::min(outputPaddingLower,
                                             extraTruncationLower);
    outputPaddingLower -= excessPaddingLower;
    extraTruncationLower -= excessPaddingLower;
    if (extraTruncationLower == params.getOutputSize(dim) -
                                outputPaddingUpper) {
      outputTruncationLower += 1 + (extraTruncationLower - 1) * stride;
    } else {
      outputTruncationLower += extraTruncationLower * stride;
    }
    extraTruncationLower = 0;
    const auto excessPaddingUpper = std::min(outputPaddingUpper,
                                             extraTruncationUpper);
    outputPaddingUpper -= excessPaddingUpper;
    extraTruncationUpper -= excessPaddingUpper;
    if (extraTruncationUpper == params.getOutputSize(dim) -
                                outputPaddingLower) {
      outputTruncationUpper += 1 + (extraTruncationUpper - 1) * stride;
    } else {
      outputTruncationUpper += extraTruncationUpper * stride;
    }
    extraTruncationUpper = 0;
  }
  // Replace unused kernel elements with zero padding.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto sliceBegin = std::max(slice.kernelBegin[dim],
                               params.kernelTransform.truncationLower[dim]);
    auto sliceEnd = std::min(slice.kernelEnd[dim],
                             static_cast<unsigned>(params.kernelShape[dim]) -
                             params.kernelTransform.truncationUpper[dim]);
    const auto transformedKernelSize = params.getTransformedKernelSize(dim);
    if (sliceBegin >= sliceEnd) {
      sliceBegin = 0;
      sliceEnd = 0;
      params.kernelTransform.truncationLower[dim] = 0;
      params.kernelTransform.truncationUpper[dim] = params.kernelShape[dim];
      params.kernelTransform.paddingLower[dim] = 0;
      params.kernelTransform.paddingUpper[dim] = transformedKernelSize;
      continue;
    }
    params.kernelTransform.truncationLower[dim] = sliceBegin;
    params.kernelTransform.paddingLower[dim] +=
        transformedKernelSize - params.getTransformedKernelSize(dim);
    params.kernelTransform.truncationUpper[dim] =
        params.kernelShape[dim] - sliceEnd;
    params.kernelTransform.paddingUpper[dim] +=
        transformedKernelSize - params.getTransformedKernelSize(dim);
  }

  // Canonicalize parameters. This may move truncation from the output to
  // the input or the kernel.
  params = canonicalizeParams(params);

  // Explicitly truncate the input.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto &inputTruncationLower = params.inputTransform.truncationLower[dim];
    auto &inputTruncationUpper = params.inputTransform.truncationUpper[dim];
    if (in) {
      *in = in->slice(inputTruncationLower,
                      params.inputFieldShape[dim] - inputTruncationUpper,
                      2 + dim);
    }
    params.inputFieldShape[dim] -= inputTruncationLower + inputTruncationUpper;
    inputTruncationLower = 0;
    inputTruncationUpper = 0;
  }

  // Explicitly truncate the kernel.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto &kernelTruncationLower = params.kernelTransform.truncationLower[dim];
    auto &kernelTruncationUpper = params.kernelTransform.truncationUpper[dim];
    if (weights) {
      *weights = weights->slice(kernelTruncationLower,
                                params.kernelShape[dim] - kernelTruncationUpper,
                                1 + dim);
    }
    params.kernelShape[dim] -= kernelTruncationLower + kernelTruncationUpper;
    kernelTruncationLower = 0;
    kernelTruncationUpper = 0;
  }
  assert(params == canonicalizeParams(params));
}

static void
iteratePartition(const ConvParams &params,
                 const Partition &partition,
                     const std::function<
                       void(const ConvIndices &,
                            const ConvSlice &)
                     > &f) {
  const auto numFieldDims = params.getNumFieldDims();
  const unsigned numOutChans = params.getNumOutputChansPerConvGroup();
  const auto outChanGrainSize = partition.outChanGrainSize;
  const auto outChanNumGrains = (numOutChans + outChanGrainSize - 1) /
                                outChanGrainSize;
  const auto batchSplit = partition.batchSplit;
  const auto outChanSplit = partition.outChanSplit;
  const auto inChanSplit = partition.inChanSplit;
  const unsigned batchSize = params.getBatchSize();
  const unsigned numInChans = params.getNumInputChansPerConvGroup();
  const auto inChanGrainSize = partition.inChanGrainSize;
  const auto inChanNumGrains = (numInChans + inChanGrainSize - 1) /
                               inChanGrainSize;
  const auto convGroupSplit = partition.convGroupSplit;
  const unsigned numConvGroups = params.getNumConvGroups();
  const auto totalFieldSplit = product(partition.fieldSplit);
  const auto totalKernelSplit = product(partition.kernelSplit);
  for (unsigned cg = 0; cg != convGroupSplit; ++cg) {
    const auto cgBegin = (cg * numConvGroups) / convGroupSplit;
    const auto cgEnd = ((cg + 1) * numConvGroups) / convGroupSplit;
    for (unsigned b = 0; b != batchSplit; ++b) {
      const auto batchBegin = (b * batchSize) / batchSplit;
      const auto batchEnd = ((b + 1) * batchSize) / batchSplit;
      for (unsigned ic = 0; ic != inChanSplit; ++ic) {
        const auto inChanGrainBegin = (ic * inChanNumGrains) / inChanSplit;
        const auto inChanGrainEnd = ((ic + 1) * inChanNumGrains) /
                                    inChanSplit;
        const auto inChanBegin = inChanGrainBegin * inChanGrainSize;
        const auto inChanEnd = std::min(inChanGrainEnd * inChanGrainSize,
                                        numInChans);
        for (unsigned k = 0; k != totalKernelSplit; ++k) {
          auto kernelIndices = unflattenIndex(partition.kernelSplit, k);
          std::vector<unsigned> kernelBegin(numFieldDims),
                                kernelEnd(numFieldDims);
          for (unsigned dim = 0; dim != numFieldDims; ++dim) {
            const auto kernelSize = params.kernelShape[dim];
            kernelBegin[dim] = (kernelIndices[dim] * kernelSize) /
                               partition.kernelSplit[dim];
            kernelEnd[dim] = ((kernelIndices[dim] + 1) * kernelSize) /
                             partition.kernelSplit[dim];
          }
          for (unsigned oc = 0; oc != outChanSplit; ++oc) {
            const auto outChanGrainBegin = (oc * outChanNumGrains) /
                                           outChanSplit;
            const auto outChanGrainEnd = ((oc + 1) * outChanNumGrains) /
                                         outChanSplit;
            const auto outChanBegin = outChanGrainBegin * outChanGrainSize;
            const auto outChanEnd = std::min(outChanGrainEnd * outChanGrainSize,
                                             numOutChans);
            for (unsigned of = 0; of != totalFieldSplit; ++of) {
              auto outIndices = unflattenIndex(partition.fieldSplit, of);
              std::vector<unsigned> outFieldBegin(numFieldDims),
                                    outFieldEnd(numFieldDims);
              for (unsigned dim = 0; dim != numFieldDims; ++dim) {
                std::tie(outFieldBegin[dim], outFieldEnd[dim]) =
                    getTileOutRange(params, partition, outIndices[dim],
                                    dim);
              }
              f({cg, b, outIndices, oc, ic, kernelIndices},
                {cgBegin, cgEnd,
                 batchBegin, batchEnd,
                 outFieldBegin,
                 outFieldEnd,
                 outChanBegin,
                 outChanEnd,
                 inChanBegin,
                 inChanEnd,
                 kernelBegin,
                 kernelEnd
                });
            }
          }
        }
      }
    }
  }
}

static std::vector<unsigned>
inversePermutation(const std::vector<unsigned> &permutation) {
  const auto rank = permutation.size();
  std::vector<unsigned> inverse(rank);
  for (unsigned i = 0; i != rank; ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

static Tensor
flattenDimsMultiStage(Tensor t, unsigned from, unsigned to,
                      unsigned kernelFactor) {
  const auto rank = t.rank();
  // Permute the dimensions so the dimension we want to flatten are at the
  // front.
  std::vector<unsigned> bringToFront = {
    from,
    to
  };
  bringToFront.reserve(rank);
  for (unsigned dim = 0; dim != rank; ++dim) {
    if (dim == from || dim == to)
      continue;
    bringToFront.push_back(dim);
  }
  t = t.dimShuffle(bringToFront);
  // Flatten from dimension into to dimension.
  auto flattenedShape = t.shape();
  flattenedShape[1] *= kernelFactor;
  if (kernelFactor != 0) {
    assert((flattenedShape[0] % kernelFactor) == 0);
    flattenedShape[0] /= kernelFactor;
  } else {
    flattenedShape[0] = 1;
  }
  t = t.reshape(flattenedShape);
  // Undo the previous permutation.
  t = t.dimShuffle(inversePermutation(bringToFront));
  return t;
}

static Tensor
flattenDims(Tensor t, unsigned from, unsigned to) {
  unsigned factor = t.dim(from);
  return flattenDimsMultiStage(t, from, to, factor);
}

static Tensor
unflattenDims(Tensor t, unsigned from, unsigned to, unsigned fromSize) {
  const auto rank = t.rank();
  // Permute the dimensions so the dimension we want to flatten are at the
  // front.
  std::vector<unsigned> bringToFront = {
    from,
    to
  };
  bringToFront.reserve(rank);
  for (unsigned dim = 0; dim != rank; ++dim) {
    if (dim == from || dim == to)
      continue;
    bringToFront.push_back(dim);
  }
  t = t.dimShuffle(bringToFront);
  // Reshape the dimensions.
  auto flattenedShape = t.shape();
  assert(flattenedShape[1] % fromSize == 0);
  assert(flattenedShape[0] == 1);
  flattenedShape[1] /= fromSize;
  flattenedShape[0] = fromSize;
  t = t.reshape(flattenedShape);
  // Undo the previous permutation.
  t = t.dimShuffle(inversePermutation(bringToFront));
  return t;
}

static Tensor dilate(Graph &graph,
                     const Tensor &t,
                     unsigned dilationFactor,
                     unsigned dim,
                     const std::string &debugPrefix) {
  const auto oldSize = t.dim(dim);
  const auto newSize = getDilatedSize(oldSize, dilationFactor);
  if (newSize == oldSize)
    return t;
  auto expandedT = t.expand({dim + 1});
  const auto dType = expandedT.elementType();
  auto zeroShape = expandedT.shape();
  zeroShape[dim + 1] = dilationFactor - 1;
  Tensor zero = graph.addConstant(dType, zeroShape, 0, debugPrefix + "/zero");
  graph.setTileMapping(zero, 0);
  return concat(expandedT, zero, dim + 1)
           .flatten(dim, dim + 2)
           .slice(0, newSize, dim);
}

// Dilate a tensor but instead of padding with zeros duplicate the nearest
// neighbouring element.
static Tensor
dilateWithNearestNeighbour(const Tensor &t, unsigned dilationFactor,
                           unsigned dim) {
  const auto oldSize = t.dim(dim);
  const auto newSize = getDilatedSize(oldSize, dilationFactor);
  if (newSize == oldSize)
    return t;
  return t.expand({dim + 1})
          .broadcast(dilationFactor, dim + 1)
          .flatten(dim, dim + 2)
          .slice(dilationFactor / 2, newSize + dilationFactor / 2, dim);
}

/**
 * Expand a spatial dimension of activations/weights to increase
 * the number of input channels, potentially in multiple stages.
 *
 * \param graph Graph in which tensors etc. are contained.
 * \param params Convolutional parameters, these will be modified
 *               to represent a convolution with the dimension
 *               expanded (by the given factor).
 * \param acts   Optional tensor which will be manipulated to
 *               perform the expansion.
 * \param weights Optional tensor which will be manipulated to
 *                perform the expansion.
 * \param dim     The spatial dimension to expand.
 * \param kernelFactor  The factor by which to expand the kernel.
 *                      This need not evenly divide the kernel
 *                      dimension, but if it does not padding will
 *                      be added.
 */
static void expandSpatialDimMultiStage(Graph &graph,
                                       ConvParams &params,
                                       Tensor *acts,
                                       Tensor *weights,
                                       unsigned dim,
                                       unsigned kernelFactor,
                                       const std::string &debugPrefix) {
  unsigned actsDimIndex = dim + 2;
  unsigned weightsDimIndex = dim + 1;
  auto &actsSize = params.inputFieldShape[dim];
  auto &weightsSize = params.kernelShape[dim];
  auto &actsTruncationLower = params.inputTransform.truncationLower[dim];
  auto &actsTruncationUpper = params.inputTransform.truncationUpper[dim];
  auto &actsDilation = params.inputTransform.dilation[dim];
  auto &actsPaddingLower = params.inputTransform.paddingLower[dim];
  auto &actsPaddingUpper = params.inputTransform.paddingUpper[dim];
  std::vector<bool>::reference actsFlip = params.inputTransform.flip[dim];
  std::vector<bool>::reference weightsFlip = params.kernelTransform.flip[dim];
  auto &weightsTruncationLower = params.kernelTransform.truncationLower[dim];
  auto &weightsTruncationUpper = params.kernelTransform.truncationUpper[dim];
  auto &weightsDilation = params.kernelTransform.dilation[dim];
  auto &weightsPaddingLower = params.kernelTransform.paddingLower[dim];
  auto &weightsPaddingUpper = params.kernelTransform.paddingUpper[dim];
  auto &outputTruncationLower = params.outputTransform.truncationLower[dim];
  auto &outputTruncationUpper = params.outputTransform.truncationUpper[dim];
  auto &stride = params.outputTransform.stride[dim];
  auto &outputPaddingLower = params.outputTransform.paddingLower[dim];
  auto &outputPaddingUpper = params.outputTransform.paddingUpper[dim];
  if (acts) {
    // Explicitly truncate.
    *acts = pad(graph, *acts,
                -static_cast<int>(actsTruncationLower),
                -static_cast<int>(actsTruncationUpper),
                actsDimIndex);
    // Explicitly dilate.
    *acts = dilate(graph, *acts, actsDilation, actsDimIndex, debugPrefix);
    // Explicitly pad.
    *acts = pad(graph, *acts, actsPaddingLower, actsPaddingUpper, actsDimIndex);
    // Explicitly flip.
    if (actsFlip) {
      *acts = acts->reverse(actsDimIndex);
    }
  }
  actsSize -= (actsTruncationLower + actsTruncationUpper);
  actsSize = getDilatedSize(actsSize, actsDilation);
  actsSize += actsPaddingLower + actsPaddingUpper;
  actsDilation = 1;
  actsTruncationLower = 0;
  actsTruncationUpper = 0;
  actsPaddingLower = 0;
  actsPaddingUpper = 0;
  actsFlip = false;
  if (weights) {
    // Explicitly truncate.
    *weights = pad(graph, *weights,
                   -static_cast<int>(weightsTruncationLower),
                   -static_cast<int>(weightsTruncationUpper),
                   weightsDimIndex);
  }
  weightsSize = params.getTruncatedKernelSize(dim);
  weightsTruncationLower = 0;
  weightsTruncationUpper = 0;

  // A partial expansion entails splitting the input dimension
  // into another dimension with size equal to the given kernelFactor
  // which is then folded into the input channels for the activations,
  // or flattened from the expanded dimension to the input channels
  // for the weights.
  //
  // In order to ensure that new input channels created by this
  // expansion can be expanded without interleaving with the original
  // input channels (allowing us to group by input channels after a partial
  // expansion) the kernel is split by uniformly subsampling kernel elements by
  // the kernel factor.

  // First pad weights/input if needed to allow expansion by the kernel
  // factor.
  bool emptyWeights = weightsSize == 0;
  const auto paddedKernelSize =
    emptyWeights ? 0 : roundToMultiple<std::size_t>(weightsSize, kernelFactor);
  const auto kernelPadding = paddedKernelSize - weightsSize;
  const auto actsPaddedSize =
    actsSize +
    getDilatedSize(paddedKernelSize, weightsDilation) -
    getDilatedSize(weightsSize, weightsDilation);
  const auto actsPadding = actsPaddedSize - actsSize;
  if (acts) {
    // If the kernel will be flipped, then pad the input such that padding
    // is at the lower end of the spatial dimension to match where the
    // kernel padding ends up.
    *acts = pad(graph, *acts,
                actsPadding * weightsFlip,
                actsPadding * !weightsFlip,
                actsDimIndex);
  }
  if (weights) {
    *weights = pad(graph, *weights, 0, kernelPadding, weightsDimIndex);
  }
  actsSize = actsPaddedSize;
  weightsSize = paddedKernelSize;

  bool fullExpansion = kernelFactor == weightsSize;
  assert(emptyWeights || (weightsSize % kernelFactor) == 0);
  const auto weightsFactoredSize =
    emptyWeights ? 1 : weightsSize / kernelFactor;
  auto weightsFactoredDilation =
    emptyWeights ? 1 : weightsDilation * kernelFactor;
  auto weightsFactoredDilatedSize =
    getDilatedSize(weightsFactoredSize, weightsFactoredDilation);
  auto actsFactoredSize =
    actsSize -
    (getDilatedSize(weightsSize, weightsDilation) +
     weightsPaddingLower + weightsPaddingUpper) +
    weightsFactoredDilatedSize;
  actsFactoredSize -= outputTruncationLower + outputTruncationUpper;
  if (fullExpansion) {
    actsFactoredSize =
      (actsFactoredSize + stride - 1) / stride;
    actsFactoredSize += outputPaddingLower + outputPaddingUpper;
  }
  assert(!fullExpansion || actsFactoredSize == params.getOutputSize(dim));
  if (acts) {
    // Expand the acts tensor.
    auto dType = acts->elementType();
    if (weightsSize == 0) {
      auto newActsShape = acts->shape();
      newActsShape[actsDimIndex] = actsFactoredSize;
      newActsShape.back() = 0;
      *acts = graph.addConstant(dType, newActsShape, 0, debugPrefix + "/acts");
      graph.setTileMapping(*acts, 0);
    } else {
      std::vector<Tensor> slices;
      auto subsampledKernelElements =
        getDilatedSize(weightsFactoredSize, kernelFactor);
      for (unsigned k = 0; k < kernelFactor; ++k) {
        auto weightOutRange =
          getOutputRangeForKernelRange(dim, {0, params.getOutputSize(dim)},
                                       {k, k + subsampledKernelElements},
                                       params);
        assert(weightOutRange.first != weightOutRange.second);
        auto weightInRange = getInputRange(dim, {0, params.getOutputSize(dim)},
                                           {k, k + subsampledKernelElements},
                                           params);
        auto slice = acts->slice(weightInRange.first,
                                 weightInRange.second,
                                 actsDimIndex);
        if (fullExpansion) {
          slice = slice.subSample(stride, actsDimIndex);
          const auto slicePaddingLower = weightOutRange.first;
          const auto slicePaddingUpper =
              params.getOutputSize(dim) - weightOutRange.second;
          slice = pad(graph, slice, slicePaddingLower, slicePaddingUpper,
                      actsDimIndex);
        }
        assert(slice.dim(actsDimIndex) == actsFactoredSize);
        slices.push_back(slice);
      }
      auto expanded = concat(slices, acts->rank() - 1);
      *acts = expanded;
    }
  }
  if (weights) {
    // Flatten the spatial dimension of the weights tensor into the input
    // channels.
    *weights = flattenDimsMultiStage(*weights, weightsDimIndex,
                                     weights->rank() - 1, kernelFactor);
  }
  actsSize = actsFactoredSize;
  params.inputChannels *= kernelFactor;
  weightsPaddingLower = 0;
  weightsPaddingUpper = 0;
  outputTruncationLower = 0;
  outputTruncationUpper = 0;
  // These transformations of the kernel cannot be applied until we
  // are fully expanding the kernel
  if (fullExpansion) {
    weightsFlip = false;
    weightsSize = 1;
    weightsDilation = 1;
    stride = 1;
    outputPaddingLower = 0;
    outputPaddingUpper = 0;
  } else {
    weightsSize /= kernelFactor;
    weightsDilation *= kernelFactor;
  }
}

static void expandSpatialDim(Graph &graph,
                             ConvParams &params,
                             Tensor *acts,
                             Tensor *weights,
                             unsigned dim,
                             const std::string &debugPrefix) {
  auto factor = params.getTruncatedKernelSize(dim);
  expandSpatialDimMultiStage(
        graph, params, acts, weights, dim, factor, debugPrefix);
}

static void
swapOperands(ConvParams &params, boost::optional<Tensor> &acts,
             boost::optional<Tensor> &weights) {
  swapOperands(params);
  std::swap(acts, weights);
  if (acts) {
    *acts = acts->dimRoll(acts->rank() - 2, 1);
  }
  if (weights) {
    *weights = weights->dimRoll(1, weights->rank() - 2);
  }
}

// A plan for how to perform a dimension expansion for the
// best possible memory usage/cycles.
// Plan is unified for activations and weights as we may need
// to modify the convolution parameters as we go and these
// must agree for both.
struct ExpandDimsPlan {
  // A dimension and a factor by which to divide the kernel in order to
  // partially expand the dimension.
  // This allows a transpose on a partially expanded tensor. This is not
  // advantageous for the weights hence this applies to activations only.
  std::pair<unsigned, unsigned> partialExpansion = std::make_pair(0U, 1U);
  // Grouping info before/after for both activations and weights
  // used to regroup either before or after the expansion.
  // Same grouped dimension before and after means no regroup
  // will occur.
  // [0] = weights, [1] = activations
  std::array<std::pair<GroupingInfo, GroupingInfo>, 2> regroup;
  // Whether to perform any present regroup operation before or
  // after the expansion/flattening of the activations/weights.
  // [0] = weights, [1] = activations
  std::array<bool, 2> regroupPost;
};

std::ostream &operator<<(std::ostream &o, const ExpandDimsPlan &p) {
  if (p.partialExpansion.second > 1) {
    o << "Partially expand spatial dimension "
      << p.partialExpansion.first << " by factor "
      << p.partialExpansion.second << "\n";
  }
  for (bool isActs : {false, true}) {
    if (p.regroup[isActs].first.first != p.regroup[isActs].second.first) {
      o << "regroup "
        << (isActs ? "activations " : "weights ")
        << (p.regroupPost[isActs] ? "after" : "before")
        << " expanding: {"
        << p.regroup[isActs].first.first << ","
        << p.regroup[isActs].first.second << "} -> {"
        << p.regroup[isActs].second.first << ","
        << p.regroup[isActs].second.second << "}\n";
    }
  }
  return o;
}

static std::vector<GroupingInfo>
determinePreprocessedGroupingFromPlan(const ConvParams &params,
                                      const Plan &plan,
                                      unsigned level,
                                      bool isActs) {
  std::vector<GroupingInfo> grouping(1);

  // Total dimensions are spatial dimensions + 3 for either
  // acts or weights. Input channels are always the last dimension.
  auto inChanDim = params.kernelShape.size() + 3 - 1;
  auto inChanGrainSize = plan.partitions[level].inChanGrainSize;

  // The final grouping will have to be some multiple of inChanGrainSize
  grouping[0] = std::make_pair(inChanDim, inChanGrainSize);
  return grouping;
}

// Get a description for how to perform a transform for efficiency in memory
// and cycles.
static ExpandDimsPlan
getExpandDimsPlan(const Graph &graph,
                  const ConvParams &params,
                  const Plan &convPlan,
                  unsigned level,
                  const Tensor &in,
                  bool isActs) {
  ExpandDimsPlan plan;
  const auto &expandDimsSpatial = convPlan.transforms[level].expandDims;
  const auto expandDims = dimsFromSpatialDims(expandDimsSpatial, isActs);
  auto grouping = detectDimGroupings(graph, in);
  // We can simply use the fully preprocessed grouping as further transforms
  // won't interleave the input channels we get from expanding and our regroup
  // operation is mapped linearly without regard for the ordering as defined
  // by the other dimensions (which may be shuffled around by other
  // transforms).
  auto destGrouping =
    determinePreprocessedGroupingFromPlan(params, convPlan, level, isActs);

  // If there was no detected grouping we've got nothing to go on unfortunately
  // so there's no special ops to help this expansion.
  if (!grouping.empty() && !destGrouping.empty() &&
      grouping[0].first != destGrouping[0].first &&
      !expandDims.empty() &&
      (std::find(expandDims.begin(), expandDims.end(),
                 grouping[0].first) == expandDims.end()) &&
      (std::find(expandDims.begin(), expandDims.end(),
                 destGrouping[0].first) == expandDims.end())) {
    auto grainSize = getMinimumRegroupGrainSize(params.dType);
    unsigned dimElems = in.dim(destGrouping[0].first);
    auto maxGroupSize = gcd(dimElems, destGrouping[0].second);
    auto nextGrouping = destGrouping[0];
    nextGrouping.second = maxGroupSize;

    // If the dimension to transpose to doesn't meet our minimum grain size
    // then we can do a special multi-stage expansion where
    // we partially expand a dimension, then transpose, then
    // perform the rest of the expansion. This is only advantageous for
    // activations where the elements are broadcasted increasing the amount
    // of data that needs transposing. For weights we can always do a
    // transpose after completely flattening the tensor.
    if (isActs && nextGrouping.first == in.rank() - 1 &&
        (maxGroupSize % grainSize) != 0) {
      unsigned expandedDimElems = dimElems;
      unsigned remainingExpansion =
        std::accumulate(expandDimsSpatial.begin(),
                        expandDimsSpatial.end(),
                        std::size_t(1),
                        [&](std::size_t t, const std::size_t dim) {
                          return t * params.getTruncatedKernelSize(dim);
                        });
      auto inputChannels = params.inputChannels;
      auto inChanGrainSize = convPlan.partitions[level].inChanGrainSize;
      for (unsigned i = 0; i != expandDimsSpatial.size(); ++i) {
        unsigned roundedElems = lcm(expandedDimElems, grainSize);
        unsigned factor = roundedElems / expandedDimElems;
        auto dim = expandDimsSpatial[i];
        auto truncatedKernelSize = params.getTruncatedKernelSize(dim);
        // We're all padding anyway
        if (truncatedKernelSize == 0)
          break;
        auto kernelPadded =
          ((truncatedKernelSize + factor - 1) / factor) * factor;
        auto kernelPadding = kernelPadded - truncatedKernelSize;
        auto inChanPaddingAfterFullExpansion =
          kernelPadding * remainingExpansion;
        // We can partially expand at this point either if:
        // * the kernel is evenly divisible by the desired factor (no padding
        // required)
        if ((truncatedKernelSize % factor) == 0 ||
        // * the padding after fully expanding is no more than would be
        //   added as a result of the input channel grain size.
            (truncatedKernelSize > 1 &&
             inChanPaddingAfterFullExpansion < inChanGrainSize) ||
        // * this is the last dimension to be expanded (padding can be safely
        //   added as it will end up in the last input channels and can be
        //   easily stripped.
            (truncatedKernelSize > 1 &&
             i == expandDimsSpatial.size() - 1)) {
          plan.partialExpansion.first = dim;
          plan.partialExpansion.second = factor;
          maxGroupSize = gcd(roundedElems, destGrouping[0].second);
          nextGrouping.second = maxGroupSize;
          plan.regroupPost[isActs] = false;
          break;
        }
        expandedDimElems *= truncatedKernelSize;
        remainingExpansion /= truncatedKernelSize;
        inputChannels *= kernelPadded;
      }
    }

    if ((maxGroupSize % grainSize) == 0 &&
        (grouping[0].second % grainSize) == 0) {
      plan.regroup[isActs].first = grouping[0];
      plan.regroup[isActs].second = nextGrouping;
      plan.regroupPost[isActs] = false;
    }
  }

  return plan;
}

static ExpandDimsPlan
mergeExpandDimsPlans(ExpandDimsPlan planActs,
                     const ExpandDimsPlan &planWeights) {
  planActs.regroup[false] = planWeights.regroup[false];
  planActs.regroupPost[false] = planWeights.regroupPost[false];
  return planActs;
}


static void
expandSpatialDims(Graph &graph, ConvParams &params,
                  Plan &plan,
                  unsigned level,
                  boost::optional<Tensor> &acts,
                  boost::optional<Tensor> &weights,
                  const ExpandDimsPlan &expandDimsPlan,
                  Sequence &expandingCopies,
                  boost::optional<ComputeSet> &transposeCS,
                  bool rearrangeActs,
                  bool rearrangeWeights,
                  const std::string &debugPrefix) {
  const auto &expandDimsSpatial = plan.transforms[level].expandDims;

  const auto &partialExpansion = expandDimsPlan.partialExpansion;
  bool hasPartialExpansion = partialExpansion.second > 1;
  unsigned inputChanPaddingLower = 0, inputChanPaddingUpper = 0;
  bool stripPadding = false;
  std::size_t nextToExpand = 0;
  if (!expandDimsSpatial.empty() && hasPartialExpansion) {
    // Fully expand up to the dimension to partially expand.
    for (; nextToExpand != expandDimsSpatial.size(); ++nextToExpand) {
      if (expandDimsSpatial[nextToExpand] == partialExpansion.first) {
        break;
      }
      expandSpatialDim(graph,
                       params,
                       acts.get_ptr(),
                       weights.get_ptr(),
                       expandDimsSpatial[nextToExpand],
                       debugPrefix);
    }
    unsigned kernelSizeBefore =
      params.getTruncatedKernelSize(partialExpansion.first);
    unsigned inputChansBefore = params.inputChannels;

    // Partially expand
    expandSpatialDimMultiStage(graph,
                               params,
                               acts.get_ptr(),
                               weights.get_ptr(),
                               partialExpansion.first,
                               partialExpansion.second,
                               debugPrefix);

    // Check for any padding added for the partial expansion
    unsigned kernelSizeAfter =
      params.getTruncatedKernelSize(partialExpansion.first);
    unsigned padding =
      (kernelSizeAfter * partialExpansion.second) - kernelSizeBefore;
    inputChanPaddingLower = 0;
    inputChanPaddingUpper = padding * inputChansBefore;
    // We can only strip the padding if it does not become interleaved
    // in the input channels as a result of further expanded dimensions.
    stripPadding =
      (inputChanPaddingUpper + inputChanPaddingLower > 0 &&
       partialExpansion.first == expandDimsSpatial.back());
  }

  // Pre-expansion regroup.
  for (bool isActs : {false, true}) {
    const auto &regroup = expandDimsPlan.regroup[isActs];
    bool regroupPreExpansion = !expandDimsPlan.regroupPost[isActs];
    if (regroup.first.first != regroup.second.first &&
        regroupPreExpansion) {
      if ((acts && isActs && rearrangeActs) ||
          (weights && !isActs && rearrangeWeights)) {
        auto &t = isActs ? *acts : *weights;
        t = regroupTensor(graph, t, expandingCopies, transposeCS,
                          regroup.first, regroup.second,
                          debugPrefix);
      }
    }
  }

  // Fully expand remaining dimensions now
  for (; nextToExpand != expandDimsSpatial.size(); ++nextToExpand) {
    const auto factor =
      params.getTruncatedKernelSize(expandDimsSpatial[nextToExpand]);
    expandSpatialDimMultiStage(graph,
                               params,
                               acts.get_ptr(),
                               weights.get_ptr(),
                               expandDimsSpatial[nextToExpand],
                               factor,
                               debugPrefix);
    // Trim any padding added by a potential earlier partial expansion
    if ((inputChanPaddingLower || inputChanPaddingUpper) && stripPadding) {
      if (acts) {
        *acts = pad(graph, *acts,
                    -static_cast<int>(inputChanPaddingLower),
                    -static_cast<int>(inputChanPaddingUpper),
                    acts->rank() - 1);
      }
      if (weights) {
        *weights = pad(graph, *weights,
                       -static_cast<int>(inputChanPaddingLower),
                       -static_cast<int>(inputChanPaddingUpper),
                       weights->rank() - 1);
      }
      params.inputChannels -= inputChanPaddingUpper + inputChanPaddingLower;
      inputChanPaddingLower = inputChanPaddingUpper = 0;
    }
  }

  // Post-expansion regroup.
  for (bool isActs : {false, true}) {
    const auto &regroup = expandDimsPlan.regroup[isActs];
    bool regroupPostExpansion = expandDimsPlan.regroupPost[isActs];
    if (regroup.first.first != regroup.second.first &&
        regroupPostExpansion) {
      if ((acts && isActs && rearrangeActs) ||
          (weights && !isActs && rearrangeWeights)) {
        auto &t = isActs ? *acts : *weights;
        t = regroupTensor(graph, t, expandingCopies, transposeCS,
                          regroup.first, regroup.second,
                          debugPrefix);
      }
    }
  }
}

static void
expandSpatialDims(Graph &graph, ConvParams &params,
                  Plan &plan, unsigned level,
                  boost::optional<Tensor> &acts,
                  boost::optional<Tensor> &weights,
                  Sequence &expandingCopies,
                  boost::optional<ComputeSet> &transposeCS,
                  bool rearrangeActs = false,
                  bool rearrangeWeights = false,
                  const std::string &debugPrefix = "") {
  const auto &expandDimsSpatial = plan.transforms[level].expandDims;
  if (expandDimsSpatial.empty()) {
    return;
  }

  ExpandDimsPlan actsTransformPlan, weightsTransformPlan;
  Tensor actsExpanded, weightsExpanded;
  if (acts && rearrangeActs) {
    actsTransformPlan =
        getExpandDimsPlan(graph, params, plan, level, *acts, true);
  }

  if (weights && rearrangeWeights) {
    weightsTransformPlan =
        getExpandDimsPlan(graph, params, plan, level, *weights, false);
  }

  ExpandDimsPlan mergedTransformPlan =
    mergeExpandDimsPlans(actsTransformPlan, weightsTransformPlan);

  // Now do a series of transforms as appropriate.
  expandSpatialDims(graph,
                    params,
                    plan,
                    level,
                    acts,
                    weights,
                    mergedTransformPlan,
                    expandingCopies,
                    transposeCS,
                    rearrangeActs,
                    rearrangeWeights,
                    debugPrefix);
}

static void
regroupIfBeneficial(Graph &graph,
                    const ConvParams &params,
                    const Plan &plan,
                    unsigned level,
                    Tensor &in,
                    bool isActs,
                    Sequence &expandingCopies,
                    boost::optional<ComputeSet> &transposeCS,
                    const std::string &debugPrefix = "") {
  auto grouping =
    detectDimGroupings(graph, in);
  auto destGrouping =
    determinePreprocessedGroupingFromPlan(params, plan, level, isActs);
  auto grainSize = getMinimumRegroupGrainSize(params.dType);
  if (!grouping.empty() && !destGrouping.empty() &&
      grouping[0].first != destGrouping[0].first &&
      (grouping[0].second % grainSize) == 0 &&
      (destGrouping[0].second % grainSize) == 0) {
    in = regroupTensor(graph, in, expandingCopies, transposeCS,
                       grouping[0], destGrouping[0],
                       debugPrefix);
  }
}

/// Apply any pre-convolution transformations implied by the plan. The
/// plan and the parameters are updated to describe the convolution operation
/// performed on the transformed input. If the \a acts or \ weights pointers are
/// not null they are updated to be views of the original tensors with
/// dimensions that match the shape expected by the convolution operation.
static void
convolutionPreprocess(Graph &graph, ConvParams &params,
                      const ConvOptions &options,
                      Plan &plan, unsigned level,
                      const std::vector<ConvIndices> &indices,
                      boost::optional<Tensor> &acts,
                      boost::optional<Tensor> &weights,
                      Sequence *rearrangeProg = nullptr,
                      Tensor *rearrangeWritten = nullptr,
                      bool rearrangeActs = false,
                      bool rearrangeWeights = false,
                      const std::string &debugPrefix = "") {
  ConvTransform &transform = plan.transforms[level];
  const auto inChanGrainSize = level < plan.partitions.size() ?
                               plan.partitions[level].inChanGrainSize :
                               plan.inChansPerGroup;
  const auto outChanGrainSize = level < plan.partitions.size() ?
                                plan.partitions[level].outChanGrainSize :
                                plan.partialChansPerGroup;
  if (transform.extraFieldDims) {
    addExtraDims(params, transform.extraFieldDims);
    if (acts) {
      *acts = acts->expand(
                std::vector<std::size_t>(transform.extraFieldDims, 2)
              );
    }
    if (weights) {
      *weights = weights->expand(
                   std::vector<std::size_t>(transform.extraFieldDims, 1)
                 );
    }
    transform.extraFieldDims = 0;
  }
  params = calculateParamsWithDeferredDilation(params,
                                               transform.dilatePostConv);
  transform.dilatePostConv.clear();
  if (transform.swapOperands) {
    swapOperands(params, acts, weights);
    transform.swapOperands = false;
  }
  boost::optional<ComputeSet> transposeCS;
  Sequence expandingCopies;
  expandSpatialDims(graph,
                    params,
                    plan,
                    level,
                    acts,
                    weights,
                    expandingCopies,
                    transposeCS,
                    rearrangeActs,
                    rearrangeWeights,
                    debugPrefix);
  transform.expandDims.clear();
  if (!transform.outChanFlattenDims.empty()) {
    boost::optional<Tensor> maybeActs, maybeWeights;
    if (acts)
      maybeActs.reset(*acts);
    if (weights)
      maybeWeights.reset(*weights);
    swapOperands(params, maybeActs, maybeWeights);
    for (auto dim : transform.outChanFlattenDims) {
      expandSpatialDim(graph,
                       params,
                       maybeActs.get_ptr(),
                       maybeWeights.get_ptr(),
                       dim,
                       debugPrefix);
      if (maybeActs) {
        *maybeActs = flattenDims(*maybeActs, dim + 2, 1);
      }
      params.batchSize *= params.inputFieldShape[dim];
      params.inputFieldShape[dim] = 1;
    }
    swapOperands(params, maybeActs, maybeWeights);
    if (acts)
      *acts = *maybeActs;
    if (weights)
      *weights = *maybeWeights;
    transform.outChanFlattenDims.clear();
  }
  if (!transform.flattenDims.empty()) {
    for (auto it = std::next(transform.flattenDims.rbegin()),
         end = transform.flattenDims.rend(); it != end; ++it) {
      const auto fromDimIndex = *it;
      const auto toDimIndex = transform.flattenDims.back();
      assert(fromDimIndex != toDimIndex);
      if (acts) {
        *acts = flattenDims(*acts, fromDimIndex + 1, toDimIndex + 1);
      }
      auto &fromDimSize =
          fromDimIndex ? params.inputFieldShape[fromDimIndex - 1] :
          params.batchSize;
      auto &toDimSize =
          toDimIndex ? params.inputFieldShape[toDimIndex - 1] :
          params.batchSize;
      toDimSize *= fromDimSize;
      fromDimSize = 1;
    }
  }
  transform.flattenDims.clear();
  // Zero pad the input / weights.
  const auto inChanGrains = (params.inputChannels + inChanGrainSize - 1) /
                            inChanGrainSize;
  const auto paddedInChans = inChanGrains * inChanGrainSize;
  const auto outChanGrains = (params.outputChannels + outChanGrainSize - 1) /
                            outChanGrainSize;
  const auto paddedOutChans = outChanGrains * outChanGrainSize;
  if (acts) {
    *acts = pad(graph, *acts, 0, paddedInChans - params.inputChannels,
                acts->rank() - 1);
  }
  if (weights) {
    *weights = pad(graph, *weights, 0, paddedInChans - params.inputChannels,
                   weights->rank() - 1);
    *weights = pad(graph, *weights, 0, paddedOutChans - params.outputChannels,
                   weights->rank() - 2);
  }
  params.inputChannels = paddedInChans;
  params.outputChannels = paddedOutChans;

  Sequence rearrangingCopies;
  if (acts && rearrangeActs) {
    regroupIfBeneficial(graph, params, plan, level, *acts, true,
                        expandingCopies, transposeCS, debugPrefix);
    auto actsRearranged =
      createInputImpl(graph, params, level, true, indices,
                      debugPrefix + "/actsRearranged",
                      plan, options);
    rearrangingCopies.add(Copy(*acts, actsRearranged));
    *rearrangeWritten = concat(*rearrangeWritten, actsRearranged.flatten());
    *acts = actsRearranged;
  }

  if (weights && rearrangeWeights) {
    regroupIfBeneficial(graph, params, plan, level, *weights, false,
                        expandingCopies, transposeCS, debugPrefix);
    auto weightsRearranged =
      createWeightsImpl(graph, params, level, true, indices,
                        debugPrefix + "/weightsRearranged",
                        plan, options);
    rearrangingCopies.add(Copy(*weights, weightsRearranged));
    *rearrangeWritten = concat(*rearrangeWritten, weightsRearranged.flatten());
    *weights = weightsRearranged;
  }
  if (rearrangeProg) {
    rearrangeProg->add(expandingCopies);
    if (transposeCS) {
      rearrangeProg->add(Execute(*transposeCS));
    }
    rearrangeProg->add(rearrangingCopies);
  }
}

static void
convolutionPreprocess(Graph &graph, ConvParams &params,
                      const ConvOptions &options,
                      Plan &plan, unsigned level,
                      const std::vector<ConvIndices> &indices,
                      Tensor &acts,
                      Tensor &weights,
                      Sequence *rearrangeProg = nullptr,
                      Tensor *rearrangeWritten = nullptr,
                      bool rearrangeActs = false,
                      bool rearrangeWeights = false,
                      const std::string &debugPrefix = "") {
  auto actsOptional = boost::make_optional(acts);
  auto weightsOptional = boost::make_optional(weights);
  convolutionPreprocess(graph, params, options, plan, level, indices,
                        actsOptional, weightsOptional,
                        rearrangeProg, rearrangeWritten, rearrangeActs,
                        rearrangeWeights, debugPrefix);
  acts = *actsOptional;
  weights = *weightsOptional;
}

// Postprocess results of convolution
// - undo any flattening of the field
// - undo any padding
static Tensor
convolutionPostprocess(Graph &graph,
                       const ConvParams &originalParams,
                       const ConvTransform &transform,
                       Tensor activations,
                       Sequence &copies,
                       const std::string &debugPrefix) {
  auto postAddExtraDimsParams = originalParams;
  if (transform.extraFieldDims) {
    addExtraDims(postAddExtraDimsParams, transform.extraFieldDims);
  }
  auto postDeferDilationParams =
      calculateParamsWithDeferredDilation(postAddExtraDimsParams,
                                          transform.dilatePostConv);
  auto postExpandParams = postDeferDilationParams;
  if (transform.swapOperands) {
    swapOperands(postExpandParams);
  }
  for (auto dim : transform.expandDims) {
    expandSpatialDim(
          graph, postExpandParams, nullptr, nullptr, dim, debugPrefix);
  }
  auto postOutChanFlattenParams = postExpandParams;
  if (!transform.outChanFlattenDims.empty()) {
    swapOperands(postOutChanFlattenParams);
    for (auto dim : transform.outChanFlattenDims) {
      expandSpatialDim(graph,
                       postOutChanFlattenParams,
                       nullptr,
                       nullptr,
                       dim,
                       debugPrefix);
      // Flatten into the batch axis (this will become the output channel
      // axis when we swap back).
      postOutChanFlattenParams.batchSize *=
          postOutChanFlattenParams.inputFieldShape[dim];
      postOutChanFlattenParams.inputFieldShape[dim] = 1;
    }
    swapOperands(postOutChanFlattenParams);
  }
  // Undo padding.
  assert(activations.dim(activations.rank() - 1) >=
         postOutChanFlattenParams.outputChannels);
  const auto outChanPadding = activations.dim(activations.rank() - 1) -
                              postOutChanFlattenParams.outputChannels;
  activations = pad(graph, activations, 0, -static_cast<int>(outChanPadding),
                    activations.rank() - 1);
  // Undo flattening of the batch / spatial fields.
  if (!transform.flattenDims.empty()) {
    for (auto it = transform.flattenDims.begin(),
         end = std::prev(transform.flattenDims.end()); it != end; ++it) {
      const auto fromDimIndex = *it;
      const auto toDimIndex = transform.flattenDims.back();
      const auto fromSize =
          fromDimIndex ?
            postOutChanFlattenParams.inputFieldShape[fromDimIndex - 1] :
            postOutChanFlattenParams.batchSize;
      activations =
          unflattenDims(activations, 1 + fromDimIndex, 1 + toDimIndex,
                        fromSize);
    }
  }
  // Undo flattening into output channels.
  for (auto it = transform.outChanFlattenDims.rbegin(),
       end = transform.outChanFlattenDims.rend(); it != end; ++it) {
    const auto spatialDim = *it;
    const auto spatialDimSize =
        postDeferDilationParams.getOutputSize(spatialDim);
    activations =
        unflattenDims(activations, 2 + spatialDim, activations.rank() - 1,
                      spatialDimSize);
  }
  // Undo the swapping of operands.
  if (transform.swapOperands) {
    activations = activations.dimShufflePartial({1, activations.rank() - 1},
                                                {activations.rank() - 1, 1});
  }
  // Perform any dilations that were deferred until after the convolution.
  if (!transform.dilatePostConv.empty()) {
    // Create a dilated padded view of the activations and copy it to a
    // new variable. It is not valid to return the view as the result as the
    // convolution function is expected to be a writable tensor.
    auto outChansPerGroup = poputil::detectInnermostGrouping(graph,
                                                                  activations);
    auto activationsView = activations;
    // View that matches the activations view except each zero element is
    // replaced with the nearest non zero element. This is used to
    // determine the tile mapping of the variable we create.
    auto mappingView = activations;
    for (const auto spatialDim : transform.dilatePostConv) {
      const auto dilation =
          postAddExtraDimsParams.inputTransform.dilation[spatialDim];
      const auto paddingLower =
          postAddExtraDimsParams.outputTransform.paddingLower[spatialDim];
      const auto paddingUpper =
          postAddExtraDimsParams.outputTransform.paddingUpper[spatialDim];
      const auto dim = 2 + spatialDim;
      activationsView =
          dilate(graph, activationsView, dilation, dim, debugPrefix);
      mappingView = dilateWithNearestNeighbour(mappingView, dilation, dim);
      activationsView = pad(graph, activationsView, paddingLower, paddingUpper,
                            dim);
      // pad with nearest neighbour.
      mappingView = pad(mappingView, paddingLower, paddingUpper, dim,
                        popops::padding::Type::EDGE);
    }
    assert(activationsView.shape() == mappingView.shape());
    activationsView = splitActivationChanGroups(activationsView,
                                                outChansPerGroup);
    mappingView = splitActivationChanGroups(mappingView, outChansPerGroup);
    activations = graph.addVariable(activationsView.elementType(),
                                    activationsView.shape(),
                                    "activations");
    graph.setTileMapping(activations, graph.getTileMapping(mappingView));
    copies.add(Copy(activationsView, activations));
    activations = unsplitActivationChanGroups(activations);
  }
  // Remove extra dimensions.
  if (transform.extraFieldDims) {
    std::vector<std::size_t> toSqueeze(transform.extraFieldDims);
    std::iota(toSqueeze.begin(), toSqueeze.end(), std::size_t(2));
    activations = activations.squeeze(toSqueeze);
  }
  return activations;
}

static Tensor
iterateTilePartitionImpl(
    Graph &graph, ConvParams params, Plan plan, unsigned level,
    bool postPreprocess, Tensor *actsPtr, Tensor *weightsPtr,
    const std::vector<ConvIndices> &indices, const ConvOptions &options,
    std::function<void (unsigned, Tensor *, Tensor *)> f) {
  boost::optional<Tensor> acts, weights;
  if (actsPtr)
    acts = *actsPtr;
  if (weightsPtr)
    weights = *weightsPtr;
  if (!postPreprocess) {
    // Transform.
    convolutionPreprocess(graph, params, options, plan, level, indices,
                          acts, weights);
  }
  Tensor out;
  if (level == plan.partitions.size()) {
    const auto &target = graph.getTarget();
    const auto tile = linearizeTileIndices(target, options, indices, plan);
    f(tile, acts.get_ptr(), weights.get_ptr());
  } else {
    const auto &partition = plan.partitions[level];
    iteratePartition(params, partition, [&](const ConvIndices &levelIndices,
                                            const ConvSlice &slice) {
      // Get sub convolution
      ConvParams subParams = params;
      auto subActs = acts;
      auto subWeights = weights;
      auto subIndices = indices;
      subIndices.push_back(levelIndices);
      getSubConvolution(slice, subParams, subActs.get_ptr(),
                        subWeights.get_ptr());
      iterateTilePartitionImpl(graph, subParams, plan, level + 1, false,
                               subActs.get_ptr(), subWeights.get_ptr(),
                               subIndices, options, f);
    });
  }
  return out;
}

/// Map the input tensor such that the exchange required during the
/// convolution operation is minimized. If \a isActs is true then the
/// tensor is mapped assuming it the activations operand in convolution
/// operation, otherwise it is mapped assuming it is the weights operand.
static void mapActivationsOrWeights(Graph &graph, ConvParams params,
                                    Plan plan, unsigned level,
                                    bool postPreprocess,
                                    const std::vector<ConvIndices> &indices,
                                    Tensor in, bool isActs,
                                    const ConvOptions &options) {
  auto flattenedIn = in.flatten();
  in = isActs ? unsplitActivationChanGroups(in) : ungroupWeights(in);

  // Build a map from the input to the set of tiles that access them.
  const auto numTiles = graph.getTarget().getNumTiles();
  TensorUseTracker useTracker(numTiles);
  iterateTilePartitionImpl(graph, params, plan, level, postPreprocess,
                           isActs ? &in : nullptr,
                           isActs ? nullptr : &in, indices, options,
                           [&](unsigned tile, Tensor *acts, Tensor *weights) {
    assert((acts && !weights) || (weights && !acts));
    useTracker.add(graph, tile, acts ? *acts : *weights);
  });
  // Limit the minimum number of bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was chosen
  // experimentally.
  const auto inType = params.dType;
  const auto inTypeSize = graph.getTarget().getTypeSize(inType);
  const auto minBytesPerTile = isActs ? 128 : 256;
  const auto minElementsPerTile =
    (minBytesPerTile + inTypeSize - 1) / inTypeSize;
  const auto grainSize =
      isActs ? plan.inChansPerGroup :
               plan.inChansPerGroup * plan.partialChansPerGroup;
  if (useTracker.empty()) {
    mapTensorLinearly(graph, in);
  } else {
    useTracker.mapTensorsByUse(graph, grainSize, minElementsPerTile, true);
  }
}

static void
mapActivations(Graph &graph, ConvParams params, Plan plan, unsigned level,
               bool postPreprocess, const std::vector<ConvIndices> &indices,
               Tensor acts, const ConvOptions &options) {
  return mapActivationsOrWeights(graph, params, plan, level, postPreprocess,
                                 indices, acts, true, options);
}

static void
mapWeights(Graph &graph, ConvParams params, Plan plan, unsigned level,
           bool postPreprocess, const std::vector<ConvIndices> &indices,
           Tensor weights, const ConvOptions &options) {
  return mapActivationsOrWeights(graph, params, plan, level, postPreprocess,
                                 indices, weights, false, options);
}

static Tensor
createInputImpl(Graph &graph, const ConvParams &params,
                unsigned level, bool postPreprocess,
                const std::vector<ConvIndices> &indices,
                const std::string &name,
                const Plan &plan, const ConvOptions &options) {
  const auto inNumChans = params.getNumInputChansPerConvGroup();
  const auto inChansPerGroup = getInChansPerGroup(plan, inNumChans);
  assert(params.getNumInputChansPerConvGroup() % inChansPerGroup == 0);
  std::vector<std::size_t> tensorShape = {
    params.getNumConvGroups(),
    params.getNumInputChansPerConvGroup() / inChansPerGroup,
    params.getBatchSize(),
  };
  tensorShape.insert(tensorShape.end(), params.inputFieldShape.begin(),
                     params.inputFieldShape.end());
  tensorShape.push_back(inChansPerGroup);
  auto t = graph.addVariable(params.dType, tensorShape, name);
  mapActivations(graph, params, plan, level,
                 postPreprocess, indices, t, options);
  t = unsplitActivationChanGroups(t);
  return t;
}

static Tensor
createInput(Graph &graph, const ConvParams &params_,
            const std::string &name,
            const ConvOptions &options,
            PlanningCache *cache) {
  auto params = canonicalizeParams(params_);
  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  // If the output channels are split sequentially then create an input that
  // is optimized to the output channels actually calculated in one step (i.e.
  // the output channels divided by the split).
  assert(params.outputChannels % plan.outChanSerialSplit == 0);
  params.outputChannels /= plan.outChanSerialSplit;
  auto input = createInputImpl(graph, params, 0, false, {},
                               name, plan, options);
  input = actsToExternalShape(input);
  return input;
}

Tensor
createInput(Graph &graph, const ConvParams &params_,
            const std::string &name,
            const poplar::OptionFlags &options_,
            PlanningCache *cache) {
  const auto options = parseConvOptions(graph.getTarget(), options_);
  return createInput(graph, params_, name, options, cache);
}

static Tensor
createWeightsImpl(Graph &graph, const ConvParams &params, unsigned level,
                  bool postPreprocess,
                  const std::vector<ConvIndices> &indices,
                  const std::string &name, const Plan &plan,
                  const ConvOptions &options) {
  const auto dType = params.dType;
  const auto inNumChans = params.getNumInputChansPerConvGroup();
  const auto outNumChans = params.getNumOutputChansPerConvGroup();
  const auto weightOutChansPerGroup =
      getWeightOutChansPerGroup(plan, outNumChans);
  assert(outNumChans % weightOutChansPerGroup == 0);
  const auto weightNumOutChanGroups = outNumChans / weightOutChansPerGroup;
  const auto weightInChansPerGroup = getWeightInChansPerGroup(plan, inNumChans);
  assert(inNumChans % weightInChansPerGroup == 0);
  const auto weightNumInChanGroups = inNumChans / weightInChansPerGroup;
  std::vector<std::size_t> weightsShape = {
    params.getNumConvGroups(),
    weightNumOutChanGroups,
    weightNumInChanGroups
  };
  weightsShape.insert(weightsShape.end(), params.kernelShape.begin(),
                      params.kernelShape.end());
  weightsShape.push_back(weightOutChansPerGroup);
  weightsShape.push_back(weightInChansPerGroup);
  auto weights = graph.addVariable(dType, weightsShape, name);
  mapWeights(graph, params, plan, level, postPreprocess,
             indices, weights, options);
  weights = ungroupWeights(weights);
  return weights;
}

static Tensor
createWeights(Graph &graph,
              const ConvParams &params_, const std::string &name,
              const ConvOptions &options,
              PlanningCache *cache) {
  auto params = canonicalizeParams(params_);
  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  // If the output channels are split sequentially then create weights
  // optimized for one step and then broadcast them.
  assert(params.outputChannels % plan.outChanSerialSplit == 0);
  params.outputChannels /= plan.outChanSerialSplit;
  auto weights =
      weightsToExternalShape(createWeightsImpl(graph, params, 0, false, {},
                                               name, plan, options));
  if (plan.outChanSerialSplit > 1) {
    // We split the weights up so broadcast out to get the full weights.
    auto fullWeights = weights;
    for (unsigned i = 1; i < plan.outChanSerialSplit; ++i) {
      fullWeights = concat(fullWeights, graph.clone(weights, name), 1);
    }
    weights = fullWeights;
  }
  return weights;
}

Tensor
createWeights(Graph &graph,
              const ConvParams &params_, const std::string &name,
              const poplar::OptionFlags &options_,
              PlanningCache *cache) {
  const auto options = parseConvOptions(graph.getTarget(), options_);
  return createWeights(graph, params_, name, options, cache);
}


static void
mapBiases(poplar::Graph &graph, const poplar::Tensor &biases,
          const poplar::Tensor &out) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  TensorUseTracker useTracker(numTiles);
  // Create a view of the output where channels are the outermost dimension.
  auto outRegrouped = out.dimShufflePartial({out.rank() - 1}, {1})
                         .flatten(2, out.rank())
                         .flatten(0, 2);
  auto outMapping = graph.getTileMapping(outRegrouped);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    for (const auto &interval : outMapping[tile]) {
      unsigned chanBegin = interval.begin() / outRegrouped.dim(1);
      unsigned chanEnd = (interval.end() + outRegrouped.dim(1) - 1) /
                         outRegrouped.dim(1);
      useTracker.add(graph, tile, biases.slice(chanBegin, chanEnd));
    }
  }
  const auto dType = out.elementType();
  const auto grainSize = target.getVectorWidth(dType);

  // Limit the minimum number of bias bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto dTypeSize = target.getTypeSize(dType);
  const auto minBytesPerTile = 8;
  const auto minElementsPerTile =
      (minBytesPerTile + dTypeSize - 1) / dTypeSize;

  useTracker.mapTensorsByUse(graph, grainSize, minElementsPerTile);
}

poplar::Tensor
createBiases(poplar::Graph &graph, const Tensor &acts_,
             const std::string &name) {
  const auto acts = actsToInternalShape(acts_, 1, acts_.dim(1));
  const auto numOutChans = acts.dim(acts.rank() - 1);
  const auto dType = acts.elementType();
  auto biases = graph.addVariable(dType, {numOutChans}, name);
  mapBiases(graph, biases, acts);
  return biases;
}

struct ConvOutputSlice {
  unsigned outXBegin;
  unsigned outXEnd;
  unsigned b;
  std::vector<unsigned> outerFieldIndices;
  unsigned outZGroup;
  unsigned cg;
  ConvOutputSlice(unsigned outXBegin, unsigned outXEnd, unsigned b,
                  std::vector<unsigned> outerFieldIndices,
                  unsigned outZGroup, unsigned cg) :
    outXBegin(outXBegin), outXEnd(outXEnd),
    b(b), outerFieldIndices(std::move(outerFieldIndices)), outZGroup(outZGroup),
    cg(cg) {}

};

static std::vector<std::vector<ConvOutputSlice>>
partitionConvOutputBetweenWorkers(const Graph &graph,
                                  unsigned batchBegin, unsigned batchEnd,
                                  const std::vector<unsigned> &outFieldBegin,
                                  const std::vector<unsigned> &outFieldEnd,
                                  unsigned outZGroupBegin,
                                  unsigned outZGroupEnd,
                                  unsigned cgBegin, unsigned cgEnd) {
  const auto numFieldDims = outFieldBegin.size();
  assert(outFieldEnd.size() == numFieldDims);
  std::vector<std::vector<ConvOutputSlice>> perWorkerConvOutputSlices;
  const auto &target = graph.getTarget();
  std::vector<unsigned> rowIterationSpace = {
    outZGroupEnd - outZGroupBegin,
    batchEnd - batchBegin,
    cgEnd - cgBegin
  };
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    rowIterationSpace.push_back(outFieldEnd[dim] - outFieldBegin[dim]);
  }
  const auto numRows = product(rowIterationSpace);
  const auto numWorkers = target.getNumWorkerContexts();
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numRows);
  rowIterationSpace.push_back(rowSplitFactor);
  const auto numPartRows = numRows * rowSplitFactor;
  const auto outXBegin = outFieldBegin.back();
  const auto outXEnd = outFieldEnd.back();
  const auto outWidth = outXEnd - outXBegin;
  for (unsigned worker = 0; worker != numWorkers; ++worker) {
    const auto begin = (worker * numPartRows) / numWorkers;
    const auto end = ((worker + 1) * numPartRows) / numWorkers;
    perWorkerConvOutputSlices.emplace_back();
    for (unsigned partRow = begin; partRow != end; ++partRow) {
      auto indices = unflattenIndex(rowIterationSpace, partRow);
      const auto ocg = outZGroupBegin + indices[0];
      const auto b = batchBegin + indices[1];
      const auto cg = cgBegin + indices[2];
      std::vector<unsigned> outerFieldIndices;
      for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
        outerFieldIndices.push_back(outFieldBegin[dim] + indices[dim + 3]);
      }
      const auto partInRow = indices.back();
      const auto workerOutXBegin =
          outXBegin + (partInRow * outWidth) / rowSplitFactor;
      const auto workerOutXEnd =
          outXBegin + ((partInRow + 1) * outWidth) / rowSplitFactor;
      if (workerOutXBegin == workerOutXEnd)
        continue;
      if (!perWorkerConvOutputSlices.back().empty() &&
          cg == perWorkerConvOutputSlices.back().back().cg &&
          b == perWorkerConvOutputSlices.back().back().b &&
          ocg == perWorkerConvOutputSlices.back().back().outZGroup &&
          outerFieldIndices ==
          perWorkerConvOutputSlices.back().back().outerFieldIndices) {
        perWorkerConvOutputSlices.back().back().outXEnd = workerOutXEnd;
      } else {
        perWorkerConvOutputSlices.back().emplace_back(workerOutXBegin,
                                                      workerOutXEnd,
                                                      b, outerFieldIndices, ocg,
                                                      cg);
      }
    }
  }
  return perWorkerConvOutputSlices;
}

static bool fitsMachineStride(const Target &target, int stride) {
  int64_t maxLimit = (1 << target.getNumStrideBits()) / 2 - 1;
  int64_t minLimit = -(1 << target.getNumStrideBits()) / 2;
  return stride >= minLimit && stride < maxLimit;
};

// Weights for output channel groups is reordered to be reverse order
static std::vector<Tensor>
reorderWeightsTensor(std::vector<Tensor> &in, unsigned numInGroups,
                     unsigned numOutGroups, unsigned numConvGroups) {
  assert(in.size() == numInGroups * numOutGroups * numConvGroups);
  std::vector<Tensor> reorderedIn;
  for (auto cg = 0U; cg != numConvGroups; ++cg) {
    for (auto ig = 0U; ig != numInGroups; ++ig) {
      for (auto ogp1 = numOutGroups; ogp1 > 0; --ogp1) {
        const auto og = ogp1 - 1;
        auto inIndex = cg * numOutGroups * numInGroups + og * numInGroups
                       + ig;
        reorderedIn.push_back(in[inIndex]);
      }
    }
  }
  return reorderedIn;
}

static void
createConvPartialAmpVertex(Graph &graph,
                           const Plan &plan,
                           unsigned tile,
                           ConvParams params,
                           ComputeSet fwdCS,
                           Tensor in,
                           Tensor weights,
                           Tensor out,
                           bool use128BitConvUnitLoad,
                           const std::string &debugPrefix) {
  assert(params == canonicalizeParams(params));
  const auto &target = graph.getTarget();
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(in.elementType() == FLOAT);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  if (convUnitWeightHeight != 1) {
    assert(weights.dim(3) % convUnitWeightHeight == 0);
    assert(params.inputTransform.truncationLower[0] == 0);
    assert(params.inputTransform.truncationUpper[0] == 0);
    assert(params.inputTransform.dilation[0] == 1);
    assert(params.inputTransform.paddingLower[0] == 0);
    assert(params.inputTransform.paddingUpper[0] == 0);
  }

  const auto numFieldDims = params.getNumFieldDims();
  const unsigned numConvGroups = out.dim(0);
  const unsigned numOutChanGroups = out.dim(1);
  const unsigned numInChanGroups = in.dim(1);
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto partialsType = out.elementType();
  const unsigned inChansPerGroup = plan.inChansPerGroup;

  auto isNonZero = [](unsigned x) {
    return x != 0;
  };

  // If the number of input channels is zero, the output could still be only
  // padding. The 1x1 vertex requires the input channels to be non-zero to
  // write zero to the output. Hence we always use a nx1 vertex if number
  // of input channels is zero.
  bool nx1Vertex =
      numInChanGroups * inChansPerGroup == 0 ||
      product(params.kernelShape) != 1 ||
      params.inputTransform.dilation != params.outputTransform.stride ||
      std::any_of(params.outputTransform.paddingLower.begin(),
                  params.outputTransform.paddingLower.end(),
                  isNonZero) ||
      std::any_of(params.outputTransform.paddingUpper.begin(),
                  params.outputTransform.paddingUpper.end(),
                  isNonZero);
  bool flipOut = params.inputTransform.flip[numFieldDims - 1];

  std::vector<Tensor> weightsWindow;
  for (unsigned cg = 0; cg != numConvGroups; ++cg) {
    for (unsigned ozg = 0; ozg < numOutChanGroups; ++ozg) {
      for (unsigned izg = 0; izg < numInChanGroups; ++izg) {
        auto window = weights[cg][ozg][izg].flatten();
        weightsWindow.push_back(window.flatten());
      }
    }
  }

  const auto contextsPerVertex = target.getNumWorkerContexts();
  // The number of n x 1 x ... 1 slices required to cover the kernel in each
  // dimension.
  auto numSubKernelSlices = params.kernelShape;
  assert(numSubKernelSlices[0] % convUnitWeightHeight == 0);
  numSubKernelSlices[0] /= convUnitWeightHeight;
  const auto numSubKernelPositions = product(numSubKernelSlices);

  auto kernelInnerElements = product(numSubKernelSlices) /
                             numSubKernelSlices[0];
  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;

  const auto convInputLoadElems =
      target.getConvUnitInputLoadElemsPerCycle(in.elementType() == FLOAT);


  std::vector<std::vector<unsigned>> worklist(contextsPerVertex *
                                              numSubKernelPositions);
  struct Partition {
    std::vector<std::size_t> outBeginIndices;
    unsigned outXWidth;
    std::vector<std::size_t> inBeginIndices;
    unsigned inXWidth;
    unsigned context;
    unsigned subKernelPosition;
  };

  std::vector<Partition> partitions;
  for (unsigned k = 0; k != numSubKernelPositions; ++k) {
    auto kernelBeginIndices = unflattenIndex(numSubKernelSlices, k);
    kernelBeginIndices[0] = kernelBeginIndices[0] * convUnitWeightHeight;
    std::vector<unsigned> tileConvOutBegin;
    std::vector<unsigned> tileConvOutSize;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      const auto kernelBeginIndex = kernelBeginIndices[dim];
      const auto kernelEndIndex =
          kernelBeginIndex + (dim == 0 ? convUnitWeightHeight : 1);
      const auto outputSize = params.getOutputSize(dim);
      auto convOutRange =
          getOutputRangeForKernelRange(dim, {0, outputSize},
                                       {kernelBeginIndex, kernelEndIndex},
                                       params);
      tileConvOutBegin.push_back(convOutRange.first);
      tileConvOutSize.push_back(convOutRange.second - convOutRange.first);
    }
    if (product(tileConvOutSize) == 0)
      continue;
    auto workerPartition =
        partitionConvPartialByWorker(params.getBatchSize(),
                                     tileConvOutSize,
                                     contextsPerVertex,
                                     params.inputTransform.dilation,
                                     params.outputTransform.stride);
    for (unsigned i = 0; i != contextsPerVertex; ++i) {
      for (const auto &partialRow : workerPartition[i]) {
        auto workerOutXBegin = tileConvOutBegin.back() + partialRow.xBegin;
        auto workerOutXEnd = tileConvOutBegin.back() + partialRow.xEnd;
        std::tie(workerOutXBegin, workerOutXEnd) =
            getOutputRangeForKernelIndex(numFieldDims - 1,
                                         {workerOutXBegin, workerOutXEnd},
                                         kernelBeginIndices.back(), params);
        const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
        if (workerOutWidth == 0)
          continue;
        std::vector<std::size_t> outBeginIndices = { partialRow.b };
        for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
          outBeginIndices.push_back(partialRow.outerFieldIndices[dim] +
                                    tileConvOutBegin[dim]);
        }
        outBeginIndices.push_back(partialRow.xBegin +
                                  tileConvOutBegin.back());
        std::vector<std::size_t> inBeginIndices = { partialRow.b };
        if (numFieldDims > 1) {
          const auto kOuterBegin = kernelBeginIndices[0];
          const auto kOuterEnd = kOuterBegin + convUnitWeightHeight;
          const auto outOuterIndex = tileConvOutBegin[0] +
                                     partialRow.outerFieldIndices[0];
          for (unsigned k = kOuterBegin; k != kOuterEnd; ++k) {
            auto inOuterIndex = getInputIndex(0, outOuterIndex, k, params);
            if (inOuterIndex != ~0U) {
            auto inOuterBeginIndex =
                  inOuterIndex +
                  (params.inputTransform.flip.front() !=
                   params.kernelTransform.flip.front() ? 1 : -1) *
                  (k - kOuterBegin) * params.kernelTransform.dilation.front();
              inBeginIndices.push_back(inOuterBeginIndex);
              break;
            }
          }
          if (inBeginIndices.size() < 2) {
            continue;
          }
        }
        for (unsigned dim = 1; dim + 1 < numFieldDims; ++dim) {
          auto inIndex =
              getInputIndex(dim,
                            tileConvOutBegin[dim] +
                                partialRow.outerFieldIndices[dim],
                            kernelBeginIndices[dim], params);
          assert(inIndex != ~0U);
          inBeginIndices.push_back(inIndex);
        }
        auto workerInXRange =
            getInputRange(numFieldDims - 1, {workerOutXBegin, workerOutXEnd},
                          kernelBeginIndices.back(), params);
        assert(workerInXRange.first != ~0U);
        inBeginIndices.push_back(workerInXRange.first);
        partitions.push_back({std::move(outBeginIndices),
                              workerOutWidth,
                              std::move(inBeginIndices),
                              workerInXRange.second - workerInXRange.first,
                              i, k});
      }
    }
  }
  assert(!partitions.empty());
  std::vector<std::size_t> inputBatchAndFieldShape = {params.getBatchSize()};
  std::vector<std::size_t> outputBatchAndFieldShape = {params.getBatchSize()};
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    inputBatchAndFieldShape.push_back(params.inputFieldShape[dim]);
    outputBatchAndFieldShape.push_back(params.getOutputSize(dim));
  }

  bool useConvPartial1x1OutVertex = !nx1Vertex;

  if (useConvPartial1x1OutVertex) {
    // In most common cases there should only be one partition per worker for
    // a 1x1 vertex. To avoid having two types of 1x1 vertices we just make it
    // a nx1 vertex if there's more than one partition.
    std::vector<unsigned> partitionsPerContext(contextsPerVertex);
    std::for_each(partitions.begin(), partitions.end(),
                  [&](const Partition &p) {
      partitionsPerContext[p.context] += 1;
    });

    // find if any of the contexts has more than one partition
    unsigned contextsHaveMoreThanOnePartition = false;
    for (auto v : partitionsPerContext) {
      if (v > 1) {
        contextsHaveMoreThanOnePartition = true;
        break;
      }
    }
    useConvPartial1x1OutVertex = !contextsHaveMoreThanOnePartition;
  }

  // create worklist now that dimensions of all splits are known
  for (const auto &p : partitions) {
    const auto outBeginOffset =
        flattenIndex(outputBatchAndFieldShape, p.outBeginIndices);
    const auto inBeginOffset =
        flattenIndex(inputBatchAndFieldShape, p.inBeginIndices);
    const auto outOffset = flipOut ? outBeginOffset + p.outXWidth - 1 :
                                     outBeginOffset;
    const auto numFieldElems =
        useConvPartial1x1OutVertex ?
          p.outXWidth : (p.outXWidth + outStrideX - 1) / outStrideX;
    const auto wIndex = p.subKernelPosition * contextsPerVertex + p.context;
    worklist[wIndex].push_back(outOffset);
    worklist[wIndex].push_back(numFieldElems);
    worklist[wIndex].push_back(inBeginOffset);
  }

  std::vector<Tensor> outWindow;
  std::vector<Tensor> inWindow;

  for (unsigned cg = 0; cg != numConvGroups; ++cg) {
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      auto o = out[cg][ozg];
      outWindow.push_back(o.flatten());
    }
    // TODO if the tile kernel size is 1 and the stride is greater than one we
    // could subsample the input instead of using input striding.
    for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
      auto window = in[cg][izg];
      inWindow.push_back(window.flatten());
    }
  }
  // This stride is what's used to move down one element in the input field by
  // the vertex.
  int inRowStride =
      getInRowStride(params, product(inputBatchAndFieldShape) /
                                   (inputBatchAndFieldShape[0]
                                    * inputBatchAndFieldShape[1]),
                     useConvPartial1x1OutVertex, convUnitWeightHeight);

  int transformedInStride =
      (static_cast<int>(inStrideX) - 1 -
       static_cast<int>(convUnitWeightHeight - 1) * inRowStride) *
      static_cast<int>(inChansPerGroup / convInputLoadElems) + 1;
  // fill in worklist
  unsigned outStrideToUse = useConvPartial1x1OutVertex ? 1 : outStrideX;
  int scaledOutStride = static_cast<int>(outStrideToUse * outChansPerGroup);

  int transformedOutStride =
      (plan.types.back().partialType == poplar::FLOAT ? -6 : -4) +
      (flipOut ? -scaledOutStride : scaledOutStride);

  int transformedInRowStride =  (inRowStride - 1) *
      static_cast<int>(inChansPerGroup / convInputLoadElems) + 1;

  // Limits for field and worklist elements
  const auto unsignedMax = std::numeric_limits<unsigned short>::max();
  const auto signedMax = std::numeric_limits<short>::max();
  const auto signedMin = std::numeric_limits<short>::min();

  bool useLimitedVer = true;
  const auto zerosInfo = outWindow[0].numElements();
  if (!fitsMachineStride(target, transformedOutStride / 2) ||
      !fitsMachineStride(target, transformedInStride) ||
      !fitsMachineStride(target, transformedInRowStride))
    useLimitedVer = false;

  if ((numConvGroups - 1 > unsignedMax) ||
      (numOutChanGroups - 1 > unsignedMax) ||
      (numInChanGroups > unsignedMax) ||
      (transformedInStride < signedMin) ||
      (transformedInStride > signedMax) ||
      (outChansPerGroup > unsignedMax) ||
      (transformedOutStride < signedMin) ||
      (transformedOutStride > signedMax))
    useLimitedVer = false;

  const auto doubleWordWrites =
      zerosInfo / (8 / target.getTypeSize(outWindow[0].elementType()));
  const auto doubleWordWritesPerWorker =
      (doubleWordWrites + contextsPerVertex - 1) / contextsPerVertex;

  if (!useConvPartial1x1OutVertex) {
    if ((kernelInnerElements - 1 > unsignedMax) ||
        (numSubKernelSlices[0] - 1 > unsignedMax) ||
        (convUnitWeightHeight - 1 > unsignedMax) ||
        (transformedInRowStride > signedMax) ||
        (transformedInRowStride < signedMin) ||
        (inChansPerGroup > unsignedMax) ||
        doubleWordWritesPerWorker > target.getRptCountMax())
      useLimitedVer = false;

    if (convUnitWeightHeight != 1 &&
        convUnitWeightHeight != 2 &&
        convUnitWeightHeight != 4)
      useLimitedVer = false;
  }
  // check if all worklist items meet range constraints
  for (auto j = 0U; j != worklist.size() && useLimitedVer; ++j) {
    const auto &vec = worklist[j];
    for (auto i = 0U; i != vec.size(); ++i) {
      // worklist is a multiple of 3.
      // i % 3 == 0 : output offset
      // i % 3 == 1 : number of field elems
      // i % 3 == 2 : input offset
      if ((i % 3) == 1) {
        if (vec[i] >  target.getRptCountMax()) {
          useLimitedVer = false;
          break;
        }
      } else {
        if (vec[i] > unsignedMax) {
          useLimitedVer = false;
          break;
        }
      }
    }
  }

  const auto worklistEntryType =
      useLimitedVer ? UNSIGNED_SHORT : UNSIGNED_INT;

  auto codeletName = useConvPartial1x1OutVertex ?
                       "poplin::ConvPartial1x1Out" :
                       "poplin::ConvPartialnx1";
  auto v =
        graph.addVertex(fwdCS,
                        templateVertex(codeletName, in.elementType(),
                                       plan.types.back().partialType,
                                       useLimitedVer ? "true" : "false",
                                       use128BitConvUnitLoad ? "true" :
                                                               "false"));

  // The parameters are modified to what the vertex uses
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"],
                  reorderWeightsTensor(weightsWindow, numInChanGroups,
                                       numOutChanGroups, numConvGroups));
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroupsM1"], numOutChanGroups - 1);
  graph.setInitialValue(v["numInGroups"], numInChanGroups);
  assert(inChansPerGroup % convInputLoadElems == 0);

  graph.setInitialValue(v["transformedInStride"], transformedInStride);

  graph.setInitialValue(v["numConvGroupsM1"], numConvGroups - 1);

  graph.setInitialValue(v["transformedOutStride"], transformedOutStride);

  // Worklists are 2D for nx1 and 1D for 1x1
  if (useConvPartial1x1OutVertex) {
    std::vector<unsigned> worklist1x1(contextsPerVertex * 3);
    for (unsigned i = 0; i < worklist.size(); ++i) {
      std::copy(std::begin(worklist[i]), std::end(worklist[i]),
                worklist1x1.begin() + 3 * i);
    }
    auto t = graph.addConstant(worklistEntryType,
                               {worklist1x1.size()},
                               worklist1x1.data(),
                               debugPrefix + "/worklists");
    graph.setTileMapping(t, tile);
    graph.connect(v["worklists"], t);
  } else {
    graph.setFieldSize(v["worklists"], worklist.size());
    for (unsigned i = 0;i < worklist.size(); ++i) {
      auto t = graph.addConstant(worklistEntryType,
                                 {worklist[i].size()},
                                 worklist[i].data(),
                                 debugPrefix + "/worklists");
      graph.setTileMapping(t, 0);
      graph.connect(v["worklists"][i], t);
    }
  }

  if (!useConvPartial1x1OutVertex) {
    graph.setInitialValue(v["kernelInnerElementsM1"],
        kernelInnerElements - 1);
    graph.setInitialValue(v["kernelOuterSizeM1"], numSubKernelSlices[0] - 1);
    graph.setInitialValue(v["ampKernelHeightM1"], convUnitWeightHeight - 1);
    graph.setInitialValue(v["transformedInRowStride"],
        transformedInRowStride);
    graph.setInitialValue(v["zerosInfo"], zerosInfo);
  }
  graph.setTileMapping(v, tile);
}

/// Return whether the specified convolution produces an output that is known
/// to be all zeros. A convolution that produces an empty output is trivially
/// a zero convolution.
static bool isZeroConvolution(const ConvParams &params) {
  if (params.inputChannels == 0 ||
      params.outputChannels == 0 ||
      params.batchSize == 0 ||
      params.numConvGroups == 0)
    return true;
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (params.outputTransform.paddingLower[dim] +
        params.outputTransform.paddingUpper[dim] ==
        params.getOutputSize(dim)) {
      return true;
    }
  }
  return false;
}

static Tensor sliceOutput(const Tensor &out, const ConvSlice &slice,
                          unsigned outChansPerGroup) {
  std::vector<std::size_t> begin, end;
  begin.push_back(slice.cgBegin); end.push_back(slice.cgEnd);
  assert(slice.outChanBegin % outChansPerGroup == 0);
  begin.push_back(slice.outChanBegin / outChansPerGroup);
  assert(slice.outChanEnd % outChansPerGroup == 0);
  end.push_back(slice.outChanEnd / outChansPerGroup);
  begin.push_back(slice.batchBegin); end.push_back(slice.batchEnd);
  const auto numFieldDims = slice.outFieldBegin.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    begin.push_back(slice.outFieldBegin[dim]);
    end.push_back(slice.outFieldEnd[dim]);
  }
  begin.push_back(0);
  end.push_back(outChansPerGroup);
  return out.slice(begin, end);
}

/// Return the tensor \a t with the specified amount of padding added to the
/// specified dimension. The padding elements are added as a new variable
/// which is concaternated to the \a padding tensors. It is the caller's
/// responsibility to initialize the padding.
static Tensor padWithVariable(Graph &graph, Tensor t, unsigned paddingLower,
                              unsigned paddingUpper, unsigned dim,
                              Tensor &padding,
                              const std::string &debugPrefix) {
  auto paddingSize = paddingLower + paddingUpper;
  auto paddingShape = t.shape();
  paddingShape[dim] = paddingSize;
  auto paddingTensor = graph.addVariable(t.elementType(), paddingShape,
                                         debugPrefix + "/zeroPadding");
  auto paddingLowerTensor = paddingTensor.slice(0, paddingLower, dim);
  auto paddingUpperTensor = paddingTensor.slice(paddingLower, paddingSize, dim);
  padding = concat(padding, paddingTensor.flatten());
  return concat({paddingLowerTensor, t, paddingUpperTensor}, dim);
}

static void createConvPartialAmpVertices(Graph &graph,
                                         const Plan &plan,
                                         unsigned tile,
                                         ConvParams params,
                                         Sequence &copies,
                                         Tensor &copyWritten,
                                         ComputeSet fwdCS,
                                         Tensor in, Tensor weights,
                                         Tensor out,
                                         bool use128BitConvUnitLoad,
                                         const std::string &debugPrefix) {
  assert(params == canonicalizeParams(params));
  const auto &target = graph.getTarget();
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(in.elementType() == FLOAT);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  if (convUnitWeightHeight != 1) {
    // Padding the input / weights using constants creates aliases of the zero
    // constant which causes a rearrangement between the exchange and compute
    // steps. This rearrangement can double amount of temporary memory required
    // Workaround this by creating a padding variable that in used in place
    // of the constant zero. The size of the variable is equal to the
    // amount of padding required so we can avoid aliasing of elements.
    auto paddingTensor = graph.addConstant(
          weights.elementType(), {0}, 0, debugPrefix + "/paddingTensor");
    graph.setTileMapping(paddingTensor, 0);
    // If we are doing an nx1 convolution we need to pad the weights to a
    // multiple of n.
    const auto kernelHeight = weights.dim(3);
    if (kernelHeight % convUnitWeightHeight != 0) {
      const auto extraKernelPadding =
          (convUnitWeightHeight - kernelHeight % convUnitWeightHeight);
      const auto extraInputPadding =
          extraKernelPadding * params.kernelTransform.dilation[0];
      unsigned extraKernelPaddingLower = 0, extraKernelPaddingUpper = 0;
      auto &flippedExtraKernelPaddingUpper =
          params.kernelTransform.flip[0] ? extraKernelPaddingLower :
                                           extraKernelPaddingUpper;
      auto &inputPaddingLower = params.inputTransform.paddingLower[0];
      auto &inputPaddingUpper = params.inputTransform.paddingUpper[0];
      auto &flippedInputPaddingUpper =
          params.inputTransform.flip[0] ? inputPaddingLower : inputPaddingUpper;
      flippedExtraKernelPaddingUpper += extraKernelPadding;
      flippedInputPaddingUpper += extraInputPadding;
      weights = padWithVariable(graph, weights, extraKernelPaddingLower,
                                extraKernelPaddingUpper, 3, paddingTensor,
                                debugPrefix);
      params.kernelShape[0] += extraKernelPadding;
    }
    // Explicitly truncate, dilate and pad the outermost spatial field of the
    // input.
    const auto inputTruncationLower =
        params.inputTransform.truncationLower[0];
    const auto inputTruncationUpper =
        params.inputTransform.truncationUpper[0];
    in = pad(graph, in, -static_cast<int>(inputTruncationLower),
             -static_cast<int>(inputTruncationUpper), 3);
    params.inputFieldShape[0] -= inputTruncationLower + inputTruncationUpper;
    params.inputTransform.truncationLower[0] = 0;
    params.inputTransform.truncationUpper[0] = 0;

    const auto inputDilation = params.inputTransform.dilation[0];
    in = dilate(graph, in, inputDilation, 3, debugPrefix);
    params.inputTransform.dilation[0] = 1;
    params.inputFieldShape[0] = getDilatedSize(params.inputFieldShape[0],
                                               inputDilation);

    const auto inputPaddingLower = params.inputTransform.paddingLower[0];
    const auto inputPaddingUpper = params.inputTransform.paddingUpper[0];
    in = padWithVariable(graph, in, inputPaddingLower, inputPaddingUpper, 3,
                         paddingTensor, debugPrefix);
    params.inputFieldShape[0] += inputPaddingLower + inputPaddingUpper;
    params.inputTransform.paddingLower[0] = 0;
    params.inputTransform.paddingUpper[0] = 0;
    if (paddingTensor.numElements() != 0) {
      auto c = graph.addConstant(paddingTensor.elementType(),
                                 paddingTensor.shape(),
                                 0,
                                 debugPrefix + "/paddingTensor");
      graph.setTileMapping(c, 0);
      graph.setTileMapping(paddingTensor, tile);
      copies.add(Copy(c, paddingTensor));
      copyWritten = concat(copyWritten, paddingTensor.flatten());
    }
  }

  const auto partialsType = out.elementType();
  auto isNonZero = [](unsigned x) {
    return x != 0;
  };
  bool nx1Vertex =
      product(params.kernelShape) != 1 ||
      params.inputTransform.dilation != params.outputTransform.stride ||
      std::any_of(params.outputTransform.paddingLower.begin(),
                  params.outputTransform.paddingLower.end(),
                  isNonZero) ||
      std::any_of(params.outputTransform.paddingUpper.begin(),
                  params.outputTransform.paddingUpper.end(),
                  isNonZero);
  bool useConvPartial1x1OutVertex = !nx1Vertex;
  const unsigned inChansPerGroup = plan.inChansPerGroup;

  // The number of n x 1 x ... 1 slices required to cover the kernel in each
  // dimension.
  auto numSubKernelSlices = params.kernelShape;
  assert(numSubKernelSlices[0] % convUnitWeightHeight == 0);
  numSubKernelSlices[0] /= convUnitWeightHeight;

  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;

  const auto inRowStrideBeforeSplit =
      getInRowStride(params, product(params.inputFieldShape)
                         / params.inputFieldShape[0],
                     useConvPartial1x1OutVertex, convUnitWeightHeight);
  const auto convInputLoadElems =
      target.getConvUnitInputLoadElemsPerCycle(in.elementType() == FLOAT);
  int transformedInStrideBeforeSplit =
      (static_cast<int>(inStrideX) - 1 -
       static_cast<int>(convUnitWeightHeight - 1) * inRowStrideBeforeSplit) *
      static_cast<int>(inChansPerGroup / convInputLoadElems) + 1;
  int transformedInRowStrideBeforeSplit = (inRowStrideBeforeSplit - 1) *
      static_cast<int>(inChansPerGroup / convInputLoadElems) + 1;

  // Find field split that satisfies machine stride bit-widths
  // Only use input striding to decide to split field as it is most likely to
  // exceed machine strides
  const auto partition =
      splitConvIntoAmpVertices(params, target.getNumStrideBits(),
                               transformedInStrideBeforeSplit,
                               transformedInRowStrideBeforeSplit);

  iteratePartition(params, partition, [&](const ConvIndices &,
                                          const ConvSlice &slice) {
    // Get sub convolution
    ConvParams subParams = params;
    Tensor subIn = unsplitActivationChanGroups(in);
    Tensor subWeights = ungroupWeights(weights);
    getSubConvolution(slice, subParams, &subIn, &subWeights);
    subIn = splitActivationChanGroups(subIn, plan.inChansPerGroup);
    subWeights = groupWeights(subWeights, plan.inChansPerGroup,
                              plan.partialChansPerGroup);
    Tensor subOut = sliceOutput(out, slice, plan.partialChansPerGroup);
    if (isZeroConvolution(subParams)) {
      zero(graph, subOut, tile, fwdCS);
    } else {
      createConvPartialAmpVertex(graph,
                                 plan,
                                 tile,
                                 subParams,
                                 fwdCS,
                                 subIn, subWeights,
                                 subOut,
                                 use128BitConvUnitLoad,
                                 debugPrefix);
    }
  });
}

static void
createConvPartialHorizontalMacVertex(Graph &graph,
                                     const Plan &plan,
                                     unsigned tile,
                                     const ConvParams &params,
                                     ComputeSet fwdCS,
                                     const Tensor &in,
                                     const Tensor &weights,
                                     const Tensor &out,
                                     const std::string &debugPrefix) {
  const auto &target = graph.getTarget();
  const auto numFieldDims = params.getNumFieldDims();
  const auto xDimIndex = numFieldDims - 1;
  const unsigned numConvGroups = out.dim(0);
  const unsigned numOutChanGroups = out.dim(1);
  const unsigned numInChanGroups = in.dim(1);
  const unsigned inChansPerGroup = plan.inChansPerGroup;
  const unsigned outChansPerGroup = plan.partialChansPerGroup;

  bool flipOut = params.inputTransform.flip[xDimIndex];

  assert(outChansPerGroup == 1);
  if (in.elementType() == HALF) {
    assert(inChansPerGroup % 2 == 0);
  }
  const auto outputFieldShape = params.getOutputFieldShape();
  const unsigned numOutFieldElems = product(outputFieldShape);
  if (numOutFieldElems == 0)
    return;

  std::vector<Tensor> outWindow;
  std::vector<Tensor> inWindow;
  std::vector<Tensor> weightsWindow;
  for (unsigned cg = 0; cg != numConvGroups; ++cg) {
    // Output Tensor slices
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      auto o = out[cg][ozg].flatten();
      outWindow.push_back(o);
    }
    // Input tensor slices
    for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
      auto i = in[cg][izg].flatten();
      inWindow.push_back(i);
    }
    // kernel tensor slices
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
        auto w = weights[cg][ozg][izg].flatten();
        weightsWindow.push_back(w);
      }
    }
  }

  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;

  const unsigned numInFieldElems = product(params.inputFieldShape);
  const unsigned numKernelFieldElems = product(params.kernelShape);
  const unsigned kernelSizeX = params.kernelShape.back();
  const auto contextsPerVertex = target.getNumWorkerContexts();
  std::vector<std::vector<unsigned>> worklist(contextsPerVertex
                                              * numKernelFieldElems);
  for (unsigned k = 0; k != numKernelFieldElems / kernelSizeX ; ++k) {
    // unflatten kernel index into a co-ordinate for the kernel
    auto kCoord = unflattenIndex(params.kernelShape, k * kernelSizeX);
    std::vector<unsigned> convOutBegin, convOutEnd;
    for (auto dim = 0U; dim + 1 != numFieldDims; ++dim ) {
      unsigned begin, end;
      std::tie(begin, end) =
        getOutputRangeForKernelIndex(dim,
                                     {0, params.getOutputSize(dim)},
                                     kCoord[dim], params);
      convOutBegin.push_back(begin);
      convOutEnd.push_back(end);
    }
    const auto convOutElems = getNumElementsInSlice(convOutBegin, convOutEnd);
    if (convOutElems == 0)
      continue;
    for (unsigned kx = 0; kx != params.kernelShape.back(); ++kx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRangeForKernelIndex(xDimIndex,
                                       {0, params.getOutputSize(xDimIndex)},
                                       kx,
                                       params);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;

      auto outFieldBegin = convOutBegin;
      outFieldBegin.push_back(convOutXBegin);
      auto outFieldEnd = convOutEnd;
      outFieldEnd.push_back(convOutXEnd);
      auto workerPartition =
          partitionConvOutputBetweenWorkers(graph, 0, params.getBatchSize(),
                                            outFieldBegin, outFieldEnd, 0, 1,
                                            0, 1);
      for (unsigned i = 0; i != contextsPerVertex; ++i) {
        for (const auto &workerSlice : workerPartition[i]) {
          auto workerOutXBegin = workerSlice.outXBegin;
          auto workerOutXEnd = workerSlice.outXEnd;
          std::tie(workerOutXBegin, workerOutXEnd) =
              getOutputRangeForKernelIndex(xDimIndex,
                                           {workerOutXBegin, workerOutXEnd},
                                           kx, params);
          const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
          if (workerOutWidth == 0)
            continue;
          std::vector<std::size_t> workerIn;
          bool validRow = true;
          for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
            auto outIndex = workerSlice.outerFieldIndices[dim];
            auto inIndex =
                getInputIndex(dim, outIndex, kCoord[dim], params);
            if (inIndex == ~0U) {
              validRow = false;
              break;
            }
            workerIn.push_back(inIndex);
          }
          if (!validRow)
            continue;
          unsigned workerInXBegin, workerInXEnd;
          std::tie(workerInXBegin, workerInXEnd) =
              getInputRange(xDimIndex, {workerOutXBegin, workerOutXEnd},
                            kx, params);
          workerIn.push_back(workerInXBegin);

          auto workerOutFieldIndicesBegin =
              vectorConvert<std::size_t>(workerSlice.outerFieldIndices);
          workerOutFieldIndicesBegin.push_back(workerOutXBegin);
          const auto outBeginOffset =
              workerSlice.b * numOutFieldElems +
              flattenIndex(outputFieldShape, workerOutFieldIndicesBegin);

          const auto inBeginOffset =
              workerSlice.b * numInFieldElems +
              flattenIndex(params.inputFieldShape, workerIn);

          auto kIndex = k * kernelSizeX + kx;
          const auto numFieldElems =
              (workerOutWidth + outStrideX - 1) / outStrideX;

          const auto outOffset = flipOut ? outBeginOffset + workerOutWidth - 1 :
                                           outBeginOffset;

          worklist[kIndex * contextsPerVertex + i].push_back(outOffset);
          worklist[kIndex * contextsPerVertex + i].push_back(numFieldElems);
          worklist[kIndex * contextsPerVertex + i].push_back(inBeginOffset);
        }
      }
    }
  }

  int transformedOutStride =
      ((flipOut ? -static_cast<int>(outStrideX) :
                   static_cast<int>(outStrideX)) - 1) * outChansPerGroup;
  const auto transformedInStride = inStrideX * inChansPerGroup;

  // Limits for field and worklist elements
  const auto unsignedMax = std::numeric_limits<unsigned short>::max();
  bool useLimitedVer = true;
  const auto zerosInfo = outWindow[0].numElements();

  const auto doubleWordWrites =
      zerosInfo / (8 / target.getTypeSize(outWindow[0].elementType()));
  const auto doubleWordWritesPerWorker =
      (doubleWordWrites + contextsPerVertex - 1) / contextsPerVertex;

  // check if field elements meet short representation
  if ((outChansPerGroup > unsignedMax) ||
      (inChansPerGroup > unsignedMax) ||
      (numOutChanGroups - 1 > unsignedMax) ||
      (numInChanGroups > unsignedMax) ||
      (numKernelFieldElems - 1 > unsignedMax) ||
      (numConvGroups - 1 > unsignedMax) ||
      doubleWordWritesPerWorker > target.getRptCountMax())
    useLimitedVer = false;

  // check if all worklist items meet range constraints
  for (auto j = 0U; j != worklist.size() && useLimitedVer; ++j) {
    const auto &vec = worklist[j];
    for (auto entry : vec) {
      if (entry > unsignedMax) {
        useLimitedVer = false;
        break;
      }
    }
  }

  if (in.elementType() == HALF) {
    // Conv planner sets a grain size of 2 for input channels per group
    if (inChansPerGroup % 2)
      useLimitedVer = false;
    else {
      const auto maxRptCount = inChansPerGroup % 4 == 0 ? inChansPerGroup / 4 :
                                                          inChansPerGroup / 2;
      if (maxRptCount > target.getRptCountMax())
        useLimitedVer = false;
    }
  } else if (in.elementType() == FLOAT) {
    const auto maxRptCount = inChansPerGroup % 2 == 0 ? inChansPerGroup / 2 :
                                                        inChansPerGroup;
    if (maxRptCount > target.getRptCountMax())
      useLimitedVer = false;
  }

  const auto worklistEntryType = useLimitedVer ? UNSIGNED_SHORT : UNSIGNED_INT;
  auto v = graph.addVertex(fwdCS,
                           templateVertex("poplin::ConvPartialHorizontalMac",
                                          in.elementType(),
                                          plan.types.back().partialType,
                                          useLimitedVer ? "true" : "false"));
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"],
      reorderWeightsTensor(weightsWindow, numInChanGroups,
                           numOutChanGroups, numConvGroups));
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroupsM1"], numOutChanGroups - 1);
  graph.setInitialValue(v["numInGroups"], numInChanGroups);
  graph.setInitialValue(v["kernelSizeM1"], numKernelFieldElems - 1);
  graph.setInitialValue(v["transformedInStride"], transformedInStride);
  graph.setInitialValue(v["transformedOutStride"], transformedOutStride);
  graph.setInitialValue(v["numConvGroupsM1"], numConvGroups - 1);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstant(worklistEntryType,
                               {worklist[i].size()},
                               worklist[i].data(),
                               debugPrefix + "/worklist");
    graph.setTileMapping(t, 0);
    graph.connect(v["worklists"][i], t);
  }
  graph.setInitialValue(v["zerosInfo"], zerosInfo);
  graph.setTileMapping(v, tile);
}

static void
createOuterProductVertex(
    Graph &graph,
    unsigned tile,
    unsigned xBegin, unsigned xEnd,
    const ConvParams &params,
    ComputeSet fwdCS,
    Tensor in,
    Tensor weights,
    const Tensor &out) {
  const auto numFieldDims = params.getNumFieldDims();
  assert(product(params.outputTransform.stride) == 1);
  assert(product(params.inputTransform.dilation) == 1);

  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    in = pad(graph, in,
             -static_cast<int>(params.inputTransform.truncationLower[dim]),
             -static_cast<int>(params.inputTransform.truncationUpper[dim]),
             3 + dim);
    in = pad(graph, in,
             static_cast<int>(params.inputTransform.paddingLower[dim]),
             static_cast<int>(params.inputTransform.paddingUpper[dim]),
             3 + dim);
    weights = pad(graph, weights,
             -static_cast<int>(params.kernelTransform.truncationLower[dim]),
             -static_cast<int>(params.kernelTransform.truncationUpper[dim]),
                  3 + dim);
    weights = pad(graph, weights,
             static_cast<int>(params.kernelTransform.paddingLower[dim]),
             static_cast<int>(params.kernelTransform.paddingUpper[dim]),
                  3 + dim);
  }

  assert(in.dim(1) == 1);
  assert(in.dim(2) == 1);

  // check all input field dimensions other than the innermost is 1
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    assert(in.dim(dim + 3) == 1);
  }
  assert(in.dim(in.rank() - 1) == 1);

  // check every field dimension of the weights tensor is 1
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    assert(weights.dim(dim + 3) == 1);
  }

  assert(weights.dim(2) == 1);
  assert(weights.dim(weights.rank() - 1) == 1);
  assert(out.dim(1) == weights.dim(1));
  assert(out.dim(2) == 1);

  // check all output field dimensions other than the innermost is 1
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    assert(out.dim(dim + 3) == 1);
  }
  assert(out.dim(3 + numFieldDims - 1) == in.dim(3 + numFieldDims - 1));
  assert(out.dim(out.rank() - 1) == weights.dim(weights.rank() - 2));
  const auto chansPerGroup = weights.dim(weights.rank() - 2);
  const auto dType = in.elementType();

  const auto numConvGroups = params.getNumConvGroups();
  for (unsigned cg = 0; cg != numConvGroups; ++cg) {
    auto inWindow = in[cg].flatten().slice(xBegin, xEnd);
    auto outWindow =
        out.slice(cg, cg + 1, 0)
           .slice(xBegin, xEnd, out.rank() - 2)
           .reshape({out.dim(1), (xEnd - xBegin) * chansPerGroup});
    auto weightsWindow = weights[cg].flatten();
    auto v = graph.addVertex(fwdCS,
                             templateVertex(
                               "poplin::OuterProduct", dType
                             ),
                             {{"in", inWindow},
                              {"weights", weightsWindow},
                              {"out", outWindow}});

    graph.setInitialValue(v["chansPerGroup"],
          weightsWindow.numElements() / outWindow.dim(0));

    graph.setTileMapping(v, tile);
  }
}

static void
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      unsigned tile,
                      ConvParams params,
                      Sequence &copies,
                      Tensor &copyWritten,
                      ComputeSet convolveCS,
                      Tensor in, Tensor weights,
                      Tensor out, bool use128BitConvUnitLoad,
                      const std::string &debugPrefix) {
#ifndef NDEBUG
  const auto numOutChans = params.getNumOutputChansPerConvGroup();
  const auto outChansPerGroup = plan.partialChansPerGroup;
  assert(numOutChans % outChansPerGroup == 0);
#endif
  graph.setTileMapping(out, tile);
  in = splitActivationChanGroups(in, plan.inChansPerGroup);
  weights = groupWeights(weights, plan.inChansPerGroup,
                         plan.partialChansPerGroup);
  if (isZeroConvolution(params)) {
    zero(graph, out, tile, convolveCS);
    return;
  }
  switch (plan.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    createConvPartialAmpVertices(graph, plan, tile, params, copies, copyWritten,
                                 convolveCS, in, weights, out,
                                 use128BitConvUnitLoad, debugPrefix);
    break;
  case Plan::Method::MAC:
    createConvPartialHorizontalMacVertex(graph,
                                         plan,
                                         tile,
                                         params,
                                         convolveCS,
                                         in,
                                         weights,
                                         out,
                                         debugPrefix);
    break;
  case Plan::Method::OUTER_PRODUCT:
    {
      const auto &target = graph.getTarget();
      const auto outputLength =
          params.getOutputSize(params.getNumFieldDims() - 1);
      const auto perWorkerRegions =
          splitRegionsBetweenWorkers(target, {{0, outputLength}}, 1);
      for (const auto &entry : perWorkerRegions) {
        assert(entry.size() == 1);
        createOuterProductVertex(graph, tile,
                                 entry[0].begin(), entry[0].end(), params,
                                 convolveCS, in, weights, out);
      }
    }
    break;
  }
}

static bool inputRearrangementIsExpensive(const ConvOptions &options) {
  // During the weight update pass we change the innermost dimension when the
  // activations / deltas are rearranged.
  return options.pass == Pass::TRAINING_WU ||
         options.pass == Pass::FC_TRAINING_WU;
}

static bool weightRearrangementIsExpensive(const ConvOptions &options) {
  // During the weight update pass we change the innermost dimension when the
  // activations / deltas are rearranged.
  return options.pass == Pass::TRAINING_WU ||
         options.pass == Pass::FC_TRAINING_WU;
}

static unsigned getPartialIndex(const ConvIndices &indices,
                                const Partition &partition) {
  const auto numFieldDims = indices.kernel.size();
  unsigned partialIndex = indices.ic;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    partialIndex =
        partialIndex * partition.kernelSplit[dim] +
        indices.kernel[dim];
  }
  return partialIndex;
}

static unsigned getOutputIndex(const ConvIndices &indices,
                               const Partition &partition) {
  assert(indices.cg < partition.convGroupSplit &&
         indices.b < partition.batchSplit &&
         indices.oc < partition.outChanSplit);
  unsigned outputIndex = indices.cg;
  outputIndex *= partition.batchSplit;
  outputIndex +=indices.b;
  const auto numFieldDims = indices.out.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    assert(indices.out[dim] < partition.fieldSplit[dim]);
    outputIndex *= partition.fieldSplit[dim];
    outputIndex += indices.out[dim];
  }
  outputIndex *= partition.outChanSplit;
  outputIndex += indices.oc;
  return outputIndex;
}

static std::vector<unsigned> getOutputDimSplits(const Partition &partition) {
  std::vector<unsigned> splits = {
    partition.convGroupSplit,
    partition.batchSplit
  };
  splits.insert(splits.end(), partition.fieldSplit.begin(),
                partition.fieldSplit.end());
  splits.push_back(partition.outChanSplit);
  return splits;
}

/// Stich each runs of \a dimSplit partial result tensors together by
/// concatenating them in the specified dimension to form a new
/// list of results that is written back to \a results.
static void stichResultsImpl(std::vector<Tensor> &results, unsigned dim,
                             unsigned dimSplit) {
  if (dimSplit == 1)
    return;
  std::vector<Tensor> stiched;
  assert(results.size() % dimSplit == 0);
  stiched.reserve(results.size() / dimSplit);
  for (auto it = results.begin(), end = results.end(); it != end;
       it += dimSplit) {
    std::vector<Tensor> slice(it, it + dimSplit);
    stiched.push_back(concat(slice, dim));
  }
  std::swap(stiched, results);
}

static Tensor stichResults(std::vector<Tensor> results,
                           std::vector<unsigned> dimSplits) {
  const auto numDims = dimSplits.size();
  for (int dim = numDims - 1; dim >= 0; --dim) {
    stichResultsImpl(results, dim, dimSplits[dim]);
  }
  assert(results.size() == 1);
  return results.front();
}

/// Stich together a number of partial result tensors to form a single tensor.
/// The 1st dimension of \a results represents the dimension to reduce over and
/// the 2nd dimension is a list of results that should be stiched together in
/// the output axes. The list of results is lexigraphically ordered by the
/// indices the partition associated with the output in the order the axes
/// have in the output tensor.
static Tensor
stichPartialResults(
    const std::vector<std::vector<boost::optional<Tensor>>> &results,
    const Partition &partition) {
  std::vector<Tensor> partials;
  partials.reserve(results.size());
  auto dimSplits = getOutputDimSplits(partition);
  for (const auto &entry : results) {
    std::vector<Tensor> r(entry.size());
    for (unsigned i = 0; i < entry.size(); ++i) {
      assert(entry[i]);
      r[i] = *entry[i];
    }
    partials.push_back(stichResults(r, dimSplits));
    partials.back() = partials.back().expand({0});
  }
  return concat(partials, 0);
}

static std::vector<std::size_t>
getPartialOutputShape(const ConvParams &params, const Plan &plan) {
  auto numOutChans = params.getNumOutputChansPerConvGroup();
  const auto outChansPerGroup = plan.partialChansPerGroup;
  std::vector<std::size_t> outShape = {
    params.getNumConvGroups(),
    numOutChans / outChansPerGroup,
    params.getBatchSize()
  };
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outShape.push_back(params.getOutputSize(dim));
  }
  outShape.push_back(outChansPerGroup);
  return outShape;
}

static boost::optional<Tensor>
convolutionImpl(Graph &graph, ConvParams params,
                Plan plan, unsigned level,
                Tensor in, Tensor weights,
                std::vector<Sequence> &copies,
                Tensor &copyWritten,
                ComputeSet convolveCS,
                std::vector<std::vector<ComputeSet>> &reduceComputeSets,
                std::vector<Sequence> &postCopies,
                const std::vector<ConvIndices> &indices,
                Tensor partials, unsigned createPartialsLevel,
                const std::string &debugPrefix,
                const ConvOptions &options) {
  // Transform.
  const auto originalParams = params;
  const auto originalTransform = plan.transforms[level];
  const auto ipuLevel = plan.transforms.size() - 2;
  bool rearrangeActs = false;
  bool rearrangeWeights = false;
  if (level == ipuLevel) {
    // If the input tensors have a different memory layout to the one expected
    // by the vertices poplar will rearrange the data using exchange code or
    // copy pointers. If the data is broadcast this rearrangement happens on
    // every tile that receives the data. We can reduce the amount of exchange
    // code / number of copy pointers required by rearranging the data once and
    // broadcasting the rearranged data. This trades increased execution time
    // for reduced memory usage. The biggest reductions in memory usage come
    // when data is broadcast to many tiles. inViewMaxBroadcastDests and
    // weightViewMaxBroadcastDests specify the maximum number of broadcast
    // destinations a tensor can have before we insert a copy to rearrange it.
    // Note these copies will be elided if the inputs already use the expected
    // memory layout and tile mapping.
    const auto inViewMaxBroadcastDests = 7U;
    const auto weightViewMaxBroadcastDests = 7U;
    const auto inNumDests =
        std::accumulate(plan.partitions.back().kernelSplit.begin(),
                        plan.partitions.back().kernelSplit.end(),
                        1U,
                        std::multiplies<unsigned>()) *
                        plan.partitions.back().outChanSplit;
    auto weightsNumDests = plan.partitions.back().batchSplit;
    for (const auto split : plan.partitions.back().fieldSplit) {
      weightsNumDests *= split;
    }
    rearrangeActs = inputRearrangementIsExpensive(options) ||
                    (inNumDests > inViewMaxBroadcastDests) ||
                    options.useAggressiveRegrouping;
    rearrangeWeights = weightRearrangementIsExpensive(options) ||
                       (weightsNumDests > weightViewMaxBroadcastDests) ||
                       options.useAggressiveRegrouping;
  }
  convolutionPreprocess(graph, params, options, plan, level, indices,
                        in, weights, &copies[level], &copyWritten,
                        rearrangeActs, rearrangeWeights, debugPrefix);
  if (level == createPartialsLevel) {
    auto partialsShape = getPartialOutputShape(params, plan);
    if (level != plan.partitions.size()) {
      const auto numPartials = plan.partitions[level].inChanSplit *
                               product(plan.partitions[level].kernelSplit);
      partialsShape.insert(partialsShape.begin(), numPartials);
    }
    partials = graph.addVariable(plan.types.back().partialType, partialsShape,
                                 debugPrefix + "/partials");
  }
  Tensor out;
  const auto resultType = plan.types[level].resultType;
  if (level == plan.partitions.size()) {
    const auto &target = graph.getTarget();
    const auto tile = linearizeTileIndices(target, options, indices, plan);
    calcPartialConvOutput(graph, plan, tile, params, copies.back(), copyWritten,
                          convolveCS, in, weights, partials,
                          options.use128BitConvUnitLoad, debugPrefix);
    out = partials;

    if (level == createPartialsLevel) {
      out = unsplitActivationChanGroups(out);
    }
    if (level > createPartialsLevel) {
      // The explicit output of the partial convolution is never used.
      return {};
    }
  } else {
    const auto &partition = plan.partitions[level];
    const auto numPartials = partition.inChanSplit *
                             product(partition.kernelSplit);
    const auto outputSplit = partition.convGroupSplit *
                             partition.batchSplit *
                             product(partition.fieldSplit) *
                             partition.outChanSplit;
    std::vector<std::vector<boost::optional<Tensor>>>
        results(numPartials, std::vector<boost::optional<Tensor>>(outputSplit));
    iteratePartition(params, partition, [&](const ConvIndices &levelIndices,
                                            const ConvSlice &slice) {
      // Get sub convolution
      ConvParams subParams = params;
      Tensor subIn = in;
      Tensor subWeights = weights;
      auto subIndices = indices;
      subIndices.push_back(levelIndices);
      getSubConvolution(slice, subParams, &subIn, &subWeights);
      auto partialIndex = getPartialIndex(levelIndices, partition);
      Tensor nextLevelPartials;
      if (level >= createPartialsLevel) {
        nextLevelPartials = partials;
        if (level == createPartialsLevel) {
          nextLevelPartials = nextLevelPartials[partialIndex];
        }
        nextLevelPartials = sliceOutput(nextLevelPartials, slice,
                                        plan.partialChansPerGroup);
      }
      auto subOut = convolutionImpl(graph, subParams, plan, level + 1, subIn,
                                    subWeights, copies, copyWritten, convolveCS,
                                    reduceComputeSets, postCopies, subIndices,
                                    nextLevelPartials, createPartialsLevel,
                                    debugPrefix, options);
      auto outputIndex = getOutputIndex(levelIndices, partition);
      results[partialIndex][outputIndex] = subOut;
    });
    // Stich together results.
    if (level < createPartialsLevel) {
      partials = stichPartialResults(results, partition);
    } else {
      if (level != createPartialsLevel) {
        // The explicit output of the partial convolution is never used.
        return {};
      }
      const auto rank = partials.rank();
      partials =
         partials.dimShufflePartial({2}, {rank - 2})
                 .reshapePartial(rank - 2, rank,
                                 {partials.dim(2) *
                                  partials.dim(rank - 1)});
    }
    // Reduce
    const auto partialType = partials.elementType();
    // Perform the reduction of partial sums.
    if (partials.dim(0) == 1) {
      out = partials.squeeze({0});
    } else {
      const auto partialsRank = partials.rank();
      const auto outChanGrainSize = partition.outChanGrainSize;
      partials = partials.reshapePartial(partialsRank - 1, partialsRank,
                                         {partials.dim(partialsRank - 1) /
                                          outChanGrainSize,
                                          outChanGrainSize})
                         .dimShufflePartial({partialsRank - 1}, {2});
      out = multiStageGroupedReduce(graph, partials, resultType,
                                    reduceComputeSets[level],
                                    debugPrefix);
      out = unsplitActivationChanGroups(out);
    }
  }
  if (out.elementType() != resultType) {
    if (reduceComputeSets[level].empty()) {
      reduceComputeSets[level].push_back(graph.addComputeSet(debugPrefix +
                                                             "/Cast"));
    }
    out = cast(graph, out, resultType, reduceComputeSets[level][0],
               debugPrefix);
  }
  // Inverse transform.
  out = convolutionPostprocess(graph,
                               originalParams,
                               originalTransform,
                               out,
                               postCopies[level],
                               debugPrefix);
  return out;
}

template <typename T>
static std::string
getShapeAsString(const std::vector<T> &shape) {
  return shape.empty() ? std::string ()
    : std::accumulate (std::next(shape.begin()), shape.end (),
                       std::to_string(shape[0]),
                       [] (std::string a, unsigned b) {
                         return a + "x" + std::to_string(b);
                       });
}

static std::string
convSuffix(const ConvParams &params) {
  std::string s = "_";
  s += getShapeAsString(params.kernelShape);
  if (std::any_of(params.outputTransform.stride.begin(),
                  params.outputTransform.stride.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_stride" + getShapeAsString(params.outputTransform.stride);
  }
  if (std::any_of(params.inputTransform.dilation.begin(),
                  params.inputTransform.dilation.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_inDilation" + getShapeAsString(params.inputTransform.dilation);
  }
  return s;
}

static bool requiresReduction(const Partition &partition) {
  if (partition.inChanSplit != 1)
    return true;
  for (const auto &split : partition.kernelSplit) {
    if (split != 1)
      return true;
  }
  return false;
}

// Get the lowest level that we can create the partials tensor
// at.
static unsigned getCreatePartialsLevel(const Plan &plan) {
  const auto numLevels = plan.partitions.size();
  unsigned level = numLevels;
  const auto &partialType = plan.types.back().partialType;
  // TODO: Currently if we create the partials as a large variable
  // with a chan grouping of one it can cause a problem in addToBias
  // detecting the chan grouping. When addToBias is replaced with
  // correct introspection we can remove this check.
  if (plan.partialChansPerGroup == 1)
    return level;
  while (level > 0) {
    const auto &transform = plan.transforms[level];
    // If this level transorms the input in anyway then stop since
    // creating partials earlier may not be the right shape.
    if (transform.swapOperands ||
        !transform.outChanFlattenDims.empty() ||
        !transform.flattenDims.empty() ||
        !transform.expandDims.empty() ||
        !transform.dilatePostConv.empty())
      break;
    // If this level casts the partials to a different type then stop.s
    if (partialType != plan.types[level].resultType)
      break;
    if (level < plan.partitions.size()) {
      // If this level is earlier than the tile level, there may be a post
      // transformation of the partials (a regrouping or reduction) in which
      // we must also stop.
      const auto &partition = plan.partitions[level];
      if (partition.outChanGrainSize != plan.partialChansPerGroup)
        break;
      if (requiresReduction(partition))
        break;
    }
    level--;
  }
  return level;
}

// Rearrange weight deltas if the operands are not swapped. Not swapping
// operands means that the innermost grouping is of output channels whereas
// the weights use the fwd channel grouping (i.e. input channel as innermost
// dimension
static Tensor
rearrangeWeightDeltas(Graph &graph,
                      const Plan &plan,
                      const Tensor &activations_,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  const auto operandSwapped =
      std::accumulate(plan.transforms.begin(), plan.transforms.end(), false,
                      [](bool oS, const ConvTransform &transform) {
                        return oS ^ transform.swapOperands;
                      });
  if (operandSwapped)
    return activations_;

  auto rank = activations_.rank();
  const auto oC = activations_.dim(rank - 1);
  const auto iC = activations_.dim(1);
  const auto dType = activations_.elementType();

  // detect channel grouping
  auto oG = detectInnermostGrouping(graph, activations_);

  if (oG == 1)
    return activations_;

  // Find largest grouping which is an integer sub-multiple of the input
  // channels
  const unsigned maxInputChannelGrouping = 16;
  unsigned iG = maxInputChannelGrouping;
  while (iC % iG != 0)
    --iG;
  if (iG == 1)
    return activations_;

  auto ungroupedActs =
      activations_.reshapePartial(rank - 1, rank, {oC / oG, oG})
                  .reshapePartial(1, 2, {iC / iG, iG})
                  .dimRoll(2, rank);

  auto cs = graph.addComputeSet(debugPrefix + "/DeltasPartialTranspose");
  auto transposedActs = partialTranspose(graph, ungroupedActs, cs);
  prog.add(Execute(cs));

  rank = transposedActs.rank();
  transposedActs =
      transposedActs.dimRoll(rank - 1, 2)
                    .reshapePartial(rank - 2, rank, {oC})
                    .reshapePartial(1, 3, {iC});
  return transposedActs;
}

void preplanConvolutions(
    const std::set<std::tuple<const Target *,
                              const ConvParams,
                              const OptionFlags *>> &convs,
    PlanningCache &cache) {
  std::set<std::pair<ConvParams, ConvOptions>> convsImpl;

  if (convs.empty())
    return;

  for (auto &conv : convs) {
    const auto options = parseConvOptions(*std::get<0>(conv),
                                          *std::get<2>(conv));
    convsImpl.emplace(std::get<1>(conv), options);
  }
  auto &commonTarget = *std::get<0>(*convs.cbegin());
  preplanConvolutionsImpl(commonTarget,  convsImpl, cache);
}

// This function takes a program that calculates a partial convolution
// for a sub-group of out channels and wraps it in a loop to
// calculate the full convolution sequentially.
Tensor
wrapConvInLoop(Graph &graph, Sequence &prog, const ConvParams &params,
               unsigned outChanSerialSplit,
               Sequence &cprog, const Tensor &partialOut,
               const Tensor &fullWeights, const Tensor &weightsIn,
               const std::string &debugPrefix) {
  if (outChanSerialSplit == 1) {
    // In this case there is no loop, the partial convolution is the
    // entire convolution.
    prog.add(cprog);
    return partialOut;
  }

  auto fullOutput = graph.clone(partialOut);
  for (unsigned i = 1; i < outChanSerialSplit; ++i) {
    fullOutput = concat(fullOutput, graph.clone(partialOut), 1);
  }
  auto index = graph.addVariable(UNSIGNED_INT, {}, "convLoopIndex");
  graph.setTileMapping(index, 0);
  Sequence loopBody;
  auto fw =
      fullWeights.reshapePartial(1, 2,
                                 {fullWeights.dim(1) / params.outputChannels,
                                  params.outputChannels});
  auto theseWeights =
      dynamicSlice(graph, fw, index.expand({0}), {1}, {1},
                   loopBody, debugPrefix + "/ConvLoop")
      .reshapePartial(1, 3, {params.outputChannels});
  loopBody.add(Copy(theseWeights, weightsIn));
  loopBody.add(cprog);
  auto fo =
      fullOutput.reshapePartial(1, 2,
                                {params.numConvGroups,
                                 fullOutput.dim(1) /
                                   (params.numConvGroups *
                                      params.outputChannels),
                                 params.outputChannels});
  auto o =
      partialOut.reshapePartial(1, 2,
                                {params.numConvGroups,
                                 partialOut.dim(1) /
                                   (params.numConvGroups *
                                      params.outputChannels),
                                 params.outputChannels});
  dynamicUpdate(graph, fo, o, index.expand({0}), {2}, {1},
                loopBody, debugPrefix + "/ConvLoop");
  auto inc = graph.addConstant(UNSIGNED_INT, {}, 1);
  graph.setTileMapping(inc, 0);
  addInPlace(graph, index, inc, loopBody, debugPrefix + "/ConvLoop");
  auto zero = graph.addConstant(UNSIGNED_INT, {}, 0);
  graph.setTileMapping(zero, 0);
  prog.add(Copy(zero, index));
  prog.add(Repeat(outChanSerialSplit, loopBody));
  return fullOutput;
}

Tensor
convolution(Graph &graph, const poplar::Tensor &in_,
            const poplar::Tensor &weights_,
            const ConvParams &params_,
            bool transposeAndFlipWeights, Sequence &prog,
            const std::string &debugPrefix,
            const poplar::OptionFlags &options_,
            PlanningCache *cache) {
  Sequence cprog;
  const auto options = parseConvOptions(graph.getTarget(), options_);
  auto params = canonicalizeParams(params_);
  auto plan = getPlan(graph.getTarget(), params, options, cache);
  auto weights = weights_;
  if (weights.rank() == params_.getNumFieldDims() + 2) {
    weights = weights.expand({0});
  }
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    auto bwdWeights = createWeights(graph, params, "bwdWeights",
                                    options, cache);
    if (bwdWeights.dim(1) && bwdWeights.dim(2)) {
      weightsTransposeChansFlipXY(
            graph, weights, bwdWeights, prog, debugPrefix);
    }
    weights = bwdWeights;
  }
  auto fullWeights = weights;
  // If the out channels are split into a sequentially calculation then
  // create the convolution code for one "chunk" and wrap it in a loop later.
  auto outChanSerialSplit = plan.outChanSerialSplit;
  if (outChanSerialSplit > 1) {
    assert(params.outputChannels % outChanSerialSplit == 0);
    params.outputChannels /= outChanSerialSplit;
    // Create a weights placeholder to copy the correct weights into at
    // each step of the loop.
    weights = graph.clone(weights.slice(0, params.outputChannels, 1));
    plan.outChanSerialSplit = 1;
  }
  auto weightsIn = weights;
  weights = weightsToInternalShape(weights);
  auto in = actsToInternalShape(in_, params.numConvGroups,
                                params.inputChannels);
  verifyInputShapes(params, in, weights);
  if (plan.useWinograd) {
    throw poputil::poplibs_error("Winograd not yet supported");
  }
  const auto layerName = debugPrefix + "/Conv" + convSuffix(params);
  const auto numLevels = plan.partitions.size() + 1;
  const auto createPartialsLevel = getCreatePartialsLevel(plan);
  std::vector<Sequence> copies(numLevels), postCopies(numLevels);
  Tensor copyWritten = graph.addVariable(params.dType, {0});
  std::vector<std::vector<ComputeSet>> reduceComputeSets(numLevels);
  auto convolveCS = graph.addComputeSet(layerName + "/Convolve");

  auto activations =
     *convolutionImpl(graph, params, plan, 0, in, weights, copies,
                      copyWritten, convolveCS, reduceComputeSets,
                      postCopies, std::vector<ConvIndices>(), Tensor(),
                      createPartialsLevel, layerName, options);
  cprog.add(WriteUndef(copyWritten));
  for (const auto &p : copies) {
    cprog.add(p);
  }
  cprog.add(Execute(convolveCS));
  for (int level = numLevels - 1; level >= 0; --level) {
    for (const auto &reduceCS : reduceComputeSets[level]) {
      cprog.add(Execute(reduceCS));
    }
    cprog.add(postCopies[level]);
  }

  // Do any rearrangements of the output in the WU pass
  if (options.pass == Pass::TRAINING_WU) {
    activations =
        rearrangeWeightDeltas(graph, plan, activations, cprog, debugPrefix);
  }

  assert(activations.elementType() == in.elementType());
  auto out = actsToExternalShape(activations);

  return wrapConvInLoop(graph, prog, params, outChanSerialSplit, cprog, out,
                        fullWeights, weightsIn, debugPrefix);
}

static uint64_t getFlops(const ConvParams &params) {
  return (2 * getNumberOfMACs(params));
}

uint64_t getFwdFlops(const ConvParams &params) {
  return getFlops(params);
}

uint64_t getBwdFlops(const ConvParams &params) {
  return getFlops(params);
}

uint64_t getWuFlops(const ConvParams &params) {
  return getFlops(params);
}

static double getPerfectCycleCount(const Graph &graph,
                                   const ConvParams &params) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  auto numMacs = getNumberOfMACs(params);
  if (params.dType == FLOAT) {
    const auto floatVectorWidth = target.getFloatVectorWidth();
    auto macCycles =
        static_cast<double>(numMacs) / (floatVectorWidth * numTiles);
    return macCycles;
  }
  assert(params.dType == HALF);
  const auto convUnitsPerTile =
      std::max(std::max(target.getFp16InFp16OutConvUnitsPerTile(),
                        target.getFp32InFp32OutConvUnitsPerTile()),
               target.getFp16InFp32OutConvUnitsPerTile());
  const auto halfVectorWidth = target.getHalfVectorWidth();
  auto macsPerCycle = convUnitsPerTile * halfVectorWidth;
  auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
  return macCycles;
}

double getFwdPerfectCycleCount(const Graph &graph,
                               const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

double getWuPerfectCycleCount(const Graph &graph, const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

/** Copy the weights in 'weightsIn' into 'weightsOut' such that
 *  each element of the kernel is transposed w.r.t. the input and output
 *  channels and flip both the X and Y axis of the kernel field.
 */
void weightsTransposeChansFlipXY(Graph &graph,
                                 const Tensor &weightsInUnGrouped,
                                 const Tensor &weightsOutUnGrouped,
                                 Sequence &prog,
                                 const std::string &debugPrefix) {
  assert(weightsInUnGrouped.rank() >= 3);
  const auto numFieldDims = weightsInUnGrouped.rank() - 3;
  const auto weightsIn =
      groupWeights(graph, weightsToInternalShape(weightsInUnGrouped));
  const auto weightsOut =
      groupWeights(graph, weightsToInternalShape(weightsOutUnGrouped));
  // weightsIn = { O/G1, I/G2, ..., G1, G2 }
  // weightsOut = { I/G3, O/G4, ..., G3, G4 }
  const auto dType = weightsIn.elementType();
  const auto GC = weightsOut.dim(0);
  const auto G1 = weightsIn.dim(weightsIn.rank() - 2);
  const auto G2 = weightsIn.dim(weightsIn.rank() - 1);
  const auto G3 = weightsOut.dim(weightsOut.rank() - 2);
  const auto G4 = weightsOut.dim(weightsOut.rank() - 1);
  const auto I = weightsOut.dim(1) * G3;
  const auto O = weightsOut.dim(2) * G4;

  // Express the rearrangement as a composition of two rearrangements such
  // that the first rearrangement avoids exchange and maximises the size of the
  // block that is rearranged in the second step. This reduces exchange code
  // since the second step involves fewer, larger messages.
  // G5 is the size of the innermost dimension after the partial transposition.
  // To avoid exchange it must divide G1. If G4 divides G1 then set G5 to G4 -
  // this results in the block size of G1 * gcd(G2, G3) elements in the
  // second step. Otherwise set G5 to G1 for a block size of gcd(G1, G4)
  // elements.
  const auto G5 = (G1 % G4 == 0) ? G4 : G1;
  Tensor partiallyTransposed;
  if (G5 == 1) {
    partiallyTransposed =
        weightsIn.reshapePartial(0, 3, {GC, O/G1, I/G2})
                 .reshapePartial(weightsIn.rank() - 2, weightsIn.rank(),
                                 {G1, G2, 1});
  } else {
    auto cs = graph.addComputeSet(debugPrefix + "/WeightTranspose");
    partiallyTransposed =
        partialTranspose(
          graph,
          weightsIn.reshapePartial(0, 3, {GC, O/G1, I/G2})
                   .reshapePartial(weightsIn.rank() - 2, weightsIn.rank(),
                                   {G1/G5, G5, G2}),
          cs
        );
    prog.add(Execute(cs));
  }

  auto flipped = partiallyTransposed;
  std::vector<Tensor> flippedSlices;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto kernelSize = partiallyTransposed.dim(3 + dim);
    for (int w = kernelSize - 1; w >= 0; --w) {
      flippedSlices.push_back(flipped.slice(w, w + 1, 3 + dim));
    }
    flipped = concat(flippedSlices, 3 + dim);
    flippedSlices.clear();
  }
  prog.add(Copy(flipped.dimShufflePartial({1,
                                           3 + numFieldDims,
                                           3 + numFieldDims + 2,
                                           2,
                                           3 + numFieldDims + 1},
                                          {1 + numFieldDims,
                                           1 + numFieldDims + 1,
                                           1 + numFieldDims + 2,
                                           1 + numFieldDims + 3,
                                           1 + numFieldDims + 4})
                       .reshapePartial(flipped.rank() - 5, flipped.rank(),
                                       {O/G4, G4, I/G3, G3})
                       .dimShufflePartial({1 + numFieldDims + 2,
                                           1 + numFieldDims,
                                           1 + numFieldDims + 3,
                                           1 + numFieldDims + 1},
                                          {1,
                                           2,
                                           3 + numFieldDims,
                                           3 + numFieldDims + 1}),
                weightsOut));
}

ConvParams
getWeightUpdateParams(ConvParams fwdParams) {
  fwdParams = canonicalizeParams(fwdParams);
  const auto numFieldDims = fwdParams.getNumFieldDims();
  auto wuFlipInput = fwdParams.inputTransform.flip;
  std::vector<bool> wuFlipKernel(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (fwdParams.kernelTransform.flip[dim]) {
      // If the kernel is flipped in the forward pass we must flip the output
      // in the weight update pass. This is equivalent to flipping both the
      // activations and the deltas in the weight update pass.
      wuFlipInput[dim] = !wuFlipInput[dim];
      wuFlipKernel[dim] = !wuFlipKernel[dim];
    }
  }
  ConvParams wuParams(
    fwdParams.dType,
    fwdParams.getNumInputChansPerConvGroup(), // batchSize
    fwdParams.inputFieldShape, // inputFieldShape
    fwdParams.getOutputFieldShape(), // kernelShape
    fwdParams.getBatchSize(), // inputChannels
    fwdParams.getNumOutputChansPerConvGroup(), // outputChannels
    fwdParams.numConvGroups, // numConvGroups
    fwdParams.inputTransform.truncationLower, // inputTruncationLower
    fwdParams.inputTransform.truncationUpper, // inputTruncationUpper
    fwdParams.inputTransform.dilation, // inputDilation
    fwdParams.inputTransform.paddingLower, // inputPaddingLower
    fwdParams.inputTransform.paddingUpper, // inputPaddingUpper
    wuFlipInput, // flipInput
    fwdParams.outputTransform.paddingLower, // kernelTruncationLower
    fwdParams.outputTransform.paddingUpper, // kernelTruncationUpper
    fwdParams.outputTransform.stride, // kernelDilation
    fwdParams.outputTransform.truncationLower, // kernelPaddingLower
    fwdParams.outputTransform.truncationUpper, // kernelPaddingUpper
    wuFlipKernel, // flipKernel
    fwdParams.kernelTransform.paddingLower, // outputTruncationLower
    fwdParams.kernelTransform.paddingUpper, // outputTruncationIpper
    fwdParams.kernelTransform.dilation, // stride
    fwdParams.kernelTransform.truncationLower, // outputPaddingLower
    fwdParams.kernelTransform.truncationUpper // outputPaddingUpper
  );
  return canonicalizeParams(wuParams);
}

Tensor
calculateWeightDeltas(Graph &graph, const Tensor &zDeltas_,
                      const Tensor &activations_,
                      const ConvParams &fwdParams,
                      Sequence &prog,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &fwdOptions,
                      PlanningCache *cache) {
  const auto numConvGroups = fwdParams.numConvGroups;
  auto zDeltas = actsToInternalShape(zDeltas_, numConvGroups,
                                     fwdParams.outputChannels);
  auto activations = actsToInternalShape(activations_, numConvGroups,
                                         fwdParams.inputChannels);
  auto params = getWeightUpdateParams(fwdParams);
  auto options = fwdOptions;
  options.set("pass", "TRAINING_WU");
  // The weight update is equivalent to a convolution where:
  // - wu conv groups = fwd conv groups
  // - wu batch size = fwd input channels
  // - wu input channels = fwd batch size
  // - wu height = fwd height
  // - wu width = fwd width
  // - wu output channels = fwd output channels
  auto activationsRearranged =
      activations.dimShufflePartial({1, activations.rank() - 1},
                                    {activations.rank() - 1, 1});
  auto deltasRearranged = zDeltas.dimShufflePartial({zDeltas.rank() - 1}, {1});
  auto weightDeltas =
      convolution(graph,
                  actsToExternalShape(activationsRearranged),
                  deltasRearranged,
                  params,
                  false,
                  prog,
                  debugPrefix,
                  options,
                  cache);
  weightDeltas = actsToInternalShape(weightDeltas, numConvGroups,
                                     params.outputChannels);
  return weightsToExternalShape(
           weightDeltas.dimShufflePartial({1}, {weightDeltas.rank() - 1})
         );
}

void
convolutionWeightUpdate(Graph &graph,
                        const Tensor &zDeltas, const Tensor &weights,
                        const Tensor &activations,
                        const ConvParams &params,
                        const Tensor &scale,
                        Sequence &prog,
                        const std::string &debugPrefix,
                        const poplar::OptionFlags &options,
                        PlanningCache *cache) {
  auto weightDeltas = calculateWeightDeltas(graph, zDeltas, activations, params,
                                            prog, debugPrefix, options, cache);
  // Add the weight deltas to the weights.
  assert(weightDeltas.shape() == weights.shape());
  scaledAddTo(graph, weights, weightDeltas, scale, prog,
              debugPrefix + "/UpdateWeights");
}

void
convolutionWeightUpdate(Graph &graph,
                        const Tensor &zDeltas, const Tensor &weights,
                        const Tensor &activations,
                        const ConvParams &params,
                        float scale,
                        Sequence &prog,
                        const std::string &debugPrefix,
                        const poplar::OptionFlags &options,
                        PlanningCache *cache) {
  auto weightDeltas = calculateWeightDeltas(graph, zDeltas, activations, params,
                                            prog, debugPrefix, options, cache);
  // Add the weight deltas to the weights.
  assert(weightDeltas.shape() == weights.shape());
  scaledAddTo(graph, weights, weightDeltas, scale, prog,
              debugPrefix + "/UpdateWeights");
}

// Return a program to update the biases tensor with the gradients derived
// from the zDeltas tensor
void
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                      const Tensor &biases,
                      const Tensor &scale,
                      const Type &partialsType,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  if (zDeltasUngrouped.rank() < 2)
    throw poplibs_error("convolutionBiasUpdate with rank " +
                       std::to_string(zDeltasUngrouped.rank()) +
                       "; must have at least channel and batch dimensions");

  std::vector<std::size_t> reduceDims(zDeltasUngrouped.rank()-1);
  std::iota(reduceDims.begin()+1, reduceDims.end(), 2);

  popops::reduceWithOutput(graph, zDeltasUngrouped, biases, reduceDims, {
            popops::Operation::ADD, true, scale
          }, prog, debugPrefix + "/BiasUpdate");
}
void
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                      const Tensor &biases,
                      float scale,
                      const Type &partialsType,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  auto scaleTensor =
      graph.addConstant(FLOAT, {}, scale, debugPrefix + "/scaleTensor");
  graph.setTileMapping(scaleTensor, 0);
  convolutionBiasUpdate(graph, zDeltasUngrouped, biases, scaleTensor,
                            partialsType, prog, debugPrefix + "/ConstLearning");
}

void
addBias(Graph &graph, const Tensor &acts, const Tensor &biases,
        Sequence &prog, const std::string &debugPrefix) {
  // TODO: Change this to an addInPlace and see what happens
  addToChannel(graph, acts, biases, 1.0, prog, debugPrefix);
}

static Plan
getFullyConnectedFwdPlanFromBwdParams(const Target &target,
                                      const ConvParams &bwdParams,
                                      const ConvOptions &bwdOptions,
                                      PlanningCache *cache) {
  assert(bwdOptions.pass == Pass::FC_TRAINING_BWD);
  auto fwdParams = bwdParams;
  std::swap(fwdParams.inputFieldShape[0], fwdParams.inputChannels);
  if (fwdParams.inputFieldShape[0] == 0) {
    // Transformed input must be greater than or equal to the transformed kernel
    // size.
    fwdParams.inputTransform.paddingUpper[0] = 1;
    fwdParams.outputTransform.truncationUpper[0] = 1;
  }
  auto fwdOptions = bwdOptions;
  fwdOptions.pass = Pass::FC_TRAINING_FWD;
  return getPlan(target, fwdParams, fwdOptions, cache);
}

static bool planSwapsOperands(const Plan &plan) {
  for (const auto &entry : plan.transforms) {
    if (entry.swapOperands)
      return true;
  }
  return false;
}

Tensor
fullyConnectedWeightTranspose(Graph &graph,
                              Tensor activations,
                              ConvParams params,
                              Sequence &prog, const std::string &debugPrefix,
                              const poplar::OptionFlags &options_,
                              PlanningCache *cache) {
  const auto options = parseConvOptions(graph.getTarget(), options_);
  if (params.getNumFieldDims() != 1) {
    throw poputil::poplibs_error("fullyConnectedWeightTranspose() expects a 1-d"
                                 " convolution");
  }
  auto bwdPlan = getPlan(graph.getTarget(), params, options, cache);
  auto fwdPlan = getFullyConnectedFwdPlanFromBwdParams(graph.getTarget(),
                                                       params, options,
                                                       cache);
  auto splitActivations =
      actsToInternalShape(activations, params.getNumConvGroups(),
                          params.inputFieldShape.back());
  const auto fwdGroupSize =
      getInChansPerGroup(fwdPlan,
                         static_cast<unsigned>(splitActivations.dim(3)));
  const auto bwdGroupSize =
      getInChansPerGroup(bwdPlan,
                         static_cast<unsigned>(splitActivations.dim(2)));
  if (fwdGroupSize == 1 || bwdGroupSize == 1 ||
      planSwapsOperands(fwdPlan) || planSwapsOperands(bwdPlan)) {
    // In this case there is no benefit to using transpose vertices to
    // rearrange.
    return actsToExternalShape(splitActivations.dimShuffle({0, 1, 3, 2}));
  }
  Tensor transposed = createInput(graph, params, "transposed", options, cache);
  auto splitTransposed =
      actsToInternalShape(transposed, params.getNumConvGroups(),
                          params.inputChannels);
  auto splitTransposedUngroupedShape = splitTransposed.shape();
  const auto dType = activations.elementType();
  splitActivations =
      splitActivations.reshape({splitActivations.dim(0),
                                splitActivations.dim(1),
                                splitActivations.dim(2) / bwdGroupSize,
                                bwdGroupSize,
                                splitActivations.dim(3) / fwdGroupSize,
                                fwdGroupSize})
                      .dimShufflePartial({3}, {4});
  splitTransposed =
      splitTransposed.reshape({splitTransposed.dim(0),
                               splitTransposed.dim(1),
                               splitTransposed.dim(2) / fwdGroupSize,
                               fwdGroupSize,
                               splitTransposed.dim(3) / bwdGroupSize,
                               bwdGroupSize})
                      .dimShufflePartial({3}, {4});
  auto firstInBlock =
      splitActivations.slice({0, 0, 0, 0, 0, 0},
                        {splitActivations.dim(0),
                         splitActivations.dim(1),
                         splitActivations.dim(2),
                         splitActivations.dim(3),
                         1,
                         1})
                      .squeeze({4, 5});
  auto blockTileMapping = graph.getTileMapping(firstInBlock);
  auto transposeCS = graph.addComputeSet(debugPrefix + "/Transpose");

  addTransposeVertices(graph, transposeCS,
                       dType, bwdGroupSize, fwdGroupSize,
                       blockTileMapping,
                       [&](size_t index) {
                         auto blockIndices =
                           poputil::unflattenIndex(firstInBlock.shape(), index);
                         return std::make_pair(
                                   splitActivations[blockIndices[0]]
                                           [blockIndices[1]]
                                           [blockIndices[2]]
                                           [blockIndices[3]].flatten(),
                                   splitTransposed[blockIndices[0]]
                                           [blockIndices[1]]
                                           [blockIndices[3]]
                                           [blockIndices[2]].flatten());
                       });
  prog.add(Execute(transposeCS));
  auto transposedWeights =
      splitTransposed.dimShufflePartial({3}, {4})
                     .reshape(splitTransposedUngroupedShape);
  return actsToExternalShape(transposedWeights);
}

void reportPlanInfo(std::ostream &out,
                    const poplar::Graph &graph,
                    const ConvParams &params,
                    const poplar::OptionFlags &options_,
                    PlanningCache *cache) {
  const auto options = parseConvOptions(graph.getTarget(), options_);
  auto plan = getPlan(graph.getTarget(), params, options, cache);
  if (options.pass != Pass::FC_TRAINING_WU &&
      options.pass != Pass::FC_TRAINING_BWD) {
    uint64_t cycles, memory;
    std::tie(cycles, memory) = estimateConvCost(graph.getTarget(),
                                                params,
                                                options,
                                                cache,
                                                plan);
    out << "  Estimated cost {cycles " << cycles
        << ", temporary bytes " << memory << "}\n";
  }
  out << plan;
}

void reportWeightUpdatePlanInfo(std::ostream &out,
                                const Graph &graph,
                                const ConvParams &fwdParams,
                                const poplar::OptionFlags &fwdOptions,
                                PlanningCache *cache) {
  auto params = getWeightUpdateParams(fwdParams);
  auto options = fwdOptions;
  options.set("pass", "TRAINING_WU");
  // The weight update is equivalent to a convolution where:
  // - wu conv groups = fwd conv groups
  // - wu batch size = fwd input channels
  // - wu input channels = fwd batch size
  // - wu height = fwd height
  // - wu width = fwd width
  // - wu output channels = fwd output channels
  reportPlanInfo(out, graph, params, options, cache);
}

} // namespace poplin

namespace std {

std::size_t
hash<poplin::ConvParams::InputTransform>::operator()(
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

std::size_t
hash<poplin::ConvParams::OutputTransform>::operator()(
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
  // TODO: specialise std::hash for poplar::Type
  boost::hash_combine(seed, std::string(p.dType.toString()));
  boost::hash_combine(seed, p.batchSize);
  boost::hash_range(seed, std::begin(p.inputFieldShape),
                    std::end(p.inputFieldShape));
  boost::hash_range(seed, std::begin(p.kernelShape), std::end(p.kernelShape));
  boost::hash_combine(seed, p.inputChannels);
  boost::hash_combine(seed, p.outputChannels);
  boost::hash_combine(seed, p.numConvGroups);
  boost::hash_combine(seed, p.inputTransform);
  boost::hash_combine(seed, p.kernelTransform);
  boost::hash_combine(seed, p.outputTransform);
  return seed;
}
} // namespace std
