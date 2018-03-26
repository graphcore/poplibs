#include "popconv/Convolution.hpp"
#include "popconv/internal/ConvPlan.hpp"
#include <limits>
#include <algorithm>
#include <boost/optional.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include "popconv/ConvUtil.hpp"
#include "popops/Pad.hpp"
#include "popops/Add.hpp"
#include "poputil/TileMapping.hpp"
#include "popops/Reduce.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexOptim.hpp"
#include "poputil/exceptions.hpp"
#include "popops/Cast.hpp"
#include "poputil/Util.hpp"
#include "Winograd.hpp"
#include "popops/Zero.hpp"
#include "popops/ElementWise.hpp"
#include <unordered_set>
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/print.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include <boost/icl/interval_map.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace popconv {

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
    throw poputil::poplib_error("Number of kernel field dimensions does not"
                               "match the number of input field dimensions");
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
      throw poputil::poplib_error(std::string("Number of ") + entry.second +
                                 " dimensions does not match the number of "
                                 "field dimensions");
    }
  }
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (inputTransform.truncationLower[dim] +
        inputTransform.truncationUpper[dim] >
        inputFieldShape[dim]) {
      throw poputil::poplib_error("Truncation for dimension " +
                                 std::to_string(dim) +
                                 " truncates by more than the size of the "
                                 "field");
    }
    if (kernelTransform.truncationLower[dim] +
        kernelTransform.truncationUpper[dim] >
        kernelShape[dim]) {
      throw poputil::poplib_error("Truncation for dimension " +
                                 std::to_string(dim) +
                                 " truncates by more than the size of the "
                                 "kernel");
    }
    const auto transformedInputSize = getTransformedInputSize(dim);
    const auto transformedKernelSize = getTransformedKernelSize(dim);
    if (transformedKernelSize == 0) {
      throw poputil::poplib_error("Transformed kernel for dimension " +
                                  std::to_string(dim) +
                                  " has zero size");
    }

    if (transformedInputSize < transformedKernelSize) {
      throw poputil::poplib_error("Transformed input size for dimension " +
                                  std::to_string(dim) +
                                  " is less than the transformed kernel size");
    }
    const auto convOutSize = getUntransformedOutputSize(dim);
    if (outputTransform.truncationLower[dim] +
        outputTransform.truncationUpper[dim] >
        convOutSize) {
      throw poputil::poplib_error("Output truncation for dimension " +
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

// Reshape the activations tensor from [N][G * C]... shape to
// [G][N]...[C].
static Tensor
actsToInternalShape(const Tensor &act, unsigned numConvGroups) {
  if (act.dim(1) % numConvGroups != 0) {
    throw poputil::poplib_error("Number of input channels is not a multiple "
                               "of the number of convolutional groups");
  }
  return act.reshapePartial(1, 2, {numConvGroups, act.dim(1) / numConvGroups})
            .dimShufflePartial({1, 2}, {0, act.rank()});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [N][G * C]... shape.
static Tensor
actsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({0, act.rank() - 1}, {1, 2})
            .reshapePartial(1, 3, {act.dim(0) * act.dim(act.rank() - 1)});
}

// Reshape the weights tensor from [G][OC][IC]... shape to
// [G]...[OC][IC].
static Tensor
weightsToInternalShape(const Tensor &act) {
  return act.dimShufflePartial({1, 2}, {act.rank() - 2, act.rank() - 1});
}

// Reshape the weights tensor from [G]...[OC][IC] shape to
// [G][OC][IC]... shape.
static Tensor
weightsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({act.rank() - 2, act.rank() - 1}, {1, 2});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [G][C1][N]...[C2]
//
// Where C1 * C2 = C
static Tensor
splitActivationChanGroups(const Tensor &act, unsigned chansPerGroup) {
  const auto rank = act.rank();
  assert(act.dim(rank - 1) % chansPerGroup == 0);
  return act.reshapePartial(rank - 1, rank,
                            {act.dim(rank - 1) / chansPerGroup, chansPerGroup})
            .dimShufflePartial({rank - 1}, {1});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [G][C1][N]...[C2]
//
// Where C1 * C2 = C
static Tensor
splitActivationChanGroups(const Tensor &act) {
  auto chansPerGroup = detectChannelGrouping(act);
  return splitActivationChanGroups(act, chansPerGroup);
}

// Reshape the activations tensor from [G][C1][N]...[C2] shape to
// [G][N]...[C]
//
// Where C1 * C2 = C
static Tensor
unsplitActivationChanGroups(const Tensor &act) {
  const auto rank = act.rank();
  return act.dimShufflePartial({1}, {rank - 2})
            .reshapePartial(rank - 2, rank, {act.dim(1) * act.dim(rank - 1)});
}

static std::pair<unsigned, unsigned>
detectWeightsChannelGrouping(const Tensor &w) {
  auto inChansPerGroup = detectChannelGrouping(w);
  const auto rank = w.rank();
  const auto w1 =
      w.reshapePartial(rank - 1, rank, {w.dim(rank - 1) / inChansPerGroup,
                                        inChansPerGroup})
       .dimRoll(rank - 1, 0);
  auto outChansPerGroup = detectChannelGrouping(w1);
  assert(outChansPerGroup % inChansPerGroup == 0);
  outChansPerGroup /= inChansPerGroup;
  return {outChansPerGroup, inChansPerGroup};
}

// Groups tensor from standard convolution weight tensor shape [G]...[OC][IC]
// to internal shape [G][OC1][IC1]...[OC2][IC2]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
static Tensor groupWeights(const Tensor &weights, unsigned inChansPerGroup,
                           unsigned outChansPerGroup) {
  const auto rank = weights.rank();
  assert(weights.dim(rank - 1) % inChansPerGroup == 0);
  assert(weights.dim(rank - 2) % outChansPerGroup == 0);
  const unsigned inChanGroups = weights.dim(rank - 1) / inChansPerGroup;
  const unsigned outChanGroups = weights.dim(rank - 2) / outChansPerGroup;

  return weights.reshapePartial(rank - 2, rank,
                                {outChanGroups, outChansPerGroup,
                                 inChanGroups, inChansPerGroup})
                .dimShufflePartial({rank - 2, rank}, {1, 2});
}


static Tensor groupWeights(const Tensor &weights) {
  unsigned inChansPerGroup, outChansPerGroup;
  std::tie(outChansPerGroup, inChansPerGroup) =
      detectWeightsChannelGrouping(weights);
  return groupWeights(weights, inChansPerGroup, outChansPerGroup);
}

// Ungroups tensors from internal shape [G][OC1][IC1]...[OC2][IC2] to
// standard convolution weight tensor shape [G]...[OC][IC]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
static Tensor ungroupWeights(const Tensor &weights) {
  const auto rank = weights.rank();
  return weights.dimShufflePartial({1, 2}, {rank - 4, rank - 2})
                .reshapePartial(rank - 4, rank,
                                {weights.dim(1) * weights.dim(rank - 2),
                                 weights.dim(2) * weights.dim(rank - 1)});
}

static unsigned
getTransformedSize(const std::vector<std::size_t> &size,
                   const ConvParams::InputTransform &transform,
                   unsigned dim) {
  assert(size[dim] >= transform.truncationLower[dim] +
         transform.truncationUpper[dim]);
  const auto truncatedSize =
      size[dim] - (transform.truncationLower[dim] +
                   transform.truncationUpper[dim]);
  const auto truncatedDilatedSize =
      getDilatedSize(truncatedSize, transform.dilation[dim]);
  int truncatedDilatedPaddedSize =
      transform.paddingLower[dim] + truncatedDilatedSize +
      transform.paddingUpper[dim];
  return truncatedDilatedPaddedSize;
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
  return transformedInputSize - transformedKernelSize + 1;
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

static void
applyTensorMapping(
    Graph &graph,
    const Tensor &t,
    const std::vector<
      std::vector<Interval>
    > &mapping) {
  auto flattened = t.flatten();
  const auto numTiles = mapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &region : mapping[tile]) {
      graph.setTileMapping(flattened.slice(region.begin(), region.end()),
                           tile);
    }
  }
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
    throw poputil::poplib_error("Input tensor does not have the expected rank");
  }
  if (weights.rank() != 3 + numFieldDims) {
    throw poputil::poplib_error("Weight tensor does not have the expected "
                                "rank");
  }
  for (unsigned i = 0; i != numFieldDims; ++i) {
    if (params.inputFieldShape[i] != in.dim(2 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw poputil::poplib_error(dimName + " of input tensor does not match "
                                  "convolution parameters");
    }
    if (params.kernelShape[i] != weights.dim(1 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw poputil::poplib_error(dimName + " of kernel does not match "
                                  "convolution parameters");
    }
  }
  if (in.dim(0) == 0) {
    throw poputil::poplib_error("Number of convolution groups equal to zero "
                                "is not supported");
  }
  if (params.numConvGroups != in.dim(0)) {
    throw poputil::poplib_error("Number of convolution groups of input tensor "
                                "does not match convolution parameters");
  }
  if (params.getBatchSize() != in.dim(1)) {
    throw poputil::poplib_error("Batchsize of input tensor does not match "
                                "convolution parameters");
  }
  if (in.dim(1) == 0) {
    throw poputil::poplib_error("Batch size of input tensor equal to zero "
                                 "is not supported");
  }
  if (params.getNumInputChansPerConvGroup() != in.dim(in.rank() - 1)) {
    throw poputil::poplib_error("Number of channels per convolution group of "
                                "input tensor does not match convolution "
                                "parameters");
  }
  if (params.numConvGroups != weights.dim(0)) {
    throw poputil::poplib_error("Number of convolution groups of weights "
                                "tensor does not match convolution parameters");
  }
  if (params.getNumOutputChansPerConvGroup() !=
      weights.dim(weights.rank() - 2)) {
    throw poputil::poplib_error("Kernel output channel size does not match "
                                "convolution parameters");
  }
  if (params.getNumInputChansPerConvGroup() !=
      weights.dim(weights.rank() - 1)) {
    throw poputil::poplib_error("Kernel input channel size does not match "
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
getOutChansPerGroup(const Plan &plan, unsigned numOutChans) {
  return gcd(plan.partialChansPerGroup, numOutChans);
}

static unsigned
linearizeConvIndices(const std::vector<unsigned> &outIndices,
                     const std::vector<unsigned> &kernelIndices,
                     unsigned ic, unsigned b, unsigned oc, unsigned cg,
                     const std::vector<unsigned> &fieldSplit,
                     const std::vector<unsigned> &kernelSplit,
                     unsigned inChanSplit, unsigned batchSplit,
                     unsigned outChanSplit, unsigned convGroupSplit) {
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
                     const std::vector<ConvIndices> &indices,
                     const Plan &plan) {
  const auto hierarchy = getTileHierarchy(target);
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
    const auto &convGroupSplit = plan.partitions[i].convGroupSplit;
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
                             fwdBatchSplit, fwdOutChanSplit,
                             convGroupSplit);
    tile = tile * hierarchy[i] + linearizedIndex;
  }
  assert(tile < target.getNumTiles());
  return tile;
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

/// Extend a partial map to a total map in the range [lower, upper). The value
/// of keys not in the partial map are based on the value of the neighbouring
/// keys that are in the map. The partial map must contain at least one entry.
template <class K, class V> static void
extendPartialMap(boost::icl::interval_map<K, V> &map,
                 K lower, K upper) {
  assert(iterative_size(map) >= 0);
  boost::icl::interval_map<K, V> extendedMap;
  for (auto begin = map.begin(), it = begin, end = map.end(); it != end;
       ++it) {
    const auto &interval = it->first;
    auto next = std::next(it);
    auto extendedIntervalLower = it == begin ? lower : interval.lower();
    auto extendedIntervalUpper = next == end ? upper : next->first.lower();
    auto extendedInterval =
        boost::icl::interval<unsigned>::right_open(extendedIntervalLower,
                                                   extendedIntervalUpper);
    extendedMap.insert({extendedInterval, std::move(it->second)});
  }
  std::swap(map, extendedMap);
}

static bool
isHaloRegion(
    const std::set<unsigned> &prevTiles,
    const std::set<unsigned> &tiles,
    const std::set<unsigned> &nextTiles) {
  if (prevTiles.size() + nextTiles.size() != tiles.size())
    return false;
  return std::includes(tiles.begin(), tiles.end(),
                       prevTiles.begin(), prevTiles.end()) &&
         std::includes(tiles.begin(), tiles.end(),
                       nextTiles.begin(), nextTiles.end());
}

static void
optimizeHaloMapping(boost::icl::interval_map<
                      unsigned, std::set<unsigned>
                    > &map) {
  // Modify the map so that "halo" regions where the uses are the union of the
  // uses of the neighbouring regions are mapped as if they were only used by
  // one of the sets of tiles. This heuristic reduces exchange code for
  // convolutional layers since the halos tend to be small and mapping them
  // independently splits up the tensor tile mapping, increasing the amount of
  // exchange code required.
  boost::icl::interval_map<unsigned, std::set<unsigned>> optimizedMap;
  for (auto begin = map.begin(), it = begin, end = map.end(); it != end;
       ++it) {
    if (it != begin && std::next(it) != end &&
        isHaloRegion(std::prev(it)->second,
                     it->second,
                     std::next(it)->second)) {
      optimizedMap.insert({it->first, it == begin ? std::next(it)->second :
                                                    std::prev(it)->second});
    } else {
      optimizedMap.insert(*it);
    }
  }
  std::swap(map, optimizedMap);
}

static std::vector<std::vector<Interval>>
calculateMappingBasedOnUsage(const Graph &graph,
                             const std::vector<std::size_t> &shape,
                             const boost::icl::interval_map<
                               unsigned, std::set<unsigned>
                             > &uses,
                             unsigned grainSize,
                             unsigned minElementsPerTile) {
  if (iterative_size(uses) == 0) {
    return poputil::calcLinearTileMapping(graph, shape,
                                         minElementsPerTile, grainSize);
  }
  boost::icl::interval_map<unsigned, std::set<unsigned>> grainToTiles;
  for (const auto &entry : uses) {
    const auto &interval = entry.first;
    unsigned grainLower = interval.lower() / grainSize;
    unsigned grainUpper = (interval.upper() - 1) / grainSize + 1;
    auto grainInterval =
        boost::icl::interval<unsigned>::right_open(grainLower, grainUpper);
    grainToTiles.insert({grainInterval, entry.second});
  }

  // Extend the grainUses map to total map.
  const auto numElements = std::accumulate(shape.begin(), shape.end(), 1UL,
                                           std::multiplies<std::size_t>());
  const unsigned numGrains = (numElements + grainSize - 1) / grainSize;
  extendPartialMap(grainToTiles, 0U, numGrains);

  optimizeHaloMapping(grainToTiles);

  // Build a map from sets of tiles to grains they use.
  std::map<std::set<unsigned>, std::vector<Interval>>
      tilesToGrains;
  for (const auto &entry : grainToTiles) {
    tilesToGrains[entry.second].emplace_back(entry.first.lower(),
                                             entry.first.upper());
  }
  const auto numTiles = graph.getTarget().getNumTiles();
  std::vector<std::vector<Interval>> mapping(numTiles);
  const auto minGrainsPerTile =
      (minElementsPerTile + grainSize - 1) / grainSize;
  for (const auto &entry : tilesToGrains) {
    const auto &tiles = entry.first;
    const auto &sharedGrains = entry.second;
    const auto perTileGrains =
        splitRegions(sharedGrains, 1, tiles.size(), minGrainsPerTile);
    unsigned i = 0;
    for (auto tile : tiles) {
      if (i == perTileGrains.size())
        break;
      mapping[tile].reserve(perTileGrains[i].size());
      for (const auto &interval : perTileGrains[i]) {
        const auto lower = interval.begin() * grainSize;
        const auto upper = std::min(interval.end() * grainSize, numElements);
        mapping[tile].emplace_back(lower, upper);
      }
      ++i;
    }
  }
  return mapping;
}

static boost::icl::discrete_interval<unsigned>
toIclInterval(const Interval &interval) {
  return boost::icl::interval<unsigned>::right_open(interval.begin(),
                                                    interval.end());
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
flattenDims(Tensor t, unsigned from, unsigned to) {
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
  flattenedShape[1] *= flattenedShape[0];
  flattenedShape[0] = 1;
  t = t.reshape(flattenedShape);
  // Undo the previous permutation.
  t = t.dimShuffle(inversePermutation(bringToFront));
  return t;
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

static Tensor dilate(Graph &graph, const Tensor &t, unsigned dilationFactor,
                     unsigned dim) {
  const auto oldSize = t.dim(dim);
  const auto newSize = getDilatedSize(oldSize, dilationFactor);
  if (newSize == oldSize)
    return t;
  auto expandedT = t.expand({dim + 1});
  const auto dType = expandedT.elementType();
  auto zeroShape = expandedT.shape();
  zeroShape[dim + 1] = dilationFactor - 1;
  Tensor zero = graph.addConstant(dType, zeroShape, 0);
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

poplar::Tensor
padWithNearestNeighbour(poplar::Tensor t, unsigned paddingLower,
                        unsigned paddingUpper, unsigned dim) {
  const auto type = t.elementType();
  if (paddingLower > 0) {
    auto padding = t.slice(0, 1, dim).broadcast(paddingLower, dim);
    t = concat(padding, t, dim);
  }
  if (paddingUpper > 0) {
    auto padding = t.slice(t.dim(dim) - 1, t.dim(dim), dim)
                    .broadcast(paddingUpper, dim);
    t = concat(t, padding, dim);
  }
  return t;
}

static void expandSpatialDim(Graph &graph, ConvParams &params,
                             Tensor *acts, Tensor *weights, unsigned dim) {
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
    *acts = dilate(graph, *acts, actsDilation, actsDimIndex);
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
  if (acts) {
    // Expand the acts tensor.
    auto dType = acts->elementType();
    auto expandedShape = acts->shape();
    expandedShape[actsDimIndex] = params.getOutputSize(dim);
    expandedShape.back() = 0;
    std::vector<Tensor> slices;
    for (unsigned k = 0; k != weightsSize; ++k) {
      Tensor slice;
      auto weightOutRange =
          getOutputRangeForKernelIndex(dim, {0, params.getOutputSize(dim)},
                                       k, params);
      if (weightOutRange.first == weightOutRange.second) {
        auto zerosShape = expandedShape;
        zerosShape.back() = acts->dim(acts->rank() - 1);
        slice = graph.addConstant(dType, zerosShape, 0);
      } else {
        auto weightInRange = getInputRange(dim, {0, params.getOutputSize(dim)},
                                           k, params);
        slice = acts->slice(weightInRange.first,
                            weightInRange.second,
                            actsDimIndex);
        slice = slice.subSample(stride, actsDimIndex);
        const auto slicePaddingLower = weightOutRange.first;
        const auto slicePaddingUpper =
            params.getOutputSize(dim) - weightOutRange.second;
        slice = pad(graph, slice, slicePaddingLower, slicePaddingUpper,
                    actsDimIndex);
        assert(slice.dim(actsDimIndex) == params.getOutputSize(dim));
      }
      slices.push_back(std::move(slice));
    }
      auto expanded = concat(slices, acts->rank() - 1);
      *acts = expanded;
  }
  if (weights) {
    // Flatten the spatial dimension of the weights tensor into the input
    // channels.
    *weights = flattenDims(*weights, weightsDimIndex, weights->rank() - 1);
  }
  actsSize = params.getOutputSize(dim);
  params.inputChannels *= weightsSize;
  weightsSize = 1;
  weightsTruncationLower = 0;
  weightsTruncationUpper = 0;
  weightsDilation = 1;
  weightsPaddingLower = 0;
  weightsPaddingUpper = 0;
  weightsFlip = false;
  outputTruncationLower = 0;
  outputTruncationUpper = 0;
  stride = 1;
  outputPaddingLower = 0;
  outputPaddingUpper = 0;
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

/// Apply any pre-convolution transformations implied by the plan. The
/// plan and the parameters are updated to describe the convolution operation
/// performed on the transformed input. If the \a acts or \ weights pointers are
/// not null they are updated to be views of the original tensors with
/// dimensions that match the shape expected by the convolution operation.
static void
convolutionPreprocess(Graph &graph, ConvParams &params,
                      Plan &plan, unsigned level,
                      boost::optional<Tensor> &acts,
                      boost::optional<Tensor> &weights) {
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
  for (auto dim : transform.expandDims) {
    expandSpatialDim(graph, params, acts.get_ptr(), weights.get_ptr(), dim);
  }
  transform.expandDims.clear();
  if (!transform.outChanFlattenDims.empty()) {
    boost::optional<Tensor> maybeActs, maybeWeights;
    if (acts)
      maybeActs.reset(*acts);
    if (weights)
      maybeWeights.reset(*weights);
    swapOperands(params, maybeActs, maybeWeights);
    for (auto dim : transform.outChanFlattenDims) {
      expandSpatialDim(graph, params, maybeActs.get_ptr(),
                       maybeWeights.get_ptr(), dim);
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
}

static void
convolutionPreprocess(Graph &graph, ConvParams &params,
                      Plan &plan, unsigned level, Tensor &acts,
                      Tensor &weights) {
  auto actsOptional = boost::make_optional(acts);
  auto weightsOptional = boost::make_optional(weights);
  convolutionPreprocess(graph, params, plan, level, actsOptional,
                        weightsOptional);
  acts = *actsOptional;
  weights = *weightsOptional;
}

// Postprocess results of convolution
// - undo any flattening of the field
// - undo any padding
static Tensor
convolutionPostprocess(Graph &graph, const ConvParams &originalParams,
                       const ConvTransform &transform,
                       Tensor activations, Sequence &copies) {
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
    expandSpatialDim(graph, postExpandParams, nullptr, nullptr, dim);
  }
  auto postOutChanFlattenParams = postExpandParams;
  if (!transform.outChanFlattenDims.empty()) {
    swapOperands(postOutChanFlattenParams);
    for (auto dim : transform.outChanFlattenDims) {
      expandSpatialDim(graph, postOutChanFlattenParams, nullptr, nullptr, dim);
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
    auto outChansPerGroup = detectChannelGrouping(activations);
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
      activationsView = dilate(graph, activationsView, dilation, dim);
      mappingView = dilateWithNearestNeighbour(mappingView, dilation, dim);
      activationsView = pad(graph, activationsView, paddingLower, paddingUpper,
                            dim);
      mappingView = padWithNearestNeighbour(mappingView, paddingLower,
                                            paddingUpper, dim);
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
    const std::vector<ConvIndices> &indices,
    std::function<void (unsigned, Tensor *, Tensor *)> f) {
  boost::optional<Tensor> acts, weights;
  if (actsPtr)
    acts = *actsPtr;
  if (weightsPtr)
    weights = *weightsPtr;
  if (!postPreprocess) {
    // Transform.
    convolutionPreprocess(graph, params, plan, level, acts, weights);
  }
  Tensor out;
  if (level == plan.partitions.size()) {
    const auto tile = linearizeTileIndices(graph.getTarget(), indices,
                                           plan);
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
                               subIndices, f);
    });
  }
  return out;
}

static Tensor
iterateTilePartition(
    Graph &graph, ConvParams params, Plan plan, Tensor *actsPtr,
    Tensor *weightsPtr, std::function<void (unsigned, Tensor *, Tensor *)> f) {
  return iterateTilePartitionImpl(graph, params, plan, 0, false, actsPtr,
                                  weightsPtr, {}, f);
}

/// Map the input tensor such that the exchange required during the
/// convolution operation is minimized. If \a isActs is true then the
/// tensor is mapped assuming it the activations operand in convolution
/// operation, otherwise it is mapped assuming it is the weights operand.
static void mapActivationsOrWeights(Graph &graph, ConvParams params,
                                    Plan plan, unsigned level,
                                    bool postPreprocess,
                                    const std::vector<ConvIndices> &indices,
                                    Tensor in, bool isActs) {
  auto flattenedIn = in.flatten();
  in = isActs ? unsplitActivationChanGroups(in) : ungroupWeights(in);

  // Build a map from the input to the set of tiles that access them.
  const auto numTiles = graph.getTarget().getNumTiles();
  std::vector<boost::icl::interval_set<unsigned>> used(numTiles);
  auto zeroElement = graph.addConstant(in.elementType(), {}, 0);
  iterateTilePartitionImpl(graph, params, plan, level, postPreprocess,
                           isActs ? &in : nullptr,
                           isActs ? nullptr : &in, indices,
                           [&](unsigned tile, Tensor *acts, Tensor *weights) {
    assert((acts && !weights) || (weights && !acts));
    Tensor flattened = acts ? acts->flatten() : weights->flatten();
    std::vector<Interval> intervals;
    const auto contiguousRegions =
        graph.getSortedContiguousRegions(flattened,
                                         {{0, flattened.numElements()}});
    for (const auto &contiguousRegion : contiguousRegions) {
      for (const auto &interval : contiguousRegion) {
        auto firstElement = flattened[interval.begin()];
        if (firstElement == zeroElement)
          continue;
        auto firstIndex = firstElement.getElementIndices()[0];
        assert(flattenedIn[firstIndex] == firstElement);
        intervals.emplace_back(firstIndex, firstIndex + interval.size());
      }
    }
    auto &useSet = used[tile];
    for (const auto &interval : intervals) {
      useSet.add(toIclInterval(interval));
    }
  });
  boost::icl::interval_map<unsigned, std::set<unsigned>> inToTiles;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    std::set<unsigned> tileSet{tile};
    for (const auto &region : used[tile]) {
      inToTiles.add(std::make_pair(region, tileSet));
    }
  }
  // Limit the minimum number of bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was chosen
  // experimentally.
  const auto inType = params.dType;
  const auto inTypeSize = graph.getTarget().getTypeSize(inType);
  const auto minBytesPerTile = isActs ? 128 : 256;
  const auto minElementsPerTile =
    (minBytesPerTile + inTypeSize - 1) / minBytesPerTile;
  const auto grainSize =
      isActs ? plan.inChansPerGroup :
               plan.inChansPerGroup * plan.partialChansPerGroup;
  auto mapping =
      calculateMappingBasedOnUsage(graph, flattenedIn.shape(), inToTiles,
                                   grainSize, minElementsPerTile);
  graph.setTileMapping(flattenedIn, mapping);
}

static void
mapActivations(Graph &graph, ConvParams params, Plan plan, unsigned level,
               bool postPreprocess, const std::vector<ConvIndices> &indices,
               Tensor acts) {
  return mapActivationsOrWeights(graph, params, plan, level, postPreprocess,
                                 indices, acts, true);
}

static void
mapWeights(Graph &graph, ConvParams params, Plan plan, unsigned level,
           bool postPreprocess, const std::vector<ConvIndices> &indices,
           Tensor weights) {
  return mapActivationsOrWeights(graph, params, plan, level, postPreprocess,
                                 indices, weights, false);
}

static Tensor
createInputImpl(Graph &graph, const ConvParams &params,
                unsigned level, bool postPreprocess,
                const std::vector<ConvIndices> &indices,
                const std::string &name,
                const Plan &plan) {
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
  mapActivations(graph, params, plan, level, postPreprocess, indices, t);
  t = unsplitActivationChanGroups(t);
  return t;
}

Tensor
createInput(Graph &graph, const ConvParams &params_,
            const std::string &name,
            const ConvOptions &options) {
  auto params = canonicalizeParams(params_);
  const auto plan = getPlan(graph, params, options);
  auto input = createInputImpl(graph, params, 0, false, {}, name, plan);
  return actsToExternalShape(input);
}

static Tensor
createWeightsImpl(Graph &graph, const ConvParams &params, unsigned level,
                  bool postPreprocess,
                  const std::vector<ConvIndices> &indices,
                  const std::string &name, const Plan &plan) {
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
  mapWeights(graph, params, plan, level, postPreprocess, indices, weights);
  weights = ungroupWeights(weights);
  return weights;
}

Tensor
createWeights(Graph &graph,
              const ConvParams &params_, const std::string &name,
              const ConvOptions &options) {
  auto params = canonicalizeParams(params_);
  const auto plan = getPlan(graph, params, options);
  return weightsToExternalShape(createWeightsImpl(graph, params, 0, false, {},
                                                  name, plan));
}

static std::vector<std::vector<poplar::Interval>>
computeBiasMapping(Graph &graph, const Tensor &out) {
  const auto &target = graph.getTarget();
  const auto dType = out.elementType();
  const auto dTypeSize = target.getTypeSize(dType);
  const auto numTiles = graph.getTarget().getNumTiles();
  const unsigned numChans = out.dim(0) * out.dim(out.rank() - 1);
  // Create a view of the output where channels are the outermost dimension.
  auto outRegrouped = out.dimShufflePartial({out.rank() - 1}, {1})
                         .reshape({numChans, out.numElements() / numChans});
  const auto outRegroupedMapping = graph.getTileMapping(outRegrouped);
  // Build a map from the bias to the set of tiles that access it.
  boost::icl::interval_map<unsigned, std::set<unsigned>> biasToTiles;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &interval : outRegroupedMapping[tile]) {
      unsigned chanBegin = interval.begin() / outRegrouped.dim(1);
      unsigned chanEnd = (interval.end() + outRegrouped.dim(1) - 1) /
                         outRegrouped.dim(1);
      auto biasInterval =
          boost::icl::interval<unsigned>::right_open(chanBegin, chanEnd);
      biasToTiles += std::make_pair(biasInterval, std::set<unsigned>({tile}));
    }
  }
  const auto grainSize = target.getVectorWidth(dType);

  // Limit the minimum number of bias bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto minBytesPerTile = 8;
  const auto minElementsPerTile =
      (minBytesPerTile + dTypeSize - 1) / dTypeSize;
  return calculateMappingBasedOnUsage(graph, {numChans}, biasToTiles,
                                      grainSize, minElementsPerTile);
}

static void
mapBiases(poplar::Graph &graph, const poplar::Tensor &biases,
          const poplar::Tensor &out) {
  auto mapping = computeBiasMapping(graph, out);
  applyTensorMapping(graph, biases, mapping);
}

poplar::Tensor
createBiases(poplar::Graph &graph, const Tensor &acts_,
             const std::string &name) {
  const auto acts = actsToInternalShape(acts_, 1);
  const auto numOutChans = acts.dim(acts.rank() - 1);
  const auto dType = acts.elementType();
  auto biases = graph.addVariable(dType, {numOutChans}, name);
  mapBiases(graph, biases, acts);
  return biases;
}

static unsigned getNumConvUnits(bool floatActivations,
                                bool floatPartial,
                                const poplar::Target &target) {
  if (floatActivations) {
    return target.getFp32InFp32OutConvUnitsPerTile();
  } else {
    return floatPartial ? target.getFp16InFp32OutConvUnitsPerTile() :
                          target.getFp16InFp16OutConvUnitsPerTile();
  }
}

// create a zero list for a tensor based on the number of elements in the tensor
// The work list assumes that the tensor elements are laid out in contiguously
// in memory
static std::vector<unsigned>
createZeroWorklist(const Target &target, const Tensor &out) {
  const auto grainSize = target.getVectorWidth(out.elementType());
  const auto contextsPerVertex = target.getNumWorkerContexts();
  auto splitZeroList = splitRegions({{0, out.numElements()}},
                                    grainSize, contextsPerVertex);
  std::vector<unsigned> zeroWorklist(2 * contextsPerVertex);
  for (auto i = 0U; i != splitZeroList.size(); ++i) {
    for (auto &region : splitZeroList[i]) {
      zeroWorklist[2 * i] = region.begin();
      zeroWorklist[2 * i + 1] = region.end() - region.begin();
    }
  }
  return zeroWorklist;
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

static void createConvPartialAmpVertex(Graph &graph,
                                       const Plan &plan,
                                       unsigned tile,
                                       ConvParams params,
                                       ComputeSet fwdCS,
                                       Tensor in, Tensor weights, Tensor out) {
  assert(params == canonicalizeParams(params));
  const auto &target = graph.getTarget();
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(in.elementType() == FLOAT);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  if (convUnitWeightHeight != 1) {
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
      weights = pad(graph, weights, extraKernelPaddingLower,
                    extraKernelPaddingUpper, 3);
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
    in = dilate(graph, in, inputDilation, 3);
    params.inputTransform.dilation[0] = 1;
    params.inputFieldShape[0] = getDilatedSize(params.inputFieldShape[0],
                                               inputDilation);

    const auto inputPaddingLower = params.inputTransform.paddingLower[0];
    const auto inputPaddingUpper = params.inputTransform.paddingUpper[0];
    in = pad(graph, in, inputPaddingLower, inputPaddingUpper, 3);
    params.inputFieldShape[0] += inputPaddingLower + inputPaddingUpper;
    params.inputTransform.paddingLower[0] = 0;
    params.inputTransform.paddingUpper[0] = 0;
  }

  const auto numFieldDims = params.getNumFieldDims();
  const unsigned numConvGroups = out.dim(0);
  const unsigned numOutChanGroups = out.dim(1);
  const unsigned numInChanGroups = in.dim(1);
  const auto outChansPerGroup = plan.partialChansPerGroup;
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
  bool flipOut = params.inputTransform.flip[numFieldDims - 1];

  std::vector<Tensor> outWindow;
  std::vector<Tensor> inWindow;
  std::vector<Tensor> weightsWindow;
  for (unsigned cg = 0; cg != numConvGroups; ++cg) {
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      auto o = out[cg][ozg].flatten();
      outWindow.push_back(o);
    }
    // TODO if the tile kernel size is 1 and the stride is greater than one we
    // could subsample the input instead of using input striding.
    for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
      auto window = in[cg][izg].flatten();
      inWindow.push_back(window);
    }
    for (unsigned ozg = 0; ozg < numOutChanGroups; ++ozg) {
      for (unsigned izg = 0; izg < numInChanGroups; ++izg) {
        auto window = weights[cg][ozg][izg].flatten();
        weightsWindow.push_back(window.flatten());
      }
    }
  }

  // This stride is what's used to move down one element in the input field by
  // the vertex.
  int inRowStride =
      inChansPerGroup * params.kernelTransform.dilation.front() *
      product(params.inputFieldShape) / params.inputFieldShape[0];
  if (params.inputTransform.flip.front() !=
      params.kernelTransform.flip.front()) {
    inRowStride = -inRowStride;
  }

  std::vector<std::size_t> inputBatchAndFieldShape = { params.getBatchSize() };
  std::copy(params.inputFieldShape.begin(), params.inputFieldShape.end(),
            std::back_inserter(inputBatchAndFieldShape));
  std::vector<std::size_t> outputBatchAndFieldShape = { params.getBatchSize() };
  const auto outputFieldShape = params.getOutputFieldShape();
  std::copy(outputFieldShape.begin(), outputFieldShape.end(),
            std::back_inserter(outputBatchAndFieldShape));
  const auto contextsPerVertex = target.getNumWorkerContexts();
  // The number of n x 1 x ... 1 slices required to cover the kernel in each
  // dimension.
  auto numSubKernelSlices = params.kernelShape;
  assert(numSubKernelSlices[0] % convUnitWeightHeight == 0);
  numSubKernelSlices[0] /= convUnitWeightHeight;
  const auto numSubKernelPositions = product(numSubKernelSlices);
  std::vector<std::vector<unsigned>> worklist(contextsPerVertex *
                                              numSubKernelPositions);
  for (unsigned k = 0; k != numSubKernelPositions; ++k) {
    auto kernelBeginIndices = unflattenIndex(numSubKernelSlices, k);
    kernelBeginIndices[0] = kernelBeginIndices[0] * convUnitWeightHeight;
    std::vector<unsigned> tileConvOutBegin;
    std::vector<unsigned> tileConvOutSize;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      const auto kernelBeginIndex = kernelBeginIndices[dim];
      const auto kernelEndIndex =
          kernelBeginIndex + (dim == 0 ? convUnitWeightHeight : 1);
      auto convOutRange =
          getOutputRangeForKernelRange(dim, {0, params.getOutputSize(dim)},
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
        outBeginIndices.push_back(partialRow.xBegin + tileConvOutBegin.back());
        const auto outBeginOffset =
            flattenIndex(outputBatchAndFieldShape, outBeginIndices);
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
                  (inRowStride < 0 ? 1 : -1) *
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
        auto workerInXBegin =
            getInputRange(numFieldDims - 1, {workerOutXBegin, workerOutXEnd},
                          kernelBeginIndices.back(), params).first;
        assert(workerInXBegin != ~0U);
        inBeginIndices.push_back(workerInXBegin);
        const auto inBeginOffset =
            flattenIndex(inputBatchAndFieldShape, inBeginIndices);
        worklist[k * contextsPerVertex + i].push_back(outBeginOffset);
        worklist[k * contextsPerVertex + i].push_back(workerOutWidth);
        worklist[k * contextsPerVertex + i].push_back(inBeginOffset);
      }
    }
  }
  unsigned numEdges = inWindow.size() + outWindow.size() + weightsWindow.size();
  auto codeletName = useConvPartial1x1OutVertex ?
                       "popconv::ConvPartial1x1Out" :
                       "popconv::ConvPartialnx1";
  auto v = graph.addVertex(fwdCS,
                           templateVertex(codeletName, in.elementType(),
                                          plan.types.back().partialType,
                                        useDeltaEdgesForConvPartials(numEdges) ?
                                                          "true" : "false"));
  auto kernelInnerElements = product(numSubKernelSlices) /
                             numSubKernelSlices[0];
  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroups"], numOutChanGroups);
  graph.setInitialValue(v["numInGroups"], numInChanGroups);
  graph.setInitialValue(v["inStride"], inStrideX) ;
  graph.setInitialValue(v["numConvGroups"], numConvGroups);
  graph.setInitialValue(v["flipOut"], flipOut);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstant(UNSIGNED_INT, {worklist[i].size()},
                               worklist[i].data());
    graph.connect(v["worklists"][i], t);
  }

  if (!useConvPartial1x1OutVertex) {
    graph.setInitialValue(v["kernelInnerElements"], kernelInnerElements);
    graph.setInitialValue(v["kernelOuterSize"], numSubKernelSlices[0]);
    graph.setInitialValue(v["outStride"], outStrideX);
    graph.setInitialValue(v["ampKernelHeight"], convUnitWeightHeight);
    graph.setInitialValue(v["inRowStride"], inRowStride);

    const auto zeroWorklist = createZeroWorklist(target, outWindow[0]);
    auto zeroWorklistTensor = graph.addConstant(UNSIGNED_INT,
                                                {zeroWorklist.size()},
                                                zeroWorklist.data());
    graph.connect(v["zeroWorklist"], zeroWorklistTensor);
  }
  graph.setTileMapping(v, tile);
}

static void
createConvPartialHorizontalMacVertex(Graph &graph,
                                     const Plan &plan,
                                     unsigned tile,
                                     const ConvParams &params,
                                     ComputeSet fwdCS,
                                     const Tensor &in,
                                     const Tensor &weights,
                                     const Tensor &out) {
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
          worklist[kIndex * contextsPerVertex + i].push_back(outBeginOffset);
          worklist[kIndex * contextsPerVertex + i].push_back(workerOutWidth);
          worklist[kIndex * contextsPerVertex + i].push_back(inBeginOffset);
        }
      }
    }
  }

  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;
  auto v = graph.addVertex(fwdCS,
                           templateVertex("popconv::ConvPartialHorizontalMac",
                                          in.elementType(),
                                          plan.types.back().partialType));
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroups"], numOutChanGroups);
  graph.setInitialValue(v["numInGroups"], numInChanGroups);
  graph.setInitialValue(v["kernelSize"], numKernelFieldElems);
  graph.setInitialValue(v["inStride"], inStrideX);
  graph.setInitialValue(v["outStride"], outStrideX);
  graph.setInitialValue(v["numConvGroups"], numConvGroups);
  graph.setInitialValue(v["flipOut"], flipOut);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstant(UNSIGNED_INT, {worklist[i].size()},
                               worklist[i].data());
    graph.connect(v["worklists"][i], t);
  }
  const auto zeroWorklist = createZeroWorklist(target, outWindow[0]);
  auto zeroWorklistTensor = graph.addConstant(UNSIGNED_INT,
                                              {zeroWorklist.size()},
                                              zeroWorklist.data());
  graph.connect(v["zeroWorklist"], zeroWorklistTensor);
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
                               "popconv::OuterProduct", dType
                             ),
                             {{"in", inWindow},
                              {"weights", weightsWindow},
                              {"out", outWindow}});
    graph.setTileMapping(v, tile);
  }
}

static bool isZeroConvolution(const ConvParams &params) {
  if (!params.getNumOutputChansPerConvGroup())
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

static Tensor
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      unsigned tile,
                      ConvParams params,
                      ComputeSet convolveCS,
                      Tensor in, Tensor weights) {
  const auto numOutChans = params.getNumOutputChansPerConvGroup();
  const auto outChansPerGroup = plan.partialChansPerGroup;
  assert(numOutChans % outChansPerGroup == 0);
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
  auto out = graph.addVariable(plan.types.back().partialType, outShape,
                               "partials");
  graph.setTileMapping(out, tile);
  in = splitActivationChanGroups(in, plan.inChansPerGroup);
  weights = groupWeights(weights, plan.inChansPerGroup,
                         plan.partialChansPerGroup);
  if (isZeroConvolution(params)) {
    zero(graph, out, tile, convolveCS);
    return unsplitActivationChanGroups(out);
  }
  switch (plan.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    createConvPartialAmpVertex(graph, plan, tile, params, convolveCS,
                               in, weights, out);
    break;
  case Plan::Method::MAC:
    createConvPartialHorizontalMacVertex(graph, plan, tile, params,
                                         convolveCS, in, weights, out);
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
  return unsplitActivationChanGroups(out);
}

static Tensor
partialGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &
        tileGroupRegions,
    const Tensor &partials,
    unsigned outDepth,
    const Type &resultType,
    ComputeSet cs) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;
  Tensor out = graph.addVariable(resultType,
                                 outDims,
                                 "partialReduceOut");
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numTileGroups = tileGroupRegions.size();
  const unsigned minGrainSize = target.getVectorWidth(resultType);
  const unsigned partialChansPerGroup = partials.dim(partials.rank()-1);
  const auto grainSize = std::max(partialChansPerGroup, minGrainSize);

  for (unsigned i = 0; i != outDepth; ++i) {
    unsigned begin = (i * partialsDepth) / outDepth;
    unsigned end = ((i + 1) * partialsDepth) / outDepth;
    std::vector<std::vector<Interval>>
        outSubMapping(numTiles);
    for (unsigned tileGroup = 0; tileGroup != numTileGroups; ++tileGroup) {
      const auto tilesInGroup = tileGroups[tileGroup].size();
      const auto tileBegin = (i * tilesInGroup) / outDepth;
      const auto tileEnd = std::max(tileBegin + 1,
                                    ((i + 1) * tilesInGroup) / outDepth);
      const auto outSplitRegions =
          splitRegions(tileGroupRegions[tileGroup], grainSize,
                       tileEnd - tileBegin);
      for (unsigned j = 0; j != outSplitRegions.size(); ++j) {
        outSubMapping[tileGroups[tileGroup][j + tileBegin]] =
            outSplitRegions[j];
      }
    }
    applyTensorMapping(graph, out[i], outSubMapping);
    popops::reduce(graph, partials.slice(begin, end), out[i],
                      outSubMapping, cs);
  }
  return out;
}

static Tensor
groupedReduce(Graph &graph,
              const std::vector<std::vector<unsigned>> &tileGroups,
              const std::vector<
                std::vector<Interval>
              > &tileGroupRegions,
              const Tensor &partials,
              const Type &resultType,
              ComputeSet cs) {
  return partialGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
         1, resultType, cs).reshape(partials[0].shape());
}

/// Return the number of reduce stages to use for a reduction of the specified
/// reduction depth.
static unsigned getNumReduceStages(unsigned partialsDepth) {
  /// Using more reduce stages affects code size as follows.
  /// If the reduction depth is p then a single stage reduction requires each
  /// tile to receive p messages. If instead we break the reduction down into n
  /// stages then each stage involves a reduction of reduce p^(1/n) messages.
  /// The total number of messages is n*p^(1/n). For large p, increase n
  /// will reducing the total number of messages received which is turn likely
  /// to also reduce the exchange code size. The thresholds below have been
  /// chosen based on benchmarking.
  if (partialsDepth >= 125)
    return 3;
  if (partialsDepth >= 16)
    return 2;
  return 1;
}

/// Return a plan for how to split a reduction into multiple stages along with
/// an estimate of the cost of the plan. The first member of the pair is a
/// vector of the depth of each partials tensor in each intermediate stage.
/// If the vector is empty there are no intermediate stages and the reduction
/// is performed in a single step. The second member of the pair is an
/// estimated cost. The cost is an estimate of the average number of messages
/// required per tile.
static std::pair<std::vector<unsigned>, float>
getMultiStageReducePlanAndCost(unsigned partialsDepth, unsigned numStages) {
  if (numStages == 1) {
    return {{}, partialsDepth};
  }
  auto nextDepthRoundDown =
      static_cast<unsigned>(
        std::pow(static_cast<double>(partialsDepth),
                 (numStages - 1.0) / numStages)
      );
  std::vector<unsigned> roundDownPlan, roundUpPlan;
  float roundDownCost, roundUpCost;
  std::tie(roundDownPlan, roundDownCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundDown, numStages - 1);
  roundDownCost += static_cast<float>(partialsDepth) / nextDepthRoundDown;
  auto nextDepthRoundUp = nextDepthRoundDown + 1;
  std::tie(roundUpPlan, roundUpCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundUp, numStages - 1);
  roundUpCost += static_cast<float>(partialsDepth) / nextDepthRoundUp;
  if (roundDownCost < roundUpCost) {
    roundDownPlan.insert(roundDownPlan.begin(), nextDepthRoundDown);
    return {roundDownPlan, roundDownCost};
  }
  roundUpPlan.insert(roundUpPlan.begin(), nextDepthRoundUp);
  return {roundUpPlan, roundUpCost};
}

static std::vector<unsigned>
getMultiStageReducePlan(unsigned partialsDepth) {
  const auto numStages = getNumReduceStages(partialsDepth);
  return getMultiStageReducePlanAndCost(partialsDepth, numStages).first;
}

static Tensor
multiStageGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &
        tileGroupRegions,
    Tensor partials,
    const Type &resultType,
    std::vector<ComputeSet> &computeSets,
    const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  auto plan = getMultiStageReducePlan(partialsDepth);
  for (unsigned i = computeSets.size(); i <= plan.size(); ++i) {
    computeSets.push_back(
      graph.addComputeSet(debugPrefix + "/Reduce" +
                             std::to_string(i))
    );
  }
  const auto partialsType = partials.elementType();
  for (unsigned i = 0; i != plan.size(); ++i) {
    partials = partialGroupedReduce(graph, tileGroups, tileGroupRegions,
                                    partials, plan[i], partialsType,
                                    computeSets[i]);
  }
  auto reduced = groupedReduce(graph, tileGroups, tileGroupRegions, partials,
                               resultType, computeSets[plan.size()]);
  return reduced;
}

static Tensor
multiStageGroupedReduce(Graph &graph,
                        Tensor partials,
                        const Type &resultType,
                        std::vector<ComputeSet> &computeSets,
                        const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  // Build a map from the output to the set of tiles that contribute partial
  // sums.
  boost::icl::interval_map<unsigned, std::set<unsigned>> outputToTiles;
  for (unsigned i = 0; i != partialsDepth; ++i) {
    const auto tileMapping = graph.getTileMapping(partials[i]);
    for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
      for (const auto &interval : tileMapping[tile]) {
        outputToTiles += std::make_pair(toIclInterval(interval),
                                        std::set<unsigned>({tile}));
      }
    }
  }
  // Build a map from sets of tiles the outputs they contribute to.
  std::map<std::set<unsigned>, std::vector<Interval>>
      tilesToOutputs;
  for (const auto &entry : outputToTiles) {
    tilesToOutputs[entry.second].emplace_back(entry.first.lower(),
                                              entry.first.upper());
  }
  std::vector<std::vector<unsigned>> tileGroups;
  std::vector<std::vector<Interval>> tileGroupRegions;
  tileGroups.reserve(tilesToOutputs.size());
  tileGroupRegions.reserve(tilesToOutputs.size());
  for (const auto &entry : tilesToOutputs) {
    tileGroups.emplace_back(entry.first.begin(), entry.first.end());
    tileGroupRegions.push_back(std::move(entry.second));
  }
  return multiStageGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
                                 resultType, computeSets, debugPrefix);
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
/// concaternating them in the specified dimension to form a new
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
stichPartialResults(const std::vector<std::vector<Tensor>> &results,
                    const Partition &partition) {
  std::vector<Tensor> partials;
  partials.reserve(results.size());
  auto dimSplits = getOutputDimSplits(partition);
  for (const auto &entry : results) {
    partials.push_back(stichResults(entry, dimSplits));
    partials.back() = partials.back().expand({0});
  }
  return concat(partials, 0);
}

static Tensor
convolutionImpl(Graph &graph, ConvParams params,
                Plan plan, unsigned level,
                Tensor in, Tensor weights,
                std::vector<Sequence> &copies,
                ComputeSet convolveCS,
                std::vector<std::vector<ComputeSet>> &reduceComputeSets,
                std::vector<Sequence> &postCopies,
                const std::vector<ConvIndices> &indices,
                const std::string &debugPrefix,
                const ConvOptions &options) {
  // Transform.
  const auto originalParams = params;
  const auto originalTransform = plan.transforms[level];
  convolutionPreprocess(graph, params, plan, level, in, weights);
  const auto ipuLevel = plan.transforms.size() - 2;
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
    const auto inViewMaxBroadcastDests =
        inputRearrangementIsExpensive(options) ? 1U : 7U;
    const auto weightViewMaxBroadcastDests =
        weightRearrangementIsExpensive(options) ? 1U : 7U;
    const auto inNumDests =
        std::accumulate(plan.partitions.back().kernelSplit.begin(),
                        plan.partitions.back().kernelSplit.end(),
                        1U,
                        std::multiplies<unsigned>()) *
                        plan.partitions.back().outChanSplit;
    if (inNumDests > inViewMaxBroadcastDests) {
      auto inRearranged = createInputImpl(graph, params, level, true, indices,
                                          "inRearranged", plan);
      copies[level].add(Copy(in, inRearranged));
      in = inRearranged;
    }
    auto weightsNumDests = plan.partitions.back().batchSplit;
    for (const auto split : plan.partitions.back().fieldSplit) {
      weightsNumDests *= split;
    }
    if (weightsNumDests > weightViewMaxBroadcastDests) {
      auto weightsRearranged = createWeightsImpl(graph, params, level, true,
                                                 indices, "weightsRearranged",
                                                 plan);
      copies[level].add(Copy(weights, weightsRearranged));
      weights = weightsRearranged;
    }
  }
  Tensor out;
  const auto resultType = plan.types[level].resultType;
  if (level == plan.partitions.size()) {
    const auto tile = linearizeTileIndices(graph.getTarget(), indices,
                                           plan);
    out =
        calcPartialConvOutput(graph, plan, tile, params, convolveCS, in,
                              weights);
  } else {
    const auto &partition = plan.partitions[level];
    const auto numPartials = partition.inChanSplit *
                             product(partition.kernelSplit);
    const auto outputSplit = partition.convGroupSplit *
                             partition.batchSplit *
                             product(partition.fieldSplit) *
                             partition.outChanSplit;
    std::vector<std::vector<Tensor>> results(numPartials,
                                             std::vector<Tensor>(outputSplit));
    iteratePartition(params, partition, [&](const ConvIndices &levelIndices,
                                            const ConvSlice &slice) {
      // Get sub convolution
      ConvParams subParams = params;
      Tensor subIn = in;
      Tensor subWeights = weights;
      auto subIndices = indices;
      subIndices.push_back(levelIndices);
      getSubConvolution(slice, subParams, &subIn, &subWeights);
      auto subOut = convolutionImpl(graph, subParams, plan, level + 1, subIn,
                                    subWeights, copies, convolveCS,
                                    reduceComputeSets, postCopies, subIndices,
                                    debugPrefix, options);
      auto partialIndex = getPartialIndex(levelIndices, partition);
      auto outputIndex = getOutputIndex(levelIndices, partition);
      results[partialIndex][outputIndex] = subOut;
    });
    // Stich together results.
    auto partials = stichPartialResults(results, partition);
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
    out = cast(graph, out, resultType, reduceComputeSets[level][0]);
  }
  // Inverse transform.
  out = convolutionPostprocess(graph, originalParams, originalTransform, out,
                               postCopies[level]);
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

Tensor
convolution(Graph &graph, const poplar::Tensor &in_,
            const poplar::Tensor &weights_,
            const ConvParams &params_,
            bool transposeAndFlipWeights, Sequence &prog,
            const std::string &debugPrefix, const ConvOptions &options) {
  auto params = canonicalizeParams(params_);
  auto weights = weights_;
  if (weights.rank() == params_.getNumFieldDims() + 2) {
    weights = weights.expand({0});
  }
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    auto bwdWeights = createWeights(graph, params, "bwdWeights", options);
    if (bwdWeights.dim(1) && bwdWeights.dim(2))
      weightsTransposeChansFlipXY(graph, weights, bwdWeights, prog,
                                  debugPrefix);
    weights = bwdWeights;
  }
  weights = weightsToInternalShape(weights);
  auto in = actsToInternalShape(in_, params.numConvGroups);
  auto plan = getPlan(graph, params, options);
  verifyInputShapes(params, in, weights);
  if (plan.useWinograd) {
    throw poputil::poplib_error("Winograd not yet supported");
  }
  const auto layerName = debugPrefix + "/Conv" + convSuffix(params);
  const auto numLevels = plan.partitions.size() + 1;
  std::vector<Sequence> copies(numLevels), postCopies(numLevels);
  std::vector<std::vector<ComputeSet>> reduceComputeSets(numLevels);
  auto convolveCS = graph.addComputeSet(layerName + "/Convolve");
  auto activations =
      convolutionImpl(graph, params, plan, 0, in, weights, copies, convolveCS,
                      reduceComputeSets, postCopies, std::vector<ConvIndices>(),
                      layerName, options);
  for (const auto &p : copies) {
    prog.add(p);
  }
  prog.add(Execute(convolveCS));
  for (int level = numLevels - 1; level >= 0; --level) {
    for (const auto &reduceCS : reduceComputeSets[level]) {
      prog.add(Execute(reduceCS));
    }
    prog.add(postCopies[level]);
  }
  assert(activations.elementType() == in.elementType());
  return actsToExternalShape(activations);
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

/**
 * Transpose the innermost pair of dimensions of the specified tensor, writing
 * the results to a new tensor.
 */
static Tensor weightsPartialTranspose(Graph &graph, Tensor in, ComputeSet cs) {
  const auto &target = graph.getTarget();
  const auto rank = in.rank();
  const auto numSrcRows = in.dim(rank - 2);
  const auto numSrcColumns = in.dim(rank - 1);
  const auto dType = in.elementType();
  auto outShape = in.shape();
  std::swap(outShape[rank - 2], outShape[rank - 1]);
  auto out = graph.addVariable(dType, outShape, "partialTranspose");
  auto inFlat = in.reshape({in.numElements() / (numSrcRows * numSrcColumns),
                            numSrcRows * numSrcColumns});
  auto outFlat = out.reshape(inFlat.shape());
  const auto transpositionMapping =
      graph.getTileMapping(inFlat.slice(0, 1, 1));
  const auto numTiles = transpositionMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerTranspositions =
        splitRegionsBetweenWorkers(target, transpositionMapping[tile], 1);
    for (const auto &entry : perWorkerTranspositions) {
      const auto v =
          graph.addVertex(cs, templateVertex("popconv::Transpose2d", dType));
      graph.setInitialValue(v["numSrcColumns"],
                            static_cast<unsigned>(numSrcColumns));
      graph.setTileMapping(v, tile);
      unsigned i = 0;
      for (const auto &interval : entry) {
        for (auto transposition = interval.begin();
             transposition != interval.end(); ++transposition) {
          graph.connect(v["src"][i], inFlat[transposition]);
          graph.connect(v["dst"][i], outFlat[transposition]);
          graph.setTileMapping(outFlat[transposition], tile);
          ++i;
        }
      }
      graph.setFieldSize(v["src"], i);
      graph.setFieldSize(v["dst"], i);
    }
  }
  return out;
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
      groupWeights(weightsToInternalShape(weightsInUnGrouped));
  const auto weightsOut =
      groupWeights(weightsToInternalShape(weightsOutUnGrouped));
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
        weightsPartialTranspose(
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

static ConvParams
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
                      const ConvOptions &fwdOptions) {
  const auto numConvGroups = fwdParams.numConvGroups;
  auto zDeltas = actsToInternalShape(zDeltas_, numConvGroups);
  auto activations = actsToInternalShape(activations_, numConvGroups);
  auto params = getWeightUpdateParams(fwdParams);
  auto options = fwdOptions;
  options.pass = Pass::TRAINING_WU;
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
                  options);
  weightDeltas = actsToInternalShape(weightDeltas, numConvGroups);
  return weightsToExternalShape(
           weightDeltas.dimShufflePartial({1}, {weightDeltas.rank() - 1})
         );
}

void
convolutionWeightUpdate(Graph &graph,
                        const Tensor &zDeltas, const Tensor &weights,
                        const Tensor &activations,
                        const ConvParams &params,
                        float learningRate,
                        Sequence &prog,
                        const std::string &debugPrefix,
                        const ConvOptions &options) {
  auto weightDeltas = calculateWeightDeltas(graph, zDeltas, activations, params,
                                            prog, debugPrefix, options);
  // Add the weight deltas to the weights.
  assert(weightDeltas.shape() == weights.shape());
  addTo(graph, weights, weightDeltas, -learningRate, prog,
        debugPrefix + "/UpdateWeights");
}

static void
convChannelReduce(Graph &graph,
                  const Tensor &in,
                  Tensor dst,
                  bool doAcc,
                  float scale,
                  bool doSquare,
                  std::vector<ComputeSet> &computeSets,
                  const Type &partialsType,
                  const std::string &debugPrefix) {

  if (computeSets.size() == 0) {
    computeSets.push_back(graph.addComputeSet(debugPrefix + "/Reduce1"));
    computeSets.push_back(graph.addComputeSet(debugPrefix + "/Reduce2"));
    computeSets.push_back(doAcc ?
                          graph.addComputeSet(debugPrefix + "/FinalUpdate")
                          : graph.addComputeSet(debugPrefix + "/FinalReduce"));
  }
  assert(computeSets.size() == 3);

  // Force convolution grouping to be 1
  auto inGrouped =
      splitActivationChanGroups(
        actsToInternalShape(in, 1)
      )[0];

  // set this to true to enable f16v8 and f32v4 instructions
  const bool useDoubleDataPathInstr = true;

  // The bias gradient is the sum of all the deltas.
  // The reduction of these deltas is done in three stages:
  //     The first stage reduces on each tile. It places the partial sum
  //     for each tile in the tensor 'tileReducedBiasDeltas[tile]'.
  //     The second stage reduces across tiles to a set of partial sums
  //     spread across the workers. It takes 'tileReducedBiasDeltas' as input
  //     and outputs to the 'biasPartials' 2-d tensor.
  //     The final stage reduces the 'biasPartials' 2-d tensor to get the
  //     final gradient for each bias, multiplies it by the learning rate and
  //     subtracts from the bias in the 'biases' tensor.
  const auto &target = graph.getTarget();
  auto dType = inGrouped.elementType();
  auto numTiles = target.getNumTiles();
  auto numOut = dst.numElements();
  auto inChansPerGroup = inGrouped.dim(inGrouped.rank() - 1);
  // Before the cross tile reduction. Reduce biases on each tile.
  auto inFlatField = inGrouped.flatten(1, inGrouped.rank() - 1);

  // Calculate which bias groups have values to reduce on each tile
  auto firstInGroup = inFlatField.slice(0, 1, 2).squeeze({2});
  auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  std::vector<std::map<unsigned, std::vector<Interval>>>
      tileLocalReductions(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    for (const auto &interval : firstInGroupMapping[tile]) {
      auto begin = interval.begin();
      auto last = interval.end() - 1;
      auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(), begin);
      auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(), last);
      for (unsigned g = beginIndices[0]; g != lastIndices[0] + 1; ++g) {
        unsigned fieldBegin = g == beginIndices[0] ?
                                   beginIndices[1] :
                                   0;
        unsigned fieldLast = g == lastIndices[0] ?
                                  lastIndices[1] :
                                  firstInGroup.dim(1) - 1;
        unsigned flatBegin = flattenIndex(firstInGroup.shape(),
                                          {g, fieldBegin});
        unsigned flatLast = flattenIndex(firstInGroup.shape(),
                                         {g, fieldLast});
        tileLocalReductions[tile][g].emplace_back(flatBegin, flatLast + 1);
      }
    }
  }

  // On each tile create vertices that reduce the on-tile elements to a single
  // value for each element on each tile stored in the
  // tensor tileReduced[tile].
  std::vector<Tensor> tileReduced;
  tileReduced.reserve(numTiles);
  auto inGroups = inGrouped.reshape({inGrouped.numElements() / inChansPerGroup,
                                     inChansPerGroup});
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto tileNumGroups = tileLocalReductions[tile].size();
    Tensor r = graph.addVariable(partialsType, {tileNumGroups, inChansPerGroup},
                                 "tileReduced");
    tileReduced.push_back(r);
    graph.setTileMapping(r, tile);
    unsigned outIndex = 0;
    for (const auto &entry : tileLocalReductions[tile]) {
      auto v = graph.addVertex(computeSets[0],
                               templateVertex(doSquare ?
                                              "popconv::ConvChanReduceSquare" :
                                              "popconv::ConvChanReduce",
                                              dType, partialsType));
      unsigned numRanges = 0;
      for (const auto &interval : entry.second) {
        auto in = inGroups.slice(interval.begin(), interval.end())
                          .flatten();
        graph.connect(v["in"][numRanges++], in);
      }
      graph.setFieldSize(v["in"], numRanges);
      graph.connect(v["out"], r[outIndex++]);
      graph.setInitialValue(v["useDoubleDataPathInstr"],
                            useDoubleDataPathInstr);
      graph.setTileMapping(v, tile);
    }
  }

  /** The number of outputs is often small. So the reduction of ouputs
   *  is done in two stages to balance compute.
   */
  auto numWorkers = target.getNumWorkerContexts() * target.getNumTiles();
  unsigned workersPerOutput, usedWorkers, maxOutputsPerWorker;
  if (numWorkers > numOut) {
    workersPerOutput = numWorkers / numOut;
    usedWorkers = workersPerOutput * numOut;
    maxOutputsPerWorker = 1;
  } else {
    workersPerOutput = 1;
    usedWorkers = numWorkers;
    maxOutputsPerWorker = (numOut + numWorkers - 1) / numWorkers;
  }
  auto partials =
      graph.addVariable(partialsType, {usedWorkers, maxOutputsPerWorker},
                        "partials");
  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
    auto tile = worker / target.getNumWorkerContexts();
    graph.setTileMapping(partials[worker].slice(0, maxOutputsPerWorker), tile);
    unsigned outBegin = (worker  * numOut) / usedWorkers;
    unsigned outEnd = ((worker  + workersPerOutput) * numOut) / usedWorkers;
    if (outBegin == outEnd)
      continue;
    unsigned numWorkerOutputs = outEnd - outBegin;
    std::vector<Tensor> toReduceSlices;
    std::vector<unsigned> numInputsPerOutput;
    for (auto o = outBegin; o != outEnd; ++o) {
      auto outGroup = o / inChansPerGroup;
      auto outInGroup = o % inChansPerGroup;
      std::vector<Tensor> inputSlices;
      for (unsigned srcTile = 0; srcTile < numTiles; ++srcTile) {
        auto it = tileLocalReductions[srcTile].find(outGroup);
        if (it == tileLocalReductions[srcTile].end())
          continue;
        unsigned i = std::distance(tileLocalReductions[srcTile].begin(),
                                   it);
        auto srcBias = tileReduced[srcTile][i][outInGroup];
        inputSlices.push_back(srcBias.expand({0}));
      }
      if (inputSlices.empty()) {
        numInputsPerOutput.push_back(0);
      } else {
        auto inputs = concat(inputSlices);
        const auto numInputs = inputs.numElements();
        auto inBegin =
            ((worker  % workersPerOutput) * numInputs) / workersPerOutput;
        unsigned inEnd =
            (((worker  % workersPerOutput) + 1) * numInputs) / workersPerOutput;
        if (inBegin != inEnd) {
          toReduceSlices.push_back(inputs.slice(inBegin, inEnd));
        }
        numInputsPerOutput.push_back(inEnd - inBegin);
      }
    }
    if (toReduceSlices.empty()) {
      auto v = graph.addVertex(computeSets[1],
                               templateVertex("popops::Zero", partialsType));
      graph.connect(v["out"], partials[worker].slice(0, maxOutputsPerWorker));
      graph.setTileMapping(v, tile);
      continue;
    }
    auto toReduce = concat(toReduceSlices);
    auto v = graph.addVertex(computeSets[1],
                             templateVertex("popconv::ConvChanReduce2",
                                            partialsType));
    graph.connect(v["in"], toReduce);
    graph.connect(v["out"], partials[worker].slice(0, numWorkerOutputs));
    graph.setInitialValue(v["numInputsPerOutput"], numInputsPerOutput);
    graph.setTileMapping(v, tile);
  }

  const auto outMapping = graph.getTileMapping(dst);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &interval : outMapping[tile]) {
      for (unsigned o = interval.begin(); o != interval.end(); ++o) {
        std::string vType;
        if (doAcc) {
          vType = templateVertex("popconv::ConvChanReduceAcc",
                                 partialsType, dType);
        } else {
          vType = templateVertex("popconv::ConvChanReduceAndScale",
                                 partialsType, dType);
        }
        auto v = graph.addVertex(computeSets[2], vType);
        unsigned numPartials = 0;
        for (unsigned srcWorker = 0; srcWorker < usedWorkers; ++srcWorker) {
          unsigned inBegin = (srcWorker * numOut) / usedWorkers;
          unsigned inEnd =
              ((srcWorker + workersPerOutput) * numOut) / usedWorkers;
          if (inBegin > o || inEnd <= o)
            continue;
          graph.connect(v["in"][numPartials++],
                        partials[srcWorker][o - inBegin]);
        }
        graph.setFieldSize(v["in"], numPartials);
        graph.connect(v["out"], dst[o]);
        graph.setInitialValue(v["K"], scale);
        graph.setTileMapping(v, tile);
      }
    }
  }
}

static Tensor
batchNormReduce(Graph &graph,
                const Tensor &actsUngrouped,
                float scale,
                bool doSquare,
                std::vector<ComputeSet> &csVec,
                const Type &partialsType,
                const std::string &debugPrefix) {
  auto t = createBiases(graph, actsUngrouped,
                        "bnReduceResult");
  convChannelReduce(graph, actsUngrouped, t, false,
                    scale, doSquare, csVec, partialsType, debugPrefix);
  return t;
}


// Return a program to update the biases tensor with the gradients derived
// from the zDeltas tensor
void
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                      const Tensor &biases,
                      float learningRate,
                      const Type &partialsType,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  std::vector< ComputeSet> csVec;
  convChannelReduce(graph, zDeltasUngrouped, biases, true,
                    -learningRate, false,
                    csVec, partialsType,
                    debugPrefix + "/BiasUpdate");
  assert(csVec.size() == 3);
  for (const auto &cs : csVec) {
    prog.add(Execute(cs));
  }
}

static void
addToChannel(Graph &graph, const Tensor &actsUngrouped,
             const Tensor &addend, float scale, Sequence &prog,
             const std::string debugPrefix) {
  const auto fnPrefix = debugPrefix + "/addToChannel";
  auto cs = graph.addComputeSet(fnPrefix);
  const auto acts =
      splitActivationChanGroups(
        actsToInternalShape(actsUngrouped, 1)
      )[0];
  const auto dType = acts.elementType();
  const auto &target = graph.getTarget();
  const auto outChansPerGroup = acts.dim(acts.rank() - 1);
  const auto addendByGroup =
      addend.reshape({addend.numElements() / outChansPerGroup,
                      outChansPerGroup});
  const auto firstInGroup = acts.slice(0, 1, acts.rank() - 1)
                                .flatten(2, acts.rank());
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(target, firstInGroupMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      VertexRef v;
      if (scale == 1.0) {
        v = graph.addVertex(cs, templateVertex("popconv::AddToChannel",
                                               dType));
      } else {
        v = graph.addVertex(cs, templateVertex("popconv::ScaledAddToChannel",
                                               dType));
      }
      graph.setTileMapping(v, tile);
      unsigned num = 0;
      for (const auto &interval : entry) {
        const auto begin = interval.begin();
        const auto end = interval.end();
        const auto last = end - 1;
        auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                                    begin);
        auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                                   last);
        for (unsigned g = beginIndices[0]; g != lastIndices[0] + 1; ++g) {
          unsigned batchBegin = g == beginIndices[0] ?
                                beginIndices[1] :
                                0;
          unsigned batchLast = g == lastIndices[0] ?
                               lastIndices[1] :
                               firstInGroup.dim(1) - 1;
          auto addendWindow = addendByGroup[g];
          for (unsigned b = batchBegin; b != batchLast + 1; ++b) {
            unsigned begin = g == beginIndices[0] && b == beginIndices[1] ?
                             beginIndices[2] :
                             0;
            unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                            lastIndices[2] :
                            firstInGroup.dim(2) - 1;
            auto actsWindow =
                acts[g][b].flatten().slice(begin * outChansPerGroup,
                                           (last + 1) * outChansPerGroup);
            graph.connect(v["acts"][num], actsWindow);
            graph.connect(v["addend"][num], addendWindow);
            ++num;
          }
        }
      }
      if (scale != 1.0) {
        graph.setInitialValue(v["scale"], scale);
      }
      graph.setFieldSize(v["acts"], num);
      graph.setFieldSize(v["addend"], num);
    }
  }
  prog.add(Execute(cs));
}

void
addBias(Graph &graph, const Tensor &acts, const Tensor &biases,
        Sequence &prog, const std::string &debugPrefix) {
  addToChannel(graph, acts, biases, 1.0, prog, debugPrefix);
}

Tensor
fullyConnectedWeightTranspose(Graph &graph,
                              Tensor activations,
                              ConvParams params,
                              Sequence &prog, const std::string &debugPrefix,
                              const ConvOptions &options) {
  if (params.getNumFieldDims() != 1) {
    throw poputil::poplib_error("fullyConnectedWeightTranspose() expects a 1-d "
                               "convolution");
  }
  auto plan = getPlan(graph, params, options);
  auto fwdPlan = plan;
  for (auto &p : fwdPlan.partitions) {
    std::swap(p.fieldAxisGrainSize.back(), p.inChanGrainSize);
    std::swap(p.fieldSplit.back(), p.inChanSplit);
  }
  fwdPlan.inChansPerGroup = fwdPlan.partitions.back().inChanGrainSize;
  Tensor transposed = createInput(graph, params, "transposed", options);
  // split activations into conv groups
  auto splitActivations =
      actsToInternalShape(activations, params.getNumConvGroups());
  auto splitTransposed =
      actsToInternalShape(transposed, params.getNumConvGroups());
  auto splitTransposedUngroupedShape = splitTransposed.shape();
  const auto fwdGroupSize =
      getInChansPerGroup(fwdPlan,
                         static_cast<unsigned>(splitActivations.dim(3)));
  const auto bwdGroupSize =
      getInChansPerGroup(plan, static_cast<unsigned>(splitActivations.dim(2)));
  const auto dType = activations.elementType();
  const auto &target = graph.getTarget();
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
  for (unsigned tile = 0; tile != blockTileMapping.size(); ++tile) {
    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(target, blockTileMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      // Create a vertex.
      const auto v =
          graph.addVertex(transposeCS,
                          templateVertex("popconv::Transpose2d", dType));
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["numSrcColumns"],
                            static_cast<unsigned>(fwdGroupSize));
      unsigned index = 0;
      for (const auto interval : entry) {
        for (auto block = interval.begin(); block != interval.end(); ++block) {
          auto blockIndices = poputil::unflattenIndex(firstInBlock.shape(),
                                                     block);
          graph.connect(v["src"][index],
                        splitActivations[blockIndices[0]]
                                        [blockIndices[1]]
                                        [blockIndices[2]]
                                        [blockIndices[3]].flatten());
          graph.connect(v["dst"][index++],
                        splitTransposed[blockIndices[0]]
                                       [blockIndices[1]]
                                       [blockIndices[3]]
                                       [blockIndices[2]].flatten());
        }
      }
      graph.setFieldSize(v["dst"], index);
      graph.setFieldSize(v["src"], index);
    }
  }
  prog.add(Execute(transposeCS));
  auto transposedWeights =
      splitTransposed.dimShufflePartial({3}, {4})
                     .reshape(splitTransposedUngroupedShape);
  return actsToExternalShape(transposedWeights);
}

void reportPlanInfo(std::ostream &out,
                    const poplar::Graph &graph,
                    const ConvParams &params, const ConvOptions &options) {
  auto plan = getPlan(graph, params, options);
  out << plan;
}

void reportWeightUpdatePlanInfo(std::ostream &out,
                                const Graph &graph,
                                const ConvParams &fwdParams,
                                const ConvOptions &fwdOptions) {
  auto params = getWeightUpdateParams(fwdParams);
  auto options = fwdOptions;
  options.pass = Pass::TRAINING_WU;
  // The weight update is equivalent to a convolution where:
  // - wu conv groups = fwd conv groups
  // - wu batch size = fwd input channels
  // - wu input channels = fwd batch size
  // - wu height = fwd height
  // - wu width = fwd width
  // - wu output channels = fwd output channels
  reportPlanInfo(out, graph, params, options);
}


static Tensor
channelMul(Graph &graph, const Tensor &actsUngrouped, const Tensor &scale,
             Sequence &prog, const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/channelMul";
  auto cs = graph.addComputeSet(fnPrefix);

  auto actsScaledUngrouped = graph.clone(actsUngrouped, fnPrefix + "/actsIn");
  const auto acts =
      splitActivationChanGroups(
        actsToInternalShape(actsUngrouped, 1)
      )[0];
  const auto actsScaled =
      splitActivationChanGroups(
        actsToInternalShape(actsScaledUngrouped, 1)
      )[0];
  const auto dType = acts.elementType();
  const auto &target = graph.getTarget();
  const auto outChansPerGroup = acts.dim(4);
  const auto scaleByGroup =
      scale.reshape({scale.numElements() / outChansPerGroup,
                      outChansPerGroup});
  const auto firstInGroup = acts.slice(0, 1, 4)
                                .reshapePartial(2, 5,
                                                {acts.dim(2) * acts.dim(3)});
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(target, firstInGroupMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      auto v = graph.addVertex(cs,
                               templateVertex("popconv::ChannelMul",
                                              dType));
      graph.setTileMapping(v, tile);
      unsigned num = 0;
      for (const auto &interval : entry) {
        const auto begin = interval.begin();
        const auto end = interval.end();
        const auto last = end - 1;
        auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                                    begin);
        auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                                   last);
        for (unsigned g = beginIndices[0]; g != lastIndices[0] + 1; ++g) {
          unsigned batchBegin = g == beginIndices[0] ?
                                beginIndices[1] :
                                0;
          unsigned batchLast = g == lastIndices[0] ?
                               lastIndices[1] :
                               firstInGroup.dim(1) - 1;
          auto scaleWindow = scaleByGroup[g];
          for (unsigned b = batchBegin; b != batchLast + 1; ++b) {
            unsigned begin = g == beginIndices[0] && b == beginIndices[1] ?
                             beginIndices[2] :
                             0;
            unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                            lastIndices[2] :
                            firstInGroup.dim(2) - 1;
            auto actsWindow =
                acts[g][b].flatten().slice(begin * outChansPerGroup,
                                           (last + 1) * outChansPerGroup);
            auto actsScaledWindow =
                actsScaled[g][b].flatten().slice(begin * outChansPerGroup,
                                                (last + 1) * outChansPerGroup);
            graph.connect(v["actsIn"][num], actsWindow);
            graph.connect(v["actsOut"][num], actsScaledWindow);
            graph.connect(v["scale"][num], scaleWindow);
            ++num;
          }
        }
      }
      graph.setFieldSize(v["actsIn"], num);
      graph.setFieldSize(v["actsOut"], num);
      graph.setFieldSize(v["scale"], num);
    }
  }
  prog.add(Execute(cs));
  return actsScaledUngrouped;
}

static Tensor computeInvStdDev(Graph &graph, const Tensor &mean,
                               const Tensor &power, float eps,
                               Sequence &prog,
                               const Type &invStdDevType,
                               const std::string debugPrefix) {
  const auto meanType = mean.elementType();
  const auto powerType = power.elementType();
  auto iStdDev = graph.clone(invStdDevType, mean, debugPrefix + "/iStdDev");

  const auto meanFlat = mean.flatten();
  const auto powerFlat = power.flatten();
  const auto iStdDevFlat = iStdDev.flatten();

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/iStdDev");

  const auto mapping = graph.getTileMapping(iStdDev);
  const auto grainSize = target.getVectorWidth(invStdDevType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(iStdDevFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popconv::InverseStdDeviation",
                                              meanType, powerType,
                                              invStdDevType),
                               {{"mean", meanFlat.slices(regions)},
                                {"power", powerFlat.slices(regions)},
                                {"iStdDev", iStdDevFlat.slices(regions)}});
      graph.setInitialValue(v["eps"], eps);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return iStdDev;
}

std::pair<Tensor, Tensor>
batchNormEstimates(Graph &graph,
                   const Tensor &acts,
                   float eps,
                   Sequence &prog,
                   const Type &partialsType,
                   const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/BN/estimates";
  assert(acts.rank() == 4);

  // mean and standard deviation have the same mapping as biases
  const auto actsShape = acts.shape();
  const auto numElements = acts.numElements() / acts.dim(1);
  const float scale = 1.0 / numElements;

  std::vector< ComputeSet> csVec;
  auto mean =
    batchNormReduce(graph, acts, scale, false, csVec, partialsType,
                    fnPrefix + "/mean");
  auto power =
    batchNormReduce(graph, acts, scale, true, csVec, partialsType,
                    fnPrefix + "/power");
  for (const auto &cs : csVec) {
    prog.add(Execute(cs));
  }

  auto iStdDev = computeInvStdDev(graph, mean, power, eps, prog,
                                  acts.elementType(), debugPrefix);

  return std::make_pair(mean, iStdDev);
}

std::pair<Tensor, Tensor>
createBatchNormParams(Graph &graph, const Tensor &acts) {
  // map beta and gamma the same way as biases
  auto gamma = createBiases(graph, acts, "gamma");
  auto beta = createBiases(graph, acts, "beta");
  return std::make_pair(gamma, beta);
}

std::pair<Tensor, Tensor>
batchNormalise(Graph &graph,
               const Tensor &acts_,
               const Tensor &gamma,
               const Tensor &beta,
               const Tensor &mean,
               const Tensor &iStdDev,
               Sequence &prog,
               const std::string &debugPrefix) {
  auto acts = acts_;
  assert(acts.rank() == 4);
  const auto fnPrefix = debugPrefix + "/BN/batchNormalise";
  auto actsZeroMean = duplicate(graph, acts, prog);
  addToChannel(graph, actsZeroMean, mean, -1.0, prog, fnPrefix + "/beta");
  auto actsWhitened =
    channelMul(graph, actsZeroMean, iStdDev, prog, fnPrefix + "/istdDev");
  auto actsOut =
    channelMul(graph, actsWhitened, gamma, prog, fnPrefix + "/gamma");
  addToChannel(graph, actsOut, beta, 1.0, prog, fnPrefix + "/beta");
  return std::make_pair(actsOut, actsWhitened);
}

Tensor
batchNormalise(Graph &graph,
               const Tensor &acts,
               const Tensor &combinedMultiplicand,
               const Tensor &addend,
               Sequence &prog,
               const std::string &debugPrefix) {
  assert(acts.rank() == 4);
  const auto fnPrefix = debugPrefix + "/BN/batchNormaliseInference";
  auto actsBN = channelMul(graph, acts, combinedMultiplicand, prog,
                           fnPrefix + "/combinedMult");
  addToChannel(graph, actsBN, addend, 1.0, prog, fnPrefix + "/combinedAdd");
  return actsBN;
}

std::pair<Tensor, Tensor>
batchNormDeltas(Graph &graph,
                const Tensor &actsWhitened,
                const Tensor &gradsIn,
                Sequence &prog,
                const Type &partialsType,
                const std::string &debugPrefix) {

  const auto fnPrefix = debugPrefix + "/BN/deltas";
  const auto gradsInMultActs =
    mul(graph, gradsIn, actsWhitened, prog, fnPrefix);

  std::vector< ComputeSet> csVec = {};
  auto numChannels = gradsInMultActs.dim(1);
  const auto concatDeltas =
    batchNormReduce(graph, concat({gradsInMultActs, gradsIn}, 1), 1.0,
                    false, csVec, partialsType, fnPrefix + "/JointGammaDelta");
  for (const auto &cs : csVec) {
    prog.add(Execute(cs));
  }
  return std::make_pair(concatDeltas.slice(0, numChannels),
                        concatDeltas.slice(numChannels, 2 * numChannels));
}

Tensor batchNormGradients(Graph &graph,
                          const Tensor &actsWhitened,
                          const Tensor &gradsIn,
                          const Tensor &gammaDelta,
                          const Tensor &betaDelta,
                          const Tensor &invStdDev,
                          const Tensor &gamma,
                          Sequence &prog,
                          const Type &partialsType,
                          const std::string &debugPrefix) {
  assert(actsWhitened.rank() == 4);
  const auto fnPrefix = debugPrefix + "/BN/gradients";
  const auto actsShape = actsWhitened.shape();
  const auto numElements = actsWhitened.numElements() / actsWhitened.dim(1);
  const float rScale = 1.0 / numElements;

  auto gradient = graph.clone(actsWhitened);
  prog.add(Copy(gradsIn, gradient));
  addTo(graph, gradient,
        channelMul(graph, actsWhitened, gammaDelta, prog, fnPrefix),
        -rScale, prog, fnPrefix + "/gamma");

  addToChannel(graph, gradient, betaDelta, -rScale, prog, fnPrefix);

  return channelMul(graph, gradient,
                    mul(graph, gamma, invStdDev, prog,
                        fnPrefix + "/gamma_x_delta"), prog, fnPrefix);
}

} // namespace conv
