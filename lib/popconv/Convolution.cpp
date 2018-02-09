#include "popconv/Convolution.hpp"
#include "popconv/internal/ConvPlan.hpp"
#include <limits>
#include <algorithm>
#include <boost/optional.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include "popconv/ConvUtil.hpp"
#include "popstd/Pad.hpp"
#include "popstd/Add.hpp"
#include "popstd/TileMapping.hpp"
#include "popreduce/Reduce.hpp"
#include "popstd/VertexTemplates.hpp"
#include "util/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexOptim.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/Cast.hpp"
#include "popstd/Util.hpp"
#include "Winograd.hpp"
#include "popstd/Zero.hpp"
#include "popstd/Operations.hpp"
#include <unordered_set>
#include "util/Compiler.hpp"
#include "util/print.hpp"
#include "util/VectorUtils.hpp"
#include <boost/icl/interval_map.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

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
    throw popstd::poplib_error("Number of kernel field dimensions does not"
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
      throw popstd::poplib_error(std::string("Number of ") + entry.second +
                                 " dimensions does not match the number of "
                                 "field dimensions");
    }
  }
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (inputTransform.truncationLower[dim] +
        inputTransform.truncationUpper[dim] >
        inputFieldShape[dim]) {
      throw popstd::poplib_error("Truncation for dimension " +
                                 std::to_string(dim) +
                                 " truncates by more than the size of the "
                                 "field");
    }
    if (kernelTransform.truncationLower[dim] +
        kernelTransform.truncationUpper[dim] >
        kernelShape[dim]) {
      throw popstd::poplib_error("Truncation for dimension " +
                                 std::to_string(dim) +
                                 " truncates by more than the size of the "
                                 "kernel");
    }
    const auto convOutSize = getUntransformedOutputSize(dim);
    if (outputTransform.truncationLower[dim] +
        outputTransform.truncationUpper[dim] >
        convOutSize) {
      throw popstd::poplib_error("Output truncation for dimension " +
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

static std::vector<unsigned>
unflattenIndexInSlice(const std::vector<unsigned> &sliceBegin,
                      const std::vector<unsigned> &sliceEnd,
                      std::size_t index) {
  const auto rank = sliceBegin.size();
  assert(sliceEnd.size() == rank);
  std::vector<unsigned> coord;
  for (int dim = rank - 1; dim >= 0; --dim) {
    const auto sliceSize = sliceEnd[dim] - sliceBegin[dim];
    coord.push_back(index % sliceSize + sliceBegin[dim]);
    index /= sliceSize;
  }
  assert(index == 0);
  std::reverse(coord.begin(), coord.end());
  return coord;
}

// Indices start from dimension 0 and may have lower rank than the slice
// dimensions.
static unsigned
flattenIndexInSlice(const std::vector<unsigned> &sliceBegin,
                    const std::vector<unsigned> &sliceEnd,
                    const std::vector<unsigned> &indices) {
  const auto rank = sliceBegin.size();
  assert(indices.size() > 0);
  assert(sliceEnd.size() == rank);
  assert(indices.size() <= rank);
  (void)rank;
  unsigned index = 0;
  for (unsigned i = 0; i != indices.size(); ++i) {
    assert(indices[i] >= sliceBegin[i]);
    assert(indices[i] < sliceEnd[i]);
    const auto sliceSize = sliceEnd[i] - sliceBegin[i];
    index = sliceSize * index + indices[i] - sliceBegin[i];
  }
  return index;
}

// Reshape the activations tensor from [N][G * C]... shape to
// [G][N]...[C].
static Tensor
actsToInternalShape(const Tensor &act, unsigned numConvGroups) {
  if (act.dim(1) % numConvGroups != 0) {
    throw popstd::poplib_error("Number of input channels is not a multiple "
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
    throw popstd::poplib_error("Input tensor does not have the expected rank");
  }
  if (weights.rank() != 3 + numFieldDims) {
    throw popstd::poplib_error("Weight tensor does not have the expected rank");
  }
  for (unsigned i = 0; i != numFieldDims; ++i) {
    if (params.inputFieldShape[i] != in.dim(2 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw popstd::poplib_error(dimName + " of input tensor does not match "
                                 "convolution parameters");
    }
    if (params.kernelShape[i] != weights.dim(1 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw popstd::poplib_error(dimName + " of kernel does not match "
                                 "convolution parameters");
    }
  }
  if (params.numConvGroups != in.dim(0)) {
    throw popstd::poplib_error("Number of convolution groups of input tensor "
                               "does not match convolution parameters");
  }
  if (params.getBatchSize() != in.dim(1)) {
    throw popstd::poplib_error("Batchsize of input tensor does not match "
                               "convolution parameters");
  }
  if (params.getNumInputChansPerConvGroup() != in.dim(in.rank() - 1)) {
    throw popstd::poplib_error("Number of channels per convolution group of "
                               "input tensor does not match convolution "
                               "parameters");
  }
  if (params.numConvGroups != weights.dim(0)) {
    throw popstd::poplib_error("Number of convolution groups of weights tensor "
                               "does not match convolution parameters");
  }
  if (params.getNumOutputChansPerConvGroup() !=
      weights.dim(weights.rank() - 2)) {
    throw popstd::poplib_error("Kernel output channel size does not match "
                               "convolution parameters");
  }
  if (params.getNumInputChansPerConvGroup() !=
      weights.dim(weights.rank() - 1)) {
    throw popstd::poplib_error("Kernel input channel size does not match "
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

static unsigned
getFlattenedIndex(const std::vector<std::size_t> &shape,
                  const std::vector<std::size_t> &indices) {
  const auto rank = shape.size();
  assert(indices.size() == rank);
  unsigned index = 0;
  for (unsigned dim = 0; dim != rank; ++dim) {
    assert(indices[dim] < shape[dim]);
    index *= shape[dim];
    index += indices[dim];
  }
  return index;
}

static void
addFlattenedRegions(const std::vector<std::size_t> &shape,
                    const std::vector<std::size_t> &begin,
                    const std::vector<std::size_t> &end,
                    std::vector<Interval> &regions) {
  const auto numDims = shape.size();
  assert(begin.size() == numDims);
  assert(end.size() == numDims);

  for (unsigned dim = 0; dim != numDims; ++dim) {
    if (begin[dim] == end[dim])
      return;
  }

  std::vector<std::size_t> indices = begin;
  bool done = false;
  while (!done) {
    unsigned regionBegin = getFlattenedIndex(shape, indices);
    unsigned regionEnd = regionBegin + (end.back() - begin.back());
    if (!regions.empty() &&
        regions.back().end() == regionBegin) {
      regions.back() = Interval(regions.back().begin(), regionEnd);
    } else {
      regions.emplace_back(regionBegin, regionEnd);
    }
    done = true;
    for (int dim = numDims - 2; dim >= 0; --dim) {
      if (indices[dim] + 1 == end[dim]) {
        indices[dim] = begin[dim];
      } else {
        ++indices[dim];
        done = false;
        break;
      }
    }
  }
}

static std::pair<unsigned, unsigned>
getTileOutRange(const ConvSlice &parentSlice, const Partition &partition,
                unsigned tileIndex, unsigned dim) {
  const auto outSize = parentSlice.getOutputSize(dim);
  const auto outOffset = parentSlice.outFieldBegin[dim];
  const auto grainSize = partition.fieldAxisGrainSize[dim];
  const auto numGrains = (outSize + grainSize - 1) / grainSize;
  const auto split = partition.fieldSplit[dim];
  const auto outGrainBegin = (tileIndex * numGrains) / split;
  const auto outGrainEnd = ((tileIndex + 1) * numGrains) / split;
  const auto outBegin = outOffset + outGrainBegin * grainSize;
  const auto outEnd = outOffset + std::min(outGrainEnd * grainSize, outSize);
  return {outBegin, outEnd};
}

static void
iteratePartition(const ConvSlice &parentSlice,
                 const Partition &partition,
                     const std::function<
                       void(const ConvIndices &,
                            const ConvSlice &)
                     > &f) {
  const auto numFieldDims = parentSlice.getNumFieldDims();
  const unsigned numOutChans = parentSlice.getNumOutputChans();
  const auto outChanGrainSize = partition.outChanGrainSize;
  const auto outChanNumGrains = (numOutChans + outChanGrainSize - 1) /
                                outChanGrainSize;
  const auto batchSplit = partition.batchSplit;
  const auto outChanSplit = partition.outChanSplit;
  const auto inChanSplit = partition.inChanSplit;
  const unsigned batchSize = parentSlice.getBatchSize();
  const unsigned numInChans = parentSlice.getNumInputChans();
  const auto inChanGrainSize = partition.inChanGrainSize;
  const auto inChanNumGrains = (numInChans + inChanGrainSize - 1) /
                               inChanGrainSize;
  const auto convGroupSplit = partition.convGroupSplit;
  const unsigned numConvGroups = parentSlice.getNumConvGroups();
  const auto totalFieldSplit = product(partition.fieldSplit);
  const auto totalKernelSplit = product(partition.kernelSplit);
  for (unsigned cg = 0; cg != convGroupSplit; ++cg) {
    const auto cgBegin = (cg * numConvGroups) / convGroupSplit;
    const auto cgEnd = ((cg + 1) * numConvGroups) / convGroupSplit;
    for (unsigned b = 0; b != batchSplit; ++b) {
      const auto batchBegin = parentSlice.batchBegin +
                              (b * batchSize) / batchSplit;
      const auto batchEnd = parentSlice.batchBegin +
                            ((b + 1) * batchSize) / batchSplit;
      for (unsigned ic = 0; ic != inChanSplit; ++ic) {
        const auto inChanGrainBegin = (ic * inChanNumGrains) / inChanSplit;
        const auto inChanGrainEnd = ((ic + 1) * inChanNumGrains) /
                                    inChanSplit;
        const auto inChanBegin = parentSlice.inChanBegin +
                                 inChanGrainBegin * inChanGrainSize;
        const auto inChanEnd = parentSlice.inChanBegin +
                               std::min(inChanGrainEnd * inChanGrainSize,
                                        numInChans);
        for (unsigned k = 0; k != totalKernelSplit; ++k) {
          auto kernelIndices = unflattenIndex(partition.kernelSplit, k);
          std::vector<unsigned> kernelBegin(numFieldDims),
                                kernelEnd(numFieldDims);
          for (unsigned dim = 0; dim != numFieldDims; ++dim) {
            const auto kernelSize = parentSlice.getKernelSize(dim);
            const auto kernelOffset = parentSlice.kernelBegin[dim];
            kernelBegin[dim] = kernelOffset +
                               (kernelIndices[dim] * kernelSize) /
                               partition.kernelSplit[dim];
            kernelEnd[dim] = kernelOffset +
                             ((kernelIndices[dim] + 1) * kernelSize) /
                             partition.kernelSplit[dim];
          }
          for (unsigned oc = 0; oc != outChanSplit; ++oc) {
            const auto outChanGrainBegin = (oc * outChanNumGrains) /
                                           outChanSplit;
            const auto outChanGrainEnd = ((oc + 1) * outChanNumGrains) /
                                         outChanSplit;
            const auto outChanBegin = parentSlice.outChanBegin +
                                      outChanGrainBegin * outChanGrainSize;
            const auto outChanEnd = parentSlice.outChanBegin +
                                    std::min(outChanGrainEnd * outChanGrainSize,
                                             numOutChans);
            for (unsigned of = 0; of != totalFieldSplit; ++of) {
              auto outIndices = unflattenIndex(partition.fieldSplit, of);
              std::vector<unsigned> outFieldBegin(numFieldDims),
                                    outFieldEnd(numFieldDims);
              for (unsigned dim = 0; dim != numFieldDims; ++dim) {
                std::tie(outFieldBegin[dim], outFieldEnd[dim]) =
                    getTileOutRange(parentSlice, partition, outIndices[dim],
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

static void
iteratePartitionsImpl(const ConvSlice &parentSlice,
                      std::vector<Partition>::const_iterator begin,
                      std::vector<Partition>::const_iterator end,
                      const std::vector<ConvIndices> &parentIndices,
                      const std::function<
                        void(const std::vector<ConvIndices> &,
                             const ConvSlice &)
                      > &f) {
  if (begin == end) {
    f(parentIndices, parentSlice);
    return;
  }
  auto indices = parentIndices;
  indices.emplace_back();
  iteratePartition(parentSlice, *begin, [&](const ConvIndices &sliceIndices,
                                            const ConvSlice &slice) {
    indices.back() = sliceIndices;
    iteratePartitionsImpl(slice, std::next(begin), end, indices, f);
  });
}

static void
iteratePartitions(const ConvSlice &parentSlice,
                  const std::vector<Partition> &partitions,
                  const std::function<
                    void(const std::vector<ConvIndices> &,
                         const ConvSlice &)
                  > &f) {
  std::vector<ConvIndices> indices;
  iteratePartitionsImpl(parentSlice, partitions.begin(), partitions.end(),
                        indices, f);
}

static ConvSlice getWholeConvSlice(const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> zeros(numFieldDims);
  ConvSlice slice = {
    0, static_cast<unsigned>(params.getNumConvGroups()),
    0, static_cast<unsigned>(params.getBatchSize()),
    zeros, vectorConvert<unsigned>(params.getOutputFieldShape()),
    0, static_cast<unsigned>(params.getNumOutputChansPerConvGroup()),
    0, static_cast<unsigned>(params.getNumInputChansPerConvGroup()),
    zeros, vectorConvert<unsigned>(params.kernelShape)
  };
  return slice;
}

static void
iterateTilePartition(const Graph &graph, const ConvParams &params,
                     const Plan &plan,
                     const std::function<
                       void(unsigned,
                            const std::vector<ConvIndices> &,
                            const ConvSlice &)
                     > &f) {
  ConvSlice parentSlice = getWholeConvSlice(params);
  iteratePartitions(parentSlice, plan.partitions,
                    [&](const std::vector<ConvIndices> &indices,
                        const ConvSlice &slice) {
    const auto tile = linearizeTileIndices(graph.getTarget(), indices, plan);
    f(tile, indices, slice);
  });
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
    return popstd::calcLinearTileMapping(graph, shape,
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

static void
addFlattenedPrevActsRegions(const std::vector<std::size_t> &actsShape,
                            const ConvSlice &slice,
                            const ConvParams &params,
                            std::vector<Interval> &regions) {
  assert(actsShape.size() >= 4);
  const auto numFieldDims = actsShape.size() - 4;
  assert(slice.outFieldBegin.size() == numFieldDims);
  assert(slice.outFieldEnd.size() == numFieldDims);
  assert(slice.kernelBegin.size() == numFieldDims);
  assert(slice.kernelEnd.size() == numFieldDims);
  const auto inChansPerGroup = actsShape.back();
  assert(slice.inChanBegin % inChansPerGroup == 0);
  std::vector<std::size_t> sliceBegin = {
    slice.cgBegin,
    slice.inChanBegin / inChansPerGroup,
    slice.batchBegin
  };
  assert(slice.inChanEnd % inChansPerGroup == 0);
  std::vector<std::size_t> sliceEnd = {
    slice.cgEnd,
    slice.inChanEnd / inChansPerGroup,
    slice.batchEnd
  };
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto inRange =
        getInputRange(dim, {slice.outFieldBegin[dim], slice.outFieldEnd[dim]},
                      {slice.kernelBegin[dim], slice.kernelEnd[dim]}, params);
    sliceBegin.push_back(inRange.first);
    sliceEnd.push_back(inRange.second);
  }
  sliceBegin.push_back(0);
  sliceEnd.push_back(actsShape.back());
  addFlattenedRegions(actsShape, sliceBegin, sliceEnd, regions);
}

static std::vector<std::vector<Interval>>
calculateActivationMapping(Graph &graph, const ConvParams &params,
                           const Plan &plan) {
  // Build a map from activations to the set of tiles that access them.
  const auto numInChans = params.getNumInputChansPerConvGroup();
  assert(numInChans % plan.inChansPerGroup == 0);
  const auto numInChanGroups = numInChans / plan.inChansPerGroup;
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<std::size_t> actsShape = {
    params.getNumConvGroups(),
    numInChanGroups,
    params.getBatchSize()
  };
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    actsShape.push_back(params.getInputSize(dim));
  }
  actsShape.push_back(plan.inChansPerGroup);
  const auto numTiles = graph.getTarget().getNumTiles();
  std::vector<boost::icl::interval_set<unsigned>> used(numTiles);
  boost::icl::interval_map<unsigned, std::set<unsigned>> actsToTiles;
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const std::vector<ConvIndices> &,
                           const ConvSlice &slice) {
    std::vector<Interval> intervals;
    addFlattenedPrevActsRegions(actsShape, slice, params, intervals);
    auto &useSet = used[tile];
    for (const auto &interval : intervals) {
      useSet.add(toIclInterval(interval));
    }
  });
  for (unsigned tile = 0; tile < numTiles; ++tile) {
     std::set<unsigned> tileSet{tile};
     for (const auto &region : used[tile]) {
        actsToTiles.add(std::make_pair(region, tileSet));
     }
  }
  // Limit the minimum number of activation bytes per tile to reduce the amount
  // of exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto actType = params.dType;
  const auto actTypeSize = graph.getTarget().getTypeSize(actType);
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + actTypeSize - 1) / minBytesPerTile;
  const auto grainSize = plan.inChansPerGroup;
  return calculateMappingBasedOnUsage(graph, actsShape, actsToTiles,
                                      grainSize, minElementsPerTile);
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
  const auto dType = t.elementType();
  auto zeroShape = t.shape();
  zeroShape[dim] = 1;
  Tensor zero = graph.addConstant(dType, zeroShape, 0);
  std::vector<Tensor> slices;
  slices.reserve(newSize);
  for (unsigned i = 0; i != newSize; ++i) {
    if (i % dilationFactor == 0) {
      const auto oldIndex = i / dilationFactor;
      slices.push_back(t.slice(oldIndex, oldIndex + 1, dim));
    } else {
      slices.push_back(zero);
    }
  }
  return concat(slices, dim);
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

static void swapOperands(ConvParams &params, Tensor *acts, Tensor *weights) {
  swapOperands(params);
  if (acts && weights) {
    std::swap(*acts, *weights);
    *acts = acts->dimRoll(acts->rank() - 2, 1);
    *weights = weights->dimRoll(1, weights->rank() - 2);
  } else {
    assert(!acts && !weights);
  }
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
convolutionPreprocess(Graph &graph, ConvParams &params, Plan &plan,
                      Tensor *acts, Tensor *weights) {
  if (plan.extraFieldDims) {
    addExtraDims(params, plan.extraFieldDims);
    if (acts) {
      *acts = acts->expand(
                std::vector<std::size_t>(plan.extraFieldDims, 2)
              );
    }
    if (weights) {
      *weights = weights->expand(
                   std::vector<std::size_t>(plan.extraFieldDims, 1)
                 );
    }
    plan.extraFieldDims = 0;
  }
  if (plan.swapOperands) {
    swapOperands(params, acts, weights);
    plan.swapOperands = false;
  }
  for (auto dim : plan.expandDims) {
    expandSpatialDim(graph, params, acts, weights, dim);
  }
  plan.expandDims.clear();
  if (!plan.outChanFlattenDims.empty()) {
    boost::optional<Tensor> maybeActs, maybeWeights;
    if (acts)
      maybeActs.reset(*acts);
    if (weights)
      maybeWeights.reset(*weights);
    swapOperands(params, maybeActs, maybeWeights);
    for (auto dim : plan.outChanFlattenDims) {
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
    plan.outChanFlattenDims.clear();
  }
  if (!plan.flattenDims.empty()) {
    for (auto it = std::next(plan.flattenDims.rbegin()),
         end = plan.flattenDims.rend(); it != end; ++it) {
      const auto fromDimIndex = *it;
      const auto toDimIndex = plan.flattenDims.back();
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
  plan.flattenDims.clear();
  const auto numInChans = params.getNumInputChansPerConvGroup();
  const auto convInChansPerGroup = plan.inChansPerGroup;
  const auto convNumChanGroups =
      (numInChans + convInChansPerGroup - 1) / convInChansPerGroup;
  const auto convNumChans = convInChansPerGroup * convNumChanGroups;

  if (convNumChans != numInChans) {
    // Zero pad the input / weights.
    if (acts) {
      *acts = pad(graph, *acts, 0, convNumChans - numInChans, acts->rank() - 1);
    }
    if (weights) {
      *weights = pad(graph, *weights, 0, convNumChans - numInChans,
                     weights->rank() - 1);
    }
    params.inputChannels = convNumChans;
  }
  const auto outNumChans = params.getNumOutputChansPerConvGroup();
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto partialNumChanGroups =
      (outNumChans + partialChansPerGroup - 1) / partialChansPerGroup;
  const auto partialNumChans = partialNumChanGroups * partialChansPerGroup;
  if (partialNumChans != outNumChans) {
    if (weights) {
      *weights = pad(graph, *weights, 0, partialNumChans - outNumChans,
                     weights->rank() - 2);
    }
    params.outputChannels = partialNumChans;
  }
}

/// Map the activations tensor such that the exchange required during the
/// convolution operation is minimized.
static void mapActivations(Graph &graph, ConvParams params,
                           Plan plan, Tensor acts) {
  // Depending on the plan the input may be transformed prior to the
  // convolution. Some activations may not be present in the transformed view
  // (for example if they are not used in the calculation due to negative
  // padding). Map the activations tensor before it is transformed so
  // activations that don't appear in the transformed view are mapped to a tile.
  mapTensorLinearly(graph, acts);
  // Apply the transform to the activations.
  convolutionPreprocess(graph, params, plan, &acts, nullptr);
  // Compute a mapping for the transformed activations tensor that minimizes
  // exchange.
  acts = splitActivationChanGroups(acts, plan.inChansPerGroup);
  auto mapping = calculateActivationMapping(graph, params, plan);
  // Apply the mapping to the transformed activations tensor. This indirectly
  // maps the original (non-transformed) tensor.
  graph.setTileMapping(acts, mapping);
}

static Tensor
createWeightsImpl(Graph &graph,
                  const ConvParams &params, const std::string &name,
                  const Plan &plan);

static Tensor
createInputImpl(Graph &graph, const ConvParams &params,
                const std::string &name,
                const Plan &plan) {
  if (plan.swapOperands) {
    auto newParams = params;
    auto newPlan = plan;
    swapOperands(newParams);
    newPlan.swapOperands = false;
    auto t = createWeightsImpl(graph, newParams, name, newPlan);
    return t.dimRoll(t.rank() - 2, 1);
  }
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
  t = unsplitActivationChanGroups(t);
  mapActivations(graph, params, plan, t);
  return t;
}

Tensor
createInput(Graph &graph, const ConvParams &params,
            const std::string &name,
            const ConvOptions &options) {
  const auto plan = getPlan(graph, params, options);
  auto input = createInputImpl(graph, params, name, plan);
  return actsToExternalShape(input);
}

static std::vector<std::vector<Interval>>
calculateWeightMapping(const Graph &graph,
                       const ConvParams &params,
                       const Plan &plan) {
  // Build a map from weights to the set of tiles that access them.
  const auto numInChans = params.getNumInputChansPerConvGroup();
  assert(numInChans % plan.inChansPerGroup == 0);
  const auto numInChanGroups = numInChans / plan.inChansPerGroup;
  const auto numOutChans = params.getNumOutputChansPerConvGroup();
  assert(numOutChans % plan.partialChansPerGroup == 0);
  const auto numOutChanGroups = numOutChans / plan.partialChansPerGroup;
  std::vector<std::size_t> weightsShape = {
    params.getNumConvGroups(),
    numOutChanGroups,
    numInChanGroups
  };
  weightsShape.insert(weightsShape.end(), params.kernelShape.begin(),
                      params.kernelShape.end());
  weightsShape.push_back(plan.partialChansPerGroup);
  weightsShape.push_back(plan.inChansPerGroup);
  boost::icl::interval_map<unsigned, std::set<unsigned>> weightsToTiles;
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const std::vector<ConvIndices> &,
                           const ConvSlice &slice) {
    std::vector<Interval> intervals;
    assert(slice.outChanBegin % plan.partialChansPerGroup == 0);
    assert(slice.outChanEnd % plan.partialChansPerGroup == 0);
    assert(slice.inChanBegin % plan.inChansPerGroup == 0);
    assert(slice.inChanEnd % plan.inChansPerGroup == 0);
    std::vector<std::size_t> sliceBegin = {
      slice.cgBegin,
      slice.outChanBegin / plan.partialChansPerGroup,
      slice.inChanBegin / plan.inChansPerGroup
    };
    sliceBegin.insert(sliceBegin.end(), slice.kernelBegin.begin(),
                      slice.kernelBegin.end());
    sliceBegin.push_back(0);
    sliceBegin.push_back(0);
    std::vector<std::size_t> sliceEnd = {
      slice.cgEnd,
      slice.outChanEnd / plan.partialChansPerGroup,
      slice.inChanEnd / plan.inChansPerGroup
    };
    sliceEnd.insert(sliceEnd.end(), slice.kernelEnd.begin(),
                    slice.kernelEnd.end());
    sliceEnd.push_back(plan.partialChansPerGroup);
    sliceEnd.push_back(plan.inChansPerGroup);
    addFlattenedRegions(weightsShape, sliceBegin, sliceEnd, intervals);
    for (const auto &interval : intervals) {
      weightsToTiles += std::make_pair(toIclInterval(interval),
                                       std::set<unsigned>({tile}));
    }
  });
  // Limit the minimum number of weight bytes per tile to reduce the
  // amount of exchange code. Increasing this constant reduces exchange code
  // size and increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto weightType = params.dType;
  const auto weightTypeSize = graph.getTarget().getTypeSize(weightType);
  const auto minBytesPerTile = 256;
  const auto minElementsPerTile =
    (minBytesPerTile + weightTypeSize - 1) / minBytesPerTile;
  unsigned grainSize = plan.partialChansPerGroup * plan.inChansPerGroup;
  return calculateMappingBasedOnUsage(graph, weightsShape, weightsToTiles,
                                      grainSize, minElementsPerTile);
}

static void mapWeights(Graph &graph, Tensor weights,
                       ConvParams params, Plan plan) {
  // Depending on the plan the weights may be transformed prior to the
  // convolution. Some weights may not be present in the transformed view
  // (for example if they are not used in the calculation due to negative
  // padding). Map the weights tensor before it is transformed so
  // weights that don't appear in the transformed view are mapped to a tile.
  mapTensorLinearly(graph, weights);
  convolutionPreprocess(graph, params, plan, nullptr, &weights);
  weights = groupWeights(weights, plan.inChansPerGroup,
                         plan.partialChansPerGroup);
  // Compute a mapping for the transformed weights tensor that minimizes
  // exchange.
  auto weightsMapping = calculateWeightMapping(graph, params, plan);
  // Apply the mapping to the transformed weights tensor. This indirectly
  // maps the original (non-transformed) tensor.
  graph.setTileMapping(weights, weightsMapping);
}

static Tensor
createWeightsImpl(Graph &graph,
                  const ConvParams &params, const std::string &name,
                  const Plan &plan) {
  if (plan.swapOperands) {
    auto newParams = params;
    auto newPlan = plan;
    swapOperands(newParams);
    newPlan.swapOperands = false;
    auto t = createInputImpl(graph, newParams, name, newPlan);
    return t.dimRoll(1, t.rank() - 2);
  }
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
  weights = ungroupWeights(weights);
  mapWeights(graph, weights, params, plan);
  return weights;
}

Tensor
createWeights(Graph &graph,
              const ConvParams &params, const std::string &name,
              const ConvOptions &options) {
  const auto plan = getPlan(graph, params, options);
  return weightsToExternalShape(createWeightsImpl(graph, params, name, plan));
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


static bool writtenRangeEqualsOutputRange(
    unsigned dim,
    std::pair<unsigned, unsigned> outRange,
    std::pair<unsigned, unsigned> kernelIndexRange,
    const ConvParams &params) {
  auto writtenRange =
      getOutputRangeForKernelRange(dim, outRange, kernelIndexRange, params);
  return writtenRange == outRange;
}

static bool writtenRangeEqualsOutputRange(
    const std::vector<unsigned> &outRangeBegin,
    const std::vector<unsigned> &outRangeEnd,
    const std::vector<unsigned> &kernelRangeBegin,
    const std::vector<unsigned> &kernelRangeEnd,
    const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (!writtenRangeEqualsOutputRange(
             dim, {outRangeBegin[dim], outRangeEnd[dim]},
             {kernelRangeBegin[dim], kernelRangeEnd[dim]},
             params
         )) {
      return false;
    }
  }
  return true;
}

static void createConvPartialAmpVertex(Graph &graph,
                                       const Plan &plan,
                                       const Type &dType,
                                       unsigned tile,
                                       const ConvSlice &slice,
                                       ConvParams params,
                                       ComputeSet fwdCS,
                                       Tensor in, Tensor weights, Tensor out) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto &target = graph.getTarget();
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto partialsType = out.elementType();
  const auto outChansPerPass = getNumConvUnits(dType == FLOAT,
                                               partialsType == FLOAT,
                                               target);
  assert(outChansPerGroup % outChansPerPass == 0);
  const auto passesPerOutputGroup = outChansPerGroup / outChansPerPass;
  bool nx1Vertex =
      getNumElementsInSlice(slice.kernelBegin, slice.kernelEnd) != 1 ||
      params.inputTransform.dilation != params.outputTransform.stride ||
      !writtenRangeEqualsOutputRange(slice.outFieldBegin, slice.outFieldEnd,
                                     slice.kernelBegin, slice.kernelEnd,
                                     params);
  bool useConvPartial1x1OutVertex = !nx1Vertex &&
                                    passesPerOutputGroup == 1;
  const auto dataPathWidth = target.getDataPathWidth();
  const bool floatActivations = dType == FLOAT;
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(floatActivations);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  const auto batchBegin = slice.batchBegin;
  const auto batchEnd = slice.batchEnd;
  assert(slice.outChanBegin % plan.partialChansPerGroup == 0);
  assert(slice.outChanEnd % plan.partialChansPerGroup == 0);
  const auto outZGroupBegin = slice.outChanBegin / plan.partialChansPerGroup;
  const auto outZGroupEnd = slice.outChanEnd / plan.partialChansPerGroup;
  const auto cgBegin = slice.cgBegin;
  const auto cgEnd = slice.cgEnd;
  assert(slice.inChanBegin % plan.inChansPerGroup == 0);
  assert(slice.inChanEnd % plan.inChansPerGroup == 0);
  const auto inZGroupBegin = slice.inChanBegin / plan.inChansPerGroup;
  const auto inZGroupEnd = slice.inChanEnd / plan.inChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;
  bool flipOut = params.inputTransform.flip[numFieldDims - 1];
  const auto convUnitInputLoadElemsPerCycle =
    target.getConvUnitInputLoadElemsPerCycle(dType == FLOAT);
  const auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();

  std::vector<Tensor> outWindow;
  std::vector<std::size_t> tileOutBatchAndFieldBegin = { batchBegin };
  tileOutBatchAndFieldBegin.insert(tileOutBatchAndFieldBegin.end(),
                                   slice.outFieldBegin.begin(),
                                   slice.outFieldBegin.end());
  std::vector<std::size_t> tileOutBatchAndFieldEnd = { batchEnd };
  tileOutBatchAndFieldEnd.insert(tileOutBatchAndFieldEnd.end(),
                                 slice.outFieldEnd.begin(),
                                 slice.outFieldEnd.end());
  for (unsigned cg = cgBegin; cg < cgEnd; ++cg) {
    for (unsigned ozg = outZGroupBegin; ozg < outZGroupEnd; ++ozg) {
      auto o = out[cg][ozg].slice(tileOutBatchAndFieldBegin,
                                  tileOutBatchAndFieldEnd)
                           .flatten();
      graph.setTileMapping(o, tile);
      outWindow.push_back(o);
    }
  }
  // Explicitly dilate the input tensor if needed
  if (convUnitWeightHeight != 1 && params.inputTransform.dilation[0] > 1) {
    const auto inputDilation = params.inputTransform.dilation[0];
    const auto inputTruncationLower = params.inputTransform.truncationLower[0];
    const auto inputTruncationUpper = params.inputTransform.truncationUpper[0];
    in = pad(graph, in, -static_cast<int>(inputTruncationLower),
             -static_cast<int>(inputTruncationUpper), 3);
    params.inputFieldShape[0] -= inputTruncationLower + inputTruncationUpper;
    params.inputTransform.truncationLower[0] = 0;
    params.inputTransform.truncationUpper[0] = 0;
    in = dilate(graph, in, inputDilation, 3);
    params.inputTransform.dilation[0] = 1;
    params.inputFieldShape[0] = getDilatedSize(params.inputFieldShape[0],
                                               inputDilation);
  }
  std::vector<std::size_t> tileInBatchAndFieldBegin = { batchBegin };
  std::vector<std::size_t> tileInBatchAndFieldEnd = { batchEnd };
  // This stride is what's used to move down one element in the input field by
  // the vertex.
  int inRowStride = inChansPerGroup * params.kernelTransform.dilation.front();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto range = getInputRange(dim,
                               {slice.outFieldBegin[dim],
                                slice.outFieldEnd[dim]},
                               {slice.kernelBegin[dim],
                                slice.kernelEnd[dim]},
                               params);
    tileInBatchAndFieldBegin.push_back(range.first);
    tileInBatchAndFieldEnd.push_back(range.second);
    if (dim != 0)
      inRowStride *= range.second - range.first;
  }
  if (params.inputTransform.flip.front() !=
      params.kernelTransform.flip.front()) {
    inRowStride = -inRowStride;
  }

  unsigned prePaddingSize = 0;
  unsigned postPaddingSize = 0;
  if (convUnitWeightHeight != 1) {
    // Pad the input window so it contains, for each position of the amp sub
    // kernel where at least one kernel element is multiplied by a non padding
    // input element, all input padding elements multiplied by the amp sub
    // kernel in that position. This avoids special handling in the vertex for
    // the beginning or end of the field. Compute the amount of padding required
    // by conservatively assuming the first non padding input element in the
    // window is multiplied by the last kernel element and the last non padding
    // input element in the window is multiplied by the first kernel element.
    prePaddingSize = (convUnitWeightHeight - 1) *
                     params.kernelTransform.dilation.front();
    postPaddingSize = prePaddingSize;
  }
  // TODO if the tile kernel size is 1 and the stride is greater than one we
  // could subsample the input instead of using input striding.
  std::vector<Tensor> inWindow;
  for (unsigned cg = cgBegin; cg < cgEnd; ++cg) {
    for (unsigned izg = inZGroupBegin; izg < inZGroupEnd; ++izg) {
      auto window = in[cg][izg].slice(tileInBatchAndFieldBegin,
                                      tileInBatchAndFieldEnd);
      window = pad(graph, window,
                   prePaddingSize,
                   postPaddingSize,
                   1);
      inWindow.push_back(window.flatten());
    }
  }
  tileInBatchAndFieldEnd[1] += prePaddingSize + postPaddingSize;

  std::vector<Tensor> weightsWindow;
  for (unsigned cg = cgBegin; cg < cgEnd; ++cg) {
    for (unsigned ozg = outZGroupBegin; ozg < outZGroupEnd; ++ozg) {
      for (unsigned izg = inZGroupBegin; izg < inZGroupEnd; ++izg) {
        auto window =
            weights[cg][ozg][izg].slice(
              vectorConvert<std::size_t>(slice.kernelBegin),
              vectorConvert<std::size_t>(slice.kernelEnd)
            );
        const auto kernelHeight = window.dim(0);
        if (kernelHeight % convUnitWeightHeight != 0) {
          // If we are doing an nx1 convolution need to pad the bottom of the
          // weights to round up to a multiple of n
          auto postPaddingSize =
               (convUnitWeightHeight - kernelHeight % convUnitWeightHeight);
          window = pad(graph, window, 0, postPaddingSize, 0);
        }
        weightsWindow.push_back(window.flatten());
      }
    }
  }

  const auto contextsPerVertex = target.getNumWorkerContexts();
  auto subKernelPositionsBegin = slice.kernelBegin;
  auto subKernelPositionsEnd = slice.kernelEnd;
  subKernelPositionsBegin[0] = 0;
  subKernelPositionsEnd[0] =
      (slice.kernelEnd[0] - slice.kernelBegin[0] + convUnitWeightHeight - 1) /
      convUnitWeightHeight;
  const auto numSubKernelPositions =
      getNumElementsInSlice(subKernelPositionsBegin,
                            subKernelPositionsEnd);
  std::vector<std::vector<unsigned>> worklist(contextsPerVertex *
                                              numSubKernelPositions);
  for (unsigned k = 0; k != numSubKernelPositions; ++k) {
    auto kernelBeginIndices = unflattenIndexInSlice(subKernelPositionsBegin,
                                                    subKernelPositionsEnd,
                                                    k);
    kernelBeginIndices[0] = kernelBeginIndices[0] * convUnitWeightHeight +
                            slice.kernelBegin[0];
    std::vector<unsigned> tileConvOutBegin;
    std::vector<unsigned> tileConvOutSize;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      const auto kernelBeginIndex = kernelBeginIndices[dim];
      const auto kernelEndIndex =
          std::min(kernelBeginIndex + (dim == 0 ? convUnitWeightHeight : 1),
                   slice.kernelEnd[dim]);
      auto convOutRange =
          getOutputRangeForKernelRange(dim, {slice.outFieldBegin[dim],
                                             slice.outFieldEnd[dim]},
                                       {kernelBeginIndex, kernelEndIndex},
                                       params);
      tileConvOutBegin.push_back(convOutRange.first);
      tileConvOutSize.push_back(convOutRange.second - convOutRange.first);
    }
    if (product(tileConvOutSize) == 0)
      continue;
    auto workerPartition =
        partitionConvPartialByWorker(batchEnd - batchBegin,
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
        std::vector<unsigned> outBeginIndices = { partialRow.b + batchBegin };
        for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
          outBeginIndices.push_back(partialRow.outerFieldIndices[dim] +
                                    tileConvOutBegin[dim]);
        }
        outBeginIndices.push_back(partialRow.xBegin + tileConvOutBegin.back());
        const auto outBeginOffset =
            flattenIndexInSlice(
              vectorConvert<unsigned>(tileOutBatchAndFieldBegin),
              vectorConvert<unsigned>(tileOutBatchAndFieldEnd),
              outBeginIndices);
        std::vector<unsigned> inBeginIndices = { partialRow.b + batchBegin };
        if (numFieldDims > 1) {
          const auto kOuterBegin = kernelBeginIndices[0];
          const auto kOuterEnd = std::min(kOuterBegin + convUnitWeightHeight,
                                          slice.kernelEnd[0]);
          const auto outOuterIndex = tileConvOutBegin[0] +
                                     partialRow.outerFieldIndices[0];
          for (unsigned k = kOuterBegin; k != kOuterEnd; ++k) {
            auto inOuterIndex = getInputIndex(0, outOuterIndex, k, params);
            if (inOuterIndex != ~0U) {
              auto inOuterBeginIndex =
                  inOuterIndex + prePaddingSize +
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
            flattenIndexInSlice(
              vectorConvert<unsigned>(tileInBatchAndFieldBegin),
              vectorConvert<unsigned>(tileInBatchAndFieldEnd),
              inBeginIndices
            );
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
                           templateVertex(codeletName, dType,
                                          plan.getPartialType(),
                                        useDeltaEdgesForConvPartials(numEdges) ?
                                                          "true" : "false"));
  unsigned kernelInnerElements = 1;
  for (unsigned dim = 1; dim < numFieldDims; ++dim) {
    kernelInnerElements *= subKernelPositionsEnd[dim] -
                           subKernelPositionsBegin[dim];
  }
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
  graph.setInitialValue(v["numOutGroups"], outZGroupEnd - outZGroupBegin);
  graph.setInitialValue(v["numInGroups"], inZGroupEnd - inZGroupBegin);
  graph.setInitialValue(v["inStride"], inStrideX) ;
  graph.setInitialValue(v["numConvGroups"], cgEnd - cgBegin);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["convUnitInputLoadElemsPerCycle"],
                        convUnitInputLoadElemsPerCycle);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  graph.setInitialValue(v["numWorkerContexts"], contextsPerVertex);
  graph.setInitialValue(v["flipOut"], flipOut);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstant(UNSIGNED_INT, {worklist[i].size()},
                               worklist[i].data());
    graph.connect(v["worklists"][i], t);
  }

  if (!useConvPartial1x1OutVertex) {
    graph.setInitialValue(v["kernelInnerElements"], kernelInnerElements);
    graph.setInitialValue(v["kernelOuterSize"], subKernelPositionsEnd[0]);
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
                                     const Type &dType,
                                     unsigned tile,
                                     const ConvSlice &slice,
                                     const ConvParams &params,
                                     ComputeSet fwdCS,
                                     const Tensor &in,
                                     const Tensor &weights,
                                     const Tensor &out) {
  const auto &target = graph.getTarget();
  const auto numFieldDims = params.getNumFieldDims();
  const auto xDimIndex = numFieldDims - 1;
  const auto batchBegin = slice.batchBegin;
  const auto batchEnd = slice.batchEnd;
  const auto outXBegin = slice.outFieldBegin.back();
  const auto outXEnd = slice.outFieldEnd.back();
  assert(slice.outChanBegin % plan.partialChansPerGroup == 0);
  assert(slice.outChanEnd % plan.partialChansPerGroup == 0);
  const auto outZGroupBegin = slice.outChanBegin / plan.partialChansPerGroup;
  const auto outZGroupEnd = slice.outChanEnd / plan.partialChansPerGroup;
  const auto cgBegin = slice.cgBegin;
  const auto cgEnd = slice.cgEnd;
  assert(slice.inChanBegin % plan.inChansPerGroup == 0);
  assert(slice.inChanEnd % plan.inChansPerGroup == 0);
  const auto inZGroupBegin = slice.inChanBegin / plan.inChansPerGroup;
  const auto inZGroupEnd = slice.inChanEnd / plan.inChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;

  const auto kernelXBegin = slice.kernelBegin.back();
  const auto kernelXEnd = slice.kernelEnd.back();

  bool flipOut = params.inputTransform.flip[xDimIndex];

  const auto dataPathWidth = target.getDataPathWidth();

  assert(outChansPerGroup == 1);
  if (dType == HALF) {
    assert(inChansPerGroup % 2 == 0);
  }
  const auto numOutElems =
      getNumElementsInSlice(slice.outFieldBegin, slice.outFieldEnd);
  if (!numOutElems)
    return;
  const auto outWidth = outXEnd - outXBegin;

  std::vector<Tensor> outWindow;
  // Output Tensor slices
  auto outSliceBegin = vectorConvert<std::size_t>(slice.outFieldBegin);
  auto outSliceEnd = vectorConvert<std::size_t>(slice.outFieldEnd);
  outSliceBegin.insert(outSliceBegin.begin(), batchBegin);
  outSliceEnd.insert(outSliceEnd.begin(), batchEnd);
  for (unsigned cg = cgBegin; cg != cgEnd; ++cg) {
    for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
      auto o = out[cg][ozg].slice(outSliceBegin, outSliceEnd).flatten();
      graph.setTileMapping(o, tile);
      outWindow.push_back(o);
    }
  }

  // Compute input field slices for output range
  std::vector<unsigned> inFieldBegin, inFieldEnd;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto inRange =
        getInputRange(dim, {slice.outFieldBegin[dim], slice.outFieldEnd[dim]},
                      {slice.kernelBegin[dim], slice.kernelEnd[dim]}, params);
    inFieldBegin.push_back(inRange.first);
    inFieldEnd.push_back(inRange.second);
  }

  const auto inXBegin = inFieldBegin.back();
  const auto inXEnd = inFieldEnd.back();
  const auto numInElems = getNumElementsInSlice(inFieldBegin, inFieldEnd);
  const auto inWidth = inXEnd - inXBegin;

  std::vector<Tensor> inWindow;
  // Input tensor slices
  auto inSliceBegin = vectorConvert<std::size_t>(inFieldBegin);
  auto inSliceEnd = vectorConvert<std::size_t>(inFieldEnd);
  inSliceBegin.insert(inSliceBegin.begin(), batchBegin);
  inSliceEnd.insert(inSliceEnd.begin(), batchEnd);
  for (unsigned cg = cgBegin; cg != cgEnd; ++cg) {
    for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
      auto i = in[cg][izg].slice(inSliceBegin, inSliceEnd).flatten();
      inWindow.push_back(i);
    }
  }

  // kernel tensor slices
  auto kernelSliceBegin = vectorConvert<std::size_t>(slice.kernelBegin);
  auto kernelSliceEnd = vectorConvert<std::size_t>(slice.kernelEnd);
  std::vector<Tensor> weightsWindow;
  for (unsigned cg = cgBegin; cg != cgEnd; ++cg) {
    for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
      for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
        auto w = weights[cg][ozg][izg].slice(kernelSliceBegin, kernelSliceEnd)
                                      .flatten();
        weightsWindow.push_back(w);
      }
    }
  }

  const auto numKernelElems = getNumElementsInSlice(slice.kernelBegin,
                                                    slice.kernelEnd);
  const auto kernelSizeX = kernelXEnd - kernelXBegin;
  const auto contextsPerVertex = target.getNumWorkerContexts();
  std::vector<std::vector<unsigned>> worklist(contextsPerVertex
                                              * numKernelElems);
  for (auto k = 0; k != numKernelElems / kernelSizeX ; ++k) {
    // unflatten kernel index into a co-ordinate for the kernel
    auto kCoord = unflattenIndexInSlice(slice.kernelBegin, slice.kernelEnd,
                                        k * kernelSizeX);
    std::vector<unsigned> convOutBegin, convOutEnd, convOutSizesForKy;
    for (auto dim = 0U; dim + 1 != numFieldDims; ++dim ) {
      unsigned begin, end;
      std::tie(begin, end) =
        getOutputRangeForKernelIndex(dim,
                                     {slice.outFieldBegin[dim],
                                      slice.outFieldEnd[dim]},
                                     kCoord[dim], params);
      convOutBegin.push_back(begin);
      convOutEnd.push_back(end);
      convOutSizesForKy.push_back(end - begin);
    }
    const auto convOutElems = getNumElementsInSlice(convOutBegin, convOutEnd);
    if (convOutElems == 0)
      continue;
    for (auto kx = kernelXBegin; kx != kernelXEnd; ++kx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRangeForKernelIndex(xDimIndex, {outXBegin, outXEnd}, kx,
                                       params);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;

      auto outFieldBegin = convOutBegin;
      outFieldBegin.push_back(convOutXBegin);
      auto outFieldEnd = convOutEnd;
      outFieldEnd.push_back(convOutXEnd);
      auto workerPartition =
          partitionConvOutputBetweenWorkers(graph, batchBegin, batchEnd,
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
          std::vector<unsigned> workerIn;
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

          const auto outBeginOffset =
              (workerSlice.b - batchBegin) * numOutElems +
              flattenIndexInSlice(slice.outFieldBegin, slice.outFieldEnd,
                                  workerSlice.outerFieldIndices) * outWidth +
              (workerOutXBegin - outXBegin);

          const auto inBeginOffset =
              (workerSlice.b - batchBegin) * numInElems +
              flattenIndexInSlice(inFieldBegin, inFieldEnd,
                                   workerIn) * inWidth +
              (workerInXBegin - inXBegin);
          auto kIndex = k * kernelSizeX + kx - kernelXBegin;
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
                                          dType, plan.getPartialType()));
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroups"], outZGroupEnd - outZGroupBegin);
  graph.setInitialValue(v["numInGroups"], inZGroupEnd - inZGroupBegin);
  graph.setInitialValue(v["kernelSize"], numKernelElems);
  graph.setInitialValue(v["inStride"], inStrideX);
  graph.setInitialValue(v["outStride"], outStrideX);
  graph.setInitialValue(v["numConvGroups"], cgEnd - cgBegin);
  graph.setInitialValue(v["flipOut"], flipOut);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["numWorkerContexts"], contextsPerVertex);
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
    unsigned cgBegin, unsigned cgEnd,
    unsigned chanGroupBegin, unsigned chanGroupEnd,
    unsigned xBegin, unsigned xEnd,
    const ConvParams &params,
    ComputeSet fwdCS,
    Tensor in,
    Tensor weights,
    const Tensor &out) {
  const auto dataPathWidth = graph.getTarget().getDataPathWidth();
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

  // check every field dimensions of the weights tensor is 1
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
  const auto outShape = out.shape();

  // create output tensor slice vectors
  std::vector<std::size_t> sliceBegin(numFieldDims + 1);
  std::vector<std::size_t> sliceEnd;
  sliceEnd.insert(sliceEnd.begin(), outShape.begin() + 1,
                  outShape.begin() + numFieldDims + 2);
  sliceBegin.push_back(xBegin);
  sliceEnd.push_back(xEnd);

  for (auto cg = cgBegin; cg != cgEnd; ++cg) {
    const auto chanBegin = chanGroupBegin * chansPerGroup;
    const auto chanEnd = chanGroupEnd * chansPerGroup;
    auto inWindow = in[cg].flatten().slice(xBegin, xEnd);
    sliceBegin[0] = chanGroupBegin;
    sliceEnd[0] = chanGroupEnd;
    auto outWindow =
        out[cg].slice(sliceBegin, sliceEnd)
               .reshape({chanGroupEnd - chanGroupBegin,
                         (xEnd - xBegin) * chansPerGroup});
    auto weightsWindow = weights[cg].flatten().slice(chanBegin, chanEnd);
    auto v = graph.addVertex(fwdCS,
                             templateVertex(
                               "popconv::OuterProduct", dType
                             ),
                             {{"in", inWindow},
                              {"weights", weightsWindow},
                              {"out", outWindow}});
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  }
}

/// Compute the sub-convolution corresponding to the specified slice of a larger
/// convolution. The parameters and tensors are updated in place to
/// the parameters and tensors for the sub-convolution.
static void
getSubConvolution(ConvSlice &slice,
                  ConvParams &params,
                  Tensor &in, Tensor &weights, Tensor &out) {
  auto tileSlice = slice;
  const auto numFieldDims = params.getNumFieldDims();
  // Explicitly truncate the convGroup, channel and batch axes.
  const auto partialChansPerGroup = weights.dim(weights.rank() - 2);
  const auto inChansPerGroup = weights.dim(weights.rank() - 1);
  assert(slice.outChanBegin % partialChansPerGroup == 0);
  assert(slice.outChanEnd % partialChansPerGroup == 0);
  const auto outZGroupBegin = slice.outChanBegin / partialChansPerGroup;
  const auto outZGroupEnd = slice.outChanEnd / partialChansPerGroup;
  assert(slice.inChanBegin % inChansPerGroup == 0);
  assert(slice.inChanEnd % inChansPerGroup == 0);
  const auto inZGroupBegin = slice.inChanBegin / inChansPerGroup;
  const auto inZGroupEnd = slice.inChanEnd / inChansPerGroup;
  params.numConvGroups = slice.getNumConvGroups();
  params.batchSize = slice.getBatchSize();
  params.inputChannels = slice.getNumInputChans() * slice.getNumConvGroups();
  params.outputChannels = slice.getNumOutputChans() * slice.getNumConvGroups();
  in = in.slice({slice.cgBegin, inZGroupBegin, slice.batchBegin},
                {slice.cgEnd, inZGroupEnd, slice.batchEnd});
  weights = weights.slice({slice.cgBegin, outZGroupBegin, inZGroupBegin},
                          {slice.cgEnd, outZGroupEnd, inZGroupEnd});
  out = out.slice({slice.cgBegin, outZGroupBegin, slice.batchBegin},
                  {slice.cgEnd, outZGroupEnd, slice.batchEnd});
  tileSlice.cgBegin = 0;
  tileSlice.cgEnd = slice.getNumConvGroups();
  tileSlice.batchBegin = 0;
  tileSlice.batchEnd = slice.getBatchSize();
  tileSlice.inChanBegin = 0;
  tileSlice.inChanEnd = slice.getNumInputChans();
  tileSlice.outChanBegin = 0;
  tileSlice.outChanEnd = slice.getNumOutputChans();
  // Explicitly truncate the spatial dimensions.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto extraTruncationLower = slice.outFieldBegin[dim];
    auto extraTruncationUpper =
        static_cast<unsigned>(params.getOutputSize(dim))
        - slice.outFieldEnd[dim];
    auto &outputPaddingLower = params.outputTransform.paddingLower[dim];
    auto &outputPaddingUpper = params.outputTransform.paddingUpper[dim];
    const auto &stride = params.outputTransform.stride[dim];
    auto &outputTruncationLower = params.outputTransform.truncationLower[dim];
    auto &outputTruncationUpper = params.outputTransform.truncationUpper[dim];
    const auto excessPaddingLower = std::min(outputPaddingLower,
                                             extraTruncationLower);
    outputPaddingLower -= excessPaddingLower;
    extraTruncationLower -= excessPaddingLower;
    if (extraTruncationLower == params.getOutputSize(dim)) {
      outputTruncationLower += 1 + (extraTruncationLower - 1) * stride;
    } else {
      outputTruncationLower += extraTruncationLower * stride;
    }
    extraTruncationLower = 0;
    const auto excessPaddingUpper = std::min(outputPaddingUpper,
                                             extraTruncationUpper);
    outputPaddingUpper -= excessPaddingUpper;
    extraTruncationUpper -= excessPaddingUpper;
    if (extraTruncationUpper == params.getOutputSize(dim)) {
      outputTruncationUpper += 1 + (extraTruncationUpper - 1) * stride;
    } else {
      outputTruncationUpper += extraTruncationUpper * stride;
    }
    extraTruncationUpper = 0;
    out = out.slice(slice.outFieldBegin[dim],
                    slice.outFieldEnd[dim],
                    3 + dim);
    tileSlice.outFieldBegin[dim] = 0;
    tileSlice.outFieldEnd[dim] = slice.getOutputSize(dim);
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
    in = in.slice(inputTruncationLower,
                  params.inputFieldShape[dim] - inputTruncationUpper,
                  3 + dim);
    params.inputFieldShape[dim] -= inputTruncationLower + inputTruncationUpper;
    inputTruncationLower = 0;
    inputTruncationUpper = 0;
  }

  // Explicitly truncate the kernel.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto &kernelTruncationLower = params.kernelTransform.truncationLower[dim];
    auto &kernelTruncationUpper = params.kernelTransform.truncationUpper[dim];
    weights = weights.slice(kernelTruncationLower,
                            params.kernelShape[dim] - kernelTruncationUpper,
                            3 + dim);
    params.kernelShape[dim] -= kernelTruncationLower + kernelTruncationUpper;
    kernelTruncationLower = 0;
    kernelTruncationUpper = 0;
    tileSlice.kernelBegin[dim] = 0;
    tileSlice.kernelEnd[dim] = params.kernelShape[dim];
  }
  assert(params == canonicalizeParams(params));

  slice = tileSlice;
}

static bool isZeroConvolution(const ConvParams &params) {
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

static void
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      const Type &dType,
                      unsigned tile,
                      ConvSlice slice,
                      ConvParams params,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out) {
  getSubConvolution(slice, params, in, weights, out);
  graph.setTileMapping(out, tile);
  if (isZeroConvolution(params)) {
    zero(graph, out, tile, fwdCS);
    return;
  }
  switch (plan.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    createConvPartialAmpVertex(graph, plan, dType, tile, slice, params, fwdCS,
                               in, weights, out);
    break;
  case Plan::Method::MAC:
    createConvPartialHorizontalMacVertex(graph, plan, dType, tile,
                                         slice, params,fwdCS, in, weights,
                                         out);
    break;
  case Plan::Method::OUTER_PRODUCT:
    {
      const auto &target = graph.getTarget();
      const auto cgBegin = slice.cgBegin;
      const auto cgEnd = slice.cgEnd;
      const auto outXBegin = slice.outFieldBegin.back();
      const auto outXEnd = slice.outFieldEnd.back();
      assert(slice.outChanBegin % plan.partialChansPerGroup == 0);
      assert(slice.outChanEnd % plan.partialChansPerGroup == 0);
      const auto outZGroupBegin = slice.outChanBegin /
                                  plan.partialChansPerGroup;
      const auto outZGroupEnd = slice.outChanEnd /
                                plan.partialChansPerGroup;
      const auto perWorkerRegions =
          splitRegionsBetweenWorkers(target, {{outXBegin, outXEnd}}, 1);
      for (const auto &entry : perWorkerRegions) {
        assert(entry.size() == 1);
        createOuterProductVertex(graph, tile,
                                 cgBegin, cgEnd,
                                 outZGroupBegin, outZGroupEnd,
                                 entry[0].begin(), entry[0].end(), params,
                                 fwdCS, in, weights, out);
      }
    }
    break;
  }
}

static Tensor
calcPartialSums(Graph &graph,
                const Plan &plan,
                const ConvParams &params,
                Tensor in, Tensor weights,
                Sequence &prog,
                const std::string &layerName) {
  const auto numBatchGroups = in.dim(2);
  const auto dType = in.elementType();
  const auto outNumChans = weights.dim(1) * weights.dim(weights.rank() - 2);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;

  const auto partialType = plan.getPartialType();
  // Calculate a set of partial sums of the convolutions.
  std::vector<std::size_t> partialsShape;
  for (const auto p : plan.partitions) {
    partialsShape.push_back(p.inChanSplit * product(p.kernelSplit));
  }
  partialsShape.push_back(params.getNumConvGroups());
  partialsShape.push_back(partialNumChanGroups);
  partialsShape.push_back(numBatchGroups);
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    partialsShape.push_back(params.getOutputSize(dim));
  }
  partialsShape.push_back(partialChansPerGroup);
  Tensor partials = graph.addVariable(partialType, partialsShape, "partials");
  ComputeSet convolveCS = graph.addComputeSet(layerName + "/Convolve");
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile,
                           const std::vector<ConvIndices> &indices,
                           const ConvSlice &slice) {
    if (slice.outChanBegin == slice.outChanEnd ||
        slice.cgBegin == slice.cgEnd)
      return;
    const auto numFieldDims = params.getNumFieldDims();
    std::vector<unsigned> partialIndices;
    const auto numLevels = plan.partitions.size();
    assert(indices.size() == numLevels);
    auto partialsSlice = partials;
    for (unsigned i = 0; i != numLevels; ++i) {
      unsigned partialIndex = indices[i].ic;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        partialIndex =
            partialIndex * plan.partitions[i].kernelSplit[dim] +
            indices[i].kernel[dim];
      }
      partialsSlice = partialsSlice[partialIndex];
    }
    calcPartialConvOutput(graph, plan, dType, tile, slice, params, convolveCS,
                          in, weights, partialsSlice);
  });
  prog.add(Execute(convolveCS));
  return partials;
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
    popreduce::reduce(graph, partials.slice(begin, end), out[i],
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

static Tensor
convolutionImpl(Graph &graph, const Plan &plan,
                const ConvParams &params,
                Tensor in, Tensor weights,
                Sequence &prog, const std::string &debugPrefix) {
  in = splitActivationChanGroups(in, plan.inChansPerGroup);
  weights = groupWeights(weights, plan.inChansPerGroup,
                         plan.partialChansPerGroup);
  const auto dType = in.elementType();
  const auto partialType = plan.getPartialType();
  auto partials =
      calcPartialSums(graph, plan, params, in, weights, prog, debugPrefix);
  const auto numLevels = plan.partitions.size();
  for (int i = static_cast<int>(numLevels) - 1; i >= 0; --i) {
    const auto resultType = i == 0 ? dType : partialType;
    std::vector<ComputeSet> reduceComputeSets;
    // For each element of the batch, we add the reduction vertices to same
    // compute sets so the batch will be executed in parallel.
    Tensor reduced;
    // Perform the reduction of partial sums.
    if (partials.dim(i) == 1) {
      if (partialType != resultType) {
        reduced = graph.clone(dType, partials, "reduced");
        if (reduceComputeSets.empty()) {
          reduceComputeSets.push_back(graph.addComputeSet(debugPrefix +
                                                             "/Cast"));
        }
        cast(graph, partials, reduced, reduceComputeSets[0]);
      } else {
        reduced = partials;
      }
      reduced = reduced.squeeze({std::size_t(i)});
    } else {
      reduced = multiStageGroupedReduce(graph, partials.dimRoll(i, 0),
                                        resultType, reduceComputeSets,
                                        debugPrefix);
    }
    for (const auto &cs : reduceComputeSets) {
      prog.add(Execute(cs));
    }
    partials = reduced;
  }
  return unsplitActivationChanGroups(partials);
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

// Postprocess results of convolution
// - undo any flattening of the field
// - undo any padding
static Tensor
convolutionPostprocess(Graph &graph, const ConvParams &originalParams,
                       const Plan &originalPlan,
                       Tensor activations) {
  auto postAddExtraDimsParams = originalParams;
  if (originalPlan.extraFieldDims) {
    addExtraDims(postAddExtraDimsParams, originalPlan.extraFieldDims);
  }
  auto postExpandParams = postAddExtraDimsParams;
  if (originalPlan.swapOperands) {
    swapOperands(postExpandParams);
  }
  for (auto dim : originalPlan.expandDims) {
    expandSpatialDim(graph, postExpandParams, nullptr, nullptr, dim);
  }
  auto postOutChanFlattenParams = postExpandParams;
  if (!originalPlan.outChanFlattenDims.empty()) {
    swapOperands(postOutChanFlattenParams);
    for (auto dim : originalPlan.outChanFlattenDims) {
      expandSpatialDim(graph, postOutChanFlattenParams, nullptr, nullptr, dim);
      // Flatten into the batch axis (this will become the output channel
      // axis when we swap back).
      postOutChanFlattenParams.batchSize *=
          postOutChanFlattenParams.inputFieldShape[dim];
      postOutChanFlattenParams.inputFieldShape[dim] = 1;
    }
    swapOperands(postOutChanFlattenParams);
  }
  const auto outNumChans =
      postOutChanFlattenParams.getNumOutputChansPerConvGroup();
  // Undo padding.
  activations = activations.slice(0, outNumChans, activations.rank() - 1);
  // Undo flattening of the batch / spatial fields.
  if (!originalPlan.flattenDims.empty()) {
    for (auto it = originalPlan.flattenDims.begin(),
         end = std::prev(originalPlan.flattenDims.end()); it != end; ++it) {
      const auto fromDimIndex = *it;
      const auto toDimIndex = originalPlan.flattenDims.back();
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
  for (auto it = originalPlan.outChanFlattenDims.rbegin(),
       end = originalPlan.outChanFlattenDims.rend(); it != end; ++it) {
    const auto spatialDim = *it;
    const auto spatialDimSize =
        postAddExtraDimsParams.getOutputSize(spatialDim);
    activations =
        unflattenDims(activations, 2 + spatialDim, activations.rank() - 1,
                      spatialDimSize);
  }
  // Undo the swapping of operands.
  if (originalPlan.swapOperands) {
    activations = activations.dimShufflePartial({1, activations.rank() - 1},
                                                {activations.rank() - 1, 1});
  }
  if (originalPlan.extraFieldDims) {
    std::vector<std::size_t> toSqueeze(originalPlan.extraFieldDims);
    std::iota(toSqueeze.begin(), toSqueeze.end(), std::size_t(2));
    activations = activations.squeeze(toSqueeze);
  }
  return activations;
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

Tensor
convolution(Graph &graph, const poplar::Tensor &in_,
            const poplar::Tensor &weights_,
            const ConvParams &params_,
            bool transposeAndFlipWeights, Sequence &prog,
            const std::string &debugPrefix, const ConvOptions &options) {
  auto params = params_;
  auto weights = weights_;
  if (weights.rank() == params_.getNumFieldDims() + 2) {
    weights = weights.expand({0});
  }
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    auto bwdWeights = createWeights(graph, params, "bwdWeights", options);
    weightsTransposeChansFlipXY(graph, weights, bwdWeights, prog, debugPrefix);
    weights = bwdWeights;
  }
  weights = weightsToInternalShape(weights);
  auto in = actsToInternalShape(in_, params.numConvGroups);
  const auto dType = in.elementType();
  auto plan = getPlan(graph, params, options);

  verifyInputShapes(params, in, weights);

  const auto originalParams = params;
  const auto originalPlan = plan;
  convolutionPreprocess(graph, params, plan, &in, &weights);

  // If the input tensors have a different memory layout to the one expected by
  // the vertices poplar will rearrange the data using exchange code or copy
  // pointers. If the data is broadcast this rearrangement happens on every tile
  // that receives the data. We can reduce the amount of exchange code / number
  // of copy pointers required by rearranging the data once and broadcasting the
  // rearranged data. This trades increased execution time for reduced memory
  // usage. The biggest reductions in memory usage come when data is broadcast
  // to many tiles. inViewMaxBroadcastDests and weightViewMaxBroadcastDests
  // specify the maximum number of broadcast destinations a tensor can have
  // before we insert a copy to rearrange it.
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
    auto inRearranged = createInputImpl(graph, params, "inRearranged", plan);
    prog.add(Copy(in, inRearranged));
    in = inRearranged;
  }
  auto weightsNumDests = plan.partitions.back().batchSplit;
  for (const auto split : plan.partitions.back().fieldSplit) {
    weightsNumDests *= split;
  }
  if (weightsNumDests > weightViewMaxBroadcastDests) {
    auto weightsRearranged = createWeightsImpl(graph, params,
                                               "weightsRearranged", plan);
    prog.add(Copy(weights, weightsRearranged));
    weights = weightsRearranged;
  }

  if (plan.useWinograd) {
    throw popstd::poplib_error("Winograd not yet supported");
  }
  const auto layerName = debugPrefix + "/Conv" + convSuffix(params);
  auto activations =
      convolutionImpl(graph, plan, params, in, weights, prog, layerName);
  activations = convolutionPostprocess(graph, originalParams, originalPlan,
                                       activations);
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
  popstd::addTo(graph, weights, weightDeltas, -learningRate, prog,
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
      auto beginIndices = popstd::unflattenIndex(firstInGroup.shape(), begin);
      auto lastIndices = popstd::unflattenIndex(firstInGroup.shape(), last);
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
      graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
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
                               templateVertex("popstd::Zero", partialsType));
      graph.connect(v["out"], partials[worker].slice(0, maxOutputsPerWorker));
      graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
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
        auto beginIndices = popstd::unflattenIndex(firstInGroup.shape(), begin);
        auto lastIndices = popstd::unflattenIndex(firstInGroup.shape(), last);
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
      graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
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
    throw popstd::poplib_error("fullyConnectedWeightTranspose() expects a 1-d "
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
          auto blockIndices = popstd::unflattenIndex(firstInBlock.shape(),
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
                                const Tensor &activations,
                                const Tensor &zDeltas,
                                const ConvParams &fwdParams,
                                const ConvOptions &options) {
  auto params = getWeightUpdateParams(fwdParams);
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
        auto beginIndices = popstd::unflattenIndex(firstInGroup.shape(), begin);
        auto lastIndices = popstd::unflattenIndex(firstInGroup.shape(), last);
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
      graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
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
  const auto dataPathWidth = target.getDataPathWidth();
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
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
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
