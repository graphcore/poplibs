#include "popconv/Convolution.hpp"
#include "ConvPlan.hpp"
#include <limits>
#include <algorithm>
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

ConvParams::ConvParams(std::string dType_,
                       std::size_t batchSize_,
                       std::vector<std::size_t> inputFieldShape_,
                       std::vector<std::size_t> kernelShape_,
                       std::size_t inputChannels_,
                       std::size_t outputChannels_,
                       std::vector<unsigned> stride_,
                       std::vector<int> inputPaddingLower_,
                       std::vector<int> inputPaddingUpper_,
                       std::vector<unsigned> inputDilation_,
                       std::vector<int> kernelPaddingLower_,
                       std::vector<int> kernelPaddingUpper_,
                       std::vector<unsigned> kernelDilation_,
                       std::size_t numConvGroups_) :
    dType(std::move(dType_)),
    batchSize(batchSize_),
    inputFieldShape(std::move(inputFieldShape_)),
    kernelShape(std::move(kernelShape_)),
    inputChannels(inputChannels_),
    outputChannels(outputChannels_),
    stride(std::move(stride_)),
    inputPaddingLower(std::move(inputPaddingLower_)),
    inputPaddingUpper(std::move(inputPaddingUpper_)),
    inputDilation(std::move(inputDilation_)),
    kernelPaddingLower(std::move(kernelPaddingLower_)),
    kernelPaddingUpper(std::move(kernelPaddingUpper_)),
    kernelDilation(std::move(kernelDilation_)),
    numConvGroups(numConvGroups_) {
  const auto numFieldDims = inputFieldShape.size();
  if (kernelShape.size() != numFieldDims) {
    throw popstd::poplib_error("Number of kernel field dimensions does not"
                               "match the number of input field dimensions");
  }
  const std::pair<std::size_t, const char *> sizes[] = {
    {stride.size(), "stride"},
    {inputPaddingLower.size(), "input padding (lower)"},
    {inputPaddingUpper.size(), "input padding (upper)"},
    {inputDilation.size(), "input dilation"},
    {kernelPaddingLower.size(), "kernel padding (lower)"},
    {kernelPaddingUpper.size(), "kernel padding (upper)"},
    {kernelDilation.size(), "kernel dilation"}
  };
  for (const auto &entry : sizes) {
    if (entry.first != numFieldDims) {
      throw popstd::poplib_error(std::string("Number of ") + entry.second +
                                 " dimensions does not match the number of "
                                 "field dimensions");
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
  os << "        stride                     ";
  printContainer(p.stride, os);
  os << "\n";
  os << "        inputPaddingLower          ";
  printContainer(p.inputPaddingLower, os);
  os << "\n";
  os << "        inputPaddingUpper          ";
  printContainer(p.inputPaddingUpper, os);
  os << "\n";
  os << "        inputDilation              ";
  printContainer(p.inputDilation, os);
  os << "\n";
  os << "        kernelPaddingLower         ";
  printContainer(p.kernelPaddingLower, os);
  os << "\n";
  os << "        kernelPaddingUpper         ";
  printContainer(p.kernelPaddingUpper, os);
  os << "\n";
  os << "        kernelDilation             ";
  printContainer(p.kernelDilation, os);
  os << "\n";
  return os;
}

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

std::size_t ConvParams::getOutputSize(unsigned dim) const {
  auto paddedDilatedInputSize = getPaddedDilatedInputSize(dim);
  auto paddedDilatedKernelSize = getPaddedDilatedKernelSize(dim);
  return absdiff(paddedDilatedInputSize, paddedDilatedKernelSize) /
                 stride[dim] + 1;
}

std::size_t ConvParams::getOutputHeight() const {
  return getOutputSize(0);
}

std::size_t ConvParams::getOutputWidth() const {
  return getOutputSize(1);
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
      std::vector<Interval<std::size_t>>
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
linearizeTileIndices(unsigned numTiles,
                     const std::vector<unsigned> &outIndices,
                     const std::vector<unsigned> &kernelIndices,
                     unsigned ic, unsigned b, unsigned oc, unsigned cg,
                     const std::vector<unsigned> &fieldTileSplit,
                     const std::vector<unsigned> &kernelTileSplit,
                     unsigned inChanTileSplit, unsigned batchTileSplit,
                     unsigned outChanTileSplit, unsigned convGroupTileSplit) {
  const auto numFieldDims = outIndices.size();
  // Use ozg as the innermost dimension to increase the chance that
  // tiles in a supertile both read the same activations. This reduces
  // exchange time when supertile send / receive is used.
  auto tile = cg;
  tile = tile * inChanTileSplit + ic;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    tile = tile * kernelTileSplit[dim] + kernelIndices[dim];
  }
  tile = tile * batchTileSplit + b;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    tile = tile * fieldTileSplit[dim] + outIndices[dim];
  }
  tile = tile * outChanTileSplit + oc;
  assert(tile < numTiles);
  return tile;
}

static unsigned
linearizeTileIndices(unsigned numTiles, const std::vector<unsigned> &outIndices,
                     const std::vector<unsigned> &kernelIndices, unsigned ic,
                     unsigned b, unsigned oc, unsigned cg, const Plan &plan) {
  auto fwdOutIndices = outIndices;
  const auto &fwdKernelIndices = kernelIndices;
  auto fwdic = ic;
  const auto fwdb = b;
  auto fwdoc = oc;
  const auto fwdcg = cg;
  auto fwdFieldTileSplit = plan.fieldTileSplit;
  const auto &fwdKernelTileSplit = plan.kernelTileSplit;
  auto fwdInChanTileSplit = plan.inChanTileSplit;
  const auto &fwdBatchTileSplit = plan.batchTileSplit;
  auto fwdOutChanTileSplit = plan.outChanTileSplit;
  const auto &convGroupTileSplit = plan.convGroupTileSplit;
  switch (plan.linearizeTileOrder) {
  case Plan::LinearizeTileOrder::FC_WU:
    // For the fully connected weight update the in group and out group are
    // swapped compared to the forward pass.
    std::swap(fwdInChanTileSplit, fwdOutChanTileSplit);
    std::swap(fwdic, fwdoc);
    break;
  case Plan::LinearizeTileOrder::FC_BWD_AS_CONV:
    // For the fully connected backward pass the width and the input channels
    // are swapped compared to the forward pass.
    {
      std::swap(fwdFieldTileSplit.back(), fwdInChanTileSplit);
      std::swap(fwdOutIndices.back(), fwdic);
    }
    break;
  case Plan::LinearizeTileOrder::STANDARD:
    break;
  }
  auto tile = linearizeTileIndices(numTiles, fwdOutIndices, fwdKernelIndices,
                                   fwdic, fwdb, fwdoc, fwdcg, fwdFieldTileSplit,
                                   fwdKernelTileSplit, fwdInChanTileSplit,
                                   fwdBatchTileSplit, fwdOutChanTileSplit,
                                   convGroupTileSplit);
  assert(tile < numTiles);
  return tile;
}

static std::pair<unsigned,unsigned>
getOutChanGroupRange(unsigned ozgIndex, unsigned partialNumChanGroups,
                     const Plan &plan) {
  const auto outChanTileSplit = plan.outChanTileSplit;
  const auto maxOutChanGroupsPerTile =
      (partialNumChanGroups + outChanTileSplit - 1) / outChanTileSplit;
  const auto outChanGroupBegin =
      std::min(ozgIndex * maxOutChanGroupsPerTile, partialNumChanGroups);
  const auto outChanGroupEnd =
      std::min((ozgIndex + 1) * maxOutChanGroupsPerTile, partialNumChanGroups);
  return {outChanGroupBegin, outChanGroupEnd};
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
                    std::vector<Interval<std::size_t>> &regions) {
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
      regions.back() = Interval<std::size_t>(regions.back().begin(),
                                             regionEnd);
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


struct ConvTileIndices {
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
  unsigned outChanGroupBegin, outChanGroupEnd;
  unsigned inChanGroupBegin, inChanGroupEnd;
  std::vector<unsigned> kernelBegin, kernelEnd;
};

static std::pair<unsigned, unsigned>
getTileOutRange(const ConvParams &params, const Plan &plan, unsigned tileIndex,
                unsigned dim) {
  const auto fieldSize = params.getOutputSize(dim);
  const auto grainSize = plan.fieldAxisGrainSize[dim];
  const auto numGrains = (fieldSize + grainSize - 1) / grainSize;
  const auto split = plan.fieldTileSplit[dim];
  const auto outGrainBegin = (tileIndex * numGrains) / split;
  const auto outGrainEnd = ((tileIndex + 1) * numGrains) / split;
  const auto outBegin = outGrainBegin * grainSize;
  const auto outEnd = std::min(outGrainEnd * grainSize, fieldSize);
  return {outBegin, outEnd};
}

static void
iterateTilePartition(const Graph &graph, const ConvParams &params,
                     const Plan &plan,
                     const std::function<
                       void(unsigned, const ConvTileIndices &,
                            const ConvSlice &)
                     > &f) {
  const auto numFieldDims = params.getNumFieldDims();
  const unsigned inNumChans = params.getNumInputChansPerConvGroup();
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(params.getNumOutputChansPerConvGroup() % partialChansPerGroup == 0);
  const auto partialNumChanGroups =
      params.getNumOutputChansPerConvGroup() / partialChansPerGroup;
  const auto batchTileSplit = plan.batchTileSplit;
  const auto outChanTileSplit = plan.outChanTileSplit;
  const auto inChanTileSplit = plan.inChanTileSplit;
  const unsigned batchSize = params.getBatchSize();
  const unsigned numInZGroups = inNumChans / inChansPerGroup;
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto convGroupTileSplit = plan.convGroupTileSplit;
  const unsigned numConvGroups = params.getNumConvGroups();
  const auto totalFieldTileSplit = product(plan.fieldTileSplit);
  const auto totalKernelTileSplit = product(plan.kernelTileSplit);
  for (unsigned cg = 0; cg != convGroupTileSplit; ++cg) {
    const auto cgBegin = (cg * numConvGroups) / convGroupTileSplit;
    const auto cgEnd = ((cg + 1) * numConvGroups) / convGroupTileSplit;
    for (unsigned b = 0; b != batchTileSplit; ++b) {
      const auto batchBegin = (b * batchSize) / batchTileSplit;
      const auto batchEnd = ((b + 1) * batchSize) / batchTileSplit;
      for (unsigned izg = 0; izg != inChanTileSplit; ++izg) {
        const auto inZGroupBegin = (izg * numInZGroups) / inChanTileSplit;
        const auto inZGroupEnd = ((izg + 1) * numInZGroups) / inChanTileSplit;
        for (unsigned k = 0; k != totalKernelTileSplit; ++k) {
          auto kernelIndices = unflattenIndex(plan.kernelTileSplit, k);
          std::vector<unsigned> kernelBegin(numFieldDims),
                                kernelEnd(numFieldDims);
          for (unsigned dim = 0; dim != numFieldDims; ++dim) {
            kernelBegin[dim] = (kernelIndices[dim] * params.kernelShape[dim]) /
                               plan.kernelTileSplit[dim];
            kernelEnd[dim] = ((kernelIndices[dim] + 1) *
                              params.kernelShape[dim]) /
                             plan.kernelTileSplit[dim];
          }
          for (unsigned ozg = 0; ozg != outChanTileSplit; ++ozg) {
            unsigned outZGroupBegin, outZGroupEnd;
            std::tie(outZGroupBegin, outZGroupEnd) =
                getOutChanGroupRange(ozg, partialNumChanGroups, plan);
            for (unsigned of = 0; of != totalFieldTileSplit; ++of) {
              auto outIndices = unflattenIndex(plan.fieldTileSplit, of);
              std::vector<unsigned> outFieldBegin(numFieldDims),
                                    outFieldEnd(numFieldDims);
              for (unsigned dim = 0; dim != numFieldDims; ++dim) {
                std::tie(outFieldBegin[dim], outFieldEnd[dim]) =
                    getTileOutRange(params, plan, outIndices[dim], dim);
              }
              const auto tile = linearizeTileIndices(numTiles, outIndices,
                                                     kernelIndices, izg, b, ozg,
                                                     cg, plan);
              f(tile,
                {cg, b, outIndices, ozg, izg, kernelIndices},
                {cgBegin, cgEnd,
                 batchBegin, batchEnd,
                 outFieldBegin,
                 outFieldEnd,
                 outZGroupBegin, outZGroupEnd,
                 inZGroupBegin, inZGroupEnd,
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

static std::vector<std::vector<Interval<std::size_t>>>
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
  std::map<std::set<unsigned>, std::vector<Interval<std::size_t>>>
      tilesToGrains;
  for (const auto &entry : grainToTiles) {
    tilesToGrains[entry.second].emplace_back(entry.first.lower(),
                                             entry.first.upper());
  }
  const auto numTiles = graph.getTarget().getNumTiles();
  std::vector<std::vector<Interval<std::size_t>>> mapping(numTiles);
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
toIclInterval(const Interval<std::size_t> &interval) {
  return boost::icl::interval<unsigned>::right_open(interval.begin(),
                                                    interval.end());
}

static void
addFlattenedPrevActsRegions(const std::vector<std::size_t> &actsShape,
                            const ConvSlice &slice,
                            const ConvParams &params,
                            std::vector<Interval<std::size_t>> &regions) {
  assert(actsShape.size() >= 4);
  const auto numFieldDims = actsShape.size() - 4;
  assert(slice.outFieldBegin.size() == numFieldDims);
  assert(slice.outFieldEnd.size() == numFieldDims);
  assert(slice.kernelBegin.size() == numFieldDims);
  assert(slice.kernelEnd.size() == numFieldDims);
  std::vector<std::size_t> sliceBegin = {
    slice.cgBegin,
    slice.inChanGroupBegin,
    slice.batchBegin
  };
  std::vector<std::size_t> sliceEnd = {
    slice.cgEnd,
    slice.inChanGroupEnd,
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

static std::vector<std::vector<Interval<std::size_t>>>
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
                       [&](unsigned tile, const ConvTileIndices &,
                           const ConvSlice &slice) {
    std::vector<Interval<std::size_t>> intervals;
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
  const auto actTypeSize = actType == "float" ? 4 : 2;
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
  const auto newSize = (oldSize - 1) * dilationFactor + 1;
  if (newSize == oldSize)
    return t;
  const auto dType = t.elementType();
  auto zeroShape = t.shape();
  zeroShape[dim] = 1;
  Tensor zero = graph.addConstantTensor(dType, zeroShape, 0);
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
  bool actsAreLarger = expandDimExpandActs(params, dim);
  Tensor *larger = actsAreLarger ? acts : weights;
  Tensor *smaller = actsAreLarger ? weights : acts;
  unsigned largerDimIndex = dim + (actsAreLarger ? 2 : 1);
  unsigned smallerDimIndex = dim + (actsAreLarger ? 1 : 2);
  auto &largerSize = actsAreLarger ? params.inputFieldShape[dim] :
                                     params.kernelShape[dim];
  auto &smallerSize = actsAreLarger ? params.kernelShape[dim] :
                                      params.inputFieldShape[dim];
  auto &largerDilation = actsAreLarger ? params.inputDilation[dim] :
                                          params.kernelDilation[dim];
  auto &smallerDilation = actsAreLarger ? params.kernelDilation[dim] :
                                          params.inputDilation[dim];
  auto &largerPaddingLower = actsAreLarger ? params.inputPaddingLower[dim] :
                                             params.kernelPaddingLower[dim];
  auto &smallerPaddingLower = actsAreLarger ? params.kernelPaddingLower[dim] :
                                              params.inputPaddingLower[dim];
  auto &largerPaddingUpper = actsAreLarger ? params.inputPaddingUpper[dim] :
                                              params.kernelPaddingUpper[dim];
  auto &smallerPaddingUpper = actsAreLarger ? params.kernelPaddingUpper[dim] :
                                              params.inputPaddingUpper[dim];
  if (larger) {
    // Explicitly dilate this axis.
    *larger = dilate(graph, *larger, largerDilation, largerDimIndex);
    // Explicitly pad this axis.
    *larger = pad(graph, *larger, largerPaddingLower, largerPaddingUpper,
                  largerDimIndex);
  }
  largerSize = (largerSize - 1) * largerDilation + 1;
  largerSize += largerPaddingLower + largerPaddingUpper;
  largerDilation = 1;
  largerPaddingLower = 0;
  largerPaddingUpper = 0;
  if (larger) {
    // Expand the larger tensor.
    auto dType = larger->elementType();
    auto expandedShape = larger->shape();
    expandedShape[largerDimIndex] = params.getOutputSize(dim);
    expandedShape.back() = 0;
    auto smallerPaddedDilatedSize =
        actsAreLarger ? params.getPaddedDilatedKernelSize(dim) :
                        params.getPaddedDilatedInputSize(dim);
    std::vector<Tensor> slices;
    for (unsigned k = 0; k != smallerSize; ++k) {
      auto dilatedPaddedK =
          static_cast<int>(k * smallerDilation) +
          smallerPaddingLower;
      Tensor slice;
      if (dilatedPaddedK >= 0 && dilatedPaddedK < smallerPaddedDilatedSize) {
        slice =
            larger->slice(dilatedPaddedK,
                          dilatedPaddedK +
                          1 +
                          (params.getOutputSize(dim) - 1) * params.stride[dim],
                          largerDimIndex);
        slice = slice.subSample(params.stride[dim], largerDimIndex);
      } else {
        auto zerosShape = expandedShape;
        zerosShape.back() = larger->dim(larger->rank() - 1);
        slice = graph.addConstantTensor(dType, zerosShape, 0);
      }
      slices.push_back(std::move(slice));
    }
    auto expanded = concat(slices, larger->rank() - 1);
    *larger = expanded;
  }
  if (smaller) {
    // Flatten the spatial dimension of the smaller tensor into the input
    // channels.
    *smaller = flattenDims(*smaller, smallerDimIndex, smaller->rank() - 1);
  }
  largerSize = params.getOutputSize(dim);
  params.inputChannels *= smallerSize;
  smallerSize = 1;
  smallerDilation = 1;
  smallerPaddingLower = 0;
  smallerPaddingUpper = 0;
  params.stride[dim] = 1;
}

/// Apply any pre-convolution transformations implied by the plan. The
/// plan and the parameters are updated to describe the convolution operation
/// performed on the transformed input. If the \a acts or \ weights pointers are
/// not null they are updated to be views of the original tensors with
/// dimensions that match the shape expected by the convolution operation.
static void
convolutionPreprocess(Graph &graph, ConvParams &params, Plan &plan,
                      Tensor *acts, Tensor *weights) {
  if (plan.swapOperands) {
    swapOperands(params);
    if (acts && weights) {
      std::swap(*acts, *weights);
      *acts = acts->dimRoll(acts->rank() - 2, 1);
      *weights = weights->dimRoll(1, weights->rank() - 2);
    } else {
      assert(!acts && !weights);
    }
    plan.swapOperands = false;
  }
  for (auto dim : plan.expandDims) {
    expandSpatialDim(graph, params, acts, weights, dim);
  }
  plan.expandDims.clear();
  for (auto dim : plan.outChanFlattenDims) {
    if (weights) {
      *weights = flattenDims(*weights, dim + 1, weights->rank() - 2);
    }
    params.outputChannels *= params.kernelShape[dim];
    params.kernelShape[dim] = 1;
  }
  plan.outChanFlattenDims.clear();
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
    return t.dimRoll(3, 1);
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
  auto t = graph.addTensor(params.dType, tensorShape, name);
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

static std::vector<std::vector<Interval<std::size_t>>>
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
                       [&](unsigned tile, const ConvTileIndices &,
                           const ConvSlice &slice) {
    std::vector<Interval<std::size_t>> intervals;
    std::vector<std::size_t> sliceBegin = {
      slice.cgBegin,
      slice.outChanGroupBegin,
      slice.inChanGroupBegin
    };
    sliceBegin.insert(sliceBegin.end(), slice.kernelBegin.begin(),
                      slice.kernelBegin.end());
    sliceBegin.push_back(0);
    sliceBegin.push_back(0);
    std::vector<std::size_t> sliceEnd = {
      slice.cgEnd,
      slice.outChanGroupEnd,
      slice.inChanGroupEnd
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
  const auto weightTypeSize = weightType == "float" ? 4 : 2;
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
    return t.dimRoll(1, 3);
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
  auto weights = graph.addTensor(dType, weightsShape, name);
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

static std::vector<std::vector<poplar::Interval<std::size_t>>>
computeBiasMapping(Graph &graph, const Tensor &out) {
  const auto &target = graph.getTarget();
  const auto dType = out.elementType();
  const auto dTypeSize = dType == "float" ? 4 : 2;
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
  const auto grainSize =
      dType == "float" ? target.getFloatVectorWidth() :
                         target.getHalfVectorWidth();
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
  auto biases = graph.addTensor(dType, {numOutChans}, name);
  mapBiases(graph, biases, acts);
  return biases;
}


static void
createConvPartial1x1OutVertex(Graph &graph,
                              const Plan &plan,
                              std::string dType,
                              unsigned tile,
                              const ConvSlice &slice,
                              const ConvParams &params,
                              ComputeSet fwdCS,
                              Tensor in, Tensor weights, Tensor out) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto batchBegin = slice.batchBegin;
  const auto batchEnd = slice.batchEnd;
  const auto outZGroupBegin = slice.outChanGroupBegin;
  const auto outZGroupEnd = slice.outChanGroupEnd;
  const auto cgBegin = slice.cgBegin;
  const auto cgEnd = slice.cgEnd;
  const auto inZGroupBegin = slice.inChanGroupBegin;
  const auto inZGroupEnd = slice.inChanGroupEnd;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto &target = graph.getTarget();
  const auto dataPathWidth = target.getDataPathWidth();
  const auto convUnitInputLoadElemsPerCycle =
    target.getConvUnitInputLoadElemsPerCycle(dType == "float");
  const auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  std::vector<Tensor> outWindow;
  std::vector<unsigned> tileOutBatchAndFieldBegin = { batchBegin };
  tileOutBatchAndFieldBegin.insert(tileOutBatchAndFieldBegin.end(),
                                   slice.outFieldBegin.begin(),
                                   slice.outFieldBegin.end());
  std::vector<unsigned> tileOutBatchAndFieldEnd = { batchEnd };
  tileOutBatchAndFieldEnd.insert(tileOutBatchAndFieldEnd.end(),
                                 slice.outFieldEnd.begin(),
                                 slice.outFieldEnd.end());
  for (unsigned cg = cgBegin; cg < cgEnd; ++cg) {
    for (unsigned ozg = outZGroupBegin; ozg < outZGroupEnd; ++ozg) {
      std::vector<std::size_t> sliceBegin = {
        cg, ozg
      };
      std::vector<std::size_t> sliceEnd = {
        cg + 1, ozg + 1
      };
      sliceBegin.insert(sliceBegin.end(), tileOutBatchAndFieldBegin.begin(),
                        tileOutBatchAndFieldBegin.end());
      sliceEnd.insert(sliceEnd.end(), tileOutBatchAndFieldEnd.begin(),
                      tileOutBatchAndFieldEnd.end());
      sliceBegin.push_back(0);
      sliceEnd.push_back(out.dim(out.rank() - 1));
      auto o = out.slice(sliceBegin, sliceEnd).flatten();
      graph.setTileMapping(o, tile);
      outWindow.push_back(o);
    }
  }
  std::vector<unsigned> tileInBatchAndFieldBegin = { batchBegin };
  std::vector<unsigned> tileInBatchAndFieldEnd = { batchEnd };
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto kernelIndex = slice.kernelBegin[dim];
    assert(slice.kernelEnd[dim] == kernelIndex + 1);
    auto range = getInputRange(dim,
                               {slice.outFieldBegin[dim],
                                slice.outFieldEnd[dim]},
                               kernelIndex,
                               params);
    tileInBatchAndFieldBegin.push_back(range.first);
    tileInBatchAndFieldEnd.push_back(range.second);
  }

  std::vector<Tensor> inWindow;
  for (unsigned cg = cgBegin; cg < cgEnd; ++cg) {
    for (unsigned izg = inZGroupBegin; izg < inZGroupEnd; ++izg) {
      std::vector<std::size_t> sliceBegin = {
        cg, izg
      };
      std::vector<std::size_t> sliceEnd = {
        cg + 1, izg + 1
      };
      sliceBegin.insert(sliceBegin.end(), tileInBatchAndFieldBegin.begin(),
                        tileInBatchAndFieldBegin.end());
      sliceEnd.insert(sliceEnd.end(), tileInBatchAndFieldEnd.begin(),
                      tileInBatchAndFieldEnd.end());
      sliceBegin.push_back(0);
      sliceEnd.push_back(in.dim(in.rank() - 1));
      inWindow.push_back(in.slice(sliceBegin, sliceEnd).flatten());
    }
  }

  std::vector<Tensor> weightsWindow;
  for (unsigned cg = cgBegin; cg < cgEnd; ++cg) {
    for (unsigned ozg = outZGroupBegin; ozg < outZGroupEnd; ++ozg) {
      for (unsigned izg = inZGroupBegin; izg < inZGroupEnd; ++izg) {
        std::vector<std::size_t> sliceBegin = {
          cg, ozg, izg
        };
        std::vector<std::size_t> sliceEnd = {
          cg + 1, ozg + 1, izg + 1
        };
        sliceBegin.insert(sliceBegin.end(), slice.kernelBegin.begin(),
                          slice.kernelBegin.end());
        sliceEnd.insert(sliceEnd.end(), slice.kernelEnd.begin(),
                        slice.kernelEnd.end());
        sliceBegin.push_back(0);
        sliceEnd.push_back(weights.dim(weights.rank() - 2));
        sliceBegin.push_back(0);
        sliceEnd.push_back(weights.dim(weights.rank() - 1));
        weightsWindow.push_back(
          weights.slice(sliceBegin,sliceEnd).flatten()
        );
      }
    }
  }

  unsigned numEdges = outWindow.size() + inWindow.size() + weightsWindow.size();

  const auto contextsPerVertex = target.getNumWorkerContexts();
  std::vector<std::vector<unsigned>> worklist(contextsPerVertex);
  std::vector<unsigned> tileConvOutBegin;
  std::vector<unsigned> tileConvOutSize;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto dimOutRange =
        getOutputRange(dim, {slice.outFieldBegin[dim], slice.outFieldEnd[dim]},
                       {slice.kernelBegin[dim], slice.kernelEnd[dim]}, params);
    if (dimOutRange.first == dimOutRange.second)
      return;
    tileConvOutBegin.push_back(dimOutRange.first);
    tileConvOutSize.push_back(dimOutRange.second - dimOutRange.first);
  }
  auto workerPartition =
      partitionConvPartialByWorker(batchEnd - batchBegin, tileConvOutSize,
                                   contextsPerVertex, params.inputDilation);
  for (unsigned i = 0; i != contextsPerVertex; ++i) {
    for (const auto &partialRow : workerPartition[i]) {
      std::vector<unsigned> outBeginIndices = {
        partialRow.b + batchBegin
      };
      std::vector<unsigned> inBeginIndices = {
        partialRow.b + batchBegin
      };
      bool inBeginIndicesValid = true;
      for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
        const auto dimOutIndex = partialRow.outerFieldIndices[dim] +
                                 tileConvOutBegin[dim];
        const auto dimInIndex =
            getInputIndex(dim, dimOutIndex, slice.kernelBegin[dim], params);
        if (dimInIndex == ~0U) {
          inBeginIndicesValid = false;
          break;
        }
        outBeginIndices.push_back(dimOutIndex);
        inBeginIndices.push_back(dimInIndex);
      }
      if (!inBeginIndicesValid)
        continue;
      auto workerOutXBegin = tileConvOutBegin.back() + partialRow.xBegin;
      auto workerOutXEnd = tileConvOutBegin.back() + partialRow.xEnd;
      std::tie(workerOutXBegin, workerOutXEnd) =
          getOutputRange(numFieldDims - 1, {workerOutXBegin, workerOutXEnd}, 0,
                         params);
      const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
      if (workerOutWidth == 0)
        continue;
      unsigned workerInXBegin, workerInXEnd;
      std::tie(workerInXBegin, workerInXEnd) =
          getInputRange(numFieldDims - 1, {workerOutXBegin, workerOutXEnd}, 0,
                        params);
      inBeginIndices.push_back(workerInXBegin);
      outBeginIndices.push_back(workerOutXBegin);
      const auto outBeginOffset =
          flattenIndexInSlice(tileOutBatchAndFieldBegin,
                              tileOutBatchAndFieldEnd,
                              outBeginIndices);
      const auto inBeginOffset =
          flattenIndexInSlice(tileInBatchAndFieldBegin,
                              tileInBatchAndFieldEnd,
                              inBeginIndices);
      worklist[i].push_back(outBeginOffset);
      worklist[i].push_back(workerOutWidth);
      worklist[i].push_back(inBeginOffset);
    }
  }
  auto v =
    graph.addVertex(fwdCS,
                    templateVertex("popconv::ConvPartial1x1Out",
                                   dType,
                                   plan.getPartialType(),
                                   useDeltaEdgesForConvPartials(numEdges) ?
                                                          "true" : "false"));
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroups"], outZGroupEnd - outZGroupBegin);
  graph.setInitialValue(v["numInGroups"], inZGroupEnd - inZGroupBegin);
  graph.setInitialValue(v["inStride"], params.stride.back());
  graph.setInitialValue(v["numConvGroups"], cgEnd - cgBegin);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["convUnitInputLoadElemsPerCycle"],
                        convUnitInputLoadElemsPerCycle);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  graph.setInitialValue(v["numWorkerContexts"], contextsPerVertex);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstantTensor("unsigned", {worklist[i].size()},
                                     worklist[i].data());
    graph.connect(v["worklists"][i], t);
  }
  graph.setTileMapping(v, tile);
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
  auto dType = out.elementType() == "float";
  const auto grainSize = dType ? target.getFloatVectorWidth() :
                                 target.getHalfVectorWidth();
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

static void
createConvPartialnx1Vertex(Graph &graph,
                           const Plan &plan,
                           std::string dType,
                           unsigned tile,
                           const ConvSlice &slice,
                           ConvParams params,
                           ComputeSet fwdCS,
                           Tensor in, Tensor weights, Tensor out) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto &target = graph.getTarget();
  const auto dataPathWidth = target.getDataPathWidth();
  const bool floatActivations = dType == "float";
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(floatActivations);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  const auto batchBegin = slice.batchBegin;
  const auto batchEnd = slice.batchEnd;
  const auto outZGroupBegin = slice.outChanGroupBegin;
  const auto outZGroupEnd = slice.outChanGroupEnd;
  const auto cgBegin = slice.cgBegin;
  const auto cgEnd = slice.cgEnd;
  const auto inZGroupBegin = slice.inChanGroupBegin;
  const auto inZGroupEnd = slice.inChanGroupEnd;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  bool flipOut = params.getPaddedDilatedKernelSize(numFieldDims - 1) >
                 params.getPaddedDilatedInputSize(numFieldDims - 1);
  const auto convUnitInputLoadElemsPerCycle =
    target.getConvUnitInputLoadElemsPerCycle (dType == "float");
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
  if (convUnitWeightHeight != 1 && params.inputDilation.front() > 1) {
    const auto inputDilation = params.inputDilation.front();
    in = dilate(graph, in, inputDilation, 3);
    params.inputDilation[0] = 1;
    params.inputFieldShape[0] =
        (params.inputFieldShape[0] - 1) * inputDilation + 1;
  }
  std::vector<std::size_t> tileInBatchAndFieldBegin = { batchBegin };
  std::vector<std::size_t> tileInBatchAndFieldEnd = { batchEnd };
  // This stride is whats use to move down one element in the input field by
  // the vertex.
  unsigned inRowStride = inChansPerGroup * params.kernelDilation.front();
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


  int dilatedPadding = params.inputPaddingLower[0] * params.kernelDilation[0];
  unsigned prePaddingSize =
      std::max<int>(dilatedPadding -
                    static_cast<int>(tileInBatchAndFieldBegin[1]), 0);
  // If the kernel is larger than the field, padding is required before the
  // field.
  auto largeKernel = params.getPaddedDilatedKernelSize(0) >
                     params.getPaddedDilatedInputSize(0);
  if (largeKernel) {
    prePaddingSize =
        std::max((convUnitWeightHeight - 1) * params.kernelDilation.front(),
                 prePaddingSize);
  }
  // If we are doing an nx1 convolution need to pad the bottom of the
  // input field for convolutions that "run off the end".
  auto postPaddingSize =
      (convUnitWeightHeight - 1) * params.kernelDilation.front();

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
      auto convOutRange =
          getOutputRange(dim, {slice.outFieldBegin[dim],
                               slice.outFieldEnd[dim]},
                         {kernelBeginIndices[dim],
                          kernelBeginIndices[dim] +
                          (dim == 0 ? convUnitWeightHeight : 1)
                         },
                         params);
      tileConvOutBegin.push_back(convOutRange.first);
      tileConvOutSize.push_back(convOutRange.second - convOutRange.first);
    }
    if (product(tileConvOutSize) == 0)
      continue;
    auto workerPartition =
        partitionConvPartialByWorker(batchEnd - batchBegin,
                                     tileConvOutSize,
                                     contextsPerVertex, params.inputDilation);
    for (unsigned i = 0; i != contextsPerVertex; ++i) {
      for (const auto &partialRow : workerPartition[i]) {
        auto workerOutXBegin = tileConvOutBegin.back() + partialRow.xBegin;
        auto workerOutXEnd = tileConvOutBegin.back() + partialRow.xEnd;
        std::tie(workerOutXBegin, workerOutXEnd) =
            getOutputRange(numFieldDims - 1, {workerOutXBegin, workerOutXEnd},
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
          const auto koBegin = kernelBeginIndices[0];
          const auto outOuterIndex = tileConvOutBegin[0] +
                                     partialRow.outerFieldIndices[0];
          for (unsigned j = 0; j != convUnitWeightHeight; ++j) {
            int inOuterIndex = getInputIndex(0, outOuterIndex, koBegin + j,
                                             params);
            if (inOuterIndex != ~0U) {
              auto inOuterBeginIndex = inOuterIndex + prePaddingSize -
                                       j * params.kernelDilation.front();
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

  auto v = graph.addVertex(fwdCS,
                           templateVertex("popconv::ConvPartialnx1",
                                          dType, plan.getPartialType(), "false",
                                        useDeltaEdgesForConvPartials(numEdges) ?
                                                          "true" : "false"));
  const auto outStrideX = params.inputDilation.back();
  unsigned kernelInnerElements = 1;
  for (unsigned dim = 1; dim < numFieldDims; ++dim) {
    kernelInnerElements *= subKernelPositionsEnd[dim] -
                           subKernelPositionsBegin[dim];
  }
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroups"], outZGroupEnd - outZGroupBegin);
  graph.setInitialValue(v["numInGroups"], inZGroupEnd - inZGroupBegin);
  graph.setInitialValue(v["kernelInnerElements"], kernelInnerElements);
  graph.setInitialValue(v["kernelOuterSize"], subKernelPositionsEnd[0]);
  graph.setInitialValue(v["inStride"], params.stride.back()) ;
  graph.setInitialValue(v["outStride"], outStrideX);
  graph.setInitialValue(v["numConvGroups"], cgEnd - cgBegin);
  graph.setInitialValue(v["flipOut"], flipOut);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["ampKernelHeight"], convUnitWeightHeight);
  graph.setInitialValue(v["inRowStride"], inRowStride);
  graph.setInitialValue(v["convUnitInputLoadElemsPerCycle"],
                        convUnitInputLoadElemsPerCycle);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  graph.setInitialValue(v["numWorkerContexts"], contextsPerVertex);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstantTensor("unsigned", {worklist[i].size()},
                                     worklist[i].data());
    graph.connect(v["worklists"][i], t);
  }

  const auto zeroWorklist = createZeroWorklist(target, outWindow[0]);
  auto zeroWorklistTensor = graph.addConstantTensor("unsigned",
                                                    {zeroWorklist.size()},
                                                    zeroWorklist.data());
  graph.connect(v["zeroWorklist"], zeroWorklistTensor);
  graph.setTileMapping(v, tile);
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


static void
createConvPartialHorizontalMacVertex(Graph &graph,
                                     const Plan &plan,
                                     const std::string &dType,
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
  const auto outZGroupBegin = slice.outChanGroupBegin;
  const auto outZGroupEnd = slice.outChanGroupEnd;
  const auto cgBegin = slice.cgBegin;
  const auto cgEnd = slice.cgEnd;
  const auto inZGroupBegin = slice.inChanGroupBegin;
  const auto inZGroupEnd = slice.inChanGroupEnd;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;

  const auto kernelXBegin = 0U;
  const auto kernelXEnd = weights.dim(weights.rank() - 3);
  assert(kernelXBegin == slice.kernelBegin.back());
  assert(kernelXEnd == slice.kernelEnd.back());

  bool flipOut = params.getPaddedDilatedKernelSize(xDimIndex) >
                 params.getPaddedDilatedInputSize(xDimIndex);

  const auto dataPathWidth = target.getDataPathWidth();

  assert(outChansPerGroup == 1);
  if (dType == "half") {
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
        getOutputRange(dim, {slice.outFieldBegin[dim], slice.outFieldEnd[dim]},
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
          getOutputRange(xDimIndex, {outXBegin, outXEnd}, kx, params);
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
              getOutputRange(xDimIndex, {workerOutXBegin, workerOutXEnd},
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

  const unsigned outStrideX = params.inputDilation.back();
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
  graph.setInitialValue(v["inStride"], params.stride.back());
  graph.setInitialValue(v["outStride"], outStrideX);
  graph.setInitialValue(v["numConvGroups"], cgEnd - cgBegin);
  graph.setInitialValue(v["flipOut"], flipOut);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["numWorkerContexts"], contextsPerVertex);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0;i < worklist.size(); ++i) {
    auto t = graph.addConstantTensor("unsigned", {worklist[i].size()},
                                     worklist[i].data());
    graph.connect(v["worklists"][i], t);
  }
  const auto zeroWorklist = createZeroWorklist(target, outWindow[0]);
  for (unsigned i = 0; i != zeroWorklist.size(); ++i) {
  }
  auto zeroWorklistTensor = graph.addConstantTensor("unsigned",
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
  assert(product(params.stride) == 1);
  assert(product(params.inputDilation) == 1);

  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    in = pad(graph, in, params.inputPaddingLower[dim],
             params.inputPaddingUpper[dim], 3 + dim);
    weights = pad(graph, weights, params.kernelPaddingLower[dim],
                  params.kernelPaddingUpper[dim], 3 + dim);
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

static void
addFlattenedPartialSumRegions(const std::vector<std::size_t> &actsShape,
                              const ConvSlice &slice,
                              const ConvParams &params,
                              std::vector<Interval<std::size_t>> &regions) {
  assert(actsShape.size() >= 4);
  const auto numFieldDims = actsShape.size() - 4;
  assert(slice.outFieldBegin.size() == numFieldDims);
  assert(slice.outFieldEnd.size() == numFieldDims);
  assert(slice.kernelBegin.size() == numFieldDims);
  assert(slice.kernelEnd.size() == numFieldDims);
  std::vector<std::size_t> sliceBegin = {
    slice.cgBegin,
    slice.outChanGroupBegin,
    slice.batchBegin
  };
  std::vector<std::size_t> sliceEnd = {
    slice.cgEnd,
    slice.outChanGroupEnd,
    slice.batchEnd
  };
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    sliceBegin.push_back(slice.outFieldBegin[dim]);
    sliceEnd.push_back(slice.outFieldEnd[dim]);
  }
  sliceBegin.push_back(0);
  sliceEnd.push_back(actsShape.back());
  addFlattenedRegions(actsShape, sliceBegin, sliceEnd, regions);
}

static void
mapPartialSums(Graph &graph, const ConvSlice &slice, unsigned tile,
               const Tensor &out) {
  assert(out.rank() >= 4);
  const auto numFieldDims = out.rank() - 4;
  assert(slice.outFieldBegin.size() == numFieldDims);
  assert(slice.outFieldEnd.size() == numFieldDims);
  assert(slice.kernelBegin.size() == numFieldDims);
  assert(slice.kernelEnd.size() == numFieldDims);
  std::vector<std::size_t> sliceBegin = {
    slice.cgBegin,
    slice.outChanGroupBegin,
    slice.batchBegin
  };
  std::vector<std::size_t> sliceEnd = {
    slice.cgEnd,
    slice.outChanGroupEnd,
    slice.batchEnd
  };
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    sliceBegin.push_back(slice.outFieldBegin[dim]);
    sliceEnd.push_back(slice.outFieldEnd[dim]);
  }
  sliceBegin.push_back(0);
  sliceEnd.push_back(out.dim(out.rank() - 1));
  std::vector<Interval<std::size_t>> regions;
  addFlattenedRegions(out.shape(), sliceBegin, sliceEnd, regions);
  Tensor flatOut = out.flatten();
  for (const auto &region : regions) {
    graph.setTileMapping(flatOut.slice(region.begin(), region.end()), tile);
  }
}

static bool writtenRangeEqualsOutputRange(
    unsigned dim,
    std::pair<unsigned, unsigned> outRange,
    std::pair<unsigned, unsigned> kernelIndexRange,
    const ConvParams &params) {
  auto writtenRange =
      getOutputRange(dim, outRange, kernelIndexRange, params);
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

static void
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      std::string dType,
                      unsigned tile,
                      const ConvSlice &slice,
                      const ConvParams &params,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out) {
  const auto outZGroupBegin = slice.outChanGroupBegin;
  const auto outZGroupEnd = slice.outChanGroupEnd;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto &target = graph.getTarget();

  bool useConvPartial1x1OutVertex = false;
  switch (plan.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    {
      const auto partialsType = out.elementType();
      const auto outChansPerPass = getNumConvUnits(dType == "float",
                                                   partialsType == "float",
                                                   target);
      assert(outChansPerGroup % outChansPerPass == 0);
      const auto passesPerOutputGroup = outChansPerGroup / outChansPerPass;
      auto equalsOne = [](unsigned x) { return x == 1; };
      bool nx1Vertex =
          getNumElementsInSlice(slice.kernelBegin, slice.kernelEnd) != 1 ||
          !std::all_of(params.inputDilation.begin(), params.inputDilation.end(),
                       equalsOne) ||
          !writtenRangeEqualsOutputRange(slice.outFieldBegin, slice.outFieldEnd,
                                         slice.kernelBegin, slice.kernelEnd,
                                         params);
      useConvPartial1x1OutVertex = !nx1Vertex &&
                                   passesPerOutputGroup == 1;
    }
    break;
  case Plan::Method::MAC:
  case Plan::Method::OUTER_PRODUCT:
    break;
  }

  if (useConvPartial1x1OutVertex) {
    createConvPartial1x1OutVertex(graph, plan, dType, tile, slice, params,
                                  fwdCS, in, weights, out);
  } else {
    mapPartialSums(graph, slice, tile, out);
    switch (plan.method) {
    default: assert(0 && "Unexpected method");
    case Plan::Method::AMP:
      createConvPartialnx1Vertex(graph, plan, dType, tile, slice, params,
                                 fwdCS, in, weights, out);
      break;
    case Plan::Method::MAC:
      createConvPartialHorizontalMacVertex(graph, plan, dType, tile,
                                           slice, params,fwdCS, in, weights,
                                           out);
      break;
    case Plan::Method::OUTER_PRODUCT:
      {
        const auto cgBegin = slice.cgBegin;
        const auto cgEnd = slice.cgEnd;
        const auto outXBegin = slice.outFieldBegin.back();
        const auto outXEnd = slice.outFieldEnd.back();
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
}

static Program
calcPartialSums(Graph &graph,
                const Plan &plan,
                const ConvParams &params,
                std::string dType,
                Tensor in, Tensor weights, Tensor partials,
                const std::string &layerName) {
  ComputeSet convolveCS = graph.addComputeSet(layerName + "/Convolve");
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const ConvTileIndices &indices,
                           const ConvSlice &slice) {
    if (slice.outChanGroupBegin == slice.outChanGroupEnd ||
        slice.cgBegin == slice.cgEnd)
      return;
    const auto numFieldDims = params.getNumFieldDims();
    unsigned partialIndex = indices.ic;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      partialIndex = partialIndex * plan.kernelTileSplit[dim] +
                     indices.kernel[dim];
    }
    calcPartialConvOutput(graph, plan, dType, tile, slice, params,
                          convolveCS, in, weights, partials[partialIndex]);
  });
  Sequence prog;
  prog.add(Execute(convolveCS));
  return prog;
}

static Tensor
partialGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval<std::size_t>>> &
        tileGroupRegions,
    const Tensor &partials,
    unsigned outDepth,
    const std::string &resultType,
    ComputeSet cs) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;
  Tensor out = graph.addTensor(resultType,
                               outDims,
                               "partialReduceOut");
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numTileGroups = tileGroupRegions.size();
  const unsigned minGrainSize =
    resultType == "float" ? target.getFloatVectorWidth() :
                            target.getHalfVectorWidth();
  const unsigned partialChansPerGroup = partials.dim(partials.rank()-1);
  const auto grainSize = std::max(partialChansPerGroup, minGrainSize);

  for (unsigned i = 0; i != outDepth; ++i) {
    unsigned begin = (i * partialsDepth) / outDepth;
    unsigned end = ((i + 1) * partialsDepth) / outDepth;
    std::vector<std::vector<Interval<std::size_t>>>
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
                std::vector<Interval<std::size_t>>
              > &tileGroupRegions,
              const Tensor &partials,
              const std::string &resultType,
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
    const std::vector<std::vector<Interval<std::size_t>>> &
        tileGroupRegions,
    Tensor partials,
    const std::string &resultType,
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
                        const std::string &resultType,
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
  std::map<std::set<unsigned>, std::vector<Interval<std::size_t>>>
      tilesToOutputs;
  for (const auto &entry : outputToTiles) {
    tilesToOutputs[entry.second].emplace_back(entry.first.lower(),
                                              entry.first.upper());
  }
  std::vector<std::vector<unsigned>> tileGroups;
  std::vector<std::vector<Interval<std::size_t>>> tileGroupRegions;
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
  const auto numBatchGroups = in.dim(2);
  const auto dType = in.elementType();
  const auto outNumChans = weights.dim(1) * weights.dim(weights.rank() - 2);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto inChanTileSplit = plan.inChanTileSplit;
  const auto kernelTileSplit = product(plan.kernelTileSplit);

  const auto partialType = plan.getPartialType();

  // Calculate a set of partial sums of the convolutions.
  std::vector<std::size_t> partialsShape = {
    inChanTileSplit * kernelTileSplit,
    params.getNumConvGroups(),
    partialNumChanGroups,
    numBatchGroups
  };
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    partialsShape.push_back(params.getOutputSize(dim));
  }
  partialsShape.push_back(partialChansPerGroup);
  Tensor partials = graph.addTensor(partialType, partialsShape, "partials");
  prog.add(calcPartialSums(graph, plan, params, dType, in, weights, partials,
                           debugPrefix));
  std::vector<ComputeSet> reduceComputeSets;
  // For each element of the batch, we add the reduction vertices to same
  // compute sets so the batch will be executed in parallel.
  Tensor reduced;
  // Perform the reduction of partial sums.
  if (partials.dim(0) == 1) {
    if (dType != partialType) {
      reduced = graph.addTensor(dType, partials.shape(), "reduced");
      if (reduceComputeSets.empty()) {
        reduceComputeSets.push_back(graph.addComputeSet(debugPrefix +
                                                           "/Cast"));
      }
      applyTensorMapping(graph, reduced, graph.getTileMapping(partials));
      cast(graph, partials, reduced, reduceComputeSets[0]);
    } else {
      reduced = partials;
    }
    reduced = reduced[0];
  } else {
    reduced = multiStageGroupedReduce(graph, partials, dType, reduceComputeSets,
                                      debugPrefix);
  }
  for (const auto &cs : reduceComputeSets) {
    prog.add(Execute(cs));
  }
  return unsplitActivationChanGroups(reduced);
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
  if (std::any_of(params.stride.begin(), params.stride.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_stride" + getShapeAsString(params.stride);
  }
  if (std::any_of(params.inputDilation.begin(), params.inputDilation.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_inDilation" + getShapeAsString(params.inputDilation);
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
  auto postExpandParams = originalParams;
  if (originalPlan.swapOperands) {
    swapOperands(postExpandParams);
  }
  for (auto dim : originalPlan.expandDims) {
    expandSpatialDim(graph, postExpandParams, nullptr, nullptr, dim);
  }
  auto postOutChanFlattenParams = postExpandParams;
  for (auto dim : originalPlan.outChanFlattenDims) {
    postOutChanFlattenParams.outputChannels *=
        postOutChanFlattenParams.kernelShape[dim];
    postOutChanFlattenParams.kernelShape[dim] = 1;
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
    const auto spatialDimSize = originalParams.getOutputSize(spatialDim);
    activations =
        unflattenDims(activations, 2 + spatialDim, activations.rank() - 1,
                      spatialDimSize);
  }
  // Undo the swapping of operands.
  if (originalPlan.swapOperands) {
    activations = activations.dimShufflePartial({1, activations.rank() - 1},
                                                {activations.rank() - 1, 1});
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
  const auto inNumDests = std::accumulate(plan.kernelTileSplit.begin(),
                                          plan.kernelTileSplit.end(),
                                          1U,
                                          std::multiplies<unsigned>()) *
                          plan.outChanTileSplit;
  if (inNumDests > inViewMaxBroadcastDests) {
    auto inRearranged = createInputImpl(graph, params, "inRearranged", plan);
    prog.add(Copy(in, inRearranged));
    in = inRearranged;
  }
  auto weightsNumDests = plan.batchTileSplit;
  for (const auto split : plan.fieldTileSplit) {
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
  if (params.dType == "float") {
    const auto floatVectorWidth = target.getFloatVectorWidth();
    auto macCycles =
        static_cast<double>(numMacs) / (floatVectorWidth * numTiles);
    return macCycles;
  }
  assert(params.dType == "half");
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
  auto out = graph.addTensor(dType, outShape, "partialTranspose");
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
          graph.addVertex(cs, templateVertex("popconv::Transpose2D", dType));
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
  for (unsigned dim = 0; dim != fwdParams.getNumFieldDims(); ++dim) {
    if (fwdParams.kernelPaddingLower[dim] != 0 ||
        fwdParams.kernelPaddingUpper[dim] != 0) {
      // Kernel padding in the forward pass translates to output truncation in
      // the weight update pass but this isn't supported yet.
      std::abort();
    }
    if (fwdParams.getPaddedDilatedKernelSize(dim) >
        fwdParams.getPaddedDilatedInputSize(dim)) {
      // If the kernel is larger than the input we need to zero pad the
      // activations - not supported for now.
      std::abort();
    }
  }
  std::vector<int> noPadding(numFieldDims);
  return ConvParams(fwdParams.dType,
                    fwdParams.getNumInputChansPerConvGroup(), // batchSize
                    {
                      fwdParams.inputFieldShape
                    }, // inputFieldShape
                    {
                      fwdParams.getOutputFieldShape()
                    }, // kernelShape
                    fwdParams.getBatchSize(), // inputChannels
                    fwdParams.getNumOutputChansPerConvGroup(), // outputChannels
                    fwdParams.kernelDilation, // stride
                    fwdParams.inputPaddingLower, // inputPaddingLower
                    fwdParams.inputPaddingUpper, // inputPaddingUpper
                    fwdParams.inputDilation,     // inputDilation
                    noPadding, // kernelPaddingLower
                    noPadding, // kernelPaddingUpper
                    fwdParams.stride, // kernelDilation
                    fwdParams.numConvGroups // numConvGroups
                    );
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
                  const std::string &partialsType,
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
  std::vector<std::map<unsigned, std::vector<Interval<std::size_t>>>>
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
    Tensor r = graph.addTensor(partialsType, {tileNumGroups, inChansPerGroup},
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
      graph.addTensor(partialsType, {usedWorkers, maxOutputsPerWorker},
                      "partials");
  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
    auto tile = worker / target.getNumWorkerContexts();
    graph.setTileMapping(partials[worker].slice(0, maxOutputsPerWorker), tile);
    unsigned outBegin = (worker  * numOut) / usedWorkers;
    unsigned outEnd = ((worker  + workersPerOutput) * numOut) / usedWorkers;
    if (outBegin == outEnd)
      continue;
    unsigned numWorkerOutputs = outEnd - outBegin;
    auto toReduce = graph.addTensor(partialsType, {0});
    std::vector<unsigned> numInputsPerOutput;
    for (auto o = outBegin; o != outEnd; ++o) {
      auto outGroup = o / inChansPerGroup;
      auto outInGroup = o % inChansPerGroup;
      auto inputs = graph.addTensor(partialsType, {0});
      for (unsigned srcTile = 0; srcTile < numTiles; ++srcTile) {
        auto it = tileLocalReductions[srcTile].find(outGroup);
        if (it == tileLocalReductions[srcTile].end())
          continue;
        unsigned i = std::distance(tileLocalReductions[srcTile].begin(),
                                   it);
        auto srcBias = tileReduced[srcTile][i][outInGroup];
        inputs = append(inputs, srcBias);
      }
      const auto numInputs = inputs.numElements();
      auto inBegin =
          ((worker  % workersPerOutput) * numInputs) / workersPerOutput;
      unsigned inEnd =
          (((worker  % workersPerOutput) + 1) * numInputs) / workersPerOutput;
      toReduce = concat(toReduce, inputs.slice(inBegin, inEnd));
      numInputsPerOutput.push_back(inEnd - inBegin);
    }
    if (toReduce.numElements() == 0) {
      auto v = graph.addVertex(computeSets[1],
                               templateVertex("popstd::Zero", partialsType));
      graph.connect(v["out"], partials[worker].slice(0, maxOutputsPerWorker));
      graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
      graph.setTileMapping(v, tile);
      continue;
    }
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
                const std::string &partialsType,
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
                      const std::string &partialsType,
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
  auto plan = getPlan(graph, params, options);
  auto fwdPlan = plan;
  std::swap(fwdPlan.fieldAxisGrainSize.back(), fwdPlan.inChansPerGroup);
  std::swap(fwdPlan.fieldTileSplit.back(), fwdPlan.inChanTileSplit);
  Tensor transposed = createInput(graph, params, "transposed", options);
  // split activations into conv groups
  auto splitActivations =
      actsToInternalShape(activations, params.getNumConvGroups());
  auto splitTransposed =
      actsToInternalShape(transposed, params.getNumConvGroups());
  auto splitTransposedUngroupedShape = splitTransposed.shape();
  const auto fwdGroupSize =
      getInChansPerGroup(fwdPlan,
                         static_cast<unsigned>(splitActivations.dim(4)));
  const auto bwdGroupSize =
      getInChansPerGroup(plan, static_cast<unsigned>(splitActivations.dim(3)));
  const auto dType = activations.elementType();
  const auto &target = graph.getTarget();
  splitActivations =
      splitActivations.reshape({splitActivations.dim(0),
                                splitActivations.dim(1) *
                                  splitActivations.dim(2),
                                splitActivations.dim(3) / bwdGroupSize,
                                bwdGroupSize,
                                splitActivations.dim(4) / fwdGroupSize,
                                fwdGroupSize})
                      .dimShufflePartial({3}, {4});
  splitTransposed =
      splitTransposed.reshape({splitTransposed.dim(0),
                               splitTransposed.dim(1) *
                                 splitTransposed.dim(2),
                               splitTransposed.dim(3) / fwdGroupSize,
                               fwdGroupSize,
                               splitTransposed.dim(4) / bwdGroupSize,
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
                          templateVertex("popconv::Transpose2D", dType));
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
                               const std::string &invStdDevType,
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
                   const std::string &partialsType,
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
                const std::string &partialsType,
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
                          const std::string &partialsType,
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
