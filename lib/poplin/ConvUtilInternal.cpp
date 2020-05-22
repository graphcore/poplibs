// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "ConvUtilInternal.hpp"
#include "poplibs_support/StructHelper.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/ConvUtil.hpp"
#include "popops/Rearrange.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/icl/interval_map.hpp>
#include <boost/optional.hpp>
#include <cassert>
#include <cmath>
#include <poplar/Tensor.hpp>
#include <unordered_map>

using namespace poplar;
using namespace poputil;

namespace poplin {

// Return a convolution where the same input, kernel and output size match the
// specified convolution and where the output is all zero.
ConvParams getZeroConv(const ConvParams &params) {
  // We represent the zero convolution as follows:
  // - truncate the input and the kernel to size zero.
  // - zero pad the input and the kernel to size one.
  // - convolve the input and kernel resulting in an output of size one.
  // - truncate the output to size zero.
  // - pad the output to match the expected output size.
  ConvParams zeroConv = params;
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> allZeros(numFieldDims, 0);
  std::vector<unsigned> allOnes(numFieldDims, 1);
  std::vector<bool> allFalse(numFieldDims, false);
  zeroConv.inputTransform.truncationLower = allZeros;
  zeroConv.inputTransform.truncationUpper =
      vectorConvert<unsigned>(params.inputFieldShape);
  zeroConv.inputTransform.dilation = allOnes;
  zeroConv.inputTransform.paddingLower = allOnes;
  zeroConv.inputTransform.paddingUpper = allZeros;
  zeroConv.inputTransform.flip = allFalse;
  zeroConv.kernelTransform.truncationLower = allZeros;
  zeroConv.kernelTransform.truncationUpper =
      vectorConvert<unsigned>(params.kernelShape);
  zeroConv.kernelTransform.dilation = allOnes;
  zeroConv.kernelTransform.paddingLower = allOnes;
  zeroConv.kernelTransform.paddingUpper = allZeros;
  zeroConv.kernelTransform.flip = allFalse;
  zeroConv.outputTransform.truncationLower = allZeros;
  zeroConv.outputTransform.truncationUpper = allOnes;
  zeroConv.outputTransform.stride = allOnes;
  zeroConv.outputTransform.paddingLower = allZeros;
  zeroConv.outputTransform.paddingUpper =
      vectorConvert<unsigned>(params.getOutputFieldShape());
  assert(zeroConv.getOutputFieldShape() == params.getOutputFieldShape());
  return zeroConv;
}

std::vector<std::vector<PartialRow>> partitionConvPartialByWorker(
    unsigned batchElements, const std::vector<unsigned> &tileConvOutSize,
    unsigned numContexts, const std::vector<unsigned> &inputDilation,
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
      (tileConvOutSize.back() + outputStride.back() - 1) / outputStride.back();
  unsigned activeRows = 1;
  std::vector<unsigned> activeRowShape;
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    auto dimActiveRows =
        (tileConvOutSize[dim] + outputStride[dim] - 1) / outputStride[dim];
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
    auto beginIndices = poputil::unflattenIndex<std::size_t>(
        {batchElements, activeRows, elementsPerRow}, beginElement);
    auto lastIndices = poputil::unflattenIndex<std::size_t>(
        {batchElements, activeRows, elementsPerRow}, lastElement);
    for (unsigned b = beginIndices[0]; b != lastIndices[0] + 1; ++b) {
      unsigned activeRowBegin = b == beginIndices[0] ? beginIndices[1] : 0;
      unsigned activeRowLast =
          b == lastIndices[0] ? lastIndices[1] : activeRows - 1;
      for (unsigned activeRow = activeRowBegin; activeRow != activeRowLast + 1;
           ++activeRow) {
        unsigned activeXBegin =
            b == beginIndices[0] && activeRow == beginIndices[1]
                ? beginIndices[2]
                : 0;
        unsigned activeXLast =
            b == lastIndices[0] && activeRow == lastIndices[1]
                ? lastIndices[2]
                : elementsPerRow - 1;
        auto outerFieldIndices =
            poputil::unflattenIndex(activeRowShape, activeRow);
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

// Reshape the activations tensor from [N][G * C]... shape to
// [G][N]...[C] where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
Tensor actsToInternalShape(const Tensor &act, unsigned numConvGroups,
                           unsigned chansPerGroup) {
  return act.reshapePartial(1, 2, {numConvGroups, chansPerGroup})
      .dimShufflePartial({1, 2}, {0, act.rank()});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [N][G * C]... shape where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
Tensor actsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({0, act.rank() - 1}, {1, 2}).flatten(1, 3);
}

// Reshape the weights tensor from [G][OC][IC]... shape to
// [G]...[OC][IC].
Tensor weightsToInternalShape(const Tensor &act) {
  return act.dimShufflePartial({1, 2}, {act.rank() - 2, act.rank() - 1});
}

// Reshape the weights tensor from [G]...[OC][IC] shape to
// [G][OC][IC]... shape.
Tensor weightsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({act.rank() - 2, act.rank() - 1}, {1, 2});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [G1][C1][N]...[G2][C2]
//
// Where
//  G1 * G2 = G
//  C1 * C2 = C
Tensor splitActivationIntoGroups(Tensor act, const unsigned convGroupsPerGroup,
                                 const unsigned chansPerGroup) {
  {
    const auto cgDim = 0;
    const auto ciDim = act.rank() - 1;

    assert(act.dim(cgDim) % convGroupsPerGroup == 0);
    assert(act.dim(ciDim) % chansPerGroup == 0);

    // reshape [G][N]...[C] into [G1][G2][N]...[C1][C2]
    // (do inner-most dimension first so as to not invalidate the indices)
    auto shape = act.shape();
    shape[ciDim] /= chansPerGroup;
    shape.insert(std::begin(shape) + ciDim + 1, chansPerGroup);
    shape[cgDim] /= convGroupsPerGroup;
    shape.insert(std::begin(shape) + cgDim + 1, convGroupsPerGroup);

    act = act.reshape(shape);
  }

  // swap the G2 and C1 dims so we have the shape [G1][C1][N]...[G2][C2]
  const auto g2Dim = 1;
  const auto c1Dim = act.rank() - 2;
  return act.dimShufflePartial({g2Dim, c1Dim}, {c1Dim, g2Dim});
}

// Reshape the activations tensor from [G1][C1][N]...[G2][C2] shape to
// [G][N]...[C]
//
// Where
//  G1 * G2 = G
//  C1 * C2 = C
poplar::Tensor unsplitActivationFromGroups(poplar::Tensor act) {
  // this is the inverse of splitActivationIntoGroups.
  // swap the G2 and C1 dims so we have the shape [G1][G2][N]...[C1][C2]
  {
    const auto c1Dim = 1;
    const auto g2Dim = act.rank() - 2;
    act = act.dimShufflePartial({c1Dim, g2Dim}, {g2Dim, c1Dim});
  }

  // reshape the [G1][G2][N]...[C1][C2] into [G][N]...[C1][C2]
  const auto g1Dim = 0;
  const auto g2Dim = 1;
  const auto c1Dim = act.rank() - 2;
  const auto c2Dim = act.rank() - 1;

  const auto &shape = act.shape();
  std::vector<std::size_t> newShape;
  newShape.reserve(shape.size() - 2);

  newShape.push_back(shape[g1Dim] * shape[g2Dim]);
  newShape.insert(std::end(newShape), std::begin(shape) + 2,
                  std::end(shape) - 2);
  newShape.push_back(shape[c1Dim] * shape[c2Dim]);

  return act.reshape(newShape);
}

// generic function for extracting some of the innermost groupings. the first
// dimension is implicitly assumed to be the innermost dim, the rest of the
// dimensions are passed in as an array.
// for eg:
//  the shape of the weights is the internal shape: [G]...[OC][IC], the memory
//  layout should be [G1][OC1][IC1]...[G2][OC2][IC2]. to find these groupings
//  we must iteratively find the innermost grouping, split that dimension and
//  move next grouping dimension next and flatten, for eg.:
//   [OC][IC] -> [OC][IC1][IC2] -> [IC1][OC][IC2] -> [IC1][OC * IC2]
//  finding the innermost grouping on this will produce the grouping of the
//  product of IC2 and OC, we can find the grouping of OC2 by dividing this by
//  IC1. we then do the same thing again to find G2.
template <std::size_t N>
std::array<unsigned, N>
detectGroupings(const poplar::Graph &graph, poplar::Tensor t,
                const std::array<unsigned, N - 1> &dims) {
  const auto rank = t.rank();

  std::array<unsigned, N> groupings;
  groupings[0] = detectInnermostGrouping(graph, t);

  for (unsigned i = 0; i < dims.size(); ++i) {
    const auto innerDim = rank - 1;

    // split dimension D into [D1][D2]
    t = t.reshapePartial(innerDim, innerDim + 1,
                         {t.dim(innerDim) / groupings[i], groupings[i]});

    // move the next dimension we want to inspect in.
    t = t.dimShufflePartial({dims[i], innerDim}, {innerDim, dims[i]});

    // combine this dimension with the previous groupings into a single
    // innermost dimension.
    t = t.flatten(innerDim, innerDim + 2);

    // find the next grouping dimension, this will be the product of all of
    // the inner dimensions so far. we factor these out later to get the
    // individual groupings.
    groupings[i + 1] = detectInnermostGrouping(graph, t);

    // note: The result obtained is incorrect if partial elements of the product
    // are assigned to a tile. If a full product is not mapped, the next
    // grouping is conservatively set to be 1 which means that product grouping
    // will be the same as the previous level.
    if (groupings[i + 1] % groupings[i] != 0) {
      groupings[i + 1] = groupings[i];
    }
  }

  // the groupings array are now in the form of {x, xy, xyz} so we need to
  // divide by the earlier groupings to get the actual grouping of each dim.
  for (unsigned i = groupings.size() - 1; i > 0; --i) {
    groupings[i] /= groupings[i - 1];
  }

  return groupings;
}

ChannelGrouping detectChannelGrouping(const poplar::Graph &graph,
                                      const poplar::Tensor &acts) {
  const unsigned gDim = 0;

  const std::array<unsigned, 1> dims{gDim};
  const auto groupings = detectGroupings<2>(graph, acts, dims);

  return {groupings[1], groupings[0]};
}

WeightChannelGrouping detectWeightsChannelGrouping(const Graph &graph,
                                                   const Tensor &weights) {
  const auto rank = weights.rank();
  const unsigned gDim = 0;
  const unsigned ocDim = rank - 2;

  const std::array<unsigned, 2> dims{ocDim, gDim};
  const auto groupings = detectGroupings<3>(graph, weights, dims);

  return {groupings[2], groupings[1], groupings[0]};
}

// Groups tensor from standard convolution weight tensor shape [G]...[OC][IC]
// to internal shape [G1][OC1][IC1]...[G2][OC2][IC2]
//
// Where
//  G1 * G2 = G
//  OC1 * OC2 = OC
//  IC1 * IC2 = IC
Tensor splitWeightsIntoGroups(Tensor weights, const unsigned convGroupsPerGroup,
                              const unsigned inChansPerGroup,
                              const unsigned outChansPerGroup) {
  {
    const auto rank = weights.rank();
    const auto gDim = 0;
    const auto ocDim = rank - 2;
    const auto icDim = rank - 1;

    assert(weights.dim(gDim) % convGroupsPerGroup == 0);
    assert(weights.dim(icDim) % inChansPerGroup == 0);
    assert(weights.dim(ocDim) % outChansPerGroup == 0);

    // reshape the tensor from [G]...[OC][IC] to [G1][G2]...[OC1][OC2][IC1][IC2]
    // (do inner-most dimension first so as to not invalidate the indices)
    auto shape = weights.shape();
    shape[icDim] /= inChansPerGroup;
    shape.insert(std::begin(shape) + icDim + 1, inChansPerGroup);
    shape[ocDim] /= outChansPerGroup;
    shape.insert(std::begin(shape) + ocDim + 1, outChansPerGroup);
    shape[gDim] /= convGroupsPerGroup;
    shape.insert(std::begin(shape) + gDim + 1, convGroupsPerGroup);

    weights = weights.reshape(shape);
  }

  // shuffle the dims so we have [G1][OC1][IC2] grouped together at the start
  // and [G2][OC2][IC2] at the end.
  const auto rank = weights.rank();
  const auto g1Dim = 0;
  const auto g2Dim = 1;
  const auto oc1Dim = rank - 4;
  const auto oc2Dim = rank - 3;
  const auto ic1Dim = rank - 2;

  // G1 and IC2 are already in the correct place so don't move them. move OC1
  // IC1 to the right of G1 and move g2Dim to the left OC2.
  return weights.dimShufflePartial({oc1Dim, ic1Dim, g2Dim},
                                   {g1Dim + 1, g1Dim + 2, oc2Dim});
}

Tensor splitWeightsFromGroups(const Graph &graph, const Tensor &weights) {
  const auto detectedGrouping = detectWeightsChannelGrouping(graph, weights);
  return splitWeightsIntoGroups(weights, detectedGrouping.convGroupsPerGroup,
                                detectedGrouping.inChansPerGroup,
                                detectedGrouping.outChansPerGroup);
}

// Ungroups tensors from internal shape [G1][OC1][IC1]...[G2][OC2][IC2] to
// standard convolution weight tensor shape [G]...[OC][IC]
//
// Where
//  G1 * G2 = G
//  OC1 * OC2 = OC
//  IC1 * IC2 = IC
Tensor unsplitWeightsFromGroups(Tensor weights) {
  // this is the inverse of splitWeightsIntoGroups.
  {
    // put G2 to the right of G1, put OC1 to the left of OC2 and IC1 to the
    // left of IC2.
    const auto rank = weights.rank();
    const auto g1Dim = 0;
    const auto oc1Dim = 1;
    const auto ic1Dim = 2;
    const auto g2Dim = rank - 3;
    const auto oc2Dim = rank - 2;
    const auto ic2Dim = rank - 1;

    weights = weights.dimShufflePartial({g2Dim, oc1Dim, ic1Dim},
                                        {g1Dim + 1, oc2Dim - 2, ic2Dim - 1});
  }

  // reshape the [G1][G2][N]...[OC1][OC2][IC1][IC2] into [G][N]...[OC][IC]
  const auto rank = weights.rank();
  const auto g1Dim = 0;
  const auto g2Dim = 1;
  const auto oc1Dim = rank - 4;
  const auto oc2Dim = rank - 3;
  const auto ic1Dim = rank - 2;
  const auto ic2Dim = rank - 1;

  const auto &shape = weights.shape();
  std::vector<std::size_t> newShape;
  newShape.reserve(shape.size() - 3);

  newShape.push_back(shape[g1Dim] * shape[g2Dim]);
  newShape.insert(std::end(newShape), std::begin(shape) + 2,
                  std::end(shape) - 4);
  newShape.push_back(shape[oc1Dim] * shape[oc2Dim]);
  newShape.push_back(shape[ic1Dim] * shape[ic2Dim]);

  weights = weights.reshape(newShape);
  return weights;
}

std::vector<unsigned> dimsFromSpatialDims(std::vector<unsigned> dims,
                                          bool isActs) {
  for (auto &d : dims)
    d += 1 + isActs;
  return dims;
}

std::vector<unsigned>
actDimsFromSpatialDims(const std::vector<unsigned> &spatialDims) {
  return dimsFromSpatialDims(spatialDims, true);
}

std::vector<unsigned>
weightDimsFromSpatialDims(const std::vector<unsigned> &spatialDims) {
  return dimsFromSpatialDims(spatialDims, false);
}

// This stride is what's used to move down one element in the input field by
// the vertex.
int getInRowStride(const ConvParams &params, unsigned fieldElems,
                   bool useConvPartial1x1OutVertex,
                   unsigned convUnitWeightHeight) {
  int inRowStride =
      params.kernelTransform.dilation.front() * static_cast<int>(fieldElems);
  if (params.inputTransform.flip.front() !=
      params.kernelTransform.flip.front()) {
    inRowStride = -inRowStride;
  }
  if (convUnitWeightHeight == 1 || useConvPartial1x1OutVertex)
    inRowStride = 1;
  return inRowStride;
}

// Split field dimensions such that the stride fits machine stride. This
// implementation only splits field such that input stride fits. The outermost
// dimension is not split
Partition splitConvIntoAmpVertices(const ConvParams &params,
                                   unsigned numMachineStrideBits, int inStride,
                                   int inRowStride) {
  const auto numFieldDims = params.inputFieldShape.size();
  std::vector<unsigned> fieldDimSplit(numFieldDims, 1U);
  int stride =
      std::abs(inStride) > std::abs(inRowStride) ? inStride : inRowStride;
  // Takes the max of the stride (i.e. positive) because of twos complement
  // strides used in the machine
  if (std::abs(inStride) == std::abs(inRowStride)) {
    stride = std::max(inStride, inRowStride);
  }

  // Exclude outermost dimension and select field with maximum input elements
  const auto fieldDimWithMaxSizeIt = std::max_element(
      std::next(params.inputFieldShape.begin()), params.inputFieldShape.end());
  if (fieldDimWithMaxSizeIt != params.inputFieldShape.end()) {
    const int machineStride = stride >= 0 ? (1 << numMachineStrideBits) / 2 - 1
                                          : (1 << numMachineStrideBits) / 2;
    auto splitFactor = (std::abs(stride) + machineStride - 1) / machineStride;
    fieldDimSplit[std::distance(params.inputFieldShape.begin(),
                                fieldDimWithMaxSizeIt)] = splitFactor;
  }

  const unsigned batchSplit = 1;
  const Split<unsigned> outChanSplit = {1, 1};
  std::vector<unsigned> kernelSplit(numFieldDims, 1U);
  const Split<unsigned> inChanSplit = {1, 1};
  const unsigned convGroupSplit = 1;
  std::vector<unsigned> fieldAxisGrainSize(numFieldDims, 1U);
  const unsigned convGroupGrainSize = 1;
  const unsigned inChanGrainSize = 1;
  const unsigned outChanGrainSize = 1;
  return {fieldDimSplit,
          batchSplit,
          outChanSplit,
          std::move(kernelSplit),
          inChanSplit,
          convGroupSplit,
          std::move(fieldAxisGrainSize),
          convGroupGrainSize,
          inChanGrainSize,
          outChanGrainSize};
}

std::vector<multiconv::internal::CreateTensorArgs>
convertToConvOptions(poplar::Graph &graph,
                     const std::vector<multiconv::CreateTensorArgs> &args) {
  std::vector<multiconv::internal::CreateTensorArgs> v;
  for (const auto &arg : args) {
    const ConvOptions options(graph.getTarget(), arg.options);
    v.push_back({arg.params, options, arg.name});
  }
  return v;
}

std::vector<multiconv::internal::ConvolutionArgs>
convertToConvOptions(poplar::Graph &graph,
                     const std::vector<multiconv::ConvolutionArgs> &args) {
  std::vector<multiconv::internal::ConvolutionArgs> v;
  for (const auto &arg : args) {
    const ConvOptions options(graph.getTarget(), arg.options);
    v.push_back({arg.inputs, arg.weights, arg.params, options});
  }
  return v;
}

std::vector<multiconv::internal::CalculateWeightDeltasArgs>
convertToConvOptions(
    poplar::Graph &graph,
    const std::vector<multiconv::CalculateWeightDeltasArgs> &args) {
  std::vector<multiconv::internal::CalculateWeightDeltasArgs> v;
  for (const auto &arg : args) {
    const ConvOptions options(graph.getTarget(), arg.options);
    v.push_back({arg.zDeltas, arg.activations, arg.params, options});
  }
  return v;
}

namespace {

// common method for multiconv::internal::ConvWeightUpdateArgs(WithScalar).
template <typename ScaleType, typename T>
std::vector<multiconv::internal::ConvWeightUpdateArgs<ScaleType>>
convertToConvOptionsImpl(poplar::Graph &graph, const std::vector<T> &args) {
  std::vector<multiconv::internal::ConvWeightUpdateArgs<ScaleType>> v;
  for (const auto &arg : args) {
    const ConvOptions options(graph.getTarget(), arg.options);
    v.push_back({arg.zDeltas, arg.weights, arg.activations, arg.scale,
                 arg.params, options});
  }
  return v;
}

} // unnamed namespace

std::vector<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>
convertToConvOptions(poplar::Graph &graph,
                     const std::vector<multiconv::ConvWeightUpdateArgs> &args) {
  return convertToConvOptionsImpl<poplar::Tensor>(graph, args);
}

std::vector<multiconv::internal::ConvWeightUpdateArgs<float>>
convertToConvOptions(
    poplar::Graph &graph,
    const std::vector<multiconv::ConvWeightUpdateArgsScalar> &args) {
  return convertToConvOptionsImpl<float>(graph, args);
}

static constexpr auto allButNumConvGroups = poplibs_support::makeStructHelper(
    &ConvParams::inputType, &ConvParams::outputType, &ConvParams::batchSize,
    &ConvParams::inputFieldShape, &ConvParams::kernelShape,
    &ConvParams::inputChannelsPerConvGroup,
    &ConvParams::outputChannelsPerConvGroup, &ConvParams::inputTransform,
    &ConvParams::kernelTransform, &ConvParams::outputTransform);

template <typename T> bool canBeCombined(const T &ca1, const T &ca2) {
  return allButNumConvGroups.eq(*ca1.params, *ca2.params) &&
         ca1.options == ca2.options;
}

template <typename T>
bool canBeCombined(const std::vector<T> &convolutionArgs) {
  auto it = convolutionArgs.begin();
  const auto first = it;
  while (++it != convolutionArgs.end()) {
    if (!canBeCombined(*first, *it)) {
      return false;
    }
  }
  return true;
}

template bool canBeCombined<multiconv::internal::CreateTensorArgs>(
    const std::vector<multiconv::internal::CreateTensorArgs> &args);
template bool canBeCombined<multiconv::internal::ConvolutionArgs>(
    const std::vector<multiconv::internal::ConvolutionArgs> &args);
template bool canBeCombined<multiconv::internal::CalculateWeightDeltasArgs>(
    const std::vector<multiconv::internal::CalculateWeightDeltasArgs> &args);
template bool canBeCombined<multiconv::internal::ConvWeightUpdateArgs<float>>(
    const std::vector<multiconv::internal::ConvWeightUpdateArgs<float>> &args);
template bool
canBeCombined<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>(
    const std::vector<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>
        &args);

template <typename T>
std::vector<std::vector<const T *>>
groupCombinables(const std::vector<T> &args) {
  std::vector<std::vector<const T *>> grouped;
  for (auto &ca : args) {
    bool found(false);
    for (auto &g : grouped) {
      if (canBeCombined(ca, *g[0])) {
        g.push_back(&ca);
        found = true;
      }
    }
    if (!found) {
      grouped.push_back({&ca});
    }
  }
  return grouped;
}

template std::vector<std::vector<const multiconv::internal::CreateTensorArgs *>>
groupCombinables<multiconv::internal::CreateTensorArgs>(
    const std::vector<multiconv::internal::CreateTensorArgs> &args);
template std::vector<std::vector<const multiconv::internal::ConvolutionArgs *>>
groupCombinables<multiconv::internal::ConvolutionArgs>(
    const std::vector<multiconv::internal::ConvolutionArgs> &args);
template std::vector<
    std::vector<const multiconv::internal::CalculateWeightDeltasArgs *>>
groupCombinables<multiconv::internal::CalculateWeightDeltasArgs>(
    const std::vector<multiconv::internal::CalculateWeightDeltasArgs> &args);
template std::vector<
    std::vector<const multiconv::internal::ConvWeightUpdateArgs<float> *>>
groupCombinables<multiconv::internal::ConvWeightUpdateArgs<float>>(
    const std::vector<multiconv::internal::ConvWeightUpdateArgs<float>> &args);
template std::vector<std::vector<
    const multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor> *>>
groupCombinables<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>(
    const std::vector<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>
        &args);

// Returns the combination (aggregates numConvGroups) of
// multiple compatible convolution parameters
CanonicalConvParams
combineConvParams(const std::vector<CanonicalConvParams> &convParams) {
  assert(!convParams.empty());
  std::size_t numConvGroups(0);
  for (const auto &cp : convParams) {
    numConvGroups += cp->numConvGroups;
  }
  ConvParams cp(*convParams[0]);
  cp.numConvGroups = numConvGroups;
  return cp;
}

multiconv::internal::CreateTensorArgs
combine(const std::vector<multiconv::internal::CreateTensorArgs> &args) {
  assert(!args.empty());
  assert(canBeCombined(args));

  std::vector<CanonicalConvParams> convParams;
  for (const auto &arg : args) {
    convParams.push_back(arg.params);
  }

  // TODO: what to do with the nane?
  return multiconv::internal::CreateTensorArgs{combineConvParams(convParams),
                                               args[0].options, args[0].name};
}

namespace {

// tensors are concatinated in the group dimension which is 1 for acts and
// 0 for weights.
constexpr unsigned actsGroupDim = 1;
constexpr unsigned weightsGroupDim = 0;

} // unnamed namespace

multiconv::internal::ConvolutionArgs combine(
    const std::vector<multiconv::internal::ConvolutionArgs> &convolutionArgs) {
  assert(!convolutionArgs.empty());
  assert(canBeCombined(convolutionArgs));
  std::vector<CanonicalConvParams> convParams;
  std::vector<poplar::Tensor> inputs;
  std::vector<poplar::Tensor> weights;
  for (const auto &cp : convolutionArgs) {
    convParams.push_back(cp.params);
    inputs.push_back(cp.inputs);
    weights.push_back(cp.weights);
  }

  return multiconv::internal::ConvolutionArgs{
      concat(inputs, actsGroupDim), concat(weights, weightsGroupDim),
      combineConvParams(convParams), convolutionArgs[0].options};
}

multiconv::internal::CalculateWeightDeltasArgs combine(
    const std::vector<multiconv::internal::CalculateWeightDeltasArgs> &args) {
  assert(!args.empty());
  assert(canBeCombined(args));

  std::vector<CanonicalConvParams> convParams;
  std::vector<poplar::Tensor> zDeltas;
  std::vector<poplar::Tensor> activations;
  for (const auto &arg : args) {
    convParams.push_back(arg.params);
    zDeltas.push_back(arg.zDeltas);
    activations.push_back(arg.activations);
  }

  return multiconv::internal::CalculateWeightDeltasArgs{
      concat(zDeltas, actsGroupDim), concat(activations, actsGroupDim),
      combineConvParams(convParams), args[0].options};
}

template <typename T>
multiconv::internal::ConvWeightUpdateArgs<T>
combine(const std::vector<multiconv::internal::ConvWeightUpdateArgs<T>> &args) {
  assert(!args.empty());
  assert(canBeCombined(args));

  std::vector<CanonicalConvParams> convParams;
  std::vector<poplar::Tensor> zDeltas;
  std::vector<poplar::Tensor> weights;
  std::vector<poplar::Tensor> activations;
  for (const auto &arg : args) {
    convParams.push_back(arg.params);
    zDeltas.push_back(arg.zDeltas);
    weights.push_back(arg.weights);
    activations.push_back(arg.activations);
  }

  // TODO: scale needs thought...
  return multiconv::internal::ConvWeightUpdateArgs<T>{
      concat(zDeltas, actsGroupDim),     concat(weights, weightsGroupDim),
      concat(activations, actsGroupDim), args[0].scale,
      combineConvParams(convParams),     args[0].options};
}

template multiconv::internal::ConvWeightUpdateArgs<float> combine<float>(
    const std::vector<multiconv::internal::ConvWeightUpdateArgs<float>> &args);
template multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>
combine<poplar::Tensor>(
    const std::vector<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>
        &args);

std::vector<poplar::Tensor>
splitOutput(const std::vector<CanonicalConvParams> &convParams,
            const poplar::Tensor &out) {
  assert(!convParams.empty());
  std::vector<Interval> intervals;
  std::size_t prev(0);
  const std::size_t outputChannelsPerConvGroup =
      convParams[0]->outputChannelsPerConvGroup;
  for (const auto &cp : convParams) {
    const auto intervalSize = cp->numConvGroups * outputChannelsPerConvGroup;
    intervals.push_back({prev, prev + intervalSize});
    prev += intervalSize;
  }
  return out.slices(intervals, actsGroupDim);
}

std::vector<poplar::Tensor>
splitInput(const std::vector<CanonicalConvParams> &convParams,
           const poplar::Tensor &in) {
  assert(!convParams.empty());
  std::vector<Interval> intervals;
  std::size_t prev(0);
  const std::size_t inChans = convParams[0]->inputChannelsPerConvGroup;
  for (const auto &cp : convParams) {
    const auto intervalSize = cp->numConvGroups * inChans;
    intervals.push_back({prev, prev + intervalSize});
    prev += intervalSize;
  }
  return in.slices(intervals, actsGroupDim);
}

std::vector<poplar::Tensor>
splitWeights(const std::vector<CanonicalConvParams> &convParams,
             const poplar::Tensor &weights) {
  assert(!convParams.empty());
  std::vector<Interval> intervals;
  std::size_t prev(0);
  for (const auto &cp : convParams) {
    const auto intervalSize = cp->numConvGroups;
    intervals.push_back({prev, prev + intervalSize});
    prev += intervalSize;
  }
  return weights.slices(intervals, weightsGroupDim);
}

std::vector<unsigned>
splitElementsInWeightedGroups(const std::vector<std::uint64_t> &groups,
                              unsigned elements) {
  if (elements < groups.size()) {
    throw poputil::poplibs_error("At least one element per group");
  }
  const double totalWeight = std::accumulate(groups.begin(), groups.end(), 0LL);
  if (!totalWeight) {
    throw poputil::poplibs_error("Total weight cannot be zero");
  }
  std::vector<unsigned> elementsPerGroup;
  double assignedElements(0);
  for (const auto &weight : groups) {
    const double e = elements * weight / totalWeight;
    elementsPerGroup.push_back(std::round(assignedElements + e) -
                               std::round(assignedElements));
    assignedElements += e;
  }
  // Each group must have at least one element
  for (auto &epg : elementsPerGroup) {
    if (!epg) {
      --*std::max_element(elementsPerGroup.begin(), elementsPerGroup.end());
      ++epg;
    }
  }
  assert(elements ==
         std::accumulate(elementsPerGroup.begin(), elementsPerGroup.end(), 0U));
  return elementsPerGroup;
}

std::vector<unsigned> splitTilesByComp(const std::vector<std::uint64_t> &flops,
                                       unsigned numTiles) {
  // The amount of tiles in each subsets should be a multiple of 2 as tile pairs
  // are used for 64-bit sends
  const unsigned atomSize(2);
  if (numTiles % atomSize > 0) {
    throw poputil::poplibs_error(
        "Number of tiles should be a multiple of atom size");
  }
  auto tiles = splitElementsInWeightedGroups(flops, numTiles / atomSize);
  std::transform(tiles.begin(), tiles.end(), tiles.begin(),
                 [](auto t) { return t * atomSize; });
  return tiles;
}

std::vector<unsigned>
splitTilesByComp(const std::vector<ConvParams> &convParams, unsigned numTiles) {
  std::vector<std::uint64_t> flops;
  for (const auto &cp : convParams) {
    flops.push_back(getFwdFlops(cp));
  }
  return splitTilesByComp(flops, numTiles);
}

unsigned getGroupIndex(const std::vector<unsigned> &groups,
                       const unsigned element) {
  unsigned acc(0);
  for (unsigned i(0); i < groups.size(); ++i) {
    acc += groups[i];
    if (acc > element) {
      return i;
    }
  }
  throw poputil::poplibs_error("Element index exceeds groups size");
}

void log(unsigned indent, const ConvParams &params) {
  namespace logging = poplibs_support::logging;

  if (logging::shouldLog(logging::Level::Info)) {
    std::string prefix(indent, ' ');
    logging::info(
        "{}input={}x({}x{}x{}) padding={}/{} truncation={}/{} dilation={} "
        "flip={}",
        prefix, params.inputFieldShape, params.getBatchSize(),
        params.getNumConvGroups(), params.getNumInputChansPerConvGroup(),
        params.inputTransform.paddingLower, params.inputTransform.paddingUpper,
        params.inputTransform.truncationLower,
        params.inputTransform.truncationUpper, params.inputTransform.dilation,
        params.inputTransform.flip);
    logging::info("{}kernel={}x({}x{}x{}) padding={}/{} truncation={}/{} "
                  "dilation={} flip={}",
                  prefix, params.kernelShape, params.getNumConvGroups(),
                  params.getNumOutputChansPerConvGroup(),
                  params.getNumInputChansPerConvGroup(),
                  params.kernelTransform.paddingLower,
                  params.kernelTransform.paddingUpper,
                  params.kernelTransform.truncationLower,
                  params.kernelTransform.truncationUpper,
                  params.kernelTransform.dilation, params.kernelTransform.flip);
    logging::info(
        "{}output={}x({}x{}x{}) padding={}/{} truncation={}/{} stride={}",
        prefix, params.getOutputFieldShape(), params.getBatchSize(),
        params.getNumConvGroups(), params.getNumOutputChansPerConvGroup(),
        params.outputTransform.paddingLower,
        params.outputTransform.paddingUpper,
        params.outputTransform.truncationLower,
        params.outputTransform.truncationUpper, params.outputTransform.stride);
  }
}

} // namespace poplin
