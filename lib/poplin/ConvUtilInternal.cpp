// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#include "ConvUtilInternal.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/icl/interval_map.hpp>
#include <boost/optional.hpp>
#include <cassert>
#include <poplar/Tensor.hpp>

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

static Tensor groupTensorAux(const Tensor &t, unsigned rank) { return t; }
static Tensor ungroupTensorAux(const Tensor &t, unsigned) { return t; }

template <typename... G>
static Tensor groupTensorAux(const Tensor &t, unsigned rank,
                             const GroupingInfo &g, G &&... gs) {
  return groupTensorAux(t.reshapePartial(g.first, g.first + 1,
                                         {t.dim(g.first) / g.second, g.second})
                            .dimRoll(g.first + 1, rank),
                        rank + 1, std::forward<G>(gs)...);
}

template <typename... G>
static Tensor ungroupTensorAux(const Tensor &t, unsigned rank,
                               const GroupingInfo &g, G &&... gs) {
  return ungroupTensorAux(
      t.dimRoll(rank, g.first + 1).flatten(g.first, g.first + 2), rank,
      std::forward<G>(gs)...);
}

template <typename... G>
static Tensor groupTensor(const Tensor &t, G &&... gs) {
  return groupTensorAux(t, t.rank(), std::forward<G>(gs)...);
}

template <typename... G>
static Tensor ungroupTensor(const Tensor &t, G &&... gs) {
  return ungroupTensorAux(t, unsigned(t.rank() - sizeof...(gs)),
                          std::forward<G>(gs)...);
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
};

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
  const unsigned inChanSplit = 1;
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

unsigned getMinimumRegroupGrainSize(const Type &type) {
  if (type == HALF) {
    return 4;
  } else if (type == FLOAT) {
    return 2;
  }
  return 1;
}

// Returns an updated grouping based on original grouping and tile mapping

static std::tuple<GroupingInfo, GroupingInfo, Graph::TileToTensorMapping>
updateGroupingInternal(const Graph &graph, const Tensor &t,
                       const GroupingInfo &from, const GroupingInfo &to) {
  auto grouped = groupTensor(t, to, from);
  auto groupedFlat = grouped.flatten(0, grouped.rank() - 2).flatten(1, 3);
  const auto tMapping = graph.getTileMapping(groupedFlat);
  const auto numTiles = tMapping.size();
  const auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
  const auto numIPUs = numTiles / tilesPerIPU;
  std::vector<std::size_t> elemsPerIpu(numIPUs);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto ipu = tile / tilesPerIPU;
    const auto &mapping = tMapping[tile];
    if (!mapping.empty()) {
      for (const auto &r : mapping) {
        elemsPerIpu[ipu] += r.size();
      }
    }
  }

  // Minimum number of elements in a group. Groups are split by a multiple of
  // this
  const unsigned minGroupsSize = 4;

  // find entry with max elements
  auto maxIt = std::max_element(std::begin(elemsPerIpu), std::end(elemsPerIpu));

  // A factor by which the number of transposes can increase by without
  // breaking the constraints set on group size
  auto additionalFactor = groupedFlat.dim(1) / (minGroupsSize * minGroupsSize);

  auto isPrime = [](unsigned num) {
    for (unsigned i = 2; i <= num / 2; ++i) {
      if (num % i == 0) {
        return false;
      }
    }
    return true;
  };

  // This limits the number of transpositions allowed on the IPU
  const auto maxTranspositionsAllowedPerIpu = numTiles;

  // actual transpose factor used. Initialise with 1 which means no additional
  // factor is applied
  unsigned transposeFactor = 1;

  // Estimate the number of transpositions on the IPU which has the maximum
  // number of elements mapped
  auto transpositionsOnIpuEstimate =
      (*maxIt + groupedFlat.dim(1) - 1) / groupedFlat.dim(1);

  bool allowIncrease =
      to.second % minGroupsSize == 0 && from.second % minGroupsSize == 0;
  while (allowIncrease && additionalFactor != 1) {
    unsigned factor = 1;
    // TODO: T12892 This assumes that typical transposes are a multiple of very
    // small primes. Investigate other methods (e.g., dividing into prime
    // factors). A method that should give good results is to find the maximum
    // GCD across different values of transpositions (i.e.
    // maxTranspositionsAllowedPerIpu, maxTranspositionsAllowedPerIpu-1, ...)
    for (unsigned x = 2; x <= additionalFactor; ++x) {
      if (additionalFactor % x == 0 && isPrime(x)) {
        factor = x;
        break;
      }
    }
    if (transpositionsOnIpuEstimate * transposeFactor * factor >
        maxTranspositionsAllowedPerIpu) {
      break;
    }
    if (additionalFactor % factor != 0 || factor == 1) {
      throw poputil::poplibs_error("Invalid factor in regrouping");
    }
    transposeFactor *= factor;
    additionalFactor /= factor;
  }

  auto updatedFrom = from;
  auto updatedTo = to;

  if (transposeFactor != 1) {
    // TODO: T12893 Optimise split once the cost of using a supervisor vertex
    // is known.
    auto factorFrom = gcd(transposeFactor, from.second / minGroupsSize);
    transposeFactor /= factorFrom;
    auto factorTo = gcd(transposeFactor, to.second / minGroupsSize);
    updatedFrom.second /= factorFrom;
    updatedTo.second /= factorTo;
  }
  return std::make_tuple(updatedFrom, updatedTo, std::move(tMapping));
}

std::pair<GroupingInfo, GroupingInfo> updateGrouping(const Graph &graph,
                                                     const Tensor &t,
                                                     const GroupingInfo &from,
                                                     const GroupingInfo &to) {
  const auto result = updateGroupingInternal(graph, t, from, to);
  return std::make_pair(std::get<0>(result), std::get<1>(result));
}

Tensor regroupTensor(Graph &graph, const Tensor &t,
                     poplar::program::Sequence &copies,
                     boost::optional<ComputeSet> &transposeCS,
                     const GroupingInfo &from_, const GroupingInfo &to_,
                     const std::string &debugPrefix) {
  GroupingInfo to, from;
  Graph::TileToTensorMapping tMapping;
  std::tie(from, to, tMapping) = updateGroupingInternal(graph, t, from_, to_);
  auto grouped = groupTensor(t, to, from);
  auto groupedFlat = grouped.flatten(0, grouped.rank() - 2).flatten(1, 3);

  if (!(from == from_ && to == to_)) {
    tMapping = graph.getTileMapping(groupedFlat);
  }

  // Explicitly copy to a single variable in order to force
  // regions to be contiguous. Performing a transpose alone
  // may leave multiple regions per-tile, one for each edge to a
  // transpose vertex.
  auto preRegroup = graph.addVariable(t.elementType(), grouped.shape(),
                                      debugPrefix + "/preRegroup");
  auto preRegroupTranspose = preRegroup.flatten(0, preRegroup.rank() - 2);
  auto preRegroupFlat =
      preRegroup.flatten(0, preRegroup.rank() - 2).flatten(1, 3);

  // Build a map giving which intervals are mapped to each
  // IPU. Track which tiles on each IPU have any elements
  // mapped.
  const auto numTiles = tMapping.size();
  const auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
  const auto numIPUs = numTiles / tilesPerIPU;

  std::vector<std::vector<unsigned>> mappedTilesByIPU(numIPUs);
  for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
    mappedTilesByIPU.reserve(tilesPerIPU);
  }
  using IntervalMap = boost::icl::interval_map<std::size_t, unsigned,
                                               boost::icl::partial_enricher>;
  using Interval = boost::icl::interval<std::size_t>;
  IntervalMap intervalsToIPU;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto ipu = tile / tilesPerIPU;
    const auto &mapping = tMapping[tile];
    if (!mapping.empty()) {
      mappedTilesByIPU[ipu].push_back(tile);
      for (const auto &i : mapping) {
        intervalsToIPU.insert(
            std::make_pair(Interval::right_open(i.begin(), i.end()), ipu));
      }
    }
  }

  // Iterate each transposition, mapping this to an IPU based on the first
  // element in each.
  auto elemsPerTransposition = preRegroupFlat.dim(1);
  std::vector<std::vector<poplar::Interval>> ipuTranspositions(numIPUs);
  for (unsigned t = 0; t < preRegroupFlat.dim(0); ++t) {
    auto it = intervalsToIPU.find(Interval::right_open(
        t * elemsPerTransposition, t * elemsPerTransposition + 1));
    assert(it != intervalsToIPU.end());
    auto ipu = it->second;
    auto &ipuTs = ipuTranspositions[ipu];
    // Try and extend the previous region if possible
    if (!ipuTs.empty() && ipuTs.back().end() == t) {
      ipuTs.back() = poplar::Interval(ipuTs.back().begin(), t + 1);
    } else {
      ipuTs.emplace_back(t, t + 1);
    }
  }

  // Finally map slices of the new tensor to transpose mapped linearly
  // across the tiles on which the original tensor was mapped on the same
  // IPU the elements of the transposition were originally mapped to.
  //
  // FIXME: This currently allows external exchange to be incurred for a
  // given transposition. This should not be allowed as it is not expected
  // but for the timebeing the padding constants added to activations
  // are just mapped to tile 0 which can be a different IPU to the one
  // on which it should actually reside. T6427 is required for this as a
  // primary usage of these regrouping functions.
  for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
    const auto &mappedTiles = mappedTilesByIPU[ipu];
    const auto &transpositions = ipuTranspositions[ipu];
    auto numTiles = mappedTiles.size();
    auto numTranspositions = std::accumulate(
        transpositions.begin(), transpositions.end(), std::size_t(0),
        [](std::size_t t, const poplar::Interval &i) { return t + i.size(); });
    if (!numTranspositions)
      continue;

    // Map transpositions on this IPU evenly across the tiles on which
    // elements of the source tensor reside.
    auto transpositionsPerTile = (numTranspositions + numTiles - 1) / numTiles;
    auto interval = transpositions.begin();
    unsigned intervalOffset = 0;
    for (unsigned i = 0; i < numTiles; ++i) {
      auto remaining = std::min(transpositionsPerTile, numTranspositions);
      numTranspositions -= remaining;
      while (remaining > 0) {
        auto n = std::min(interval->size() - intervalOffset, remaining);
        auto slice =
            preRegroupFlat.slice(interval->begin() + intervalOffset,
                                 interval->begin() + intervalOffset + n, 0);
        graph.setTileMapping(slice, mappedTiles[i]);
        remaining -= n;
        intervalOffset += n;
        if (interval->begin() + intervalOffset == interval->end()) {
          ++interval;
          intervalOffset = 0;
        }
      }
    }
  }

  copies.add(program::Copy(grouped, preRegroup));

  // Finally, transpose
  if (!transposeCS) {
    transposeCS = graph.addComputeSet(debugPrefix + "/Transpose");
  }
  auto partiallyTransposed =
      partialTranspose(graph, preRegroup, *transposeCS, debugPrefix);

  return ungroupTensor(partiallyTransposed, from, to);
}

} // namespace poplin
