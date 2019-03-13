#include "ConvUtilInternal.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplibs_support/gcd.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include <boost/icl/interval_map.hpp>
#include <boost/optional.hpp>
#include <poplar/Tensor.hpp>
#include <cassert>

using namespace poplar;

namespace poplin {

static Tensor groupTensorAux(const Tensor &t, unsigned rank) {
  return t;
}
static Tensor ungroupTensorAux(const Tensor &t, unsigned) {
  return t;
}

template <typename ...G>
static Tensor groupTensorAux(const Tensor &t, unsigned rank,
                             const GroupingInfo &g, G&&... gs) {
  return groupTensorAux(
      t.reshapePartial(g.first, g.first + 1,
                       {t.dim(g.first) / g.second, g.second})
       .dimRoll(g.first + 1, rank),
      rank + 1,
      std::forward<G>(gs)...);
}

template <typename ...G>
static Tensor ungroupTensorAux(const Tensor &t, unsigned rank,
                               const GroupingInfo &g, G&&... gs) {
  return ungroupTensorAux(
      t.dimRoll(rank, g.first + 1)
       .flatten(g.first, g.first + 2),
      rank,
      std::forward<G>(gs)...);
}

template <typename ...G>
static Tensor groupTensor(const Tensor &t, G&&... gs) {
  return groupTensorAux(t, t.rank(), std::forward<G>(gs)...);
}

template <typename ...G>
static Tensor ungroupTensor(const Tensor &t, G&&... gs) {
  return ungroupTensorAux(t, unsigned(t.rank() - sizeof...(gs)),
                          std::forward<G>(gs)...);
}


std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(unsigned batchElements,
                             const std::vector<unsigned> &tileConvOutSize,
                             unsigned numContexts,
                             const std::vector<unsigned> &inputDilation,
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
      (tileConvOutSize.back() + outputStride.back() - 1) /
      outputStride.back();
  unsigned activeRows = 1;
  std::vector<unsigned> activeRowShape;
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    auto dimActiveRows = (tileConvOutSize[dim] + outputStride[dim] - 1) /
                         outputStride[dim];
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
    auto beginIndices =
        poputil::unflattenIndex<std::size_t>({batchElements, activeRows,
                                             elementsPerRow}, beginElement);
    auto lastIndices =
        poputil::unflattenIndex<std::size_t>({batchElements, activeRows,
                                             elementsPerRow}, lastElement);
    for (unsigned b = beginIndices[0]; b != lastIndices[0] + 1; ++b) {
      unsigned activeRowBegin = b == beginIndices[0] ?
                                beginIndices[1] :
                                0;
      unsigned activeRowLast = b == lastIndices[0] ?
                               lastIndices[1] :
                               activeRows - 1;
      for (unsigned activeRow = activeRowBegin; activeRow != activeRowLast + 1;
           ++activeRow) {
        unsigned activeXBegin =
            b == beginIndices[0] && activeRow == beginIndices[1] ?
              beginIndices[2] : 0;
        unsigned activeXLast =
            b == lastIndices[0] && activeRow == lastIndices[1] ?
              lastIndices[2] : elementsPerRow - 1;
        auto outerFieldIndices = poputil::unflattenIndex(activeRowShape,
                                                        activeRow);
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
Tensor
actsToInternalShape(const Tensor &act, unsigned numConvGroups,
                    unsigned chansPerGroup) {
  return act.reshapePartial(1, 2, {numConvGroups, chansPerGroup})
            .dimShufflePartial({1, 2}, {0, act.rank()});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [N][G * C]... shape where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
Tensor
actsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({0, act.rank() - 1}, {1, 2}).flatten(1, 3);
}

// Reshape the weights tensor from [G][OC][IC]... shape to
// [G]...[OC][IC].
Tensor
weightsToInternalShape(const Tensor &act) {
  return act.dimShufflePartial({1, 2}, {act.rank() - 2, act.rank() - 1});
}

// Reshape the weights tensor from [G]...[OC][IC] shape to
// [G][OC][IC]... shape.
Tensor
weightsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({act.rank() - 2, act.rank() - 1}, {1, 2});
}

// Reshape the activations tensor from [G][N]...[C] shape to
// [G][C1][N]...[C2]
//
// Where C1 * C2 = C
Tensor
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
Tensor
splitActivationChanGroups(const Graph &graph, const Tensor &act) {
  auto chansPerGroup = detectChannelGrouping(graph, act);
  return splitActivationChanGroups(act, chansPerGroup);
}

// Reshape the activations tensor from [G][C1][N]...[C2] shape to
// [G][N]...[C]
//
// Where C1 * C2 = C
Tensor
unsplitActivationChanGroups(const Tensor &act) {
  const auto rank = act.rank();
  return act.dimShufflePartial({1}, {rank - 2})
            .reshapePartial(rank - 2, rank, {act.dim(1) * act.dim(rank - 1)});
}

std::pair<unsigned, unsigned>
detectWeightsChannelGrouping(const Graph &graph, const Tensor &w) {
  auto inChansPerGroup = detectChannelGrouping(graph, w);
  const auto rank = w.rank();
  const auto w1 =
      w.reshapePartial(rank - 1, rank, {w.dim(rank - 1) / inChansPerGroup,
                                        inChansPerGroup})
       .dimRoll(rank - 1, 0).flatten(rank - 1, rank + 1);
  auto outChansPerGroup = detectChannelGrouping(graph, w1);

  // The innermost dimension of the tensor should detect the product of the
  // input and output channels per group. The result obtained is incorrect
  // if partial elements of the product are assigned to a tile. If a full
  // product is not mapped, the outChansPerGroup is conservatively set to be
  // 1.
  if (outChansPerGroup % inChansPerGroup == 0)
    outChansPerGroup /= inChansPerGroup;
  else
    outChansPerGroup = 1;
  return {outChansPerGroup, inChansPerGroup};
}

// Groups tensor from standard convolution weight tensor shape [G]...[OC][IC]
// to internal shape [G][OC1][IC1]...[OC2][IC2]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
Tensor groupWeights(const Tensor &weights, unsigned inChansPerGroup,
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


Tensor groupWeights(const Graph &graph, const Tensor &weights) {
  unsigned inChansPerGroup, outChansPerGroup;
  std::tie(outChansPerGroup, inChansPerGroup) =
      detectWeightsChannelGrouping(graph, weights);
  return groupWeights(weights, inChansPerGroup, outChansPerGroup);
}

// Ungroups tensors from internal shape [G][OC1][IC1]...[OC2][IC2] to
// standard convolution weight tensor shape [G]...[OC][IC]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
Tensor ungroupWeights(const Tensor &weights) {
  const auto rank = weights.rank();
  return weights.dimShufflePartial({1, 2}, {rank - 4, rank - 2})
                .reshapePartial(rank - 4, rank,
                                {weights.dim(1) * weights.dim(rank - 2),
                                 weights.dim(2) * weights.dim(rank - 1)});
}

std::vector<unsigned>
dimsFromSpatialDims(std::vector<unsigned> dims, bool isActs) {
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
Partition
splitConvIntoAmpVertices(const ConvParams &params,
                         unsigned numMachineStrideBits,
                         int inStride, int inRowStride) {
  const auto numFieldDims = params.inputFieldShape.size();
  std::vector<unsigned> fieldDimSplit(numFieldDims, 1U);
  int stride = std::abs(inStride) > std::abs(inRowStride) ? inStride :
                                                            inRowStride;
  // Takes the max of the stride (i.e. positive) because of twos complement
  // strides used in the machine
  if (std::abs(inStride) == std::abs(inRowStride)) {
    stride = std::max(inStride, inRowStride);
  }

  // Exclude outermost dimension and select field with maximum input elements
  const auto fieldDimWithMaxSizeIt =
      std::max_element(std::next(params.inputFieldShape.begin()),
                       params.inputFieldShape.end());
  if (fieldDimWithMaxSizeIt != params.inputFieldShape.end()) {
    const int machineStride = stride >= 0 ?
          (1 << numMachineStrideBits) / 2 - 1 :
          (1 << numMachineStrideBits) / 2;
    auto splitFactor = (std::abs(stride) + machineStride - 1) / machineStride;
    fieldDimSplit[std::distance(params.inputFieldShape.begin(),
                                fieldDimWithMaxSizeIt)] = splitFactor;
  }
  unsigned batchSplit = 1;
  unsigned outChanSplit = 1;
  std::vector<unsigned> kernelSplit(numFieldDims, 1U);
  unsigned inChanSplit = 1;
  unsigned convGroupSplit = 1;
  std::vector<unsigned> fieldAxisGrainSize(numFieldDims, 1U);
  unsigned inChanGrainSize = 1;
  unsigned outChanGrainSize = 1;
  return {
    fieldDimSplit,
    batchSplit,
    outChanSplit,
    kernelSplit,
    inChanSplit,
    convGroupSplit,
    fieldAxisGrainSize,
    inChanGrainSize,
    outChanGrainSize
  };
}

std::vector<GroupingInfo>
detectDimGroupings(const Graph &graph, const Tensor &t) {
  std::vector<GroupingInfo> info;

  auto dims = t.rank();
  auto groupedT = t;
  unsigned totalGrouping = 1;
  while (true) {
    unsigned grouping = 1;
    unsigned groupedDim = 0;

    for (std::size_t d = 0; d < dims; ++d) {
      // Skip singular dimensions
      if (groupedT.dim(d) == 1)
        continue;
      // Detect grouping of this dim along with previous groupings
      auto permutation =
        groupedT.dimRoll(d, dims - 1).flatten(dims - 1, groupedT.rank());
      auto g = detectChannelGrouping(graph, permutation);
      // Even though we may already have found some grouping, the new
      // grouping we find may not be a multiple of totalGrouping if
      // there is a grouping in a weirdly sized combination of dimensions
      // so bottom out at 1 so that the gcd below gives the desired result.
      auto thisGrouping = g % totalGrouping ? 1u : g / totalGrouping;
      thisGrouping = gcd<unsigned>(thisGrouping, groupedT.dim(d));
      if (thisGrouping > grouping) {
        groupedDim = d;
        grouping = thisGrouping;
      }
    }

    // No more groupings to be found, we're done.
    if (grouping == 1)
      break;

    info.emplace_back(groupedDim, grouping);
    totalGrouping *= grouping;
    assert((groupedT.dim(groupedDim) % grouping) == 0);
    // Roll the grouping to the back for the next round
    groupedT = groupedT.reshapePartial(groupedDim,
                                       groupedDim + 1,
                                       {groupedT.dim(groupedDim) / grouping,
                                       grouping})
                       .dimRoll(groupedDim + 1, dims);
  }

  return info;
}

unsigned
getMinimumRegroupGrainSize(const Type &type) {
  if (type == HALF) {
    return 4;
  } else if (type == FLOAT) {
    return 2;
  }
  return 1;
}

Tensor
regroupTensor(Graph &graph, const Tensor &t,
              poplar::program::Sequence &copies,
              boost::optional<ComputeSet> &transposeCS,
              const GroupingInfo &from, const GroupingInfo &to,
              const std::string &debugPrefix) {
  auto grouped = groupTensor(t, to, from);
  auto groupedFlat = grouped.flatten(0, grouped.rank() - 2)
                            .flatten(1, 3);

  // Explicitly copy to a single variable in order to force
  // regions to be contiguous. Performing a transpose alone
  // may leave multiple regions per-tile, one for each edge to a
  // transpose vertex.
  auto preRegroup =
    graph.addVariable(t.elementType(), grouped.shape(),
                      debugPrefix + "/preRegroup");
  auto preRegroupFlat =
    preRegroup.flatten(0, preRegroup.rank() - 2)
              .flatten(1, 3);

  // Build a map giving which intervals are mapped to each
  // IPU. Track which tiles on each IPU have any elements
  // mapped.
  const auto tMapping = graph.getTileMapping(groupedFlat);
  auto numTiles = tMapping.size();
  auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
  auto numIPUs = numTiles / tilesPerIPU;
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
    auto it = intervalsToIPU.find(
      Interval::right_open(t * elemsPerTransposition,
                           t * elemsPerTransposition + 1));
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
  // on which it should actually reside.
  for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
    const auto &mappedTiles = mappedTilesByIPU[ipu];
    const auto &transpositions = ipuTranspositions[ipu];
    auto numTiles = mappedTiles.size();
    auto numTranspositions =
      std::accumulate(transpositions.begin(), transpositions.end(),
                      std::size_t(0),
                      [](std::size_t t, const poplar::Interval &i) {
                        return t + i.size();
                      });
    if (!numTranspositions)
      continue;

    // Map transpositions on this IPU evenly across the tiles on which
    // elements of the source tensor reside.
    auto transpositionsPerTile =
      (numTranspositions + numTiles - 1) / numTiles;
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
  auto partiallyTransposed = partialTranspose(graph, preRegroup, *transposeCS);

  return ungroupTensor(partiallyTransposed, from, to);
}


} // namespace poplin
