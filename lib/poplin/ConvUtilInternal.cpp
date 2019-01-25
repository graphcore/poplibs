#include "ConvUtilInternal.hpp"

#include "poplin/ConvUtil.hpp"
#include "poplibs_support/gcd.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"

#include <poplar/Tensor.hpp>

using namespace poplar;

namespace poplin {

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
      auto g = detectChannelGrouping(graph,
          groupedT.dimRoll(d, dims - 1).flatten(dims - 1, groupedT.rank()));
      auto thisGrouping = gcd<unsigned>(g / totalGrouping, groupedT.dim(d));
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

} // namespace poplin
