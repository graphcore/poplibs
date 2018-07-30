#include "ConvUtilInternal.hpp"

#include "popconv/ConvUtil.hpp"
#include "poplibs_support/gcd.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"

#include <poplar/Tensor.hpp>

using namespace poplar;

namespace popconv {

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
actsToInternalShape(const Tensor &act, unsigned numConvGroups) {
  if (act.dim(1) % numConvGroups != 0) {
    throw poputil::poplib_error("Number of input channels is not a multiple "
                               "of the number of convolutional groups");
  }
  return act.reshapePartial(1, 2, {numConvGroups, act.dim(1) / numConvGroups})
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
splitActivationChanGroups(const Tensor &act) {
  auto chansPerGroup = detectChannelGrouping(act);
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


Tensor groupWeights(const Tensor &weights) {
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
Tensor ungroupWeights(const Tensor &weights) {
  const auto rank = weights.rank();
  return weights.dimShufflePartial({1, 2}, {rank - 4, rank - 2})
                .reshapePartial(rank - 4, rank,
                                {weights.dim(1) * weights.dim(rank - 2),
                                 weights.dim(2) * weights.dim(rank - 1)});
}

} // namespace popconv
