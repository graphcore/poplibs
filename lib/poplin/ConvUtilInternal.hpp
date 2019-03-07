// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_ConvUtilInternal_hpp
#define poplin_ConvUtilInternal_hpp

/// A collection of utility functions that are internal to poplin.

#include <vector>
#include "poplin/ConvUtil.hpp"
#include <poplar/Tensor.hpp>
#include "poplin/internal/ConvPlan.hpp"
#include <boost/optional.hpp>

namespace poplin {

inline unsigned absdiff(unsigned a, unsigned b) {
  return a < b ? b - a : a - b;
}

struct PartialRow {
  unsigned b;
  std::vector<unsigned> outerFieldIndices;
  unsigned xBegin;
  unsigned xEnd;
  PartialRow(unsigned b, std::vector<unsigned> outerFieldIndices,
             unsigned xBegin, unsigned xEnd) :
    b(b),
    outerFieldIndices(std::move(outerFieldIndices)),
    xBegin(xBegin),
    xEnd(xEnd) {}
};

std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(unsigned batchElements,
                             const std::vector<unsigned> &tileConvOutSize,
                             unsigned numContexts,
                             const std::vector<unsigned> &inputDilation,
                             const std::vector<unsigned> &stride);


// Reshape the activations tensor from [N][G * C]... shape to
// [G][N]...[C] where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
poplar::Tensor
actsToInternalShape(const poplar::Tensor &act, unsigned numConvGroups,
                    unsigned chansPerGroup);

// Reshape the activations tensor from [G][N]...[C] shape to
// [N][G * C]... shape where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
poplar::Tensor
actsToExternalShape(const poplar::Tensor &act) ;

// Reshape the weights tensor from [G][OC][IC]... shape to
// [G]...[OC][IC].
poplar::Tensor
weightsToInternalShape(const poplar::Tensor &act) ;

// Reshape the weights tensor from [G]...[OC][IC] shape to
// [G][OC][IC]... shape.
poplar::Tensor
weightsToExternalShape(const poplar::Tensor &act) ;

// Reshape the activations tensor from [G][N]...[C] shape to
// [G][C1][N]...[C2]
//
// Where C1 * C2 = C
poplar::Tensor
splitActivationChanGroups(const poplar::Tensor &act, unsigned chansPerGroup);

// Reshape the activations tensor from [G][N]...[C] shape to
// [G][C1][N]...[C2]
//
// Where C1 * C2 = C
poplar::Tensor
splitActivationChanGroups(const poplar::Graph &graph,
                          const poplar::Tensor &act);

// Reshape the activations tensor from [G][C1][N]...[C2] shape to
// [G][N]...[C]
//
// Where C1 * C2 = C
poplar::Tensor
unsplitActivationChanGroups(const poplar::Tensor &act);

std::pair<unsigned, unsigned>
detectWeightsChannelGrouping(const poplar::Graph &graph,
                             const poplar::Tensor &w);

// Groups tensor from standard convolution weight tensor shape [G]...[OC][IC]
// to internal shape [G][OC1][IC1]...[OC2][IC2]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
poplar::Tensor groupWeights(const poplar::Tensor &weights,
                            unsigned inChansPerGroup,
                            unsigned outChansPerGroup);


poplar::Tensor groupWeights(const poplar::Graph &graph,
                            const poplar::Tensor &weights);

// Ungroups tensors from internal shape [G][OC1][IC1]...[OC2][IC2] to
// standard convolution weight tensor shape [G]...[OC][IC]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
poplar::Tensor ungroupWeights(const poplar::Tensor &weights);

std::vector<unsigned>
dimsFromSpatialDims(std::vector<unsigned> spatialDims, bool isActs);

std::vector<unsigned>
actDimsFromSpatialDims(const std::vector<unsigned> &spatialDims);

std::vector<unsigned>
weightDimsFromSpatialDims(const std::vector<unsigned> &spatialDims);

// Stride is what's used to move down one element in the input field by
// the vertex. fieldsElems is the number of field elements in all but the
// outermost dimension
int getInRowStride(const ConvParams &params, unsigned fieldElems,
                   bool useConvPartial1x1OutVertex,
                   unsigned convUnitWeightHeight);

// Split field dimensions such that the stride fits machine stride. This
// implementation only splits field such that input stride fits. The outermost
// dimension is not split
Partition
splitConvIntoAmpVertices(const ConvParams &params,
                         unsigned numMachineStrideBits,
                         int inStride, int inRowStride);

// Returns a list with the innermost grouped dimension first
// moving outwards, with groupings for each. The same dimension may appear
// more than once. This uses detectChannelGrouping iteratively.
using GroupingInfo = std::pair<unsigned, unsigned>;
std::vector<GroupingInfo>
detectDimGroupings(const poplar::Graph &graph, const poplar::Tensor &t);

// Based on minimum requirement for targeting Transpose fast path(s).
// If there is no fast path available, we will fall back on a single
// element transpose which presupposes that this will still be more
// beneficial in terms of cycles/memory than no transpose at all which
// may not be true but doesn't come up with the data type support we
// currently have in convolutions.
unsigned
getMinimumRegroupGrainSize(const poplar::Type &type);

// Takes a tensor, an optional compute set (which will be
// created if it is not initialised already), and returns
// a tensor of the same shape but with the elements in tile
// memory rearranged according to the grouping info provided.
// This maps the transposition across only the tiles to which
// the original tensor is already mapped, though it may map
// transpositions across these tiles in whichever way in order
// to better balance the regrouping operation.
//
// The grouped dimensions may not be split over multiple
// IPUs and all elements in the product of the groups are
// assumed to reside on the same tile.
poplar::Tensor
regroupTensor(poplar::Graph &graph, const poplar::Tensor &t,
              poplar::program::Sequence &copies,
              boost::optional<poplar::ComputeSet> &transposeCS,
              const GroupingInfo &from, const GroupingInfo &to,
              const std::string &debugPrefix);
} // End namespace poplin

#endif // poplin_ConvUtilInternal_hpp
