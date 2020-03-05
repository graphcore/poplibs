// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_ConvUtilInternal_hpp
#define poplin_ConvUtilInternal_hpp

/// A collection of utility functions that are internal to poplin.

#include "ConvPlan.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/TileMapping.hpp"
#include <boost/optional.hpp>
#include <poplar/Tensor.hpp>
#include <vector>

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
             unsigned xBegin, unsigned xEnd)
      : b(b), outerFieldIndices(std::move(outerFieldIndices)), xBegin(xBegin),
        xEnd(xEnd) {}
};

ConvParams getZeroConv(const ConvParams &params);

std::vector<std::vector<PartialRow>> partitionConvPartialByWorker(
    unsigned batchElements, const std::vector<unsigned> &tileConvOutSize,
    unsigned numContexts, const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride);

// Reshape the activations tensor from [N][G * C]... shape to
// [G][N]...[C] where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
poplar::Tensor actsToInternalShape(const poplar::Tensor &act,
                                   unsigned numConvGroups,
                                   unsigned chansPerGroup);

// Reshape the activations tensor from [G][N]...[C] shape to
// [N][G * C]... shape where N is the batch size, ... is the set of spatial
// dimensions (usually [W][H]), G is the number of groups and C is the number
// of channels in each group.
poplar::Tensor actsToExternalShape(const poplar::Tensor &act);

// Reshape the weights tensor from [G][OC][IC]... shape to
// [G]...[OC][IC].
poplar::Tensor weightsToInternalShape(const poplar::Tensor &act);

// Reshape the weights tensor from [G]...[OC][IC] shape to
// [G][OC][IC]... shape.
poplar::Tensor weightsToExternalShape(const poplar::Tensor &act);

// Reshape the activations tensor from [G][N]...[C] shape to
// [G1][C1][N]...[G2][C2]
//
// Where
//  G1 * G2 = G
//  C1 * C2 = C
poplar::Tensor splitActivationIntoGroups(poplar::Tensor act,
                                         unsigned convGroupsPerGroup,
                                         unsigned chansPerGroup);

// Reshape the activations tensor from [G1][C1][N]...[G2][C2] shape to
// [G][N]...[C]
//
// Where
//  G1 * G2 = G
//  C1 * C2 = C
poplar::Tensor unsplitActivationFromGroups(poplar::Tensor act);

// Groups tensor from standard convolution weight tensor shape [G]...[OC][IC]
// to internal shape [G1][OC1][IC1]...[G2][OC2][IC2]
//
// Where
//  G1 * G2 = G
//  OC1 * OC2 = OC
//  IC1 * IC2 = IC
poplar::Tensor splitWeightsIntoGroups(poplar::Tensor weights,
                                      unsigned convGroupsPerGroup,
                                      unsigned inChansPerGroup,
                                      unsigned outChansPerGroup);

poplar::Tensor splitWeightsFromGroups(const poplar::Graph &graph,
                                      const poplar::Tensor &weights);

// Ungroups tensors from internal shape [G1][OC1][IC1]...[G2][OC2][IC2] to
// standard convolution weight tensor shape [G]...[OC][IC]
//
// Where
//  G1 * G2 = G
//  OC1 * OC2 = OC
//  IC1 * IC2 = IC
poplar::Tensor unsplitWeightsFromGroups(poplar::Tensor weights);

struct ChannelGrouping {
  unsigned convGroupsPerGroup;
  unsigned chansPerGroup;
};

// the activations should be allocated with a double grouping of
// convGroupsPerGroup and the {in,out}ChansPerGroup as the 2 innermost
// dimensions. this function uses tensor introspection to attempt to work out
// what those groupings are.
ChannelGrouping detectChannelGrouping(const poplar::Graph &graph,
                                      const poplar::Tensor &acts);

struct WeightChannelGrouping {
  unsigned convGroupsPerGroup;
  unsigned outChansPerGroup;
  unsigned inChansPerGroup;
};

// the weights should be allocated with a triple grouping of convGroupsPerGroup,
// outChansPerGroup and inChansPerGroup as the 3 innermost dimensions. this
// function uses tensor introspection to attempt to work out what those
// groupings are.
WeightChannelGrouping
detectWeightsChannelGrouping(const poplar::Graph &graph,
                             const poplar::Tensor &weights);

std::vector<unsigned> dimsFromSpatialDims(std::vector<unsigned> spatialDims,
                                          bool isActs);

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
Partition splitConvIntoAmpVertices(const ConvParams &params,
                                   unsigned numMachineStrideBits, int inStride,
                                   int inRowStride);

// Based on minimum requirement for targeting Transpose fast path(s).
// If there is no fast path available, we will fall back on a single
// element transpose which presupposes that this will still be more
// beneficial in terms of cycles/memory than no transpose at all which
// may not be true but doesn't come up with the data type support we
// currently have in convolutions.
unsigned getMinimumRegroupGrainSize(const poplar::Type &type);

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
poplar::Tensor regroupTensor(poplar::Graph &graph, const poplar::Tensor &t,
                             poplar::program::Sequence &copies,
                             boost::optional<poplar::ComputeSet> &transposeCS,
                             const poputil::GroupingInfo &from,
                             const poputil::GroupingInfo &to,
                             const std::string &debugPrefix);

} // End namespace poplin

#endif // poplin_ConvUtilInternal_hpp
