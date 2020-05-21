// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef poplin_ConvUtilInternal_hpp
#define poplin_ConvUtilInternal_hpp

/// A collection of utility functions that are internal to poplin.

#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include "MultiConvolutionInternal.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/MultiConvolution.hpp"
#include "poputil/VarStructure.hpp"
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

// Converts OptionFlags in multiconv input structures to ConvOptions
std::vector<multiconv::internal::CreateTensorArgs>
convertToConvOptions(poplar::Graph &graph,
                     const std::vector<multiconv::CreateTensorArgs> &args);

std::vector<multiconv::internal::ConvolutionArgs>
convertToConvOptions(poplar::Graph &graph,
                     const std::vector<multiconv::ConvolutionArgs> &args);

std::vector<multiconv::internal::CalculateWeightDeltasArgs>
convertToConvOptions(
    poplar::Graph &graph,
    const std::vector<multiconv::CalculateWeightDeltasArgs> &args);

std::vector<multiconv::internal::ConvWeightUpdateArgs<poplar::Tensor>>
convertToConvOptions(poplar::Graph &graph,
                     const std::vector<multiconv::ConvWeightUpdateArgs> &args);

std::vector<multiconv::internal::ConvWeightUpdateArgs<float>>
convertToConvOptions(
    poplar::Graph &graph,
    const std::vector<multiconv::ConvWeightUpdateArgsScalar> &args);

// Checks that multiple ConvolutionArgs can be combined
template <typename T> bool canBeCombined(const std::vector<T> &convolutionArgs);

// Returns a vector of groups of combinable convolution arguments
template <typename T>
std::vector<std::vector<const T *>>
groupCombinables(const std::vector<T> &args);

// Returns the combination (aggregates convolution parameters and concatenates
// input tensors) of multiple compatible convolution arguments.
multiconv::internal::CreateTensorArgs
combine(const std::vector<multiconv::internal::CreateTensorArgs> &args);
multiconv::internal::ConvolutionArgs
combine(const std::vector<multiconv::internal::ConvolutionArgs> &args);
multiconv::internal::CalculateWeightDeltasArgs combine(
    const std::vector<multiconv::internal::CalculateWeightDeltasArgs> &args);
template <typename T>
multiconv::internal::ConvWeightUpdateArgs<T>
combine(const std::vector<multiconv::internal::ConvWeightUpdateArgs<T>> &args);

template <typename T>
std::vector<T> combine(const std::vector<std::vector<const T *>> &groups) {
  std::vector<T> args;
  for (const auto &group : groups) {
    std::vector<T> combinedArgs;
    for (const auto ca : group) {
      combinedArgs.push_back(*ca);
    }
    args.push_back(combine(combinedArgs));
  }
  return args;
}

// Splits the result of a combined multi-convolution
std::vector<poplar::Tensor>
splitOutput(const std::vector<CanonicalConvParams> &convParams,
            const poplar::Tensor &out);
std::vector<poplar::Tensor>
splitInput(const std::vector<CanonicalConvParams> &convParams,
           const poplar::Tensor &in);
std::vector<poplar::Tensor>
splitWeights(const std::vector<CanonicalConvParams> &convParams,
             const poplar::Tensor &in);

template <typename T, typename F>
std::vector<poplar::Tensor>
split(const std::vector<std::vector<const T *>> &groups,
      const std::vector<poplar::Tensor> &outs, const F &splitFn) {
  std::vector<poplar::Tensor> splitOuts;
  for (unsigned i(0); i < groups.size(); ++i) {
    std::vector<CanonicalConvParams> convParams;
    for (const auto ca : groups[i]) {
      convParams.push_back(ca->params);
    }
    const auto s = splitFn(convParams, outs[i]);
    splitOuts.insert(splitOuts.end(), s.begin(), s.end());
  }
  return splitOuts;
}

// Given a vector of group weights and a number of elements,
// distribute them among the groups proportionally to their weights.
// If noEmptyGroups is set, at least one element is assigned to every group
extern std::vector<unsigned>
splitElementsInWeightedGroups(const std::vector<uint64_t> &groups,
                              unsigned elements);

// Given a vector of operation FLOPs and a number of tiles,
// return a tile subset for each operation proportional to its FLOPs
extern std::vector<unsigned>
splitTilesByComp(const std::vector<uint64_t> &flops, unsigned numTiles);

/// Given a vector of convolution parameters and a number of tiles,
/// return a distribution of tiles according to the expected FLOPs for each
/// convolution
std::vector<unsigned>
splitTilesByComp(const std::vector<ConvParams> &convParams, unsigned numTiles);

// Given a vector of group sizes and an element index,
// returns the index of the group that contains that element
// assuming that elements are sequentially assigned to each group
unsigned getGroupIndex(const std::vector<unsigned> &groups,
                       const unsigned element);

// print ConvParams to INFO log level.
void log(unsigned indent, const ConvParams &params);

} // End namespace poplin

#endif // poplin_ConvUtilInternal_hpp
