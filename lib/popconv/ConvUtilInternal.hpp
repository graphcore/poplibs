// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popconv_ConvUtilInternal_hpp
#define popconv_ConvUtilInternal_hpp

/// A collection of utility functions that are internal to popconv.

#include <vector>
#include <poplar/Tensor.hpp>

namespace popconv {

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
actsToInternalShape(const poplar::Tensor &act, unsigned numConvGroups) ;

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
splitActivationChanGroups(const poplar::Tensor &act);

// Reshape the activations tensor from [G][C1][N]...[C2] shape to
// [G][N]...[C]
//
// Where C1 * C2 = C
poplar::Tensor
unsplitActivationChanGroups(const poplar::Tensor &act);

std::pair<unsigned, unsigned>
detectWeightsChannelGrouping(const poplar::Tensor &w);

// Groups tensor from standard convolution weight tensor shape [G]...[OC][IC]
// to internal shape [G][OC1][IC1]...[OC2][IC2]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
poplar::Tensor groupWeights(const poplar::Tensor &weights,
                            unsigned inChansPerGroup,
                            unsigned outChansPerGroup);


poplar::Tensor groupWeights(const poplar::Tensor &weights);

// Ungroups tensors from internal shape [G][OC1][IC1]...[OC2][IC2] to
// standard convolution weight tensor shape [G]...[OC][IC]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
poplar::Tensor ungroupWeights(const poplar::Tensor &weights);

} // End namespace popconv

#endif // popconv_ConvUtilInternal_hpp
