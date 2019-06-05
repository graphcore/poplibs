// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_ConvUtil_hpp
#define poplin_ConvUtil_hpp
#include <poplin/Convolution.hpp>
#include <tuple>
#include <vector>

/// A collection of utility functions to assist calculation of input/output
/// ranges when moving a 2-dimensional kernel over a larger 2-dimensional
/// space (e.g. in convolution or pooling layers

namespace poplin {

/// Return the output size when the specified dilation is applied to an
/// input of the specified size.
unsigned getDilatedSize(unsigned size, unsigned dilation);

/// Return the index of the input element that is multiplied by the specified
/// kernel index to produce the specified output.
/// Return ~0U if there is no such input element.
unsigned
getInputIndex(unsigned dim, unsigned outputIndex, unsigned kernelIndex,
              const ConvParams &params);

/// Return the index of the kernel element that is multiplied by the specified
/// input index to produce the specified output.
/// Return ~0U if there is no such kernel element.
unsigned
getKernelIndex(unsigned dim, unsigned outputIndex,
               unsigned inputIndex, const ConvParams &params);

/// Given an output range, return the subset whose calculation
/// involves the specified kernel index.
std::pair<unsigned, unsigned>
getOutputRangeForKernelIndex(unsigned dim,
                             std::pair<unsigned, unsigned> outputRange,
                             unsigned kernelIndex, const ConvParams &params);

/// Given an output range, return the subset whose calculation
/// involves the specified input.
std::pair<unsigned, unsigned>
getOutputRangeForInputIndex(unsigned dim,
                            std::pair<unsigned, unsigned> outputRange,
                            unsigned inputIndex, const ConvParams &params);

/// Given an output range, return the subset whose calculation
/// involves the specified range of kernel indicies.
std::pair<unsigned, unsigned>
getOutputRangeForKernelRange(unsigned dim,
                             std::pair<unsigned, unsigned> outputRange,
                             std::pair<unsigned, unsigned> kernelIndexRange,
                             const ConvParams &params);

/// Given an output range, return the subset whose calculation
/// involves the specified range of input indicies.
std::pair<unsigned, unsigned>
getOutputRangeForInputRange(unsigned dim,
                            std::pair<unsigned, unsigned> outputRange,
                            std::pair<unsigned, unsigned> inputRange,
                            const ConvParams &params);

/// Return the input range that is associated with
/// the specified kernel index when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              unsigned kernelIndex, const ConvParams &params);

/// Return the kernel range that is associated with
/// the specified input index when calculating the specified output range.
std::pair<unsigned, unsigned>
getKernelRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               unsigned inputIndex, const ConvParams &params);

/// Return the input range that is associated with the specified kernel index
/// range when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              std::pair<unsigned, unsigned> kernelIndexRange,
              const ConvParams &params);

/// Return the kernel range that is associated with the specified input index
/// range when calculating the specified output range.
std::pair<unsigned, unsigned>
getKernelRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               std::pair<unsigned, unsigned> inputRange,
               const ConvParams &params);

inline std::pair<unsigned, unsigned>
getInputRange(unsigned dim, unsigned outputIndex,
              std::pair<unsigned, unsigned> kernelIndexRange,
              const ConvParams &params) {
  return getInputRange(dim, {outputIndex, outputIndex + 1},
                       kernelIndexRange, params);
}

inline std::pair<unsigned, unsigned>
getInputRange(unsigned dim, unsigned outputIndex, const ConvParams &params) {
  return getInputRange(dim, outputIndex, {0, params.kernelShape[dim]},
                       params);
}

inline std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              const ConvParams &params) {
  return getInputRange(dim, outputRange, {0, params.kernelShape[dim]},
                       params);
}

ConvParams canonicalizeParams(const ConvParams &params);

// Given a set of parameters, return the set of params that
// represent the convolution to be applied to the output gradients
// to get the input gradients (provided the weights have been
// transposed in the channel axes and flipped in the spatial axes).
ConvParams getGradientParams(const ConvParams &params);

// Given a set of convolution parameters, return the set of params that
// represent the convolution to be applied to the output gradients to get the
// weight update gradients
ConvParams getWeightUpdateParams(ConvParams fwdParams);

// Determines if a fast transposition may be used based on the machine model,
// data type and transposition parameters
bool useFastTranspose(const poplar::Target &target,
                      const poplar::Type &type,
                      unsigned numRows,
                      unsigned numColumns,
                      unsigned numTranspositions);

///
/// Transpositions of a set of matrices stored on multiple tiles.
/// This adds all the needed vertices on the graph.
///
/// \param graph, cs  The graph and compute set to add the vertices to.
///
/// \param dType, rows, cols   The type and dimensions of the matrices to be
///                 transposed, the same for all of them.
///
/// \param mapping  A vector with <num tiles> elements, where each element is a
///                 vector of intervals indicating which matrices to be
///                 transposed are mapped (possibly partially) on each tile.
///
/// \param getInOut A function:   pair<Tensor, Tensor> getInOut(size_t index),
///                 which, given as input an index inside the intervals
///                 specified in 'mapping', returns a std::pair of Tensors
///                 (in, out) which are the input and output matrix for the
///                 'index' transposition. The 'in' and 'out' return values are
///                 2D matrices, but they must be flattened to a single
///                 dimension.
///
void
addTransposeVertices(poplar::Graph &graph,
                     poplar::ComputeSet &cs,
                     poplar::Type dType, unsigned rows, unsigned cols,
                     const poplar::Graph::TileToTensorMapping &mapping,
                     std::function<
                          std::pair<const poplar::Tensor,
                                    const poplar::Tensor>(size_t)> getInOut);

/// Transpose the innermost pair of dimensions of the specified tensor, writing
/// the results to a new tensor. This function assumes order of the underlying
/// storage matches the order of the elements in the tensor. This function is
/// optimized for group sizes that are typical of the underlying memory
/// layout of convolution activatons / weights - it may be inefficient for
/// other group sizes.
poplar::Tensor
partialTranspose(poplar::Graph &graph,
                 const poplar::Tensor &in,
                 poplar::ComputeSet cs,
                 const std::string &debugPrefix =  "");

// Take a tensor \p in_ and try to regroup it depending on whether
// a second tensor \p ref_ has a different grouping. If the same dimension
// is grouped, the original tensor is returned. The tensors should be of the
// same shape and of shape [N][C][...] where N is the batch size, C is the
// number of channels and ... is a nD field.
//
// The regrouping operation returns a tensor of the same shape but with the
// elements in tile memory rearranged according to the grouping info deduced
// from the \p ref_ tensor. This maps the transposition across only the tiles
// to which the original tensor is already mapped, though it may map
// transpositions across these tiles in whichever way in order
// to better balance the regrouping operation.
//
// The grouped dimensions may not be split over multiple
// IPUs and all elements in the product of the groups are
// assumed to reside on the same tile.
poplar::Tensor
regroupIfBeneficial(poplar::Graph &graph,
                    const poplar::Tensor &in_,
                    const poplar::Tensor &ref_,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");
}
#endif // poplin_ConvUtil_hpp
