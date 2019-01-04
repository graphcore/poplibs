// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#ifndef popops_Scatter_hpp
#define popops_Scatter_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popops {

/**
 * The scatter operation generates a result which is the value of the input
 * array `operand`, with several slices (at indices specified by
 * `scatter_indices`) updated with the values in `updates`.
 *
 *  \param graph                        The poplar graph
 *  \param operand                      Array to be scattered into
 *  \param indices                      Array containing the starting indices of
 *                                      the slices that must be scattered to
 *  \param updates                      Array containing the values that must be
 *                                      used for scattering
 *  \param indexVectorDim               The dimension in scatter_indices that
 *                                      contains the starting indices
 *  \param updateWindowDims             The set of dimensions in updates shape
 *                                      that are window dimensions
 *  \param insertWindowDims             The set of window dimensions that must
 *                                      be inserted into updates shape
 *  \param scatterDimsToOperandDims     A dimensions map from the scatter
 *                                      indices to the operand index space. This
 *                                      array is interpreted as mapping i to
 *                                      scatterDimsToOperandDims[i] . It has
 *                                      to be one-to-one and total
 *  \param prog                         The program to be extended
 *  \param debugPrefix                  The prefix prepended to debugging info
 *
 *  \note This is a near direct port of
 * https://www.tensorflow.org/xla/operation_semantics#scatter from
 * tensorflow/compiler/xla/service/scatter_expander.cc
 */
void scatter(poplar::Graph &graph, const poplar::Tensor &operand,
             const poplar::Tensor &indices, const poplar::Tensor &updates,
             std::size_t indexVectorDim, std::vector<unsigned> updateWindowDims,
             std::vector<std::size_t> insertWindowDims,
             std::vector<unsigned> scatterDimsToOperandDims,
             poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_DynamicSlice_hpp
