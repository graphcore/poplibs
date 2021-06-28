// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplin_FullyConnectedInternal_hpp
#define poplin_FullyConnectedInternal_hpp

#include "poplibs_support/PlanConstraints.hpp"
#include "poplin/ConvParams.hpp"
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>

namespace poplin {
namespace matmul {
class PlanningCache;
}

class PlanningCache;

/** Search for convolution plan and return plan constraints.
 *
 *  \param graph      The Poplar graph.
 *  \param params     Convolution parameters.
 *  \param options_   The structure describing options on how the
 *                    grouped multiplication should be implemented. See
 *                    matMul().
 *  \param cache      Optional pointer to a planning cache to use.
 */
poplibs_support::PlanConstraints
getPlanConstraints(const poplar::Graph &graph, const ConvParams &params,
                   const poplar::OptionFlags &options_ = {},
                   PlanningCache *cache = nullptr);

/** Get serial splits for the dimensions of the output matrix
 *
 *  Returns a tuple of (a) groupSplits
 *                     (b) outer field split for left side argument
 *                     (c) outer field split for right side argument
 *                     (d) inner field split
 */
std::tuple<unsigned, unsigned, unsigned, unsigned>
getMatMulSerialSplits(poplibs_support::PlanConstraints &planConstraints);

// TODO! These should be moved to the external API once we have more information
// on how these are going to be used.
/** Report the serial splitting of a grouped matrix of given size and
 *  options.
 *
 *  If C is of dimension [G,M,N] then serial splitting of G, M, N are given
 *  by the returned triplet of values.
 *
 *  \param graph      The Poplar graph.
 *  \param inputType  The data type of the elements of input matrices
 *  \param outputType The data type of the elements of the output matrix
 *  \param aShape     The shape of the left matrix in the multiplicaation.
 *  \param bShape     The shape of the right matrix in the multiplication.
 *  \param options_   The structure describing options on how the
 *                    grouped multiplication should be implemented. See
 *                    matMul().
 *  \param cache      Optional pointer to a planning cache to use.
 *
 *  \returns          Plan Constraints
 */
poplibs_support::PlanConstraints groupedMatMulPlanConstraints(
    const poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::OptionFlags &options_ = {},
    matmul::PlanningCache *cache = nullptr);

} // namespace poplin

#endif // poplin_FullyConnectedInternal_hpp
