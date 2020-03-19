// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popops_Rearrange_hpp
#define popops_Rearrange_hpp

#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>

#include <functional>

#include <poputil/VarStructure.hpp>

namespace popops {
namespace rearrange {

/** Determine if a fast transposition codelet may be used based on the given
 *  target/data type/no. of rows/no. of columns.
 *
 *  \param target The target the operation will be targeted at.
 *  \param type The data type of the tensor to transpose.
 *  \param numRows The no. of rows in each transposition to perform.
 *  \param numColumns The no. of columns in each transposition to perform.
 *
 * \returns A boolean indicating whether or not the fast transposition codelets
 *          can be targeted based on the given parameters.
 */
bool canUseFastTranspose(const poplar::Target &target, const poplar::Type &type,
                         unsigned numRows, unsigned numColumns,
                         unsigned numTranspositions);

/// Transposes of a set of matrices stored on multiple tiles.
/// This adds all the needed vertices on the graph.
///
/// \param graph, cs  The graph and compute set to add the vertices to.
///
/// \param dType, rows, cols   The type and dimensions of the matrices to be
///                 transposed, the same for all of them.
///
/// \param mapping  A vector with 'number of tiles' elements, where each element
///                 is a vector of intervals indicating which matrices to be
///                 transposed are mapped (possibly partially) on each tile.
///
/// \param getInOut A function: `pair<Tensor, Tensor> getInOut(size_t index)`,
///                 which, given as input an index inside the intervals
///                 specified in 'mapping', returns a std::pair of Tensors
///                 (in, out) which are the input and output matrix for the
///                 'index' transposition. The 'in' and 'out' return values are
///                 2D matrices, but they must be flattened to a single
///                 dimension.
///
void addTransposeVertices(
    poplar::Graph &graph, const poplar::ComputeSet &cs,
    const poplar::Type &dType, unsigned rows, unsigned cols,
    const poplar::Graph::TileToTensorMapping &mapping,
    std::function<std::pair<const poplar::Tensor, const poplar::Tensor>(size_t)>
        getInOut);

/// Transpose the innermost pair of dimensions of the specified tensor, writing
/// the results to a new tensor. This function assumes order of the underlying
/// storage matches the order of the elements in the tensor. This function is
/// optimized for group sizes that are typical of the underlying memory
/// layout of convolution activations / weights - it may be inefficient for
/// other group sizes.
poplar::Tensor partialTranspose(poplar::Graph &graph, const poplar::Tensor &in,
                                const poplar::ComputeSet &cs,
                                const std::string &debugPrefix = "");

/** Get the smallest grouping we can transpose between for the given type
 *  using fast transposition codelets.
 *
 *  \param type The data type to be transposed.
 *
 *  \returns The smallest size of grouping that can be efficiently transposed
 *           for the given type.
 */
unsigned getMinimumRegroupGrainSize(const poplar::Type &type);

/** Insert copies or other operations into the given programs/compute sets
 *  to transform the grouping found on the given tensor from \p from to
 *  \p to. This is a no-op for a one-dimensional tensor.
 *
 *  \param graph       The graph to add the operation to.
 *  \param t           The tensor to regroup.
 *  \param copies      A poplar sequence to add pre-arranging copies to.
 *  \param transposeCS A compute set that may or may not have vertices
 *                     added to it to perform the regrouping operation.
 *  \param from        A grouping that is applied to the given tensor \p t to
 *                     rearrange from.
 *  \param to          A grouping wanted on the returned tensor.
 *  \param debugPrefix An optional string to be prepended to any debug
 *                     info.
 *
 *  \returns A tensor with the contents of \p t but laid out such that
 *           it has the grouping specified in \p to.
 */
poplar::Tensor regroupTensor(poplar::Graph &graph, const poplar::Tensor &t,
                             poplar::program::Sequence &copies,
                             const poplar::ComputeSet &transposeCS,
                             const poputil::GroupingInfo &from,
                             const poputil::GroupingInfo &to,
                             const std::string &debugPrefix);

/** If possible and runtime efficient, add an operation to rearrange the given
 *  tensor in memory such that the grouping of the resulting tensor matches
 *  that of the reference tensor, or a factor of that grouping if it
 *  balances memory usage across the target better.
 *
 *  \param graph       The graph to add the operation to.
 *  \param in          The tensor to maybe regroup.
 *  \param ref         A reference tensor which will be introspected to find a
 *                     grouping to apply to the returned tensor.
 *  \param prog        A poplar sequence to add the regrouping operation to.
 *  \param debugPrefix An optional string to be prepended to any debug info.
 *
 *  \returns A tensor with the contents of the given tensor \p in rearranged in
 *           memory to have a grouping matching \p ref.
 */
poplar::Tensor regroupIfBeneficial(poplar::Graph &graph,
                                   const poplar::Tensor &in,
                                   const poplar::Tensor &ref,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix = "");

/** If possible and runtime efficient, add an operation to rearrange the given
 *  tensor in memory such that the resulting tensor has a grouping in the
 *  innermost dimension equivalent to, or a factor of the given preferred
 *  grouping if it balances memory usage across the target better.
 *
 *  \param graph             The graph to add the operation to.
 *  \param in                The tensor to maybe regroup.
 *  \param preferredGrouping A size of grouping of the innermost dimension of
 *                           the given tensor to regroup to.
 *  \param prog              A poplar sequence to add the regrouping operation
 *                           to.
 *  \param debugPrefix       An optional string to be prepended to any debug
 * info.
 *
 *  \returns A tensor with the contents of the given tensor \p in rearranged in
 *           memory to have a grouping matching \p ref.
 */
poplar::Tensor regroupIfBeneficial(poplar::Graph &graph,
                                   const poplar::Tensor &in,
                                   std::size_t preferredGrouping,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix = "");

} // end namespace rearrange
} // end namespace popops

#endif // popops_Rearrange_hpp
