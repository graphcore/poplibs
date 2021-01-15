// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for collectives.
 *
 */

#ifndef popops_Collectives_hpp
#define popops_Collectives_hpp

#include "popops/CollectiveTypes.hpp"
#include "popops/Operation.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace popops {

/**
 * Given a tensor of rank 2 reduce across the outermost dimension using the
 * specified reduction operator. This function assumes index \c i in the
 * outermost dimension is mapped to IPU \c i. The result is distributed over
 * IPUs such that each IPU has a slice of the final result.
 */
/*[INTERNAL]
 * **Collectives options**
 *
 * * `method` (auto, clockwise_ring, anticlockwise_ring,
 *   bidirectional_ring_pair, meet_in_middle_ring, quad_directional_ring)
 *   [=auto]
 *
 *   The method to be used.
 *
 *   * **auto:** Automatically decide on the most optimal method.
 *
 *   * **clockwise_ring:** Send fragments clockwise around the ring. The number
 *     of fragments is equal to the number of IPUs in the ring.
 *
 *   * **anticlockwise_ring:** Send fragments anticlockwise around the ring. The
 *     number of fragments is equal to the number of IPUs in the ring.
 *
 *   * **bidirectional_ring_pair:** Split the data into two halves and use the
 *     clockwise ring algorithm on one half and the anticlockwise ring
 *     algorithm on the other in order to fully utilize the links in both
 *     directions. The number of fragments is equal to twice the number of
 *     IPUs in the ring.
 *
 *   * **meet_in_middle_ring:** Send half the fragments half way around the ring
 *     in the clockwise direction and half the fragments half way around the
 *     ring in the anticlockwise direction, meeting in the middle. The number
 *     of fragments is equal to the number of IPUs in the ring. The
 *     disadvantage compared to the "bidirectional_ring_pair" method is that
 *     the usage of available bandwidth is not quite optimal, in particular
 *     the final step only uses the links in one direction (assuming an even
 *     number of IPUs). The advantage is the that it requires fewer steps and
 *     allows the use of larger fragments.
 *
 *   * **quad_directional_ring:** Divide fragments in four and send each quarter
 *     around one of two rings using the mirrored and non mirrored ring pattern.
 */
/**
 * \param graph The graph.
 * \param toReduce The tensor to reduce. Each partial should be mapped
 *                 identically to the others across the IPUs within the rank.
 * \param op The reduction operator
 *           (for example, popops::CollectiveOperator::ADD).
 * \param prog The program sequence to add operations to.
 * \param debugContext Optional debug information.
 * \param options Collective options (not currently used).
 * \return A vector of chunks, where chunk \c i resides on IPU \c i.
 *        The chunks may have different numbers of elements (for example, when
 *        the number of IPUs does not exactly divide the number of elements).
 */
Chunks reduceScatter(poplar::Graph &graph, const poplar::Tensor &toReduce,
                     popops::CollectiveOperator op,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext = {},
                     const poplar::OptionFlags &options = {});
/** \deprecated
 *  **deprecated** Use reduceScatter with popops::CollectiveOperator instead */
inline Chunks reduceScatter(poplar::Graph &graph,
                            const poplar::Tensor &toReduce,
                            popops::Operation op,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return reduceScatter(graph, toReduce, operationToCollectiveOperator(op), prog,
                       debugPrefix, options);
}

/** Broadcast data distributed over all IPUs. This function assumes
 *  chunk \c i is mapped to IPU \c i.
 *
 *  \param graph The graph.
 *  \param toGather The chunks to gather.
 *  \param prog The program sequence to add operations to.
 *  \param debugContext Optional debug information.
 *  \param options Collective options. See reduceScatter().
 *  \return A 2D tensor that contains a copy of the data for each IPU.
 *          Index \c i in the outermost dimension of the result is mapped
 *          to IPU \c i.
 */
poplar::Tensor allGather(poplar::Graph &graph, const Chunks &toGather,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {},
                         const poplar::OptionFlags &options = {});

/** Perform an all-reduce operation on the specified tensor. This operation
 *  reduces across the outermost dimension of the input and produces a tensor
 *  with the same shape where the innermost dimension is the result of the
 *  reduction and the outermost dimension is a number of copies of the result.
 *  This function assumes index \c i in the outermost dimension of the input is
 *  mapped to IPU \c i. Index \c i in the outermost dimension of the result is
 *  mapped to IPU \c i.
 *
 *  \param graph The graph.
 *  \param toReduce The tensor to reduce. Each partial should be mapped
 *                  identically to the others across the ipus within the rank.
 *  \param op The reduction operator
 *            (for example, popops::CollectiveOperator::ADD).
 *  \param prog The program sequence to add operations to.
 *  \param debugContext Optional debug information.
 *  \param options Collective options. See reduceScatter().
 *  \return A tensor with the same shape as \p toReduce, where the innermost
 *          dimension is the result of the reduction and the outermost dimension
 *          has a number of copies of the result.
 */
poplar::Tensor allReduce(poplar::Graph &graph, const poplar::Tensor &toReduce,
                         popops::CollectiveOperator op,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {},
                         const poplar::OptionFlags &options = {});
/** \deprecated
 *  **deprecated** Use allReduce with popops::CollectiveOperator instead */

inline poplar::Tensor allReduce(poplar::Graph &graph,
                                const poplar::Tensor &toReduce,
                                popops::Operation op,
                                poplar::program::Sequence &prog,
                                const poplar::DebugContext &debugContext = {},
                                const poplar::OptionFlags &options = {}) {
  return allReduce(graph, toReduce, operationToCollectiveOperator(op), prog,
                   debugContext, options);
}

} // End namespace popops

#endif // popops_Collectives_hpp
