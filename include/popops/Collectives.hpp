// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for collectives.
 *
 */

#ifndef popops_Collectives_hpp
#define popops_Collectives_hpp

#include "popops/Operation.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace popops {

/**
 * Supported collective operators.
 */
enum class CollectiveOperator {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND, ///< Only supports boolean operands.
  LOGICAL_OR,  ///< Only supports boolean operands.
  SQUARE_ADD,  ///< Squares each element before applying ADD reduction.
  LOCAL,       ///< Do nothing and keep the local value.
};

/**
 * Parse token from input stream \is to \op. Valid input values are the
 * stringified enumerations, for example "ADD" or "MUL".
 * \return The original input stream.
 */
std::istream &operator>>(std::istream &is, CollectiveOperator &op);

/**
 * Write \op to output stream \os. The value written is the stringified
 * enumeration, for example "ADD" or "MUL".
 * \return The original output stream.
 */
std::ostream &operator<<(std::ostream &os, const CollectiveOperator &op);

/** Convert from popops::Operation to popops::CollectiveOperator */
CollectiveOperator operationToCollectiveOperator(const Operation &col);

/// Represents a section of a tensor mapped to an IPU.
struct Chunk {
  poplar::Tensor tensor;
  /// Ring index (data parallel index).
  unsigned index;
  /// Offset within rank (model parallel index).
  unsigned offset;
  Chunk() = default;
  Chunk(poplar::Tensor tensor, unsigned index, unsigned offset)
      : tensor(tensor), index(index), offset(offset) {}
};

/// A vector of Chunk data.
struct Chunks {
  /// Used to undo shuffles introduced by scatter.
  poplar::Tensor originalInput;
  /// Chunks produced by the scatter step.
  std::vector<Chunk> chunks;
  Chunks() = default;
  Chunks(unsigned size) : chunks(std::vector<Chunk>(size)) {}
};

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
 *   bidirectional_ring_pair, meet_in_middle_ring) [=auto]
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

/** Perform an all-reduce operation on the specified replicated tensor.
 *  This operation reduces across the tensors that the replicated tensor is a
 *  handle for. The result is returned as a replicated tensor.
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::allReduce instead.
 *
 *  \param graph The replicated graph the input tensor belongs to.
 *  \param data The replicated tensor to reduce.
 *  \param op The reduction operator
 *            (for example, popops::CollectiveOperator::ADD).
 *  \param prog The program sequence to add operations to.
 *  \param debugContext Optional debug information.
 *  \param options Collective options. See reduceScatter().
 *  \return A replicated tensor with the reduction of \p data.
 */
poplar::Tensor
replicatedAllReduce(poplar::Graph &graph, const poplar::Tensor &data,
                    popops::CollectiveOperator op,
                    poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});
/** \deprecated
 *  **deprecated** Use gcl::allReduce with popops::CollectiveOperator instead */
inline poplar::Tensor
replicatedAllReduce(poplar::Graph &graph, const poplar::Tensor &data,
                    popops::Operation op, poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {}) {
  return replicatedAllReduce(graph, data, operationToCollectiveOperator(op),
                             prog, debugContext, options);
}

/** Perform an all-reduce operation on the specified replicated tensor.
 *  This operation reduces across the tensors that the replicated tensor is a
 *  handle for.
 *
 *  Same as replicatedAllReduce() but writes the result to the output tensor
 *  instead of creating a new one.
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::allReduceToDestination instead.
 *
 *  \param graph The replicated graph the input tensor belongs to.
 *  \param data The replicated tensor to reduce.
 *  \param output The result of the reduction of \p data.
 *  \param op The reduction operator
 *            (for example, popops::CollectiveOperator::ADD).
 *  \param prog The program sequence to add operations to.
 *  \param debugContext Optional debug information.
 *  \param options Collective options. See reduceScatter().
 */
void replicatedAllReduceWithOutput(
    poplar::Graph &graph, const poplar::Tensor &data, poplar::Tensor &output,
    popops::CollectiveOperator op, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** \deprecated
 *  **deprecated** Use gcl::allReduceWithOutput instead */
inline void
replicatedAllReduceWithOutput(poplar::Graph &graph, const poplar::Tensor &data,
                              poplar::Tensor &output, popops::Operation op,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext = {},
                              const poplar::OptionFlags &options = {}) {
  replicatedAllReduceWithOutput(graph, data, output,
                                operationToCollectiveOperator(op), prog,
                                debugContext, options);
}

/** Perform an all-reduce operation on the specified replicated tensor.
 *  This operation reduces across the tensors the replicated tensor is a handle
 *  for. The result is written to back to the input data tensor.
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::allReduceInPlace instead.
 *
 *  \param graph The replicated graph the input tensor belongs to.
 *  \param data The replicated tensor to reduce and written to.
 *  \param op The reduction operator (for example, `CollectiveOperatior::ADD`)
 *  \param prog The program sequence to add operations to.
 *  \param debugContext Optional debug information.
 *  \param options Collective options. See reduceScatter().
 */
void replicatedAllReduceInPlace(poplar::Graph &graph, poplar::Tensor &data,
                                popops::CollectiveOperator op,
                                poplar::program::Sequence &prog,
                                const poplar::DebugContext &debugContext = {},
                                const poplar::OptionFlags &options = {});
/** \deprecated
 *  **deprecated** Use gcl::allReduceInPlace instead */
inline void replicatedAllReduceInPlace(
    poplar::Graph &graph, poplar::Tensor &data, popops::Operation op,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  replicatedAllReduceInPlace(graph, data, operationToCollectiveOperator(op),
                             prog, debugPrefix, options);
}

/** Perform an all-reduce operation on the specified replicated tensor.
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::allReduce instead.
 */
poplar::Tensor
replicatedAllReduce(poplar::Graph &graph, poplar::Graph &parentGraph,
                    const poplar::Tensor &data, popops::CollectiveOperator op,
                    poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});

/** \deprecated
 *  **deprecated** Use gcl::allReduce instead */
inline poplar::Tensor
replicatedAllReduce(poplar::Graph &graph, poplar::Graph &parentGraph,
                    const poplar::Tensor &data, popops::Operation op,
                    poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {}) {
  return replicatedAllReduce(graph, parentGraph, data,
                             operationToCollectiveOperator(op), prog,
                             debugContext, options);
}

/** Reduce the replicated rank-1 tensor \p toReduce with the result scattered
 *  across the replicas.
 *
 *  For an input of shape [numElements] mapped to a single IPU per replica, the
 *  output will have shape [ceil(numElements / replicationFactor)]. If
 *  replicationFactor does not evenly divide numElements, the result is
 *  zero-padded. For instance:
 *  * Before:
 *    * Replica0: toReduce[x0, y0, z0]
 *    * Replica1: toReduce[x1, y1, z1]
 *  * After:
 *    * Replica0: result[op(x0, x1), op(y0, y1)]
 *    * Replica1: result[op(z0, z1), 0]
 *
 *  For an input of shape [numElementsIPU0 + numElementsIPU1 + ...] mapped to
 *  multiple IPUs per replica, the output will have shape:
 *  [ceil(numElementsIPU0 /
 *  replicationFactor) + ceil(numElementsIPU1 / replicationFactor) + ...] with
 *  the result grouped per IPU. If replicationFactor does not evenly divide the
 *  number of elements on an IPU, the result is zero-padded per IPU.
 *  For instance:
 *  * Before:
 *    * Replica0: toReduce[x0,   y0,   z0,   w0]
 *    * Replica1: toReduce[x1,   y1,   z1,   w1]
 *    * Replica2: toReduce[x2,   y2,   z2,   w2]
 *    * Replica3: toReduce[x3,   y3,   z3,   w3]
 *    * Mapping:  toReduce[IPU0, IPU0, IPU0, IPU1]
 *  * After:
 *    * Replica0: result[op(x0, x1, x2, x3), op(w0, w1, w2, w3)]
 *    * Replica1: result[op(y0, y1, y2, y3), 0]
 *    * Replica2: result[op(z0, z1, z2, z3), 0]
 *    * Replica3: result[0,                  0]
 *    * Mapping:  result[IPU0,               IPU1]
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::reduceScatter instead.
 */
poplar::Tensor
replicatedReduceScatter(poplar::Graph &graph, const poplar::Tensor &toReduce,
                        popops::CollectiveOperator op,
                        poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {});
/** \deprecated
 *  **deprecated** Use gcl::reduceScatter instead */
inline poplar::Tensor
replicatedReduceScatter(poplar::Graph &graph, const poplar::Tensor &toReduce,
                        popops::Operation op, poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {}) {
  return replicatedReduceScatter(graph, toReduce,
                                 operationToCollectiveOperator(op), prog,
                                 debugContext, options);
}

/** Gather the replicated tensor \p toGather and return the result so each
 *  replica will have a copy of **all** other replicas' \p toGather tensors. For
 *  instance:
 *
 *  * Before:
 *    * Replica0: toGather[x,y]
 *    * Replica1: toGather[z,w]
 *    * Replica2: toGather[x1, y1]
 *
 *  * After allGather: \n
 *    * Replica0: result[x,y,z,w,x1,y1] \n
 *    * Replica1: result[x,y,z,w,x1,y1] \n
 *    * Replica2: result[x,y,z,w,x1,y1] \n
 *
 *  For an input of shape [incomingShape] the output will be
 *  [replicationFactor][incomingShape].
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::allGather instead.
 */
poplar::Tensor
replicatedAllGather(poplar::Graph &graph, const poplar::Tensor &toGather,
                    poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});

/** Perform an all-to-all exchange of the elements of the input tensor based on
 *  replica ID. The shape of the input must have the number of replicas in the
 *  graph as its first or only dimension. That dimension will be used to split
 *  up the tensor being sent, with each replica sending all splits except for
 *  the split index which matches its replica ID. That is, replica 2 will not
 *  send input[2] and so on.
 *
 *  The replica receiving the slice will copy that incoming slice into the
 *  output at the index which matches the replica ID of the replica which sent
 *  it. For instance:
 *  * Input tensor:
 *    * Replica0: Tensor T[x0,x1,x2]
 *    * Replica1: Tensor T[y0,y1,y2]
 *    * Replica2: Tensor T[z0,z1,z2]
 *  * Output tensor:
 *    * Replica0: Tensor T[x0,y0,z0]
 *    * Replica1: Tensor T[x1,y1,z1]
 *    * Replica2: Tensor T[x2,y2,z2]
 *
 * \deprecated
 *
 * **Deprecated:** This function is deprecated and will be removed in a future
 * release. Use gcl::allToAll instead.
 */
poplar::Tensor
allToAllPersonalizedExchange(poplar::Graph &graph, const poplar::Tensor &input,
                             poplar::program::Sequence &sequence,
                             const poplar::DebugContext &debugContext = {},
                             const poplar::OptionFlags &options = {});

} // End namespace popops

#endif // popops_Collectives_hpp
