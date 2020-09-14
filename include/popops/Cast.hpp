// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Casts between tensor types.
 *
 */

#ifndef popops_Cast_hpp
#define popops_Cast_hpp

#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <vector>

namespace popops {

/** Cast elements of the specified \p src tensor to \p dstType, returning the
 * result as a new tensor.
 *
 * Note: If `dstType == src.elementType()`, then the operation is a copy.
 *
 * \param graph         The graph that the operation will be added to.
 * \param src           Source tensor to cast.
 * \param dstType       Type of the destination tensor.
 * \param prog          Program to add the cast operation to.
 * \param debugPrefix   Name of the operation, for debugging.
 * \return              The resultant cast tensor.
 */
poplar::Tensor cast(poplar::Graph &graph, const poplar::Tensor &src,
                    const poplar::Type &dstType,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

/** Create a program to copy tensor casting between types (for example,
 * half->float).
 *
 * Precondition: `src.shape() == dst.shape()`
 *
 * Note: If `dst.elementType() == src.elementType()`, then the operation is
 * just a copy.
 *
 * \param graph         The graph that the operation will be added to.
 * \param src           Source tensor.
 * \param dst           Destination tensor.
 * \param debugPrefix   Name of the operation, for debugging.
 * \return              The program to perform this operation.
 */
poplar::program::Program cast(poplar::Graph &graph, poplar::Tensor src,
                              poplar::Tensor dst,
                              const std::string &debugPrefix = "");

/** Create vertices to copy element wise from the \p src tensor to the \p dst
 * tensor casting between types (for example, half->float).
 * The vertices are added to the specified compute set.
 *
 * Precondition: `src.shape() == dst.shape()`
 *
 * \param graph     The graph that the operation will be added to.
 * \param src       Source tensor.
 * \param dst       Destination tensor.
 * \param cs        Compute set to add the vertices to.
 */
void cast(poplar::Graph &graph, poplar::Tensor src, poplar::Tensor dst,
          poplar::ComputeSet cs);

/** Create vertices to cast elements of the specified \p src tensor to
 * \p dstType, returning the result as a new tensor. The vertices are added to
 * the specified compute set.
 *
 * \param graph         The graph that the operation will be added to.
 * \param src           Source tensor.
 * \param dstType       Destination type.
 * \param cs            Compute set to add the vertices to.
 * \param debugPrefix   Name of the operation, for debugging.
 * \return              Resultant destination tensor.
 */
poplar::Tensor cast(poplar::Graph &graph, poplar::Tensor src,
                    const poplar::Type &dstType, poplar::ComputeSet cs,
                    const std::string &debugPrefix = "");

/** Helper function which checks the relative error in the tensor \p input
 * when casting it to type \p outputType. The result is a single element bool
 * tensor which is set to true if the error is less than \p tolerance.
 *
 * Preconditions:
 *  - `input.elementType() == FLOAT`
 *  - `outputType == HALF`
 *  - `input.numElements() == 1`
 *
 * \param graph         The graph that the operation will be added to.
 * \param input         Input tensor.
 * \param outputType    Output type after the cast operation.
 * \param tolerance     Allowed tolerance in error from cast operation.
 * \param prog          Program to add the check onto.
 * \param debugPrefix   Name of the operation, for debugging.
 * \return              Boolean tensor indicating that the error is less
 *                      than \p tolerance.
 * \throw poputil::poplibs_error If either \p input or \p outputType
 * are not either half or float.
 */
poplar::Tensor checkAccuracyWhenCast(poplar::Graph &graph,
                                     const poplar::Tensor &input,
                                     poplar::Type outputType, double tolerance,
                                     poplar::program::Sequence &prog,
                                     const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_Cast_hpp
