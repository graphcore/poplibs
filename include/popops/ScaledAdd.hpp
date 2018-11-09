// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_ScaledAdd_hpp
#define popops_ScaledAdd_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Add the elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations A += k * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param k            The scalar to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledAddTo(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 float k, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Add the elements of one tensor each multiplied by a (scalar) tensor to
 *  another tensor.
 *
 *  Performs the calculations A += factor * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param factor       The scalar tensor to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledAddTo(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 poplar::Tensor factor, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Subtract the elements of one tensor multiplied by a scalar from another
 * tensor.
 *
 *  Performs the calculations A -= k * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param k            The scalar to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledSubtractFrom(poplar::Graph &graph,
                        poplar::Tensor A, poplar::Tensor B,
                        float k, poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "");
}


#endif // popops_ScaledAdd_hpp
