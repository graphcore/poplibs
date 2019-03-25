// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_ScaledAdd_hpp
#define popops_ScaledAdd_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Add the elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations A += scaleB * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledAddTo(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 float scaleB, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Add the elements of one tensor each multiplied by a (scalar) tensor to
 *  another tensor.
 *
 *  Performs the calculations A += scaleB * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledAddTo(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 poplar::Tensor scaleB, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Subtract the elements of one tensor multiplied by a scalar from another
 * tensor.
 *
 *  Performs the calculations A -= scaleB * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor providing the elements to subtract
 *                     (must be of the same shape as \A).
 * \param scaleB       The scalar to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledSubtractFrom(poplar::Graph &graph,
                        poplar::Tensor A, poplar::Tensor B,
                        float scaleB, poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "");

/** Subtract the elements of one tensor each multiplied by a (scalar) tensor
 *  from another tensor.
 *
 *  Performs the calculations A -= scaleB * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor providing the elements to subtract
 *                     (must be of the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledSubtractFrom(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 poplar::Tensor scaleB, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The 2 scaling factors are (scalar) tensors.
 *
 *  Performs the calculations A = scaleA * A + scaleB * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The scalar tensor to multiply elements of A with before
 *                     addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor scaleA,
        poplar::Tensor B, poplar::Tensor scaleB,
        poplar::program::Sequence &prog, const std::string &debugPrefix ="");

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The 2 scaling factors are constants.
 *
 *  Performs the calculations A = scaleA * A + scaleB * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The constant to multiply elements of A with before
 *                     addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The constant to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, float scaleA,
        poplar::Tensor B, float scaleB,
        poplar::program::Sequence &prog, const std::string &debugPrefix ="");


}

#endif // popops_ScaledAdd_hpp
