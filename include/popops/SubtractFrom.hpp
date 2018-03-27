// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_SubtractFrom_hpp
#define popops_SubtractFrom_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Subtract elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations A += -k * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The tensor of elements to subtract (must be of the
 *                     same shape as \A).
 * \param k            The scalar to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     subtract will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void subtractFrom(poplar::Graph &graph,
           poplar::Tensor A, poplar::Tensor B,
           float k, poplar::program::Sequence &prog,
           const std::string &debugPrefix = "");


/** Subtract the elements of one tensor to another tensor.
 *
 *  Performs the calculations A -= B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The tensor of elements to subtract (must be of the
 *                     same shape as \A).
 * \param prog         A sequence program to which the code performing the
 *                     subtract will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void subtractFrom(poplar::Graph &graph,
           poplar::Tensor A, poplar::Tensor B,
           poplar::program::Sequence &prog,
           const std::string &debugPrefix = "");

}

#endif // popops_SubtractFrom_hpp
