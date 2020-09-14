// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Perform logical AND of tensor elements.
 *
 */

#ifndef popops_AllTrue_hpp
#define popops_AllTrue_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/**
 * Given a boolean tensor, compute the logical AND of all its elements.
 * A new variable is created to store the result.
 * \param graph         The Poplar graph.
 * \param A             The boolean tensor.
 * \param prog          The program sequence to add this operation to.
 * \param debugPrefix   A debug name for the operation,
 * \returns             A variable that holds the result of the operation.
 * \throw poputil::poplibs_error If the elements of \p A are not booleans.
 */
poplar::Tensor allTrue(poplar::Graph &graph, poplar::Tensor A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_AllTrue_hpp
