// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Supporting functions for element-wise operations.
 *
 */

#ifndef _popops_ElementWiseUtil_hpp_
#define _popops_ElementWiseUtil_hpp_

#include <poplar/Graph.hpp>

#include <vector>

namespace popops {

/** Create a tensor for use as the output of an element-wise operation
 *  (operation with no dependency between more than one element of
 *  the output and any given element of any input tensor).
 *
 *  Use the mapping of this tensor to map element-wise operations to tiles
 *  to produce an operation that is computationally balanced across tiles
 *  and which minimises exchange.
 *
 *  All input tensors must have the same shape.
 *
 *  \param graph            A graph to add the tensor to and which the inputs
 *                          belong to.
 *  \param inputs           List of input tensors for the element-wise
 *                          operation.
 *  \param outputType       The element type of the tensor.
 *  \param debugName        Debug name given to the tensor.
 *
 *  \return A tensor with the same shape as the given inputs, with a complete
 *          tile mapping.
 */
poplar::Tensor createOutputForElementWiseOp(
    poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
    const poplar::Type &outputType, const std::string &debugName = "");

} // end namespace popops

#endif // _popops_ElementWiseUtil_hpp_
