// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popops_SelectScalarInRows_hpp
#define popops_SelectScalarInRows_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popops {

/**
 * For each row in the 2D tensor params, select a single scalar value.
 * Aggregate the resulting scalars into a 1D tensor.
 *
 * \param params  A 2D tensor, element-type must be either float or half
 * \param indices A 1D tensor, element-type must be unsigned integer.
 *
 * The size of the \p indices tensor must be equal to the size of dimension-0 of
 * \p params. The ith element of \p indices represents an index in the ith row
 * of the params tensor.
 *
 * \return A 1D tensor containing in the ith position the scalar
 * `params[indices[i]]`. \n
 * If ith element of the \p indices tensor is less than 0 or
 * greater than the width of \p params then a NaN is stored into the ith element
 * of the output. If the ith element of the \p indices tensor is equal to
 * \p MASKED_LABEL_CODE then zero is stored into the ith element of the output.
 *
 */
poplar::Tensor selectScalarFromRows(poplar::Graph &graph,
                                    const poplar::Tensor &params,
                                    const poplar::Tensor &indices,
                                    poplar::program::Sequence &prog,
                                    const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_SelectScalarInRows_hpp
