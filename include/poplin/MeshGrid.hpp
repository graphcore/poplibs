// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_MeshGrid_hpp
#define poplin_MeshGrid_hpp

#include <poplar/Graph.hpp>

namespace poplin {

/** Create a constant variable that contains values equally spaced in the
 *  specified closed range [left, right].
 * \param graph Graph to which the variable is added.
 * \param left The first value in the range.
 * \param right The last value in the range.
 * \param type Data type of variable to create. Must be FLOAT or HALF.
 * \return Constant Tensor of rank 1 (vector) containing the linspace values.
 */
poplar::Tensor linspace(poplar::Graph &graph, const poplar::Type &type,
                        float left, float right, size_t count,
                        const std::string &debugPrefix = "");

/** Create a coordinate grid for each axis by broadcasting the input tensors.
 * This 2D specialisation only supports two inputs that must be of rank 1
 * (vectors) and hence the output coordinate grids are always two matrices
 * (so two outputs of rank two).
 *
 * TODO: T12862 Implement for general case of N inputs -> N outputs of rank N.
 *
 * \param graph Graph to which the variables are added.
 * \param x Co-ordinates for the x-axis
 * \param y Co-ordinates for the y-axis
 * \return A list of (two) tensors that form co-ordinate grids for each input
 *         axis. These output tensors will be views of the inputs (reshaped and
 *         broadcast)
 */
std::vector<poplar::Tensor> meshgrid2d(poplar::Graph &graph, poplar::Tensor x,
                                       poplar::Tensor y);

} // end namespace poplin

#endif // poplin_MeshGrid_hpp
