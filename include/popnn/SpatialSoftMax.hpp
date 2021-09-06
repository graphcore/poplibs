// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/** \file
 *  Functions for spatial softmax.
 */

#ifndef poplin_SpatialSoftMax_hpp
#define poplin_SpatialSoftMax_hpp

#include <poplar/Graph.hpp>
#include <popnn/NonLinearity.hpp>

namespace popnn {

/** Implements a spatial softmax specialised for 2D input fields. This computes
 * the expected coordinates (normalised to be in [-1.0, 1.0]) for every 2D
 * field in the input tensor. A (trainable) temperature scalar is added which
 * normalises the softmax across the fields.
 *
 * The output of the spatial softmax (first tensor in the returned pair) is
 * a set of expected x and y coordinates for the maximum activation in each
 * field. This result has shape {F, 2} where F is the number of fields.
 * Y-coordinates run down the first column and X-coordinates down the second
 * column to preserve (row,column) indexing order into the original fields.
 *
 * \param graph          Graph to which variables and vertices will be added.
 * \param prog           Program to which operations will be added.
 * \param fields         The input Tensor. Must have rank 3. Interpretation is
 *                       a set of 2D scalar fields of identical height (H) and
 *                       width (W) given by the two inner dimensions (so shape
 *                       is {F, H, W} where F is the number of fields).
 * \param temperature    Initial value for the softmax scaling/normalisation
 *                       parameter.
 * \param debugContext   Optional debug information.
 * \param disableSoftmax Turns off softmax computation in this function. This is
 *                       useful if you have already computed a softmax over all
 *                       the fields due to other processing or for test/debug.
 *
 * \return A pair of tensors. First is the output of the spatial-softmax, second
 *         is scalar temperature variable.
 **/
std::pair<poplar::Tensor, poplar::Tensor>
spatialSoftMax2D(poplar::Graph &graph, poplar::program::Sequence &prog,
                 const poplar::Tensor &fields, float temperature,
                 bool disableSoftmax = false,
                 const poplar::DebugContext &debugContext = {});

} // end namespace popnn

#endif // poplin_SpatialSoftMax_hpp
