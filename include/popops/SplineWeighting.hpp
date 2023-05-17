// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
/** \file
 *
 * SplineWeighting operation.
 *
 */

#ifndef popops_SplineWeighting_hpp
#define popops_SplineWeighting_hpp

#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"

namespace popops {

/** Calculate spline weighting.
 *
 * That is, given a two-dimensional \p input tensor with shape
 * numEdges * numInputChannels, three-dimensional \p weight tensor and
 * two-dimensional \p basis and \p weightIndex tensors, calculate output
 * features weighted by a continuous B-spline kernel functions. \p basis and
 * \p weightIndex tensors are outputs from SplineBasis operation with shape
 * numEdges * numSplines. \p weight tensor shape must be
 * numEdges * numInputChannels * numOutputChannels and \p output tensor shape is
 * numEdges * numOutputChannels.
 *
 *  \param graph        The graph to add any required vertices.
 *  \param input        Input features tensor.
 *  \param weight       Weight tensor.
 *  \param basis        B-spline basis functions coefficients tensor.
 *  \param weightIndex  Tensor with weight indices for spline coefficients.
 *  \param prog         Sequence to which the programs that perform the
 *                      calculations are added.
 *  \param debugContext  Optional debug information.
 *
 *  \returns An output tensor with features weighted by a continuous
 *           B-spline kernel functions
 *
 *  \throw poputil::poplibs_error If \p input \p basis or \p weightIndex are not
 *         two-dimensional.
 *  \throw poputil::poplibs_error If \p weight tensor is not three-dimensional.
 *  \throw poputil::poplibs_error If elements of \p input are not float or half
 *         type.
 *  \throw poputil::poplibs_error If either elements of \p weight \p basis or
 *         \p output not have the same type as elements of \p input tensor.
 *  \throw poputil::poplibs_error If elements of \p weightIndex tensor are not
 *         integer type.
 *  \throw poputil::poplibs_error If either size of dimension 0 of \p basis or
 *         dimension 0 of \p weightIndex or dimension 0 of \p output do not
 *         match size of dimension 0 \p input tensor.
 *  \throw poputil::poplibs_error If size of dimension 1 of \p basis does not
 *         match the size of dimension 1 of \p weightIndex tensor.
 *  \throw poputil::poplibs_error If size of dimension 1 of \p weight does not
 *         match the size of dimension 1 of \p input tensor.
 *  \throw poputil::poplibs_error If size of dimension 2 of \p weight does not
 *         match the size of dimension 1 of \p output tensor.
 */
poplar::Tensor splineWeighting(poplar::Graph &graph,
                               const poplar::Tensor &input,
                               const poplar::Tensor &weight,
                               const poplar::Tensor &basis,
                               const poplar::Tensor &weightIndex,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {});

} // end namespace popops

#endif // popops_SplineWeighting_hpp
