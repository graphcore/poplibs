// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
/** \file
 *
 * SplineBasis operation.
 *
 */

#ifndef popops_SplineBasis_hpp
#define popops_SplineBasis_hpp

#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"

namespace popops {

/** Calculate B-spline basis.
 *
 * That is, given a two-dimensional \p pseudo tensor with shape
 * numEdges * numDims and one-dimensional \p kernelSize and \p isOpenSpline
 * tensors with length numDims, calculate \p basis tensor of shape
 * numEdges * numSplines containing B-spline basis functions coefficients for
 * the given \p degree. The \p weightIndex output contains weight index for each
 * spline coefficient.
 *
 *  \param graph        The graph to add any vertices needed for calculating
                         \p basis and \p weightIndex outputs.
 *  \param pseudo       Pseudo coordinates, of shape numEdges * numDims.
 *  \param kernelSize   One-dimensional tensor containing kernel size at each
                        dimension of edge's pseudo coordinates.
 *  \param isOpenSpline One-dimenstional tensor that for each dimension encodes
                        whether open or closed B-spline basis must be used.
 *  \param basis        Two-dimensional output tensor with shape
                        numEdges * numSplines for B-spline basis functions
                        coefficients.
 *  \param weightIndex  Two-dimensional output tensor with shape
                        numEdges * numSplines for weight indices for each spline
                        coefficient.
 *  \param prog         Sequence to which the programs that perform the
 *                      calculations are added.
 *  \param debugContext  Optional debug information.
 *  \throw poputil::poplibs_error If \p pseudo is not two-dimensional.
 *  \throw poputil::poplibs_error If \p kernelSize or \p isOpenSpline  are not
           one-dimensional.
 *  \throw poputil::poplibs_error If \p kernelSize and \p isOpenSpline do not
 *         have the same size as number of columns in \p pseudo tensor.
 *  \throw poputil::poplibs_error If elements of \p pseudo are not float or half
 *         type.
 *  \throw poputil::poplibs_error If elements of \p kernelSize are not integer
 *         type.
 *  \throw poputil::poplibs_error If elements of \p isOpenSpline are not
 *         unsigned char type.
 *  \throw poputil::poplibs_error If degree is neither 1, 2 or 3.
 */
void splineBasis(poplar::Graph &graph, const poplar::Tensor &pseudo,
                 const poplar::Tensor &kernelSize,
                 const poplar::Tensor &isOpenSpline,
                 const poplar::Tensor &basis, const poplar::Tensor &weightIndex,
                 unsigned degree, poplar::program::Sequence &prog,
                 const poplar::DebugContext &debugContext = {});

} // end namespace popops

#endif // popops_SplineBasis_hpp
