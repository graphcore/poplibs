// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Log of softmax functions.
 */

#ifndef popnn_LogSoftmax_hpp
#define popnn_LogSoftmax_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {

/** Update tensor \p t by computing log of softmax in-place.
 *
 * \param graph             The graph to add the operation to.
 * \param t                 The tensor to apply the log of softmax to.
 * \param prog              The sequence to add the operation to.
 * \param debugContext      Optional debug information.
 */
void logSoftmaxInPlace(poplar::Graph &graph, poplar::Tensor t,
                       poplar::program::Sequence &prog,
                       const poplar::DebugContext &debugContext = {});

/** Compute the log of the softmax to tensor \p t and return the result.
 *
 * \param graph             The graph to add the operation to.
 * \param t                 The tensor to apply the non-linearity to.
 * \param prog              The sequence to add the operation to.
 * \param debugContext      Optional debug information.
 *
 * \returns A new tensor containing the contents of \p t with the given
 *          log of the softmax applied.
 */
poplar::Tensor logSoftmax(poplar::Graph &graph, poplar::Tensor t,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {});

} // end namespace popnn

#endif // popnn_LogSoftmax_hpp
