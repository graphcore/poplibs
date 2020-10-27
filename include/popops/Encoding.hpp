// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Encoding and generating ranges of integers.
 *
 */

#ifndef popops_Encoding_hpp
#define popops_Encoding_hpp

#include "EncodingConstants.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Type.hpp"
#include <string>

namespace popops {

/** Encode a given set of indices as a set of one-hot vectors
 * per-index with a hot element at that index.
 * That is, given a one-dimensional \p indices tensor with length N and a
 * two-dimensional \p encoded tensor with shape N * x, \p encoded is a tensor
 * with a single element equal to 1, and all others equal 0. The single hot
 * element in each row is given by the indices in \p indices.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the encoding to.
 *  \param encoded      Tensor to encode output to.
 *  \param indices      1-dimensional tensor containing indices to encode
 *                      as one-hot vectors. A codepoint
 *                      \c MASKED_LABEL_CODE is reserved to indicate
 *                      that the encoding is not done for that index.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 * \param debugContext  Optional debug information.
 *  \throw poputil::poplibs_error If \p encoded is not two dimensional.
 *  \throw poputil::poplibs_error If \p indices and \p encoded do not
 *         have the same number of rows.
 *  \throw poputil::poplibs_error If elements of \p indices are not an
 *         integer type.
 */
void encodeOneHot(poplar::Graph &graph, const poplar::Tensor &indices,
                  const poplar::Tensor &encoded,
                  poplar::program::Sequence &prog,
                  const poplar::DebugContext &debugContext = {});

/** Encode a given set of indices as a set of one-hot vectors
 * per-index with a hot element at that index.
 * That is, given a one-dimensional \p indices tensor with length N and a
 * two-dimensional \p encoded tensor with shape N * x \p encoded is a tensor
 * with a single element equal to \p on, and all others equal to \p off as
 * given by the user. The single hot element in each row is given by the
 * indices in \p indices.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the encoding to.
 *  \param encoded      Tensor to encode output to.
 *  \param indices      1-dimensional tensor containing indices to encode
 *                      as one-hot vectors.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 * \param debugContext  Optional debug information.
 *  \param on           Value which represents the "On" state in the one hot
 *                      encoded output.
 *  \param off          Value which represents the "Off" state.
 *  \throw poputil::poplibs_error If \p encoded is not two dimensional.
 *  \throw poputil::poplibs_error If \p indices and \p encoded do not
 *         have the same number of rows.
 *  \throw poputil::poplibs_error If elements of \p indices are not an
 *         integer type.
 */
void encodeOneHot(poplar::Graph &graph, const poplar::Tensor &indices,
                  const poplar::Tensor &encoded,
                  poplar::program::Sequence &prog, const poplar::Tensor &on,
                  const poplar::Tensor &off,
                  const poplar::DebugContext &debugContext = {});

/** Fill a tensor with a right-open range of unsigned integers:
 *         [startInteger, startInteger + length),
 * where length is the number of elements in the mapped 1-D output tensor \p t.
 * The output tensor \p t must be of type UNSIGNED_INT.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the operation.
 *  \param t            1-D tensor to write the encoded output to.
 *                      The tensor must be mapped.
 *  \param startInteger The start value in the output range.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 * \param debugContext  Optional debug information.
 * \throw poputil::poplibs_error If the rank of \p t is greater than 1.
 * \throw poputil::poplibs_error If the type of \p t is not UNSIGNED_INT.
 */
void iota(poplar::Graph &graph, const poplar::Tensor &t, unsigned startInteger,
          poplar::program::Sequence &prog,
          const poplar::DebugContext &debugContext = {});

/** Fill a tensor with a right-open range of signed integers:
 *         [startInteger, startInteger + length),
 * where length is the number of elements in the mapped 1-D output tensor \p t.
 * The output tensor \p t must be of type INT.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the operation.
 *  \param t            1-D tensor to write the encoded output to.
 *                      The tensor must be mapped.
 *  \param startInteger The start value in the output range.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 * \param debugContext  Optional debug information.
 * \throw poputil::poplibs_error If the rank of \p t is greater than 1.
 * \throw poputil::poplibs_error If the type of \p t is not INT.
 */
void iota(poplar::Graph &graph, const poplar::Tensor &t, int startInteger,
          poplar::program::Sequence &prog,
          const poplar::DebugContext &debugContext = {});

} // end namespace popops

#endif // popops_Encoding_hpp
