// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

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
 * i.e. given a 1-dimensional \p indices tensor with length N and a
 * 2-dimensional \p encoded tensor with shape N * x \p encoded is a tensor with
 * a single element equal to 1, and all others equal 0. The single hot element
 * in each row is given by the indices in \p indices.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the encoding to.
 *  \param encoded      Tensor to encode output to.
 *  \param indices      1-dimensional tensor containing indices to encode
 *                      as one-hot vectors. A codepoint
 *                      MASKED_LABEL_CODE is reserved to indicate
 *                      that the encoding is not done for that index.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 *  \param debugPrefix  Optional debug prefix for programs/variables
 *                      used to perform the encoding.
 */
void encodeOneHot(poplar::Graph &graph, const poplar::Tensor &indices,
                  const poplar::Tensor &encoded,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

/** Encode a given set of indices as a set of one-hot vectors
 * per-index with a hot element at that index.
 * i.e. given a 1-dimensional \p indices tensor with length N and a
 * 2-dimensional \p encoded tensor with shape N * x \p encoded is a tensor with
 * a single element equal to \p on, and all others equal to \p off as given by
 * the user. The single hot element in each row is given by the indices in
 * \p indices.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the encoding to.
 *  \param encoded      Tensor to encode output to.
 *  \param indices      1-dimensional tensor containing indices to encode
 *                      as one-hot vectors.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 *  \param debugPrefix  Optional debug prefix for programs/variables
 *                      used to perform the encoding.
 *  \param on           Value which represents the "On" state in the one hot
 *                      encoded output.
 *  \param off          Value which represents the "Off" state.
 */
void encodeOneHot(poplar::Graph &graph, const poplar::Tensor &indices,
                  const poplar::Tensor &encoded,
                  poplar::program::Sequence &prog, const poplar::Tensor &on,
                  const poplar::Tensor &off,
                  const std::string &debugPrefix = "");

/** Returns a right-open range of integers
 *         [startInteger, startInteger + length)
 * where, length is the number of elements of mapped 1-D output tensor \p t.
 * Output tensor can be of type INT or UNSIGNED_INT.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the operation.
 *  \param t            1-D tensor to write the encoded output to.
 *                      Tensor must be mapped.
 *  \param startInteger The start integer in the output range.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 *  \param debugPrefix  Optional debug prefix for programs/variables
 *                      used to perform the encoding.
 */
void iota(poplar::Graph &graph, const poplar::Tensor &t, unsigned startInteger,
          poplar::program::Sequence &prog, const std::string &debugPrefix);

/** Returns a right-open range of integers
 *         [startInteger, startInteger + length)
 * where, length is the number of elements of mapped 1-D output tensor \p t.
 * Output tensor can be of type INT or UNSIGNED_INT.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the operation.
 *  \param t            1-D tensor to write the encoded output to.
 *                      Tensor must be mapped.
 *  \param startInteger The start integer in the output range.
 *  \param prog         Sequence which the programs that perform the
 *                      encoding are added to.
 *  \param debugPrefix  Optional debug prefix for programs/variables
 *                      used to perform the encoding.
 */
void iota(poplar::Graph &graph, const poplar::Tensor &t, int startInteger,
          poplar::program::Sequence &prog, const std::string &debugPrefix);

} // end namespace popops

#endif // popops_Encoding_hpp
