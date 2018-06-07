// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Encoding_hpp
#define popops_Encoding_hpp

#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Type.hpp"
#include <string>

namespace popops {

/** Encode a given set of indices as a set of one-hot vectors
 *  per-index with a hot element at that index.
 *  i.e. given a 1-dimensional `indices` tensor with length N returns a
 *  2-dimensional tensor with shape N * `length` where each row of the
 *  tensor has a single element equal to 1, and all others equal 0.
 *  The single hot element in each row is given by the indices in `indices`.
 *
 *  \param graph        The graph to add the tensor and any vertices
 *                      needed for the encoding to.
 *  \param encodedType  Type of elements in the returned tensor.
 *  \param indices      1-dimensional tensor containing indices to encode
 *                      as one-hot vectors.
 *  \param length       The length of the one-hot vectors. Must be at least
 *                      large enough for the max index in indices (max + 1).
 *  \param prog         Sequence to add programs to to perform the encoding.
 *  \param debugPrefix  Optional debug prefix for programs/variables
 *                      used to perform the encoding.
 */
poplar::Tensor
encodeOneHot(poplar::Graph &graph,
             const poplar::Type &encodedType,
             const poplar::Tensor &indices,
             unsigned length,
             poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

} // end namespace popops

#endif  // popops_Encoding_hpp
