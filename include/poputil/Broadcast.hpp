// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poputil_Broadcast_hpp
#define poputil_Broadcast_hpp
#include <poplar/Tensor.hpp>

namespace poputil {

/** Match dimensions of a tensor to a shape by broadcasting using numpy style
 *  broadcast rules:
 *
 *     1) If the rank of the tensor is expand to
 *        the dimensions to the left with dimensions of size 1 to match
 *        the rank of the required shape.
 *     2) For each dimension, the size of the dimension in the
 *        tensors must be the same as the required shape or
 *        must have size 1.
 *        In the case where it is of size one the tensor is broadcast in
 *        that dimension to match the shape. If neither of these
 *        conditions hold then an exception is thrown.
 *
 *  \param a The tensor to broadcast to match the shape.
 *           This will be updated in place with broadcast dimensions.
 *  \param shape
 *           The shape to match.
 */
void broadcastToMatch(poplar::Tensor &a, const std::vector<std::size_t> &shape);


  /** Match dimensions of two tensors by broadcasting using numpy style
   *  broadcast rules:
   *
   *     1) If the rank of one tensor is less than the other then extend
   *        the dimensions to the left with dimensions of size 1.
   *     2) For each dimension, the size of the dimension in both
   *        tensors must be the same or one of them must have size 1.
   *        In the case where one is of size one the tensor is broadcast in
   *        that dimension to match the other. If neither of these
   *        conditions hold then an exception is thrown.
   *
   *  \param a First tensor to match. This will be updated in place with
   *           broadcast dimensions.
   *  \param b Second tensor to match. This will be updated in place with
   *           broadcast dimensions.
   */
  void broadcastToMatch(poplar::Tensor &a, poplar::Tensor &b);

}

#endif // poputil_Broadcast_hpp
