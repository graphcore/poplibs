// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef poputil_Broadcast_hpp
#define poputil_Broadcast_hpp
#include <poplar/Tensor.hpp>

namespace poputil {

/** Insert singleton dimensions into either of two tensors such that their
 *  ranks match following numpy style expansion rules. The tensor with the
 *  lower rank has singleton dimensions inserted as outer-most dimensions.
 *
 *  \param a First tensor to match.
 *  \param b Second tensor to match.
 */
void expandToMatchRanks(poplar::Tensor &a, poplar::Tensor &b);

/** Match dimensions of a tensor to a shape by broadcasting using numpy style
 *  broadcast rules:
 *
 *     1) If the rank of the tensor is expand to
 *        the dimensions to the left with dimensions of size 1 to match
 *        the rank of the required shape.
 *
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
 *
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

/** Match dimensions of three tensors by broadcasting using numpy style
 *  broadcast rules:
 *
 *     1) If the rank of one tensor is less than the other then extend
 *        the dimensions to the left with dimensions of size 1.
 *
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
 *  \param c Third tensor to match. This will be updated in place with
 *           broadcast dimensions.
 */
void broadcastToMatch(poplar::Tensor &a, poplar::Tensor &b, poplar::Tensor &c);

/** Test if the given tensors can be broadcast to match one another
 *  using the rules for broadcastToMatch.
 *
 *  \param a First tensor to match.
 *  \param b Second tensor to match.
 *
 *  \return True if the two tensors may be broadcast to match one another
 *          and false if they do not match following the broadcastToMatch
 *          broadcast rules.
 */
bool canBroadcastToMatch(const poplar::Tensor &a, const poplar::Tensor &b);

} // namespace poputil

#endif // poputil_Broadcast_hpp
