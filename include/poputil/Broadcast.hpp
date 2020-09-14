// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/** \file Broadcast.hpp
 *
 * Functions to provide numpy-like tensor matching and broadcasting.
 *
 */

#ifndef poputil_Broadcast_hpp
#define poputil_Broadcast_hpp
#include <poplar/Tensor.hpp>

namespace poputil {

/** Match dimensions of two tensors using numpy-style expansion rules.
 *
 *  Insert singleton dimensions into either of the two tensors so that their
 *  ranks match, following numpy-style expansion rules. The tensor with the
 *  lower rank has singleton dimensions inserted as the outermost dimensions.
 *
 *  \param a First tensor to match.
 *  \param b Second tensor to match.
 */
void expandToMatchRanks(poplar::Tensor &a, poplar::Tensor &b);

/** Match dimensions of a tensor to a shape using numpy-style
 *  broadcast rules:
 *
 *  1) If the rank of the tensor is less than the required shape then
 *     expand to the left by adding dimensions of size 1 to match the rank
 *     required.
 *
 *  2) For each dimension, the size of the dimension in the tensor must be the
 *     same as the required shape or must be 1. In the case where it is of size
 *     1, the tensor is broadcast in that dimension to match the shape. If
 *     neither of these conditions hold then an exception is thrown.
 *
 *  \param a The tensor to broadcast to match the shape.
 *           This will be updated in place with broadcast dimensions.
 *  \param shape
 *           The shape to match.
 *  \throw poputil::poplibs_error If \p a cannot be broadcast to match \p shape.
 */
void broadcastToMatch(poplar::Tensor &a, const std::vector<std::size_t> &shape);

/** Match dimensions of two tensors using numpy-style
 *  broadcast rules:
 *
 *  1) If the rank of one tensor is less than the other then extend
 *     the dimensions to the left with dimensions of size 1 to match the rank
 *     required.
 *
 *  2) For each dimension, the size of each dimension in both
 *     tensors must be the same or one of them must have size 1.
 *     In the case where one is of size 1, the tensor is broadcast in
 *     that dimension to match the other. If neither of these
 *     conditions hold then an exception is thrown.
 *
 *  \param a First tensor to match. This will be updated in place with
 *           broadcast dimensions.
 *  \param b Second tensor to match. This will be updated in place with
 *           broadcast dimensions.
 * \throw poputil::poplibs_error If \p a cannot be broadcast to match a
 *  dimension.
 */
void broadcastToMatch(poplar::Tensor &a, poplar::Tensor &b);

/** Match dimensions of three tensors using numpy-style
 *  broadcast rules:
 *
 *  1) If the rank of one tensor is less than the other then extend
 *     the dimensions to the left with dimensions of size 1 to match the rank
 *     required.
 *
 *  2) For each dimension, the size of each dimension in both
 *     tensors must be the same or one of them must have size 1.
 *     In the case where one is of size 1, the tensor is broadcast in
 *     that dimension to match the other. If neither of these
 *     conditions hold then an exception is thrown.
 *
 *  \param a First tensor to match. This will be updated in place with
 *           broadcast dimensions.
 *  \param b Second tensor to match. This will be updated in place with
 *           broadcast dimensions.
 *  \param c Third tensor to match. This will be updated in place with
 *           broadcast dimensions.
 * \throw poputil::poplibs_error If \p a cannot be broadcast to match a
 *  dimension.
 */
void broadcastToMatch(poplar::Tensor &a, poplar::Tensor &b, poplar::Tensor &c);

/** Test if the given tensors can be broadcast to match one another
 *  using the rules for broadcastToMatch().
 *
 *  \param a First tensor to match.
 *  \param b Second tensor to match.
 *
 *  \return True if the two tensors may be broadcast to match one another
 *          and false if they cannot be matched with the broadcastToMatch()
 *          rules.
 */
bool canBroadcastToMatch(const poplar::Tensor &a, const poplar::Tensor &b);

} // namespace poputil

#endif // poputil_Broadcast_hpp
