// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef poplin_ChannelOps_hpp
#define poplin_ChannelOps_hpp

#include <boost/variant.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include <string>

namespace poplin {

/** Add a vector to every channel in an activation tensor. The shape of
 * `actsUngrouped` is [N][C]... where N is the batch size, C is the
 * number of channels and ... are the spatial dimensions.
 *
 * \param addend A vector of length C that is effectively broadcast in every
 *               other dimension and added to `actsUngrouped` (multiplied by
 *               `scale`).
 */
void addToChannel(
    poplar::Graph &graph, const poplar::Tensor &actsUngrouped,
    const poplar::Tensor &addend, float scale,
    boost::variant<poplar::ComputeSet &, poplar::program::Sequence &> csOrProg,
    const std::string debugPrefix);

/// Similar to addToChannel, but performs a multiply instead of add, and isn't
/// in-place.
poplar::Tensor channelMul(
    poplar::Graph &graph, const poplar::Tensor &actsUngrouped,
    const poplar::Tensor &scale,
    boost::variant<poplar::ComputeSet &, poplar::program::Sequence &> csOrProg,
    const std::string &debugPrefix);

} // namespace poplin

#endif // poplin_ChannelOps_hpp
