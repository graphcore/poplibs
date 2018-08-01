#ifndef popconv_ChannelOps_hpp
#define popconv_ChannelOps_hpp

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>

#include <string>

namespace popconv {

// Add a vector to every channel in an activation tensor. The shape of
// `actsUngrouped` is [N][C]... where N is the batch size, C is the
// number of channels and ... are the spatial dimensions.
//
// `addend` is a vector of length C that is effectively broadcast in every
// other dimension and added to `actsUngrouped` (multiplied by `scale`).
void addToChannel(poplar::Graph &graph,
                  const poplar::Tensor &actsUngrouped,
                  const poplar::Tensor &addend,
                  float scale,
                  poplar::program::Sequence &prog,
                  const std::string debugPrefix);

// Similar to addToChannel, but performs a multiply instead of add, and isn't
// in-place.
poplar::Tensor channelMul(poplar::Graph &graph,
                          const poplar::Tensor &actsUngrouped,
                          const poplar::Tensor &scale,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix);

}

#endif // popconv_ChannelOps_hpp
