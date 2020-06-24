// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnectedOnTile_hpp
#define popsparse_FullyConnectedOnTile_hpp

#include <poplar/Graph.hpp>

#include <popsparse/SparseTensor.hpp>

#include "FullyConnectedVector.hpp"

#include <boost/variant.hpp>

namespace popsparse {

namespace fullyconnected {

enum class OnTileMethod;
struct Options;

// Operation to perform sparse fully connected forward pass on a tile.
void onTileImpl(
    poplar::Graph &graph, const poplar::ComputeSet &cs, unsigned tile,
    const fullyconnected::OnTileMethod &method, bool zeroPartials,
    const boost::variant<unsigned, poplar::Tensor> &subGroupIdToProcess,
    const Vector<std::size_t> &shape, const poplar::Tensor &metaInfoBuckets,
    const poplar::Tensor &weights, const poplar::Tensor &acts,
    const poplar::Tensor &partials, const std::string &debugPrefix);

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedOnTile_hpp
