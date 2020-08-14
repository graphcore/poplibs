// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnectedOnTile_hpp
#define popsparse_FullyConnectedOnTile_hpp

#include <poplar/Graph.hpp>

#include <popsparse/SparseTensor.hpp>

#include "FullyConnectedVector.hpp"

#include <array>
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
    const poplar::Tensor &partials,
    const std::array<std::size_t, 2> &blockDimensions,
    const std::string &debugPrefix);

// Describes the desired ordering in memory of activations given to onTileImpl
// dependent on the method. For the time being this is the only information
// needed to lay out operands optimally for the operation.
//
// Assumes internal shape of activations
std::vector<unsigned> getOnTileActsOrdering(const OnTileMethod &method);

// Describes the desired ordering in memory of partials given to onTileImpl
// dependent on the method. For the time being this is the only information
// needed to lay out operands optimally for the operation.
std::vector<unsigned> getOnTilePartialsOrdering(const OnTileMethod &method);

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedOnTile_hpp
