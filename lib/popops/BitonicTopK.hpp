// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef _popops_BitonicTopK_hpp_
#define _popops_BitonicTopK_hpp_

#include <optional>
#include <vector>

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace popops {
namespace bitonic {

poplar::Tensor createTopKInputImpl(poplar::Graph &graph,
                                   const poplar::Type &type,
                                   const std::vector<std::size_t> &shape,
                                   const poplar::DebugNameAndId &dnai = {});

/// Implementation of topK using bitonic sort based method.
/// Returns a pair of top k values in t, and matching permutation
/// of \p other if it was given.
std::pair<poplar::Tensor, poplar::Tensor>
topKImpl(poplar::Graph &graph, poplar::program::Sequence &prog,
         const poplar::Tensor &t, const std::optional<poplar::Tensor> &other,
         const unsigned k, const bool largest, const bool sorted,
         const bool ascendingOrder, const bool otherIsSecondaryKey,
         const bool sortOtherInReverseOrder,
         const poplar::DebugNameAndId &dnai = {});

} // end namespace bitonic
} // end namespace popops

#endif // _popops_BitonicTopK_hpp_
