// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poputil_mock/Mock.hpp>

namespace poputil_mock {
MockPoputil *mockPoputil_ = nullptr;
} // namespace poputil_mock

namespace poputil {

void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned minElementsPerTile, unsigned grainSize) {
  poputil_mock::mockPoputil_->mapTensorLinearly(graph, t, minElementsPerTile,
                                                grainSize);
}

void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t) {
  poputil_mock::mockPoputil_->mapTensorLinearly(graph, t);
}

unsigned getTileImbalance(const poplar::Graph &graph, const poplar::Tensor &t,
                          unsigned minElementsPerTile, unsigned grainSize) {
  return poputil_mock::mockPoputil_->getTileImbalance(
      graph, t, minElementsPerTile, grainSize);
}

std::pair<poplar::Tensor, unsigned>
cloneAndExpandAliasing(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned offset,
                       const poplar::DebugContext &debugContext) {
  return poputil_mock::mockPoputil_->cloneAndExpandAliasing(graph, t, offset,
                                                            debugContext);
}

} // namespace poputil
