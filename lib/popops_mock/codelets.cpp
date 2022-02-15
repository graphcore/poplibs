// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popops_mock/Mock.hpp>

namespace popops {

void addCodelets(poplar::Graph &graph) {
  popops_mock::mockPopops_->addCodelets(graph);
}

} // namespace popops
