// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplin_mock/Mock.hpp>

namespace poplin {

void addCodelets(poplar::Graph &graph) {
  poplin_mock::mockPoplin_->addCodelets(graph);
}

} // namespace poplin
