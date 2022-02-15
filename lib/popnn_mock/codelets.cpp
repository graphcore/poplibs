// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popnn_mock/Mock.hpp>

namespace popnn_mock {
MockPopnn *mockPopnn_ = nullptr;
} // namespace popnn_mock

namespace popnn {

void addCodelets(poplar::Graph &graph) {
  popnn_mock::mockPopnn_->addCodelets(graph);
}

} // namespace popnn
