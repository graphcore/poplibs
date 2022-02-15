// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poprand_mock/Mock.hpp>

namespace poprand_mock {
MockPoprand *mockPoprand_ = nullptr;
} // namespace poprand_mock

namespace poprand {

void addCodelets(poplar::Graph &graph) {
  poprand_mock::mockPoprand_->addCodelets(graph);
}

} // namespace poprand
