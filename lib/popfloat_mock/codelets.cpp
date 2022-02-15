// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popfloat_mock/Mock.hpp>

namespace popfloat_mock {
MockPopfloat *mockPopfloat_ = nullptr;
} // namespace popfloat_mock

namespace popfloat::experimental {

void addCodelets(poplar::Graph &graph) {
  popfloat_mock::mockPopfloat_->experimental_addCodelets(graph);
}

} // namespace popfloat::experimental
