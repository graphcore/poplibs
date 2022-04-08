// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popops_mock/Mock.hpp>

namespace popops {

poplar::Tensor createOutputForElementWiseOp(
    poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
    const poplar::Type &outputType, const poplar::DebugContext &debugContext) {
  return popops_mock::mockPopops_->createOutputForElementWiseOp(
      graph, inputs, outputType, debugContext);
}

} // end namespace popops
