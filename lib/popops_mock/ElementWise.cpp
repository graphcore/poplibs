// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popops_mock/Mock.hpp>

namespace popops_mock {
MockPopops *mockPopops_ = nullptr;
} // namespace popops_mock

namespace popops {

void mapInPlace(poplar::Graph &graph, const expr::Expr &expr,
                const std::vector<poplar::Tensor> &ts,
                poplar::program::Sequence &prog,
                const poplar::DebugContext &debugContext,
                const poplar::OptionFlags &options) {
  popops_mock::mockPopops_->mapInPlace(graph, expr, ts, prog, debugContext,
                                       options);
}

poplar::Tensor map(poplar::Graph &graph, const expr::Expr &expr,
                   const std::vector<poplar::Tensor> &ts,
                   poplar::program::Sequence &prog,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options) {
  return popops_mock::mockPopops_->map(graph, expr, ts, prog, debugContext,
                                       options);
}

} // namespace popops
