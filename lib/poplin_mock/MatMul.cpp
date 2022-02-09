// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplin_mock/Mock.hpp>

namespace poplin_mock {
MockPoplin *mockPoplin_ = nullptr;
} // namespace poplin_mock

namespace poplin {

poplar::Tensor matMulGrouped(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const poplar::Type &outputType,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &options,
                             matmul::PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulGrouped(graph, A, B, prog, outputType,
                                                 debugContext, options, cache);
}

void matMulGroupedReportPlan(std::ostream &out, const poplar::Graph &graph,
                             const poplar::Type &inputType,
                             const poplar::Type &outputType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const poplar::OptionFlags &options,
                             matmul::PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulGroupedReportPlan(
      out, graph, inputType, outputType, aShape, bShape, options, cache);
}

poplar::Tensor createMatMulGroupedInputLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, matmul::PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulGroupedInputLHS(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

poplar::Tensor createMatMulGroupedInputRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, matmul::PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulGroupedInputRHS(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

} // namespace poplin
