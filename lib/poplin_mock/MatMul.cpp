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
                             PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulGrouped(graph, A, B, prog, outputType,
                                                 debugContext, options, cache);
}

void matMulGroupedWithOutput(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B, poplar::Tensor &out,
                             poplar::program::Sequence &prog,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &options_,
                             PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulGroupedWithOutput(
      graph, A, B, out, prog, debugContext, options_, cache);
}

void matMulWithOutput(poplar::Graph &graph, const poplar::Tensor &A_,
                      const poplar::Tensor &B_, poplar::Tensor &out,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options_,
                      PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulWithOutput(
      graph, A_, B_, out, prog, debugContext, options_, cache);
}

void matMulGroupedReportPlan(std::ostream &out, const poplar::Graph &graph,
                             const poplar::Type &inputType,
                             const poplar::Type &outputType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const poplar::OptionFlags &options,
                             PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulGroupedReportPlan(
      out, graph, inputType, outputType, aShape, bShape, options, cache);
}

void matMulReportPlan(std::ostream &out, const poplar::Graph &graph,
                      const poplar::Type &inputType,
                      const poplar::Type &outputType,
                      const std::vector<std::size_t> &aShape,
                      const std::vector<std::size_t> &bShape,
                      const poplar::OptionFlags &options,
                      PlanningCache *cache) {
  return poplin_mock::mockPoplin_->matMulReportPlan(
      out, graph, inputType, outputType, aShape, bShape, options, cache);
}

poplar::Tensor createMatMulInputLHS(poplar::Graph &graph,
                                    const poplar::Type &inputType,
                                    const poplar::Type &outputType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options,
                                    PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulInputLHS(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

poplar::Tensor createMatMulInputRHS(poplar::Graph &graph,
                                    const poplar::Type &inputType,
                                    const poplar::Type &outputType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options,
                                    PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulInputRHS(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

poplar::Tensor createMatMulOutput(poplar::Graph &graph,
                                  const poplar::Type &inputType,
                                  const poplar::Type &outputType,
                                  const std::vector<std::size_t> &aShape,
                                  const std::vector<std::size_t> &bShape,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags &options,
                                  PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulOutput(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

poplar::Tensor createMatMulGroupedInputLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulGroupedInputLHS(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

poplar::Tensor createMatMulGroupedInputRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulGroupedInputRHS(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

poplar::Tensor createMatMulGroupedOutput(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  return poplin_mock::mockPoplin_->createMatMulGroupedOutput(
      graph, inputType, outputType, aShape, bShape, debugContext, options,
      cache);
}

} // namespace poplin
