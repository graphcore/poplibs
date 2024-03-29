// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poplin_Mock_hpp
#define poplin_Mock_hpp

#include <gmock/gmock.h>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>

namespace poplin_mock {

class MockPoplin {
public:
  // MatMul.hpp

  MOCK_METHOD(::poplar::Tensor, matMulGrouped,
              (::poplar::Graph &, const ::poplar::Tensor &,
               const ::poplar::Tensor &, ::poplar::program::Sequence &,
               const ::poplar::Type &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(void, matMulGroupedWithOutput,
              (::poplar::Graph &, const ::poplar::Tensor &,
               const ::poplar::Tensor &, ::poplar::Tensor &,
               ::poplar::program::Sequence &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(void, matMulWithOutput,
              (::poplar::Graph &, const ::poplar::Tensor &,
               const ::poplar::Tensor &, ::poplar::Tensor &,
               ::poplar::program::Sequence &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(void, matMulGroupedReportPlan,
              (std::ostream &, const ::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::OptionFlags &,
               ::poplin::PlanningCache *));

  MOCK_METHOD(void, matMulReportPlan,
              (std::ostream &, const ::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::OptionFlags &,
               ::poplin::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulInputLHS,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulInputRHS,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulOutput,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulGroupedInputLHS,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulGroupedInputRHS,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulGroupedOutput,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &, ::poplin::PlanningCache *));

  // codelets.hpp

  MOCK_METHOD(void, addCodelets, (::poplar::Graph &));
};

extern MockPoplin *mockPoplin_;

} // namespace poplin_mock

#endif // poplin_Mock_hpp
