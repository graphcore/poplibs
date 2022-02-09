// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poplin_Mock_hpp
#define poplin_Mock_hpp

#include <gmock/gmock.h>
#include <poplin/MatMul.hpp>

namespace poplin_mock {

class MockPoplin {
public:
  MOCK_METHOD(::poplar::Tensor, matMulGrouped,
              (::poplar::Graph &, const ::poplar::Tensor &,
               const ::poplar::Tensor &, ::poplar::program::Sequence &,
               const ::poplar::Type &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &,
               ::poplin::matmul::PlanningCache *));

  MOCK_METHOD(void, matMulGroupedReportPlan,
              (std::ostream &, const ::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::OptionFlags &,
               ::poplin::matmul::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulGroupedInputLHS,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &,
               ::poplin::matmul::PlanningCache *));

  MOCK_METHOD(::poplar::Tensor, createMatMulGroupedInputRHS,
              (::poplar::Graph &, const ::poplar::Type &,
               const ::poplar::Type &, const std::vector<std::size_t> &,
               const std::vector<std::size_t> &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &,
               ::poplin::matmul::PlanningCache *));
};

extern MockPoplin *mockPoplin_;

} // namespace poplin_mock

#endif // poplin_Mock_hpp
