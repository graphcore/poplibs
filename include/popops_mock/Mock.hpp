// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popops_Mock_hpp
#define popops_Mock_hpp

#include <gmock/gmock.h>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

namespace popops_mock {

class MockPopops {
public:
  // ElementWise.hpp

  MOCK_METHOD(void, mapInPlace,
              (::poplar::Graph &, const ::popops::expr::Expr &,
               const std::vector<::poplar::Tensor> &,
               ::poplar::program::Sequence &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &));

  MOCK_METHOD(poplar::Tensor, map,
              (::poplar::Graph &, const ::popops::expr::Expr &,
               const std::vector<::poplar::Tensor> &,
               ::poplar::program::Sequence &, const ::poplar::DebugContext &,
               const ::poplar::OptionFlags &));

  // codelets.hpp

  MOCK_METHOD(void, addCodelets, (::poplar::Graph &));
};

extern MockPopops *mockPopops_;

} // namespace popops_mock

#endif // popops_Mock_hpp
