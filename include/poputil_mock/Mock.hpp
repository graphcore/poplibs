// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poputil_Mock_hpp
#define poputil_Mock_hpp

#include <gmock/gmock.h>
#include <poputil/TileMapping.hpp>

namespace poputil_mock {

class MockPoputil {
public:
  MOCK_METHOD(void, mapTensorLinearly,
              (::poplar::Graph &, const ::poplar::Tensor &, unsigned,
               unsigned));

  MOCK_METHOD(void, mapTensorLinearly,
              (::poplar::Graph &, const ::poplar::Tensor &));

  MOCK_METHOD(unsigned, getTileImbalance,
              (const ::poplar::Graph &, const ::poplar::Tensor &, unsigned,
               unsigned));

  MOCK_METHOD((std::pair<::poplar::Tensor, unsigned>), cloneAndExpandAliasing,
              (poplar::Graph &, const poplar::Tensor &, unsigned,
               const poplar::DebugContext &));
};

extern MockPoputil *mockPoputil_;

} // namespace poputil_mock

#endif // poputil_Mock_hpp
