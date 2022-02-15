// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popfloat_Mock_hpp
#define popfloat_Mock_hpp

#include <gmock/gmock.h>
#include <popfloat/experimental/codelets.hpp>

namespace popfloat_mock {

class MockPopfloat {
public:
  // experimental/codelets.hpp

  MOCK_METHOD(void, experimental_addCodelets, (::poplar::Graph &));
};

extern MockPopfloat *mockPopfloat_;

} // namespace popfloat_mock

#endif // popfloat_Mock_hpp
