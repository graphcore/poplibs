// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poprand_Mock_hpp
#define poprand_Mock_hpp

#include <gmock/gmock.h>
#include <poprand/codelets.hpp>

namespace poprand_mock {

class MockPoprand {
public:
  // codelets.hpp

  MOCK_METHOD(void, addCodelets, (::poplar::Graph &));
};

extern MockPoprand *mockPoprand_;

} // namespace poprand_mock

#endif // poprand_Mock_hpp
