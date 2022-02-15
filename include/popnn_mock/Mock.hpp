// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popnn_Mock_hpp
#define popnn_Mock_hpp

#include <gmock/gmock.h>
#include <popnn/codelets.hpp>

namespace popnn_mock {

class MockPopnn {
public:
  // codelets.hpp

  MOCK_METHOD(void, addCodelets, (::poplar::Graph &));
};

extern MockPopnn *mockPopnn_;

} // namespace popnn_mock

#endif // popnn_Mock_hpp
