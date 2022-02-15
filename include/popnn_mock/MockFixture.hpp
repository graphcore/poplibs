// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popnn_Mock_Fixture_hpp
#define popnn_Mock_Fixture_hpp

#include <popnn_mock/Mock.hpp>

namespace popnn_mock {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPopnnFixture {
public:
  MockPopnnFixture() {
    mockPopnn_ = static_cast<popnn_mock::MockPopnn *>(&mockPopnn);
  }

  ~MockPopnnFixture() { mockPopnn_ = nullptr; }

protected:
  Mock<MockPopnn> mockPopnn;
};

} // namespace popnn_mock

#endif // popnn_Mock_Fixture_hpp
