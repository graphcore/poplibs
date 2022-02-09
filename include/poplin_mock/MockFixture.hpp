// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poplin_Mock_Fixture_hpp
#define poplin_Mock_Fixture_hpp

#include <poplin_mock/Mock.hpp>

namespace poplin_mock {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPoplinFixture {
public:
  MockPoplinFixture() {
    mockPoplin_ = static_cast<poplin_mock::MockPoplin *>(&mockPoplin);
  }

  ~MockPoplinFixture() { mockPoplin_ = nullptr; }

protected:
  Mock<MockPoplin> mockPoplin;
};

} // namespace poplin_mock

#endif // poplin_Mock_Fixture_hpp
