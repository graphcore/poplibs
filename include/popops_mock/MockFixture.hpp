// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popops_Mock_Fixture_hpp
#define popops_Mock_Fixture_hpp

#include <gmock/gmock.h>
#include <popops_mock/Mock.hpp>

namespace popops_mock {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPopopsFixture {
public:
  MockPopopsFixture() {
    mockPopops_ = static_cast<popops_mock::MockPopops *>(&mockPopops);
  }

  ~MockPopopsFixture() { mockPopops_ = nullptr; }

protected:
  Mock<MockPopops> mockPopops;
};

} // namespace popops_mock

#endif // popops_Mock_Fixture_hpp
