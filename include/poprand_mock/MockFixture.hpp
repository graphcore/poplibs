// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poprand_Mock_Fixture_hpp
#define poprand_Mock_Fixture_hpp

#include <poprand_mock/Mock.hpp>

namespace poprand_mock {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPoprandFixture {
public:
  MockPoprandFixture() {
    mockPoprand_ = static_cast<poprand_mock::MockPoprand *>(&mockPoprand);
  }

  ~MockPoprandFixture() { mockPoprand_ = nullptr; }

protected:
  Mock<MockPoprand> mockPoprand;
};

} // namespace poprand_mock

#endif // poprand_Mock_Fixture_hpp
