// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popfloat_Mock_Fixture_hpp
#define popfloat_Mock_Fixture_hpp

#include <popfloat_mock/Mock.hpp>

namespace popfloat_mock {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPopfloatFixture {
public:
  MockPopfloatFixture() {
    mockPopfloat_ = static_cast<popfloat_mock::MockPopfloat *>(&mockPopfloat);
  }

  ~MockPopfloatFixture() { mockPopfloat_ = nullptr; }

protected:
  Mock<MockPopfloat> mockPopfloat;
};

} // namespace popfloat_mock

#endif // popfloat_Mock_Fixture_hpp
