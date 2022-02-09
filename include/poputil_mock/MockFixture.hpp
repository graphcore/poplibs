// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poputil_Mock_Fixture_hpp
#define poputil_Mock_Fixture_hpp

#include <poputil_mock/Mock.hpp>

namespace poputil_mock {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPoputilFixture {
public:
  MockPoputilFixture() {
    mockPoputil_ = static_cast<poputil_mock::MockPoputil *>(&mockPoputil);
  }

  ~MockPoputilFixture() { mockPoputil_ = nullptr; }

protected:
  Mock<MockPoputil> mockPoputil;
};

} // namespace poputil_mock

#endif // poputil_Mock_Fixture_hpp
