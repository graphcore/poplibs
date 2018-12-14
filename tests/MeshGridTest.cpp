#define BOOST_TEST_MODULE MeshGridTest

#include <boost/test/unit_test.hpp>
#include <poplin/MeshGrid.hpp>
#include "TestDevice.hpp"

#include <vector>

struct TestCase {
  poplar::Type type;
  float left;
  float right;
  size_t count;
  std::string name;
  std::vector<float> values;
};

BOOST_AUTO_TEST_CASE(LinSpace) {
  auto device = createTestDevice(TEST_TARGET);
  poplar::Graph g(device.getTarget());

  // Define some linspace arguments with expected result values:
  std::vector<TestCase> testCases = {
    {poplar::FLOAT, 11.f, 42.f, 2, "two", {11.f, 42.f}},
    {poplar::FLOAT, -1.f, 1.f, 3, "small", {-1.f, 0.f, 1.f}},
    {poplar::FLOAT, 10.f, -10.f, 5, "backwards", {10.f, 5.f, 0.f, -5.f, -10.f}}
  };

  poplar::program::Sequence prog;

  for (auto c : testCases) {
    auto var1 = poplin::linspace(g, c.type, c.left, c.right, c.count);
    auto var2 = g.clone(c.type, var1);
    g.setTileMapping(var2, 0);
    prog.add(poplar::program::Copy(var1, var2));
    g.createHostRead(c.name, var2);
  }

  poplar::Engine e(g, prog);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.run();

    for (auto c : testCases) {
      std::vector<float> result(c.values.size(), 0.f);
      e.readTensor(c.name, result.data());
      for (auto i = 0u; i < result.size(); ++i) {
        BOOST_CHECK_EQUAL(result.at(i), c.values.at(i));
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(MeshGrid) {
  auto device = createTestDevice(TEST_TARGET);
  poplar::Graph g(device.getTarget());

  auto xCoords = poplin::linspace(g, poplar::FLOAT, -1.f, 1.f, 3);
  auto yCoords = poplin::linspace(g, poplar::FLOAT, -2.f, 2.f, 2);
  auto grids = poplin::meshgrid2d(g, xCoords, yCoords);

  auto gridXOut = g.clone(poplar::FLOAT, grids.at(0));
  auto gridYOut = g.clone(poplar::FLOAT, grids.at(1));
  g.setTileMapping(gridXOut, 0);
  g.setTileMapping(gridYOut, 0);

  const auto rowsOut = 2u;
  const auto colsOut = 3u;
  BOOST_CHECK_EQUAL(gridXOut.shape()[0], rowsOut);
  BOOST_CHECK_EQUAL(gridXOut.shape()[1], colsOut);
  BOOST_CHECK_EQUAL(gridYOut.shape()[0], rowsOut);
  BOOST_CHECK_EQUAL(gridYOut.shape()[1], colsOut);

  poplar::program::Sequence prog = {
    poplar::program::Copy(grids.at(0), gridXOut),
    poplar::program::Copy(grids.at(1), gridYOut)
  };

  g.createHostRead("xs", gridXOut);
  g.createHostRead("ys", gridYOut);

  poplar::Engine e(g, prog);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.run();

    // In Poplar, matrices will come back row major so these are
    // the expected flat results:
    const std::vector<float> correctXs =
      {-1.f, 0.f, 1.f,
       -1.f, 0.f, 1.f};
    const std::vector<float> correctYs =
      {-2.f, -2.f, -2.f,
        2.f,  2.f,  2.f};

    std::vector<float> resultX(2*3);
    e.readTensor("xs", resultX.data());
    for (auto i = 0u; i < resultX.size(); ++i) {
      BOOST_CHECK_EQUAL(resultX.at(i), correctXs.at(i));
    }

    std::vector<float> resultY(2*3);
    e.readTensor("ys", resultY.data());
    for (auto i = 0u; i < resultY.size(); ++i) {
      BOOST_CHECK_EQUAL(resultY.at(i), correctYs.at(i));
    }
  });
}
