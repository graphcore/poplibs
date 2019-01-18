#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include "popops/codelets.hpp"
#include "poplibs_test/Util.hpp"

#define BOOST_TEST_MODULE ScaledAdd2D_integral
#include <boost/test/included/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

// test data of lengths 1 to 16
template <typename T>
struct TestData;

template <>
struct TestData<int> {
  const std::vector<std::vector<int>> data{
    {17},
    {-27, -50},
    {-13, 45, 49},
    {38, -17, -7, -49},
    {-32, 33, -14, 9, -38},
    {1, -43, -2, 41, 13, -33},
    {-38, -34, -26, 20, 50, -31, -11},
    {-42, -15, 31, -11, -27, -10, -11, -25},
    {31, -47, 39, -14, 49, -12, 42, -39, -35},
    {19, -12, 4, -44, -5, -50, 30, 1, 34, 13},
    {13, -16, 42, -50, -12, -22, -28, 10, 48, 46, -45},
    {-14, -45, 50, -3, -1, 33, 38, 34, -23, -30, -11, 26},
    {16, 35, -7, -37, 0, -19, 14, -11, -30, 27, 38, -29, -25},
    {0, -34, 39, -27, 3, -18, 39, 41, -27, 12, 47, 48, 50, 42},
    {50, -39, 46, -2, 1, 21, 27, 7, 37, -48, 13, 49, -42, 4, -28},
    {-48, 12, 45, -44, 27, 20, -15, 5, -34, 38, -42, 25, -49, -31, 30, 36},
  };

  const std::vector<std::vector<int>> deltas{
    {-29},
    {-24, 25},
    {28, -22, -36},
    {21, -14, 20, -37},
    {19, -38, 22, -29, 2},
    {50, -12, -10, 8, -17, -22},
    {-47, 19, 44, 12, 15, -37, 49},
    {8, -23, 31, 13, 10, 23, -19, 3},
    {-12, -42, 47, -50, 11, -30, 8, 16, 27},
    {44, 19, -48, -35, 39, 24, -49, -36, -15, -27},
    {16, -37, -43, 32, 29, 16, -50, 37, -10, 3, 12},
    {7, 0, 3, 13, -11, 2, -6, -6, 27, -13, -36, 22},
    {50, 44, 8, -1, -37, -37, -39, -13, -1, -45, 12, -49, -40},
    {40, 15, 39, 44, 29, 18, 39, 0, -3, -32, -40, 13, 18, 40},
    {19, 10, -8, 11, -47, 19, 31, -16, 1, -9, -47, 50, 2, 48, 35},
    {-10, 10, 46, -15, -10, 17, 41, -2, -27, 42, 0, 37, -14, 39, 3, 45},
  };

};

template <>
struct TestData<unsigned> {
  const std::vector<std::vector<unsigned>> data{
    {7},
    {46, 9},
    {2, 1, 31},
    {33, 3, 13, 48},
    {50, 29, 28, 19, 45},
    {22, 48, 3, 9, 15, 32},
    {20, 22, 5, 49, 32, 5, 26},
    {11, 21, 28, 5, 13, 4, 41, 11},
    {34, 27, 12, 49, 46, 48, 32, 38, 15},
    {14, 37, 47, 49, 43, 42, 14, 43, 11, 18},
    {40, 48, 29, 39, 3, 14, 9, 46, 5, 48, 24},
    {31, 47, 6, 37, 44, 31, 25, 13, 25, 49, 2, 17},
    {13, 20, 42, 1, 48, 32, 19, 50, 20, 32, 23, 42, 18},
    {2, 41, 22, 35, 15, 30, 0, 28, 6, 20, 22, 13, 22, 18},
    {48, 18, 34, 7, 26, 47, 48, 37, 5, 39, 35, 9, 20, 9, 30},
    {24, 16, 3, 46, 19, 40, 32, 4, 26, 7, 2, 30, 38, 23, 27, 43},
  };

  const std::vector<std::vector<unsigned>> deltas{
    {5},
    {24, 23},
    {41, 26, 30},
    {36, 4, 12, 34},
    {16, 27, 47, 6, 10},
    {11, 36, 6, 13, 30, 40},
    {36, 6, 42, 39, 24, 22, 37},
    {12, 48, 7, 18, 23, 24, 30, 45},
    {50, 33, 18, 35, 23, 10, 45, 3, 11},
    {6, 21, 2, 20, 15, 0, 14, 41, 40, 9},
    {49, 2, 27, 11, 1, 28, 36, 26, 32, 48, 5},
    {18, 14, 34, 21, 36, 2, 17, 31, 32, 20, 10, 31},
    {43, 0, 27, 40, 13, 17, 7, 30, 26, 21, 27, 32, 3},
    {32, 49, 22, 35, 18, 11, 43, 19, 14, 28, 5, 8, 2, 34},
    {9, 9, 9, 29, 6, 30, 10, 33, 44, 38, 28, 45, 26, 28, 29},
    {42, 44, 22, 45, 1, 18, 9, 40, 39, 4, 8, 34, 34, 50, 43, 49},
  };

};
const int k = 9;

template <typename T>
void testScaledAdd2D(const char *vertex, const Type &type,
                          const bool &constantFactor, const bool &doSubtract) {
  const TestData<T> testData;
  const auto &data = testData.data;
  const auto &deltas = testData.deltas;

   // Generate the expected result
  std::vector<std::vector<T>> expected(data.size());
  for(unsigned i = 0; i < data.size(); i++) {
    for(unsigned j = 0; j < data[i].size(); j++) {
      if(doSubtract)
        expected[i].push_back(data[i][j] - deltas[i][j] * k);
      else
        expected[i].push_back(data[i][j] + deltas[i][j] * k);
    }
  }

  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto cs = graph.addComputeSet("cs");
  auto v = graph.addVertex(cs, vertex);
  graph.setTileMapping(v, 0);
  graph.setFieldSize(v["data"], data.size());
  graph.setFieldSize(v["deltas"], deltas.size());

  if(constantFactor) {
    graph.setInitialValue(v["K"], k);
  }
  else {
    auto factorTensor = graph.addVariable(type, {});
    graph.setTileMapping(factorTensor, 0);
    graph.connect(v["factor"], factorTensor);
    graph.setInitialValue(factorTensor, k);
  }
  // create tensors for each of the input rows.
  assert(data.size() == deltas.size());
  for (unsigned i = 0; i < data.size(); ++i) {
    const auto size = data[i].size();
    assert(size == deltas[i].size());

    auto datumTensor = graph.addVariable(type, {size});
    graph.setTileMapping(datumTensor, 0);
    graph.connect(v["data"][i], datumTensor);
    graph.createHostRead("datum" + std::to_string(i), datumTensor);
    graph.createHostWrite("datum" + std::to_string(i), datumTensor);

    auto deltaTensor = graph.addVariable(type, {size});
    graph.setTileMapping(deltaTensor, 0);
    graph.connect(v["deltas"][i], deltaTensor);
    graph.createHostWrite("delta" + std::to_string(i), deltaTensor);
  }

  Execute prog(cs);
  Engine e(graph, prog);

  device.bind([&](const Device &d) {
    e.load(d);

    // write tensors to the device.
    for (unsigned i = 0; i < data.size(); ++i) {
      const auto &datum = data[i];
      const auto &delta = deltas[i];

      e.writeTensor("datum" + std::to_string(i), datum.data());
      e.writeTensor("delta" + std::to_string(i), delta.data());
    }

    e.run();

    // check results against the expected output.
    for (unsigned i = 0; i < data.size(); ++i) {
      const auto &datum = data[i];
      const auto size = datum.size();

      std::vector<T> actual(size);
      e.readTensor("datum" + std::to_string(i), actual.data());

      for (unsigned j = 0; j < i; ++j) {
        BOOST_CHECK(actual[j] == expected[i][j]);
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(ScaledAdd2DIntConst) {
  testScaledAdd2D<int>("popops::ScaledAdd2D<int,true>", INT, true, false);
}

BOOST_AUTO_TEST_CASE(ScaledAdd2DUnsignedIntConst) {
  testScaledAdd2D<unsigned>("popops::ScaledAdd2D<unsigned int,true>",
                                                    UNSIGNED_INT, true, false);
}
BOOST_AUTO_TEST_CASE(ScaledAdd2DIntTensor) {
  testScaledAdd2D<int>("popops::ScaledAdd2D<int,false>", INT, false, false);
}

BOOST_AUTO_TEST_CASE(ScaledAdd2DUnsignedIntTensor) {
  testScaledAdd2D<unsigned>("popops::ScaledAdd2D<unsigned int,false>",
                                                    UNSIGNED_INT, false, false);
}
BOOST_AUTO_TEST_CASE(ScaledSubtract2DIntTensor) {
  testScaledAdd2D<int>("popops::ScaledSubtract2D<int>", INT, false, true);
}

BOOST_AUTO_TEST_CASE(ScaledSubtract2DUnsignedIntTensor) {
  testScaledAdd2D<unsigned>("popops::ScaledSubtract2D<unsigned int>",
                                                    UNSIGNED_INT, false, true);
}
