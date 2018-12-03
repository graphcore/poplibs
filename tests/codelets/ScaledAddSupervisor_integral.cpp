#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include "popops/codelets.hpp"
#include "poplibs_test/Util.hpp"

#define BOOST_TEST_MODULE ScaledAddSupervisor_integral
#include <boost/test/included/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

#define N 50

template <typename T>
struct TestData;

template <>
struct TestData<int> {
  constexpr static const std::array<int, N> data = {
    -17, 34, 46, 34, -41, 6, -38, 17,
    -47, 25, -13, 46, 24, 26, -45, 30,
    18, 43, 29, -41, 43, 7, -26, 33,
    -35, -10, 15, 49, -42, 42, 41, -37,
    -13, 34, 38, -39, 20, 13, -17, 13,
    -42, -5, -14, -36, 32, 43, 36, -4,
    49, 19
  };

  constexpr static const std::array<int, N> deltas = {
    3, -7, -6, -13, 6, -50, 42, 11,
    -21, -41, -29, -30, 3, -37, 46,
    4, 36, 43, 12, -1, 10, 46, 23,
    46, 32, -24, 2, 30, 38, 0, -32,
    18, -45, 41, -39, -38, 27, -12, -35,
    33, 12, -43, 45, 8, 32, -36, -33, 43,
    -35, 1
  };

  constexpr static const std::array<int, N> expected = {
    10, -29, -8, -83, 13, -444, 340, 116,
    -236, -344, -274, -224, 51, -307, 369, 66,
    342, 430, 137, -50, 133, 421, 181, 447,
    253, -226, 33, 319, 300, 42, -247, 125,
    -418, 403, -313, -381, 263, -95, -332, 310,
    66, -392, 391, 36, 320, -281, -261, 383,
    -266, 28
  };
};

constexpr const std::array<int, N> TestData<int>::data;
constexpr const std::array<int, N> TestData<int>::deltas;
constexpr const std::array<int, N> TestData<int>::expected;

template <>
struct TestData<unsigned> {
  constexpr static const std::array<unsigned, N> data = {
    22, 0, 44, 79, 13, 32, 16, 30,
    53, 29, 32, 79, 98, 28, 1, 49,
    54, 20, 91, 64, 88, 29, 3, 23,
    28, 86, 97, 92, 11, 40, 68, 12,
    23, 11, 94, 82, 10, 69, 91, 48,
    100, 53, 48, 50, 95, 26, 28, 13,
    44, 45
  };

  constexpr static const std::array<unsigned, N> deltas = {
    28, 46, 39, 70, 72, 67, 16, 47,
    13, 81, 82, 15, 25, 89, 85, 34,
    46, 58, 53, 6, 81, 23, 61, 66,
    61, 23, 74, 70, 27, 97, 46, 95,
    10, 62, 54, 51, 92, 80, 47, 20,
    86, 67, 51, 54, 14, 26, 16, 34,
    22, 92
  };

  constexpr static const std::array<unsigned, N> expected = {
    274, 414, 395, 709, 661, 635, 160, 453,
    170, 758, 770, 214, 323, 829, 766, 355,
    468, 542, 568, 118, 817, 236, 552, 617,
    577, 293, 763, 722, 254, 913, 482, 867,
    113, 569, 580, 541, 838, 789, 514, 228,
    874, 656, 507, 536, 221, 260, 172, 319,
    242, 873
  };
};

constexpr const std::array<unsigned, N> TestData<unsigned>::data;
constexpr const std::array<unsigned, N> TestData<unsigned>::deltas;
constexpr const std::array<unsigned, N> TestData<unsigned>::expected;

template <typename T>
void testScaledAddSupervisor(const char *vertex, const Type &type,
                                                const bool &constantFactor) {
  const auto &data = TestData<T>::data;
  const auto &deltas = TestData<T>::deltas;
  const auto &expected = TestData<T>::expected;

  Device device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popops::addCodelets(graph);

  Sequence prog;

  // create a ComputeSet for each test case of size = 1...N
  for (unsigned i = 1; i <= N; ++i) {
    auto cs = graph.addComputeSet("cs" + std::to_string(i));
    auto v = graph.addVertex(cs, vertex);
    graph.setTileMapping(v, 0);

    auto dataTensor = graph.addVariable(type, {i});
    graph.setTileMapping(dataTensor, 0);
    graph.connect(v["data"], dataTensor);

    graph.createHostWrite("data" + std::to_string(i), dataTensor);
    graph.createHostRead("data" + std::to_string(i), dataTensor);

    auto deltasTensor = graph.addVariable(type, {i});
    graph.setTileMapping(deltasTensor, 0);
    graph.connect(v["deltas"], deltasTensor);
    graph.createHostWrite("deltas" + std::to_string(i), deltasTensor);

    if(constantFactor) {
      graph.setInitialValue(v["K"], 9);
    }
    else {
      auto factorTensor = graph.addVariable(type, {});
      graph.setTileMapping(factorTensor, 0);
      graph.connect(v["factor"], factorTensor);
      graph.setInitialValue(factorTensor, 9);
    }
    prog.add(Execute(cs));
  }

  Engine e(graph, prog);
  e.load(device);

  for (unsigned i = 1; i <= N; ++i) {
    e.writeTensor("data" + std::to_string(i), data.data());
    e.writeTensor("deltas" + std::to_string(i), deltas.data());
  }

  e.run();

  std::array<T, N> actual;
  for (unsigned i = 1; i <= N; ++i) {
    e.readTensor("data" + std::to_string(i), actual.data());
    for (unsigned j = 0; j < i; ++j) {
      BOOST_CHECK(actual[j] == expected[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorHalfConstant) {
  testScaledAddSupervisor<int>(
        "popops::ScaledAddSupervisor<int,int,true>", INT, true);
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorFloatConstant) {
  testScaledAddSupervisor<unsigned>(
        "popops::ScaledAddSupervisor<unsigned int,unsigned int,true>",
                                                        UNSIGNED_INT, true);
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorHalfTensor) {
  testScaledAddSupervisor<int>(
        "popops::ScaledAddSupervisor<int,int,false>", INT, false);
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorFloatTensor) {
  testScaledAddSupervisor<unsigned>(
        "popops::ScaledAddSupervisor<unsigned int,unsigned int,false>",
                                                        UNSIGNED_INT, false);
}
