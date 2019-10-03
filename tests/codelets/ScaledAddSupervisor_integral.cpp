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

  static std::array<int, N> expected;
};

constexpr const std::array<int, N> TestData<int>::data;
constexpr const std::array<int, N> TestData<int>::deltas;
std::array<int, N> TestData<int>::expected;

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

  static  std::array<unsigned, N> expected;
};

constexpr const std::array<unsigned, N> TestData<unsigned>::data;
constexpr const std::array<unsigned, N> TestData<unsigned>::deltas;
std::array<unsigned, N> TestData<unsigned>::expected;

template <typename T>
void testScaledAddSupervisor(const char *vertex, const Type &type,
                          const bool &constantFactor, const bool &doSubtract) {
  const auto &data = TestData<T>::data;
  const auto &deltas = TestData<T>::deltas;
  auto &expected = TestData<T>::expected;
  const int k = 9;

   // Generate the expected result
  for(unsigned i = 0; i < data.size(); i++) {
    if(doSubtract)
      expected[i] = data[i] - deltas[i] * k;
    else
      expected[i] = data[i] + deltas[i] * k;
  }

  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence prog;

  // create a ComputeSet for each test case of size = 1...N
  for (unsigned i = 1; i <= N; ++i) {
    auto cs = graph.addComputeSet("cs" + std::to_string(i));
    auto v = graph.addVertex(cs, vertex);
    graph.setTileMapping(v, 0);

    auto dataTensor = graph.addVariable(type, {i});
    graph.setTileMapping(dataTensor, 0);
    graph.connect(v["A"], dataTensor);

    graph.createHostWrite("data" + std::to_string(i), dataTensor);
    graph.createHostRead("data" + std::to_string(i), dataTensor);

    auto deltasTensor = graph.addVariable(type, {i});
    graph.setTileMapping(deltasTensor, 0);
    graph.connect(v["B"], deltasTensor);
    graph.createHostWrite("deltas" + std::to_string(i), deltasTensor);

    graph.setInitialValue(v["size"], i);
    if(constantFactor) {
      graph.setInitialValue(v["scaleB"], k);
    }
    else {
      auto factorTensor = graph.addVariable(type, {});
      graph.setTileMapping(factorTensor, 0);
      graph.connect(v["scaleB"], factorTensor.reshape({1}));
      graph.setInitialValue(factorTensor, 9);
    }
    prog.add(Execute(cs));
  }

  Engine e(graph, prog);

  const char *pdata = reinterpret_cast<const char *>(data.data());
  const char *pdeltas = reinterpret_cast<const char *>(deltas.data());

  device.bind([&](const Device &d) {
    e.load(d);

    for (unsigned i = 1; i <= N; ++i) {

      e.writeTensor("data" + std::to_string(i), pdata,
                    pdata + i * target.getTypeSize(type));
      e.writeTensor("deltas" + std::to_string(i), pdeltas,
                    pdeltas + i * target.getTypeSize(type));
    }

    e.run();

    std::array<T, N> actual;
    char *pactual = reinterpret_cast<char *>(actual.data());
    for (unsigned i = 1; i <= N; ++i) {
      e.readTensor("data" + std::to_string(i), pactual, pactual +
                   i * target.getTypeSize(type));
      for (unsigned j = 0; j < i; ++j) {
        BOOST_CHECK(actual[j] == expected[j]);
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorIntConstant) {
  testScaledAddSupervisor<int>(
        "popops::ScaledAddSupervisor<int,int,int,true,false>", INT, true,
        false);
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorUnsignedIntConstant) {
  testScaledAddSupervisor<unsigned>(
        "popops::ScaledAddSupervisor<unsigned int,unsigned int,"
        "unsigned int,true,false>",
        UNSIGNED_INT, true, false);
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorIntTensor) {
  testScaledAddSupervisor<int>(
        "popops::ScaledAddSupervisor<int,int,int,false,false>", INT, false,
        false);
}

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorUnsignedIntTensor) {
  testScaledAddSupervisor<unsigned>(
        "popops::ScaledAddSupervisor<unsigned int,unsigned int,"
        "unsigned int,false,false>",
        UNSIGNED_INT, false, false);
}

BOOST_AUTO_TEST_CASE(ScaledSubtractSupervisorIntTensor) {
  testScaledAddSupervisor<int>(
        "popops::ScaledSubtractSupervisor<int,int,false>", INT, false, true);
}

BOOST_AUTO_TEST_CASE(ScaledSubtractSupervisorUnsignedIntTensor) {
  testScaledAddSupervisor<unsigned>(
        "popops::ScaledSubtractSupervisor<unsigned int,unsigned int,false>",
                                                    UNSIGNED_INT, false, true);
}
