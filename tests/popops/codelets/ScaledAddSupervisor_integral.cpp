// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ScaledAddSupervisor_integral
#include "poplibs_test/Util.hpp"
#include "popops/codelets.hpp"
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;

#define N 50

template <typename T> struct TestData;

template <> struct TestData<int> {
  constexpr static const std::array<int, N> data = {
      -17, 34,  46,  34,  -41, 6,   -38, 17, -47, 25,  -13, 46,  24,
      26,  -45, 30,  18,  43,  29,  -41, 43, 7,   -26, 33,  -35, -10,
      15,  49,  -42, 42,  41,  -37, -13, 34, 38,  -39, 20,  13,  -17,
      13,  -42, -5,  -14, -36, 32,  43,  36, -4,  49,  19};

  constexpr static const std::array<int, N> deltas = {
      3,   -7, -6,  -13, 6,   -50, 42,  11,  -21, -41, -29, -30, 3,
      -37, 46, 4,   36,  43,  12,  -1,  10,  46,  23,  46,  32,  -24,
      2,   30, 38,  0,   -32, 18,  -45, 41,  -39, -38, 27,  -12, -35,
      33,  12, -43, 45,  8,   32,  -36, -33, 43,  -35, 1};
};

constexpr const std::array<int, N> TestData<int>::data;
constexpr const std::array<int, N> TestData<int>::deltas;

template <> struct TestData<unsigned> {
  constexpr static const std::array<unsigned, N> data = {
      22, 0,  44, 79, 13, 32, 16,  30, 53, 29, 32, 79, 98, 28, 1,  49, 54,
      20, 91, 64, 88, 29, 3,  23,  28, 86, 97, 92, 11, 40, 68, 12, 23, 11,
      94, 82, 10, 69, 91, 48, 100, 53, 48, 50, 95, 26, 28, 13, 44, 45};

  constexpr static const std::array<unsigned, N> deltas = {
      28, 46, 39, 70, 72, 67, 16, 47, 13, 81, 82, 15, 25, 89, 85, 34, 46,
      58, 53, 6,  81, 23, 61, 66, 61, 23, 74, 70, 27, 97, 46, 95, 10, 62,
      54, 51, 92, 80, 47, 20, 86, 67, 51, 54, 14, 26, 16, 34, 22, 92};
};

constexpr const std::array<unsigned, N> TestData<unsigned>::data;
constexpr const std::array<unsigned, N> TestData<unsigned>::deltas;

template <typename T>
void testScaledAddSupervisor(const char *vertex, const Type &type,
                             const bool &constantFactor,
                             const bool &doSubtract) {
  const auto &data = TestData<T>::data;
  const auto &deltas = TestData<T>::deltas;
  const int k = 9;

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto &target = device.getTarget();
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence prog;

  auto dataTensor = graph.addVariable(type, {N});
  poputil::mapTensorLinearly(graph, dataTensor);

  graph.createHostWrite("data", dataTensor);

  auto deltasTensor = graph.addVariable(type, {N});
  poputil::mapTensorLinearly(graph, deltasTensor);
  graph.createHostWrite("deltas", deltasTensor);

  std::vector<Tensor> outs;
  outs.reserve(N);

  // put a test case on each tile.
  auto cs = graph.addComputeSet("cs");
  for (unsigned i = 1; i <= N; ++i) {
    const unsigned tile = (i - 1) % target.getTilesPerIPU();

    auto v = graph.addVertex(cs, vertex);
    graph.setTileMapping(v, tile);

    Interval interval = {0, i};
    auto A = graph.addVariable(type, {N});
    graph.setTileMapping(A, tile);

    prog.add(Copy(dataTensor, A));
    outs.push_back(A);

    graph.connect(v["A"], A.slice(interval));
    graph.connect(v["B"], deltasTensor.slice(interval));

    graph.setInitialValue(v["size"], i);
    if (constantFactor) {
      graph.setInitialValue(v["scaleB"], k);
    } else {
      auto factorTensor = graph.addVariable(type, {});
      graph.setTileMapping(factorTensor, tile);
      graph.connect(v["scaleB"], factorTensor.reshape({1}));
      graph.setInitialValue(factorTensor, 9);
    }
  }
  prog.add(Execute(cs));

  auto out = concat(outs);
  graph.createHostRead("out", out);

  Engine e(graph, prog);

  const char *pdata = reinterpret_cast<const char *>(data.data());
  const char *pdeltas = reinterpret_cast<const char *>(deltas.data());

  // one tensor for each slice {0..N}
  std::vector<char> outBuffer(N * N * target.getTypeSize(type));

  device.bind([&](const Device &d) {
    e.load(d);

    e.writeTensor("data", pdata, pdata + N * target.getTypeSize(type));
    e.writeTensor("deltas", pdeltas, pdeltas + N * target.getTypeSize(type));

    e.run();

    e.readTensor("out", outBuffer.data(), outBuffer.data() + outBuffer.size());
  });

  std::array<T, N> expected;
  std::array<T, N> actual;
  std::copy(&data[0], &data[N], std::begin(expected));

  for (unsigned i = 0; i < N; ++i) {
    const auto start = i * N * target.getTypeSize(type);
    copy(target, type, outBuffer.data() + start, actual.data(), N);

    // Generate the next required result given the length of the test has
    // increased by one. Earlier expected results have already been computed.
    // Later expected results remain equal to the original input value until
    // overwritten.
    expected[i] =
        doSubtract ? data[i] - deltas[i] * k : data[i] + deltas[i] * k;

    BOOST_TEST(actual == expected, boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_SUITE(ScaledAddSupervisorIntConstant)

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorIntConstant) {
  testScaledAddSupervisor<int>(
      "popops::ScaledAddSupervisor<int,int,int,true,false>", INT, true, false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ScaledAddSupervisorUnsignedIntConstant)

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorUnsignedIntConstant) {
  testScaledAddSupervisor<unsigned>(
      "popops::ScaledAddSupervisor<unsigned int,unsigned int,"
      "unsigned int,true,false>",
      UNSIGNED_INT, true, false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ScaledAddSupervisorIntTensor)

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorIntTensor) {
  testScaledAddSupervisor<int>(
      "popops::ScaledAddSupervisor<int,int,int,false,false>", INT, false,
      false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ScaledAddSupervisorUnsignedIntTensor)

BOOST_AUTO_TEST_CASE(ScaledAddSupervisorUnsignedIntTensor) {
  testScaledAddSupervisor<unsigned>(
      "popops::ScaledAddSupervisor<unsigned int,unsigned int,"
      "unsigned int,false,false>",
      UNSIGNED_INT, false, false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ScaledSubtractSupervisorIntTensor)

BOOST_AUTO_TEST_CASE(ScaledSubtractSupervisorIntTensor) {
  testScaledAddSupervisor<int>(
      "popops::ScaledSubtractSupervisor<int,int,int,false>", INT, false, true);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ScaledSubtractSupervisorUnsignedIntTensor)

BOOST_AUTO_TEST_CASE(ScaledSubtractSupervisorUnsignedIntTensor) {
  testScaledAddSupervisor<unsigned>("popops::ScaledSubtractSupervisor<unsigned "
                                    "int,unsigned int,unsigned int,false>",
                                    UNSIGNED_INT, false, true);
}

BOOST_AUTO_TEST_SUITE_END()
