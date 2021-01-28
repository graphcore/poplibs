// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TriangularSolveTest
#include <poplibs_support/TestDevice.hpp>

#include <iostream>

#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/TriangularSolve.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplin;
using namespace poplibs_support;
namespace bu = boost::unit_test;
namespace bud = boost::unit_test::data;

namespace {

void writeTensor(const poplar::Target &target, poplar::Engine &engine,
                 poplar::Type type, StringRef handle,
                 const std::vector<float> &values) {
  if (type == HALF) {
    std::vector<char> buf(values.size() * target.getTypeSize(HALF));
    copyFloatToDeviceHalf(target, values.data(), buf.data(), values.size());
    engine.writeTensor(handle, buf.data(), buf.data() + buf.size());
  } else if (type == FLOAT) {
    engine.writeTensor(handle, values.data(), values.data() + values.size());
  } else {
    throw std::runtime_error("invalid type");
  }
}

void readTensor(const poplar::Target &target, poplar::Engine &engine,
                poplar::Type type, StringRef handle,
                std::vector<float> &values) {
  if (type == HALF) {
    std::vector<char> buf(values.size() * target.getTypeSize(HALF));
    engine.readTensor(handle, buf.data(), buf.data() + buf.size());
    copyDeviceHalfToFloat(target, buf.data(), values.data(), values.size());
  } else if (type == FLOAT) {
    engine.readTensor(handle, values.data(), values.data() + values.size());
  } else {
    throw std::runtime_error("invalid type");
  }
}

template <typename T, std::size_t Tiles = 4>
void deviceTriangularSolve(poplar::Type type, const std::vector<T> &a,
                           std::vector<std::size_t> a_shape,
                           const std::vector<T> &b,
                           std::vector<std::size_t> b_shape, bool left_side,
                           bool lower, bool unit_diagonal,
                           std::size_t block_size) {
  auto device = createTestDevice(TEST_TARGET, 1, Tiles);
  auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  auto matmuls = getTriangularSolveMatMulPrePlanParameters(
      type, type, a_shape, b_shape, left_side, lower, block_size, {});

  matmul::PlanningCache cache;
  std::set<MatMulPlanParams> params;
  BOOST_TEST(cache.size() == 0);
  for (auto &param : matmuls) {
    params.emplace(&target, param.first, &param.second);
  }
  preplanMatMuls(params, cache);
  BOOST_TEST(cache.size() == matmuls.size());

  Tensor tA = graph.addVariable(type, a_shape, "A");
  Tensor tB = graph.addVariable(type, b_shape, "B");

  BOOST_REQUIRE_EQUAL(tA.numElements(), a.size());
  BOOST_REQUIRE_EQUAL(tB.numElements(), b.size());

  auto aRank = tA.rank();
  auto bRank = tB.rank();

  BOOST_REQUIRE_EQUAL(aRank, bRank);
  BOOST_REQUIRE_GE(aRank, 2);

  BOOST_REQUIRE_EQUAL(tA.dim(aRank - 1), tA.dim(aRank - 2));
  BOOST_REQUIRE_EQUAL(left_side ? tB.dim(bRank - 2) : tB.dim(bRank - 1),
                      tA.dim(aRank - 1));

  poputil::mapTensorLinearly(graph, tA);
  poputil::mapTensorLinearly(graph, tB);

  graph.createHostWrite("a", tA);
  graph.createHostWrite("b", tB);

  Sequence seq;
  Tensor tX = triangularSolve(graph, tA, tB, left_side, lower, unit_diagonal,
                              block_size, seq, "triangular-solve", {}, &cache);

  BOOST_TEST(cache.size() == matmuls.size()); // All matmuls were preplanned.
  BOOST_TEST(tX.shape() == tB.shape(), boost::test_tools::per_element());
  tA = poplin::triangularMask(graph, tA, lower, unit_diagonal, seq,
                              "triangular-mask-verify");

  tA = tA.rank() >= 3 ? tA.flatten(0, tA.rank() - 2) : tA.expand({0});
  tX = tX.rank() >= 3 ? tX.flatten(0, tX.rank() - 2) : tX.expand({0});

  Tensor tR = left_side
                  ? poplin::matMulGrouped(graph, tA, tX, seq, tA.elementType(),
                                          "triangular-solve-verify")
                  : poplin::matMulGrouped(graph, tX, tA, seq, tA.elementType(),
                                          "triangular-solve-verify");

  tR = tR.reshape(b_shape);
  graph.createHostRead("r", tR);

  std::vector<T> result(b.size());
  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    writeTensor(target, eng, type, "a", a);
    writeTensor(target, eng, type, "b", b);
    eng.run();

    readTensor(target, eng, type, "r", result);
  });

  // boost test tools limitation: you can't use both tolerance and per_element
  // decorators
  for (std::size_t i = 0; i < b.size(); ++i) {
    BOOST_REQUIRE_CLOSE(b[i], result[i],
                        type == HALF ? 0.3 : 0.0001); // percentage!
  }
}

template <typename T, typename E>
std::vector<T> generateIota(std::vector<E> shape) {
  auto n =
      std::accumulate(shape.begin(), shape.end(), E(1), std::multiplies<E>());
  std::vector<T> data(n);
  int x = 0;
  for (auto &v : data)
    v = 1 + (x++ % 8) / 8.0f; // use small numbers for the test
  return data;
}

template <typename T, std::size_t Tiles = 4>
void deviceTriangularSolveIota(poplar::Type type,
                               std::vector<std::size_t> a_shape,
                               std::vector<std::size_t> b_shape, bool left_side,
                               bool lower, bool unit_diagonal,
                               std::size_t block_size) {
  auto a = generateIota<T>(a_shape);
  auto b = generateIota<T>(b_shape);
  return deviceTriangularSolve<T, Tiles>(type, a, a_shape, b, b_shape,
                                         left_side, lower, unit_diagonal,
                                         block_size);
}

} // namespace

BOOST_DATA_TEST_CASE(TriangularSolveCase,
                     bud::xrange(2) * bud::xrange(2) * bud::xrange(2) *
                         bud::xrange(2) * bud::xrange(1, 8, 6) * bud::xrange(2),
                     half_type, left_side, lower, unit_diagonal, k,
                     block_solver) {
  static constexpr std::size_t n = 5;
  std::size_t block_size = std::max<std::size_t>(n, k);
  if (block_solver) {
    block_size = (block_size + 1) / 2;
  }
  using Shape = std::vector<std::size_t>;
  auto bShape = left_side ? Shape{1, 2, n, std::size_t(k)}
                          : Shape{1, 2, std::size_t(k), n};
  deviceTriangularSolveIota<float>(half_type ? HALF : FLOAT, {1, 2, n, n},
                                   bShape, left_side, lower, unit_diagonal,
                                   block_size);
}
