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

void VerifyDotProducts(
    std::vector<std::pair<MatMulParams, poplar::OptionFlags>> &matmuls,
    std::size_t start, std::size_t g, std::size_t an, std::size_t bn) {
  for (std::size_t i = 1; i < an; ++i) {
    auto &params = matmuls[start++].first;
    BOOST_TEST(params.aShape.size() == 3);
    BOOST_TEST(params.aShape[0] == g);
    BOOST_TEST(params.aShape[1] == 1);
    BOOST_TEST(params.aShape[2] == i);

    BOOST_TEST(params.bShape.size() == 3);
    BOOST_TEST(params.bShape[0] == g);
    BOOST_TEST(params.bShape[1] == i);
    BOOST_TEST(params.bShape[2] == bn);
  }
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

BOOST_AUTO_TEST_CASE(TriangularSolvePreplanning30x30BlockSize4) {
  const std::vector<std::size_t> aShape = {8, 30, 30};
  const std::vector<std::size_t> bShape = {8, 30, 3};
  const std::size_t blockSize = 4;
  auto matmuls = getTriangularSolveMatMulPrePlanParameters(
      FLOAT, FLOAT, aShape, bShape, true, true, blockSize, {});
  // Main dimensions of the solver will be padded to be at least pow(2, n) *
  // blockSize. In this case padded tensor size will be 32x32 and n = 3 Divide
  // and conquer algorithm above will use matmul of sizes 16x16, 8x8 and 4x4.
  // Additional matmuls are dot-product in forward/back substitution (an - 1)
  BOOST_TEST(matmuls.size() == 3 + blockSize - 1);
  const auto &matmul1 = matmuls[0].first;
  const auto &matmul2 = matmuls[1].first;
  const auto &matmul3 = matmuls[2].first;

  BOOST_TEST(matmul1.aShape.size() == 3);
  BOOST_TEST(matmul1.aShape[0] == 8);
  BOOST_TEST(matmul1.aShape[1] == 16);
  BOOST_TEST(matmul1.aShape[2] == 16);

  BOOST_TEST(matmul1.bShape.size() == 3);
  BOOST_TEST(matmul1.bShape[0] == 8);
  BOOST_TEST(matmul1.bShape[1] == 16);
  BOOST_TEST(matmul1.bShape[2] == 16);

  BOOST_TEST(matmul2.aShape.size() == 3);
  BOOST_TEST(matmul2.aShape[0] == 8);
  BOOST_TEST(matmul2.aShape[1] == 8);
  BOOST_TEST(matmul2.aShape[2] == 8);

  BOOST_TEST(matmul2.bShape.size() == 3);
  BOOST_TEST(matmul2.bShape[0] == 8);
  BOOST_TEST(matmul2.bShape[1] == 8);
  BOOST_TEST(matmul2.bShape[2] == 8);

  BOOST_TEST(matmul3.aShape.size() == 3);
  BOOST_TEST(matmul3.aShape[0] == 8);
  BOOST_TEST(matmul3.aShape[1] == 4);
  BOOST_TEST(matmul3.aShape[2] == 4);

  BOOST_TEST(matmul3.bShape.size() == 3);
  BOOST_TEST(matmul3.bShape[0] == 8);
  BOOST_TEST(matmul3.bShape[1] == 4);
  BOOST_TEST(matmul3.bShape[2] == 4);

  VerifyDotProducts(matmuls, 3, 8, blockSize, 4);
}

BOOST_AUTO_TEST_CASE(TriangularSolvePreplanning4x4BlockSize4) {
  const std::vector<std::size_t> aShape = {1, 4, 4};
  const std::vector<std::size_t> bShape = {1, 4, 2};
  const std::size_t blockSize = 4;
  auto matmuls = getTriangularSolveMatMulPrePlanParameters(
      FLOAT, FLOAT, aShape, bShape, true, true, blockSize, {});
  // No block solver, only dot products
  BOOST_TEST(matmuls.size() == 3);
  VerifyDotProducts(matmuls, 0, 1, 4, 2);
}
