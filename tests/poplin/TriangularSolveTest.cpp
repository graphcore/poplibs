// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplibs_support/TestDevice.hpp>

#include <iostream>

#include <boost/program_options.hpp>

#include <poplibs_test/Util.hpp>

#include <poplar/Engine.hpp>
#include <poplin/ConvPreplan.hpp>
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
bool deviceTriangularSolve(poplar::Type type, const std::vector<T> &a,
                           std::vector<std::size_t> a_shape,
                           const std::vector<T> &b,
                           std::vector<std::size_t> b_shape, bool left_side,
                           bool lower, bool unit_diagonal,
                           std::size_t block_size,
                           const DeviceType &deviceType) {
  auto device = createTestDevice(deviceType, 1, Tiles);
  auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  poplar::OptionFlags options;
  options.set("blockSize", std::to_string(block_size));

  auto matmuls = getTriangularSolveMatMulPrePlanParameters(
      type, type, a_shape, b_shape, left_side, lower, options);

  PlanningCache cache;
  std::set<MatMulPlanParams> params;
  assert(cache.size() == 0);
  for (auto &param : matmuls) {
    params.emplace(&target, param.first, &param.second);
  }
  preplan({}, params, cache);
  assert(cache.size() == matmuls.size());

  Tensor tA = graph.addVariable(type, a_shape, "A");
  Tensor tB = graph.addVariable(type, b_shape, "B");

  assert(tA.numElements() == a.size());
  assert(tB.numElements() == b.size());

  assert(tA.rank() == tB.rank());
  assert(tA.rank() >= 2);

  assert(tA.dim(tA.rank() - 1) == tA.dim(tA.rank() - 2));
  assert(left_side ? tB.dim(tB.rank() - 2)
                   : tB.dim(tB.rank() - 1) == tA.dim(tA.rank() - 1));

  poputil::mapTensorLinearly(graph, tA);
  poputil::mapTensorLinearly(graph, tB);

  graph.createHostWrite("a", tA);
  graph.createHostWrite("b", tB);

  Sequence seq;
  Tensor tX = triangularSolve(graph, tA, tB, left_side, lower, unit_diagonal,
                              seq, "triangular-solve", options, &cache);

  assert(cache.size() == matmuls.size()); // All matmuls were preplanned.
  assert(tX.shape() == tB.shape());
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

  const double absTolerance = type == HALF ? 0.3 : 0.0001;
  const double relTolerance = type == HALF ? 0.01 : 0.001;
  bool matchesModel = poplibs_test::util::checkIsClose(
      "result", result.data(), {b.size()}, b.data(), b.size(), relTolerance,
      absTolerance);
  return matchesModel;
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
bool deviceTriangularSolveIota(poplar::Type type,
                               std::vector<std::size_t> a_shape,
                               std::vector<std::size_t> b_shape, bool left_side,
                               bool lower, bool unit_diagonal,
                               std::size_t block_size,
                               const DeviceType &deviceType) {
  auto a = generateIota<T>(a_shape);
  auto b = generateIota<T>(b_shape);
  return deviceTriangularSolve<T, Tiles>(type, a, a_shape, b, b_shape,
                                         left_side, lower, unit_diagonal,
                                         block_size, deviceType);
}

} // namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = TEST_TARGET;
  poplar::Type type = poplar::HALF;
  bool left_side = false;
  bool lower = false;
  bool unit_diagonal = false;
  bool block_solver = false;
  unsigned k = 1;
  unsigned n = 32;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type", po::value<DeviceType>(&deviceType)->required(), "Device type")
    ("type", po::value<poplar::Type>(&type)->required(), "Data type")
    ("left-side", po::value<bool>(&left_side)->required())
    ("lower", po::value<bool>(&lower)->required())
    ("unit-diagonal", po::value<bool>(&unit_diagonal)->required())
    ("block-solver", po::value<bool>(&block_solver)->required())
    ("k", po::value<unsigned>(&k)->required())
    ("n", po::value<unsigned>(&n)->required())
    ;
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  std::size_t block_size = block_solver ? 4 : n;
  using Shape = std::vector<std::size_t>;
  auto bShape = left_side ? Shape{1, 2, n, std::size_t(k)}
                          : Shape{1, 2, std::size_t(k), n};
  bool success = deviceTriangularSolveIota<float>(
      type, {1, 2, n, n}, bShape, left_side, lower, unit_diagonal, block_size,
      deviceType);
  if (!success) {
    std::cerr << "Test failed\n";
    return 1;
  }

  return 0;
}
