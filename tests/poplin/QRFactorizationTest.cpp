// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplibs_support/TestDevice.hpp>

#include <iostream>

#include <boost/program_options.hpp>

#include <poplibs_test/Util.hpp>

#include <poplar/Engine.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <poplin/experimental/QRFactorization.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplin;
using namespace poplin::experimental;
using namespace poplibs_support;

namespace {
struct Errors {
  float ANormErr;
  float QNormErr;
};

std::vector<float> generateInput(const std::vector<std::size_t> shape) {
  const auto nelems =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());
  std::vector<float> data(nelems);
  int x = 0;
  for (auto &v : data)
    v = 1 + (x++ % 8) / 8.0f; // use small numbers for the test
  return data;
}

std::vector<float> generateIdentity(const std::size_t dimension) {
  std::vector<float> data(dimension * dimension);
  for (std::size_t i = 0; i < dimension; i++)
    for (std::size_t j = 0; j < dimension; j++)
      data[i * dimension + j] = float(i == j);
  return data;
}

void initAuxTensors(Graph &graph, Tensor &ARef, Tensor &QRef, Tensor &errA,
                    Tensor &errQ, const std::size_t m, const std::size_t n) {
  ARef = graph.addVariable(FLOAT, {m, n}, "ARef");
  QRef = graph.addVariable(FLOAT, {m, m}, "QRef");
  errA = graph.addVariable(FLOAT, {}, "errA");
  errQ = graph.addVariable(FLOAT, {}, "errQ");

  poputil::mapTensorLinearly(graph, ARef);
  poputil::mapTensorLinearly(graph, QRef);
  poputil::mapTensorLinearly(graph, errA);
  poputil::mapTensorLinearly(graph, errQ);
}

void saveAQValues(Sequence &main, const Tensor &A, const Tensor &Q,
                  const Tensor &ARef, const Tensor &QRef) {
  main.add(Copy(A, ARef));
  main.add(Copy(Q, QRef));
}

void computeErrors(Graph &graph, Sequence &main, Tensor &A, Tensor &Q,
                   Tensor &ARef, Tensor &QRef, Tensor &errA, Tensor &errQ) {

  // AError = ||A - QR|| / ||A||
  matMulWithOutput(graph, Q, A, A, main);
  subWithOutput(graph, ARef, A, A, main);
  Tensor norm = mul(graph, A, A, main);
  norm = reduce(graph, norm, FLOAT, {0, 1}, Operation::ADD, main);
  Tensor normRef = mul(graph, ARef, ARef, main);
  normRef = reduce(graph, normRef, FLOAT, {0, 1}, Operation::ADD, main);
  divWithOutput(graph, norm, normRef, errA, main);

  // QError = ||I - QQ ^ T||
  matMulWithOutput(graph, Q.transpose(), Q, Q, main);
  subWithOutput(graph, QRef, Q, Q, main);
  norm = mul(graph, Q, Q, main);
  reduceWithOutput(graph, norm, errQ, {0, 1}, Operation::ADD, main);
}

Errors testQRFactorization(const DeviceType &deviceType, const std::size_t m,
                           const std::size_t n, const unsigned tiles,
                           const int rowsPerIteration) {
  const Type type = FLOAT;
  auto device = createTestDevice(deviceType, 1, tiles);
  auto &target = device.getTarget();

  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  std::array<Tensor, 2> matrices =
      createQRFactorizationMatrices(graph, type, m, n, {});
  Tensor &A = matrices[0];
  Tensor &Q = matrices[1];
  const auto a = generateInput({m, n});
  const auto identity = generateIdentity(m);

  graph.createHostWrite("a", A);
  graph.createHostWrite("q", Q);

  Sequence main;

  Tensor ARef, QRef, errA, errQ;
  initAuxTensors(graph, ARef, QRef, errA, errQ, m, n);
  saveAQValues(main, A, Q, ARef, QRef);

  OptionFlags options;
  options.set("rowsPerIteration", std::to_string(rowsPerIteration));
  QRFactorization(graph, A, Q, main, {}, options);

  computeErrors(graph, main, A, Q, ARef, QRef, errA, errQ);

  graph.createHostRead("errA", errA);
  graph.createHostRead("errQ", errQ);

  Errors errors;
  Engine eng(graph, main);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("a", a.data(), a.data() + a.size());
    eng.writeTensor("q", identity.data(), identity.data() + identity.size());
    eng.run();
    eng.readTensor("errA", &errors.ANormErr, &errors.ANormErr + 1);
    eng.readTensor("errQ", &errors.QNormErr, &errors.QNormErr + 1);
  });
  return errors;
}
} // namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = TEST_TARGET;
  std::size_t m, n;
  unsigned tiles = 1;
  int rowsPerIteration = 32;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type", po::value<DeviceType>(&deviceType)->required(),
     "Device type")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use")
    ("m", po::value<std::size_t>(&m)->required(),
     "Number of rows in the input matrix")
    ("n", po::value<std::size_t>(&n)->required(),
     "Number of columns in the input matrix")
    ("rows-per-iteration",
     po::value<int>(&rowsPerIteration)->implicit_value(rowsPerIteration),
     "Value to set the 'rowsPerIteration' option in QR factorization.")
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

  const auto errors =
      testQRFactorization(deviceType, m, n, tiles, rowsPerIteration);
  constexpr float floatPrecision = 1.1920929e-7;
  const float threshold = m * floatPrecision;
  if (errors.ANormErr > threshold || errors.QNormErr > threshold) {
    std::cerr << "Test failed, ||A - QR|| / ||A|| = " << errors.ANormErr
              << " ||I - QQ ^ T|| = " << errors.QNormErr << std::endl;
    return 1;
  }

  return 0;
}
