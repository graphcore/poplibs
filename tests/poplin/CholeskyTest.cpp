// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CholeskyTest
#include <poplibs_support/TestDevice.hpp>

#include <iostream>

#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <poplin/Cholesky.hpp>
#include <poplin/MatMul.hpp>
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

std::vector<float> createTriMat(std::size_t numBatches, std::size_t N,
                                bool lower, float batchIncrement = 100.0f) {
  float start = 1;
  std::vector<float> v(numBatches * N * N);
  for (std::size_t b = 0, k = 0; b < numBatches; ++b, k += N * N) {
    float f = start;
    for (std::size_t i = 0; i < N; i++) {
      for (std::size_t j = 0; j < N; j++, f += 1.0f) {
        // Generate the same values (tranposed)
        if (lower)
          v[k + i * N + j] = j > i ? 0 : f;
        else
          v[k + j * N + i] = j > i ? 0 : f;
      }
    }
    start += batchIncrement;
  }
  return v;
}

} // namespace

BOOST_DATA_TEST_CASE(CholeskyTest,
                     bud::make({20, 32}) * bud::make({16}) * bud::make({1, 2}) *
                         bud::make({true, false}),
                     N_, blockSize_, numBatches_, lower_) {
  std::size_t N = N_;
  std::size_t blockSize = blockSize_;
  std::size_t numBatches = numBatches_;
  bool lower = lower_;

  std::cout << "Running test: N = " << N << ", lower = " << lower
            << ", blockSize = " << blockSize << ", numBatches = " << numBatches
            << std::endl;

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  std::vector<float> Tv = createTriMat(numBatches, N, lower);

  Sequence prog;

  auto T = graph.addVariable(poplar::FLOAT, {numBatches, N, N}, "T");
  poputil::mapTensorLinearly(graph, T);
  graph.createHostWrite("T", T);

  auto A =
      lower ? poplin::matMulGrouped(graph, T, poplin::transposeGroupedMatrix(T),
                                    prog, T.elementType())
            : poplin::matMulGrouped(graph, poplin::transposeGroupedMatrix(T), T,
                                    prog, T.elementType());
  graph.createHostRead("A", A);

  poplar::OptionFlags options{{"blockSize", std::to_string(blockSize)}};
  auto matmulOptPairs = getCholeskyMatMulPrePlanParameters(
      A.elementType(), A.shape(), lower, options);

  std::set<MatMulPlanParams> params;
  for (auto &pair : matmulOptPairs)
    params.emplace(&target, pair.first, &pair.second);
  matmul::PlanningCache cache;
  preplanMatMuls(params, cache);
  BOOST_TEST(cache.size() == matmulOptPairs.size());

  if (N > blockSize) {
    BOOST_TEST(cache.size() > 0);
  } else {
    BOOST_TEST(cache.size() == 0);
  }

  poplar::DebugContext debugContext;

  auto T2 =
      poplin::cholesky(graph, A, lower, prog, debugContext, options, &cache);
  BOOST_TEST(cache.size() == matmulOptPairs.size());

  auto A2 =
      lower
          ? poplin::matMulGrouped(graph, T2, poplin::transposeGroupedMatrix(T2),
                                  prog, T2.elementType())
          : poplin::matMulGrouped(graph, poplin::transposeGroupedMatrix(T2), T2,
                                  prog, T2.elementType());
  graph.createHostRead("A2", A2);

  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("T", Tv.data());
    eng.run();

    std::vector<float> Av(Tv.size());
    eng.readTensor("A", Av.data());

    std::vector<float> A2v(Tv.size());
    eng.readTensor("A2", A2v.data());

    for (std::size_t i = 0; i < Av.size(); i++) {
      BOOST_REQUIRE_CLOSE(Av[i], A2v[i], 0.001f);
    }
  });
}
