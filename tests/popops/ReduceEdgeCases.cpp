// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ReduceEdgeCases
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/unit_test.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_test::util;
using namespace poplibs_support;

const OptionFlags options;

BOOST_AUTO_TEST_CASE(Reduce_Nop_ADD_float) {

  // Tests for nop reductions, where the reduced dimension is 1, or
  // any of the input dimensions are 0.

  // Workaround GCC 5 bug.
  using TestCase =
      std::tuple<std::vector<std::size_t>,  // Input shape
                 std::vector<std::size_t>,  // Reduced dimensions
                 std::vector<std::size_t>>; // Expected output shape

  std::vector<TestCase> testCases = {
      TestCase{{2, 1, 2, 3}, {1, 2}, {2, 3}},
      TestCase{{2, 3, 4, 0}, {3}, {2, 3, 4}},
      TestCase{{2, 3, 4, 0}, {0}, {3, 4, 0}},
      TestCase{{1, 1, 1}, {1}, {1, 1}},
      TestCase{{1, 1, 1, 0}, {0, 1}, {1, 0}},
      TestCase{{0, 1, 2}, {}, {0, 1, 2}},
      TestCase{{0, 1, 2, 3}, {3}, {0, 1, 2}},
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence prog;
  for (const auto &testCase : testCases) {
    const auto &inShape = std::get<0>(testCase);
    const auto &dims = std::get<1>(testCase);
    const auto &outShape = std::get<2>(testCase);

    auto in = graph.addVariable(FLOAT, inShape, "in");
    poputil::mapTensorLinearly(graph, in);

    auto out =
        popops::reduce(graph, in, FLOAT, dims, popops::Operation::ADD, prog);
    BOOST_TEST(out.shape() == outShape);
  }
}

BOOST_AUTO_TEST_CASE(ReduceIntermediatePrec) {
  // Test that we can accumulate in higher precision by adding lots of small
  // values to a large value such that if it were done with half precision
  // accumulation all the smaller terms would be lost.
  auto tdevice = createTestDevice(TEST_TARGET);
  const auto &target = tdevice.getTarget();
  Graph graph(target);

  popops::addCodelets(graph);

  const auto N = 100;
  Tensor input = graph.addVariable(HALF, {N});
  poputil::mapTensorLinearly(graph, input);

  Sequence prog;

  auto out = reduce(graph, input, {0}, popops::Operation::ADD, prog);

  std::vector<float> hInput(N);
  hInput[0] = 8192;
  for (unsigned i = 1; i < N; ++i)
    hInput[i] = 1;

  graph.setInitialValue(input, poplar::ArrayRef<float>(hInput));
  graph.createHostRead("out", out);

  Engine engine(graph, prog, options);
  tdevice.bind([&](const Device &device) {
    engine.load(device);
    engine.run(0);

    std::vector<char> hVal(target.getTypeSize(HALF));
    float val;

    engine.readTensor("out", hVal.data(), hVal.data() + hVal.size());

    copyDeviceHalfToFloat(target, hVal.data(), &val, 1);

    // In the half precision range > 8192 the representation will round to
    // multiples of 8
    BOOST_CHECK_EQUAL(val, 8192 + ((N - 1) / 8) * 8);
  });
}

BOOST_AUTO_TEST_CASE(Reduce_Huge_ADD_float) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  // create a huge amount of partials and map them all to the same tile to blow
  // the vector list 12-bit count limit.
  auto in = graph.addVariable(HALF, {{3, 10500, 3}}, "in");
  graph.setTileMapping(in, 0);

  Sequence prog;
  popops::reduce(graph, in, HALF, {1}, popops::Operation::ADD, prog);

  // we expect this to throw an out of memory exception but NOT an exception
  // complaining about the number of partials.
  try {
    Engine e(graph, prog, {{"debug.allowOutOfMemory", "true"}});
  } catch (const poplar::graph_memory_allocation_error &) {
  };
}

BOOST_AUTO_TEST_CASE(Avoid_subword_mapping_single_element_per_tile) {
  const unsigned numOutputs = 4;
  auto device = createTestDevice(TEST_TARGET, 1, numOutputs);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto inHalf = graph.addVariable(HALF, {{numOutputs, 100}}, "inH");
  auto inFloat = graph.addVariable(HALF, {{numOutputs, 100}}, "inF");

  for (unsigned i = 0; i != numOutputs; ++i) {
    graph.setTileMapping(inHalf[i], i);
    graph.setTileMapping(inFloat[i], i);
  }

  Sequence prog;
  auto outHalf =
      popops::reduce(graph, inHalf, HALF, {1}, popops::Operation::ADD, prog);
  auto outFloat =
      popops::reduce(graph, inHalf, FLOAT, {1}, popops::Operation::ADD, prog);

  // check over how many tiles the output is mapped
  auto tMap = graph.getTileMapping(outHalf);
  auto tilesContainingOutHalf =
      std::accumulate(tMap.begin(), tMap.end(), 0U,
                      [](unsigned num, const std::vector<Interval> &mapping) {
                        return num + !mapping.empty();
                      });

  tMap = graph.getTileMapping(outFloat);
  auto tilesContainingOutFloat =
      std::accumulate(tMap.begin(), tMap.end(), 0U,
                      [](unsigned num, const std::vector<Interval> &mapping) {
                        return num + !mapping.empty();
                      });

  // The output should be remapped
  BOOST_CHECK_NE(tilesContainingOutHalf, numOutputs);
  // The output should not be remapped
  BOOST_CHECK_EQUAL(tilesContainingOutFloat, numOutputs);
}
