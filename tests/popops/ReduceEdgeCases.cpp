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

struct TestCase {
  std::vector<std::size_t> inShape;
  std::vector<std::size_t> dims;
  std::vector<std::size_t> outShape;
};

std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &test) {
  os << '[';
  StringRef sep = "";
  for (size_t item : test) {
    os << sep << item;
    sep = ", ";
  }
  return os << ']';
}

std::ostream &operator<<(std::ostream &os, const TestCase &test) {
  return os << "TestCase{inShape=" << test.inShape << ", dims=" << test.dims
            << ", outShape=" << test.outShape << "}";
}

// Call popops::reduce or popops::reduceMany depending on `useReduceMany` with
// a single input tensor and expect a single output tensor.
static Tensor reduceWrapper(bool useReduceMany, Graph &graph, const Tensor &in,
                            const std::vector<size_t> &dims, Sequence &prog) {
  if (!useReduceMany)
    return popops::reduce(graph, in, FLOAT, dims, popops::Operation::ADD, prog);
  else {
    std::vector<Tensor> outs;
    SingleReduceOp op = {in, dims, popops::Operation::ADD, FLOAT};
    popops::reduceMany(graph, {std::move(op)}, outs, prog);
    BOOST_TEST(outs.size() == 1);
    return outs[0];
  }
}

BOOST_AUTO_TEST_CASE(Reduce_Nop_ADD_float) {
  // Tests for nop reductions, where the reduced dimension is 1, or
  // any of the input dimensions are 0.
  std::vector<TestCase> testCases = {
      TestCase{{2, 1, 2, 3}, {1, 2}, {2, 3}},
      TestCase{{2, 3, 4, 0}, {3}, {2, 3, 4}},
      TestCase{{2, 3, 4, 0}, {0}, {3, 4, 0}},
      TestCase{{1, 1, 1}, {1}, {1, 1}},
      TestCase{{1, 1, 1, 0}, {0, 1}, {1, 0}},
      TestCase{{0, 1, 2}, {}, {0, 1, 2}},
      TestCase{{0, 1, 2, 3}, {3}, {0, 1, 2}},
      TestCase{{2, 2}, {0, 1}, {}},
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence prog;
  for (const auto &[inShape, dims, outShape] : testCases) {
    auto in = graph.addVariable(FLOAT, inShape, "in");
    poputil::mapTensorLinearly(graph, in);
    for (bool useReduceMany : {false, true}) {
      Tensor out = reduceWrapper(useReduceMany, graph, in, dims, prog);
      BOOST_TEST(out.shape() == outShape);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReduceCheckProgsAreOptimised) {
  // Any reduction that only removes dimensions of size 1 will leave the number
  // of output elements the same and so should be optimised to a simpler
  // expression such as a copy/cast or map expression. Unless the compute set
  // overload of reduce is used.
  const std::array<TestCase, 2> tests = {
      TestCase{{1}, {0}, {}},
      TestCase{{2, 2, 1}, {2}, {2, 2}},
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  for (const auto &test : tests) {
    for (bool useReduceMany : {false, true}) {
      std::string reduceName = (useReduceMany ? "reduceMany" : "reduce");
      Sequence prog({}, DebugContext{reduceName + "CheckProgsAreOptimised"});
      Tensor in = graph.addVariable(FLOAT, test.inShape,
                                    VariableMappingMethod::LINEAR, "in");
      Tensor out = reduceWrapper(useReduceMany, graph, in, test.dims, prog);
      BOOST_TEST(out.shape() == test.outShape);
      // This is quicker and simpler than compiling the graph and iterating
      // through the execution profile. However this is a terrible test
      // because it will still pass if there's a copy and a bunch of junk.
      // TODO: deserialize the JSON and check the results properly.
      std::stringstream s;
      dumpProgram(graph, prog, s);
      auto progJson = s.str();
      BOOST_TEST_MESSAGE(test << " with " << reduceName
                              << " => progJson=" << progJson);
      BOOST_TEST(progJson.find("Copy") != std::string::npos);
      BOOST_TEST(progJson.find("Execute") == std::string::npos);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReduceCheckNoProgsNeeded) {
  // When the output tensor is empty no reduction needs to happen.
  const std::array<TestCase, 2> tests = {
      TestCase{{2, 0, 1}, {2}, {2, 0}},
      TestCase{{0, 1, 2}, {}, {0, 1, 2}},
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  for (const auto &test : tests) {
    for (bool useReduceMany : {false, true}) {
      std::string reduceName = (useReduceMany ? "reduceMany" : "reduce");
      Sequence prog({}, DebugContext{reduceName + "CheckNoProgsNeeded"});
      Tensor in = graph.addVariable(FLOAT, test.inShape,
                                    VariableMappingMethod::LINEAR, "in");
      Tensor out = reduceWrapper(useReduceMany, graph, in, test.dims, prog);
      BOOST_TEST(out.shape() == std::vector<size_t>(test.outShape));
      BOOST_TEST(out.numElements() == 0);
      std::stringstream s;
      dumpProgram(graph, prog, s);
      auto progJson = s.str();
      BOOST_TEST_MESSAGE(test << " with " << reduceName
                              << " => progJson=" << progJson);
      BOOST_TEST(progJson.find("Copy") == std::string::npos);
      BOOST_TEST(progJson.find("Execute") == std::string::npos);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReduceCheckOutputFromEmptyInputIsFilled) {
  // When the inputs are empty but the output tensor is not empty the
  // output tensor needs to be created and filled with starting values
  // using an Execute program.
  const std::array<TestCase, 2> tests = {
      TestCase{{0}, {0}, {}},
      TestCase{{2, 0, 1}, {1}, {2, 1}},
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  for (const auto &test : tests) {
    for (bool useReduceMany : {false, true}) {
      std::string reduceName = (useReduceMany ? "reduceMany" : "reduce");
      Sequence prog(
          {}, DebugContext{reduceName + "CheckOutputFromEmptyInputIsFilled"});
      Tensor in = graph.addVariable(FLOAT, test.inShape,
                                    VariableMappingMethod::LINEAR, "in");
      Tensor out = reduceWrapper(useReduceMany, graph, in, test.dims, prog);
      BOOST_TEST(out.shape() == std::vector<size_t>(test.outShape));
      BOOST_TEST(out.numElements() != 0);
      std::stringstream s;
      dumpProgram(graph, prog, s);
      auto progJson = s.str();
      BOOST_TEST_MESSAGE(test << " with " << reduceName
                              << " => progJson=" << progJson);
      BOOST_TEST(progJson.find("Copy") == std::string::npos);
      BOOST_TEST(progJson.find("Execute") != std::string::npos);
    }
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
