// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NaNTest

#include "popops/NaN.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/ElementWise.hpp"
#include "popops/codelets.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include <boost/multi_array.hpp>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;

static constexpr std::size_t D0 = 6;
static constexpr std::size_t D1 = 5;
static constexpr std::size_t D2 = 4;
static constexpr std::size_t D3 = 3;

void hasNaNTest(const bool introduceNaN, const Type &type, std::size_t testSize,
                unsigned numTiles, bool twoDimensionalVertices) {
  std::mt19937 randomEngine;
  boost::random::uniform_real_distribution<double> dist(0., 10.);

  boost::multi_array<double, 4> input(boost::extents[D0][D1][D2][D3]);

  for (unsigned i = 0; i < input.num_elements(); ++i) {
    *(input.data() + i) = dist(randomEngine);
  }

  // Fill last element
  const std::vector<std::size_t> shape = {D0, D1, D2, D3};
  auto indices = poputil::unflattenIndex(shape, testSize - 1);

  if (introduceNaN) {
    input[indices[0]][indices[1]][indices[2]][indices[3]] = NAN;
  }

  // fill NANs outside
  std::fill(input.data() + testSize, input.data() + input.num_elements(), NAN);

  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  Tensor inputT;
  if (twoDimensionalVertices) {
    static_assert(D0 % 2 == 0);
    auto var1 = graph.addVariable(type, {D0 / 2, D1, D2, D3}, "input1");
    auto var2 = graph.addVariable(type, {D0 / 2, D1, D2, D3}, "input2");
    poputil::mapTensorLinearly(graph, var1);
    poputil::mapTensorLinearly(graph, var2);

    inputT = concat(var1, var2);
  } else {
    inputT = graph.addVariable(type, {D0, D1, D2, D3}, "input");
    poputil::mapTensorLinearly(graph, inputT);
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  auto rawHostInput = allocateHostMemoryForTensor(
      inputT, "input", graph, uploadProg, downloadProg, tmap);
  copy(target, input, type, rawHostInput.get());

  Sequence prog;
  const auto out =
      popops::hasNaN(graph, inputT.flatten().slice(0, testSize), prog);
  auto rawHostOutput = allocateHostMemoryForTensor(
      out, "out", graph, uploadProg, downloadProg, tmap);

  const poplar::OptionFlags options{{"debug.floatPointOpException", "false"}};
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  boost::multi_array<bool, 1> result(boost::extents[1]);
  copy(target, BOOL, rawHostOutput.get(), result);

  BOOST_TEST(result[0] == introduceNaN);
}

#define ENUMERATE_FULL_TEST(dType, addNaN, twoD)                               \
  BOOST_AUTO_TEST_SUITE(HasNaN##_suite)                                        \
  BOOST_AUTO_TEST_CASE(HasNan##_##dType##_##addNaN##_##twoD##_full) {          \
    const auto sizeToTest = D0 * D1 * D2 * D3;                                 \
    hasNaNTest(addNaN, dType, sizeToTest, 4, twoD);                            \
  }                                                                            \
  BOOST_AUTO_TEST_SUITE_END()

#define ENUMERATE_REM_TESTS(dType, addNaN, startOffset)                        \
  BOOST_AUTO_TEST_SUITE(HasNaN##_suite)                                        \
                                                                               \
  BOOST_AUTO_TEST_CASE(HasNan##_##dType##_##addNaN##_##startOffset##_rem1) {   \
    hasNaNTest(addNaN, dType, startOffset + 1, 1, false);                      \
  }                                                                            \
                                                                               \
  BOOST_AUTO_TEST_CASE(HasNan##_##dType##_##addNaN##_##startOffset##_rem2) {   \
    hasNaNTest(addNaN, dType, startOffset + 2, 1, false);                      \
  }                                                                            \
                                                                               \
  BOOST_AUTO_TEST_CASE(HasNan##_##dType##_##addNaN##_##startOffset##_rem3) {   \
    hasNaNTest(addNaN, dType, startOffset + 3, 1, false);                      \
  }                                                                            \
  BOOST_AUTO_TEST_SUITE_END()

// Enumerate tests
ENUMERATE_FULL_TEST(FLOAT, true, true)
ENUMERATE_FULL_TEST(FLOAT, true, false)
ENUMERATE_FULL_TEST(FLOAT, false, true)
ENUMERATE_FULL_TEST(FLOAT, false, false)
ENUMERATE_FULL_TEST(HALF, true, true)
ENUMERATE_FULL_TEST(HALF, true, false)
ENUMERATE_FULL_TEST(HALF, false, true)
ENUMERATE_FULL_TEST(HALF, false, false)
ENUMERATE_REM_TESTS(FLOAT, true, 0)
ENUMERATE_REM_TESTS(FLOAT, false, 0)
ENUMERATE_REM_TESTS(HALF, true, 0)
ENUMERATE_REM_TESTS(HALF, false, 0)
ENUMERATE_REM_TESTS(HALF, true, 4)
ENUMERATE_REM_TESTS(HALF, false, 4)
