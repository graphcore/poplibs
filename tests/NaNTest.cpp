// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE NaNTest

#include "popops/NaN.hpp"
#include "TestDevice.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/ElementWise.hpp"
#include "popops/codelets.hpp"
#include "poputil/TileMapping.hpp"
#include <boost/multi_array.hpp>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

static constexpr std::size_t D0 = 6;
static constexpr std::size_t D1 = 5;
static constexpr std::size_t D2 = 4;
static constexpr std::size_t D3 = 3;

void hasNaNTest(const bool introduceNaN, const Type &type) {
  std::mt19937 randomEngine;
  boost::random::uniform_real_distribution<double> dist(0., 10.);

  boost::multi_array<double, 4> input(boost::extents[D0][D1][D2][D3]);

  for (unsigned i = 0; i < input.num_elements(); ++i) {
    *(input.data() + i) = dist(randomEngine);
  }

  if (introduceNaN) {
    input[0][1][2][2] = NAN;
  }

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  const auto inputT = graph.addVariable(type, {D0, D1, D2, D3}, "input");
  poputil::mapTensorLinearly(graph, inputT);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  auto rawHostInput = allocateHostMemoryForTensor(
      inputT, "input", graph, uploadProg, downloadProg, tmap);
  copy(target, input, type, rawHostInput.get());

  Sequence prog;
  const auto out = popops::hasNaN(graph, inputT, prog);
  auto rawHostOutput = allocateHostMemoryForTensor(
      out, "out", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  boost::multi_array<bool, 1> result(boost::extents[1]);
  copy(target, BOOL, rawHostOutput.get(), result);

  BOOST_TEST(result[0] == introduceNaN);
}

BOOST_AUTO_TEST_CASE(HasNaN_Float_False) { hasNaNTest(false, FLOAT); }

BOOST_AUTO_TEST_CASE(HasNaN_Float_True) { hasNaNTest(true, FLOAT); }

BOOST_AUTO_TEST_CASE(HasNaN_Half_False) { hasNaNTest(false, HALF); }

BOOST_AUTO_TEST_CASE(HasNaN_Half_True) { hasNaNTest(true, HALF); }
