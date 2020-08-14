// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ChannelMul2d

#include "../lib/popops/ExprOpUtil.hpp"
#include <boost/multi_array.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>

// Tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplibs_support;

const OptionFlags options;

struct TestCase {
  poplar::Type type;
  std::vector<std::size_t> scaleLen;
  std::vector<std::size_t> actsLen;
};

struct TestCaseData {
  // These are all the scales/acts concatenated together.
  std::vector<double> allScales;
  std::vector<double> allActsIn;

  std::unique_ptr<char[]> rawAllScales;
  std::unique_ptr<char[]> rawAllActsIn;
  std::unique_ptr<char[]> rawAllActsOut;
};

static bool channelMul2DTests(const std::vector<TestCase> &cases) {

  const std::size_t overwriteLen = 32;

  auto device = createTestDevice(TEST_TARGET, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // One compute set, with a vertex for each test case.
  auto cs = graph.addComputeSet("cs");

  Sequence uploadProg;
  Sequence downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::vector<TestCaseData> tcData(cases.size());

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto &tc = cases[i];

    // Check the actsLen's are ok.
    if (tc.actsLen.size() != tc.scaleLen.size())
      return false;

    for (unsigned a = 0; a < tc.actsLen.size(); ++a) {
      if (tc.actsLen[a] % tc.scaleLen[a] != 0) {
        return false;
      }
    }

    std::size_t totalScaleLen =
        std::accumulate(tc.scaleLen.begin(), tc.scaleLen.end(), 0);
    std::size_t totalActsLen =
        std::accumulate(tc.actsLen.begin(), tc.actsLen.end(), 0);

    std::cout << "Test case [" << i << "]: "
              << " scaleLen.size(): " << tc.scaleLen.size()
              << " actsLen.size(): " << tc.actsLen.size()
              << " totalScaleLen: " << totalScaleLen
              << " totalActsLen: " << totalActsLen
              << " type: " << tc.type.toString() << "\n";

    std::string suffix = "_" + std::to_string(i);

    auto allScales =
        graph.addVariable(tc.type, {totalScaleLen}, "allScales" + suffix);
    auto allActsIn =
        graph.addVariable(tc.type, {totalActsLen}, "allActsIn" + suffix);
    auto allActsOut = graph.addVariable(tc.type, {totalActsLen + overwriteLen},
                                        "allActsOut" + suffix);
    graph.setTileMapping(allScales, 0);
    graph.setTileMapping(allActsIn, 0);
    graph.setTileMapping(allActsOut, 0);

    std::string templateVertexName =
        templateVertex("popops::BroadcastVectorInner2D",
                       popops::expr::BinaryOpType::MULTIPLY, tc.type);

    auto v = graph.addVertex(cs, templateVertexName);

    // Connect the acts and scale subvectors.

    graph.setInitialValue(v["n"], tc.actsLen.size());
    graph.setFieldSize(v["data"], tc.actsLen.size());
    graph.setFieldSize(v["out"], tc.actsLen.size());
    graph.setFieldSize(v["B"], tc.actsLen.size());
    graph.setFieldSize(v["BLen"], tc.actsLen.size());
    graph.setFieldSize(v["dataBlockCount"], tc.actsLen.size());

    std::size_t actsPos = 0;
    std::size_t scalePos = 0;
    for (unsigned a = 0; a < tc.actsLen.size(); ++a) {
      graph.connect(v["data"][a],
                    allActsIn.slice(actsPos, actsPos + tc.actsLen[a]));
      graph.connect(v["out"][a],
                    allActsOut.slice(actsPos, actsPos + tc.actsLen[a]));
      graph.connect(v["B"][a],
                    allScales.slice(scalePos, scalePos + tc.scaleLen[a]));

      actsPos += tc.actsLen[a];
      scalePos += tc.scaleLen[a];

      auto actsBlockCount = tc.actsLen[a] / tc.scaleLen[a];
      uint16_t actsBlockCount16 = actsBlockCount;

      if (actsBlockCount16 != actsBlockCount)
        return false;

      graph.setInitialValue(v["BLen"][a], tc.scaleLen[a]);
      graph.setInitialValue(v["dataBlockCount"][a], actsBlockCount16);
    }

    graph.setTileMapping(v, 0);

    tcData[i].rawAllScales = allocateHostMemoryForTensor(
        allScales, "allScale" + suffix, graph, uploadProg, downloadProg, tmap);
    tcData[i].rawAllActsIn = allocateHostMemoryForTensor(
        allActsIn, "allActsIn" + suffix, graph, uploadProg, downloadProg, tmap);
    tcData[i].rawAllActsOut =
        allocateHostMemoryForTensor(allActsOut, "allActsOut" + suffix, graph,
                                    uploadProg, downloadProg, tmap);

    tcData[i].allScales.resize(totalScaleLen);
    tcData[i].allActsIn.resize(totalActsLen);

    std::mt19937 randomEngine;
    writeRandomValues(target, allScales.elementType(),
                      tcData[i].allScales.data(),
                      tcData[i].allScales.data() + tcData[i].allScales.size(),
                      -2., 2., randomEngine);
    writeRandomValues(target, allActsIn.elementType(),
                      tcData[i].allActsIn.data(),
                      tcData[i].allActsIn.data() + tcData[i].allActsIn.size(),
                      -2., 2., randomEngine);

    copy(target, tcData[i].allScales.data(), tcData[i].allScales.size(),
         allScales.elementType(), tcData[i].rawAllScales.get());
    copy(target, tcData[i].allActsIn.data(), tcData[i].allActsIn.size(),
         allActsIn.elementType(), tcData[i].rawAllActsIn.get());
    // Set actsOut to 1, to verify there are no overwrites.
    std::vector<double> ones(allActsOut.numElements(), 1.0);
    copy(target, ones.data(), ones.size(), allActsOut.elementType(),
         tcData[i].rawAllActsOut.get());
  }

  std::cout << "Executing engine\n";

  auto prog = Execute(cs);
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);

  attachStreams(engine, tmap);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  std::cout << "Checking results\n";

  // Check the results for each test case.

  for (std::size_t i = 0; i < tcData.size(); ++i) {
    const auto &tc = cases[i];

    std::cout << "Checking case [" << i << "]\n";

    // Convert back to double.
    std::vector<double> allActsOut(tcData[i].allActsIn.size() + overwriteLen,
                                   0.0);

    copy(target, tc.type, tcData[i].rawAllActsOut.get(), allActsOut.data(),
         allActsOut.size());

    // Calculate the correct answer.
    std::vector<double> allActsOutRef(allActsOut.size(), 1.0);

    std::size_t actsPos = 0;
    std::size_t scalePos = 0;
    for (unsigned a = 0; a < tc.actsLen.size(); ++a) {
      for (std::size_t k = 0; k < tc.actsLen[a]; ++k) {
        allActsOutRef[actsPos + k] =
            tcData[i].allActsIn[actsPos + k] *
            tcData[i].allScales[scalePos + (k % tc.scaleLen[a])];
      }
      actsPos += tc.actsLen[a];
      scalePos += tc.scaleLen[a];
    }

    const double absoluteTolerance =
        tc.type == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    const double relativeTolerance =
        tc.type == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;

    auto matchesModel = checkIsClose(
        "out", allActsOut.data(), {allActsOut.size()}, allActsOutRef.data(),
        allActsOut.size(), relativeTolerance, absoluteTolerance);
    if (!matchesModel)
      return false;
  }

  return true;
}

const bool isNotSim = !isSimulator(TEST_TARGET);

BOOST_AUTO_TEST_CASE(ChannelMul2DTiny) {
  std::vector<TestCase> cases = {
      {HALF, {1, 4, 8, 5}, {15, 12, 32, 15}},
      {FLOAT, {1, 4, 8, 5}, {15, 12, 32, 15}},
  };
  BOOST_TEST(channelMul2DTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMul2DSmall,
                     *boost::unit_test::precondition(enableIfNotSim())) {
  std::vector<TestCase> cases = {
      {HALF, {1, 4, 8, 12, 16}, {480, 480, 480, 480, 480}},
      {HALF, {1, 4, 8, 5, 8}, {15, 12, 40, 15, 168}},
      {FLOAT, {1, 4, 8, 12, 16}, {480, 480, 480, 480, 480}},
      {FLOAT, {1, 4, 8, 5, 8}, {15, 12, 40, 15, 168}},
  };
  BOOST_TEST(channelMul2DTests(cases));
}

std::size_t maxBlockCount() {
  // This is the maximum acts_block_count the vertex supports.
  return 4095;
}

BOOST_AUTO_TEST_CASE(ChannelMul2DLarge1_half,
                     *boost::unit_test::precondition(enableIfNotSim())) {
  std::vector<TestCase> cases = {
      {HALF, {1, 1}, {maxBlockCount(), 80}},
  };
  BOOST_TEST(channelMul2DTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMul2DLarge8_half,
                     *boost::unit_test::precondition(enableIfNotSim())) {
  std::vector<TestCase> cases = {
      {HALF, {8, 8}, {8000, 80}},
  };
  BOOST_TEST(channelMul2DTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMul2DLarge1_float,
                     *boost::unit_test::precondition(enableIfNotSim())) {
  std::vector<TestCase> cases = {
      {FLOAT, {1, 1}, {maxBlockCount(), 80}},
  };
  BOOST_TEST(channelMul2DTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMul2DLarge8_float,
                     *boost::unit_test::precondition(enableIfNotSim())) {
  std::vector<TestCase> cases = {
      {FLOAT, {8, 8}, {8000, 80}},
  };
  BOOST_TEST(channelMul2DTests(cases));
}
