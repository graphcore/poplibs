#define BOOST_TEST_MODULE AddToChannel2d

#include <boost/test/unit_test.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <popconv/codelets.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <iostream>
#include <functional>
#include <limits>
#include <boost/multi_array.hpp>
#include "TestDevice.hpp"

// Tolerances used in tests
#define FLOAT_REL_TOL  0.01
#define HALF_REL_TOL   0.1
#define FLOAT_ABS_TOL  1e-6
#define HALF_ABS_TOL   1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;

const OptionFlags options {
  {"target.textSectionSizeInBytes", "0xa000"},
  {"target.workerStackSizeInBytes", "0x400" }

};

struct TestCase {
  poplar::Type type;
  std::vector<std::size_t> addendLen;
  std::vector<std::size_t> actsLen;
  float scale;
};

struct TestCaseData {
  // These are all the addends/acts concatenated together.
  std::vector<double> allAddends;
  std::vector<double> allActs;

  std::unique_ptr<char[]> rawAllAddends;
  std::unique_ptr<char[]> rawAllActs;
};

static bool addToChannel2DTests(const std::vector<TestCase> &cases) {

  const std::size_t overwriteLen = 32;

  auto device = createTestDevice(TEST_TARGET, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(device);
  popconv::addCodelets(graph);

  // One compute set, with a vertex for each test case.
  auto cs = graph.addComputeSet("cs");

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::vector<TestCaseData> tcData(cases.size());

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto &tc = cases[i];

    // Check the actsLen's are ok.
    if (tc.actsLen.size() != tc.addendLen.size())
      return false;

    for (unsigned a = 0; a < tc.actsLen.size(); ++a) {
      if (tc.actsLen[a] % tc.addendLen[a] != 0) {
        return false;
      }
    }

    std::size_t totalAddendLen = std::accumulate(tc.addendLen.begin(),
                                                 tc.addendLen.end(),
                                                 0);
    std::size_t totalActsLen = std::accumulate(tc.actsLen.begin(),
                                               tc.actsLen.end(),
                                               0);

    std::cout << "Test case [" << i << "]: "
              << " addendLen.size(): " << tc.addendLen.size()
              << " actsLen.size(): " << tc.actsLen.size()
              << " totalAddendLen: " << totalAddendLen
              << " totalActsLen: " << totalActsLen
              << " scale: " << tc.scale
              << " type: " << tc.type.toString()
              << "\n";

    std::string suffix = "_" + std::to_string(i);

    auto allAddends = graph.addVariable(tc.type, {totalAddendLen},
                                        "allAddends" + suffix);
    graph.setTileMapping(allAddends, 0);
    auto allActs = graph.addVariable(tc.type, {totalActsLen + overwriteLen},
                                     "allActs" + suffix);
    graph.setTileMapping(allActs, 0);

    auto vertexName = tc.scale == 1.0f ? "popconv::AddToChannel2D"
                                       : "popconv::ScaledAddToChannel2D";

    auto v = graph.addVertex(cs, templateVertex(vertexName, tc.type));

    // Connect the acts and addend subvectors.

    graph.setInitialValue(v["n"], tc.actsLen.size());
    graph.setFieldSize(v["acts"], tc.actsLen.size());
    graph.setFieldSize(v["addend"], tc.actsLen.size());
    graph.setFieldSize(v["addendLen"], tc.actsLen.size());
    graph.setFieldSize(v["actsBlockCount"], tc.actsLen.size());

    std::size_t actsPos = 0;
    std::size_t addendPos = 0;
    for (unsigned a = 0; a < tc.actsLen.size(); ++a) {
      graph.connect(v["acts"][a],
                    allActs.slice(actsPos, actsPos + tc.actsLen[a]));
      graph.connect(v["addend"][a],
                    allAddends.slice(addendPos, addendPos + tc.addendLen[a]));

      actsPos += tc.actsLen[a];
      addendPos += tc.addendLen[a];

      auto actsBlockCount = tc.actsLen[a] / tc.addendLen[a];
      uint16_t actsBlockCount16 = actsBlockCount;

      if (actsBlockCount16 != actsBlockCount)
        return false;

      graph.setInitialValue(v["addendLen"][a], tc.addendLen[a]);
      graph.setInitialValue(v["actsBlockCount"][a], actsBlockCount16);
    }

    if (tc.scale != 1.0f)
      graph.setInitialValue(v["scale"], tc.scale);

    graph.setTileMapping(v, 0);

    tcData[i].rawAllAddends = allocateHostMemoryForTensor(allAddends,
                                                          "allAddend" + suffix,
                                                          graph, uploadProg,
                                                          downloadProg, tmap);
    tcData[i].rawAllActs = allocateHostMemoryForTensor(allActs,
                                                       "allActs" + suffix,
                                                       graph, uploadProg,
                                                       downloadProg, tmap);

    tcData[i].allAddends.resize(totalAddendLen);
    tcData[i].allActs.resize(totalActsLen + overwriteLen);

    std::mt19937 randomEngine;
    writeRandomValues(target,
                      allAddends.elementType(),
                      tcData[i].allAddends.data(),
                      tcData[i].allAddends.data() + tcData[i].allAddends.size(),
                      -2, 2,
                      randomEngine);
    writeRandomValues(target,
                      allActs.elementType(),
                      tcData[i].allActs.data(),
                      tcData[i].allActs.data() + tcData[i].allActs.size(),
                      -2, 2,
                      randomEngine);

    copy(target, tcData[i].allAddends.data(), tcData[i].allAddends.size(),
         allAddends.elementType(), tcData[i].rawAllAddends.get());
    copy(target, tcData[i].allActs.data(), tcData[i].allActs.size(),
         allActs.elementType(), tcData[i].rawAllActs.get());
  }

  std::cout << "Executing engine\n";

  auto prog = Execute(cs);
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);

  engine.load(device);
  attachStreams(engine, tmap);

  engine.run(0);

  std::cout << "Checking results\n";

  // Check the results for each test case.

  for (std::size_t i = 0; i < tcData.size(); ++i) {
    const auto &tc = cases[i];

    std::cout << "Checking case [" << i << "]\n";

    // Convert back to double.
    std::vector<double> allActsOut(tcData[i].allActs.size(), 0.0);

    copy(target, tc.type, tcData[i].rawAllActs.get(),
         allActsOut.data(), allActsOut.size());

    // Calculate the correct answer.
    auto allActsRef = tcData[i].allActs;

    std::size_t actsPos = 0;
    std::size_t addendPos = 0;
    for (unsigned a = 0; a < tc.actsLen.size(); ++a) {
      for (std::size_t k = 0; k < tc.actsLen[a]; ++k) {
        allActsRef[actsPos + k] +=
            tcData[i].allAddends[addendPos + (k % tc.addendLen[a])] * tc.scale;
      }
      actsPos += tc.actsLen[a];
      addendPos += tc.addendLen[a];
    }

    const double absoluteTolerance = tc.type == FLOAT ? FLOAT_ABS_TOL :
                                                        HALF_ABS_TOL;
    const double relativeTolerance = tc.type == FLOAT ? FLOAT_REL_TOL :
                                                        HALF_REL_TOL;

    auto matchesModel = checkIsClose("out",
                                     allActsOut.data(),
                                     {allActsOut.size()},
                                     allActsRef.data(),
                                     allActsOut.size(),
                                     relativeTolerance, absoluteTolerance);
    if (!matchesModel)
      return false;
  }

  return true;
}

void runAddToChannel2DTests(std::vector<TestCase> cases) {
  // Test ScaledAddToChannel
  BOOST_TEST(addToChannel2DTests(cases));

  // Test AddToChannel
  for (auto &tc : cases)
    tc.scale = 1.0f;
  BOOST_TEST(addToChannel2DTests(cases));
}

BOOST_AUTO_TEST_CASE(AddToChannel2DSmall) {
  std::vector<TestCase> cases = {
    {HALF, {1, 4, 8, 12, 16}, {480, 480, 480, 480, 480}, 3.0f},
    {HALF, {1, 4, 8, 5, 8}, {15, 12, 40, 15, 168}, 3.0f},
    {FLOAT, {1, 4, 8, 12, 16}, {480, 480, 480, 480, 480}, 3.0f},
    {FLOAT, {1, 4, 8, 5, 8}, {15, 12, 40, 15, 168}, 3.0f},
  };
  runAddToChannel2DTests(cases);
}

std::size_t maxBlockCount() {
  // This is the maximum acts_block_count the vertex supports.
  return 4095;
}

BOOST_AUTO_TEST_CASE(AddToChannel2DLarge1_half) {
  std::vector<TestCase> cases = {
    {HALF, {1, 1}, {maxBlockCount(), 80}, 3.0f},
  };
  runAddToChannel2DTests(cases);
}

BOOST_AUTO_TEST_CASE(AddToChannel2DLarge8_half) {
  std::vector<TestCase> cases = {
    {HALF, {8, 8}, {8000, 80}, 3.0f},
  };
  runAddToChannel2DTests(cases);
}

BOOST_AUTO_TEST_CASE(AddToChannel2DLarge1_float) {
  std::vector<TestCase> cases = {
    {FLOAT, {1, 1}, {maxBlockCount(), 80}, 3.0f},
  };
  runAddToChannel2DTests(cases);
}

BOOST_AUTO_TEST_CASE(AddToChannel2DLarge8_float) {
  std::vector<TestCase> cases = {
    {FLOAT, {8, 8}, {8000, 80}, 3.0f},
  };
  runAddToChannel2DTests(cases);
}
