#define BOOST_TEST_MODULE ChannelMul

#include <boost/test/unit_test.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplin/codelets.hpp>
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
#define HALF_ABS_TOL   1e-3

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;

const OptionFlags options {
  {"target.workerStackSizeInBytes", "0x400" }
};

struct TestCase {
  poplar::Type type;
  std::size_t scaleLen;
  std::size_t actsLen;
};

struct TestCaseData {
  std::vector<double> scale;
  std::vector<double> actsIn;

  std::unique_ptr<char[]> rawScale;
  std::unique_ptr<char[]> rawActsIn;
  std::unique_ptr<char[]> rawActsOut;
};

static bool channelMulTests(const std::vector<TestCase> &cases) {

  const std::size_t overwriteLen = 32;

  auto device = createTestDevice(TEST_TARGET, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);

  // One compute set, with a vertex for each test case.
  auto cs = graph.addComputeSet("cs");

  Sequence uploadProg;
  Sequence downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::vector<TestCaseData> tcData(cases.size());

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto &tc = cases[i];
    if (tc.actsLen % tc.scaleLen != 0) {
      return false;
    }

    std::cout << "Test case [" << i << "]: "
              << " scaleLen: " << tc.scaleLen
              << " actsLen: " << tc.actsLen
              << " type: " << tc.type.toString()
              << "\n";

    std::string suffix = "_" + std::to_string(i);

    auto scale = graph.addVariable(tc.type, {tc.scaleLen},
                                   "scale" + suffix);
    auto actsIn = graph.addVariable(tc.type, {tc.actsLen},
                                    "actsIn" + suffix);
    auto actsOut = graph.addVariable(tc.type, {tc.actsLen + overwriteLen},
                                     "actsOut" + suffix);
    graph.setTileMapping(scale, 0);
    graph.setTileMapping(actsIn, 0);
    graph.setTileMapping(actsOut, 0);

    auto vertexName = "poplin::ChannelMul";

    auto v = graph.addVertex(cs,
                             templateVertex(vertexName, tc.type),
                             {
                               {"actsIn", actsIn},
                               {"actsOut", actsOut},
                               {"scale", scale}
                             });

    auto actsBlockCount = tc.actsLen / tc.scaleLen;
    auto actsBlockCountPacked = ((actsBlockCount / 6) << 3)
                               | (actsBlockCount % 6);
    uint16_t actsBlockCountPacked16 = actsBlockCountPacked;

    if (actsBlockCountPacked16 != actsBlockCountPacked)
      return false;

    graph.setInitialValue(v["actsBlockCountPacked"], actsBlockCountPacked16);

    graph.setTileMapping(v, 0);

    tcData[i].rawScale =
        allocateHostMemoryForTensor(scale, "scale" + suffix, graph,
                                    uploadProg, downloadProg, tmap);
    tcData[i].rawActsIn =
        allocateHostMemoryForTensor(actsIn, "actsIn" + suffix, graph,
                                    uploadProg, downloadProg, tmap);
    tcData[i].rawActsOut =
        allocateHostMemoryForTensor(actsOut, "actsOut" + suffix, graph,
                                    uploadProg, downloadProg, tmap);

    tcData[i].scale.resize(tc.scaleLen);
    tcData[i].actsIn.resize(tc.actsLen);

    std::mt19937 randomEngine;
    writeRandomValues(target,
                      scale.elementType(),
                      tcData[i].scale.data(),
                      tcData[i].scale.data() + tcData[i].scale.size(),
                      -2, 2,
                      randomEngine);
    writeRandomValues(target,
                      actsIn.elementType(),
                      tcData[i].actsIn.data(),
                      tcData[i].actsIn.data() + tcData[i].actsIn.size(),
                      -2, 2,
                      randomEngine);

    copy(target, tcData[i].scale.data(), tcData[i].scale.size(),
         scale.elementType(), tcData[i].rawScale.get());
    copy(target, tcData[i].actsIn.data(), tcData[i].actsIn.size(),
         actsIn.elementType(), tcData[i].rawActsIn.get());
    // Set actsOut to 1, to verify there are no overwrites.
    std::vector<double> ones(actsOut.numElements(), 1.0);
    copy(target, ones.data(), ones.size(),
         actsOut.elementType(), tcData[i].rawActsOut.get());
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
    std::vector<double> actsOut(tcData[i].actsIn.size() + overwriteLen, 0.0);

    copy(target, tc.type, tcData[i].rawActsOut.get(),
         actsOut.data(), actsOut.size());

    // Check the answer.
    std::vector<double> actsOutRef(actsOut.size(), 1.0);
    for (std::size_t k = 0; k < tc.actsLen; ++k)
      actsOutRef[k] = tcData[i].actsIn[k] * tcData[i].scale[k % tc.scaleLen];

    const double absoluteTolerance = tc.type == FLOAT ? FLOAT_ABS_TOL :
                                                        HALF_ABS_TOL;
    const double relativeTolerance = tc.type == FLOAT ? FLOAT_REL_TOL :
                                                        HALF_REL_TOL;

    auto matchesModel = checkIsClose("out",
                                     actsOut.data(),
                                     {actsOut.size()},
                                     actsOutRef.data(),
                                     actsOut.size(),
                                     relativeTolerance, absoluteTolerance);
    if (!matchesModel)
      return false;
  }

  return true;
}

const bool isNotSim = TEST_TARGET != DeviceType::Sim;

BOOST_AUTO_TEST_CASE(ChannelMulTiny) {
  std::vector<TestCase> cases = {
    {HALF, 1, 8},
    {HALF, 4, 16},
    {HALF, 8, 32},
    {HALF, 5, 15},
    {FLOAT, 1, 8},
    {FLOAT, 4, 16},
    {FLOAT, 8, 32},
    {FLOAT, 5, 15},
  };
  BOOST_TEST(channelMulTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMulSmall,
                     *boost::unit_test::enable_if<isNotSim>()) {
  std::vector<TestCase> cases = {
    {HALF, 1, 480},
    {HALF, 4, 480},
    {HALF, 8, 480},
    {HALF, 12, 480},
    {HALF, 16, 480},
    {HALF, 1, 15},
    {HALF, 4, 12},
    {HALF, 8, 40},
    {HALF, 5, 15},
    {HALF, 8, 168},

    {FLOAT, 1, 480},
    {FLOAT, 4, 480},
    {FLOAT, 8, 480},
    {FLOAT, 12, 480},
    {FLOAT, 16, 480},
    {FLOAT, 1, 15},
    {FLOAT, 4, 12},
    {FLOAT, 8, 40},
    {FLOAT, 5, 15},
    {FLOAT, 8, 168},
  };
  BOOST_TEST(channelMulTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMulLarge1_half,
                     *boost::unit_test::enable_if<isNotSim>()) {
  std::vector<TestCase> cases = {
    {HALF, 1, 8000},
  };
  BOOST_TEST(channelMulTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMulLarge8_half,
                     *boost::unit_test::enable_if<isNotSim>()) {
  std::vector<TestCase> cases = {
    {HALF, 8, 8000},
  };
  BOOST_TEST(channelMulTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMulLarge1_float,
                     *boost::unit_test::enable_if<isNotSim>()) {
  std::vector<TestCase> cases = {
    {FLOAT, 1, 8000},
  };
  BOOST_TEST(channelMulTests(cases));
}

BOOST_AUTO_TEST_CASE(ChannelMulLarge8_float,
                     *boost::unit_test::enable_if<isNotSim>()) {
  std::vector<TestCase> cases = {
    {FLOAT, 8, 8000},
  };
  BOOST_TEST(channelMulTests(cases));
}

// Above an addend length over 2044, we switch to scalar code. Check that works.
BOOST_AUTO_TEST_CASE(ChannelMul_MaxChannels_MultipleOfFour_half,
                     *boost::unit_test::enable_if<isNotSim>()) {
  for (std::size_t scaleLen = 2040; scaleLen <= 2052; scaleLen += 4) {
    std::vector<TestCase> cases = {
      {HALF, scaleLen, scaleLen * 4},
    };
    BOOST_TEST(channelMulTests(cases));
  }
}
