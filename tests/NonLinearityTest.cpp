// Simple test case for IPU nonLinearity
//
#define BOOST_TEST_MODULE NonLinearityTest
#include <popnn/NonLinearity.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <iostream>
#include <poplibs_test/Util.hpp>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popnn;
using namespace poplibs_test::util;

const OptionFlags options {
  {"target.textSectionSizeInBytes", "0x9000"}
};

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define TOL 0.1 //tolerance of 0.1%
#define FLOAT_ATOL 1e-20
#define HALF_ATOL 1e-7

BOOST_AUTO_TEST_CASE(NonLinearity,
                    *utf::tolerance<float>(fpc::percent_tolerance<float>(TOL))
                    *utf::tolerance<double>(fpc::percent_tolerance<double>(TOL))
                     ) {
  if (TEST_TARGET == DeviceType::Sim)
    // test disabled until T3905 is fixed
    return;
  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();
  Graph graph(device);
  popnn::addCodelets(graph);
  //layer parameters

  const unsigned zNGroups = 1;
  const std::size_t zChunk = 1;
  const std::size_t ySize = 100;
  const std::size_t xSize = 30;
  auto actF = graph.addVariable(FLOAT, {1, zNGroups, ySize, xSize, zChunk},
                                "actF");
  auto actH = graph.addVariable(HALF, {1, zNGroups, ySize, xSize, zChunk},
                                "actH");
  auto deltaF = graph.addVariable(FLOAT, {1, zNGroups, ySize, xSize, zChunk},
                                  "actF");
  auto deltaH = graph.addVariable(HALF, {1, zNGroups, ySize, xSize, zChunk},
                                  "actH");

  // arbitraray mappings
  mapTensorLinearly(graph, actF);
  mapTensorLinearly(graph, actH);
  mapTensorLinearly(graph, deltaF);
  mapTensorLinearly(graph, deltaH);

  graph.createHostWrite("inF", actF);
  graph.createHostWrite("inH", actH);
  graph.createHostRead("outF", actF);
  graph.createHostRead("outH",actH);
  graph.createHostWrite("inDeltaF", deltaF);
  graph.createHostWrite("inDeltaH", deltaH);
  graph.createHostRead("outDeltaF", deltaF);
  graph.createHostRead("outDeltaH", deltaH);

  const auto batchSize=1;

  // test inputs calculated in harness
  boost::multi_array<double, 4>
    hActIn(boost::extents[batchSize][ySize][xSize][zChunk]);
  boost::multi_array<double, 4>
    hDeltaIn(boost::extents[batchSize][ySize][xSize][zChunk]);

  // outputs calculated by target code
  auto rawHActOutF = allocateHostMemoryForTensor(target, actF);
  auto rawHActOutH = allocateHostMemoryForTensor(target, actH);
  auto rawHActInF = allocateHostMemoryForTensor(target, actF);
  auto rawHActInH = allocateHostMemoryForTensor(target, actH);
  auto rawHDeltaOutF = allocateHostMemoryForTensor(target, deltaF);
  auto rawHDeltaOutH = allocateHostMemoryForTensor(target, deltaH);
  auto rawHDeltaInF = allocateHostMemoryForTensor(target, deltaF);
  auto rawHDeltaInH = allocateHostMemoryForTensor(target, deltaH);
  boost::multi_array<double, 4>
    hActOutF(boost::extents[batchSize][ySize][xSize][zChunk]);
  boost::multi_array<double, 4>
    hActOutH(boost::extents[batchSize][ySize][xSize][zChunk]);
  boost::multi_array<double, 4>
    hDeltaOutF(boost::extents[batchSize][ySize][xSize][zChunk]);
  boost::multi_array<double, 4>
    hDeltaOutH(boost::extents[batchSize][ySize][xSize][zChunk]);

  // reference results calculated in harness
  boost::multi_array<double, 4>
    hRefActOut(boost::extents[batchSize][ySize][xSize][zChunk]);
  boost::multi_array<double, 4>
    hRefDeltaOut(boost::extents[batchSize][ySize][xSize][zChunk]);

  //initialse hInF[][] to arbitrary values
  float val = -100.0;
  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned y = 0; y < ySize; ++y) {
      for (unsigned x = 0; x < xSize; ++x) {
        for (unsigned chan = 0; chan < zChunk; chan++) {
          hRefDeltaOut[b][y][x][chan]
            = hDeltaIn[b][y][x][chan] = val / 200;
           hActIn[b][y][x][chan] = val + 1000 * chan;
        }
        val += 7.01;
        if (val > 200)
          val -= 400;
      }
    }
  }

  for (auto n : {NonLinearityType::RELU,
                 NonLinearityType::SIGMOID,
                 NonLinearityType::TANH,
                 }) {
    //Check backward gradient calculations
    std::cerr << "Check nl type " << poplibs_test::asString(n) << "\n";
    //Check forward activation calculation
    hRefActOut = hActIn;
    poplibs_test::nonLinearity(n, hRefActOut);
    // build and run the target code
    auto fwdProg = Sequence();
    nonLinearity(graph, n, actF, fwdProg);
    nonLinearity(graph, n, actH, fwdProg);;
    Engine fwdEng(graph, fwdProg, options);
    fwdEng.load(device);
    copy(target, hActIn, FLOAT, rawHActInF.get());
    fwdEng.writeTensor("inF", rawHActInF.get());
    copy(target, hActIn, HALF, rawHActInH.get());
    fwdEng.writeTensor("inH", rawHActInH.get());
    fwdEng.run();
    fwdEng.readTensor("outF", rawHActOutF.get());
    fwdEng.readTensor("outH", rawHActOutH.get());
    copy(target, HALF, rawHActOutH.get(), hActOutH);
    copy(target, FLOAT, rawHActOutF.get(), hActOutF);

    BOOST_TEST(
      checkIsClose("outF", hActOutF, hRefActOut, TOL, FLOAT_ATOL));
    BOOST_TEST(
      checkIsClose("outH", hActOutH, hRefActOut, TOL, HALF_ATOL));

    hRefDeltaOut = hDeltaIn;
    poplibs_test::bwdNonLinearity(n, hActIn, hRefDeltaOut);
    // build and run the target code
    auto bwdProg = Sequence();
    auto deltaFF = nonLinearityInputGradient(graph, n, actF, deltaF, bwdProg);
    bwdProg.add(Copy(deltaFF, deltaF));
    auto deltaHH = nonLinearityInputGradient(graph, n, actH, deltaH, bwdProg);
    bwdProg.add(Copy(deltaHH, deltaH));
    Engine bwdEng(graph, bwdProg, options);
    bwdEng.load(device);
    copy(target, hActIn, FLOAT, rawHActInF.get());
    bwdEng.writeTensor("inF", rawHActInF.get());
    copy(target, hActIn, HALF, rawHActInH.get());
    bwdEng.writeTensor("inH", rawHActInH.get());
    copy(target, hDeltaIn, FLOAT, rawHDeltaInF.get());
    bwdEng.writeTensor("inDeltaF", rawHDeltaInF.get());
    copy(target, hDeltaIn, HALF, rawHDeltaInH.get());
    bwdEng.writeTensor("inDeltaH", rawHDeltaInH.get());
    bwdEng.run();
    bwdEng.readTensor("outDeltaF", rawHDeltaOutF.get());
    bwdEng.readTensor("outDeltaH", rawHDeltaOutH.get());
    copy(target, HALF, rawHDeltaOutH.get(), hDeltaOutH);
    copy(target, FLOAT, rawHDeltaOutF.get(), hDeltaOutF);


    BOOST_TEST(
      checkIsClose("deltaOutF", hDeltaOutF, hRefDeltaOut, TOL, FLOAT_ATOL));
    BOOST_TEST(
      checkIsClose("deltaOutH", hDeltaOutH, hRefDeltaOut, TOL, HALF_ATOL));
  }
}

BOOST_AUTO_TEST_CASE(NonLinearitySoftMax,
                    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.1))
                    *utf::tolerance<double>(fpc::percent_tolerance<double>(0.1))
                     ) {
  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();
  Graph graph(device);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);

  // support only 2D
  const auto nl = NonLinearityType::SOFTMAX;
  const unsigned batchSize = 2;
  const unsigned numChannels = 128;

  auto actF = graph.addVariable(FLOAT, {batchSize, numChannels}, "actF");
  auto actH = graph.addVariable(HALF, {batchSize, numChannels}, "actH");

  graph.createHostWrite("inF", actF);
  graph.createHostWrite("inH", actH);
  graph.createHostRead("outF", actF);
  graph.createHostRead("outH",actH);

  // arbitrary mappings
  mapTensorLinearly(graph, actF);
  mapTensorLinearly(graph, actH);

  auto rawHActOutF = allocateHostMemoryForTensor(target, actF);
  auto rawHActOutH = allocateHostMemoryForTensor(target, actH);
  auto rawHActInF = allocateHostMemoryForTensor(target, actF);
  auto rawHActInH = allocateHostMemoryForTensor(target, actH);

  boost::multi_array<double, 2>
    hActIn(boost::extents[batchSize][numChannels]),
    hActOutF(boost::extents[batchSize][numChannels]),
    hActOutH(boost::extents[batchSize][numChannels]);

  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned c = 0; c < numChannels; ++c) {
      double sample = (1.0 - 2 * (c&1)) * (1 + b) * 0.01 * c;
      hActIn[b][c] = sample;
    }
  }

  std::cerr << "Check nl type " << poplibs_test::asString(nl) << "\n";

  auto hActOut = hActIn;
  poplibs_test::nonLinearity(nl, hActOut);
  // build and run the target code
  auto fwdProg = Sequence();
  nonLinearity(graph, nl, actF, fwdProg);
  nonLinearity(graph, nl, actH, fwdProg);
  Engine fwdEng(graph, fwdProg, options);
  fwdEng.load(device);

  copy(target, hActIn, FLOAT, rawHActInF.get());
  fwdEng.writeTensor("inF", rawHActInF.get());
  copy(target, hActIn, HALF, rawHActInH.get());
  fwdEng.writeTensor("inH", rawHActInH.get());
  fwdEng.writeTensor("inF", rawHActInF.get());
  fwdEng.writeTensor("inH", rawHActInH.get());
  fwdEng.run();
  fwdEng.readTensor("outF", rawHActOutF.get());
  fwdEng.readTensor("outH", rawHActOutH.get());
  copy(target, HALF, rawHActOutH.get(), hActOutH);
  copy(target, FLOAT, rawHActOutF.get(), hActOutF);


  BOOST_TEST(
    checkIsClose("actOutF", hActOutF, hActOut, TOL, FLOAT_ATOL));
  BOOST_TEST(
    checkIsClose("actOutH", hActOutH, hActOut, TOL, HALF_ATOL));
}
