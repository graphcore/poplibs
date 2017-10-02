// Simple test case for IPU nonLinearity
//
#define BOOST_TEST_MODULE NonLinearityTest
#include <popnn/NonLinearity.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/TileMapping.hpp>
#include <popnn/codelets.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <iostream>
#include <poplib_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;
using namespace popnn;
using namespace poplib_test::util;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define TOL 0.1 //tolerance of 0.1%
#define ATOL 1e-30
BOOST_AUTO_TEST_CASE(NonLinearity,
                    *utf::tolerance<half>(fpc::percent_tolerance<half>(TOL))
                    *utf::tolerance<float>(fpc::percent_tolerance<float>(TOL))
                    *utf::tolerance<double>(fpc::percent_tolerance<double>(TOL))
                     ) {
  Graph graph(createIPUModelDevice());
  popnn::addCodelets(graph);
  //layer parameters

  const unsigned zNGroups = 1;
  const std::size_t zChunk = 1;
  const std::size_t ySize = 100;
  const std::size_t xSize = 30;
  auto actF = graph.addTensor("float", {1, zNGroups, ySize, xSize, zChunk},
                              "actF");
  auto actH = graph.addTensor("half", {1, zNGroups, ySize, xSize, zChunk},
                              "actH");
  auto deltaF = graph.addTensor("float", {1, zNGroups, ySize, xSize, zChunk},
                              "actF");
  auto deltaH = graph.addTensor("half", {1, zNGroups, ySize, xSize, zChunk},
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
  float hActInF[batchSize][ySize][xSize][zChunk];
  half  hActInH[batchSize][ySize][xSize][zChunk];
  float hDeltaInF[batchSize][ySize][xSize][zChunk];
  half  hDeltaInH[batchSize][ySize][xSize][zChunk];
  boost::multi_array<double, 4>
    hActIn(boost::extents[batchSize][ySize][xSize][zChunk]);
  boost::multi_array<double, 4>
    hDeltaIn(boost::extents[batchSize][ySize][xSize][zChunk]);

  // outputs calculated by target code
  auto rawHActOutF = allocateHostMemoryForTensor(actF);
  auto rawHActOutH = allocateHostMemoryForTensor(actH);
  auto rawHDeltaOutF = allocateHostMemoryForTensor(deltaF);
  auto rawHDeltaOutH = allocateHostMemoryForTensor(deltaH);
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
            = hDeltaInH[b][y][x][chan]
            = hDeltaInF[b][y][x][chan]
            = hDeltaIn[b][y][x][chan] = val / 20;
          hActInH[b][y][x][chan] = hActInF[b][y][x][chan] =
           hActIn[b][y][x][chan] = val + 1000 * chan;
        }
        val += 7.01;
        if (val > 200)
          val -= 400;
      }
    }
  }

  for (enum NonLinearityType n : {NON_LINEARITY_RELU,
                                  NON_LINEARITY_SIGMOID,
                                  NON_LINEARITY_TANH,
                                  }) {
    //Check backward gradient calculations
    if (n == NON_LINEARITY_SOFTMAX) {
      continue;
    }
    std::cerr<<"Check nl type "<< n << "\n";
    //Check forward activation calculation
    hRefActOut = hActIn;
    poplib_test::nonLinearity(n, hRefActOut);
    // build and run the target code
    auto fwdProg = Sequence();
    nonLinearity(graph, n, actF, fwdProg);
    nonLinearity(graph, n, actH, fwdProg);;
    Engine fwdEng(graph, fwdProg);

    fwdEng.writeTensor("inF", hActInF);
    fwdEng.writeTensor("inH", hActInH);
    fwdEng.run();
    fwdEng.readTensor("outF", rawHActOutF.get());
    fwdEng.readTensor("outH", rawHActOutH.get());
    copy("half", rawHActOutH.get(), hActOutH);
    copy("float", rawHActOutF.get(), hActOutF);

    BOOST_TEST(
      checkIsClose("hRefActOutF", hActOutF, hRefActOut, TOL, ATOL));
    BOOST_TEST(
      checkIsClose("hRefActOutH", hActOutH, hRefActOut, TOL, ATOL));

    hRefDeltaOut = hDeltaIn;
    poplib_test::bwdNonLinearity(n, hActIn, hRefDeltaOut);
    // build and run the target code
    auto bwdProg = Sequence();
    auto deltaFF = nonLinearityInputGradient(graph, n, actF, deltaF, bwdProg);
    bwdProg.add(Copy(deltaFF, deltaF));
    auto deltaHH = nonLinearityInputGradient(graph, n, actH, deltaH, bwdProg);
    bwdProg.add(Copy(deltaHH, deltaH));
    Engine bwdEng(graph, bwdProg);
    bwdEng.writeTensor("inF", hActInF);
    bwdEng.writeTensor("inH", hActInH);
    bwdEng.writeTensor("inDeltaF", hDeltaInF);
    bwdEng.writeTensor("inDeltaH", hDeltaInH);
    bwdEng.run();
    bwdEng.readTensor("outDeltaF", rawHDeltaOutF.get());
    bwdEng.readTensor("outDeltaH", rawHDeltaOutH.get());
    copy("half", rawHDeltaOutH.get(), hDeltaOutH);
    copy("float", rawHDeltaOutF.get(), hDeltaOutF);

    for (unsigned b = 0; b < batchSize; ++b) {
      for (unsigned y = 0; y < xSize; ++y) {
        for (unsigned x = 0; x < xSize; ++x) {
          for (unsigned chan = 0; chan < zChunk; chan++) {
            BOOST_TEST(hDeltaOutF[b][y][x][chan]
                       == (float)hRefDeltaOut[b][y][x][chan]);
            BOOST_TEST((float)hDeltaOutH[b][y][x][chan]
                       == (float)hRefDeltaOut[b][y][x][chan]);
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(NonLinearitySoftMax,
                    *utf::tolerance<half>(fpc::percent_tolerance<half>(1))
                    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.1))
                    *utf::tolerance<double>(fpc::percent_tolerance<double>(0.1))
                     ) {
  Graph graph(createIPUModelDevice());
  popnn::addCodelets(graph);
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);

  // support only 2D
  const auto nl = NON_LINEARITY_SOFTMAX;
  const unsigned batchSize = 2;
  const unsigned numChannels = 128;

  auto actF = graph.addTensor("float", {batchSize, numChannels}, "actF");
  auto actH = graph.addTensor("half", {batchSize, numChannels}, "actH");

  graph.createHostWrite("inF", actF);
  graph.createHostWrite("inH", actH);
  graph.createHostRead("outF", actF);
  graph.createHostRead("outH",actH);

  // arbitrary mappings
  mapTensorLinearly(graph, actF);
  mapTensorLinearly(graph, actH);

  float hActInF[batchSize][numChannels];
  half hActInH[batchSize][numChannels];
  float hActOutF[batchSize][numChannels];
  half hActOutH[batchSize][numChannels];

  boost::multi_array<double, 2>
    hActIn(boost::extents[batchSize][numChannels]);

  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned c = 0; c < numChannels; ++c) {
      double sample = (1.0 - 2 * (c&1)) * (1 + b) * 0.01 * c;
      hActInH[b][c] = hActInF[b][c] =  hActIn[b][c] = sample;
    }
  }

  std::cerr<<"Check nl type "<< nl << "\n";

  auto hActOut = hActIn;
  poplib_test::nonLinearity(nl, hActOut);
  // build and run the target code
  auto fwdProg = Sequence();
  nonLinearity(graph, nl, actF, fwdProg);
  nonLinearity(graph, nl, actH, fwdProg);
  Engine fwdEng(graph, fwdProg);

  fwdEng.writeTensor("inF", hActInF);
  fwdEng.writeTensor("inH", hActInH);
  fwdEng.run();
  fwdEng.readTensor("outF", hActOutF);
  fwdEng.readTensor("outH", hActOutH);

  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned c = 0; c < numChannels; ++c) {
      BOOST_TEST(hActOutF[b][c] == hActOut[b][c]);
      BOOST_TEST((float)hActOutH[b][c] == hActOut[b][c]);
    }
  }
}
