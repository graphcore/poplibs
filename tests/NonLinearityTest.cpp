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
#include <poplib_test/NonLinearity.hpp>
#include <iostream>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;
using namespace popnn;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(NonLinearity,
                    *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.1))
                    *utf::tolerance<double>(fpc::percent_tolerance<double>(0.1))
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
  float hActOutF[batchSize][ySize][xSize][zChunk];
  half  hActOutH[batchSize][ySize][xSize][zChunk];
  float hDeltaOutF[batchSize][ySize][xSize][zChunk];
  half  hDeltaOutH[batchSize][ySize][xSize][zChunk];

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
                                  NON_LINEARITY_TANH}) {
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
    fwdEng.readTensor("outF", hActOutF);
    fwdEng.readTensor("outH", hActOutH);

    for (unsigned b = 0; b < batchSize; ++b) {
      for (unsigned y = 0; y < xSize; ++y) {
        for (unsigned x = 0; x < xSize; ++x) {
          for (unsigned chan = 0; chan < zChunk; chan++) {
            BOOST_TEST(hActOutF[b][y][x][chan]
                       == (float)hRefActOut[b][y][x][chan]);
            BOOST_TEST(hActOutH[b][y][x][chan]
                       == (half)hRefActOut[b][y][x][chan]);
          }
        }
      }
    }

    //Check backward gradient calculations

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
    bwdEng.readTensor("outDeltaF", hDeltaOutF);
    bwdEng.readTensor("outDeltaF", hDeltaOutH);

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
