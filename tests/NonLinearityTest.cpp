// Simple test case for IPU nonLinearity
//
#define BOOST_TEST_MODULE NonLinearityTest
#include <popnn/NonLinearity.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <popnn/MaxPool.hpp>
#include <popnn/ActivationMapping.hpp>
#include <popnn/Net.hpp>
#include <iostream>

using namespace poplar;
using namespace poplar::program;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

static float sigmoid(float x)
{
  return (1. / (1. + exp(-x)));
}

BOOST_AUTO_TEST_CASE(NonLinearity,
                     *utf::tolerance<float>(
                     fpc::percent_tolerance<float>(0.0))) {
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env, createIPUModelDevice());

  //layer parameters

  const unsigned zNGroups = 1;
  const std::size_t zChunk = 1;
  const std::size_t ySize = 100;
  const std::size_t xSize = 30;
  auto actF = graph.addTensor("float", {1, zNGroups, ySize, xSize, zChunk},
                              "actF");
  auto actH = graph.addTensor("half", {1, zNGroups, ySize, xSize, zChunk},
                              "actH");

  // arbitraray mappings
  mapActivations(graph, actF);
  mapActivations(graph, actH);

  const auto nNL = NON_LINEARITY_SIGMOID+1;

  // test inputs calculated in harness
  float hInF[ySize][xSize][zChunk];
  half  hInH[ySize][xSize][zChunk];

  // outputs calculated by target code
  float hOutF[nNL][ySize][xSize][zChunk];
  half  hOutH[nNL][ySize][xSize][zChunk];

  // reference results calculated in harness
  float hRefF[nNL][ySize][xSize][zChunk];

  //initialse hInF[][] to arbitrary values
  float val = -100.0;
  for (unsigned y = 0; y < ySize; ++y) {
    for (unsigned x = 0; x < xSize; ++x) {
      for (unsigned chan = 0; chan < zNGroups; chan++) {
        hInF[y][x][chan] = val + 1000 * chan;
        hInH[y][x][chan] = hInF[y][x][chan];
      }
      val += 7.01;
      if (val > 200)
        val -= 400;
    }
  }

  //relu calculation
  for (unsigned chan = 0; chan < zNGroups; chan++) {
    for (unsigned y = 0; y < xSize; ++y) {
      for (unsigned x = 0; x < xSize; ++x) {
        auto hIn = hInF[y][x][chan];
        hRefF[NON_LINEARITY_NONE][y][x][chan] = hIn;
        hRefF[NON_LINEARITY_RELU][y][x][chan] = hIn > 0 ? hIn : 0;
        hRefF[NON_LINEARITY_SIGMOID][y][x][chan] = sigmoid(hIn);
      }
    }
  }

  // build and run the target code
  auto prog = Sequence();
  for (enum NonLinearityType n = NON_LINEARITY_NONE;
     n != NON_LINEARITY_SIGMOID;
     n=(enum NonLinearityType)((unsigned int)n+1)) {
    prog.add(Copy(actF, hInF));
    prog.add(fwdNonLinearity(graph, actF, n));
    prog.add(Copy(hOutF[n], actF));
    prog.add(Copy(actH, hInH));
    prog.add(fwdNonLinearity(graph, actH, n));
    prog.add(Copy(hOutH[n], actH));
  }
    Engine eng(graph, prog);
  eng.run();

  //Check forward activation calculation
  for (enum NonLinearityType n = NON_LINEARITY_NONE;
       n != NON_LINEARITY_SIGMOID;
       n=(enum NonLinearityType)((unsigned int)n+1)) {

    for (unsigned chan = 0; chan < zNGroups; chan++) {
      for (unsigned y = 0; y < xSize; ++y) {
        for (unsigned x = 0; x < xSize; ++x) {
          BOOST_TEST(hOutF[n][y][x][chan] == hRefF[n][y][x][chan]);
          BOOST_TEST(hOutH[n][y][x][chan] == hRefF[n][y][x][chan]);
        }
      }
    }
  }
}
