// Simple test case for IPU maxPool and maxPoolBackward
//
#define BOOST_TEST_MODULE MaxPoolTest
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <popnn/MaxPool.hpp>
#include <popnn/ActivationMapping.hpp>
#include <popnn/Net.hpp>
#include <iostream>
using namespace poplar;
using namespace poplar::program;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(MaxPool,
                     *utf::tolerance<float>(
                     fpc::percent_tolerance<float>(0.0))) {
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env, createIPUModelDevice());

  //layer parameters
  const unsigned kernelSize = 3;
  const unsigned padding = 1;
  const unsigned stride = 2;
  const unsigned zNGroups = 1;
  const std::size_t zChunk = 1;
  const std::size_t prevSize = 100;
  const std::size_t nextSize = (prevSize + padding) / stride;
  auto actIn = graph.addTensor("float", {1,
                                         zNGroups, prevSize, prevSize, zChunk},
                               "actIn");
  auto actOut = graph.addTensor("float", {1,
                                          zNGroups, nextSize, nextSize, zChunk},
                                "actOut");
  auto errIn = graph.addTensor("float", {1,
                                         zNGroups, nextSize, nextSize, zChunk},
                               "errIn");
  auto errOut = graph.addTensor("float", {1,
                                          zNGroups, prevSize, prevSize, zChunk},
                                "errOut");
  // arbitraray mappings
  mapActivations(graph, actIn);
  mapActivations(graph, actOut);
  mapActivations(graph, errIn);
  mapActivations(graph, errOut);

  auto maxPoolFwd = maxpool::maxPool(graph, kernelSize, stride, padding,
                                     actIn, actOut);
  auto maxPoolBwd = maxpool::maxPoolBackward(graph, kernelSize, stride, padding,
                                     actIn, actOut, errIn, errOut);
  // test inputs calculated in harness
  float hIn[prevSize][prevSize][zChunk];
  float hErrsIn[nextSize][nextSize][zChunk];

  // outputs calculated by target code
  float hActOut[nextSize][nextSize][zChunk];
  float hErrsOut[prevSize][prevSize][zChunk];

  // reference results calculated in harness
  float hRefActOut[nextSize][nextSize][zChunk];
  float hRefErrsOut[prevSize][prevSize][zChunk];

  //initialse hRefErrsOut[][] to zeros and hIn to arbitrary values
  float val = -100.0;
  for (unsigned y = 0; y < prevSize; ++y) {
    for (unsigned x = 0; x < prevSize; ++x) {
      for (unsigned chan = 0; chan < zNGroups; chan++) {
        hIn[y][x][chan] = val + 1000 * chan;
        hRefErrsOut[y][x][chan] = 0;
      }
      val += 7.01;
      if (val > 200)
        val -= 400;
    }
  }

  //initialise arbitrary errors
  float err = 1.0;
  for (unsigned y = 0; y < nextSize; ++y) {
    for (unsigned x = 0; x < nextSize; ++x) {
      for (unsigned chan = 0; chan < zNGroups; chan++) {
        hErrsIn[y][x][chan] = err;
      }
      err *= 10;
      if (err >= 1e5)
        err = 1.0;
    }
  }

  //forward maxpool calculation
  for (unsigned chan = 0; chan < zNGroups; chan++) {
    for (unsigned y = 0; y < nextSize; ++y) {
      for (unsigned x = 0; x < nextSize; ++x) {
        float runningMax = -std::numeric_limits<float>::infinity();
        for (unsigned j = 0; j < kernelSize; j++) {
          if (y * stride + j < padding)
            continue;
          if (y * stride + j - padding >= prevSize)
            continue;
          for (unsigned i = 0; i < kernelSize; i++) {
            if (x * stride + i < padding)
              continue;
            if (x * stride + i - padding >= prevSize)
              continue;
            auto yPrevPos = y * stride + j - padding;
            auto xPrevPos = x * stride + i - padding;
            if (runningMax < hIn[yPrevPos][xPrevPos][chan])
              runningMax = hIn[yPrevPos][xPrevPos][chan];
          }
        }
        hRefActOut[y][x][chan] = runningMax;
      }
    }
  }

  //maxpool backwards pass
  for (unsigned chan = 0; chan < zNGroups; chan++) {
    for (unsigned y = 0; y < nextSize; ++y) {
      for (unsigned x = 0; x < nextSize; ++x) {
        for (unsigned j = 0; j < kernelSize; j++) {
          if (y * stride + j < padding)
            continue;
          if (y * stride + j - padding >= prevSize)
            continue;
          for (unsigned i = 0; i < kernelSize; i++) {
            if (x * stride + i < padding)
              continue;
            if (x * stride + i - padding >= prevSize)
              continue;
            auto yPrevPos = y * stride + j - padding;
            auto xPrevPos = x * stride + i - padding;
            if (hRefActOut[y][x][chan] ==  hIn[yPrevPos][xPrevPos][chan])
              hRefErrsOut[yPrevPos][xPrevPos][chan] += hErrsIn[y][x][chan];
          }
        }
      }
    }
  }

  // build and run the target code
  auto prog = Sequence(Copy(actIn, hIn),
                       maxPoolFwd,
                       Copy(&hActOut, actOut),
                       Copy(errIn, &hErrsIn),
                       maxPoolBwd,
                       Copy(&hErrsOut[0], errOut));
  Engine eng(graph, prog);
  eng.run();

  //Check forward activation calculation
  for (unsigned chan = 0; chan < zNGroups; chan++) {
    for (unsigned y = 0; y < nextSize; ++y) {
      for (unsigned x = 0; x < nextSize; ++x) {
        BOOST_TEST(hActOut[y][x][chan] == hRefActOut[y][x][chan]);
      }
    }
  }

  // Check backward error propagation
  for (unsigned chan = 0; chan < zNGroups; chan++) {
    for (unsigned y = 0; y < prevSize; ++y) {
      for (unsigned x = 0; x < prevSize; ++x) {
        BOOST_TEST(hErrsOut[y][x][chan] == hRefErrsOut[y][x][chan]);
      }
    }
  }
}
