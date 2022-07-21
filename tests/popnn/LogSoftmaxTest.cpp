// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
// Simple test case for test log of softmax
//
#define BOOST_TEST_MODULE NonLinearityTest
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popnn/LogSoftmax.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popnn;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define TOL 0.1
#define FLOAT_ATOL 1e-20
#define HALF_ATOL 1e-7

void validateLogSoftmax(unsigned batchSize, unsigned numChannels) {
  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  auto actF = graph.addVariable(FLOAT, {batchSize, numChannels}, "actF");
  auto actH = graph.addVariable(HALF, {batchSize, numChannels}, "actH");

  // arbitrary mappings
  mapTensorLinearly(graph, actF);
  mapTensorLinearly(graph, actH);

  graph.createHostWrite("inF", actF);
  graph.createHostWrite("inH", actH);
  graph.createHostRead("outF", actF);
  graph.createHostRead("outH", actH);

  std::vector<std::pair<std::string, HostMemory>> tmap;
  Sequence uploadProg, downloadProg;

  auto rawHActF = allocateHostMemoryForTensor(actF, "actF", graph, uploadProg,
                                              downloadProg, tmap);
  auto rawHActH = allocateHostMemoryForTensor(actH, "actH", graph, uploadProg,
                                              downloadProg, tmap);

  boost::multi_array<double, 2> hActIn(boost::extents[batchSize][numChannels]),
      hOutRef(boost::extents[batchSize][numChannels]),
      hActOutF(boost::extents[batchSize][numChannels]),
      hActOutH(boost::extents[batchSize][numChannels]);

  // Reference computation
  for (unsigned b = 0; b < batchSize; ++b) {
    double maxBatch = std::numeric_limits<double>::lowest();
    for (unsigned c = 0; c < numChannels; ++c) {
      double sample = (1.0 - 2 * (c & 1)) * (1 + b) * 0.01 * c;
      hActIn[b][c] = sample;
      maxBatch = std::max(maxBatch, sample);
    }
    // compute sum of exponent
    double sum = 0.0;
    for (unsigned c = 0; c < numChannels; ++c) {
      sum += std::exp(hActIn[b][c] - maxBatch);
    }

    for (unsigned c = 0; c < numChannels; ++c) {
      hOutRef[b][c] = hActIn[b][c] - maxBatch - std::log(sum);
    }
  }

  // To test 1D
  if (batchSize == 1) {
    actF = actF.squeeze({0});
    actH = actH.squeeze({0});
  }

  // build and run the target code: non-inplace followed by in-place
  auto prog = Sequence();
  auto outF = popnn::logSoftmax(graph, actF, prog);
  auto outH = popnn::logSoftmax(graph, actH, prog);
  popnn::logSoftmaxInPlace(graph, actF, prog);
  popnn::logSoftmaxInPlace(graph, actH, prog);

  auto rawHOutF = allocateHostMemoryForTensor(outF, "outF", graph, uploadProg,
                                              downloadProg, tmap);
  auto rawHOutH = allocateHostMemoryForTensor(outH, "outH", graph, uploadProg,
                                              downloadProg, tmap);

  copy(target, hActIn, FLOAT, rawHActF.get());
  copy(target, hActIn, HALF, rawHActH.get());

  Engine fwdEng(graph, Sequence{uploadProg, prog, downloadProg});
  attachStreams(fwdEng, tmap);
  device.bind([&](const Device &d) { fwdEng.loadAndRun(d); });

  // inplace variant
  copy(target, FLOAT, rawHActF.get(), hActOutF);
  copy(target, HALF, rawHActH.get(), hActOutH);
  BOOST_TEST(checkIsClose("actOutF", hActOutF, hOutRef, TOL, FLOAT_ATOL));
  BOOST_TEST(checkIsClose("actOutH", hActOutH, hOutRef, TOL, HALF_ATOL));

  // non-inplace variant
  copy(target, FLOAT, rawHOutF.get(), hActOutF);
  copy(target, HALF, rawHOutH.get(), hActOutH);
  BOOST_TEST(checkIsClose("actOutF", hActOutF, hOutRef, TOL, FLOAT_ATOL));
  BOOST_TEST(checkIsClose("actOutH", hActOutH, hOutRef, TOL, HALF_ATOL));
}

BOOST_AUTO_TEST_CASE(logSoftmax_1D) { validateLogSoftmax(1, 100); }

BOOST_AUTO_TEST_CASE(logSoftmax_2D) { validateLogSoftmax(4, 100); }
