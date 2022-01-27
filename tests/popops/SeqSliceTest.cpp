// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DynamicSliceTest
#include <boost/multi_array.hpp>
#include <boost/test/framework.hpp>
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <poplar/Engine.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/print.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/SequenceSlice.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <sstream>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;
using namespace poplibs_test::util;
using namespace poplar_test;

using poplibs_support::toString;

void variableslice(const std::vector<uint32_t> &nElem,
                   const std::vector<uint32_t> &srcIndices,
                   const std::vector<uint32_t> &dstIndices,
                   bool planAsEmbedding, bool slicerZeros,
                   unsigned numTiles = 1) {
  BOOST_CHECK(srcIndices.size() == dstIndices.size());
  BOOST_CHECK(srcIndices.size() == nElem.size());

  // This test should pass with large T - but graph construction becomes
  // slow (a couple of minutes for T=1024)

  const auto elemType = HALF;
  const auto T = numTiles; // tiles
  const auto D = 1501;     // dictionary size
  const auto E = 8;        // embedding size

  std::set<unsigned> usedDstIndices;
  unsigned maxDstIdx = 0;
  for (unsigned i = 0; i != nElem.size(); ++i) {
    if (maxDstIdx < dstIndices[i] + nElem[i])
      maxDstIdx = dstIndices[i] + nElem[i];
    // Multiple writes to the same output location gives an undefined result,
    // so ensure we don't try to test for it.
    for (unsigned j = 0; j != nElem[i]; ++j) {
      BOOST_CHECK(usedDstIndices.insert(dstIndices[i] + j).second);
    }
    BOOST_CHECK(dstIndices[i] + nElem[i] < D);
  }

  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  // Source tensor layout uses multislice planning.
  const auto options = OptionFlags();
  auto plan = SlicePlan();
  if (planAsEmbedding) {
    plan = embedding::plan(graph, elemType, D, E, {srcIndices.size()}, options);
  }
  // Map the tensor carefully to ensure balance and minimise edge pointers
  auto t = createSliceableTensor(graph, elemType, {D, E}, {0}, {1}, plan,
                                 options, "t");
  Sequence prog;
  auto s = graph.clone(t.slice(0, maxDstIdx));

  auto tN = graph.addConstant(UNSIGNED_INT, {nElem.size()}, nElem.data(), "tN");
  auto tSrcOffset = graph.addConstant(UNSIGNED_INT, {srcIndices.size()},
                                      srcIndices.data(), "tSrcOffset");
  auto tDstOffset = graph.addConstant(UNSIGNED_INT, {dstIndices.size()},
                                      dstIndices.data(), "tDstOffset");
  graph.setTileMapping(tN, 0);
  graph.setTileMapping(tSrcOffset, 0);
  graph.setTileMapping(tDstOffset, 0);

  sequenceSlice(graph, t, s, tN, tSrcOffset, tDstOffset, slicerZeros, prog,
                "MultisliceTest");

  BOOST_CHECK_EQUAL(s.rank(), t.rank());
  BOOST_CHECK_EQUAL(s.dim(1), E);

  float initialOutputValue = -1.5;
  graph.createHostWrite("in", t, true);
  graph.createHostWrite("out", s, true);
  graph.createHostRead("out", s, true);
  std::vector<float> hIn(t.numElements());
  std::vector<float> hOut(s.numElements(), initialOutputValue);
  std::iota(hIn.begin(), hIn.end(), 1u);
  if (elemType == HALF) {
    // Keep half values in the range where we can retain their precision
    for (auto &e : hIn) {
      if (e >= 1024) {
        // Deliberately not 1024
        e = std::fmod(e, 1000);
        // don't let them become zero to keep checking simple
        if (e == 0)
          e = 0.5;
      }
    }
  }
  OptionFlags engineOptions{{"showExecutionSteps", "true"},
                            {"showVarStorage", "true"}};
  // Engine creation will fail for non-cpu targets if many edge pointers or
  // significant exchange is required; this should not happen if
  // createSliceableTensor() has given a good layout
  auto &target = graph.getTarget();
  unsigned tSizeInBytes = t.numElements() * target.getTypeSize(elemType);
  unsigned sSizeInBytes = s.numElements() * target.getTypeSize(elemType);
  std::vector<unsigned char> tByteBuf(tSizeInBytes);
  std::vector<unsigned char> sByteBuf(sSizeInBytes);
  copy(target, hOut.data(), s.numElements(), elemType, sByteBuf.data());
  copy(target, hIn.data(), t.numElements(), elemType, tByteBuf.data());

  const auto dir = TempDir::create();
  Engine eng(graph, prog, {});
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("out", sByteBuf.data(), sByteBuf.data() + sByteBuf.size());
    eng.writeTensor("in", tByteBuf.data(), tByteBuf.data() + tByteBuf.size());
    eng.run();
    eng.readTensor("out", sByteBuf.data(), sByteBuf.data() + sByteBuf.size());
  });
  copy(target, elemType, sByteBuf.data(), hOut.data(), s.numElements());
  unsigned numNonZero = 0, numInitValues = 0;
  unsigned outIdx = 0;
  for (const auto &e : hOut) {
    BOOST_TEST_MESSAGE("MSlice Output[" << outIdx++ << "] = " << e);
    if (e != 0)
      ++numNonZero;
    if (e == initialOutputValue)
      ++numInitValues;
  }
  unsigned numExpectedWrites = 0;
  for (unsigned i = 0; i != srcIndices.size(); ++i) {
    auto d = srcIndices[i];
    numExpectedWrites += nElem[i] * E;
    for (unsigned elem = 0; elem != E * nElem[i]; ++elem) {
      unsigned expected = hIn[d * E + elem];

      BOOST_CHECK_EQUAL(hOut[dstIndices[i] * E + elem], expected);
    }
  }

  if (slicerZeros)
    BOOST_CHECK_EQUAL(numExpectedWrites, numNonZero);
  if (!slicerZeros)
    BOOST_CHECK_EQUAL(numExpectedWrites, hOut.size() - numInitValues);
}

BOOST_AUTO_TEST_CASE(RegionSizeVariation) {
  variableslice({3, 2, 1, 0, 1, 2, 3, 4, 2, 2},
                {2, 0, 2, 1, 80, 70, 60, 50, 40, 30},
                {20, 30, 40, 50, 60, 70, 80, 90, 25, 35}, true, true, 1);
  variableslice({3, 2, 1, 0, 1, 2, 3, 4, 2, 2},
                {2, 0, 2, 1, 80, 70, 60, 50, 40, 30},
                {20, 30, 40, 50, 60, 70, 80, 90, 25, 35}, true, false, 2);
  variableslice({3, 2, 1, 0, 1, 2, 3, 4, 2, 2},
                {2, 0, 2, 1, 80, 70, 60, 50, 40, 30},
                {20, 30, 40, 50, 60, 70, 80, 90, 25, 35}, true, true, 4);
}

BOOST_AUTO_TEST_CASE(MixedParams) {
  variableslice({3, 2, 1, 0, 1, 2, 3, 4, 2, 2},
                {2, 0, 2, 1, 80, 70, 60, 50, 40, 30},
                {20, 30, 40, 50, 60, 70, 80, 90, 25, 35}, true, false);
}

BOOST_AUTO_TEST_CASE(VariableSlice5) {
  variableslice({1, 1, 1, 1, 1}, {100, 0, 50, 48, 49}, {0, 1, 2, 3, 4}, false,
                false);
  variableslice({1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}, {100, 0, 50, 48, 49}, false,
                true);
}

BOOST_AUTO_TEST_CASE(VariableSlice2) {
  variableslice({1, 1}, {100, 0}, {0, 1}, false, false);
  variableslice({1, 1}, {0, 1}, {100, 0}, false, true);
}

BOOST_AUTO_TEST_CASE(VariableSlice10) {
  variableslice({1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {2, 1, 3, 4, 80, 70, 60, 50, 40, 30},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, false, false);
  variableslice({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                {2, 1, 3, 4, 80, 70, 60, 50, 40, 30}, false, true);
}

BOOST_AUTO_TEST_CASE(VariableSlice5_AsEmbedding) {
  variableslice({1, 1, 1, 1, 1}, {100, 0, 50, 48, 49}, {0, 1, 2, 3, 4}, true,
                false);
  variableslice({1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}, {100, 0, 50, 48, 49}, true,
                true);
}

BOOST_AUTO_TEST_CASE(VariableSlice2_AsEmbedding) {
  variableslice({1, 1}, {100, 0}, {0, 1}, true, false);
  variableslice({1, 1}, {0, 1}, {100, 0}, true, true);
}

BOOST_AUTO_TEST_CASE(VariableSlice10_AsEmbedding) {
  variableslice({1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {2, 1, 3, 4, 80, 70, 60, 50, 40, 30},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, true, false);
  variableslice({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                {2, 1, 3, 4, 80, 70, 60, 50, 40, 30}, true, true);
}

BOOST_AUTO_TEST_CASE(VariableSizes) {
  variableslice({5, 4, 3, 2, 1}, {100, 90, 80, 70, 50}, {10, 20, 30, 40, 50},
                false, false);
  variableslice({1, 2, 3, 4, 5}, {100, 90, 80, 70, 50}, {1, 2, 4, 7, 11}, true,
                false);
}

BOOST_AUTO_TEST_CASE(LargeVariableSizes) {
  variableslice({55, 44, 33, 22, 11}, {1000, 901, 800, 701, 500},
                {100, 200, 301, 401, 501}, false, false);
  variableslice({11, 22, 33, 44, 55}, {100, 900, 800, 700, 500},
                {100, 201, 402, 56, 111}, false, true);
}
