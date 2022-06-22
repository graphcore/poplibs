// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DynamicSliceTest
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/test/framework.hpp>
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <poplar/Engine.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <poplar/Quarter.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/print.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Fill.hpp>
#include <popops/Operation.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <pva/pva.hpp>
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

constexpr bool useDSMapper = true;

#define NUM_DIMS 3

struct TestData {
  std::vector<size_t> tDims, sDims;
  boost::multi_array<float, 3> hInit; // fullsized initialiser
  boost::multi_array<float, 3> hSub;  // subTensor, either in or out
  boost::multi_array<float, 3> hUpdateOut;
  TestData(std::vector<size_t> t, std::vector<size_t> s,
           const std::vector<std::vector<std::vector<float>>> &initialData)
      : tDims(t), sDims(s) {
    assert(t.size() == 3);
    assert(s.size() == 3);
    hInit.resize(boost::extents[t[0]][t[1]][t[2]]);
    hSub.resize(boost::extents[s[0]][s[1]][s[2]]);
    hUpdateOut.resize(boost::extents[t[0]][t[1]][t[2]]);
    for (unsigned i = 0; i != initialData.size(); ++i) {
      for (unsigned j = 0; j != initialData[i].size(); ++j) {
        for (unsigned k = 0; k != initialData[i][j].size(); ++k) {
          hInit[i][j][k] = initialData[i][j][k];
        }
      }
    }
  }
};

// Small 3 test data
static const unsigned dimA = 3, dimB = 4, dimC = 2;
static std::vector<size_t> smallTestShape = {dimA, dimB, dimC};
std::vector<std::vector<std::vector<float>>> smallTestData = {
    {{111, 112}, {121, 122}, {131, 132}, {141, 142}},
    {{211, 212}, {221, 222}, {231, 232}, {241, 242}},
    {{311, 312}, {321, 322}, {331, 332}, {341, 342}}};

// long delay data
#define LONG_OUTER 32
#define MAX_DELAY 200
#define ELEM_PER_TAP 4
static std::vector<size_t> delayTestShape = {LONG_OUTER, MAX_DELAY,
                                             ELEM_PER_TAP};
std::vector<std::vector<std::vector<float>>> GenDelayData() {
  std::vector<std::vector<std::vector<float>>> result;
  result.reserve(LONG_OUTER);
  for (unsigned i = 0; i != LONG_OUTER; ++i) {
    result.emplace_back();
    for (unsigned j = 0; j != MAX_DELAY; ++j) {
      result[i].emplace_back();
      for (unsigned k = 0; k != ELEM_PER_TAP; ++k)
        result[i][j].emplace_back((1 + 3 * i + j) + k * (1.0 / ELEM_PER_TAP));
    }
  }
  return result;
}

// map t's specified dimension across tiles
static void MapAcrossTiles(Graph &graph, size_t tilesPerIPU, const Tensor &t) {
  auto nTilesForT = std::min(t.dim(0), tilesPerIPU);
  auto elemPerSlice = t.numElements() / t.dim(0);
  Graph::TileToTensorMapping map;
  for (unsigned a = 0; a != nTilesForT; ++a) {
    std::vector<Interval> submap;
    auto iBegin = a * elemPerSlice;
    {
      auto iEnd =
          (a == nTilesForT - 1) ? t.numElements() : iBegin + elemPerSlice;
      auto interval = Interval(iBegin, iEnd);
      submap.emplace_back(interval);
      map.emplace_back(submap);
    }
  }
  graph.setTileMapping(t, map);
}

static boost::multi_array<float, 3>
refSlice(const std::vector<size_t> &sShape,
         const boost::multi_array<float, 3> &t,
         const std::vector<size_t> &offsets) {
  auto tShape = t.shape();
  boost::multi_array<float, 3> result(
      boost::extents[sShape[0]][sShape[1]][sShape[2]]);
  for (unsigned a = 0; a != sShape[0]; ++a) {
    for (unsigned b = 0; b != sShape[1]; ++b) {
      for (unsigned c = 0; c != sShape[2]; ++c) {
        auto refA = (offsets[0] + a) % tShape[0];
        auto refB = (offsets[1] + b) % tShape[1];
        auto refC = (offsets[2] + c) % tShape[2];
        auto value = t[refA][refB][refC];
        result[a][b][c] = value;
      }
    }
  }
  return result;
}

static boost::multi_array<float, 3>
refUpdate(const boost::multi_array<float, 3> &t,
          const boost::multi_array<float, 3> &s,
          const std::vector<size_t> &offsets) {
  auto tShape = t.shape();
  auto sShape = s.shape();
  boost::multi_array<float, 3> result(
      boost::extents[tShape[0]][tShape[1]][tShape[2]]);
  result = t;
  for (unsigned a = 0; a != sShape[0]; ++a) {
    for (unsigned b = 0; b != sShape[1]; ++b) {
      for (unsigned c = 0; c != sShape[2]; ++c) {
        auto refA = (offsets[0] + a) % tShape[0];
        auto refB = (offsets[1] + b) % tShape[1];
        auto refC = (offsets[2] + c) % tShape[2];
        auto value = s[a][b][c];
        result[refA][refB][refC] = value;
      }
    }
  }
  return result;
}

static void checkResult(const boost::multi_array<float, 3> &m,
                        const boost::multi_array<float, 3> &ref) {
  auto shape = m.shape();

  std::stringstream ss;
  for (unsigned a = 0; a != shape[0]; ++a) {
    ss << "[" << a << "] {";
    for (unsigned b = 0; b != shape[1]; ++b) {
      std::string sep = "";
      ss << "{";
      for (unsigned c = 0; c != shape[2]; ++c) {
        auto result = m[a][b][c];
        auto refResult = ref[a][b][c];
        ss << sep << result << " == " << refResult;
        sep = ", ";

        BOOST_CHECK_EQUAL(result, refResult);
      }
      ss << "}";
    }
    ss << "}";
  }
  BOOST_TEST_MESSAGE(ss.str());
}

// Check dynamicSliceND() extracts \a sliceSizes elements from the \a sliceDims
// dimensions for all possible offsets.
void sliceTestND(unsigned tilesPerIPU, const std::vector<size_t> &testShape,
                 const std::vector<std::vector<std::vector<float>>> &testBase,
                 const std::vector<std::size_t> &sliceDims,
                 const std::vector<std::size_t> &sliceSizes,
                 bool withOutput = false,
                 std::size_t maxTestedOffsetsPerDim = 25) {
  BOOST_TEST_MESSAGE(
      "Test " << boost::unit_test::framework::current_test_case().p_name);

  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<size_t> t1Shape = testShape;
  auto tWantedOffsets =
      graph.addVariable(UNSIGNED_INT, {sliceDims.size()}, "wantedOffsets");
  graph.setTileMapping(tWantedOffsets, 0);

  Tensor t1;
  if (useDSMapper) {
    // Test with the old test-specific allocator
    t1 = graph.addVariable(FLOAT, t1Shape, "t1");
    BOOST_TEST_MESSAGE("Created tensor t1: " << t1);
    MapAcrossTiles(graph, tilesPerIPU, t1);
  } else {
    // Test with the new reference tensor allocator.
    t1 = createSliceableTensor(graph, FLOAT, testShape, sliceDims, sliceSizes,
                               2, "t1");
  }
  BOOST_TEST_MESSAGE("t1 is " << t1 << " mapping "
                              << toString(graph.getTileMapping(t1)));

  auto prog = Sequence();
  poplar::Tensor tOut;
  if (withOutput) {
    tOut = createSliceTensor(graph, t1, sliceDims, sliceSizes, 1, "tOut")
               .squeeze({0});
    dynamicSliceWithOutput(graph, tOut, t1, tWantedOffsets, sliceDims,
                           sliceSizes, prog, "DSND");
  } else {
    tOut = dynamicSlice(graph, t1, tWantedOffsets, sliceDims, sliceSizes, prog,
                        "DSND");
  }

  const auto tOutShape = tOut.shape();
  BOOST_TEST_MESSAGE("output tensor is "
                     << tOut << " mapping "
                     << toString(graph.getTileMapping(tOut)));

  // Check output Tensor shape is correct
  std::vector<size_t> wantedShape = t1.shape();
  for (unsigned i = 0; i != sliceDims.size(); ++i) {
    wantedShape[sliceDims[i]] = sliceSizes[i];
  }
  for (unsigned d = 0; d != t1.rank(); ++d) {
    auto expectedSize = wantedShape[d] ? wantedShape[d] : t1.dim(d);
    BOOST_CHECK_EQUAL(tOutShape[d], expectedSize);
  }

  graph.createHostWrite("in", t1);
  graph.createHostWrite("selector", tWantedOffsets);
  graph.createHostRead("out", tOut);

  BOOST_TEST_MESSAGE("Creating engine");
  const auto dir = TempDir::create();
  Engine eng(graph, prog,
             {{"autoReport.outputGraphProfile", "true"},
              {"autoReport.directory", dir.getPath()}});
  device.bind([&](const Device &d) {
    eng.load(d);

    TestData testData(t1Shape, wantedShape, testBase);

    eng.writeTensor("in", testData.hInit.data(),
                    testData.hInit.data() + t1.numElements());

    std::vector<unsigned> nOffsets(t1.rank(), 1);
    for (auto dim : sliceDims) {
      nOffsets[dim] = std::min(t1.dim(dim), maxTestedOffsetsPerDim);
    }
    assert(t1.rank() == NUM_DIMS);
    for (unsigned sliceA = 0; sliceA != nOffsets[0]; ++sliceA) {
      for (unsigned sliceB = 0; sliceB != nOffsets[1]; ++sliceB) {
        for (unsigned sliceC = 0; sliceC != nOffsets[2]; ++sliceC) {
          unsigned offsets[NUM_DIMS] = {sliceA, sliceB, sliceC};
          unsigned hOffsets[NUM_DIMS];
          for (unsigned i = 0; i != sliceDims.size(); ++i) {
            hOffsets[i] = offsets[sliceDims[i]];
          }
          std::vector<size_t> checkOffsets = {{sliceA, sliceB, sliceC}};
          eng.writeTensor("selector", hOffsets, &hOffsets[sliceDims.size()]);
          for (unsigned i = 0; i != testData.hUpdateOut.num_elements(); ++i)
            testData.hUpdateOut.data()[i] = 0.0;
          BOOST_TEST_MESSAGE("Engine run " << toString(checkOffsets));
          eng.run();
          eng.readTensor("out", testData.hSub.data(),
                         testData.hSub.data() + tOut.numElements());
          boost::multi_array<float, 3> refResult =
              refSlice(wantedShape, testData.hInit, checkOffsets);
          checkResult(testData.hSub, refResult);
        }
      }
    }
  });
}

static void subTestSmallSlice(unsigned tilesPerIPU,
                              const std::vector<std::size_t> &sliceDims,
                              const std::vector<std::size_t> &sliceSizes,
                              bool withOutput = false) {
  sliceTestND(tilesPerIPU, smallTestShape, smallTestData, sliceDims, sliceSizes,
              withOutput);
}

BOOST_AUTO_TEST_SUITE(SingleDim)

// Test empty slice
BOOST_AUTO_TEST_CASE(Slice_Empty) { subTestSmallSlice(5, {}, {}); }
BOOST_AUTO_TEST_CASE(Slice_Empty_With_Output) { subTestSmallSlice(5, {}, {}); }

// Test slicing of a single dimension
BOOST_AUTO_TEST_CASE(Slice_5_0_1) { subTestSmallSlice(5, {0}, {1}); }
BOOST_AUTO_TEST_CASE(Slice_5_0_2) { subTestSmallSlice(5, {0}, {2}); }
BOOST_AUTO_TEST_CASE(Slice_5_1_1) { subTestSmallSlice(5, {1}, {1}); }
BOOST_AUTO_TEST_CASE(Slice_5_1_2) { subTestSmallSlice(5, {1}, {2}); }
BOOST_AUTO_TEST_CASE(Slice_5_2_1) { subTestSmallSlice(5, {2}, {1}); }
BOOST_AUTO_TEST_CASE(Slice_5_2_2) { subTestSmallSlice(5, {2}, {2}); }
BOOST_AUTO_TEST_CASE(Slice_5_0_1_With_Output) {
  subTestSmallSlice(5, {0}, {1}, true);
}
BOOST_AUTO_TEST_CASE(Slice_5_0_2_With_Output) {
  subTestSmallSlice(5, {0}, {2}, true);
}
BOOST_AUTO_TEST_CASE(Slice_5_1_1_With_Output) {
  subTestSmallSlice(5, {1}, {1}, true);
}
BOOST_AUTO_TEST_CASE(Slice_5_1_2_With_Output) {
  subTestSmallSlice(5, {1}, {2}, true);
}
BOOST_AUTO_TEST_CASE(Slice_5_2_1_With_Output) {
  subTestSmallSlice(5, {2}, {1}, true);
}
BOOST_AUTO_TEST_CASE(Slice_5_2_2_With_Output) {
  subTestSmallSlice(5, {2}, {2}, true);
}

BOOST_AUTO_TEST_SUITE_END()

// Multidimensional slicing
BOOST_AUTO_TEST_SUITE(MultiDim)

// dimensions 1 & 2
BOOST_AUTO_TEST_CASE(ND_1_1_0) { subTestSmallSlice(5, {0, 1}, {1, 1}); }
BOOST_AUTO_TEST_CASE(ND_1_1_0_With_Output) {
  subTestSmallSlice(5, {0, 1}, {1, 1}, true);
}
// all 3 dimensions
BOOST_AUTO_TEST_CASE(ND_1_1_1) { subTestSmallSlice(5, {0, 1, 2}, {1, 1, 1}); }
BOOST_AUTO_TEST_CASE(ND_1_1_1_With_Output) {
  subTestSmallSlice(5, {0, 1, 2}, {1, 1, 1}, true);
}
// dimensions 0 and 2, producing 2xdimBx2 output
BOOST_AUTO_TEST_CASE(ND_2_0_2) { subTestSmallSlice(5, {0, 2}, {2, 2}); }
BOOST_AUTO_TEST_CASE(ND_2_0_2_With_Output) {
  subTestSmallSlice(5, {0, 2}, {2, 2}, true);
}
// 2x2x2 outputs
BOOST_AUTO_TEST_CASE(ND_2_4_2) {
  // The same result has as for 2_0_2 but with an extra compute set and
  // additional testing of dim1 at all 4 offsets
  subTestSmallSlice(5, {0, 1, 2}, {2, 4, 2});
}
BOOST_AUTO_TEST_CASE(ND_2_4_2_With_Output) {
  // The same result has as for 2_0_2 but with an extra compute set and
  // additional testing of dim1 at all 4 offsets
  subTestSmallSlice(5, {0, 1, 2}, {2, 4, 2}, true);
}

BOOST_AUTO_TEST_SUITE_END()

// large-buffer update
BOOST_AUTO_TEST_SUITE(LargeBuffer)

BOOST_AUTO_TEST_CASE(circTest) {
  auto delayTestData = GenDelayData();
  sliceTestND(24, delayTestShape, delayTestData, {1}, {1});
}

BOOST_AUTO_TEST_CASE(circTestWithOutput) {
  auto delayTestData = GenDelayData();
  sliceTestND(24, delayTestShape, delayTestData, {1}, {1}, true);
}

BOOST_AUTO_TEST_SUITE_END()

// Dynamic update
// Check dynamicSliceND() extracts \a sliceSizes elements from the \a sliceDims
// dimensions for all possible offsets.
void updateTestND(unsigned tilesPerIPU, const std::vector<size_t> &testShape,
                  const std::vector<std::vector<std::vector<float>>> &testBase,
                  const std::vector<std::size_t> &sliceDims,
                  const std::vector<std::size_t> &sliceSizes) {
  BOOST_TEST_MESSAGE(
      "Test " << boost::unit_test::framework::current_test_case().p_name);

  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<size_t> t1Shape = testShape;

  std::vector<size_t> subShape = t1Shape;
  for (unsigned i = 0; i != sliceDims.size(); ++i) {
    subShape[sliceDims[i]] = sliceSizes[i];
  }
  Tensor t1, s1;

  if (useDSMapper) {
    // Test with the old test-specific allocator
    t1 = graph.addVariable(FLOAT, t1Shape, "t1");
    BOOST_TEST_MESSAGE("Created tensor t1: " << t1);
    s1 = graph.addVariable(FLOAT, subShape, "s1");
    BOOST_TEST_MESSAGE("Created tensor s1: " << s1);
    MapAcrossTiles(graph, tilesPerIPU, t1);
    MapAcrossTiles(graph, tilesPerIPU, s1);
  } else {
    // Test with the new reference tensor allocator.
    t1 = createSliceableTensor(graph, FLOAT, t1Shape, sliceDims, sliceSizes, 2,
                               "t1");
    s1 = createSliceTensor(graph, t1, sliceDims, sliceSizes, 1, "s1")
             .squeeze({0});
  }
  auto tWantedOffsets =
      graph.addVariable(UNSIGNED_INT, {sliceDims.size()}, "wantedOffsets");
  graph.setTileMapping(tWantedOffsets, 0);
  BOOST_TEST_MESSAGE("t1 is " << t1 << " mapping "
                              << toString(graph.getTileMapping(t1)));
  BOOST_TEST_MESSAGE("s1 is " << s1 << " mapping "
                              << toString(graph.getTileMapping(t1)));

  auto prog = Sequence();

  dynamicUpdate(graph, t1, s1, tWantedOffsets, sliceDims, sliceSizes, prog,
                "DSUpdate");

  graph.createHostWrite("in", t1);
  graph.createHostWrite("update", s1);
  graph.createHostWrite("selector", tWantedOffsets);
  graph.createHostRead("out", t1);

  BOOST_TEST_MESSAGE("Creating engine");
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);

    TestData testData(t1Shape, subShape, testBase);

    for (unsigned a = 0; a != subShape[0]; ++a) {
      for (unsigned b = 0; b != subShape[1]; ++b) {
        for (unsigned c = 0; c != subShape[2]; ++c) {
          testData.hSub[a][b][c] = testData.hInit[a][b][c] * 0.001;
        }
      }
    }
    eng.writeTensor("update", testData.hSub.data(),
                    testData.hSub.data() + s1.numElements());

    std::vector<unsigned> nOffsets(t1.rank(), 1);
    for (auto dim : sliceDims) {
      nOffsets[dim] = t1.dim(dim);
    }
    assert(t1.rank() == NUM_DIMS);
    for (unsigned sliceA = 0; sliceA != nOffsets[0]; ++sliceA) {
      for (unsigned sliceB = 0; sliceB != nOffsets[1]; ++sliceB) {
        for (unsigned sliceC = 0; sliceC != nOffsets[2]; ++sliceC) {
          unsigned offsets[NUM_DIMS] = {sliceA, sliceB, sliceC};
          unsigned hOffsets[NUM_DIMS];
          for (unsigned i = 0; i != sliceDims.size(); ++i) {
            hOffsets[i] = offsets[sliceDims[i]];
          }
          std::vector<size_t> checkOffsets = {{sliceA, sliceB, sliceC}};
          eng.writeTensor("in", testData.hInit.data(),
                          testData.hInit.data() + t1.numElements());
          eng.writeTensor("selector", hOffsets, &hOffsets[sliceDims.size()]);
          for (unsigned i = 0; i != testData.hUpdateOut.num_elements(); ++i)
            testData.hUpdateOut.data()[i] = 0.0;
          BOOST_TEST_MESSAGE("Engine run " << toString(checkOffsets));
          eng.run();
          eng.readTensor("out", testData.hUpdateOut.data(),
                         testData.hUpdateOut.data() + t1.numElements());

          boost::multi_array<float, 3> refResult =
              refUpdate(testData.hInit, testData.hSub, checkOffsets);
          checkResult(testData.hUpdateOut, refResult);
        }
      }
    }
  });
}

static void testSmallUpdate(unsigned tilesPerIPU,
                            const std::vector<std::size_t> &sliceDims,
                            const std::vector<std::size_t> &sliceSizes) {
  updateTestND(tilesPerIPU, smallTestShape, smallTestData, sliceDims,
               sliceSizes);
}

BOOST_AUTO_TEST_SUITE(Update)

// Test full update
BOOST_AUTO_TEST_CASE(Update_Full) {
  subTestSmallSlice(5, {0, 1, 2}, smallTestShape);
}

BOOST_AUTO_TEST_CASE(Update_Empty) {
  // result should be the same as the full update with offsets 0,0,0
  subTestSmallSlice(5, {}, {});
}

// Test insertion of a single dimension
BOOST_AUTO_TEST_CASE(Update_5_0) { testSmallUpdate(5, {0}, {1}); }
BOOST_AUTO_TEST_CASE(Update_5_1) { testSmallUpdate(5, {1}, {1}); }
BOOST_AUTO_TEST_CASE(Update_5_2) { testSmallUpdate(5, {2}, {1}); }
// Test insertion of a single element
BOOST_AUTO_TEST_CASE(Update_5_element) {
  testSmallUpdate(5, {0, 1, 2}, {1, 1, 1});
}
// Test insertion of a 2x2 element
BOOST_AUTO_TEST_CASE(Update_5_2x2) { testSmallUpdate(5, {0, 1}, {2, 2}); }

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Misc)

// Two large contiguous regions which exceed the maximum size allowed by the
// 2D vertex fields. Only creates additional vertices for IPU2.
void largeRegions() {
  constexpr static unsigned tilesPerIPU = 1;
  const auto dataType = poplar::HALF;

  auto device =
      createTestDevice(TEST_TARGET, 1, tilesPerIPU,
                       /* compileIPUCode */ true /* for exchange code size */);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  popops::addCodelets(graph);
  Sequence uploadProg, prog, downloadProg;

  const auto numColumns = 32000U;

  const auto sliceOffset = 1;
  auto offset = graph.addConstant(UNSIGNED_INT, {1}, sliceOffset);
  graph.setTileMapping(offset, 0);
  auto t1 = graph.addVariable(dataType, {1, numColumns}, "Base tensor1");
  auto t2 = graph.addVariable(dataType, {1, numColumns}, "Base tensor2");
  auto baseTensor = concat({t1, t2});
  graph.setTileMapping(baseTensor, 0);
  auto subTensor = popops::dynamicSlice(graph, baseTensor, offset, {0}, {1},
                                        prog, "dynamicSlice");

  const auto numElements = baseTensor.numElements();
  const auto firstSliceData = 1.0;
  const auto secondSliceData = -1.0;
  graph.createHostWrite("baseTWr", baseTensor, true);
  graph.createHostWrite("subTWr", subTensor, true);
  graph.createHostRead("subTRd", subTensor, true);

  std::vector<float> hIn(numElements);
  std::fill(hIn.begin(), hIn.begin() + numElements / 2, firstSliceData);
  std::fill(hIn.begin() + numElements / 2, hIn.end(), secondSliceData);
  std::vector<char> rawIn(target.getTypeSize(HALF) * hIn.size());
  poplar::copyFloatToDeviceHalf(target, hIn.data(), rawIn.data(), hIn.size());

  std::vector<float> hOut(subTensor.numElements());
  std::vector<char> rawOut(target.getTypeSize(HALF) * hOut.size());

  Engine e(graph, std::move(prog));

  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("baseTWr", rawIn.data(), rawIn.data() + rawIn.size());
    e.writeTensor("subTWr", rawOut.data(), rawOut.data() + rawOut.size());
    e.run();
    e.readTensor("subTRd", rawOut.data(), rawOut.data() + rawOut.size());
  });

  poplar::copyDeviceHalfToFloat(target, rawOut.data(), hOut.data(),
                                hOut.size());

  // Now check tensor
  const auto value = sliceOffset == 1 ? secondSliceData : firstSliceData;
  for (unsigned i = 0; i != hOut.size(); ++i) {
    BOOST_CHECK_EQUAL(hOut[i], value);
  }
}

BOOST_AUTO_TEST_CASE(LargeRegionsToSplitTest) { largeRegions(); }

// Check that slices happen in the best order possible. Note that this currently
// abuses Graph::outputComputeGraph(). If this test breaks because of changes
// there I suggest you just delete this test.
BOOST_AUTO_TEST_CASE(SliceOrder) {
  // Input Tensor size: 100 x 50 x 10
  // Output Tensor size: 90 x 15 x 8

  // dims: [2, 0, 1]
  // sizes: [9, 90, 15]

  // It should be smart enough to realise that it should reorder the slicing
  // so that it slices the dimensions in the order [1, 2, 0] (and
  // idxOrder should be [2, 0, 1]).

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  std::vector<size_t> t1Shape = {100, 50, 10};
  std::vector<std::size_t> sliceDims = {2, 0, 1};
  std::vector<std::size_t> sliceSizes = {9, 90, 15};

  auto offset = graph.addVariable(UNSIGNED_INT, {t1Shape.size()}, "offset");
  graph.setTileMapping(offset, 0);

  Tensor input;
  if (useDSMapper) {
    input = graph.addVariable(FLOAT, t1Shape, "input");
    MapAcrossTiles(graph, graph.getTarget().getTilesPerIPU(), input);
  } else {
    input = createSliceableTensor(graph, FLOAT, t1Shape, sliceDims, sliceSizes,
                                  2, "input");
  }

  auto prog = Sequence();

  auto out = dynamicSlice(graph, input, offset, sliceDims, sliceSizes, prog);

  // Check that the graph is correct... in the ugliest way possible.
  std::stringstream computeGraph;
  graph.outputComputeGraph(computeGraph, {prog});

  // "/dynamicSlice_d1/slice" should be before "/dynamicSlice_d2/slice" and
  // so on.

  std::string cg = computeGraph.str();
  auto d0_idx = cg.find("dynamicSlice_d0/slice");
  auto d1_idx = cg.find("dynamicSlice_d1/slice");
  auto d2_idx = cg.find("dynamicSlice_d2/slice");
  const auto npos = std::string::npos;
  BOOST_CHECK(d0_idx != npos && d1_idx != npos && d2_idx != npos);
  BOOST_CHECK(d1_idx < d2_idx);
  BOOST_CHECK(d2_idx < d0_idx);
}

#if 0
BOOST_AUTO_TEST_CASE(ImbalanceTest) {
  auto T = 4u;
  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  auto N = 1024ul;
  auto M = 1024ul;

  auto t = graph.addVariable(FLOAT, {N, M}, "t");

  mapTensorLinearly(graph, t);
  // Check that no matter which way the tensor is sliced, the result is
  // balanced across tiles.
  Sequence prog;
  auto offset = graph.addVariable(UNSIGNED_INT, {1}, "offset");
  graph.setTileMapping(offset, 0);

  auto s1 = dynamicSlice(graph, t, offset, {0}, {1}, prog, "").flatten();
  BOOST_CHECK_EQUAL(s1.dim(0), M);
  BOOST_CHECK_EQUAL(getTileImbalance(graph, s1), 0);
  auto s2 =
      dynamicSlice(graph, t.transpose(), offset, {0}, {1}, prog, "").flatten();
  BOOST_CHECK_EQUAL(s2.dim(0), N);
  BOOST_CHECK_EQUAL(getTileImbalance(graph, s2), 0);
}
#endif
BOOST_AUTO_TEST_CASE(LargeTensorSlice) {
  // This test should pass with large T - but graph construction becomes
  // slow (a couple of minutes for T=1024)
  auto T = 64u;
  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  auto N = 32 * 1024ul;
  auto M = 2 * T; // multiple elements can be stored on each Tile
  // Map the tensor carefully to ensure balance and minimise edge pointers
  auto tmpOrder = graph.addVariable(HALF, {T, N, M / T}, "t");
  mapTensorLinearly(graph, tmpOrder);
  auto t = tmpOrder.dimShuffle({1, 0, 2}).reshape({N, M});

  Sequence prog;
  auto offset = graph.addVariable(UNSIGNED_INT, {1}, "offset");
  graph.setTileMapping(offset, 0);
  auto s1 = dynamicSlice(graph, t, offset, {0}, {1}, prog, "").flatten();
  BOOST_CHECK_EQUAL(s1.dim(0), M);
  BOOST_CHECK_EQUAL(getTileImbalance(graph, s1), 0);

  auto tempDir = TempDir::create();
  poplar::OptionFlags engineOptions;
  engineOptions.set("autoReport.outputExecutionProfile", "true");
  engineOptions.set("autoReport.directory", tempDir.getPath());
  OptionFlags profileOptions{{"showExecutionSteps", "true"},
                             {"showVarStorage", "true"}};
  // Actually build the graph to check that it fits onto the target
  // This will fail if many edge pointers or significant exchange is required
  Engine eng(graph, prog, engineOptions);

  std::stringstream ss;
  eng.printProfileSummary(ss, profileOptions);
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_CASE(SliceableTensorFromSlice) {
  // This test checks the expected properties of the returned tensor from
  // createSliceableTensorFromSlice which are that the tensor has the
  // same contiguous regions in each slice on each tile, and that
  // the mapping of each individual slice to tiles is the same as that
  // of the given reference slice.
  constexpr std::size_t numTiles = 64;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());
  auto slice = graph.addVariable(HALF, {20, 10, 10}, "s");

  // Shuffle contiguous regions.
  slice = slice.dimShuffle({2, 0, 1});
  // Map shuffled stuff.
  mapTensorLinearly(graph, slice, 1, 1);

  const std::vector<std::size_t> dims = {0, 1};
  const std::vector<std::size_t> numSlices = {2, 3};
  std::vector<std::size_t> expectedShape = slice.shape();
  for (std::size_t i = 0; i < dims.size(); ++i) {
    expectedShape[dims[i]] *= numSlices[i];
  }
  // Expect the most sliced dimension to be sliced first as
  // this reduces the work for later slices.
  const std::vector<std::size_t> expectedOrder = {1, 0};
  auto totalNumSlices =
      std::accumulate(numSlices.begin(), numSlices.end(), std::size_t(1),
                      std::multiplies<std::size_t>());
  auto t = createSliceableTensorFromSlice(graph, slice, dims, numSlices, "t");
  const auto tShape = t.shape();

  BOOST_CHECK_EQUAL(t.numElements(), slice.numElements() * totalNumSlices);
  BOOST_CHECK_EQUAL_COLLECTIONS(tShape.begin(), tShape.end(),
                                expectedShape.begin(), expectedShape.end());
  std::vector<Tensor> slices(totalNumSlices);
  std::vector<Tensor *> slicePtrs(totalNumSlices);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      slices[i * 2 + j] = t.slice(i * 20, (i + 1) * 20, 1)
                              .slice(j * 10, (j + 1) * 10, 0)
                              .flatten();
      slicePtrs[i * 2 + j] = &slices[i * 2 + j];
    }
  }
  auto sliceFlat = slice.flatten();
  graph.reorderToSimplify(&sliceFlat, slicePtrs);
  const auto referenceMapping = graph.getTileMapping(sliceFlat);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      // Expect each slice to be contiguous on each tile when reordered
      // to be contiguous with respect to the reference slice on each tile.
      const auto tSlice = slices[i * 2 + j];
      BOOST_CHECK_EQUAL(tSlice.numElements(), slice.numElements());

      const auto mapping = graph.getTileMapping(tSlice);
      BOOST_CHECK(mapping == referenceMapping);
      for (unsigned tile = 0; tile < mapping.size(); ++tile) {
        if (!mapping[tile].empty()) {
          auto contiguousRegions =
              graph.getSortedContiguousRegions(tSlice, mapping[tile]);
          BOOST_CHECK_EQUAL(contiguousRegions.size(), 1);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(GetSliceMapping) {
  // Check the mapping can be retrieved using a very simple test case
  // just to see that the function runs
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  std::vector<size_t> tShape = {100, 64};
  auto input =
      createSliceableTensor(graph, FLOAT, tShape, {0}, {1}, 2, "input");

  auto mapping = getSliceMapping(graph, input, {0}, {1});
  std::vector<std::vector<Interval>> expectedMapping = {
      {{0, 16}}, {{16, 32}}, {{32, 48}}, {{48, 64}}};

  auto checkResult = [&]() {
    if (mapping.size() != expectedMapping.size()) {
      return false;
    }
    for (unsigned i = 0; i < mapping.size(); i++) {
      if (mapping[i].size() != expectedMapping[i].size()) {
        return false;
      }
      for (unsigned j = 0; j < mapping[i].size(); j++) {
        if (mapping[i][j] != expectedMapping[i][j]) {
          return false;
        }
      }
    }
    return true;
  };
  BOOST_TEST(checkResult());
}

BOOST_AUTO_TEST_SUITE_END()

void multislice(const std::vector<unsigned> &indicies,
                const std::vector<std::size_t> &indiciesShape,
                bool planAsEmbedding, bool dynamic = true,
                bool createSliceableInputs = true,
                unsigned E = 8, // unsliced dim
                boost::optional<unsigned> dictSize = boost::none,
                bool remapOutOfBoundIndices = false,
                bool paddingIndexUsed = false) {
  // This test should pass with large T - but graph construction becomes
  // slow (a couple of minutes for T=1024)
  assert(indiciesShape.size() == 2); // max 2 dims supported by this test
  assert(indicies.size() == indiciesShape[0] * indiciesShape[1]);
  const auto T = 16;                               // tiles
  const auto D = dictSize ? *dictSize : 1501 * 10; // dictionary size
  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<std::size_t> sliceDims{0};
  std::vector<std::size_t> sliceSizes{1};

  OptionFlags sliceOptions;
  sliceOptions.set("remapOutOfBoundIndices",
                   remapOutOfBoundIndices ? "true" : "false");
  sliceOptions.set("paddingIndexUsed", paddingIndexUsed ? "true" : "false");

  auto plan = SlicePlan();
  if (planAsEmbedding) {
    plan = embedding::plan(graph, FLOAT, D, E, {indicies.size()}, sliceOptions);
  }

  // Map the tensor carefully to ensure balance and minimise edge pointers
  Tensor t;
  if (!createSliceableInputs && planAsEmbedding) {
    t = graph.addVariable(FLOAT, {E, D}, "t");
    mapTensorLinearly(graph, t, 0, 4);
    t = t.transpose();
  } else {
    t = createSliceableTensor(graph, FLOAT, {D, E}, sliceDims, sliceSizes, plan,
                              sliceOptions, "t");
  }

  Sequence prog;

  auto offsetInit =
      graph.addConstant(UNSIGNED_INT, indiciesShape, indicies.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indicies.size(), plan,
                                    sliceOptions, "offset");
  prog.add(Copy(offsetInit, offset));
  poplar::Tensor s;
  if (dynamic) {
    s = multiSlice(graph, t, offset, sliceDims, sliceSizes, prog, plan,
                   sliceOptions, "MultisliceTest");
  } else {
    s = multiSlice(graph, t, indicies, sliceDims[0], prog, "MultisliceTest",
                   sliceOptions);
  }

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indiciesShape[0]);
  BOOST_CHECK_EQUAL(s.dim(1), indiciesShape[1]);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("in", t, true);
  graph.createHostRead("out", s, true);
  std::vector<uint32_t> hIn(t.numElements());
  std::vector<uint32_t> hOut(s.numElements());
  std::iota(hIn.begin(), hIn.end(), 0u);
  auto tempDir = TempDir::create();
  poplar::OptionFlags engineOptions;
  engineOptions.set("autoReport.outputExecutionProfile", "true");
  engineOptions.set("autoReport.directory", tempDir.getPath());
  OptionFlags profileOptions{{"showExecutionSteps", "true"},
                             {"showVarStorage", "true"}};
  // Engine creation will fail for non-cpu targets if many edge pointers or
  // significant exchange is required; this should not happen if
  // createSliceableTensor() has given a good layout
  Engine eng(graph, prog, engineOptions);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn.data(), hIn.data() + hIn.size());
    eng.run();
    eng.readTensor("out", hOut.data(), hOut.data() + hOut.size());
  });
  unsigned outIdx = 0;
  for (const auto &e : hOut)
    BOOST_TEST_MESSAGE("MSlice Output[" << outIdx++ << "] = " << e);
  for (unsigned i = 0; i != indicies.size(); ++i) {
    auto d = indicies[i];
    if (d >= D && remapOutOfBoundIndices) {
      d = 0;
    }
    auto notPadding = d < D;
    for (unsigned elem = 0; elem != E; ++elem) {
      unsigned expected = notPadding ? hIn[d * E + elem] : 0;
      BOOST_CHECK_EQUAL(hOut[i * E + elem], expected);
    }
  }
  std::stringstream ss;
  eng.printProfileSummary(ss, profileOptions);
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_SUITE(MultiSlice)

// test the looping multislice
BOOST_AUTO_TEST_CASE(MultiSlice5) {
  multislice({100, 0, 50, 48, 49}, {5, 1}, false);
  multislice({100, 0, 50, 48, 49}, {5, 1}, false, /* dynamic = */ false);
}

// test the inlined multislice
BOOST_AUTO_TEST_CASE(MultiSlice2) {
  multislice({100, 0}, {2, 1}, false);
  multislice({100, 0}, {2, 1}, false, /* dynamic = */ false);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiSlice10) {
  multislice({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false);
  multislice({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1},
             /* dynamic = */ false);
}

// Test out of bound indices for static multislice
BOOST_AUTO_TEST_CASE(MultiSliceStaticWithOutOfBoundsIndices) {
  multislice({8, 1, 20, 22}, {4, 1}, false,
             /* dynamic = */ false, true, /* E = */ 8, 20,
             /* remapOutOfBoundIndices */ true);
  multislice({8, 1, 20, 20}, {4, 1}, false,
             /* dynamic = */ false, true, /* E = */ 8, 20,
             /* remapOutOfBoundIndices = */ false,
             /* usePaddingIndex = */ true);
  multislice({20, 20, 20, 20}, {4, 1}, false,
             /* dynamic = */ false, true, /* E = */ 8, 20,
             /* remapOutOfBoundIndices = */ false,
             /* usePaddingIndex = */ true);
}

// Test out of bound indices for static multislice
BOOST_AUTO_TEST_CASE(MultiSliceDynamicWithOutOfBoundsIndices) {
  multislice({8, 1, 20, 22}, {4, 1}, false,
             /* dynamic = */ true, true, /* E = */ 8, 20,
             /* remapOutOfBoundIndices */ true);
  multislice({8, 1, 20, 20}, {4, 1}, false,
             /* dynamic = */ true, true, /* E = */ 8, 20,
             /* remapOutOfBoundIndices = */ false,
             /* usePaddingIndex = */ true);
  multislice({20, 20, 20, 20}, {4, 1}, false,
             /* dynamic = */ true, true, /* E = */ 8, 20,
             /* remapOutOfBoundIndices = */ false,
             /* usePaddingIndex = */ true);
}

// test the looping multislice
BOOST_AUTO_TEST_CASE(MultiSlice5_AsEmbedding) {
  multislice({100, 0, 50, 48, 49}, {5, 1}, true);
  multislice({100, 0, 50, 48, 49}, {5, 1}, true, /* dynamic = */ false);
}

// test the inlined multislice
BOOST_AUTO_TEST_CASE(MultiSlice2_AsEmbedding) {
  multislice({100, 0}, {2, 1}, true);
  multislice({100, 0}, {2, 1}, true, /* dynamic = */ false);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiSlice10_AsEmbedding) {
  multislice({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true);
  multislice({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true,
             /* dynamic = */ false);

  multislice({2, 1, 2, 1, 201, 70, 60, 50, 40, 30}, {10, 1}, true,
             /* dynamic = */ true, true, /* E = */ 8, /* D = */ 200,
             /* remapOutOfBoundIndices */ true);
  multislice({2, 1, 2, 1, 200, 70, 60, 50, 200, 30}, {10, 1}, true,
             /* dynamic = */ true, true, /* E = */ 8, /* D = */ 200,
             /* remapOutOfBoundIndices = */ false,
             /* usePaddingIndex = */ true);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiSlicePlannedRegroup) {
  multislice({2, 1, 2, 1, 80, 70, 60, 50}, {8, 1}, true,
             /* dynamic = */ true, false, 16, 256);
}

// test heuristic which checks for mapping of a slice.
// if this doesn't kick in we will run out of memory on some
// tiles hence we check for an error constructing the engine.
BOOST_AUTO_TEST_CASE(MultiSlicePoorlyMapped) {
  const unsigned numTiles = 16u;
  const unsigned D = 1501u * 10u;
  const unsigned E = 8u;

  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  const std::vector<std::size_t> sliceDims{0};
  const std::vector<std::size_t> sliceSizes{1};
  const std::vector<std::uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  const auto sliceOptions = OptionFlags();
  const auto plan = SlicePlan();
  // We'll create this with the sliced dimension as the innermost
  // so that our rearrangement is hopefully niceish anyway mostly
  // for test speed purposes.
  auto t = graph.addVariable(FLOAT, {E, D}, "t").transpose();
  // We'll map this with the unsliced dimension as innermost such
  // that the first slice will most certainly reside on a single tile.
  // This is the case where there is plenty of data that will be
  // transferred to a single tile and so we should spread it to
  // slice properly.
  mapTensorLinearly(graph, t, 0, 1);

  Sequence prog;

  auto offsetInit = graph.addConstant(UNSIGNED_INT, {indices.size()},
                                      indices.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indices.size(), plan,
                                    sliceOptions, "offset");

  prog.add(Copy(offsetInit, offset));
  auto s = multiSlice(graph, t, offset, sliceDims, sliceSizes, prog, plan,
                      sliceOptions, "multiSliceTest");

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indices.size());
  BOOST_CHECK_EQUAL(s.dim(1), 1);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("in", t, true);
  graph.createHostRead("out", s, true);
  std::vector<std::uint32_t> hIn(t.numElements());
  std::vector<std::uint32_t> hOut(t.numElements());
  std::iota(hIn.begin(), hIn.end(), 0u);

  auto tempDir = TempDir::create();
  poplar::OptionFlags engineOptions;
  engineOptions.set("autoReport.outputExecutionProfile", "true");
  engineOptions.set("autoReport.directory", tempDir.getPath());
  Engine e(graph, prog, engineOptions);
  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("in", hIn.data(), hIn.data() + hIn.size());
    e.run();
    e.readTensor("out", hOut.data(), hOut.data() + hOut.size());
  });

  unsigned outIdx = 0;
  for (const auto &e : hOut)
    BOOST_TEST_MESSAGE("MSlice Output[" << outIdx++ << "] = " << e);
  for (unsigned i = 0; i != indices.size(); ++i) {
    auto d = indices[i];
    for (unsigned elem = 0; elem != E; ++elem) {
      unsigned expected = hIn[d * E + elem];
      BOOST_CHECK_EQUAL(hOut[i * E + elem], expected);
    }
  }
  std::stringstream ss;
  e.printProfileSummary(ss, {
                                {"showExecutionSteps", "true"},
                                {"showVarStorage", "true"},
                            });
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_SUITE_END()

void multiupdate(const std::vector<uint32_t> &indicies,
                 const std::vector<std::size_t> &indiciesShape,
                 bool planAsEmbedding,
                 boost::optional<popops::Operation> op = boost::none,
                 const Type &dType = HALF, float updateScaling = 1.0,
                 const unsigned E = 8, // unsliced dim
                 boost::optional<unsigned> dictSize = boost::none,
                 bool useFloatScalingForHalf = false, bool dynamic = true,
                 bool createSliceableInputs = true,
                 bool remapOutOfBoundIndices = false,
                 bool paddingIndexUsed = false) {
  const bool updateOp = op != boost::none;
  const bool opUsesScale = updateOp && *op == popops::Operation::ADD;

  const Type scaleTensorType =
      dType == HALF && useFloatScalingForHalf ? FLOAT : dType;

  // This test should pass with large T - but graph construction becomes
  // slow (a couple of minutes for T=1024)
  assert(indiciesShape.size() == 2); // max 2 dims supported by this test
  assert(indicies.size() == indiciesShape[0] * indiciesShape[1]);
  // Use fewer tiles for operations that don't use scale as tests using
  // simulator and H/w are run for those ops and we want to reduce the test
  // time.
  const auto T = (updateOp && !opUsesScale) ? 4U : 16; // tiles
  const auto D = dictSize == boost::none ? T == 4 ? 203U * 10 : 1501U * 10
                                         : *dictSize; // dictionary
  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<std::size_t> sliceDims{0};
  std::vector<std::size_t> sliceSizes{1};
  Tensor scale;

  auto sliceOptions = OptionFlags();
  if (op == boost::none) {
    sliceOptions.set({{"operationForUpdate", "none"}});
  } else if (*op == Operation::ADD) {
    sliceOptions.set({{"operationForUpdate", "add"}});
  } else if (*op == Operation::MAX) {
    sliceOptions.set({{"operationForUpdate", "max"}});
  }

  sliceOptions.set("remapOutOfBoundIndices",
                   remapOutOfBoundIndices ? "true" : "false");
  sliceOptions.set("paddingIndexUsed", paddingIndexUsed ? "true" : "false");
  auto plan = SlicePlan();
  if (planAsEmbedding) {
    sliceOptions.set({{"usedForSlice", "false"}, {"usedForUpdate", "true"}});
    plan = embedding::plan(graph, dType, D, E, {indicies.size()}, sliceOptions);
  }

  // Map the tensor carefully to ensure balance and minimise edge pointers
  Tensor t, s;
  if (!createSliceableInputs && planAsEmbedding) {
    t = graph.addVariable(dType, {E, D}, "t");
    mapTensorLinearly(graph, t, 0, 4);
    t = t.transpose();
    s = graph.addVariable(dType, {E, 1, indicies.size()}, "s");
    mapTensorLinearly(graph, s, 0, 4);
    s = s.dimShuffle({2, 1, 0});
  } else {
    t = createSliceableTensor(graph, dType, {D, E}, sliceDims, sliceSizes, plan,
                              sliceOptions, "t");
    s = createSliceTensor(graph, dType, {D, E}, sliceDims, sliceSizes,
                          indicies.size(), plan, sliceOptions, "s");
  }

  Sequence prog;
  auto offsetInit =
      graph.addConstant(UNSIGNED_INT, indiciesShape, indicies.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indicies.size(), plan,
                                    sliceOptions, "offset");
  prog.add(Copy(offsetInit, offset));
  if (!updateOp) {
    multiUpdate(graph, t, s, offset, sliceDims, sliceSizes, prog, plan,
                sliceOptions, "MultisliceTest");
  } else {
    if (*op == popops::Operation::ADD) {
      scale = graph.addVariable(scaleTensorType, {}, "scale");
      graph.setTileMapping(scale, 0);

      if (dynamic) {
        multiUpdateAdd(graph, t, s, offset, scale, sliceDims, sliceSizes, prog,
                       plan, sliceOptions, "MultisliceTest");
      } else {
        multiUpdateAdd(graph, t, s, indicies, scale, sliceDims[0], prog,
                       "MultisliceTest", sliceOptions);
      }
    } else if (*op == popops::Operation::MAX) {
      BOOST_CHECK(dynamic);
      multiUpdateMax(graph, t, s, offset, sliceDims, sliceSizes, prog, plan,
                     sliceOptions, "MultisliceTest");

    } else {
      std::cerr << "\n Unsupported op in multiUpdateOp\n";
      BOOST_CHECK(false);
    }
  }

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indiciesShape[0]);
  BOOST_CHECK_EQUAL(s.dim(1), indiciesShape[1]);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("inS", s, true);
  graph.createHostWrite("inT", t, true);
  graph.createHostRead("outT", t, true);
  if (updateOp && opUsesScale)
    graph.createHostWrite("scale", scale, true);
  std::vector<float> hIn(s.numElements());
  // This to get value in range for the addition
  const float outBaseValue = 100.0f;

  auto updateSlices = [&](std::vector<float> &h) {
    const unsigned numElements = h.size();
    // Most tests use index 0 and 1. Change those entries such that unsliced
    // dimension has at most 16 unique entries where possible. This tests that
    // consecutive 4 half reads will be different
    for (unsigned i = 0; i != std::min(2 * E, numElements); ++i) {
      h[i] += (i % 16);
    }
  };

  std::vector<float> hOut(t.numElements(), outBaseValue);
  updateSlices(hOut);

  // This test checks halves - some of these entries will be >maxHalf so the
  // test may fail if large offsets are indexed
  for (unsigned i = 0; i != hIn.size(); ++i)
    hIn[i] = i + 1.0f;

  auto target = device.getTarget();
  std::vector<char> rawIn(target.getTypeSize(dType) * hIn.size());
  std::vector<char> rawOut(target.getTypeSize(dType) * hOut.size());
  std::vector<char> rawScaleIn(target.getTypeSize(scaleTensorType) * 1);

  std::vector<float> scalingF = {updateScaling};
  copy(target, scalingF, scaleTensorType, rawScaleIn.data());
  copy(target, hIn, dType, rawIn.data());
  copy(target, hOut, dType, rawOut.data());

  OptionFlags profileOptions{{"showExecutionSteps", "true"},
                             {"showVarStorage", "true"}};
  // Engine creation will fail for non-cpu targets if many edge pointers or
  // significant exchange is required; this should not happen if
  // createSliceableTensor() has given a good layout
  auto tempDir = TempDir::create();
  poplar::OptionFlags engineOptions;
  engineOptions.set("autoReport.outputExecutionProfile", "true");
  engineOptions.set("autoReport.directory", tempDir.getPath());
  Engine eng(graph, prog, engineOptions);
  device.bind([&](const Device &d) {
    eng.load(d);
    if (updateOp && opUsesScale) {
      eng.writeTensor("scale", rawScaleIn.data(),
                      rawScaleIn.data() + rawScaleIn.size());
    }
    eng.writeTensor("inT", rawOut.data(), rawOut.data() + rawOut.size());
    eng.writeTensor("inS", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("outT", rawOut.data(), rawOut.data() + rawOut.size());
  });
  copy(target, dType, rawOut.data(), hOut);
  unsigned outIdx = 0;
  for (const auto &e : hOut) {
    if (e != outBaseValue)
      BOOST_TEST_MESSAGE("MUpdate Output[" << outIdx << "] = " << e);
    outIdx++;
  }
  std::vector<float> expected(t.numElements(), outBaseValue);
  updateSlices(expected);
  for (unsigned i = 0; i != indicies.size(); ++i) {
    auto d = indicies[i];
    if (remapOutOfBoundIndices && d >= D)
      d = 0;
    if (d >= D)
      continue;
    for (unsigned elem = 0; elem != E; ++elem) {
      if (!updateOp) {
        expected[d * E + elem] = hIn[i * E + elem];
      } else {
        if (*op == popops::Operation::ADD) {
          expected[d * E + elem] += updateScaling * hIn[i * E + elem];
        } else if (*op == popops::Operation::MAX) {
          expected[d * E + elem] =
              std::max(expected[d * E + elem], hIn[i * E + elem]);
        }
      }
    }
  }
  for (unsigned i = 0; i != expected.size(); ++i)
    BOOST_CHECK_EQUAL(hOut[i], expected[i]);

  std::stringstream ss;
  eng.printProfileSummary(ss, profileOptions);
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_SUITE(MultiUpdate)

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate5) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, false);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate2) { multiupdate({100, 0}, {2, 1}, false); }

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdate10) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false);
}

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd5) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, false, popops::Operation::ADD, HALF,
              0.5);
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, false, popops::Operation::ADD, HALF,
              0.5, 8, boost::none, false, /* dynamic = */ false);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2) {
  multiupdate({100, 0}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5);
  multiupdate({100, 0}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5, 8,
              boost::none, false, /* dynamic = */ false);
}

// test the inlined multiupdate wih remapping/padding
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2OutOfBoundDynamicWithoutPlan) {
  multiupdate({20, 0}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5,
              /* E = */ 8, /* D = */ 20, false, /* dynamic = */ true, true,
              /* remapOutOfBoundIndices = */ true,
              /* paddingIndexUsed = */ false);
  multiupdate({20, 0}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5,
              /* E = */ 8, /* D = */ 20, false, /* dynamic = */ true, true,
              /* remapOutOfBoundIndices = */ false,
              /* paddingIndexUsed = */ true);
}

// test the inlined multiupdate wih remapping/padding
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2OutOfBoundStaticWithoutPlan) {
  multiupdate({20, 0}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5,
              /* E = */ 8, /* D = */ 20, false, /* dynamic = */ false, true,
              /* remapOutOfBoundIndices = */ true,
              /* paddingIndexUsed = */ false);
  multiupdate({20, 0}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5,
              /* E = */ 8, /* D = */ 20, false, /* dynamic = */ false, true,
              /* remapOutOfBoundIndices = */ false,
              /* paddingIndexUsed = */ true);
}

// test the inlined multiupdate wih remapping/padding
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2OutOfBoundDynamicWithPlan) {
  multiupdate({20, 1, 2, 4}, {4, 1}, true, popops::Operation::ADD, HALF, 0.5,
              /* E = */ 16, /* D = */ 20, false, /* dynamic = */ true, true,
              /* remapOutOfBoundIndices = */ true,
              /* paddingIndexUsed = */ false);
  multiupdate({20, 1, 2, 4}, {4, 1}, true, popops::Operation::ADD, HALF, 0.5,
              /* E = */ 16, /* D = */ 20, false, /* dynamic = */ true, true,
              /* remapOutOfBoundIndices = */ false,
              /* paddingIndexUsed = */ true);
}

// test the looping multiupdate with duplicate indices
BOOST_AUTO_TEST_CASE(MultiUpdateAddDup5) {
  multiupdate({100, 0, 50, 0, 100}, {5, 1}, false, popops::Operation::ADD, HALF,
              0.5);
  multiupdate({100, 0, 50, 0, 100}, {5, 1}, false, popops::Operation::ADD, HALF,
              0.5, 8, boost::none, true, /* dynamic = */ false);
}

// test the inlined multiupdate with duplicate indices
BOOST_AUTO_TEST_CASE(MultiUpdateAddDup2) {
  multiupdate({100, 100}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5);
  multiupdate({100, 100}, {2, 1}, false, popops::Operation::ADD, HALF, 0.5, 8,
              boost::none, true, /* dynamic = */ false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdatePlan)

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate5Plan) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, true);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate2Plan) {
  multiupdate({100, 0}, {2, 1}, true, Operation::ADD);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdate10Plan) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true,
              Operation::MAX);
}

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd5Plan) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, true, popops::Operation::ADD, HALF,
              0.5);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2Plan) {
  multiupdate({100, 0}, {2, 1}, true, popops::Operation::ADD, HALF, 0.5);
}

BOOST_AUTO_TEST_CASE(MultiUpdateAddRegroup) {

  multiupdate({100, 0, 50, 0, 100, 1, 2, 4}, {8, 1}, true,
              popops::Operation::ADD, HALF, 1.0, 16, 128, false, true, false);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdatePlanLookupParallelSplit)

// Test lookup split with ADD
BOOST_AUTO_TEST_CASE(MultiUpdateAddLookupParallelSplitPlan) {
  multiupdate({20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 2, 2,  3,
               3,  4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10},
              {30, 1}, true, popops::Operation::ADD, HALF, 0.5, 8, 21);
}

// Test lookup split with None
BOOST_AUTO_TEST_CASE(MultiUpdateNoLookupParallelSplitPlan) {
  multiupdate({20, 0, 1, 2, 3, 4, 5, 6, 7, 7, 1, 1, 2, 2,  3,
               3,  4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10},
              {30, 1}, true, boost::none, HALF, 0.5, 8, 21);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateSingles)

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Singles) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::ADD, HALF, 0.5);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateAddHalfScale)
// Unsliced dimension of 2 per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAddHalfScaleUnsliced2_10Multiples) {
  multiupdate({0, 3, 1, 0, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::ADD, HALF, 0.5, 32, boost::none, true);
}

// Unsliced dimension of 4 per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAddHalfScaleUnsliced4_10Multiples) {
  multiupdate({0, 3, 1, 0, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::ADD, HALF, 0.5, 64, boost::none, true);
}

// Unsliced dimension of 8 per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAddHalfScaleUnsliced8_10Multiples) {
  multiupdate({0, 3, 1, 0, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::ADD, HALF, 0.5, 128, boost::none, true);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateMaxHalf)
BOOST_AUTO_TEST_CASE(MultiUpdateMaxHalf_10Singles) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::MAX, HALF, 0.5);
}

BOOST_AUTO_TEST_CASE(MultiUpdateMaxHalf_10Multiples) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::MAX, HALF, 0.5, 64);
}

// test the fast vertex with multiple updates per tile
BOOST_AUTO_TEST_CASE(MultiUpdateMaxHalf_10Multiples_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true,
              popops::Operation::MAX, HALF, 0.5, 64);
}

BOOST_AUTO_TEST_CASE(MultiUpdateMax5Half) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, false, popops::Operation::MAX,
              HALF);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateMaxFloat)

BOOST_AUTO_TEST_CASE(MultiUpdateMaxFloat_10Singles) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::MAX, FLOAT, 0.5);
}

BOOST_AUTO_TEST_CASE(MultiUpdateMaxFloat_10Multiples) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::MAX, FLOAT, 0.5, 64);
}

BOOST_AUTO_TEST_CASE(MultiUpdateMaxFloat_10Multiples_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true,
              popops::Operation::MAX, FLOAT, 0.5, 64);
}

BOOST_AUTO_TEST_CASE(MultiUpdateMax5Float) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, false, popops::Operation::MAX,
              FLOAT);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateMultiples)

// test the fast vertex with multiple updates per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Multiples) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false,
              popops::Operation::ADD, HALF, 0.5, 64);
}

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate5_AsEmbedding) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, true);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate2_AsEmbedding) {
  multiupdate({100, 0}, {2, 1}, true);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdate10_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true);
}

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd5_AsEmbedding) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, true, popops::Operation::ADD, HALF,
              0.5);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2_AsEmbedding) {
  multiupdate({100, 0}, {2, 1}, true, popops::Operation::ADD, HALF, 0.5);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Singles_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true,
              popops::Operation::ADD, HALF, 0.5);
}

// test the fast vertex with multiple updates per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Multiples_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true,
              popops::Operation::ADD, HALF, 0.5, 64);
}

// test heuristic which checks for mapping of a slice.
// if this doesn't kick in we will run out of memory on some
// tiles hence we check for an error constructing the engine.
static void multiUpdatePoorlyMapped(bool accumulate) {
  const unsigned numTiles = 16u;
  const unsigned D = 1501u * 10u;
  const unsigned E = 8u;

  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  const std::vector<std::size_t> sliceDims{0};
  const std::vector<std::size_t> sliceSizes{1};
  const std::vector<std::uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  const auto sliceOptions = OptionFlags();
  const auto plan = SlicePlan();
  // We'll create this with the sliced dimension as the innermost
  // so that our rearrangement is hopefully niceish anyway mostly
  // for test speed purposes.
  auto t = graph.addVariable(FLOAT, {E, D}, "t").transpose();
  // We'll map this with the unsliced dimension as innermost such
  // that the first slice will most certainly reside on a single tile.
  // This is the case where there is plenty of data that will be
  // transferred to a single tile and so we should spread it to
  // slice properly.
  mapTensorLinearly(graph, t, 0, 1);

  auto s = createSliceTensor(graph, FLOAT, {D, E}, sliceDims, sliceSizes,
                             indices.size(), plan, sliceOptions, "s");

  Sequence prog;

  auto offsetInit = graph.addConstant(UNSIGNED_INT, {indices.size()},
                                      indices.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indices.size(), plan,
                                    sliceOptions, "offset");

  prog.add(Copy(offsetInit, offset));
  auto scale = graph.addVariable(FLOAT, {}, "scale");
  graph.setTileMapping(scale, 0);
  if (accumulate) {
    multiUpdateAdd(graph, t, s, offset, scale, sliceDims, sliceSizes, prog,
                   plan, sliceOptions, "multiUpdateAddTest");
  } else {
    multiUpdate(graph, t, s, offset, sliceDims, sliceSizes, prog, plan,
                sliceOptions, "multiUpdateTest");
  }

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indices.size());
  BOOST_CHECK_EQUAL(s.dim(1), 1);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("inS", s, true);
  graph.createHostWrite("inT", t, true);
  graph.createHostRead("outT", t, true);
  if (accumulate) {
    graph.createHostWrite("scale", scale, true);
  }
  std::vector<float> hIn(s.numElements());
  const float outBaseValue = 100.0f;
  std::vector<float> hOut(t.numElements(), outBaseValue);
  std::iota(hIn.begin(), hIn.end(), 0.0f);
  const float updateScaling = 0.5f;
  std::vector<float> hScale(1, updateScaling);

  auto tempDir = TempDir::create();
  poplar::OptionFlags engineOptions;
  engineOptions.set("autoReport.outputExecutionProfile", "true");
  engineOptions.set("autoReport.directory", tempDir.getPath());
  Engine e(graph, prog, engineOptions);
  device.bind([&](const Device &d) {
    e.load(d);
    if (accumulate) {
      e.writeTensor("scale", hScale.data(), hScale.data() + hScale.size());
    }
    e.writeTensor("inS", hIn.data(), hIn.data() + hIn.size());
    e.writeTensor("inT", hOut.data(), hOut.data() + hOut.size());
    e.run();
    e.readTensor("outT", hOut.data(), hOut.data() + hOut.size());
  });
  unsigned outIdx = 0;
  for (const auto &e : hOut) {
    if (e != outBaseValue) {
      BOOST_TEST_MESSAGE("MUpdate Output[" << outIdx << "] = " << e);
    }
    outIdx++;
  }
  std::vector<float> expected(t.numElements(), outBaseValue);
  for (unsigned i = 0; i != indices.size(); ++i) {
    auto d = indices[i];
    for (unsigned elem = 0; elem != E; ++elem) {
      if (!accumulate) {
        expected[d * E + elem] = hIn[i * E + elem];
      } else {
        expected[d * E + elem] += updateScaling * hIn[i * E + elem];
      }
    }
  }
  for (unsigned i = 0; i != expected.size(); ++i) {
    BOOST_CHECK_EQUAL(hOut[i], expected[i]);
  }

  std::stringstream ss;
  e.printProfileSummary(ss, {
                                {"showExecutionSteps", "true"},
                                {"showVarStorage", "true"},
                            });
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_CASE(MultiUpdatePoorlyMapped) {
  multiUpdatePoorlyMapped(false);
}

BOOST_AUTO_TEST_CASE(MultiUpdateAddPoorlyMapped) {
  multiUpdatePoorlyMapped(true);
}

void multiUpdatePoorlyMappedSlices() {

  constexpr static std::size_t E = 24;
  constexpr static std::size_t V = 400;
  constexpr static std::size_t I = 50;
  constexpr static unsigned tilesPerIPU = 16;

  auto device =
      createTestDevice(TEST_TARGET, 1, tilesPerIPU,
                       /* compileIPUCode */ true /* for exchange code size */);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  const std::vector<std::size_t> sliceDims = {0};
  const std::vector<std::size_t> sliceSizes = {1};

  std::vector<std::uint32_t> indices(I);
  std::iota(indices.begin(), indices.end(), 0);

  // Currently if the factor by which we broadcast the slices is greater than
  // or equal to 4 we should rearrange before broadcasting to reduce the code.
  // We will test this by finding a rough bound on exchange code size with this
  // optimisation and testing that it doesn't increase by roughly
  // (unslicedDimSplit * slicedDimSplit) times in future
  // (i.e. that the optimisation is still in place).
  const OptionFlags &sliceOptions = {
      {"planConstraints", "{\"unslicedDimSplit\": 6, \"slicedDimSplit\": 2, "
                          "\"lookupParallelSplit\": 1}"}};

  const SlicePlan &plan = embedding::plan(graph, HALF, V, E, {I}, sliceOptions);

  auto t = createSliceableTensor(graph, HALF, {V, E}, sliceDims, sliceSizes,
                                 plan, sliceOptions, "t");
  auto s = graph.addVariable(HALF, {I, 1, E}, "s");
  mapTensorLinearly(graph, s, 0, 1);

  Sequence prog;

  auto offsetInit = graph.addConstant(UNSIGNED_INT, {indices.size()},
                                      indices.data(), "offsetInit");
  graph.setTileMapping(offsetInit, 0);

  auto offset = createIndicesTensor(graph, sliceDims, indices.size(), plan,
                                    sliceOptions, "offset");

  prog.add(Copy(offsetInit, offset));

  auto scale = graph.addVariable(HALF, {}, "scale");
  graph.setTileMapping(scale, 0);

  multiUpdateAdd(graph, t, s, offset, scale, sliceDims, sliceSizes, prog, plan,
                 sliceOptions, "multiUpdateAddTest");

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indices.size());
  BOOST_CHECK_EQUAL(s.dim(1), 1);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("inS", s, true);
  graph.createHostWrite("inT", t, true);
  graph.createHostRead("outT", t, true);
  graph.createHostWrite("scale", scale, true);
  std::vector<float> hIn(s.numElements());
  const float outBaseValue = 100.0f;
  std::vector<float> hOut(t.numElements(), outBaseValue);
  std::iota(hIn.begin(), hIn.end(), 0.0f);
  const float updateScaling = 0.5f;
  std::vector<float> hScale(1, updateScaling);

  std::vector<char> rawIn(target.getTypeSize(HALF) * hIn.size());
  std::vector<char> rawOut(target.getTypeSize(HALF) * hOut.size());
  std::vector<char> rawScaleIn(target.getTypeSize(HALF) * hScale.size());
  poplar::copyFloatToDeviceHalf(target, hScale.data(), rawScaleIn.data(),
                                hScale.size());
  poplar::copyFloatToDeviceHalf(target, hIn.data(), rawIn.data(), hIn.size());
  poplar::copyFloatToDeviceHalf(target, hOut.data(), rawOut.data(),
                                hOut.size());

  const auto dir = TempDir::create();
  Engine e(graph, prog,
           {{"autoReport.outputExecutionProfile", "true"},
            {"autoReport.directory", dir.getPath()}});
  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("scale", rawScaleIn.data(),
                  rawScaleIn.data() + rawScaleIn.size());
    e.writeTensor("inS", rawIn.data(), rawIn.data() + rawIn.size());
    e.writeTensor("inT", rawOut.data(), rawOut.data() + rawOut.size());
    e.run();
    e.readTensor("outT", rawOut.data(), rawOut.data() + rawOut.size());
  });
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), hOut.data(),
                                hOut.size());
  unsigned outIdx = 0;
  for (const auto &e : hOut) {
    if (e != outBaseValue) {
      BOOST_TEST_MESSAGE("MUpdate Output[" << outIdx << "] = " << e);
    }
    outIdx++;
  }
  std::vector<float> expected(t.numElements(), outBaseValue);
  for (unsigned i = 0; i != indices.size(); ++i) {
    auto d = indices[i];
    for (unsigned elem = 0; elem != E; ++elem) {
      expected[d * E + elem] += updateScaling * hIn[i * E + elem];
    }
  }
  for (unsigned i = 0; i != expected.size(); ++i) {
    BOOST_CHECK_EQUAL(hOut[i], expected[i]);
  }

  const auto &report = e.getReport();
  std::uint64_t exchangeCodeBytes{};
  std::uint64_t vertexStateBytes{};
  for (const auto &t : report.compilation().tiles()) {
    const auto cats = t.memory().category();
    const auto exCode = cats.internalExchangeCode();
    exchangeCodeBytes += exCode.nonInterleaved().nonOverlapped() +
                         exCode.interleaved().nonOverlapped() +
                         exCode.overflowed().nonOverlapped();

    vertexStateBytes +=
        cats.vertexInstanceState().nonInterleaved().nonOverlapped() +
        cats.copyDescriptor().nonInterleaved().nonOverlapped() +
        cats.vectorListDescriptor().nonInterleaved().nonOverlapped() +
        cats.vertexFieldData().nonInterleaved().nonOverlapped();
  }
  BOOST_TEST_MESSAGE("Total exchange code is " << exchangeCodeBytes);
  BOOST_TEST_MESSAGE("Total vertex state bytes is " << vertexStateBytes);
  BOOST_TEST_MESSAGE("Total exchange code and vertex state bytes is "
                     << exchangeCodeBytes + vertexStateBytes);

  std::stringstream ss;
  e.printProfileSummary(ss, {
                                {"showExecutionSteps", "true"},
                                {"showVarStorage", "true"},
                            });
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_CASE(MultiUpdateAddPoorlyMappedSlices) {
  multiUpdatePoorlyMappedSlices();
}

BOOST_AUTO_TEST_SUITE_END()

// Build and run a small model to check for cpu-specific target problems
void smallAndSimple() {
  const int num_ipus = 1;

  poplar::Device device = poplar::Device::createCPUDevice();
  bool have_device = false;
  if (device.attach()) {
    have_device = true;
  }
  BOOST_CHECK(have_device);

  Target target = device.getTarget();

  std::cout << "Number of IPUs: " << num_ipus << "\n";

  std::cout << "Creating graph\n";
  Graph graph(device, poplar::replication_factor(1));
  popops::addCodelets(graph);
  popops::SlicePlan plan;

  auto ids = popops::createIndicesTensor(graph, {0}, 2, plan, {}, "ids");
  auto input = popops::createSliceableTensor(graph, INT, {3, 3}, {0}, {1}, plan,
                                             {}, "input");
  auto updates = popops::createSliceTensor(graph, INT, {2, 3}, {0}, {1}, 2,
                                           plan, {}, "input");
  auto scale =
      graph.addVariable(INT, {}, VariableMappingMethod::LINEAR, "scale");

  Sequence sequence;
  popops::multiUpdateAdd(graph, input, updates, ids, scale, {0}, {1}, sequence,
                         plan, {}, "update");

  Engine engine(graph, sequence);
  engine.load(device);
  engine.run();
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE(CpuChecks)
BOOST_AUTO_TEST_CASE(SmallAndSimple) { smallAndSimple(); }
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(CheckIndexValidation)
void indexChecks(bool update) {
  const int numIPUs = 1;
  const unsigned tilesPerIPU = 4;

  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  std::cout << "Number of IPUs: " << numIPUs << "\n";
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  constexpr unsigned nonSliceableSize = 1;
  constexpr unsigned sliceableSize = 10;
  constexpr unsigned numIndices = 5;
  std::vector<std::size_t> tShape = {sliceableSize, nonSliceableSize};
  std::vector<std::size_t> sShape = {numIndices, nonSliceableSize};
  OptionFlags optionFlags{{"validateIndices", "true"}};

  popops::SlicePlan plan;
  auto ids =
      popops::createIndicesTensor(graph, {0}, numIndices, plan, {}, "ids");
  graph.createHostWrite("ids", ids);
  auto embedding = popops::createSliceableTensor(graph, INT, tShape, {0}, {1},
                                                 plan, {}, "embedding");
  auto subT = popops::createSliceTensor(graph, INT, sShape, {0}, {1},
                                        numIndices, plan, {}, "subT");
  auto scale =
      graph.addVariable(INT, {}, VariableMappingMethod::LINEAR, "scale");

  Sequence sequence;
  if (update) {
    popops::multiUpdateAdd(graph, embedding, subT, ids, scale, {0}, {1},
                           sequence, plan, optionFlags, "update");
  } else {
    subT = popops::multiSlice(graph, embedding, ids, {0}, {1}, sequence, plan,
                              optionFlags, "slice");
  }

  Engine engine(graph, sequence);
  const std::vector<unsigned> checkValues{0, sliceableSize - 1, sliceableSize,
                                          sliceableSize + 1, unsigned(-1)};
  device.bind([&](const Device &d) {
    for (const auto checkValue : checkValues) {
      bool expectPass = checkValue < sliceableSize;
      BOOST_TEST_MESSAGE("Expecting indices to be "
                         << (expectPass ? "valid" : "invalid"));
      engine.load(d);
      std::vector<unsigned> hIds{checkValue, 4, 3, 2, 1};
      engine.writeTensor("ids", &hIds.data()[0], &hIds.data()[hIds.size()]);
      if (expectPass) {
        BOOST_CHECK_NO_THROW(engine.run());
      } else {
        BOOST_CHECK_THROW(engine.run(), poplar::runtime_error);
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(SliceIndexChecks) { indexChecks(false); }
BOOST_AUTO_TEST_CASE(UpdateIndexChecks) { indexChecks(true); }
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(CheckQuarterMetadata)

void checkQuarterMetadata(void) {
  const unsigned tilesPerIPU = 4;

  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  constexpr unsigned nonSliceableSize = 1;
  constexpr unsigned sliceableSize = 10;
  constexpr unsigned numIndices = 5;
  std::vector<std::size_t> tShape = {sliceableSize, nonSliceableSize};
  std::vector<std::size_t> sShape = {numIndices, nonSliceableSize};
  OptionFlags optionFlags{{"validateIndices", "true"}};

  popops::SlicePlan plan;
  auto ids =
      popops::createIndicesTensor(graph, {0}, numIndices, plan, {}, "ids");
  graph.createHostWrite("ids", ids);

  // Create an input tensor with an associated metadata value to check
  auto embeddingMetadata =
      createConstantMetadataTensor(graph, QuarterMetadata::Format::F143, 1);
  auto embedding = popops::createSliceableTensor(graph, QUARTER, tShape, {0},
                                                 {1}, plan, {}, "embedding");
  Sequence sequence;
  sequence.add(Copy(embeddingMetadata, embedding.getMetadata()));
  // The resulting multiSliced result should have the same metadata
  auto subT = popops::multiSlice(graph, embedding, ids, {0}, {1}, sequence,
                                 plan, optionFlags, "slice");
  poplar::ArrayRef<unsigned> offsets = {0};
  auto subTConst =
      popops::multiSlice(graph, embedding, offsets, 0, sequence, "slice", {});

  // A multiupdate should not affect the metadata of the embedding
  auto dummy = popops::multiSlice(graph, embedding, ids, {0}, {1}, sequence,
                                  plan, optionFlags, "slice");
  // copy new metadata here, so as to leave the original value of metadata in
  // embedding intact for test.
  sequence.add(Copy(poputil::createConstantMetadataTensor(
                        graph, QuarterMetadata::Format::F143, 3),
                    dummy.getMetadata()));
  popops::multiUpdate(graph, embedding, dummy, ids, {0}, {1}, sequence, plan,
                      optionFlags, "slice");

  // The resulting dynamicSliced tensor should have the same metadata
  auto idsDynamicSlice = graph.addConstant<unsigned>(UNSIGNED_INT, {1}, 0u);
  graph.setTileMapping(idsDynamicSlice, 0);
  auto subTDynamicSlice = popops::dynamicSlice(
      graph, embedding, idsDynamicSlice, {0}, {1}, sequence);

  // When an input is provided the metadata should be preserved
  auto subTWithOutput = popops::createSliceTensor(
      graph, QUARTER, {1}, {0}, {1}, 1, plan, {}, "subTWithOutput");
  graph.setInitialValue(
      subTWithOutput.getMetadata(),
      QuarterMetadata(QuarterMetadata::Format::F143, 1).getBinary());
  popops::dynamicSliceWithOutput(graph, subTWithOutput, embedding,
                                 idsDynamicSlice, {0}, {1}, sequence);

  // Create an input tensor with an associated metadata value to check
  // grouped result metadata
  unsigned groupSize = 2;
  auto groupedPlan =
      popops::embedding::plan(graph, QUARTER, groupSize, 1, 1, {1, 1}, {});
  auto grouped = popops::createGroupedSliceableTensor(
      graph, QUARTER, groupSize, tShape, {0}, {1}, groupedPlan, {},
      "embedding");
  auto groupedIds = popops::createGroupedIndicesTensor(
      graph, groupSize, {0}, numIndices, groupedPlan, {}, "ids");

  // Indices must be legal even though we're not really testing dynamic slice
  popops::fill<unsigned>(graph, groupedIds, sequence, 0u, "zero indices");

  graph.setInitialValue(
      grouped.getMetadata(),
      QuarterMetadata(QuarterMetadata::Format::F143, 1).getBinary());
  auto subTGrouped =
      popops::groupedMultiSlice(graph, grouped, groupedIds, {0}, {1}, sequence,
                                groupedPlan, optionFlags, "slice");

  graph.createHostRead("embeddingMeta", embedding.getMetadata());

  graph.createHostRead("subTMeta", subT.getMetadata());
  graph.createHostRead("subTMetaConst", subTConst.getMetadata());
  graph.createHostRead("subTMetaDynamicSlice", subTDynamicSlice.getMetadata());
  graph.createHostRead("subTWithOutput", subTWithOutput.getMetadata());
  graph.createHostRead("subTGrouped", grouped.getMetadata());

  Engine engine(graph, sequence);
  device.bind([&](const Device &d) {
    engine.load(d);
    std::vector<unsigned> hIds{0, 4, 3, 2, 1};
    engine.writeTensor("ids", &hIds.data()[0], &hIds.data()[hIds.size()]);

    engine.run(0);
    const auto numTests = 5;
    std::vector<QuarterMetadata> refMetadata(1);
    std::vector<QuarterMetadata> hostMetadata(numTests);
    engine.readTensor("embeddingMeta", &refMetadata[0], &refMetadata[1]);

    engine.readTensor("subTMeta", &hostMetadata[0], &hostMetadata[1]);
    engine.readTensor("subTMetaConst", &hostMetadata[1], &hostMetadata[2]);
    engine.readTensor("subTMetaDynamicSlice", &hostMetadata[2],
                      &hostMetadata[3]);
    engine.readTensor("subTWithOutput", &hostMetadata[3], &hostMetadata[4]);
    engine.readTensor("subTGrouped", &hostMetadata[4], &hostMetadata[5]);

    for (unsigned i = 0; i < numTests; i++) {
      BOOST_CHECK(refMetadata[0] == hostMetadata[i]);
    }
  });
}

BOOST_AUTO_TEST_CASE(CheckQuarter) { checkQuarterMetadata(); }
BOOST_AUTO_TEST_SUITE_END()
