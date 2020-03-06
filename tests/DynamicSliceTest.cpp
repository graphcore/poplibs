// Copyright (c) 2017 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE DynamicSliceTest
#include "TestDevice.hpp"
#include <boost/multi_array.hpp>
#include <boost/test/framework.hpp>
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <poplar/Engine.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <poplibs_support/print.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <sstream>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

using poplibs_support::toString;

constexpr bool useDSMapper = true;

const OptionFlags options{{"target.workerStackSizeInBytes", "0x1f0"}};

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
};

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
                 const std::vector<std::size_t> &sliceSizes) {
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
  auto tOut = dynamicSlice(graph, t1, tWantedOffsets, sliceDims, sliceSizes,
                           prog, "DSND");

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
  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);

    TestData testData(t1Shape, wantedShape, testBase);

    eng.writeTensor("in", testData.hInit.data(),
                    testData.hInit.data() + t1.numElements());

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
                              const std::vector<std::size_t> &sliceSizes) {
  sliceTestND(tilesPerIPU, smallTestShape, smallTestData, sliceDims,
              sliceSizes);
}

BOOST_AUTO_TEST_SUITE(SingleDim)

// Test empty slice
BOOST_AUTO_TEST_CASE(Slice_Empty) { subTestSmallSlice(5, {}, {}); }

// Test slicing of a single dimension
BOOST_AUTO_TEST_CASE(Slice_5_0_1) { subTestSmallSlice(5, {0}, {1}); }
BOOST_AUTO_TEST_CASE(Slice_5_0_2) { subTestSmallSlice(5, {0}, {2}); }
BOOST_AUTO_TEST_CASE(Slice_5_1_1) { subTestSmallSlice(5, {1}, {1}); }
BOOST_AUTO_TEST_CASE(Slice_5_1_2) { subTestSmallSlice(5, {1}, {2}); }
BOOST_AUTO_TEST_CASE(Slice_5_2_1) { subTestSmallSlice(5, {2}, {1}); }
BOOST_AUTO_TEST_CASE(Slice_5_2_2) { subTestSmallSlice(5, {2}, {2}); }

BOOST_AUTO_TEST_SUITE_END()

// Multidimensional slicing
BOOST_AUTO_TEST_SUITE(MultiDim)

// dimensions 1 & 2
BOOST_AUTO_TEST_CASE(ND_1_1_0) { subTestSmallSlice(5, {0, 1}, {1, 1}); }
// all 3 dimensions
BOOST_AUTO_TEST_CASE(ND_1_1_1) { subTestSmallSlice(5, {0, 1, 2}, {1, 1, 1}); }
// dimensions 0 and 2, producing 2xdimBx2 output
BOOST_AUTO_TEST_CASE(ND_2_0_2) { subTestSmallSlice(5, {0, 2}, {2, 2}); }
// 2x2x2 outputs
BOOST_AUTO_TEST_CASE(ND_2_4_2) {
  // The same result has as for 2_0_2 but with an extra compute set and
  // additional testing of dim1 at all 4 offsets
  subTestSmallSlice(5, {0, 1, 2}, {2, 4, 2});
}

BOOST_AUTO_TEST_SUITE_END()

// large-buffer update
BOOST_AUTO_TEST_SUITE(LargeBuffer)

BOOST_AUTO_TEST_CASE(circTest) {
  auto delayTestData = GenDelayData();
  sliceTestND(24, delayTestShape, delayTestData, {1}, {1});
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
  Engine eng(graph, prog, options);
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

  auto d0_idx = cg.find("/dynamicSlice_d0/slice");
  auto d1_idx = cg.find("/dynamicSlice_d1/slice");
  auto d2_idx = cg.find("/dynamicSlice_d2/slice");
  const auto npos = std::string::npos;
  BOOST_CHECK(d0_idx != npos && d1_idx != npos && d2_idx != npos);
  BOOST_CHECK(d1_idx < d2_idx);
  BOOST_CHECK(d2_idx < d0_idx);
}

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

  OptionFlags engineOptions{{"showExecutionSteps", "true"},
                            {"showVarStorage", "true"}};
  // Actually build the graph to check that it fits onto the target
  // This will fail if many edge pointers or significant exchange is required
  Engine eng(graph, prog, options);

  std::stringstream ss;
  eng.printProfileSummary(ss, engineOptions);
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

BOOST_AUTO_TEST_SUITE_END()

void multislice(const std::vector<uint32_t> &indicies,
                const std::vector<std::size_t> &indiciesShape,
                bool planAsEmbedding) {
  // This test should pass with large T - but graph construction becomes
  // slow (a couple of minutes for T=1024)
  assert(indiciesShape.size() == 2); // max 2 dims supported by this test
  assert(indicies.size() == indiciesShape[0] * indiciesShape[1]);
  const auto T = 16;        // tiles
  const auto D = 1501 * 10; // dictionary size
  const auto E = 8;         // embedding size
  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<std::size_t> sliceDims{0};
  std::vector<std::size_t> sliceSizes{1};

  const auto options = OptionFlags();
  auto plan = SlicePlan();
  if (planAsEmbedding) {
    plan = embedding::plan(graph, FLOAT, D, E, {indicies.size()}, options);
  }
  // Map the tensor carefully to ensure balance and minimise edge pointers
  auto t = createSliceableTensor(graph, FLOAT, {D, E}, sliceDims, sliceSizes,
                                 plan, options, "t");
  Sequence prog;

  auto offsetInit =
      graph.addConstant(UNSIGNED_INT, indiciesShape, indicies.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indicies.size(), plan,
                                    options, "offset");
  prog.add(Copy(offsetInit, offset));
  auto s = multiSlice(graph, t, offset, sliceDims, sliceSizes, prog, plan,
                      options, "MultisliceTest");

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indiciesShape[0]);
  BOOST_CHECK_EQUAL(s.dim(1), indiciesShape[1]);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("in", t, true);
  graph.createHostRead("out", s, true);
  std::vector<uint32_t> hIn(t.numElements());
  std::vector<uint32_t> hOut(s.numElements());
  std::iota(hIn.begin(), hIn.end(), 0u);
  OptionFlags engineOptions{{"showExecutionSteps", "true"},
                            {"showVarStorage", "true"}};
  // Engine creation will fail for non-cpu targets if many edge pointers or
  // significant exchange is required; this should not happen if
  // createSliceableTensor() has given a good layout
  Engine eng(graph, prog, options);
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
    for (unsigned elem = 0; elem != E; ++elem) {
      unsigned expected = hIn[d * E + elem];
      BOOST_CHECK_EQUAL(hOut[i * E + elem], expected);
    }
  }
  std::stringstream ss;
  eng.printProfileSummary(ss, engineOptions);
  BOOST_TEST_MESSAGE(ss.str());
}

BOOST_AUTO_TEST_SUITE(MultiSlice)

// test the looping multislice
BOOST_AUTO_TEST_CASE(MultiSlice5) {
  multislice({100, 0, 50, 48, 49}, {5, 1}, false);
}

// test the inlined multislice
BOOST_AUTO_TEST_CASE(MultiSlice2) { multislice({100, 0}, {2, 1}, false); }

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiSlice10) {
  multislice({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false);
}

// test the looping multislice
BOOST_AUTO_TEST_CASE(MultiSlice5_AsEmbedding) {
  multislice({100, 0, 50, 48, 49}, {5, 1}, true);
}

// test the inlined multislice
BOOST_AUTO_TEST_CASE(MultiSlice2_AsEmbedding) {
  multislice({100, 0}, {2, 1}, true);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiSlice10_AsEmbedding) {
  multislice({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true);
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

  const auto options = OptionFlags();
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
                                    options, "offset");

  prog.add(Copy(offsetInit, offset));
  auto s = multiSlice(graph, t, offset, sliceDims, sliceSizes, prog, plan,
                      options, "multiSliceTest");

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indices.size());
  BOOST_CHECK_EQUAL(s.dim(1), 1);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("in", t, true);
  graph.createHostRead("out", s, true);
  std::vector<std::uint32_t> hIn(t.numElements());
  std::vector<std::uint32_t> hOut(t.numElements());
  std::iota(hIn.begin(), hIn.end(), 0u);

  Engine e(graph, prog, options);
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
                 bool planAsEmbedding, bool accumulate = false,
                 float updateScaling = 1.0,
                 const unsigned E = 8) // embedding size
{
  // This test should pass with large T - but graph construction becomes
  // slow (a couple of minutes for T=1024)
  assert(indiciesShape.size() == 2); // max 2 dims supported by this test
  assert(indicies.size() == indiciesShape[0] * indiciesShape[1]);
  const auto T = 16;        // tiles
  const auto D = 1501 * 10; // dictionary size
  auto device = createTestDevice(TEST_TARGET, 1, T);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<std::size_t> sliceDims{0};
  std::vector<std::size_t> sliceSizes{1};
  Tensor scale;

  const auto options = OptionFlags();
  auto plan = SlicePlan();
  if (planAsEmbedding) {
    plan = embedding::plan(graph, HALF, D, E, {indicies.size()}, options);
  }

  // Map the tensor carefully to ensure balance and minimise edge pointers
  auto t = createSliceableTensor(graph, HALF, {D, E}, sliceDims, sliceSizes,
                                 plan, options, "t");
  auto s = createSliceTensor(graph, HALF, {D, E}, sliceDims, sliceSizes,
                             indicies.size(), plan, options, "s");
  Sequence prog;
  auto offsetInit =
      graph.addConstant(UNSIGNED_INT, indiciesShape, indicies.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indicies.size(), plan,
                                    options, "offset");
  prog.add(Copy(offsetInit, offset));
  if (!accumulate) {
    multiUpdate(graph, t, s, offset, sliceDims, sliceSizes, prog, plan, options,
                "MultisliceTest");
  } else {
    scale = graph.addVariable(HALF, {}, "scale");
    graph.setTileMapping(scale, 0);
    multiUpdateAdd(graph, t, s, offset, scale, sliceDims, sliceSizes, prog,
                   plan, options, "MultisliceTest");
  }

  BOOST_CHECK_EQUAL(s.rank(), t.rank() + 1);
  BOOST_CHECK_EQUAL(s.dim(0), indiciesShape[0]);
  BOOST_CHECK_EQUAL(s.dim(1), indiciesShape[1]);
  BOOST_CHECK_EQUAL(s.dim(2), E);

  graph.createHostWrite("inS", s, true);
  graph.createHostWrite("inT", t, true);
  graph.createHostRead("outT", t, true);
  if (accumulate)
    graph.createHostWrite("scale", scale, true);
  std::vector<float> hIn(s.numElements());
  const float outBaseValue = 100.0f;
  std::vector<float> hOut(t.numElements(), outBaseValue);
  // This test checks halves - some of these entries will be >maxHalf so the
  // test may fail if large offsets are indexed
  for (unsigned i = 0; i != hIn.size(); ++i)
    hIn[i] = i + 1.0f;

  auto target = device.getTarget();
  std::vector<char> rawIn(target.getTypeSize(HALF) * hIn.size());
  std::vector<char> rawOut(target.getTypeSize(HALF) * hOut.size());
  std::vector<char> rawScaleIn(target.getTypeSize(HALF) * 1);
  poplar::copyFloatToDeviceHalf(target, &updateScaling, rawScaleIn.data(), 1);
  poplar::copyFloatToDeviceHalf(target, hIn.data(), rawIn.data(), hIn.size());
  poplar::copyFloatToDeviceHalf(target, hOut.data(), rawOut.data(),
                                hOut.size());

  OptionFlags engineOptions{{"showExecutionSteps", "true"},
                            {"showVarStorage", "true"}};
  // Engine creation will fail for non-cpu targets if many edge pointers or
  // significant exchange is required; this should not happen if
  // createSliceableTensor() has given a good layout
  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    if (accumulate) {
      eng.writeTensor("scale", rawScaleIn.data(),
                      rawScaleIn.data() + rawScaleIn.size());
    }
    eng.writeTensor("inT", rawOut.data(), rawOut.data() + rawOut.size());
    eng.writeTensor("inS", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("outT", rawOut.data(), rawOut.data() + rawOut.size());
  });
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), hOut.data(),
                                hOut.size());
  unsigned outIdx = 0;
  for (const auto &e : hOut) {
    if (e != outBaseValue)
      BOOST_TEST_MESSAGE("MUpdate Output[" << outIdx << "] = " << e);
    outIdx++;
  }
  std::vector<float> expected(t.numElements(), outBaseValue);
  for (unsigned i = 0; i != indicies.size(); ++i) {
    auto d = indicies[i];
    for (unsigned elem = 0; elem != E; ++elem) {
      if (!accumulate) {
        expected[d * E + elem] = hIn[i * E + elem];
      } else {
        expected[d * E + elem] += updateScaling * hIn[i * E + elem];
      }
    }
  }
  for (unsigned i = 0; i != expected.size(); ++i)
    BOOST_CHECK_EQUAL(hOut[i], expected[i]);

  std::stringstream ss;
  eng.printProfileSummary(ss, engineOptions);
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
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, false, true, 0.5);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2) {
  multiupdate({100, 0}, {2, 1}, false, true, 0.5);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateSingles)

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Singles) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false, true, 0.5);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MultiUpdateMultiples)

// test the fast vertex with multiple updates per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Multiples) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, false, true, 0.5,
              64);
}

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate5_AsEmbedding) {
  // Currently unhandled
  BOOST_CHECK_THROW(multiupdate({100, 0, 50, 48, 49}, {5, 1}, true),
                    poputil::poplibs_error);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdate2_AsEmbedding) {
  // Currently unhandled
  BOOST_CHECK_THROW(multiupdate({100, 0}, {2, 1}, true),
                    poputil::poplibs_error);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdate10_AsEmbedding) {
  // Currently unhandled
  BOOST_CHECK_THROW(
      multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true),
      poputil::poplibs_error);
}

// test the looping multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd5_AsEmbedding) {
  multiupdate({100, 0, 50, 48, 49}, {5, 1}, true, true, 0.5);
}

// test the inlined multiupdate
BOOST_AUTO_TEST_CASE(MultiUpdateAdd2_AsEmbedding) {
  multiupdate({100, 0}, {2, 1}, true, true, 0.5);
}

// test the fast vertex
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Singles_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true, true, 0.5);
}

// test the fast vertex with multiple updates per tile
BOOST_AUTO_TEST_CASE(MultiUpdateAdd10Multiples_AsEmbedding) {
  multiupdate({2, 1, 2, 1, 80, 70, 60, 50, 40, 30}, {10, 1}, true, true, 0.5,
              64);
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

  const auto options = OptionFlags();
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
                             indices.size(), plan, options, "s");

  Sequence prog;

  auto offsetInit = graph.addConstant(UNSIGNED_INT, {indices.size()},
                                      indices.data(), "offset");
  graph.setTileMapping(offsetInit, 0);
  auto offset = createIndicesTensor(graph, sliceDims, indices.size(), plan,
                                    options, "offset");

  prog.add(Copy(offsetInit, offset));
  auto scale = graph.addVariable(FLOAT, {}, "scale");
  graph.setTileMapping(scale, 0);
  if (accumulate) {
    multiUpdateAdd(graph, t, s, offset, scale, sliceDims, sliceSizes, prog,
                   plan, options, "multiUpdateAddTest");
  } else {
    multiUpdate(graph, t, s, offset, sliceDims, sliceSizes, prog, plan, options,
                "multiUpdateTest");
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

  Engine e(graph, prog, options);
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
      {"planConstraints",
       "{\"unslicedDimSplit\": 4, \"slicedDimSplit\": 4, \"lookupSplit\": 1}"}};

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

  Engine e(graph, prog, options);
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

  const auto &graphProfile = e.getGraphProfile();
  const auto exchangeCodeBytes =
      graphProfile["memory"]["byCategory"]["internalExchangeCode"]
                  ["nonInterleaved"]["nonOverlapped"]
                      .sumUint() +
      graphProfile["memory"]["byCategory"]["internalExchangeCode"]
                  ["interleaved"]["nonOverlapped"]
                      .sumUint() +
      graphProfile["memory"]["byCategory"]["internalExchangeCode"]["overflowed"]
                  ["nonOverlapped"]
                      .sumUint();
  const auto vertexStateBytes =
      graphProfile["memory"]["byCategory"]["vertexInstanceState"]
                  ["nonInterleaved"]["nonOverlapped"]
                      .sumUint() +
      graphProfile["memory"]["byCategory"]["copyDescriptor"]["nonInterleaved"]
                  ["nonOverlapped"]
                      .sumUint() +
      graphProfile["memory"]["byCategory"]["vectorListDescriptor"]
                  ["nonInterleaved"]["nonOverlapped"]
                      .sumUint() +
      graphProfile["memory"]["byCategory"]["vertexFieldData"]["nonInterleaved"]
                  ["nonOverlapped"]
                      .sumUint();
  BOOST_TEST_MESSAGE("Total exchange code is " << exchangeCodeBytes);
  BOOST_TEST_MESSAGE("Total vertex state bytes is " << vertexStateBytes);
  BOOST_TEST_MESSAGE("Total exchange code and vertex state bytes is "
                     << exchangeCodeBytes + vertexStateBytes);

  // measuredBytesWithOptimisation indicates the bytes for exchange code and
  // vertex state assuming we rearrange inputs to the update before
  // broadcasting. This was measured at the time of implementation.
  constexpr static unsigned measuredBytesWithOptimisation = 6799;
  BOOST_CHECK_LE(exchangeCodeBytes + vertexStateBytes,
                 measuredBytesWithOptimisation * 2u);

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
  Graph graph(device, 0, poplar::replication_factor(1));
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
