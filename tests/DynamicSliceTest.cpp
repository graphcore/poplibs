#define BOOST_TEST_MODULE DynamicSliceTest
#include <iostream>
#include <vector>
#include <sstream>
#include <boost/test/unit_test.hpp>
#include <boost/test/framework.hpp>
#include <popops/DynamicSlice.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include <poplar/Program.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Interval.hpp>
#include <poplibs_support/print.hpp>
#include <boost/multi_array.hpp>
#include <cassert>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

const OptionFlags options {
  {"target.workerStackSizeInBytes", "0x1f0"}
};

#define NUM_DIMS 3

struct TestData {
  std::vector<size_t> tDims, sDims;
  boost::multi_array<float, 3> hInit; // fullsized initialiser
  boost::multi_array<float, 3> hSub; // subTensor, either in or out
  boost::multi_array<float, 3> hUpdateOut;
  TestData(std::vector<size_t> t, std::vector<size_t> s,
           const std::vector<std::vector<std::vector<float>>> &initialData) :
      tDims(t), sDims(s) {
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
#define MAX_DELAY  200
#define ELEM_PER_TAP  4
static std::vector<size_t> delayTestShape =
  {LONG_OUTER, MAX_DELAY, ELEM_PER_TAP};
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
static void MapAcrossTiles(Graph &graph, size_t tilesPerIPU, const Tensor &t)
{
  auto nTilesForT = std::min(t.dim(0), tilesPerIPU);
  auto elemPerSlice = t.numElements() / t.dim(0);
  Graph::TileToTensorMapping map;
  for (unsigned a = 0; a != nTilesForT; ++a) {
    std::vector<Interval> submap;
    auto iBegin = a * elemPerSlice;
    {
      auto iEnd = (a == nTilesForT-1) ? t.numElements()
                                      : iBegin + elemPerSlice;
      auto interval = Interval(iBegin, iEnd);
      submap.emplace_back(interval);
      map.emplace_back(submap);
    }
  }
  graph.setTileMapping(t, map);
}

static boost::multi_array<float, 3> refSlice(
    const std::vector<size_t> &sShape,
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
        result[a][b][c]= value;
      }
    }
  }
  return result;
}

static boost::multi_array<float, 3> refUpdate(
    const boost::multi_array<float, 3> &t,
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
                        const boost::multi_array<float, 3> &ref)
{
  auto shape = m.shape();

  for (unsigned a = 0; a != shape[0]; ++a) {
    std::cerr << "[" << a << "] {";
    for (unsigned b = 0; b != shape[1]; ++b) {
      std::string sep = "";
      std::cerr<<"{";
      for (unsigned c = 0; c != shape[2]; ++c) {
        auto result = m[a][b][c];
        auto refResult = ref[a][b][c];
        std::cerr << sep << result << " == "<< refResult;
        sep = ", ";

        BOOST_CHECK_EQUAL(result, refResult);
      }
      std::cerr<<"}";
    }
    std::cerr << "}\n";
  }
}

// Check dynamicSliceND() extracts \a sliceSizes elements from the \a sliceDims
// dimensions for all possible offsets.
void sliceTestND(unsigned tilesPerIPU,
               const std::vector<size_t> &testShape,
               const std::vector<std::vector<std::vector<float>>> &testBase,
               const std::vector<std::size_t> &sliceDims,
               const std::vector<std::size_t> &sliceSizes)
{
  std::cerr << "\nTest "
            << boost::unit_test::framework::current_test_case().p_name << "\n";
  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<size_t> t1Shape = testShape;
  auto t1 = graph.addVariable(FLOAT, t1Shape, "t1");
  std::cerr<<"Created tensor t1: " << t1 << "\n";
  auto tWantedOffsets = graph.addVariable(UNSIGNED_INT, {sliceDims.size()},
                                          "wantedOffsets");
  graph.setTileMapping(tWantedOffsets, 0);

  MapAcrossTiles(graph, tilesPerIPU, t1);
  std::cerr << "t1 is " << t1
            << " mapping " << graph.getTileMapping(t1) << "\n";

  auto prog = Sequence();
  auto tOut = dynamicSlice(graph, t1, tWantedOffsets, sliceDims, sliceSizes,
                           prog, "DSND");

  const auto tOutShape = tOut.shape();
  std::cerr << "output tensor is " << tOut
            << " mapping " << graph.getTileMapping(tOut) << "\n";

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

  std::cerr << "Creating engine\n";
  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);

    TestData testData(t1Shape, wantedShape, testBase);

    eng.writeTensor("in", testData.hInit.data());

    std::vector<unsigned> nOffsets(t1.rank(), 1);
    for (auto dim : sliceDims) {
      nOffsets[dim] = t1.dim(dim);
    }
    assert(t1.rank()==NUM_DIMS);
    for (unsigned sliceA = 0; sliceA != nOffsets[0]; ++sliceA) {
      for (unsigned sliceB = 0; sliceB != nOffsets[1]; ++sliceB) {
        for (unsigned sliceC = 0; sliceC != nOffsets[2]; ++sliceC) {
          unsigned offsets[NUM_DIMS] = {sliceA, sliceB, sliceC};
          unsigned hOffsets[NUM_DIMS];
          for (unsigned i = 0; i != sliceDims.size(); ++i) {
            hOffsets[i] = offsets[sliceDims[i]];
          }
          std::vector<size_t> checkOffsets = { { sliceA, sliceB, sliceC } };
          eng.writeTensor("selector", hOffsets);
          for (unsigned i = 0; i != testData.hUpdateOut.num_elements(); ++i)
            testData.hUpdateOut.data()[i] = 0.0;
          std::cerr<<"\nEngine run " << checkOffsets << "\n";
          eng.run();
          eng.readTensor("out", testData.hSub.data());
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
                              const std::vector<std::size_t> &sliceSizes)
{
  sliceTestND(tilesPerIPU, smallTestShape, smallTestData,
            sliceDims, sliceSizes);
}

// Test empty slice
BOOST_AUTO_TEST_CASE(Slice_Empty){
  subTestSmallSlice(5, {}, {});
}

// Test slicing of a single dimension
BOOST_AUTO_TEST_CASE(Slice_5_0_1){
  subTestSmallSlice(5, {0}, {1});
}
BOOST_AUTO_TEST_CASE(Slice_5_0_2){
  subTestSmallSlice(5, {0}, {2});
}
BOOST_AUTO_TEST_CASE(Slice_5_1_1){
  subTestSmallSlice(5, {1}, {1});
}
BOOST_AUTO_TEST_CASE(Slice_5_1_2){
  subTestSmallSlice(5, {1}, {2});
}
BOOST_AUTO_TEST_CASE(Slice_5_2_1){
  subTestSmallSlice(5, {2}, {1});
}
BOOST_AUTO_TEST_CASE(Slice_5_2_2){
  subTestSmallSlice(5, {2}, {2});
}

// Multidimensional slicing

// dimensions 1 & 2
BOOST_AUTO_TEST_CASE(ND_1_1_0){
  subTestSmallSlice(5, {0, 1}, {1, 1});
}
// all 3 dimensions
BOOST_AUTO_TEST_CASE(ND_1_1_1){
  subTestSmallSlice(5, {0, 1, 2}, {1, 1, 1});
}
// dimensions 0 and 2, producing 2xdimBx2 output
BOOST_AUTO_TEST_CASE(ND_2_0_2){
  subTestSmallSlice(5, {0, 2}, {2, 2});
}
// 2x2x2 outputs
BOOST_AUTO_TEST_CASE(ND_2_4_2){
  // The same result has as for 2_0_2 but with an extra compute set and
  // additional testing of dim1 at all 4 offsets
  subTestSmallSlice(5, {0, 1, 2}, {2, 4, 2});
 }

// large-buffer update
BOOST_AUTO_TEST_CASE(circTest){
  auto delayTestData = GenDelayData();
  sliceTestND(24, delayTestShape, delayTestData,
              {1}, {1});
}

// Dynamic update
// Check dynamicSliceND() extracts \a sliceSizes elements from the \a sliceDims
// dimensions for all possible offsets.
void updateTestND(unsigned tilesPerIPU,
                  const std::vector<size_t> &testShape,
                  const std::vector<std::vector<std::vector<float>>> &testBase,
                  const std::vector<std::size_t> &sliceDims,
                  const std::vector<std::size_t> &sliceSizes)
{
  std::cerr << "\nTest "
            << boost::unit_test::framework::current_test_case().p_name << "\n";
  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  std::vector<size_t> t1Shape = testShape;
  auto t1 = graph.addVariable(FLOAT, t1Shape, "t1");
  std::cerr<<"Created tensor t1: " << t1 << "\n";

  std::vector<size_t> subShape = t1.shape();
  for (unsigned i = 0; i != sliceDims.size(); ++i) {
    subShape[sliceDims[i]] = sliceSizes[i];
  }
  auto s1 = graph.addVariable(FLOAT, subShape, "s1");
  std::cerr<<"Created tensor s1: " << s1 << "\n";
  auto tWantedOffsets = graph.addVariable(UNSIGNED_INT, {sliceDims.size()},
                                          "wantedOffsets");
  graph.setTileMapping(tWantedOffsets, 0);

  MapAcrossTiles(graph, tilesPerIPU, t1);
  MapAcrossTiles(graph, tilesPerIPU, s1);
  std::cerr << "t1 is " << t1
            << " mapping " << graph.getTileMapping(t1) << "\n";
  std::cerr << "s1 is " << s1
            << " mapping " << graph.getTileMapping(t1) << "\n";

  auto prog = Sequence();

  dynamicUpdate(graph, t1, s1, tWantedOffsets, sliceDims, sliceSizes,
                           prog, "DSUpdate");


  graph.createHostWrite("in", t1);
  graph.createHostWrite("update", s1);
  graph.createHostWrite("selector", tWantedOffsets);
  graph.createHostRead("out", t1);

  std::cerr << "Creating engine\n";
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
    eng.writeTensor("update", testData.hSub.data());

    std::vector<unsigned> nOffsets(t1.rank(), 1);
    for (auto dim : sliceDims) {
      nOffsets[dim] = t1.dim(dim);
    }
    assert(t1.rank()==NUM_DIMS);
    for (unsigned sliceA = 0; sliceA != nOffsets[0]; ++sliceA) {
      for (unsigned sliceB = 0; sliceB != nOffsets[1]; ++sliceB) {
        for (unsigned sliceC = 0; sliceC != nOffsets[2]; ++sliceC) {
          unsigned offsets[NUM_DIMS] = {sliceA, sliceB, sliceC};
          unsigned hOffsets[NUM_DIMS];
          for (unsigned i = 0; i != sliceDims.size(); ++i) {
            hOffsets[i] = offsets[sliceDims[i]];
          }
          std::vector<size_t> checkOffsets = { { sliceA, sliceB, sliceC } };
          eng.writeTensor("in", testData.hInit.data());
          eng.writeTensor("selector", hOffsets);
          for (unsigned i = 0; i != testData.hUpdateOut.num_elements(); ++i)
            testData.hUpdateOut.data()[i] = 0.0;
          std::cerr<<"\nEngine run " << checkOffsets << "\n";
          eng.run();
          eng.readTensor("out", testData.hUpdateOut.data());

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
                               const std::vector<std::size_t> &sliceSizes)
{
  updateTestND(tilesPerIPU, smallTestShape, smallTestData,
               sliceDims, sliceSizes);
}

// Test full update


BOOST_AUTO_TEST_CASE(Update_Full){
  subTestSmallSlice(5, {0, 1, 2}, smallTestShape);
}

BOOST_AUTO_TEST_CASE(Update_Empty){
  // result should be the same as the full update with offsets 0,0,0
  subTestSmallSlice(5, {}, {});
}

// Test insertion of a single dimension
BOOST_AUTO_TEST_CASE(Update_5_0){
  testSmallUpdate(5, {0}, {1});
}
BOOST_AUTO_TEST_CASE(Update_5_1){
  testSmallUpdate(5, {1}, {1});
}
BOOST_AUTO_TEST_CASE(Update_5_2){
  testSmallUpdate(5, {2}, {1});
}
// Test insertion of a single element
BOOST_AUTO_TEST_CASE(Update_5_element){
  testSmallUpdate(5, {0, 1, 2}, {1, 1, 1});
}
// Test insertion of a 2x2 element
BOOST_AUTO_TEST_CASE(Update_5_2x2){
  testSmallUpdate(5, {0, 1}, {2, 2});
}

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

  auto input = graph.addVariable(FLOAT, t1Shape, "input");
  auto offset = graph.addVariable(UNSIGNED_INT, { t1Shape.size() }, "offset");

  MapAcrossTiles(graph, graph.getTarget().getTilesPerIPU(), input);
  graph.setTileMapping(offset, 0);

  auto prog = Sequence();

  std::vector<std::size_t> sliceDims = {2, 0, 1};
  std::vector<std::size_t> sliceSizes = {9, 90, 15};

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

  OptionFlags engineOptions {
    {"showExecutionSteps", "true"},
    {"showVarStorage", "true"}
  };
  // Actually build the graph to check that it fits onto the target
  // This will fail if many edge pointers or significant exchange is required
  Engine eng(graph, prog, options);
  eng.printProfileSummary(std::cout, engineOptions);
}
