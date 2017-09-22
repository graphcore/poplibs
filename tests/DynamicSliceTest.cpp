#define BOOST_TEST_MODULE DynamicSliceTest
#include <iostream>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/test/framework.hpp>
#include <popstd/DynamicSlice.hpp>
#include <popstd/TileMapping.hpp>
#include <popstd/codelets.hpp>
#include <poplar/Program.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Interval.hpp>
#include <util/print.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

#define NUM_DIMS 3
static const unsigned dimA = 3, dimB = 4, dimC = 2;
static float hIn[dimA][dimB][dimC] = {
  {{111, 112}, {121, 122}, {131, 132}, {141, 142}},
  {{211, 212}, {221, 222}, {231, 232}, {241, 242}},
  {{311, 312}, {321, 322}, {331, 332}, {341, 342}}};
static float hZero[dimA * dimB * dimC];
static float hOut[dimA * dimB * dimC];

static void checkResult(const float *m, const std::vector<size_t> shape,
                        const std::vector<size_t> refShape,
                        const std::vector<size_t> offsets)
{
  assert(shape.size() == 3);

  for (unsigned a = 0; a != shape[0]; ++a) {
    std::cerr << "[" << a << "] {";
    for (unsigned b = 0; b != shape[1]; ++b) {
      std::string sep = "";
      std::cerr<<"{";
      for (unsigned c = 0; c != shape[2]; ++c) {
        auto result = m[((a * shape[1]) + b) * shape[2] + c];
        auto refA = (offsets[0] + a) % refShape[0];
        auto refB = (offsets[1] + b) % refShape[1];
        auto refC = (offsets[2] + c) % refShape[2];
        std::cerr << sep <<result<<" == "<<hIn[refA][refB][refC];
        sep = ", ";

        BOOST_CHECK_EQUAL(result, hIn[refA][refB][refC]);
      }
      std::cerr<<"}";
    }
    std::cerr << "}\n";
  }
}

// Check dynamicSliceND() extracts \a sliceSizes elements from the \a sliceDims
// dimensions for all possible offsets.
void subTestND(unsigned tilesPerIPU,
               const std::vector<std::size_t> &sliceDims,
               const std::vector<std::size_t> &sliceSizes)
{
  std::cerr << "\nTest "
            << boost::unit_test::framework::current_test_case().p_name << "\n";
  DeviceInfo devInfo;
  devInfo.tilesPerIPU = tilesPerIPU;
  Graph graph(createIPUModelDevice(devInfo));
  popstd::addCodelets(graph);
  std::vector<size_t> t1Shape = {dimA, dimB, dimC};
  auto t1 = graph.addTensor("float", t1Shape, "t1");
  std::cerr<<"Created tensor t1: " << t1 << "\n";
  auto tWantedOffsets = graph.addTensor("unsigned", {sliceDims.size()},
                                        "wantedOffsets");
  graph.setTileMapping(tWantedOffsets, 0);

  // map t1's major dimension across tiles
  auto nTilesForT1 = std::min(dimA, devInfo.tilesPerIPU);
  Graph::TileToTensorMapping t1Map;
  for (unsigned a = 0; a != nTilesForT1; ++a) {
    std::vector<Interval<std::size_t>> submap;
    auto elemPerSlice = dimB * dimC;
    auto iBegin = a * elemPerSlice;
    auto iEnd = (a == nTilesForT1-1) ? t1.numElements()
                                     : iBegin + elemPerSlice;
    auto interval = Interval<std::size_t>(iBegin, iEnd);
    submap.emplace_back(interval);
    t1Map.emplace_back(submap);
  }
  graph.setTileMapping(t1, t1Map);
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
  Engine eng(graph, prog);
  eng.writeTensor("in", hIn);

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
        memcpy(hOut, hZero, sizeof(hOut));
        std::cerr<<"\nEngine run " << checkOffsets << "\n";
        eng.run();
        eng.readTensor("out", hOut);
        checkResult(hOut, tOutShape, t1Shape, checkOffsets);
      }
    }
  }
}

// Test slicing of a single dimension
BOOST_AUTO_TEST_CASE(Slice_5_0_1){
  subTestND(5, {0}, {1});
}
BOOST_AUTO_TEST_CASE(Slice_5_0_2){
  subTestND(5, {0}, {2});
}
BOOST_AUTO_TEST_CASE(Slice_5_1_1){
  subTestND(5, {1}, {1});
}
BOOST_AUTO_TEST_CASE(Slice_5_1_2){
  subTestND(5, {1}, {2});
}
BOOST_AUTO_TEST_CASE(Slice_5_2_1){
  subTestND(5, {2}, {1});
}
BOOST_AUTO_TEST_CASE(Slice_5_2_2){
  subTestND(5, {2}, {2});
}

// Multidimensional slicing

// dimensions 1 & 2
BOOST_AUTO_TEST_CASE(ND_1_1_0){
  subTestND(5, {0, 1}, {1, 1});
}
// all 3 dimensions
BOOST_AUTO_TEST_CASE(ND_1_1_1){
  subTestND(5, {0, 1, 2}, {1, 1, 1});
}
// dimensions 0 and 2, producing 2xdimBx2 output
BOOST_AUTO_TEST_CASE(ND_2_0_2){
  subTestND(5, {0, 2}, {2, 2});
}
// 2x2x2 outputs
BOOST_AUTO_TEST_CASE(ND_2_4_2){
  // The same result has as for 2_0_2 but with an extra compute set and
  // additional testing of dim1 at all 4 offsets
  subTestND(5, {0, 1, 2}, {2, 4, 2});
 }
