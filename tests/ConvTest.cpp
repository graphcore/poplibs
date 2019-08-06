#define BOOST_TEST_MODULE ConvTest
#include <boost/test/unit_test.hpp>
#include <poplin/Convolution.hpp>
#include "ConvUtilInternal.hpp"
#include "poputil/TileMapping.hpp"
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplin;

BOOST_AUTO_TEST_CASE(MappingSplitOutChansSerially) {
  constexpr std::size_t numTiles = 16;
  constexpr std::size_t split = 4;
  constexpr std::size_t numOutChans = 16;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  static_assert(numOutChans % split == 0, "output channels must be evenly "
                                          "divisible by the serial split");
  const auto params = ConvParams(HALF,
                                 4 /* batchSize */,
                                 {8, 8} /* inputFieldShape */,
                                 {3, 3} /* kernelShape */,
                                 16 /* inputChannels */,
                                 numOutChans /* outputChannels */,
                                 1 /* numConvGroups */);

  const OptionFlags options{
    {"planConstraints",
      R"({"0":{"partition":{"outChanSplit":{"serial":)" +
      std::to_string(split) +
      R"(}}}})",
    },
    {"enableSerialConvolutions", "true"}
  };

  PlanningCache cache;
  const auto weights = createWeights(graph, params, "weights", options, &cache);

  std::stringstream ss;
  reportPlanInfo(ss, graph, params, options, &cache);
  BOOST_TEST_MESSAGE(ss.str());

  // Check each of the splits in the serial partition has the same mapping
  // for efficient dynamic slicing.
  const auto outChansPerSplit = numOutChans / split;
  const auto referenceMapping =
    graph.getTileMapping(weights.slice(0, outChansPerSplit, 1));
  for (std::size_t s = 1; s < split; ++s) {
    const auto slice = weights.slice(s * outChansPerSplit,
                                     (s + 1) * outChansPerSplit, 1);
    BOOST_CHECK(graph.getTileMapping(slice) == referenceMapping);
  }
}

BOOST_AUTO_TEST_CASE(CreateSliceableOutputFromSlice) {
  // This test checks the expected properties of the returned tensor from
  // createSliceableOutputFromSlice which are that the tensor has the
  // same contiguous regions in each slice on each tile, and that
  // the mapping of each individual slice to tiles is the same as that
  // of the given reference slice.
  constexpr std::size_t numTiles = 64;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());
  auto slice = graph.addVariable(HALF, {20, 10, 10}, "s");

  // Shuffle contiguous regions.
  slice = slice.dimShuffle({2,0,1});
  // Map shuffled stuff.
  poputil::mapTensorLinearly(graph, slice, 1, 1);

  const std::size_t dim = 1;
  const std::size_t numSlices = 3;
  std::vector<std::size_t> expectedShape = slice.shape();
  expectedShape[dim] *= numSlices;
  auto t = createSliceableOutputFromSlice(graph, slice, dim, numSlices, "t");
  const auto tShape = t.shape();

  BOOST_CHECK_EQUAL(t.numElements(), slice.numElements() * numSlices);
  BOOST_CHECK_EQUAL_COLLECTIONS(tShape.begin(), tShape.end(),
                                expectedShape.begin(), expectedShape.end());
  std::vector<Tensor> slices(numSlices);
  std::vector<Tensor*> slicePtrs(numSlices);
  for (std::size_t i = 0; i < 3; ++i) {
    slices[i] = t.slice(i * 20, (i + 1) * 20, 1).flatten();
    slicePtrs[i] = &slices[i];
  }
  auto sliceFlat = slice.flatten();
  graph.reorderToSimplify(&sliceFlat, slicePtrs);
  const auto referenceMapping = graph.getTileMapping(sliceFlat);
  for (std::size_t i = 0; i < 3; ++i) {
    // Expect each slice to be contiguous on each tile when reordered
    // to be contiguous with respect to the reference slice on each tile.
    const auto tSlice = slices[i];
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
