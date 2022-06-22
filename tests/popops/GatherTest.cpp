// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GatherTest
#include <poplibs_support/TestDevice.hpp>

#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <poplibs_test/TempDir.hpp>
#include <popops/Gather.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;

template <typename T>
std::vector<T>
deviceGather(const std::vector<T> &in, const std::vector<std::size_t> &in_shape,
             const std::vector<int> &indices,
             const std::vector<std::size_t> &indices_shape,
             std::size_t index_vector_dim,
             const std::vector<std::size_t> &offset_dims,
             const std::vector<std::size_t> &slice_sizes,
             const std::vector<std::size_t> &collapsed_slice_dims,
             const std::vector<unsigned> &start_index_map,
             const std::vector<std::size_t> &out_shape,
             // expected {outputTiles, elemPerTile}; not checked if zero
             const std::pair<unsigned, unsigned> &expected = {0, 0},
             unsigned tileCount = 4) {
  auto device = createTestDevice(TEST_TARGET, 1, tileCount);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tIn = createGatherInput(graph, equivalent_device_type<T>().value,
                                 in_shape, slice_sizes, start_index_map);

  BOOST_TEST(in_shape == tIn.shape(), boost::test_tools::per_element());
  Tensor tIndices = graph.addVariable(equivalent_device_type<unsigned>().value,
                                      indices_shape);

  mapTensorLinearly(graph, tIndices);

  BOOST_REQUIRE_EQUAL(tIndices.numElements(), indices.size());

  poplar::Tensor tOut =
      gather(graph, tIn, tIndices, index_vector_dim, offset_dims, slice_sizes,
             collapsed_slice_dims, start_index_map, seq, {},
             {{"remapOutOfBoundIndices", "true"}});

  auto outMapping = graph.getTileMapping(tOut, true);

  if (expected.first != 0) {
    // typically the output for each slice to be one tile per element; each tile
    // will have one output for each slice performed. This may not hold when
    // multiple dimensions are sliced
    auto expectedNumTiles = expected.first;
    auto expectedElemPerTile = expected.second;

    unsigned nOutTiles = 0;
    for (const auto &tileMapping : outMapping) {
      if (tileMapping.size() == 0)
        continue;
      ++nOutTiles;
      unsigned nElements = 0;
      for (auto &interval : tileMapping) {
        nElements += interval.size();
      }

      // The following check assumes that the output is mapped one element per
      // tile. This is not necessary when elements are sub-word or multiple
      // elements are allocated per tile
      BOOST_CHECK(nElements == expectedElemPerTile);
    }
    BOOST_CHECK(nOutTiles == expectedNumTiles);
  }

  BOOST_TEST(out_shape == tOut.shape(), boost::test_tools::per_element());

  graph.createHostWrite("in", tIn, true);
  graph.createHostWrite("indices", tIndices, true);
  graph.createHostRead("out", tOut, true);

  Engine eng(graph, seq);
  std::vector<T> out(tOut.numElements());
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in.data(), in.data() + in.size());
    eng.writeTensor("indices", indices.data(), indices.data() + indices.size());
    eng.run();

    eng.readTensor("out", out.data(), out.data() + out.size());
  });

  return out;
}

BOOST_AUTO_TEST_CASE(GatherTestCase0) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {0, 2};
  std::vector<int> result = {1, 2, 3, 7, 8, 9};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2}, 1, {1}, {1, 3}, {0}, {0},
                          {2, 3}, {3, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase1) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {0, 2};
  std::vector<int> result = {1, 3, 4, 6, 7, 9};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2}, 1, {0}, {3, 1}, {1}, {1},
                          {3, 2}, {3, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase2) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {0, 2, 2, 1};
  std::vector<int> result = {1, 3, 4, 6, 7, 9, 3, 2, 6, 5, 9, 8};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2}, 2, {1}, {3, 1}, {1},
                          {1}, {2, 3, 2}, {3, 4}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase3) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {0, 2, 2, 1, 1, 2, 2, 0};
  std::vector<int> result = {3, 8, 6, 7};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2, 2}, 2, {}, {1, 1},
                          {0, 1}, {0, 1}, {2, 2}, {1, 4}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase4) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {0, 2, 2, 1, 1, 2, 2, 0};
  std::vector<int> result = {3, 8, 6, 7};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2, 2}, 2, {1, 2}, {1, 1},
                          {}, {0, 1}, {2, 1, 1, 2}, {1, 4}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase5) {
  std::vector<int> input = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                            5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  std::vector<int> indices = {0, 0, 1, 0};
  std::vector<int> result = {-1, 1, -4, 4};

  BOOST_TEST(deviceGather(input, {3, 3, 2}, indices, {2, 2}, 1, {1}, {1, 1, 2},
                          {0, 1}, {0, 1}, {2, 2}, {2, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase6) {
  std::vector<int> input = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                            5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  std::vector<int> indices = {0, 0, 1, 0};
  std::vector<int> result = {-2, 2, -1, 1};

  BOOST_TEST(deviceGather(input, {3, 3, 2}, indices, {2, 2}, 0, {1}, {1, 1, 2},
                          {0, 1}, {0, 1}, {2, 2}, {2, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase7) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {2, 1, 1, 1};
  std::vector<int> result = {8, 5};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2}, 0, {1, 2}, {1, 1}, {},
                          {0, 1}, {2, 1, 1}, {1, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase8) {
  std::vector<int> input = {};
  std::vector<int> indices = {0, 2};
  std::vector<int> result = {};

  BOOST_TEST(deviceGather(input, {3, 0}, indices, {2}, 1, {1}, {1, 0}, {0}, {0},
                          {2, 0}, {0, 0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase9) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {2, 7, 2, 1, 1, 1, 5, 1, 2147483647, 1, 1, 2};
  std::vector<int> result = {7, 8, 5, 2, 2, 6};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {6, 2}, 1, {1, 2}, {1, 1}, {},
                          {0, 1}, {6, 1, 1}, {1, 6}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase10) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {2, -2, 2, 1, 1, 1, -500, 1, -2147483648, 1, 1, 2};
  std::vector<int> result = {7, 8, 5, 2, 2, 6};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {6, 2}, 1, {1, 2}, {1, 1}, {},
                          {0, 1}, {6, 1, 1}, {1, 6}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase11) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> indices = {1};
  std::vector<int> result = {7, 8, 9, 10, 11, 12};

  BOOST_TEST(deviceGather(input, {2, 3, 2}, indices, {}, 0, {0, 1, 2},
                          {1, 3, 2}, {}, {0}, {1, 3, 2}, {3, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase12) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> indices = {1, 1, 2, 1};
  std::vector<int> result = {5, 8};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2}, 1, {}, {1, 1}, {0, 1},
                          {0, 1}, {2}, {1, 2}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase13) {
  // clang-format off
  std::vector<int> input = {
    1, 9, 17,
    5, 13, 21,

    2, 10, 18,
    6, 14, 22,

    3, 11, 19,
    7, 15, 23,

    4, 12, 20,
    8, 16, 24
  };
  std::vector<int> indices = {
    0, 2,
    1, 0,
    2, 3
  };
  std::vector<int> result = {
    3,  9, 20,
    7, 13, 24
  };
  // clang-format on

  BOOST_TEST(result == deviceGather(input, {4, 2, 3, 1, 1}, indices, {1, 3, 2},
                                    2, {1, 3, 4}, {1, 2, 1, 1, 1}, {0, 2},
                                    {2, 0}, {1, 2, 3, 1, 1}),
             boost::test_tools::per_element());
}

// Just check the shape matches
// Case spotted in //tensorflow/compiler/tests:reverse_sequence_op_test_poplar
BOOST_AUTO_TEST_CASE(GatherTestCase_TF_reverse_sequence_op_shape) {
  std::vector<int> input(48);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> indices(6);
  std::array<int, 24> result = {0,  0,  0,  3,  3,  3,  6,  6,  6,  9,  9,  9,
                                12, 12, 12, 15, 15, 15, 18, 18, 18, 21, 21, 21};

  BOOST_TEST(result == deviceGather(input, {8, 2, 3, 1, 1}, indices, {2, 3}, 0,
                                    {0, 1, 3, 4}, {4, 2, 1, 1, 1}, {2}, {2, 0},
                                    {4, 2, 3, 1, 1}, {2, 12}),
             boost::test_tools::per_element());
}
BOOST_AUTO_TEST_CASE(GatherTestCase14) {
  const unsigned nRows = 5;
  const unsigned nCols = 7;
  const unsigned nOut = 8;
  std::vector<int> input(nRows * nCols);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> indices = {1, 2, 4, 0, 3, 1, 2, 4};
  // clang-format off
  std::vector<int> expected = {
    7,  8,  9,  10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20,
    28, 29, 30, 31, 32, 33, 34,
    0,  1,  2,  3,  4,  5,  6,
    21, 22, 23, 24, 25, 26, 27,
    7,  8,  9,  10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20,
    28, 29, 30, 31, 32, 33, 34,
  };
  // clang-format on
  auto result = deviceGather(input, {nRows, nCols}, indices, {nOut}, 1, {1},
                             {1, nCols}, {0}, {0}, {nOut, nCols});
  BOOST_TEST(result == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase15) {
  const unsigned shrink = 8; // set to 1 for full-size test
  const unsigned nRows = 16667;
  const unsigned nCols = 1200 / shrink;
  static_assert(nCols * shrink == 1200, "shrink must divide 1200");
  const unsigned nOut = 75;
  std::vector<int> input(nRows * nCols);
  std::vector<int> expected(nOut * nCols, 1);

  std::iota(input.begin(), input.end(), 0);

  std::vector<int> indices(nOut, 1);

  for (unsigned o = 0; o != nOut; ++o)
    for (unsigned col = 0; col != nCols; ++col)
      expected[o * nCols + col] = input[indices[o] * nCols + col];
  const unsigned wantedTiles = []() {
    auto device = createTestDeviceFullSize(TEST_TARGET);
    return device.getTarget().getTilesPerIPU() / shrink;
  }();
  auto result =
      deviceGather(input, {nRows, nCols}, indices, {nOut}, 1, {1}, {1, nCols},
                   {0}, {0}, {nOut, nCols}, {nCols, nOut}, wantedTiles);

  BOOST_TEST(result == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase16) {
  const unsigned nBatch = 2;
  const unsigned nRows = 3;
  const unsigned nCols = 7;
  const unsigned nOut = 2;
  std::vector<int> input(nBatch * nRows * nCols);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> indices = {1, 3};
  // clang-format off
  std::vector<int> expected = {
    1, 3,
    8, 10,
    15, 17,
    22, 24,
    29, 31,
    36, 38,
  };
  // clang-format on
  auto result =
      deviceGather(input, {nBatch, nRows, nCols}, indices, {nOut}, 1, {0, 1},
                   {nBatch, nRows, 1}, {2}, {2}, {nBatch, nRows, nOut});
  BOOST_TEST(result == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCaseBatchDimInMiddle) {
  const unsigned nBatch = 2;
  const unsigned nRows = 3;
  const unsigned nCols = 4;
  const unsigned nOut = 2;
  std::vector<int> input(nBatch * nRows * nCols);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> indices = {1, 0};
  // clang-format off
  std::vector<int> expected = {
     4,  5,  6,  7,
     0,  1,  2,  3,
    16, 17, 18, 19,
    12, 13, 14, 15
  };
  // clang-format on
  auto result =
      deviceGather(input, {nBatch, nRows, nCols}, indices, {nOut}, 1, {0, 1, 3},
                   {nBatch, 1, nCols}, {}, {1}, {nBatch, 1, nOut, nCols});
  BOOST_TEST(result == expected, boost::test_tools::per_element());
}
