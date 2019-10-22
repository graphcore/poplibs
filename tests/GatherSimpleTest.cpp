// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE GatherSimpleTest
#include "TestDevice.hpp"

#include <iostream>

#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <popops/Gather.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

template <typename T>
std::vector<T> deviceGather(const std::vector<T> &in,
                            const std::vector<std::size_t> &in_shape,
                            const std::vector<int> &indices,
                            const std::vector<std::size_t> &indices_shape,
                            unsigned axis, unsigned tile_count = 4) {
  auto device = createTestDevice(TEST_TARGET, 1, tile_count);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tIn = createGatherInput(graph, equivalent_device_type<T>().value,
                                 in_shape, axis, {}, "tIn");
  Tensor tIndices = graph.addVariable(equivalent_device_type<unsigned>().value,
                                      indices_shape, "tIndices");

  poputil::mapTensorLinearly(graph, tIndices);

  BOOST_REQUIRE_EQUAL(tIn.numElements(), in.size());
  BOOST_REQUIRE_EQUAL(tIndices.numElements(), indices.size());

  poplar::Tensor tOut = gather(graph, tIn, tIndices, axis, seq, {});

  graph.createHostWrite("in", tIn, true);
  graph.createHostWrite("indices", tIndices);
  graph.createHostRead("out", tOut, true);

  Engine eng(graph, seq);
  std::vector<T> out(tOut.numElements());
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in.data());
    eng.writeTensor("indices", indices.data());
    eng.run();

    eng.readTensor("out", out.data());
    eng.printProfileSummary(std::cout);
  });

  return out;
}
BOOST_AUTO_TEST_CASE(GatherSimpleTestCase0) {
  // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // [0, 2]
  std::vector<int> indices = {0, 2};
  // [[1, 2, 3], [7, 8, 9]]
  std::vector<int> result = {1, 2, 3, 7, 8, 9};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2}, 0) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherSimpleTestCase1) {
  // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // [0, 2]
  std::vector<int> indices = {0, 2};
  // [[1, 3], [4, 6], [7, 9]]
  std::vector<int> result = {1, 3, 4, 6, 7, 9};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2}, 1) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherSimpleTestCase2) {
  // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // [0, 2, 2, 1]
  std::vector<int> indices = {0, 2, 2, 1};
  // [[1, 2, 3], [7, 8, 9], [7, 8, 9], [4, 5, 6]]
  std::vector<int> result = {1, 2, 3, 7, 8, 9, 7, 8, 9, 4, 5, 6};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {4}, 0) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherSimpleTestCase3) {
  // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // [0, 2, 2, 1]
  std::vector<int> indices = {0, 2, 2, 1};
  // [[1, 3, 3, 2], [4, 6, 6, 5], [7, 9, 9, 8]]
  std::vector<int> result = {1, 3, 3, 2, 4, 6, 6, 5, 7, 9, 9, 8};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {4}, 1) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherSimpleTestCase4) {
  // [[[0], [1], ..., [67998], [67999]]]
  std::vector<int> input(68000);
  std::iota(input.begin(), input.end(), 0);
  // [0, 2]
  std::vector<int> indices = {0, 2};
  // [[[0], [2]]]
  std::vector<int> result = {0, 2};

  BOOST_TEST(deviceGather(input, {1, 68000, 1}, indices, {2}, 1) == result,
             boost::test_tools::per_element());
}

// A large example for profiling
// Change the `createTestDevice(TEST_TARGET, 1, 4);` to
// `createTestDevice(TEST_TARGET, 1, 1216);`
BOOST_AUTO_TEST_CASE(GatherSimpleBigTestCase) {
  const unsigned testBackoff = 8;      // reduce to 1 for a fullsized test
  const unsigned dictSize = 48 * 1024; // dictSize=192kB as they're ints
  const unsigned embeddingSize = 1000 / testBackoff;
  std::vector<int> input(dictSize * embeddingSize);
  std::vector<int> indices = {0, 1};
  deviceGather(input, {dictSize, embeddingSize}, indices, {2}, 0,
               1216 / testBackoff);
}

BOOST_AUTO_TEST_CASE(GatherSimpleBigTestCase2) {
  const unsigned dictSize = 16667;
  const unsigned embeddingSize = 1200 / 16;
  std::vector<int> input(dictSize * embeddingSize);
  std::vector<int> indices(50);
  std::iota(indices.begin(), indices.end(), 1);
  deviceGather(input, {dictSize, embeddingSize}, indices, {50}, 0, 1216 / 16);
}
