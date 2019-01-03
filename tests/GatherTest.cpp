// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE GatherTest
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

template <typename T, std::size_t N1, std::size_t N2>
std::vector<T>
deviceGather(std::array<T, N1> in, std::vector<std::size_t> in_shape,
             std::array<int, N2> indices,
             std::vector<std::size_t> indices_shape,
             std::size_t index_vector_dim, std::vector<std::size_t> offset_dims,
             std::vector<std::size_t> slice_sizes,
             std::vector<std::size_t> collapsed_slice_dims,
             std::vector<unsigned> start_index_map) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(equivalent_device_type<T>().value, in_shape);
  Tensor tIndices =
      graph.addVariable(equivalent_device_type<int>().value, indices_shape);

  poputil::mapTensorLinearly(graph, tIn);
  poputil::mapTensorLinearly(graph, tIndices);

  BOOST_REQUIRE_EQUAL(tIn.numElements(), N1);
  BOOST_REQUIRE_EQUAL(tIndices.numElements(), N2);

  poplar::Tensor tOut =
      gather(graph, tIn, tIndices, index_vector_dim, offset_dims, slice_sizes,
             collapsed_slice_dims, start_index_map, seq);

  graph.createHostWrite("in", tIn);
  graph.createHostWrite("indices", tIndices);
  graph.createHostRead("out", tOut);

  Engine eng(graph, seq);
  std::vector<T> out(tOut.numElements());
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in.data());
    eng.writeTensor("indices", indices.data());
    eng.run();

    eng.readTensor("out", out.data());
  });

  return out;
}

BOOST_AUTO_TEST_CASE(GatherTestCase0) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {0, 2};
  std::vector<int> result = {1, 2, 3, 7, 8, 9};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2}, 1, {1}, {1, 3}, {0},
                          {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase1) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {0, 2};
  std::vector<int> result = {1, 3, 4, 6, 7, 9};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2}, 1, {0}, {3, 1}, {1},
                          {1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase2) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 4> indices = {0, 2, 2, 1};
  std::vector<int> result = {1, 3, 4, 6, 7, 9, 3, 2, 6, 5, 9, 8};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2}, 2, {1}, {3, 1}, {1},
                          {1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase3) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 8> indices = {0, 2, 2, 1, 1, 2, 2, 0};
  std::vector<int> result = {3, 8, 6, 7};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2, 2}, 2, {}, {1, 1},
                          {0, 1}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase4) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 8> indices = {0, 2, 2, 1, 1, 2, 2, 0};
  std::vector<int> result = {3, 8, 6, 7};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2, 2}, 2, {1, 2}, {1, 1},
                          {}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase5) {
  std::array<int, 18> input = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                               5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  std::array<int, 4> indices = {0, 0, 1, 0};
  std::vector<int> result = {-1, 1, -4, 4};

  BOOST_TEST(deviceGather(input, {3, 3, 2}, indices, {2, 2}, 1, {1}, {1, 1, 2},
                          {0, 1}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase6) {
  std::array<int, 18> input = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                               5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  std::array<int, 4> indices = {0, 0, 1, 0};
  std::vector<int> result = {-2, 2, -1, 1};

  BOOST_TEST(deviceGather(input, {3, 3, 2}, indices, {2, 2}, 0, {1}, {1, 1, 2},
                          {0, 1}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase7) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 4> indices = {2, 1, 1, 1};
  std::vector<int> result = {8, 5};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {2, 2}, 0, {1, 2}, {1, 1}, {},
                          {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase8) {
  std::array<int, 0> input = {};
  std::array<int, 2> indices = {0, 2};
  std::vector<int> result = {};

  BOOST_TEST(deviceGather(input, {3, 0}, indices, {2}, 1, {1}, {1, 0}, {0},
                          {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase9) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 12> indices = {2, 7, 2, 1, 1, 1, 5, 1, 2147483647, 1, 1, 2};
  std::vector<int> result = {7, 8, 5, 2, 2, 6};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {6, 2}, 1, {1, 2}, {1, 1}, {},
                          {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase10) {
  std::array<int, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 12> indices = {2,    -2, 2,           1, 1, 1,
                                 -500, 1,  -2147483648, 1, 1, 2};
  std::vector<int> result = {7, 8, 5, 2, 2, 6};

  BOOST_TEST(deviceGather(input, {3, 3}, indices, {6, 2}, 1, {1, 2}, {1, 1}, {},
                          {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GatherTestCase11) {
  std::array<int, 12> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::array<int, 1> indices = {1};
  std::vector<int> result = {7, 8, 9, 10, 11, 12};

  BOOST_TEST(deviceGather(input, {2, 3, 2}, indices, {}, 0, {0, 1, 2},
                          {1, 3, 2}, {}, {0}) == result,
             boost::test_tools::per_element());
}
