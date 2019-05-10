// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE ScatterTest
#include "TestDevice.hpp"

#include <iostream>

#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <popops/Scatter.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

template <typename T, std::size_t N1, std::size_t N2, std::size_t N3>
std::array<T, N1> deviceScatter(
    std::array<T, N1> in, std::vector<std::size_t> in_shape,
    std::array<int, N2> indices, std::vector<std::size_t> indices_shape,
    std::array<T, N3> updates, std::vector<std::size_t> updates_shape,
    std::size_t index_vector_dim, std::vector<unsigned> update_window_dims,
    std::vector<std::size_t> insert_window_dims,
    std::vector<unsigned> scatter_dims_to_operand_dims) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(equivalent_device_type<T>().value, in_shape);
  Tensor tIndices =
      graph.addVariable(equivalent_device_type<int>().value, indices_shape);
  Tensor tUpdates =
      graph.addVariable(equivalent_device_type<T>().value, updates_shape);

  poputil::mapTensorLinearly(graph, tIn);
  poputil::mapTensorLinearly(graph, tIndices);
  poputil::mapTensorLinearly(graph, tUpdates);

  BOOST_REQUIRE_EQUAL(tIn.numElements(), N1);
  BOOST_REQUIRE_EQUAL(tIndices.numElements(), N2);
  BOOST_REQUIRE_EQUAL(tUpdates.numElements(), N3);

  scatter(graph, tIn, tIndices, tUpdates, index_vector_dim, update_window_dims,
          insert_window_dims, scatter_dims_to_operand_dims, seq);

  graph.createHostWrite("in", tIn);
  graph.createHostWrite("indices", tIndices);
  graph.createHostWrite("update", tUpdates);
  graph.createHostRead("out", tIn);

  std::array<T, N1> out;
  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in.data(), in.data() + in.size());
    eng.writeTensor("indices", indices.data(), indices.data() + indices.size());
    eng.writeTensor("update", updates.data(), updates.data() + updates.size());
    eng.run();

    eng.readTensor("out", out.data(), out.data() + out.size());
  });

  return out;
}

BOOST_AUTO_TEST_CASE(ScatterTestCase0) {
  std::array<int, 9> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {0, 2};
  std::array<int, 6> updates = {10, 20, 30, 70, 80, 90};
  std::array<int, 9> result = {10, 20, 30, 4, 5, 6, 70, 80, 90};

  BOOST_TEST(deviceScatter(operand, {3, 3}, indices, {2}, updates, {2, 3}, 1,
                           {1}, {0}, {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase1) {
  std::array<int, 9> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {0, 1};
  std::array<int, 6> updates = {10, 20, 30, 40, 50, 60};
  std::array<int, 9> result = {10, 20, 30, 40, 50, 60, 7, 8, 9};

  BOOST_TEST(deviceScatter(operand, {3, 3}, indices, {2}, updates, {2, 3}, 1,
                           {1}, {0}, {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase2) {
  std::array<int, 9> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {0, 2};
  std::array<int, 6> updates = {{10, 30, 40, 60, 70, 90}};
  std::array<int, 9> result = {10, 2, 30, 40, 5, 60, 70, 8, 90};

  BOOST_TEST(deviceScatter(operand, {3, 3}, indices, {2}, updates, {3, 2}, 1,
                           {0}, {1}, {1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase3) {
  std::array<float, 4> operand = {0, 0, 0, 0};
  std::array<int, 3> indices = {0, 0, 0};
  std::array<float, 4> updates = {0.12, 0.28, 0.018, 0.42};
  std::array<float, 4> result = {0.12, 0.28, 0.018, 0.42};

  BOOST_CHECK(deviceScatter(operand, {1, 2, 2, 1}, indices, {1, 3}, updates,
                            {1, 2, 2, 1}, 1, {1, 2, 3}, {0},
                            {0, 2, 1}) == result);
}

BOOST_AUTO_TEST_CASE(ScatterTestCase4) {
  std::array<int, 9> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {0, 2};
  std::array<int, 6> updates = {10, 20, 30, 70, 80, 90};
  std::array<int, 9> result = {10, 20, 30, 4, 5, 6, 70, 80, 90};

  BOOST_TEST(deviceScatter(operand, {3, 3}, indices, {2}, updates, {2, 3}, 1,
                           {1}, {0}, {0}) == result);
}

BOOST_AUTO_TEST_CASE(ScatterTestCase5) {
  std::array<float, 18> operand = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                                   5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  std::array<int, 4> indices = {0, 0, 1, 0};
  std::array<float, 4> updates = {-10, 10, -40, 40};
  std::array<float, 18> result = {-10, 10, -2, 2,  -3, 3,  -40, 40, -5,
                                  5,   -6, 6,  -7, 7,  -8, 8,   -9, 9};

  BOOST_TEST(deviceScatter(operand, {3, 3, 2}, indices, {2, 2}, updates, {2, 2},
                           1, {1}, {0, 1}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase6) {
  std::array<float, 18> operand = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                                   5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  std::array<int, 4> indices = {0, 0, 0, 1};
  std::array<float, 4> updates = {-10, 10, -20, 20};
  std::array<float, 18> result = {-10, 10, -20, 20, -3, 3,  -4, 4,  -5,
                                  5,   -6, 6,   -7, 7,  -8, 8,  -9, 9};

  BOOST_TEST(deviceScatter(operand, {3, 3, 2}, indices, {2, 2}, updates, {2, 2},
                           0, {1}, {0, 1}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase7) {
  std::array<float, 9> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 2> indices = {1, 1};
  std::array<float, 1> updates = {10};
  std::array<float, 9> result = {1, 2, 3, 4, 10, 6, 7, 8, 9};

  BOOST_TEST(deviceScatter(operand, {3, 3}, indices, {2}, updates, {1, 1}, 0,
                           {0, 1}, {}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase8) {
  std::array<float, 9> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 4> indices = {2, 1, 1, 1};
  std::array<float, 2> updates = {10, 20};
  std::array<float, 9> result = {1, 2, 3, 4, 20, 6, 7, 10, 9};

  BOOST_TEST(deviceScatter(operand, {3, 3}, indices, {2, 2}, updates, {2, 1, 1},
                           0, {1, 2}, {}, {0, 1}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase9) {
  std::array<float, 0> operand = {};
  std::array<int, 2> indices = {0, 2};
  std::array<float, 0> updates = {};
  std::array<float, 0> result = {};

  BOOST_TEST(deviceScatter(operand, {3, 0}, indices, {2}, updates, {2, 0}, 1,
                           {1}, {0}, {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase10) {
  std::array<float, 12> operand = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::array<int, 1> indices = {1};
  std::array<float, 6> updates = {10, 20, 30, 40, 50, 60};
  std::array<float, 12> result = {1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60};

  BOOST_TEST(deviceScatter(operand, {2, 3, 2}, indices, {}, updates, {1, 3, 2},
                           0, {0, 1, 2}, {}, {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase11) {
  std::array<float, 4> operand = {1, 2, 3, 4};
  std::array<int, 1> indices = {1};
  std::array<float, 1> updates = {25};
  std::array<float, 4> result = {1, 25, 3, 4};

  BOOST_TEST(deviceScatter(operand, {4}, indices, {}, updates, {}, 0, {}, {0},
                           {0}) == result,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ScatterTestCase12) {
  std::array<float, 3> operand = {1, 2, 3};
  std::array<int, 0> indices = {};
  std::array<float, 0> updates = {};
  std::array<float, 3> result = {1, 2, 3};

  BOOST_TEST(deviceScatter(operand, {3}, indices, {0}, updates, {0}, 1, {}, {0},
                           {0}) == result,
             boost::test_tools::per_element());
}
