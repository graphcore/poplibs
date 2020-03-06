// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE ScatterTest
#include "TestDevice.hpp"

#include <iostream>

#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
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

  popops::UpdateComputationFunc update = [](Graph &g, Tensor &a, Tensor &b,
                                            Sequence &prog) {
    return add(g, a, b, prog);
  };

  scatter(graph, tIn, tIndices, tUpdates, index_vector_dim, update_window_dims,
          insert_window_dims, scatter_dims_to_operand_dims, update, seq);

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

// Test that the scatter sums the elements of the update tensor
BOOST_AUTO_TEST_CASE(ScatterUpdateTestCase0) {
  std::array<int, 1> operand = {0};
  std::array<int, 9> indices = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<int, 9> updates = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<int, 1> result = {45};

  BOOST_TEST(deviceScatter(operand, {1}, indices, {9}, updates, {9}, 1, {1},
                           {0}, {0}) == result,
             boost::test_tools::per_element());
}
