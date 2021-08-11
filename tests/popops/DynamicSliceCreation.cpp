// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DynamicSliceCreation
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <pva/pva.hpp>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;

int testDim(unsigned sliceDim, const Type &type) {
  auto device = createTestDeviceFullSize(TEST_TARGET);
  Target target = device.getTarget();

  std::cout << "Creating graph" << std::endl;
  Graph graph(target);
  popops::addCodelets(graph);

  Sequence seq;
  const size_t DIM = 812;
  std::vector<size_t> inputShape = {15, DIM, 512};
  std::vector<size_t> sliceDims = {sliceDim};
  std::vector<size_t> sliceSizes = {1};
  auto sliceIndices = graph.addVariable(UNSIGNED_INT, {1});
  graph.setTileMapping(sliceIndices, 0);

  auto input = popops::createSliceableTensor(graph, type, inputShape, sliceDims,
                                             sliceSizes, 0, "input");
  // There should be no more than a single region per tile
  BOOST_CHECK(input.slice(0, 1, sliceDim).getVarRegions().size() <=
              target.getNumTiles());

  {
    auto start = std::chrono::high_resolution_clock::now();
    auto sliced = popops::dynamicSlice(graph, input, sliceIndices, sliceDims,
                                       sliceSizes, seq, "dyn_slice");
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    BOOST_TEST_MESSAGE("Time for dynamicSlice " +
                       std::to_string(elapsed.count()) + " s");
  }

  {
    std::vector<size_t> slice_shape = inputShape;
    slice_shape[sliceDim] = 1;
    auto update = popops::createSliceableTensor(
        graph, type, slice_shape, sliceDims, sliceSizes, 0, "update");
    auto start = std::chrono::high_resolution_clock::now();
    popops::dynamicUpdate(graph, input, update, sliceIndices, sliceDims,
                          sliceSizes, seq, "dyn_update_slice");
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    BOOST_TEST_MESSAGE("Time for dynamicUpdate " +
                       std::to_string(elapsed.count()) + " s");
  }

  // flat input as host-writable to ensure it is always live
  graph.createHostWrite("in", input, true);

  OptionFlags options{{"showVarStorage", "true"}};
  // Actually create the engine just to check that memory has not exploded
  Engine eng(graph, seq);
  bool verbose = false;
  if (verbose)
    eng.printProfileSummary(std::cerr, options);
  const auto tile0Memory =
      eng.getReport().compilation().tiles()[0].memory().total().excludingGaps();
  BOOST_TEST_FRAMEWORK_MESSAGE("blah");
  BOOST_TEST_MESSAGE("Tile0 memory = " + std::to_string(tile0Memory));
  BOOST_CHECK(tile0Memory <= input.numElements() + 10240);

  return 0;
}

BOOST_AUTO_TEST_CASE(HalfDim0,
                     *boost::unit_test::precondition(enableIfIpuModel())) {
  testDim(0, HALF);
}

BOOST_AUTO_TEST_CASE(HalfDim1,
                     *boost::unit_test::precondition(enableIfIpuModel())) {
  testDim(1, HALF);
}

BOOST_AUTO_TEST_CASE(HalfDim2,
                     *boost::unit_test::precondition(enableIfIpuModel())) {
  testDim(2, HALF);
}

BOOST_AUTO_TEST_CASE(UlongLongDim2,
                     *boost::unit_test::precondition(enableIfIpuModel())) {
  testDim(2, UNSIGNED_LONGLONG);
}
