// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the Dynamic Slice adn Dynamic Slice update 2d vertices
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#define BOOST_TEST_MODULE MultiSliceCodeletTest
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;

// Define a number of tests to run:
struct TestParams {
  unsigned rows;
  unsigned columns;
  unsigned baseOffset;
  unsigned numBaseElements;
  unsigned regionSize;
  bool update;
};

std::vector<TestParams> TestList = {
    {100, 8, 1, 80, 8, false},
    {80, 7, 1, 80, 7, false},
    {100, 8, 1, 80, 8, true},
    {80, 7, 1, 80, 7, false},
};

std::vector<unsigned> offsetsTest = {2, 1, 2, 1, 80, 0, 60, 55, 40, 30};

//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void MultiSliceHost(std::vector<unsigned> &offsets, std::vector<double> &baseT,
                    std::vector<double> &subT, unsigned baseOffset,
                    unsigned numBaseElements, unsigned short regionSize) {
  for (unsigned o = 0; o != offsets.size(); ++o) {
    auto baseIdx = offsets[o];
    if (baseIdx < baseOffset || baseIdx >= baseOffset + numBaseElements) {
      // this slice is not a part of baseT so we can skip it.
      continue;
    }
    baseIdx -= baseOffset;

    for (unsigned e = 0; e != regionSize; ++e) {
      subT[o * regionSize + e] = baseT[baseIdx * regionSize + e];
    }
  }
}
//*************************************************
// C test function, based on the original C version of the vertex
// Similar to above, but copy in the opposite direction
//*************************************************
void MultiUpdateHost(std::vector<unsigned> &offsets, std::vector<double> &baseT,
                     std::vector<double> &subT, unsigned baseOffset,
                     unsigned numBaseElements, unsigned short regionSize) {

  for (unsigned o = 0; o != offsets.size(); ++o) {
    auto baseIdx = offsets[o];
    if (baseIdx < baseOffset || baseIdx >= baseOffset + numBaseElements) {
      // this slice is not a part of baseT so we can skip it.
      continue;
    }
    baseIdx -= baseOffset;

    for (unsigned e = 0; e != regionSize; ++e) {
      baseT[baseIdx * regionSize + e] = subT[o * regionSize + e];
    }
  }
}
//*************************************************
// Main Test function for MultiSlice, MultiUpdate
//
// Overview:
//
// Output memory space is initialised as all zero.
// Input memory space is intitalised with a simple test pattern
// Run a series of tests that copy a varying number of items.
// The results are put into a memory area large enough to
// hold the largest test result, so often the other items are
// expected to be zero.  This is checked as well as the "wanted" data.
//*************************************************
void MultiSliceCodeletTest(const Type &dataType) {

  // determine the sizes of arrays required
  auto test_count = TestList.size();

  const auto maxRows = std::max_element(TestList.begin(), TestList.end(),
                                        [](TestParams &a, TestParams &b) {
                                          return (a.rows < b.rows);
                                        })
                           ->rows;

  const auto maxColumns = std::max_element(TestList.begin(), TestList.end(),
                                           [](TestParams &a, TestParams &b) {
                                             return (a.columns < b.columns);
                                           })
                              ->columns;

  // Whole data array size - oversize so we verify areas not overwritten
  unsigned total_size = maxColumns * maxRows;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern, dummy data to check its overwritten when
  // it should be, and not when its not. Also need to limit data range otherwise
  // it will fail a check on small data types (HALF). 2048 is reperesenting
  // lagest integer number that can be represented bit-exact by HALFs
  // (significant)
  for (unsigned i = 0; i < total_size; i++)
    inTest[i] = (i + 1) % 2048;

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  // Test In and out tensor
  Tensor in = graph.addVariable(dataType, {total_size}, "Input");
  Tensor out = graph.addVariable(dataType, {total_size}, "Output");
  Tensor offsets =
      graph.addVariable(UNSIGNED_INT, {offsetsTest.size()}, "Offsets");
  graph.setTileMapping(in, 0);
  graph.setTileMapping(out, 0);
  graph.setTileMapping(offsets, 0);

  // allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto input = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                           downloadProg, tmap);
  auto output = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  auto offs = allocateHostMemoryForTensor(offsets, "offsets", graph, uploadProg,
                                          downloadProg, tmap);

  // Make multiple programs to test dynamic slice, each selecting
  // different slices, for different output sizes and offsets
  std::vector<Program> programs;

  for (unsigned tests = 0; tests < test_count; tests++) {
    auto rows = TestList[tests].rows;
    auto columns = TestList[tests].columns;
    auto baseOffset = TestList[tests].baseOffset;
    auto numBaseElements = TestList[tests].numBaseElements;
    auto regionSize = TestList[tests].regionSize;
    auto update = TestList[tests].update;

    Sequence sequence;

    ComputeSet testComputeSet = graph.addComputeSet("computeMultiSlice");

    auto vertexClass = templateVertex("popops::MultiSlice", dataType);
    auto base = in.slice(0, rows * columns);
    auto sub = out.slice(0, regionSize * offsets.numElements());
    if (update) {
      vertexClass = templateVertex("popops::MultiUpdate", dataType);
      base = out.slice(0, rows * columns);
      sub = in.slice(0, regionSize * offsets.numElements());
    }

    auto dsVertex =
        graph.addVertex(testComputeSet, vertexClass,
                        {{"offsets", offsets}, {"baseT", base}, {"subT", sub}});
    graph.setInitialValue(dsVertex["baseOffset"], baseOffset);
    graph.setInitialValue(dsVertex["numBaseElements"], numBaseElements);
    graph.setInitialValue(dsVertex["regionSize"], regionSize);
    graph.setTileMapping(dsVertex, 0);

    popops::zero(graph, out, sequence, "Zero output");
    sequence.add(Execute(testComputeSet));
    programs.push_back(sequence);
  }

  const auto uploadProgIndex = programs.size();
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size();
  programs.push_back(std::move(downloadProg));

  // Run each program and compare host and IPU result
  Engine engine(graph, programs);
  attachStreams(engine, tmap);

  // Put test inputs into an array of the correct type ready to use
  std::vector<double> outHost(total_size);

  for (unsigned tests = 0; tests < test_count; tests++) {
    auto baseOffset = TestList[tests].baseOffset;
    auto numBaseElements = TestList[tests].numBaseElements;
    auto regionSize = TestList[tests].regionSize;
    auto update = TestList[tests].update;

    copy(target, inTest.data(), inTest.size(), dataType, input.get());
    copy(target, offsetsTest.data(), offsetsTest.size(), UNSIGNED_INT,
         offs.get());

    device.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(tests);
      engine.run(downloadProgIndex);
    });

    copy(target, dataType, output.get(), outHost.data(), outHost.size());

    // Host generated result, start with 0s
    for (unsigned i = 0; i < total_size; i++)
      outTest[i] = 0;

    if (update) {
      MultiUpdateHost(offsetsTest, outTest, inTest, baseOffset, numBaseElements,
                      regionSize);
    } else {
      MultiSliceHost(offsetsTest, inTest, outTest, baseOffset, numBaseElements,
                     regionSize);
    }

    // Check the result, in the outTest array
    // Always check the whole output memory to catch any overwrites
    bool check = checkIsClose("Test_" + std::to_string(tests), outHost.data(),
                              {outHost.size()}, outTest.data(), outTest.size(),
                              0.0, 0.0);
    BOOST_CHECK(check);
  }
}

BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_half) {
  MultiSliceCodeletTest(HALF);
}
BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_float) {
  MultiSliceCodeletTest(FLOAT);
}
BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_int) { MultiSliceCodeletTest(INT); }
BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_unsigned) {
  MultiSliceCodeletTest(UNSIGNED_INT);
}
