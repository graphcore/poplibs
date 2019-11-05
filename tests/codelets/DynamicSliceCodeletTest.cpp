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

#define BOOST_TEST_MODULE DynamicSliceCodeletTest
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;

// Define a number of tests to run:
struct TestParams {
  unsigned offset;
  unsigned numBaseElements;
  unsigned numSubElements;
  unsigned numRegions;
  unsigned rows;
  unsigned columns;
  unsigned dstColumn;
  bool update;
};

std::vector<TestParams> TestList = {
    {0, 1, 2, 2, 8, 1, 1, false}, {0, 2, 1, 1, 4, 2, 1, false},
    {1, 3, 2, 1, 4, 3, 2, false}, {1, 4, 2, 1, 4, 4, 2, false},
    {1, 1, 7, 1, 7, 7, 3, false}, {0, 2, 2, 1, 4, 8, 3, false},
    {0, 2, 2, 1, 4, 9, 3, false}, {0, 2, 2, 1, 4, 15, 0, false},

    {0, 1, 2, 2, 8, 1, 1, true},  {0, 2, 1, 1, 4, 2, 1, true},
    {1, 3, 2, 1, 4, 3, 2, true},  {1, 4, 2, 1, 4, 4, 2, true},
    {1, 1, 7, 1, 7, 7, 3, true},  {0, 2, 2, 1, 4, 8, 3, true},
    {0, 2, 2, 1, 4, 9, 3, true},  {0, 2, 2, 1, 4, 15, 0, true},
    // TODO: T12986 Next tests are for MK2 as currently they don't fit into
    // memory
    // Set numBaseElements to anything higher than 65535
    //{0, 66000, 1, 2, 132000, 1, 0, false},
    //{0, 66000, 1, 2, 132000, 1, 0, true},
};

//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void DynamicSlice2dHost(unsigned offset, std::vector<double *> &baseT,
                        std::vector<double *> &subT, unsigned numBaseElements,
                        unsigned short numSubElements,
                        unsigned short numRegions, unsigned short regionSize) {
  for (unsigned r = 0; r != numRegions; ++r) {
    unsigned baseSlice = offset;
    if (baseSlice >= numBaseElements)
      baseSlice -= numBaseElements;
    unsigned subIdx = r * numSubElements;

    for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
      auto baseIdx = r * numBaseElements + baseSlice;
      for (unsigned e = 0; e != regionSize; e++) {
        subT[subIdx][e] = baseT[baseIdx][e];
      }
      subIdx++;
      baseSlice++;
      if (baseSlice >= numBaseElements)
        baseSlice -= numBaseElements;
    }
  }
}
//*************************************************
// C test function, based on the original C version of the vertex
// Similar to above, but copy in the opposite direction
//*************************************************
void DynamicUpdateSlice2dHost(unsigned offset, std::vector<double *> &baseT,
                              std::vector<double *> &subT,
                              unsigned numBaseElements,
                              unsigned short numSubElements,
                              unsigned short numRegions,
                              unsigned short regionSize) {
  for (unsigned r = 0; r != numRegions; ++r) {
    unsigned baseSlice = offset;
    if (baseSlice >= numBaseElements)
      baseSlice -= numBaseElements;
    unsigned subIdx = r * numSubElements;

    for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
      auto baseIdx = r * numBaseElements + baseSlice;
      for (unsigned e = 0; e != regionSize; e++) {
        baseT[baseIdx][e] = subT[subIdx][e];
      }
      subIdx++;
      baseSlice++;
      if (baseSlice >= numBaseElements)
        baseSlice -= numBaseElements;
    }
  }
}
//*************************************************
// Main Test function for DynamicSlice2d, DynamicUpdateSlice2d
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
void DynamicSliceCodeletTest(const Type &dataType) {

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

  // Check max sizes of regions so that the test method generates legal copies
  const auto maxBaseRegions = std::max_element(
      TestList.begin(), TestList.end(), [](TestParams &a, TestParams &b) {
        return (a.numRegions * a.numBaseElements <
                b.numRegions * b.numBaseElements);
      });

  const auto maxSubRegions = std::max_element(
      TestList.begin(), TestList.end(), [](TestParams &a, TestParams &b) {
        return (a.numRegions * a.numSubElements <
                b.numRegions * b.numSubElements);
      });

  if (maxBaseRegions->numRegions * maxBaseRegions->numBaseElements > maxRows) {
    throw std::logic_error(
        "Number of base regions exceeds number of destination rows, test will"
        " generate invalid references.");
  }
  if (maxSubRegions->numRegions * maxSubRegions->numSubElements > maxRows) {
    throw std::logic_error(
        "Number of sub regions exceeds number of destination rows, test will"
        " generate invalid references.");
  }
  // Whole data array size - oversize so we verify areas not overwritten
  auto total_size = maxColumns * maxRows;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern, dummy data to check its overwritten when
  // it should be, and not when its not. Also need to limit data range otherwise
  // it will fail a check on small data types (HALF)
  for (unsigned i = 0; i < total_size; i++)
    inTest[i] = (i + 1) % 65519;

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  // Test In and out tensor
  Tensor in = graph.addVariable(dataType, {maxRows, maxColumns}, "Input");
  Tensor out = graph.addVariable(dataType, {maxRows, maxColumns}, "Output");
  graph.setTileMapping(in, 0);
  graph.setTileMapping(out, 0);

  // allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto input = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                           downloadProg, tmap);
  auto output = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  // Make multiple programs to test dynamic slice, each selecting
  // different slices, for different output sizes and offsets
  std::vector<Program> programs;

  for (unsigned tests = 0; tests < test_count; tests++) {
    auto offset = TestList[tests].offset;
    auto numBaseElements = TestList[tests].numBaseElements;
    auto numSubElements = TestList[tests].numSubElements;
    auto numRegions = TestList[tests].numRegions;
    auto rows = TestList[tests].rows;
    auto columns = TestList[tests].columns;
    auto dstColumn = TestList[tests].dstColumn;
    auto update = TestList[tests].update;

    Sequence sequence;

    ComputeSet testComputeSet = graph.addComputeSet("computeDynamicSlice");

    auto vertexClass = templateVertex("popops::DynamicSlice2d", dataType);
    auto base = in.slice({0, 0}, {rows, columns});
    auto sub = out.slice({0, dstColumn}, {rows, columns + dstColumn});
    if (update) {
      vertexClass = templateVertex("popops::DynamicUpdateSlice2d", dataType);
      base = out.slice({0, 0}, {rows, columns});
      sub = in.slice({0, dstColumn}, {rows, columns + dstColumn});
    }

    auto dsVertex =
        graph.addVertex(testComputeSet, vertexClass,
                        {{"offset", offset}, {"baseT", base}, {"subT", sub}});
    graph.setInitialValue(dsVertex["numBaseElements"], numBaseElements);
    graph.setInitialValue(dsVertex["numSubElements"], numSubElements);
    graph.setInitialValue(dsVertex["numRegions"], numRegions);
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
    auto offset = TestList[tests].offset;
    auto numBaseElements = TestList[tests].numBaseElements;
    auto numSubElements = TestList[tests].numSubElements;
    auto numRegions = TestList[tests].numRegions;
    auto columns = TestList[tests].columns;
    auto dstColumn = TestList[tests].dstColumn;
    auto update = TestList[tests].update;

    copy(target, inTest.data(), inTest.size(), dataType, input.get());

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

    // Build vectors of pointers to the regions to copy for the host version
    std::vector<double *> hostBaseT(numBaseElements * numRegions);
    std::vector<double *> hostSubT(numSubElements * numRegions);

    if (update) {
      for (unsigned i = 0; i < numBaseElements * numRegions; i++)
        hostBaseT[i] = &outTest[i * maxColumns];

      for (unsigned i = 0; i < numSubElements * numRegions; i++)
        hostSubT[i] = &inTest[i * maxColumns + dstColumn];
      DynamicUpdateSlice2dHost(offset, hostBaseT, hostSubT, numBaseElements,
                               numSubElements, numRegions, columns);
    } else {
      for (unsigned i = 0; i < numBaseElements * numRegions; i++)
        hostBaseT[i] = &inTest[i * maxColumns];

      for (unsigned i = 0; i < numSubElements * numRegions; i++)
        hostSubT[i] = &outTest[i * maxColumns + dstColumn];
      DynamicSlice2dHost(offset, hostBaseT, hostSubT, numBaseElements,
                         numSubElements, numRegions, columns);
    }
    // Check the result, in the outTest array
    // Always check the whole output memory to catch any overwrites
    bool check = checkIsClose("Test_" + std::to_string(tests), outHost.data(),
                              {outHost.size()}, outTest.data(), outTest.size(),
                              0.0, 0.0);
    BOOST_CHECK(check);
  }
}
BOOST_AUTO_TEST_CASE(DynamicSliceCodeletTest_float) {
  DynamicSliceCodeletTest(FLOAT);
}
BOOST_AUTO_TEST_CASE(DynamicSliceCodeletTest_half) {
  DynamicSliceCodeletTest(HALF);
}
BOOST_AUTO_TEST_CASE(DynamicSliceCodeletTest_int) {
  DynamicSliceCodeletTest(INT);
}
BOOST_AUTO_TEST_CASE(DynamicSliceCodeletTest_unsigned) {
  DynamicSliceCodeletTest(UNSIGNED_INT);
}
