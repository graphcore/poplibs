// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Test for the Dynamic Slice adn Dynamic Slice update 2d vertices
//
#define BOOST_TEST_MODULE DynamicSliceCodeletTest
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplibs_support;

// Define a number of tests to run:
struct TestParams {
  unsigned offset;
  unsigned numBaseElements;
  unsigned numSubElements;
  unsigned regionSize;

  // Note that 'dstOffset' is NOT a field of the vertex, but is used to offset
  // the destination (subT) inside the bigger tensor that is allocated for it,
  // for the purpose of testing different alignment.
  unsigned dstOffset;
  bool update;
};

// For float, half, int and unsigned char we run these common tests.
std::vector<TestParams> tests = {
    // Test with small regions
    {0, 4, 3, 1, 0, false},
    {0, 4, 3, 1, 1, false},
    {0, 4, 3, 1, 2, false},
    {0, 4, 3, 2, 1, false},
    {0, 4, 3, 3, 1, false},
    {0, 4, 3, 4, 1, false},
    {0, 4, 3, 5, 1, false},
    {0, 4, 3, 6, 1, false},
    {0, 4, 3, 7, 1, false},
    {0, 4, 3, 8, 1, false},
    {0, 4, 3, 9, 1, false},
    {0, 4, 3, 10, 1, false},
    {0, 4, 3, 11, 1, false},
    {0, 4, 3, 27, 1, false},
    {0, 4, 3, 31, 1, false},
    {0, 4, 3, 32, 1, false},
    {0, 4, 3, 33, 1, false},
    // half vertex has different handling for <= 3 atoms
    {0, 3, 1, 4, 0, false}, // 4 atoms at all offsets
    {0, 3, 1, 4, 1, false},
    {0, 3, 1, 4, 2, false},
    {0, 3, 1, 4, 3, false},
    {0, 3, 1, 27, 0, false}, // 4 atoms/worker plus 3 at the end
    {0, 3, 1, 27, 1, false},
    {0, 3, 1, 27, 2, false},
    {0, 3, 1, 27, 3, false},
    {0, 3, 1, 27 + 24 * 2 + 1, 0, false + 1}, // 16 atoms/worker + 3 at the end
    {0, 3, 1, 27 + 24 * 2 + 1, 1, false},
    {0, 3, 1, 27 + 24 * 2 + 1, 2, false},
    {0, 3, 1, 27 + 24 * 2 + 1, 3, false},
    {0, 1, 2, 1, 1, false},
    {0, 1, 2, 2, 1, false},
    {0, 1, 2, 3, 1, false},
    {0, 1, 2, 4, 1, false},
    {0, 1, 2, 5, 1, false},
    {0, 1, 2, 6, 1, false},
    {1, 3, 2, 7, 0, false},
    {0, 4, 4, 8, 1, false},
    {2, 4, 5, 9, 0, false},
    {0, 4, 4, 10, 1, false},
    {2, 4, 5, 11, 0, false},
    {0, 2, 2, 12, 1, false},
    {3, 5, 5, 13, 0, false},
    {3, 5, 5, 31, 0, false},

    {0, 1, 1, 6, 1, true},
    {1, 2, 2, 7, 0, true},
    {0, 4, 4, 8, 1, true},
    {2, 4, 4, 9, 0, true},
    {0, 2, 2, 12, 1, true},
    {3, 5, 5, 13, 0, true},
};
// For 8 bit types we run a few more tests to tests some more of the
// combinations of offsets and sizes. We break them down among different 8 bit
// types, despite the code being common for all of them, but we still want to
// test that the C++ declaration and assembly label entry points are all there.
std::vector<TestParams> testsSchar = {
    {1, 1, 2, 1, 1, false}, {1, 1, 2, 2, 1, false}, {1, 1, 2, 3, 1, false},
    {1, 1, 2, 4, 1, false}, {1, 1, 2, 5, 1, false}, {1, 1, 2, 6, 1, false},
    {1, 1, 2, 7, 1, false}, {1, 1, 2, 8, 1, false}, {1, 1, 2, 9, 1, false},
};
std::vector<TestParams> testsChar = {
    {2, 1, 2, 1, 1, false}, {2, 1, 2, 2, 1, false}, {2, 1, 2, 3, 1, false},
    {2, 1, 2, 4, 1, false}, {2, 1, 2, 5, 1, false}, {2, 1, 2, 6, 1, false},
    {2, 1, 2, 7, 1, false}, {2, 1, 2, 8, 1, false}, {2, 1, 2, 9, 1, false},
};
std::vector<TestParams> testsBool = {
    {3, 1, 2, 1, 1, false}, {3, 1, 2, 2, 1, false}, {3, 1, 2, 3, 1, false},
    {3, 1, 2, 4, 1, false}, {3, 1, 2, 5, 1, false}, {3, 1, 2, 6, 1, false},
    {3, 1, 2, 7, 1, false}, {3, 1, 2, 8, 1, false}, {3, 1, 2, 9, 1, false},
};

//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void DynamicSliceSupervisorHost(unsigned offset, std::vector<double> &baseT,
                                std::vector<double> &subT,
                                unsigned numBaseElements,
                                unsigned short numSubElements,
                                unsigned short regionSize,
                                unsigned short dstOffset) {
  unsigned baseSlice = offset;

  if (baseSlice >= numBaseElements)
    baseSlice = 0;
  for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
    for (unsigned e = 0; e != regionSize; e++) {
      subT[subSlice * regionSize + e + dstOffset] =
          baseT[baseSlice * regionSize + e];
    }
    baseSlice++;
    if (baseSlice >= numBaseElements)
      baseSlice = 0;
  }
}
//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void DynamicUpdateSliceSupervisorHost(
    unsigned offset, std::vector<double> &baseT, std::vector<double> &subT,
    unsigned numBaseElements, unsigned short numSubElements,
    unsigned short regionSize, unsigned short dstOffset) {
  unsigned baseSlice = offset;

  if (baseSlice >= numBaseElements)
    baseSlice = 0;
  for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
    for (unsigned e = 0; e != regionSize; e++) {
      baseT[baseSlice * regionSize + e + dstOffset] =
          subT[subSlice * regionSize + e];
    }
    baseSlice++;
    if (baseSlice >= numBaseElements)
      baseSlice = 0;
  }
}

//*************************************************
// Main Test function for DynamicSliceSupervisor, DynamicUpdateSliceSupervisor
//
// Overview:
//
// Output memory space is initialised as all zero.
// Input memory space is initialised with a simple test pattern
// Run a series of tests that copy a varying number of items.
// The results are put into a memory area large enough to
// hold the largest test result, so often the other items are
// expected to be zero.  This is checked as well as the "wanted" data.
//*************************************************
void DynamicSliceCodeletTest(const Type &dataType,
                             const std::vector<TestParams> &TestList = tests) {

  // determine the sizes of arrays required
  auto test_count = TestList.size();

  const auto maxRegionSize =
      std::max_element(TestList.begin(), TestList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.regionSize < b.regionSize);
                       })
          ->regionSize;

  const auto maxDstOffset =
      std::max_element(TestList.begin(), TestList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.dstOffset < b.dstOffset);
                       })
          ->dstOffset;

  // Check max sizes of regions so that the test method generates legal copies
  const auto maxElements =
      std::max_element(TestList.begin(), TestList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (std::max(a.numSubElements, a.numBaseElements) <
                                 std::max(b.numSubElements, b.numBaseElements));
                       });

  const auto maxRows =
      std::max(maxElements->numBaseElements, maxElements->numSubElements);
  // Whole data array size - oversize foe the smaller tests
  // so we verify areas not overwritten
  auto total_size = maxRegionSize * maxRows + maxDstOffset;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern, dummy data to check its overwritten when
  // it should be, and not when its not
  for (unsigned i = 0; i < total_size; i++)
    inTest[i] = (dataType == BOOL) ? 1 : i + 1;

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  // Test In and out tensor
  Tensor in = graph.addVariable(dataType, {total_size}, "Input");
  Tensor out = graph.addVariable(dataType, {total_size}, "Output");
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
    auto regionSize = TestList[tests].regionSize;
    auto dstOffset = TestList[tests].dstOffset;
    auto update = TestList[tests].update;

    Sequence sequence;

    ComputeSet testComputeSet = graph.addComputeSet("computeDynamicSlice");

    auto vertexClass = templateVertex("popops::DynamicSlice1D", dataType);
    auto base = in.slice(0, numBaseElements * regionSize);
    auto sub = out.slice(dstOffset, numSubElements * regionSize + dstOffset);
    if (update) {
      vertexClass = templateVertex("popops::DynamicUpdateSlice1D", dataType);
      base = out.slice(dstOffset, numBaseElements * regionSize + dstOffset);
      sub = in.slice(0, numSubElements * regionSize);
    }

    auto dsVertex =
        graph.addVertex(testComputeSet, vertexClass,
                        {{"offset", offset}, {"baseT", base}, {"subT", sub}});
    graph.setInitialValue(dsVertex["numBaseElements"], numBaseElements);
    graph.setInitialValue(dsVertex["numSubElements"], numSubElements);
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

  device.bind([&](const Device &d) {
    engine.load(d);
    for (unsigned tests = 0; tests < test_count; tests++) {
      BOOST_TEST_MESSAGE("Starting subtest " << tests << "/ " << test_count);
      auto offset = TestList[tests].offset;
      auto numBaseElements = TestList[tests].numBaseElements;
      auto numSubElements = TestList[tests].numSubElements;
      auto regionSize = TestList[tests].regionSize;
      auto dstOffset = TestList[tests].dstOffset;
      auto update = TestList[tests].update;

      copy(target, inTest.data(), inTest.size(), dataType, input.get());

      engine.run(uploadProgIndex);
      engine.run(tests);
      engine.run(downloadProgIndex);

      copy(target, dataType, output.get(), outHost.data(), outHost.size());

      // Host generated result, start with 0s
      for (unsigned i = 0; i < total_size; i++)
        outTest[i] = 0;

      // Run the host version of the codelet to compare against - either
      // update or non update version
      if (update) {
        DynamicUpdateSliceSupervisorHost(offset, outTest, inTest,
                                         numBaseElements, numSubElements,
                                         regionSize, dstOffset);
      } else {
        DynamicSliceSupervisorHost(offset, inTest, outTest, numBaseElements,
                                   numSubElements, regionSize, dstOffset);
      }

      // Check the result, in the outTest array
      // Always check the whole output memory to catch any overwrites
      bool check = checkIsClose("Test_" + std::to_string(tests), outHost.data(),
                                {outHost.size()}, outTest.data(),
                                outTest.size(), 0.0, 0.0);
      BOOST_CHECK(check);
    }
  });
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_float) {
  DynamicSliceCodeletTest(FLOAT);
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_half) {
  DynamicSliceCodeletTest(HALF);
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_int) {
  DynamicSliceCodeletTest(INT);
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_uchar) {
  DynamicSliceCodeletTest(UNSIGNED_CHAR);
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_schar) {
  DynamicSliceCodeletTest(SIGNED_CHAR, testsSchar);
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_char) {
  DynamicSliceCodeletTest(CHAR, testsChar);
}
BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_bool) {
  DynamicSliceCodeletTest(BOOL, testsBool);
}
