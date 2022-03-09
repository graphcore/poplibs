// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Test for the Dynamic Slice adn Dynamic Slice update 2d vertices
//
#define BOOST_TEST_MODULE MultiSliceCodeletTest
#include "poputil/VertexTemplates.hpp"
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/Operation.hpp>
#include <popops/OperationDefUtil.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <gccs/Algorithm.hpp>

#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;
using namespace popops;

// Define a number of tests to run:
struct TestParams {
  unsigned rows;
  unsigned columns;
  unsigned baseOffset;
  unsigned numBaseElements;
  unsigned regionSize;
  bool update;
  bool indicesAreSorted;
  std::optional<Operation> op;
  bool floatScale;        // only applicable for scaled update vertices
  bool splitSingleRegion; // split region - use first index
};

// Scale used for MultiUpdateAdd. Use a fixed scale and always use float
// type.
const float scaleForMultiUpdateAdd = 0.5;

std::vector<TestParams> TestList = {
    {100, 8, 1, 80, 8, false, false, std::nullopt, false},
    {80, 7, 1, 80, 7, false, true, std::nullopt, false},
    {100, 8, 1, 80, 8, true, true, Operation::ADD, true},
    {100, 8, 1, 80, 8, true, false, Operation::ADD, true},
    // This should split sliced dimension per worker such that for float
    // only one element gets allocated per worker
    {120, 1, 0, 2, 1, true, false, std::nullopt, false},

    // This to check branch path for float region size 2
    {120, 1, 0, 8, 2, true, true, Operation::ADD, true},
    {120, 1, 0, 8, 2, true, false, Operation::ADD, false},

    // This to check for branch path for half region size 4
    {120, 1, 0, 16, 4, true, true, Operation::ADD, true},
    {120, 1, 0, 16, 4, true, false, Operation::ADD, false},

    {80, 7, 1, 80, 7, true, true, Operation::MAX, false},
    {80, 7, 1, 80, 7, true, false, Operation::MAX, false},

    // single region Multi-slice within offset range
    {80, 64, 1, 80, 64, false, false, std::nullopt, false, true},
    // single region Multi-slice outside offset range
    {80, 64, 4, 80, 64, false, false, std::nullopt, false, true},

    // single region Multi-update within offset range
    {80, 64, 1, 80, 64, true, false, std::nullopt, false, true},
    {80, 64, 4, 80, 64, true, false, std::nullopt, false, true},

};

// nust have the same size to stream to device
std::vector<unsigned> offsetsTestUnsorted = {2, 2, 6, 5, 79, 0, 60, 55, 40, 30};
std::vector<unsigned> offsetsTestSorted = {2, 2, 2, 5, 5, 30, 40, 40, 40, 50};

//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void MultiSliceHost(std::vector<unsigned> &offsets, std::vector<double> &baseT,
                    std::vector<double> &subT, unsigned baseOffset,
                    unsigned numBaseElements, unsigned short regionSize,
                    unsigned singleRegionSplit) {
  unsigned numOffsets =
      singleRegionSplit ? std::min(offsets.size(), 1UL) : offsets.size();

  for (unsigned o = 0; o != numOffsets; ++o) {
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
                     unsigned numBaseElements, unsigned short regionSize,
                     std::optional<Operation> op, bool singleRegionSplit) {

  unsigned numOffsets =
      singleRegionSplit ? std::min(offsets.size(), 1UL) : offsets.size();

  for (unsigned o = 0; o != numOffsets; ++o) {
    auto baseIdx = offsets[o];
    if (baseIdx < baseOffset || baseIdx >= baseOffset + numBaseElements) {
      // this slice is not a part of baseT so we can skip it.
      continue;
    }
    baseIdx -= baseOffset;

    for (unsigned e = 0; e != regionSize; ++e) {
      if (op == std::nullopt) {
        baseT[baseIdx * regionSize + e] = subT[o * regionSize + e];
      } else {
        if (*op == Operation::MAX) {
          baseT[baseIdx * regionSize + e] = std::max(
              subT[o * regionSize + e], baseT[baseIdx * regionSize + e]);

        } else if (*op == Operation::ADD) {
          baseT[baseIdx * regionSize + e] +=
              subT[o * regionSize + e] * scaleForMultiUpdateAdd;
        }
      }
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
  // it will fail a check on small data types (8bit types and HALF). 2048 is the
  // largest integer number that can be represented bit-exact by HALF's
  // (significant)
  unsigned limit = (dataType == QUARTER || dataType == CHAR ||
                    dataType == UNSIGNED_CHAR || dataType == SIGNED_CHAR)
                       ? 127
                       : 2048;
  for (unsigned i = 0; i < total_size; i++)
    inTest[i] = (dataType == BOOL) ? 1 : (i + 1) % limit;

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  assert(offsetsTestSorted.size() == offsetsTestUnsorted.size());

  // Test In and out tensor
  Tensor in = graph.addVariable(dataType, {total_size}, "Input");
  Tensor out = graph.addVariable(dataType, {total_size}, "Output");
  Tensor offsets =
      graph.addVariable(UNSIGNED_INT, {offsetsTestUnsorted.size()}, "Offsets");
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

  auto scaleTensorF = graph.addVariable(poplar::FLOAT, {});
  auto scaleTensorH = graph.addVariable(poplar::HALF, {});
  graph.setTileMapping(scaleTensorF, 0);
  graph.setTileMapping(scaleTensorH, 0);

  auto scaleF = allocateHostMemoryForTensor(scaleTensorF, "scaleF", graph,
                                            uploadProg, downloadProg, tmap);
  auto scaleH = allocateHostMemoryForTensor(scaleTensorH, "scaleH", graph,
                                            uploadProg, downloadProg, tmap);

  // Make multiple programs to test dynamic slice, each selecting
  // different slices, for different output sizes and offsets
  std::vector<Program> programs;

  // allow multi-update op only for HALF, FLOAT, UNSIGNED_INT and INT
  const auto allowMultiUpdateOp = dataType == HALF || dataType == FLOAT ||
                                  dataType == UNSIGNED_INT || dataType == INT;

  // allow only multi-update ADD on HALF and FLOAT types. i.e. a subset of
  // allowed multi-update op
  const auto allowMultiUpdateAdd = dataType == HALF || dataType == FLOAT;

  // The decision on whether a test does a single region split is decided based
  // on the test configuration.
  std::vector<bool> singleRegionSplitUsed(test_count);

  for (unsigned tests = 0; tests < test_count; tests++) {
    auto rows = TestList[tests].rows;
    auto columns = TestList[tests].columns;
    auto baseOffset = TestList[tests].baseOffset;
    auto numBaseElements = TestList[tests].numBaseElements;
    auto regionSize = TestList[tests].regionSize;
    auto update = TestList[tests].update;
    auto indicesAreSorted = TestList[tests].indicesAreSorted;
    auto splitSingleRegion = TestList[tests].splitSingleRegion;
    auto op = TestList[tests].op;
    auto scaleIsFloat = TestList[tests].floatScale || dataType == FLOAT;

    const auto isMultiUpdateAdd = op != std::nullopt && *op == Operation::ADD;
    if ((isMultiUpdateAdd && !allowMultiUpdateAdd) ||
        (op != std::nullopt && !allowMultiUpdateOp)) {
      continue;
    }
    Sequence sequence;

    ComputeSet testComputeSet = graph.addComputeSet("computeMultiSlice");

    auto vertexClass = templateVertex("popops::MultiSlice", dataType);
    auto base = in.slice(0, rows * columns);
    auto sub = out.slice(0, regionSize * offsets.numElements());
    if (update) {
      if (op == std::nullopt) {
        vertexClass = templateVertex("popops::MultiUpdate", dataType);
      } else {
        auto subWordWrites = ((regionSize * target.getTypeSize(dataType)) %
                              target.getAtomicStoreGranularity()) != 0;
        if (*op == Operation::MAX) {
          vertexClass = templateVertex("popops::MultiUpdateOp", dataType,
                                       subWordWrites, *op);
        } else if (*op == Operation::ADD) {
          vertexClass =
              templateVertex("popops::ScaledMultiUpdateOp", dataType,
                             scaleIsFloat ? FLOAT : HALF, subWordWrites, *op);
        }
      }
      base = out.slice(0, rows * columns);
      sub = in.slice(0, regionSize * offsets.numElements());
    }

    auto dsVertex =
        graph.addVertex(testComputeSet, vertexClass,
                        {{"offsets", offsets}, {"baseT", base}, {"subT", sub}});
    graph.setInitialValue(dsVertex["baseOffset"], baseOffset);
    graph.setInitialValue(dsVertex["numBaseElements"], numBaseElements);
    graph.setInitialValue(dsVertex["regionSize"], regionSize);
    graph.setInitialValue(dsVertex["indicesAreSorted"], indicesAreSorted);

    // the dimension that is split depends on whether this is an update or slice
    unsigned grainSize =
        std::max(static_cast<unsigned>(target.getAtomicStoreGranularity() /
                                       target.getTypeSize(dataType)),
                 1U);
    unsigned elemsToSplit = update ? numBaseElements : offsets.numElements();
    // We can split region only if the region size is a sub-multiple of the
    // atomic size granularity
    const bool vertexHasSplitRegionField = !update || op == std::nullopt;

    if (splitSingleRegion && vertexHasSplitRegionField &&
        (regionSize % target.getAtomicStoreGranularity() == 0)) {
      elemsToSplit = regionSize;
    } else {
      splitSingleRegion = false;
    }
    singleRegionSplitUsed.at(tests) = splitSingleRegion;

    unsigned numGrains = gccs::ceildiv(elemsToSplit, grainSize);
    unsigned grainsPerWorker =
        gccs::ceildiv(numGrains, target.getNumWorkerContexts());

    // This is not exactly how elements are split in graph construction but
    // we just want to get some work division that doesn't cause write hazards.
    auto maxElems = std::min(elemsToSplit, grainsPerWorker * grainSize);

    if (vertexHasSplitRegionField) {
      graph.setInitialValue(dsVertex["splitSingleRegion"], splitSingleRegion);
    }
    graph.setInitialValue(dsVertex["maxElementsPerWorker"], maxElems);
    if (update && isMultiUpdateAdd) {
      graph.connect(dsVertex["scale"],
                    scaleIsFloat ? scaleTensorF : scaleTensorH);
    }
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

  for (unsigned tests = 0, progNum = 0; tests < test_count; tests++) {
    auto baseOffset = TestList[tests].baseOffset;
    auto numBaseElements = TestList[tests].numBaseElements;
    auto regionSize = TestList[tests].regionSize;
    auto update = TestList[tests].update;
    auto indicesAreSorted = TestList[tests].indicesAreSorted;
    auto op = TestList[tests].op;
    auto scaleIsFloat = TestList[tests].floatScale || dataType == FLOAT;
    auto singleRegionSplit = singleRegionSplitUsed.at(tests);

    const auto isMultiUpdateAdd = op != std::nullopt && *op == Operation::ADD;
    if ((isMultiUpdateAdd && !allowMultiUpdateAdd) ||
        (op != std::nullopt && !allowMultiUpdateOp)) {
      continue;
    }
    auto &offsetsTest =
        indicesAreSorted ? offsetsTestSorted : offsetsTestUnsorted;
    copy(target, inTest.data(), inTest.size(), dataType, input.get());
    copy(target, offsetsTest.data(), offsetsTest.size(), UNSIGNED_INT,
         offs.get());

    if (allowMultiUpdateAdd && isMultiUpdateAdd) {
      if (scaleIsFloat) {
        copy(target, &scaleForMultiUpdateAdd, 1, FLOAT, scaleF.get());
      } else {
        copy(target, &scaleForMultiUpdateAdd, 1, HALF, scaleH.get());
      }
    }

    device.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(progNum);
      engine.run(downloadProgIndex);
    });

    copy(target, dataType, output.get(), outHost.data(), outHost.size());

    // Host generated result, start with 0s
    for (unsigned i = 0; i < total_size; i++)
      outTest[i] = 0;

    if (update) {
      MultiUpdateHost(offsetsTest, outTest, inTest, baseOffset, numBaseElements,
                      regionSize, op, singleRegionSplit);
    } else {
      MultiSliceHost(offsetsTest, inTest, outTest, baseOffset, numBaseElements,
                     regionSize, singleRegionSplit);
    }

    // Check the result, in the outTest array
    // Always check the whole output memory to catch any overwrites
    bool check = checkIsClose("Test_" + std::to_string(tests), outHost.data(),
                              {outHost.size()}, outTest.data(), outTest.size(),
                              0.0, 0.0);
    BOOST_CHECK(check);
    ++progNum;
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

BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_uchar) {
  MultiSliceCodeletTest(UNSIGNED_CHAR);
}

BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_schar) {
  MultiSliceCodeletTest(SIGNED_CHAR);
}
BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_char) {
  MultiSliceCodeletTest(CHAR);
}
BOOST_AUTO_TEST_CASE(MultiSliceCodeletTest_bool) {
  MultiSliceCodeletTest(BOOL);
}
