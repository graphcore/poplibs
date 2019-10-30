// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the transpose2d vertex
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poplibs_test/Util.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#define BOOST_TEST_MODULE Transpose2dTest
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplin;

// Define a number of tests to run:
struct TestParams {
  unsigned rows;
  unsigned cols;
  unsigned matrices;
  bool force2d;
};

std::vector<TestParams> TestList = {
    {1, 10, 1, false},  {7, 1, 2, false},   {8, 4, 1, false},
    {24, 4, 2, false},  {4, 4, 3, false},   {4, 4, 1, false},
    {5, 7, 2, false},   {16, 16, 3, true},  {16, 16, 3, false},
    {12, 16, 2, true},  {12, 16, 2, false}, {8, 8, 1, false},
    {8, 9, 1, false},   {9, 4, 1, false},   {4, 4, 1, true},
    {8, 4, 1, true},    {16, 4, 2, true},   {16, 4, 5, false},
    {16, 4, 6, false},  {16, 4, 15, false}, {16, 4, 18, false},
    {16, 4, 31, false},
};

//*************************************************
// Main Test function for Transpose 2d
//
// Overview:
// define max_matrices of size max_rows,MAX_COLUMNS
// Run a series of tests that transpose a varying number
// of matrices, but also select various small subsections/slices
// of data to transpose.
// The results are put into a memory area large enough to
// hold max_matrices of max_rowsxMAX_COLUMNS but often much of the data
// is expected to be zero.  This is checked as well as the "wanted" data.
//*************************************************
void TransposeTest(const Type &dataType, bool useSupervisorVertex) {

  // determine the sizes of arrays required
  auto test_count = TestList.size();

  auto max_rows = std::max_element(TestList.begin(), TestList.end(),
                                   [](TestParams &a, TestParams &b) {
                                     return (a.rows < b.rows);
                                   })
                      ->rows;
  auto max_cols = std::max_element(TestList.begin(), TestList.end(),
                                   [](TestParams &a, TestParams &b) {
                                     return (a.cols < b.cols);
                                   })
                      ->cols;
  auto max_matrices = std::max_element(TestList.begin(), TestList.end(),
                                       [](TestParams &a, TestParams &b) {
                                         return (a.matrices < b.matrices);
                                       })
                          ->matrices;
  // Whole data array size
  auto total_size = max_rows * max_cols * max_matrices;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern.
  for (unsigned i = 0; i < total_size; i++) {
    // We don't want numbers that are outside the 'half' precision (for
    // integers):  -2048 <= HALF <= +2048
    inTest[i] = (int(i) % 4096) - 2048;
  }

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  // Input data
  Tensor in = graph.addVariable(dataType, {max_matrices, max_rows * max_cols},
                                "Input Data");
  graph.setTileMapping(in, 0);

  // Result data
  Tensor out = graph.addVariable(dataType, {max_matrices, max_rows * max_cols},
                                 "Output");
  graph.setTileMapping(out, 0);

  // allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto input = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                           downloadProg, tmap);

  auto output = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  // Make multiple programs to test Transpose 2D each using
  // different input slices
  std::vector<Program> programs(test_count);

  for (std::size_t tests = 0; tests < test_count; tests++) {
    auto matrices = TestList[tests].matrices;
    auto rows = TestList[tests].rows;
    auto cols = TestList[tests].cols;

    Sequence sequence;

    ComputeSet testComputeSet = graph.addComputeSet("computeTranspose2d");
    const auto fastVariant =
        useFastTranspose(target, dataType, rows, cols, matrices) &&
        !TestList[tests].force2d;

    std::string vertexName = "poplin::Transpose2d";
    if (fastVariant) {
      vertexName = useSupervisorVertex ? "poplin::TransposeSupervisor"
                                       : "poplin::Transpose";
    }

    const auto vertexClass = templateVertex(vertexName, dataType);

    auto transVertex = graph.addVertex(testComputeSet, vertexClass);
    graph.setTileMapping(transVertex, 0);

    // Different slices of the same input data to test looping decisions
    auto sliceIn = in.slice({0, 0}, {matrices, rows * cols});
    auto sliceOut = out.slice({0, 0}, {matrices, rows * cols});

    if (fastVariant) {
      graph.connect(transVertex["src"], sliceIn.flatten());
      graph.connect(transVertex["dst"], sliceOut.flatten());
      graph.setInitialValue(transVertex["numSrcColumnsD4"], cols / 4);
      graph.setInitialValue(transVertex["numSrcRowsD4"], rows / 4);
      if (!useSupervisorVertex) {
        graph.setInitialValue(transVertex["numTranspositionsM1"], matrices - 1);
      } else {
        // We will run one supervisor vertex, starting the 6 workers.
        // The first 'workerCount' workers (1<=workerCount<=6) will
        // transpose 'numTranspositions' matrices and (6-workerCount)
        // workers transposing (numTranspositions-1) matrices.
        // Note that (6-workerCount) and/or (numTranspositions-1) might
        // be zero.
        unsigned numWorkerContexts = target.getNumWorkerContexts();
        unsigned workerCount = numWorkerContexts, numTranspositions = 1;
        if (matrices <= numWorkerContexts) {
          workerCount = matrices;
        } else {
          numTranspositions = matrices / workerCount;
          unsigned rem = matrices % workerCount;
          if (rem > 0) {
            workerCount = rem;
            numTranspositions += 1;
          }
        }
        graph.setInitialValue(transVertex["numTranspositions"],
                              numTranspositions);
        graph.setInitialValue(transVertex["workerCount"], workerCount);
      }
    } else {
      graph.connect(transVertex["src"], sliceIn);
      graph.connect(transVertex["dst"], sliceOut);
      graph.setInitialValue(transVertex["numSrcColumns"], cols);
      graph.setInitialValue(transVertex["numSrcRows"], rows);
    }

    popops::zero(graph, out, sequence, "Zero output");
    sequence.add(Execute(testComputeSet));
    programs[tests] = sequence;
  }
  const auto uploadProgIndex = programs.size();
  programs.push_back(uploadProg);
  const auto downloadProgIndex = programs.size();
  programs.push_back(downloadProg);

  // Run each program and compare host and IPU result
  Engine engine(graph, programs);
  attachStreams(engine, tmap);

  // Put test inputs into an array of the correct type ready to use
  std::vector<double> outHost(total_size);

  for (std::size_t tests = 0; tests < test_count; tests++) {
    auto matrices = TestList[tests].matrices;
    auto rows = TestList[tests].rows;
    auto cols = TestList[tests].cols;

    copy(target, inTest.data(), inTest.size(), dataType, input.get());

    device.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(tests);
      engine.run(downloadProgIndex);
    });

    copy(target, dataType, output.get(), outHost.data(), outHost.size());
    // Host generated result, start with zeros
    for (unsigned i = 0; i < total_size; i++)
      outTest[i] = 0;
    // Then transpose the same portion of the input as the code under test
    for (unsigned k = 0; k < matrices; k++) {
      int index = k * max_rows * max_cols;
      for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
          outTest[i + (j * rows) + (k * max_rows * max_cols)] = inTest[index++];
        }
      }
    }
    // Check the result, in the outTest array
    // Always check the whole output memory to catch any overwrites
    bool check = checkIsClose("Test_" + std::to_string(tests), outHost.data(),
                              {outHost.size()}, outTest.data(), outTest.size(),
                              0.0, 0.0);
    BOOST_CHECK(check);
  }
}
BOOST_AUTO_TEST_CASE(TransposeTest_half_true) { TransposeTest(HALF, true); }
BOOST_AUTO_TEST_CASE(TransposeTest_float_false) { TransposeTest(FLOAT, false); }
BOOST_AUTO_TEST_CASE(TransposeTest_half_false) { TransposeTest(HALF, false); }
