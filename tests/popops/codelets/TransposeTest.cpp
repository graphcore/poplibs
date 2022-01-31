// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Test for the transpose2d vertex

#define BOOST_TEST_MODULE TransposeTest
#include "../../../lib/popops/RearrangeUtil.hpp"
#include "poputil/VertexTemplates.hpp"
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popops/Rearrange.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops::rearrange;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplin;
using namespace poplibs_support;

// Define a number of tests to run:
struct TestParams {
  unsigned rows;
  unsigned cols;
  unsigned matrices;
  bool force2d;
  bool splitTranspose; // only supported with matrices = 1 and force2d = false
};

std::vector<TestParams> SmallTestList = {
    {1, 10, 1, false, false},  {7, 1, 2, false, false},
    {8, 4, 1, false, false},   {24, 4, 2, false, false},
    {4, 4, 3, false, false},   {4, 4, 1, false, false},
    {5, 7, 2, false, false},   {16, 16, 3, true, false},
    {16, 16, 3, false, false}, {12, 16, 2, true, false},
    {12, 16, 2, false, false}, {8, 8, 1, false, false},
    {8, 9, 1, false, false},   {9, 4, 1, false, false},
    {4, 4, 1, true, false},    {8, 4, 1, true, false},
    {16, 4, 2, true, false},   {16, 4, 5, false, false},
    {16, 4, 6, false, false},  {16, 4, 15, false, false},
    {16, 4, 18, false, false}, {16, 4, 31, false, false},
};

std::vector<TestParams> T19548TestList = {
    {512, 4, 1, true, false},
};

std::vector<TestParams> T33035TestList = {{2052, 16, 1, false, false}};

std::vector<TestParams> splitTranspose1DTest = {
    {4, 16, 1, false, true}, {4, 52, 1, false, true},  {16, 4, 1, false, true},
    {52, 4, 1, false, true}, {52, 12, 1, false, true}, {52, 12, 2, false, true},
    {52, 12, 3, false, true}};

std::vector<TestParams> quarterTypeTestList = {
    {8, 8, 1, false, false},
    {16, 24, 13, false, false},
    {8, 32, 1, false, false},
    {32, 4, 3, false, false},
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
void TransposeTest(const Type &dataType, bool useMultiVertex,
                   const std::vector<TestParams> &testList) {

  // determine the sizes of arrays required
  auto test_count = testList.size();

  auto max_rows =
      std::max_element(testList.begin(), testList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.rows < b.rows);
                       })
          ->rows;
  auto max_cols =
      std::max_element(testList.begin(), testList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.cols < b.cols);
                       })
          ->cols;
  auto max_matrices =
      std::max_element(testList.begin(), testList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.matrices < b.matrices);
                       })
          ->matrices;

  // Whole data array size
  auto test_size = max_rows * max_cols * max_matrices;
  auto total_size = test_count * test_size;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  bool signedType = (dataType == HALF || dataType == FLOAT || dataType == INT ||
                     dataType == SHORT);

  // Initialise input pattern.
  // We don't want numbers that are outside the 'half'
  // precision (for integers):  -2048 <= HALF <= +2048
  // Or similarly for `quarter` where the minimum range is 16.  As we're
  // treating quarters as unsigned chars for test then use only positive values
  const int range = dataType == QUARTER ? 16 : 2048;
  std::generate_n(inTest.data(), inTest.size(),
                  [i = 0, signedType, range]() mutable {
                    return (int(i++) % (2 * range)) - (signedType ? range : 0);
                  });

  auto device = createTestDevice(TEST_TARGET, 1, test_count);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  // Input data
  Tensor in = graph.addVariable(
      dataType, {test_count, max_matrices, max_rows * max_cols}, "Input Data");

  // Result data
  Tensor out = graph.addVariable(
      dataType, {test_count, max_matrices, max_rows * max_cols}, "Output");

  // allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto input = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                           downloadProg, tmap);

  auto output = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  Sequence prog;
  ComputeSet cs = graph.addComputeSet("testTranpose");

  for (std::size_t test = 0; test < test_count; test++) {
    // put each test on a different tile
    graph.setTileMapping(in[test], test);
    graph.setTileMapping(out[test], test);

    auto matrices = testList[test].matrices;
    auto rows = testList[test].rows;
    auto cols = testList[test].cols;
    auto splitTranspose = testList[test].splitTranspose;

    // Zero output
    const auto zero =
        graph.addConstant(out.elementType(), out[test].shape(), 0);
    graph.setTileMapping(zero, test);
    prog.add(Copy(zero, out[test]));

    const auto fastVariant =
        canUseFastTranspose(target, dataType, rows, cols, matrices) &&
        !testList[test].force2d;

    std::string vertexName = "popops::Transpose2D";
    if (fastVariant) {
      vertexName = "popops::Transpose1DSingleWorker";
      if (useMultiVertex) {
        if (splitTranspose && matrices <= 3) {
          vertexName = "popops::SplitTranspose1D";
        } else {
          vertexName = "popops::Transpose1D";
        }
      }
    }

    const auto vertexClass = templateVertex(vertexName, dataType);

    auto transVertex = graph.addVertex(cs, vertexClass);
    graph.setTileMapping(transVertex, test);

    // Different slices of the same input data to test looping decisions
    auto sliceIn = in[test].slice({0, 0}, {matrices, rows * cols});
    auto sliceOut = out[test].slice({0, 0}, {matrices, rows * cols});

    if (fastVariant) {
      const unsigned subTransposeSize = dataType == QUARTER ? 8 : 4;
      graph.connect(transVertex["src"], sliceIn.flatten());
      graph.connect(transVertex["dst"], sliceOut.flatten());
      graph.setInitialValue(transVertex["numSrcColumnsD4Or8"],
                            cols / subTransposeSize);
      graph.setInitialValue(transVertex["numSrcRowsD4Or8"],
                            rows / subTransposeSize);
      if (!useMultiVertex) {
        graph.setInitialValue(transVertex["numTranspositionsM1"], matrices - 1);
      } else {
        unsigned numWorkerContexts = target.getNumWorkerContexts();

        if (splitTranspose) {
          auto workList = popops::internal::createSplitTranspose1DWorkList(
              rows, cols, matrices, numWorkerContexts, subTransposeSize);
          auto t = graph.addConstant(UNSIGNED_SHORT, {workList.size()},
                                     workList.data());
          graph.setTileMapping(t, test);
          graph.connect(transVertex["workList"], t);
        } else {

          // We will run one supervisor vertex, starting the 6 workers.
          // The first 'workerCount' workers (1<=workerCount<=6) will
          // transpose 'numTranspositions' matrices and (6-workerCount)
          // workers transposing (numTranspositions-1) matrices.
          // Note that (6-workerCount) and/or (numTranspositions-1) might
          // be zero.
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
      }
    } else {
      graph.connect(transVertex["src"], sliceIn);
      graph.connect(transVertex["dst"], sliceOut);
      graph.setInitialValue(transVertex["numSrcColumns"], cols);
      graph.setInitialValue(transVertex["numSrcRows"], rows);
    }
  }

  prog.add(Execute(cs));

  std::vector<Program> programs;
  const auto testProgIndex = programs.size();
  programs.push_back(prog);
  const auto uploadProgIndex = programs.size();
  programs.push_back(uploadProg);
  const auto downloadProgIndex = programs.size();
  programs.push_back(downloadProg);

  // Run each program and compare host and IPU result
  Engine engine(graph, programs);
  attachStreams(engine, tmap);

  // Put test inputs into an array of the correct type ready to use
  std::vector<double> outHost(total_size);

  copy(target, inTest.data(), inTest.size(), dataType, input.get());

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(uploadProgIndex);
    engine.run(testProgIndex);
    engine.run(downloadProgIndex);
  });

  copy(target, dataType, output.get(), outHost.data(), outHost.size());

  // Host generated result, start with zeros
  std::fill_n(outTest.data(), outTest.size(), 0);

  for (std::size_t test = 0; test < test_count; test++) {
    auto matrices = testList[test].matrices;
    auto rows = testList[test].rows;
    auto cols = testList[test].cols;

    const int testIndex = test * test_size;

    // Then transpose the same portion of the input as the code under test
    for (unsigned k = 0; k < matrices; k++) {
      int inIndex = k * max_rows * max_cols;
      for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
          const int outIndex = i + (j * rows) + (k * max_rows * max_cols);
          outTest[testIndex + outIndex] = inTest[testIndex + inIndex++];
        }
      }
    }
  }

  // Check the result, in the outTest array
  // Always check the whole output memory to catch any overwrites
  bool check = checkIsClose("TestTranspose", outHost.data(), {outHost.size()},
                            outTest.data(), outTest.size(), 0.0, 0.0);
  BOOST_CHECK(check);
}

BOOST_AUTO_TEST_SUITE(Transpose2d)

BOOST_AUTO_TEST_CASE(TransposeTest_half_true) {
  TransposeTest(HALF, true, SmallTestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_unsigned_short_true) {
  TransposeTest(UNSIGNED_SHORT, true, SmallTestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_short_true) {
  TransposeTest(SHORT, true, SmallTestList);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TransposeFast_16bit)

BOOST_AUTO_TEST_CASE(TransposeTest_half_false) {
  TransposeTest(HALF, false, SmallTestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_unsigned_short_false) {
  TransposeTest(UNSIGNED_SHORT, false, SmallTestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_short_false) {
  TransposeTest(SHORT, false, SmallTestList);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TransposeFast_Float)

BOOST_AUTO_TEST_CASE(TransposeTest_float_false) {
  TransposeTest(FLOAT, false, SmallTestList);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TransposeFast_Integral)

BOOST_AUTO_TEST_CASE(TransposeTest_unsigned_int_false) {
  TransposeTest(UNSIGNED_INT, false, SmallTestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_int_false) {
  TransposeTest(INT, false, SmallTestList);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(T19548)

BOOST_AUTO_TEST_CASE(TransposeTest_float_false_T19548) {
  TransposeTest(FLOAT, false, T19548TestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_unsigned_int_false_T19548) {
  TransposeTest(UNSIGNED_INT, false, T19548TestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_int_false_T19548) {
  TransposeTest(INT, false, T19548TestList);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(T33035)
BOOST_AUTO_TEST_CASE(TransposeTest_half_false_T33035) {
  TransposeTest(HALF, false, T33035TestList);
}
BOOST_AUTO_TEST_CASE(TransposeTest_half_true_T33035) {
  TransposeTest(HALF, true, T33035TestList);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(SplitTranspose1D)

BOOST_AUTO_TEST_CASE(TransposeTest_half_true_SplitTranspose) {
  TransposeTest(HALF, true, splitTranspose1DTest);
}

BOOST_AUTO_TEST_CASE(TransposeTest_quarter_true_SplitTranspose,
                     *boost::unit_test::precondition(enableIfIpu21Sim())) {
  TransposeTest(QUARTER, true, splitTranspose1DTest);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Transpose_quarter)

BOOST_AUTO_TEST_CASE(TransposeTest_quarter_true,
                     *boost::unit_test::precondition(enableIfIpu21Sim())) {
  TransposeTest(QUARTER, true, quarterTypeTestList);
}

BOOST_AUTO_TEST_CASE(TransposeTest_quarter_false,
                     *boost::unit_test::precondition(enableIfIpu21Sim())) {
  TransposeTest(QUARTER, false, SmallTestList);
}

BOOST_AUTO_TEST_SUITE_END()
