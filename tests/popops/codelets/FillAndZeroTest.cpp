// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// Tests for the `popops::fill()`, `popops::zero()` functions and
// the fill and fill2d vertices.

#include <poplar/Engine.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>

#include <popops/Fill.hpp>
#include <popops/Zero.hpp>

#define BOOST_TEST_MODULE FillTest
#include <poplibs_support/TestDevice.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;

// Describes the shape of the tensor slice to fill or zero.
struct TestParams {
  unsigned rows;
  unsigned columns;
  unsigned offset;
};

using ProgramBuilderType =
    std::function<Sequence(const TestParams &, poplar::Graph &, Tensor &)>;

// Overview:
//
// Memory is initialised as values 0 to N.
// Run a series of tests that fill a varying number of items.
// The results are put into a memory area large enough to
// hold the largest test result, so often the other items are
// expected to be the original initialised value.
// This is checked as well as the "wanted" data.
template <typename FillValueType, size_t N>
void Test(const std::array<TestParams, N> &testList, const Type &dataType,
          FillValueType fillValue, ProgramBuilderType programBuilder) {
  // determine the sizes of arrays required
  const auto test_count = testList.size();

  // TODO: Make these constexpr when we have C++17.
  const auto max_rows =
      std::max_element(testList.begin(), testList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.rows < b.rows);
                       })
          ->rows;

  const auto max_columns =
      std::max_element(testList.begin(), testList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.columns < b.columns);
                       })
          ->columns;

  const auto max_offset =
      std::max_element(testList.begin(), testList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.offset < b.offset);
                       })
          ->offset;
  // Whole data array size
  const auto total_size = (max_columns + max_offset) * max_rows;

  // Program generated test data
  std::vector<double> expected(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern, dummy data to check its overwritten when
  // it should be, and not when its not
  for (unsigned i = 0; i < total_size; i++)
    inTest[i] = i;

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  // Test data tensor
  Tensor data =
      graph.addVariable(dataType, {max_rows, max_columns + max_offset}, "Data");
  graph.setTileMapping(data, 0);

  // allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto input = allocateHostMemoryForTensor(data, "in", graph, uploadProg,
                                           downloadProg, tmap);

  // Make multiple programs to test fill, each using
  // different slices, for different output sizes and offsets
  std::vector<Program> programs;
  programs.reserve(test_count + 2); // + 2 for upload/download

  for (unsigned tests = 0; tests < test_count; tests++) {
    auto rows = testList[tests].rows;
    auto columns = testList[tests].columns;
    auto offset = testList[tests].offset;

    // Exclude zero2d tests for int, unsigned
    if (dataType == HALF || dataType == FLOAT || rows < 2) {
      // Different slices to test looping decisions, with offset
      Tensor sliceOut;
      if (rows == 1) {
        sliceOut = data.reshape({total_size});
        sliceOut = sliceOut.slice(offset, columns + offset);
      } else {
        sliceOut = data.slice({0, offset}, {rows, columns + offset});
      }

      programs.push_back(programBuilder(testList[tests], graph, sliceOut));
    }
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
    auto rows = testList[tests].rows;
    auto columns = testList[tests].columns;
    auto offset = testList[tests].offset;

    // Exclude zero2d tests for int, unsigned
    if (dataType == HALF || dataType == FLOAT || rows < 2) {
      copy(target, inTest.data(), inTest.size(), dataType, input.get());

      device.bind([&](const Device &d) {
        engine.load(d);
        engine.run(uploadProgIndex);
        engine.run(tests);
        engine.run(downloadProgIndex);
      });

      copy(target, dataType, input.get(), outHost.data(), outHost.size());

      // Generate the expected result.
      std::copy(std::begin(inTest), std::end(inTest), std::begin(expected));
      for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < columns; j++) {
          expected[j + offset + (i * (max_columns + max_offset))] = fillValue;
        }
      }

      // Check the result, in the expected vector
      // Always check the whole output memory to catch any overwrites
      bool check = checkIsClose("Test_" + std::to_string(tests), outHost.data(),
                                {outHost.size()}, expected.data(),
                                expected.size(), 0.0, 0.0);
      BOOST_CHECK(check);
    }
  }
}

// Test the Fill and Fill2d vertices.
template <typename FillType>
void FillVerticesTest(const Type &dataType, FillType fillValue) {
  constexpr std::array<TestParams, 20> testList = {{
      {1, 1, 0}, {1, 2, 0}, {1, 3, 0}, {1, 4, 0}, {1, 5, 0},
      {1, 6, 0}, {1, 7, 0}, {1, 8, 0}, {1, 1, 1}, {1, 2, 1},
      {1, 3, 1}, {1, 4, 1}, {1, 5, 1}, {1, 6, 1}, {1, 7, 1},
      {1, 8, 1}, {2, 1, 1}, {2, 4, 1}, {2, 7, 1}, {2, 8, 1},
  }};

  Test(testList, dataType, fillValue,
       [dataType, fillValue](const TestParams &test, poplar::Graph &graph,
                             Tensor &sliceOut) {
         ComputeSet testComputeSet = graph.addComputeSet("computeFill");

         const auto vertexClass = poputil::templateVertex(
             test.rows > 1 ? "popops::Fill2d" : "popops::Fill", dataType);

         auto fillVertex = graph.addVertex(testComputeSet, vertexClass);
         graph.setTileMapping(fillVertex, 0);
         graph.setInitialValue(fillVertex["in"], fillValue);
         graph.connect(fillVertex["out"], sliceOut);

         Sequence sequence;
         sequence.add(Execute(testComputeSet));
         return sequence;
       });
}

BOOST_AUTO_TEST_CASE(FillVerticesTest_half) {
  FillVerticesTest<float>(HALF, 3.0);
}
BOOST_AUTO_TEST_CASE(FillVerticesTest_float) {
  FillVerticesTest<float>(FLOAT, 1.23456789);
}
BOOST_AUTO_TEST_CASE(FillVerticesTest_int) { FillVerticesTest<int>(INT, 4); }
BOOST_AUTO_TEST_CASE(FillIncorrectVertexTest) {
  constexpr float fillValue = 5.0;
  constexpr std::array<TestParams, 1> testList = {{{2, 1, 0}}};
  ProgramBuilderType programBuilder =
      [](const TestParams &test, poplar::Graph &graph, Tensor &sliceOut) {
        ComputeSet testComputeSet = graph.addComputeSet("computeFill");

        const auto vertexClass = poputil::templateVertex("popops::Fill", HALF);

        auto fillVertex = graph.addVertex(testComputeSet, vertexClass);
        graph.setTileMapping(fillVertex, 0);
        graph.setInitialValue(fillVertex["in"], fillValue);
        graph.connect(fillVertex["out"], sliceOut);

        Sequence sequence;
        sequence.add(Execute(testComputeSet));
        return sequence;
      };
  BOOST_CHECK_THROW(Test(testList, HALF, fillValue, programBuilder),
                    poplar::graph_connection_error);
}

BOOST_AUTO_TEST_CASE(FillTest) {
  // The overloads of fill are tested by `ZeroTest` (as zero() is implemented
  // using fill()) so there's no need to test them here as well.

  // Check that `popops::fill()` works with a non-zero value.
  {
    constexpr float fillValue = 3.0;
    constexpr std::array<TestParams, 3> testList = {
        {{0, 0, 0}, {1, 1, 0}, {2, 1, 0}}};
    Test(testList, HALF, fillValue,
         [](const TestParams &test, poplar::Graph &graph, Tensor &sliceOut) {
           Sequence sequence;
           popops::fill(graph, sliceOut, sequence, fillValue);
           return sequence;
         });
  }
}

BOOST_AUTO_TEST_CASE(ZeroTest) {
  constexpr std::array<TestParams, 3> testList = {
      {{0, 0, 0}, {1, 1, 0}, {2, 1, 0}}};

  // Test the program overload.
  {
    Test(testList, HALF, 0,
         [](const TestParams &test, poplar::Graph &graph, Tensor &sliceOut) {
           Sequence sequence;
           popops::zero(graph, sliceOut, sequence);
           return sequence;
         });
  }

  // Test the compute set overloads.
  {
    Test(testList, INT, 0,
         [](const TestParams &test, poplar::Graph &graph, Tensor &sliceOut) {
           Sequence sequence;

           auto cs = graph.addComputeSet("ZeroWithTileMapping");
           auto tFlat = sliceOut.flatten();
           graph.reorderToSimplify(&tFlat, {});
           popops::zero(graph, tFlat, graph.getTileMapping(tFlat), cs);
           sequence.add(Execute(cs));

           return sequence;
         });
  }

  {
    Test(testList, FLOAT, 0,
         [](const TestParams &test, poplar::Graph &graph, Tensor &sliceOut) {
           Sequence sequence;

           auto cs = graph.addComputeSet("ZeroWithSingleTile");
           popops::zero(graph, sliceOut, 0, cs);
           sequence.add(Execute(cs));

           return sequence;
         });
  }

  {
    Test(testList, UNSIGNED_INT, 0,
         [](const TestParams &test, poplar::Graph &graph, Tensor &sliceOut) {
           Sequence sequence;

           auto cs = graph.addComputeSet("ZeroWithTileRegions");
           popops::zero(graph, sliceOut, {{0, sliceOut.numElements()}}, 0, cs);
           sequence.add(Execute(cs));

           return sequence;
         });
  }
}