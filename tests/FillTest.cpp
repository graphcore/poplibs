// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// Test for the popops::fill function.
// This test is a modified version of the ZeroTest.cpp test.
#define BOOST_TEST_MODULE FillTest
#include <poplibs_support/TestDevice.hpp>

#include <poplar/Engine.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/Fill.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;

// Define a number of tests to run:
struct TestParams {
  unsigned rows;
  unsigned columns;
  unsigned offset;
};

constexpr std::array<TestParams, 20> TestList = {{
    {1, 1, 0}, {1, 2, 0}, {1, 3, 0}, {1, 4, 0}, {1, 5, 0}, {1, 6, 0}, {1, 7, 0},
    {1, 8, 0}, {1, 1, 1}, {1, 2, 1}, {1, 3, 1}, {1, 4, 1}, {1, 5, 1}, {1, 6, 1},
    {1, 7, 1}, {1, 8, 1}, {2, 1, 1}, {2, 4, 1}, {2, 7, 1}, {2, 8, 1},
}};

//*************************************************
// Main test function for popops::fill().
//
// Overview:
//
// Memory is initialised as values 0 to N.
// Run a series of tests that fill a varying number of items.
// The results are put into a memory area large enough to
// hold the largest test result, so often the other items are
// expected to nbe the original initialised value.
// This is checked as well as the "wanted" data.
//*************************************************
template <typename FillType>
void FillTest(const Type &dataType, FillType fillValue) {

  // determine the sizes of arrays required
  const auto test_count = TestList.size();

  // TODO: Make these constexpr when we have C++17.
  const auto max_rows =
      std::max_element(TestList.begin(), TestList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.rows < b.rows);
                       })
          ->rows;

  const auto max_columns =
      std::max_element(TestList.begin(), TestList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.columns < b.columns);
                       })
          ->columns;

  const auto max_offset =
      std::max_element(TestList.begin(), TestList.end(),
                       [](const TestParams &a, const TestParams &b) {
                         return (a.offset < b.offset);
                       })
          ->offset;
  // Whole data array size
  const auto total_size = (max_columns + max_offset) * max_rows;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern, dummy data to check its overwritten when
  // it should be, and not when its not
  for (unsigned i = 0; i < total_size; i++)
    inTest[i] = 1;

  auto device = createTestDevice(TEST_TARGET);
  Target target = device.getTarget();

  // Create Graph object
  Graph graph(target);

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
    auto rows = TestList[tests].rows;
    auto columns = TestList[tests].columns;
    auto offset = TestList[tests].offset;

    Sequence sequence;

    // Different slices to test looping decisions, with offset
    Tensor sliceOut;
    if (rows == 1) {
      sliceOut = data.reshape({total_size});
      sliceOut = sliceOut.slice(offset, columns + offset);
    } else {
      sliceOut = data.slice({0, offset}, {rows, columns + offset});
    }

    popops::fill(graph, sliceOut, sequence, fillValue);

    programs.push_back(std::move(sequence));
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
    auto rows = TestList[tests].rows;
    auto columns = TestList[tests].columns;
    auto offset = TestList[tests].offset;

    copy(target, inTest.data(), inTest.size(), dataType, input.get());

    device.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(tests);
      engine.run(downloadProgIndex);
    });

    copy(target, dataType, input.get(), outHost.data(), outHost.size());

    // Host generated result, start with 1s
    for (unsigned i = 0; i < total_size; i++)
      outTest[i] = inTest[i];
    // Then fill the same portion of the input as the code under test
    for (unsigned i = 0; i < rows; i++) {
      for (unsigned j = 0; j < columns; j++) {
        outTest[j + offset + (i * (max_columns + max_offset))] = fillValue;
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

BOOST_AUTO_TEST_CASE(FillTest_bool) { FillTest<bool>(BOOL, true); }
BOOST_AUTO_TEST_CASE(FillTest_half) { FillTest<float>(HALF, 3.0); }
BOOST_AUTO_TEST_CASE(FillTest_float) { FillTest<float>(FLOAT, 1.23456789); }
BOOST_AUTO_TEST_CASE(FillTest_int) { FillTest<int>(INT, 4); }
