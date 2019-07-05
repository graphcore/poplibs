#define BOOST_TEST_MODULE SelectScalarFromRowsTest
#include "TestDevice.hpp"

#include <cmath>

#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <popops/UpdateScalarInRows.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include "popops/EncodingConstants.hpp"
#include "ScalarInFromRowsCommon.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

template <std::size_t N>
void findReferenceResult(std::vector<float> &in,
                         std::array<unsigned, N> indices,
                         std::vector<std::size_t> in_shape,
                         std::vector<float> &out) {
  std::copy(&in[0], &in[in.size()], out.begin());

  for (unsigned i = 0; i < N; i++) {
     if (indices[i] != MASKED_LABEL_CODE) {
      if (indices[i] > in_shape[1]) {
         for (unsigned j = 0; j < in_shape[0]; j++) {
          out[in_shape[1] * i + j] = nanf("");
        }
      } else {
        out[in_shape[1] * i + indices[i]]--;
      }
    }
  }
}

template <typename T, std::size_t N>
std::vector<T> deviceUpdateScalarInRows(
               const std::vector<T> &in,
               std::vector<std::size_t> &in_shape,
               std::array<unsigned, N> &indices,
               std::vector<std::size_t> indices_shape,
               Type inputType,
               unsigned mapFor2dCheck = false,
               bool rearrange = false,
               const std::vector<unsigned> sliceWidths = {},
               const std::vector<unsigned> sliceWidthGroups = {}) {

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  auto seq = Sequence();
  popops::addCodelets(graph);

  const auto padColumns =
             sliceWidthGroups.size() != 0 ? sliceWidthGroups.size() - 1 : 0;
  auto tIn = graph.addVariable(inputType,
                               {in_shape[0] * in_shape[1] + padColumns});
  auto tIndices = graph.addVariable(equivalent_device_type<unsigned>().value,
                                    indices_shape);

  graph.setTileMapping(tIn, 0);
  if (mapFor2dCheck) {
    // Map part of the tensor onto the last tile to cause the 2D vertex to be
    // created
    graph.setTileMapping(tIn.flatten()[0], 2);
    graph.setTileMapping(tIn.flatten()[2], 2);
  }
  mapTensorLinearly(graph, tIndices);

  BOOST_REQUIRE_EQUAL(tIn.numElements() - padColumns, in.size());
  BOOST_REQUIRE_EQUAL(tIndices.numElements(), N);
  if(sliceWidths.size()) {
    const unsigned requiredInSize = in_shape[0] *
                                    std::accumulate(sliceWidths.begin(),
                                                    sliceWidths.end(), 0);
    BOOST_REQUIRE_EQUAL (in.size(), requiredInSize);
  }
  Tensor tInRearranged;
  std::vector<float> inputRearranged(tIn.numElements());
  if (rearrange) {
    rearrangeTensor(tIn.flatten(), sliceWidths, sliceWidthGroups,
                    tInRearranged, in_shape[0]);

    rearrangeInput(in, inputRearranged, sliceWidths,
                   sliceWidthGroups, in_shape[0]);
  }
  updateScalarInRows(graph, rearrange ?
                     tInRearranged.reshape(in_shape) : tIn.reshape(in_shape),
                     tIndices, seq);

  graph.createHostWrite("in", tIn);
  graph.createHostWrite("indices", tIndices);
  graph.createHostRead("out", rearrange ? tInRearranged : tIn);

  Engine eng(graph, seq);
  std::vector<T> out(tIn.numElements() - padColumns);
  auto rawBufSizeHalf = target.getTypeSize(HALF) * tIn.numElements();

  std::vector<char> rawIn(rawBufSizeHalf), rawOut(rawBufSizeHalf);

  device.bind([&](const Device &d) {
    eng.load(d);
     if (inputType == HALF) {
      copyFloatToDeviceHalf(target,
                            rearrange ? inputRearranged.data() : in.data(),
                            rawIn.data(),
                            tIn.numElements());
      eng.writeTensor("in", rawIn.data());
    } else {
      eng.writeTensor("in", rearrange ? inputRearranged.data() : in.data());
    }
    eng.writeTensor("indices", indices.data());
    eng.run();

    if (inputType ==HALF) {
      eng.readTensor("out", rawOut.data());
      copyDeviceHalfToFloat(target, rawOut.data(), out.data(),
                            tIn.numElements());
    } else {
      eng.readTensor("out", out.data());
    }
  });

  return out;
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloat) {
  std::vector<float>  input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {0, 2, 1000};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  auto deviceResult = deviceUpdateScalarInRows(input, in_shape, indices, {3},
                                               FLOAT);
  for (unsigned i = 0; i < result.size(); i++) {
    if (std::isnan(result[i]) && std::isnan(deviceResult[i])) {
      result[i] = -1.1f;
      deviceResult[i] = -1.1f;
    }
  }
  BOOST_TEST(deviceResult == result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestHalf) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {2, 1, 0};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, HALF) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestHalf2D) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {2, 1, 0};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, HALF,
             true) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloatMaskedLabel) {
  std::vector<float>  input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {0, 2, MASKED_LABEL_CODE};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloatColumnLayout) {
  std::vector<std::size_t> in_shape = {3, 4};
  const unsigned dataSize = in_shape[0] * in_shape[1];
  std::vector<float> input(dataSize);
  std::array<unsigned, 3> indices = {2, 3, 1};
  // Column groups with these widths:
  std::vector<unsigned> sliceWidths = {1, 2, 1};
  // Number of column groups in each region (just 1 region)
  std::vector<unsigned> sliceWidthGroups =
                        {static_cast<unsigned>(sliceWidths.size())};
  std::vector<float> result(dataSize);
  for (unsigned i = 0; i < dataSize; i++) {
    input[i] = i;
  }
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT,
             false, true, sliceWidths, sliceWidthGroups) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloatIrregularColumnLayout) {
  std::vector<std::size_t> in_shape = {3, 100};
  const unsigned dataSize = in_shape[0] * in_shape[1];
  std::vector<float> input(dataSize);
  std::array<unsigned, 3> indices = {20, 50, 15};
  // Column groups with these widths:
  std::vector<unsigned> sliceWidths = {25, 45, 30};
  // Number of column groups in each region (just 1 region)
  std::vector<unsigned> sliceWidthGroups =
                        {static_cast<unsigned>(sliceWidths.size())};
  std::vector<float> result(dataSize);
  for (unsigned i = 0; i < dataSize; i++) {
    input[i] = i;
  }
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT,
             true, true, sliceWidths, sliceWidthGroups) ==
             result, boost::test_tools::per_element());
}


BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloatMultiRegion) {
  std::vector<std::size_t> in_shape = {3, 1000};
  const unsigned dataSize = in_shape[0] * in_shape[1];
  std::vector<float> input(dataSize);
  std::array<unsigned, 3> indices = {200, 500, 100};
  // Column groups with these widths:
  // It is necessary to make 13 regions to have a worker process 3 regions
  // and therefore test computation of region start properly
  std::vector<unsigned> sliceWidths = {50, 100,
                                       10,
                                       70, 60,
                                       60, 10, 20,
                                       10, 20,
                                       50, 40, 10,
                                       10, 20, 50, 50,
                                       10, 20, 30,
                                       20, 30, 40,
                                       30,
                                       40, 30,
                                       10, 20, 30,
                                       20, 20, 10
                                       };
  // Number of column groups in each region
  std::vector<unsigned> sliceWidthGroups = {2, 1, 2, 3, 2, 3,
                                            4, 3, 3, 1, 2, 3,
                                            3};
  std::vector<float> result(dataSize);
  for (unsigned i = 0; i < dataSize; i++) {
    input[i] = i;
  }
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT,
             false, true, sliceWidths, sliceWidthGroups) ==
             result, boost::test_tools::per_element());
}
