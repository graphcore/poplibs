#define BOOST_TEST_MODULE SelectScalarFromRowsTest
#include "TestDevice.hpp"

#include <iostream>

#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <popops/UpdateScalarInRows.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include "popops/EncodingConstants.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

template <std::size_t N1, std::size_t N2>
void findReferenceResult(std::array<float, N1> in,
                         std::array<unsigned, N2> indices,
                         std::vector<std::size_t> in_shape,
                         std::vector<float> &out) {
  std::copy(&in[0], &in[N1], out.begin());

  for (unsigned i = 0; i < N2; i++) {
     if(indices[i] != MASKED_LABEL_CODE) {
      out[in_shape[1] * i + indices[i]]--;
    }
  }
}

template <typename T, std::size_t N1, std::size_t N2>
std::vector<T> deviceUpdateScalarInRows(
               std::array<T, N1> in,
               std::vector<std::size_t> in_shape,
               std::array<unsigned, N2> indices,
               std::vector<std::size_t> indices_shape,
               Type inputType) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  auto seq = Sequence();
  popops::addCodelets(graph);

  auto tIn = graph.addVariable(inputType, in_shape);
  auto tIndices = graph.addVariable(equivalent_device_type<unsigned>().value,
                                    indices_shape);

  mapTensorLinearly(graph, tIn);
  // Map part of the tensor onto the last tile to cause the 2D vertex to be
  // created
  auto tInSlice = tIn.slice({0, 0}, {2, in_shape[1]});
  graph.setTileMapping(tInSlice, 3);
  mapTensorLinearly(graph, tIndices);

  BOOST_REQUIRE_EQUAL(tIn.numElements(), N1);
  BOOST_REQUIRE_EQUAL(tIndices.numElements(), N2);

  updateScalarInRows(graph, tIn, tIndices, seq);

  graph.createHostWrite("in", tIn);
  graph.createHostWrite("indices", tIndices);
  graph.createHostRead("out", tIn);

  Engine eng(graph, seq);
  std::vector<T> out(tIn.numElements());
  auto rawBufSizeHalf = target.getTypeSize(HALF) * tIn.numElements();

  std::vector<char> rawIn(rawBufSizeHalf), rawOut(rawBufSizeHalf);

  device.bind([&](const Device &d) {
    eng.load(d);
     if (inputType == HALF) {
      copyFloatToDeviceHalf(target, &in[0], rawIn.data(), tIn.numElements());
      eng.writeTensor("in", rawIn.data());
    } else {
      eng.writeTensor("in", in.data());
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
  std::array<float, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {0, 2, 1};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestHalf) {
  std::array<float, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {2, 1, 0};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, HALF) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloatMaskedLabel) {
  std::array<float, 9> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<unsigned, 3> indices = {0, 2, MASKED_LABEL_CODE};
  std::vector<float> result(9);
  std::vector<std::size_t> in_shape = {3, 3};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT) ==
             result, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(UpdateScalarInRowsTestFloatLarger) {
  std::array<float, 3000> input;
  std::array<unsigned, 3> indices = {100, 400, 900};
  std::vector<float> result(3000);
  for (unsigned i = 0; i < 3000; i++) {
    input[i] = i;
  }
  std::vector<std::size_t> in_shape = {3, 1000};
  findReferenceResult(input, indices, in_shape, result);

  BOOST_TEST(deviceUpdateScalarInRows(input, in_shape, indices, {3}, FLOAT) ==
             result, boost::test_tools::per_element());
}
