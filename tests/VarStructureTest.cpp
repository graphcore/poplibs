#define BOOST_TEST_MODULE VarStructureTest
#include <boost/test/unit_test.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poputil/VarStructure.hpp>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

BOOST_AUTO_TEST_CASE(CreatePartitionableTensorNoSplit) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {3, 8, 18};
  const std::vector<std::size_t> splits = {1, 1, 1};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  // This should be equivalent to `graph.addVariable`
  BOOST_CHECK_EQUAL(t.getContiguousRegions().size(), 1);
  BOOST_CHECK(t.shape() == shape);
}

BOOST_AUTO_TEST_CASE(CreatePartitionableTensorEvenSplit) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {3, 8, 18};
  const std::vector<std::size_t> splits = {3, 2, 9};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  const auto firstP = shape[0] / splits[0];
  for (unsigned first = 0; first < splits[0]; ++first) {
    const auto firstT =
      t.slice(first * firstP, (first + 1) * firstP, 0);
    const auto secondP = shape[1] / splits[1];
    for (unsigned second = 0; second < splits[1]; ++second) {
      const auto secondT =
        firstT.slice(second * secondP, (second + 1) * secondP, 1);
      const auto thirdP = shape[2] / splits[2];
      for (unsigned third = 0; third < splits[2]; ++third) {
        const auto thirdT =
          secondT.slice(third * thirdP, (third + 1) * thirdP, 2);
        BOOST_CHECK_EQUAL(thirdT.getContiguousRegions().size(), 1);
      }
    }
  }
  BOOST_CHECK(t.shape() == shape);
}

static Tensor
getPartitionSlice(const Tensor &t,
                  unsigned d,
                  unsigned dim,
                  unsigned i) {
  const auto dimElems = t.dim(dim);
  const auto ceil = ceildiv(dimElems, d);
  return t.slice(std::min(i * ceil, dimElems),
                 std::min((i + 1) * ceil, dimElems), dim);
}

BOOST_AUTO_TEST_CASE(CreatePartitionableTensorUnevenSplit) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {5, 8, 48};
  const std::vector<std::size_t> splits = {4, 3, 10};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  for (unsigned first = 0; first < splits[0]; ++first) {
    const auto firstT = getPartitionSlice(t, splits[0], 0, first);
    for (unsigned second = 0; second < splits[1]; ++second) {
      const auto secondT = getPartitionSlice(firstT, splits[1], 1, second);
      for (unsigned third = 0; third < splits[2]; ++third) {
        const auto thirdT = getPartitionSlice(secondT, splits[2], 2, third);
        // Some partitions may have 0 elements when split unevenly.
        BOOST_CHECK_LE(thirdT.getContiguousRegions().size(), 1);
      }
    }
  }
  BOOST_CHECK(t.shape() == shape);
}

BOOST_AUTO_TEST_CASE(CreatePartitionableTensorZeroElements) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {3, 8, 0, 18};
  const std::vector<std::size_t> splits = {1, 1, 1, 1};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  // This should be equivalent to `graph.addVariable`
  BOOST_CHECK_EQUAL(t.getContiguousRegions().size(), 0);
  BOOST_CHECK(t.shape() == shape);
}

BOOST_AUTO_TEST_CASE(IterateTensorPartitionsNoSplits) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {3, 8, 18};
  const std::vector<std::size_t> splits = {1, 1, 1};
  // The iteration function is defined in terms of createPartitionableTensor
  // therefore we use this to test it.
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  iterateTensorPartitions(t, splits,
    [](const std::vector<std::size_t> &i,
       const Tensor &s) {
      BOOST_CHECK_EQUAL(s.getContiguousRegions().size(), 1);
      BOOST_CHECK_EQUAL(i.size(), 3);
      BOOST_CHECK_EQUAL(s.rank(), 3);
      BOOST_CHECK_EQUAL(
        std::accumulate(i.begin(), i.end(), std::size_t(0)), 0);
    });
}

BOOST_AUTO_TEST_CASE(IterateTensorPartitionsEvenSplit) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {3, 8, 18};
  const std::vector<std::size_t> splits = {3, 2, 9};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  iterateTensorPartitions(t, splits,
    [](const std::vector<std::size_t> &i,
       const Tensor &s) {
      BOOST_CHECK_EQUAL(s.getContiguousRegions().size(), 1);
    });
}

BOOST_AUTO_TEST_CASE(IterateTensorPartitionsUnevenSplit) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {5, 8, 48};
  const std::vector<std::size_t> splits = {4, 3, 10};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  iterateTensorPartitions(t, splits,
    [](const std::vector<std::size_t> &i,
       const Tensor &s) {
      // Some partitions may have 0 elements when split unevenly.
      BOOST_CHECK_LE(s.getContiguousRegions().size(), 1);
    });
}

BOOST_AUTO_TEST_CASE(IterateTensorPartitionsZeroElements) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  const std::vector<std::size_t> shape = {3, 8, 0, 18};
  const std::vector<std::size_t> splits = {1, 1, 1, 1};
  auto t = createPartitionableTensor(graph, FLOAT, shape, splits, "t");

  // This should still function okay and give the correct slices.
  iterateTensorPartitions(t, splits,
    [](const std::vector<std::size_t> &i,
       const Tensor &s) {
      BOOST_CHECK_EQUAL(s.numElements(), 0);
      BOOST_CHECK_EQUAL(s.getContiguousRegions().size(), 0);
    });
}
