// Copyright (c) 2018 Graphcore Ltd, All rights reserved.

#define BOOST_TEST_MODULE SortTest
#include "TestDevice.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <popops/Sort.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

template <typename T, std::size_t N>
std::array<T, N> deviceSort(std::array<T, N> in,
                            std::vector<std::size_t> shape = {N},
                            unsigned dim = 0) {
  BOOST_REQUIRE(dim < shape.size());

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(equivalent_device_type<T>().value, shape);
  poputil::mapTensorLinearly(graph, tIn);

  BOOST_REQUIRE_EQUAL(tIn.numElements(), N);

  Tensor tOut = sort(graph, tIn, dim, seq);

  graph.createHostWrite("in", tIn);
  graph.createHostRead("out", tOut);

  std::array<T, N> out;
  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in.data(), in.data() + in.size());
    eng.run();

    eng.readTensor("out", out.data(), out.data() + out.size());
  });

  return out;
}

template <typename T1, typename T2, std::size_t N>
std::array<T2, N> deviceSortKV(std::array<T1, N> key, std::array<T2, N> value,
                               std::vector<std::size_t> shape = {N},
                               unsigned dim = 0) {
  BOOST_REQUIRE(dim < shape.size());

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tKey = graph.addVariable(equivalent_device_type<T1>().value, shape);
  Tensor tValue = graph.addVariable(equivalent_device_type<T2>().value, shape);
  poputil::mapTensorLinearly(graph, tKey);
  poputil::mapTensorLinearly(graph, tValue);

  BOOST_REQUIRE_EQUAL(tKey.numElements(), N);
  BOOST_REQUIRE_EQUAL(tValue.numElements(), N);

  Tensor tOut = sortKeyValue(graph, tKey, tValue, dim, seq);

  graph.createHostWrite("key", tKey);
  graph.createHostWrite("value", tValue);
  graph.createHostRead("out", tOut);

  std::array<T2, N> out;
  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("key", key.data(), key.data() + key.size());
    eng.writeTensor("value", value.data(), value.data() + value.size());
    eng.run();

    eng.readTensor("out", out.data(), out.data() + out.size());
  });

  return out;
}

BOOST_AUTO_TEST_CASE(DeviceSortFloat) {
  std::array<float, 64> in;
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(-1024, 1024);
  std::generate(std::begin(in), std::end(in), std::bind(dist, gen));
  auto out = deviceSort(in);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order
  BOOST_CHECK(std::is_sorted(std::begin(out), std::end(out)));

  out = deviceSort(in, {4, 4, 4}, 2);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto begin = out.data() + (i * 16 + j * 4);
      const auto end = out.data() + (i * 16 + (j + 1) * 4);

      BOOST_CHECK(std::is_sorted(begin, end));
    }
  }

  out = deviceSort(in, {4, 4, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[i * 16 + (j + 1) * 4 + k]);
      }
    }
  }

  out = deviceSort(in, {4, 4, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[(i + 1) * 16 + j * 4 + k]);
      }
    }
  }

  out = deviceSort(in, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(out[i * 4 + j] <= out[(i + 1) * 4 + j]);
    }
  }

  out = deviceSort(in, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    const auto begin = out.data() + (i * 4);
    const auto end = out.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}

BOOST_AUTO_TEST_CASE(DeviceSortInt) {
  std::array<int, 64> in;
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(-1024, 1024);
  std::generate(std::begin(in), std::end(in), std::bind(dist, gen));
  auto out = deviceSort(in);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order
  BOOST_CHECK(std::is_sorted(std::begin(out), std::end(out)));

  out = deviceSort(in, {4, 4, 4}, 2);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto begin = out.data() + (i * 16 + j * 4);
      const auto end = out.data() + (i * 16 + (j + 1) * 4);

      BOOST_CHECK(std::is_sorted(begin, end));
    }
  }

  out = deviceSort(in, {4, 4, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[i * 16 + (j + 1) * 4 + k]);
      }
    }
  }

  out = deviceSort(in, {4, 4, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[(i + 1) * 16 + j * 4 + k]);
      }
    }
  }

  out = deviceSort(in, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(out[i * 4 + j] <= out[(i + 1) * 4 + j]);
    }
  }

  out = deviceSort(in, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    const auto begin = out.data() + (i * 4);
    const auto end = out.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}

BOOST_AUTO_TEST_CASE(DeviceSortKVFloat) {
  std::array<float, 64> in;
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(-1024, 1024);
  std::generate(std::begin(in), std::end(in), std::bind(dist, gen));
  auto out = deviceSortKV(in, in);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order
  BOOST_CHECK(std::is_sorted(std::begin(out), std::end(out)));

  out = deviceSortKV(in, in, {4, 4, 4}, 2);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto begin = out.data() + (i * 16 + j * 4);
      const auto end = out.data() + (i * 16 + (j + 1) * 4);

      BOOST_CHECK(std::is_sorted(begin, end));
    }
  }

  out = deviceSortKV(in, in, {4, 4, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[i * 16 + (j + 1) * 4 + k]);
      }
    }
  }

  out = deviceSortKV(in, in, {4, 4, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[(i + 1) * 16 + j * 4 + k]);
      }
    }
  }

  out = deviceSortKV(in, in, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(out[i * 4 + j] <= out[(i + 1) * 4 + j]);
    }
  }

  out = deviceSortKV(in, in, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    const auto begin = out.data() + (i * 4);
    const auto end = out.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}

BOOST_AUTO_TEST_CASE(DeviceSortKVInt) {
  std::array<int, 64> in;
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(-1024, 1024);
  std::generate(std::begin(in), std::end(in), std::bind(dist, gen));
  auto out = deviceSortKV(in, in);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order
  BOOST_CHECK(std::is_sorted(std::begin(out), std::end(out)));

  out = deviceSortKV(in, in, {4, 4, 4}, 2);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto begin = out.data() + (i * 16 + j * 4);
      const auto end = out.data() + (i * 16 + (j + 1) * 4);

      BOOST_CHECK(std::is_sorted(begin, end));
    }
  }

  out = deviceSortKV(in, in, {4, 4, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[i * 16 + (j + 1) * 4 + k]);
      }
    }
  }

  out = deviceSortKV(in, in, {4, 4, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(out[i * 16 + j * 4 + k] <= out[(i + 1) * 16 + j * 4 + k]);
      }
    }
  }

  out = deviceSortKV(in, in, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(out[i * 4 + j] <= out[(i + 1) * 4 + j]);
    }
  }

  out = deviceSortKV(in, in, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    const auto begin = out.data() + (i * 4);
    const auto end = out.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}
