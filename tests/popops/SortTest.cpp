// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE SortTest
#include <poplibs_support/TestDevice.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
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
using namespace poplibs_support;

template <typename T, std::size_t N>
std::array<T, N>
deviceSort(std::array<T, N> in, const Type dType, bool inPlace = false,
           std::vector<std::size_t> shape = {N}, unsigned dim = 0) {
  BOOST_REQUIRE(dim < shape.size());

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(dType, shape);
  poputil::mapTensorLinearly(graph, tIn);

  BOOST_REQUIRE_EQUAL(tIn.numElements(), N);

  Tensor tOut;
  if (inPlace) {
    sortInPlace(graph, tIn, dim, seq);
    tOut = tIn;
  } else {
    tOut = sort(graph, tIn, dim, seq);
  }

  graph.createHostWrite("in", tIn);
  graph.createHostRead("out", tOut);

  std::array<T, N> out;
  Engine eng(graph, seq);

  auto target = graph.getTarget();
  auto rawBufSize = target.getTypeSize(dType) * N;
  std::vector<char> rawIn(rawBufSize), rawOut(rawBufSize);
  if constexpr (std::is_same_v<T, float>) {
    if (dType == HALF) {
      poplar::copyFloatToDeviceHalf(target, in.data(), rawIn.data(), N);
    }
  }

  device.bind([&](const Device &d) {
    eng.load(d);
    if (dType == HALF) {
      eng.writeTensor("in", rawIn.data(), rawIn.data() + rawIn.size());
    } else {
      eng.writeTensor("in", in.data(), in.data() + in.size());
    }
    eng.run();
    if (dType == HALF) {
      eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
    } else {
      eng.readTensor("out", out.data(), out.data() + out.size());
    }
  });

  if constexpr (std::is_same_v<T, float>) {
    if (dType == HALF) {
      poplar::copyDeviceHalfToFloat(target, rawOut.data(), out.data(), N);
    }
  }

  return out;
}

template <typename T1, typename T2, std::size_t N>
std::array<T2, N>
deviceSortKV(std::array<T1, N> key, std::array<T2, N> value, const Type keyType,
             const Type valueType, bool inPlace = false,
             std::vector<std::size_t> shape = {N}, unsigned dim = 0) {
  BOOST_REQUIRE(dim < shape.size());

  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  auto seq = Sequence();
  popops::addCodelets(graph);

  Tensor tKey = graph.addVariable(keyType, shape);
  Tensor tValue = graph.addVariable(valueType, shape);
  poputil::mapTensorLinearly(graph, tKey);
  poputil::mapTensorLinearly(graph, tValue);

  BOOST_REQUIRE_EQUAL(tKey.numElements(), N);
  BOOST_REQUIRE_EQUAL(tValue.numElements(), N);

  Tensor tOut;
  if (inPlace) {
    sortKeyValueInPlace(graph, tKey, tValue, dim, seq);
    tOut = tValue;
  } else {
    tOut = sortKeyValue(graph, tKey, tValue, dim, seq);
  }

  graph.createHostWrite("key", tKey);
  graph.createHostWrite("value", tValue);
  graph.createHostRead("out", tOut);

  auto target = graph.getTarget();
  std::vector<char> rawKey(target.getTypeSize(keyType) * N);
  std::vector<char> rawValue(target.getTypeSize(valueType) * N);
  std::vector<char> rawOut(target.getTypeSize(valueType) * N);
  if constexpr (std::is_same_v<T1, float>) {
    if (keyType == HALF) {
      poplar::copyFloatToDeviceHalf(target, key.data(), rawKey.data(), N);
    }
  }
  if constexpr (std::is_same_v<T2, float>) {
    if (valueType == HALF) {
      poplar::copyFloatToDeviceHalf(target, value.data(), rawValue.data(), N);
    }
  }
  std::array<T2, N> out;
  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    if (keyType == HALF) {
      eng.writeTensor("key", rawKey.data(), rawKey.data() + rawKey.size());
    } else {
      eng.writeTensor("key", key.data(), key.data() + key.size());
    }
    if (valueType == HALF) {
      eng.writeTensor("value", rawValue.data(),
                      rawValue.data() + rawValue.size());
    } else {
      eng.writeTensor("value", value.data(), value.data() + value.size());
    }
    eng.run();

    if (valueType == HALF) {
      eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
    } else {
      eng.readTensor("out", out.data(), out.data() + out.size());
    }
  });

  if constexpr (std::is_same_v<T2, float>) {
    if (valueType == HALF) {
      poplar::copyDeviceHalfToFloat(target, rawOut.data(), out.data(), N);
    }
  }

  return out;
}

template <typename T>
void DeviceSortTest(const Type dType = equivalent_device_type<T>().value,
                    bool inPlace = false) {
  std::array<T, 64> in;
  boost::random::mt19937 gen;
  auto rangeLowLimit = dType == UNSIGNED_INT ? 0 : -1024;
  boost::random::uniform_int_distribution<> dist(rangeLowLimit, 1024);
  std::generate(std::begin(in), std::end(in), std::bind(dist, gen));
  auto out = deviceSort(in, dType, inPlace);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order
  BOOST_CHECK(std::is_sorted(std::begin(out), std::end(out)));

  out = deviceSort(in, dType, inPlace, {4, 4, 4}, 2);
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

  out = deviceSort(in, dType, inPlace, {4, 4, 4}, 1);
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

  out = deviceSort(in, dType, inPlace, {4, 4, 4}, 0);
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

  out = deviceSort(in, dType, inPlace, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(out[i * 4 + j] <= out[(i + 1) * 4 + j]);
    }
  }

  out = deviceSort(in, dType, inPlace, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    const auto begin = out.data() + (i * 4);
    const auto end = out.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}

template <typename T>
void DeviceSortKVTest(const Type dType = equivalent_device_type<T>().value,
                      bool inPlace = false) {
  std::array<T, 64> in;
  boost::random::mt19937 gen;
  auto rangeLowLimit = dType == UNSIGNED_INT ? 0 : -1024;
  boost::random::uniform_int_distribution<> dist(rangeLowLimit, 1024);
  std::generate(std::begin(in), std::end(in), std::bind(dist, gen));
  auto out = deviceSortKV(in, in, dType, dType, inPlace);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order
  BOOST_CHECK(std::is_sorted(std::begin(out), std::end(out)));

  out = deviceSortKV(in, in, dType, dType, inPlace, {4, 4, 4}, 2);
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

  out = deviceSortKV(in, in, dType, dType, inPlace, {4, 4, 4}, 1);
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

  out = deviceSortKV(in, in, dType, dType, inPlace, {4, 4, 4}, 0);
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

  out = deviceSortKV(in, in, dType, dType, inPlace, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(out[i * 4 + j] <= out[(i + 1) * 4 + j]);
    }
  }

  out = deviceSortKV(in, in, dType, dType, inPlace, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(in), std::end(in), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (int i = 0; i < 15; ++i) {
    const auto begin = out.data() + (i * 4);
    const auto end = out.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}

BOOST_AUTO_TEST_CASE(DeviceSortTestAllTypes) {
  DeviceSortTest<float>(FLOAT);
  DeviceSortTest<float>(HALF);
  DeviceSortTest<unsigned>(UNSIGNED_INT);
  DeviceSortTest<int>(INT);
}

BOOST_AUTO_TEST_CASE(DeviceSortInPlaceTestAllTypes) {
  DeviceSortTest<float>(FLOAT, true);
  DeviceSortTest<float>(HALF, true);
  DeviceSortTest<unsigned>(UNSIGNED_INT, true);
  DeviceSortTest<int>(INT, true);
}

BOOST_AUTO_TEST_CASE(DeviceSortKVTestAllTypes) {
  DeviceSortKVTest<float>();
  DeviceSortKVTest<float>(HALF);
  DeviceSortKVTest<unsigned>();
  DeviceSortKVTest<int>();
}

BOOST_AUTO_TEST_CASE(DeviceSortKVInPlaceTestAllTypes) {
  DeviceSortKVTest<float>(FLOAT, true);
  DeviceSortKVTest<float>(HALF, true);
  DeviceSortKVTest<unsigned>(UNSIGNED_INT, true);
  DeviceSortKVTest<int>(INT, true);
}

BOOST_AUTO_TEST_CASE(DeviceSortKVFloatUInt) {
  std::array<float, 64> key;
  std::array<unsigned, 64> value;
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<> dist(-1024, 1024);
  std::generate(std::begin(key), std::end(key), std::bind(dist, gen));
  std::iota(value.begin(), value.end(), 0);
  auto keyType = equivalent_device_type<float>().value;
  auto valueType = equivalent_device_type<unsigned>().value;
  auto out = deviceSortKV(key, value, keyType, valueType);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order
  std::array<float, 64> keyPermuted;
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  BOOST_CHECK(std::is_sorted(std::begin(keyPermuted), std::end(keyPermuted)));

  out = deviceSortKV(key, value, keyType, valueType, true, {4, 4, 4}, 2);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto begin = keyPermuted.data() + (i * 16 + j * 4);
      const auto end = keyPermuted.data() + (i * 16 + (j + 1) * 4);

      BOOST_CHECK(std::is_sorted(begin, end));
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {4, 4, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(keyPermuted[i * 16 + j * 4 + k] <=
                    keyPermuted[i * 16 + (j + 1) * 4 + k]);
      }
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {4, 4, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(keyPermuted[i * 16 + j * 4 + k] <=
                    keyPermuted[(i + 1) * 16 + j * 4 + k]);
      }
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(keyPermuted[i * 4 + j] <= keyPermuted[(i + 1) * 4 + j]);
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 15; ++i) {
    const auto begin = keyPermuted.data() + (i * 4);
    const auto end = keyPermuted.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}

BOOST_AUTO_TEST_CASE(DeviceSortKVFloatInt) {
  std::array<float, 64> key;
  std::array<int, 64> value;
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<> dist(-1024, 1024);
  std::generate(std::begin(key), std::end(key), std::bind(dist, gen));
  std::iota(value.begin(), value.end(), 0);
  auto keyType = equivalent_device_type<float>().value;
  auto valueType = equivalent_device_type<unsigned>().value;
  auto out = deviceSortKV(key, value, keyType, valueType);

  // Check that we have the same elements in some order
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order
  std::array<float, 64> keyPermuted;
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  BOOST_CHECK(std::is_sorted(std::begin(keyPermuted), std::begin(keyPermuted)));

  out = deviceSortKV(key, value, keyType, valueType, true, {4, 4, 4}, 2);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto begin = keyPermuted.data() + (i * 16 + j * 4);
      const auto end = keyPermuted.data() + (i * 16 + (j + 1) * 4);

      BOOST_CHECK(std::is_sorted(begin, end));
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {4, 4, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(keyPermuted[i * 16 + j * 4 + k] <=
                    keyPermuted[i * 16 + (j + 1) * 4 + k]);
      }
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {4, 4, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        BOOST_CHECK(keyPermuted[i * 16 + j * 4 + k] <=
                    keyPermuted[(i + 1) * 16 + j * 4 + k]);
      }
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {16, 4}, 0);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK(keyPermuted[i * 4 + j] <= keyPermuted[(i + 1) * 4 + j]);
    }
  }

  out = deviceSortKV(key, value, keyType, valueType, true, {16, 4}, 1);
  BOOST_CHECK(
      std::is_permutation(std::begin(value), std::end(value), std::begin(out)));

  // Check that the elements are in sorted order on the specified dimension
  for (std::size_t i = 0; i < out.size(); ++i) {
    keyPermuted[i] = key[out[i]];
  }
  for (int i = 0; i < 15; ++i) {
    const auto begin = keyPermuted.data() + (i * 4);
    const auto end = keyPermuted.data() + ((i + 1) * 4);

    BOOST_CHECK(std::is_sorted(begin, end));
  }
}
