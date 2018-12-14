#define BOOST_TEST_MODULE EncodingTests

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include "poplar/IPUModel.hpp"
#include "popops/codelets.hpp"
#include "popops/Encoding.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/TileMapping.hpp"
#include "poplibs_test/Util.hpp"
#include "TestDevice.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <boost/random.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace popops;

static inline std::vector<std::uint64_t>
getRandomIndices(std::size_t numIndices,
                 std::size_t length) {
  std::vector<std::uint64_t> indices(numIndices);
  std::mt19937 randomEngine;
  boost::random::uniform_int_distribution<std::uint64_t> dist(0, length - 1);
  for (std::size_t i = 0; i < numIndices; i++) {
    indices[i] = dist(randomEngine);
  }
  return indices;
}

static inline boost::multi_array<double, 2>
getEncodedModel(const std::vector<std::uint64_t> &indices,
                std::size_t length) {
  const auto numIndices = indices.size();
  boost::multi_array<double, 2>
    encoded(boost::extents[numIndices][length]);
  for (std::size_t i = 0; i < numIndices; ++i) {
    std::fill_n(&encoded[i][0], encoded[i].size(), 0);
    encoded[i][indices[i]] = 1;
  }
  return encoded;
}

template <typename IndexType>
static inline void
copyIndices(const std::vector<std::uint64_t> &indices,
            char *out) {
  auto *typed = reinterpret_cast<IndexType*>(out);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    BOOST_CHECK(indices[i] <= std::numeric_limits<IndexType>::max());
    typed[i] = static_cast<IndexType>(indices[i]);
  }
}

static inline void
copyIndices(const std::vector<std::uint64_t> &indices,
            const Type &indexType,
            char *mem) {
  if (indexType == UNSIGNED_INT) {
    copyIndices<unsigned>(indices, mem);
  } else if (indexType == INT) {
    copyIndices<int>(indices, mem);
  }
}

static bool encodeTest(std::size_t numIndices,
                       std::size_t length,
                       const Type &indicesType,
                       const Type &encodedType) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);
  auto indices = graph.addVariable(indicesType, {numIndices}, "indices");
  poputil::mapTensorLinearly(graph, indices);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto rawHostIndices =
    allocateHostMemoryForTensor(indices, "indices", graph, uploadProg,
                                downloadProg, tmap);

  auto randIndices = getRandomIndices(numIndices, length);
  auto modelEncoded = getEncodedModel(randIndices, length);
  copyIndices(randIndices, indicesType, rawHostIndices.get());

  auto prog = Sequence();
  auto encoded = graph.addVariable(encodedType, {numIndices, length},
                                   VariableMappingMethod::LINEAR, "encoded");
  encodeOneHot(graph, indices, encoded, prog, "/OneHotEncodeTest");
  auto rawHostEncoded =
    allocateHostMemoryForTensor(encoded, "encoded", graph, uploadProg,
                                downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  boost::multi_array<double, 2>
    hostEncoded(boost::extents[numIndices][length]);
  copy(target, encodedType, rawHostEncoded.get(), hostEncoded);

  // No calculation just assignment here and 0 or 1 exactly representable
  // by any type.
  constexpr double relativeTolerance = 0, absoluteTolerance = 0;
  auto matchesModel = checkIsClose("encoded", hostEncoded,
                                   modelEncoded,
                                   relativeTolerance, absoluteTolerance);
  return matchesModel;
}

#define TEST_NAME(name, n, l, iType, eType) \
  name ## _ ## n ## x ## l ## _ ## iType ## _ ## eType

#define TEST_TYPE(name, n, l, iType, eType) \
  BOOST_AUTO_TEST_CASE(TEST_NAME(name, n, l, iType, eType)) { \
    auto matchesModel = encodeTest(n, l, iType, eType); \
    BOOST_CHECK(matchesModel); \
  }


#define ENUMERATE_VALID_TYPE_TESTS(name, n, l) \
  TEST_TYPE(name, n, l, UNSIGNED_INT, FLOAT) \
  TEST_TYPE(name, n, l, UNSIGNED_INT, HALF) \
  TEST_TYPE(name, n, l, UNSIGNED_INT, UNSIGNED_INT) \
  TEST_TYPE(name, n, l, UNSIGNED_INT, INT) \
  TEST_TYPE(name, n, l, INT, FLOAT) \
  TEST_TYPE(name, n, l, INT, HALF) \
  TEST_TYPE(name, n, l, INT, UNSIGNED_INT) \
  TEST_TYPE(name, n, l, INT, INT)

ENUMERATE_VALID_TYPE_TESTS(EncodeOneHot, 1, 1)
ENUMERATE_VALID_TYPE_TESTS(EncodeOneHot, 10, 100)
