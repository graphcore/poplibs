// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SplineConvTests

#include "popops/SplineBasis.hpp"
#include "poplar/IPUModel.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/codelets.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include <boost/multi_array.hpp>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <poplibs_support/TestDevice.hpp>
#include <random>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace popops;
using namespace poplibs_support;

static inline float splineBasisForwardRef(float v, unsigned kMod,
                                          unsigned degree) {
  if (degree == 1) {
    return 1. - v - kMod + 2. * v * kMod;
  } else if (degree == 2) {
    if (kMod == 0)
      return 0.5 * v * v - v + 0.5;
    else if (kMod == 1)
      return -v * v + v + 0.5;
    else
      return 0.5 * v * v;
  } else if (degree == 3) {
    if (kMod == 0)
      return (1. - v) * (1. - v) * (1. - v) / 6.;
    else if (kMod == 1)
      return (3. * v * v * v - 6. * v * v + 4.) / 6.;
    else if (kMod == 2)
      return (-3. * v * v * v + 3. * v * v + 3. * v + 1.) / 6.;
    else
      return v * v * v / 6.;
  } else {
    return static_cast<float>(-1.);
  }
}

static inline std::tuple<boost::multi_array<float, 2>,
                         boost::multi_array<int, 2>>
getSplineBasisRef(const boost::multi_array<float, 2> &pseudo,
                  const std::vector<int> &kernelSize,
                  const std::vector<unsigned char> &isOpenSpline,
                  unsigned degree) {

  const auto numEdges = pseudo.shape()[0];
  const auto numDims = pseudo.shape()[1];
  const auto numSplines = (size_t)(std::pow(degree + 1, numDims) + 0.5);

  boost::multi_array<float, 2> basis(boost::extents[numEdges][numSplines]);
  boost::multi_array<int, 2> weightIndex(boost::extents[numEdges][numSplines]);

  unsigned k, wi, wiOffset;
  float b;
  for (unsigned e = 0; e < numEdges; ++e) {
    for (unsigned s = 0; s < numSplines; ++s) {
      k = s, wi = 0, wiOffset = 1;
      b = 1.;
      for (unsigned d = 0; d < numDims; d++) {
        unsigned kMod = k % (degree + 1);
        k /= degree + 1;

        auto v = pseudo[e][d];
        v *= kernelSize[d] - degree * isOpenSpline[d];

        wi += ((static_cast<unsigned>(v) + kMod) % kernelSize[d]) * wiOffset;
        wiOffset *= kernelSize[d];

        v -= floor(v);
        v = splineBasisForwardRef(v, kMod, degree);
        b *= v;
      }
      basis[e][s] = b;
      weightIndex[e][s] = wi;
    }
  }

  return std::tie(basis, weightIndex);
}

static bool splineBasisTest(std::size_t numEdges, std::size_t numDims,
                            const Type &inputType, unsigned degree) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  auto pseudo = graph.addVariable(inputType, {numEdges, numDims}, "pseudo");
  poputil::mapTensorLinearly(graph, pseudo);

  auto kernelSize = graph.addVariable(INT, {numDims}, "kernelSize");
  poputil::mapTensorLinearly(graph, kernelSize);

  auto isOpenSpline =
      graph.addVariable(UNSIGNED_CHAR, {numDims}, "isOpenSpline");
  poputil::mapTensorLinearly(graph, isOpenSpline);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostPseudo = allocateHostMemoryForTensor(
      pseudo, "pseudo", graph, uploadProg, downloadProg, tmap);
  auto rawHostKernelSize = allocateHostMemoryForTensor(
      kernelSize, "kernelSize", graph, uploadProg, downloadProg, tmap);
  auto rawHostIsOpenSpline = allocateHostMemoryForTensor(
      isOpenSpline, "isOpenSpline", graph, uploadProg, downloadProg, tmap);

  // Input data buffers
  boost::multi_array<float, 2> randPseudo(boost::extents[numEdges][numDims]);
  std::vector<int> randKernelSize(numDims);
  std::vector<unsigned char> randIsOpenSpline(numDims);

  // Generate input data
  std::mt19937 randomEngine;
  boost::random::uniform_int_distribution<int> randDist(1, 10);
  boost::random::uniform_real_distribution<float> randDistFloat(0.0f, 1.0f);
  boost::random::uniform_int_distribution<unsigned char> randDistBool(0, 1);

  std::generate_n(randPseudo.data(), randPseudo.num_elements(),
                  [&] { return randDistFloat(randomEngine); });

  for (auto &el : randKernelSize) {
    el = randDist(randomEngine);
  }
  for (auto &el : randIsOpenSpline) {
    el = randDistBool(randomEngine);
  }

  // Copy and convert the data from the initialised buffers to the transfer
  // buffers (still on host)
  const auto copyBuffer = [&](auto &buf, auto size,
                              std::unique_ptr<char[]> &rawBuf,
                              const Type dataType) {
    copy(target, buf.data(), size, dataType, rawBuf.get());
    // For HALF, we copy and convert back into the (float) host buffers so
    // that the host buffers contain the exact HALF values (which are exactly
    // representable in float). This helps with the validation for the
    // comparison operators
    if (dataType == HALF)
      copy(target, dataType, rawBuf.get(), buf.data(), size);
  };

  copyBuffer(randPseudo, randPseudo.num_elements(), rawHostPseudo, inputType);
  copyBuffer(randKernelSize, randKernelSize.size(), rawHostKernelSize, INT);
  copyBuffer(randIsOpenSpline, randIsOpenSpline.size(), rawHostIsOpenSpline,
             UNSIGNED_CHAR);

  auto prog = Sequence();

  const size_t numSplines = (size_t)(std::pow(degree + 1, numDims) + 0.5);

  // Add two outputs of the SplineBasis op.
  auto basis = graph.addVariable(inputType, {numEdges, numSplines},
                                 VariableMappingMethod::LINEAR, "basis");
  auto weightIndex =
      graph.addVariable(INT, {numEdges, numSplines},
                        VariableMappingMethod::LINEAR, "weightIndex");

  splineBasis(graph, pseudo, kernelSize, isOpenSpline, basis, weightIndex,
              degree, prog, "/SplineBasis");

  auto rawHostBasis = allocateHostMemoryForTensor(
      basis, "basis", graph, uploadProg, downloadProg, tmap);

  auto rawHostWeightIndex = allocateHostMemoryForTensor(
      weightIndex, "weightIndex", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg});
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  boost::multi_array<float, 2> hostBasis(boost::extents[numEdges][numSplines]);
  boost::multi_array<int, 2> hostWeightIndex(
      boost::extents[numEdges][numSplines]);
  copy(target, inputType, rawHostBasis.get(), hostBasis);
  copy(target, INT, rawHostWeightIndex.get(), hostWeightIndex);

  const auto [refBasis, refWeightIndex] =
      getSplineBasisRef(randPseudo, randKernelSize, randIsOpenSpline, degree);
  const double rtol = inputType == HALF ? 10e-3 : 10e-5;
  const double atol = inputType == HALF ? 10e-4 : 10e-8;

  auto basisCheck = checkIsClose("basis", hostBasis, refBasis, rtol, atol);

  auto indexCheck =
      std::equal(std::begin(hostWeightIndex), std::end(hostWeightIndex),
                 std::begin(refWeightIndex));

  return basisCheck && indexCheck;
}

#define TEST_NAME(name, edges, dims, iType, degree)                            \
  name##_##edges##x##dims##_##iType##_##degree

#define TEST_TYPE(name, edges, dims, iType, degree)                            \
  BOOST_AUTO_TEST_CASE(TEST_NAME(name, edges, dims, iType, degree)) {          \
    auto result = splineBasisTest(edges, dims, iType, degree);                 \
    BOOST_CHECK(result);                                                       \
  }

#define ENUMERATE_VALID_TYPE_TESTS(name, edges, dims, degree)                  \
  TEST_TYPE(name, edges, dims, HALF, degree)                                   \
  TEST_TYPE(name, edges, dims, FLOAT, degree)

ENUMERATE_VALID_TYPE_TESTS(SplineBasis, 6, 1, 1)
ENUMERATE_VALID_TYPE_TESTS(SplineBasis, 6, 2, 1)
ENUMERATE_VALID_TYPE_TESTS(SplineBasis, 6, 1, 2)
ENUMERATE_VALID_TYPE_TESTS(SplineBasis, 6, 1, 3)
ENUMERATE_VALID_TYPE_TESTS(SplineBasis, 64, 3, 2)
ENUMERATE_VALID_TYPE_TESTS(SplineBasis, 1024, 3, 2)
