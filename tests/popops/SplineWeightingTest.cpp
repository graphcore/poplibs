// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SplineWeightingTests

#include "popops/SplineWeighting.hpp"
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

static inline boost::multi_array<float, 2>
getSplineWeightingRef(const boost::multi_array<float, 2> &input,
                      const boost::multi_array<float, 3> &weight,
                      const boost::multi_array<float, 2> &basis,
                      const boost::multi_array<int, 2> &weightIndex) {

  const auto numEdges = input.shape()[0];
  const auto numInCh = input.shape()[1];
  const auto numOutCh = weight.shape()[2];
  const auto numSplines = basis.shape()[1];
  boost::multi_array<float, 2> output(boost::extents[numEdges][numOutCh]);

  for (unsigned e = 0; e < numEdges; ++e) {
    for (unsigned oc = 0; oc < numOutCh; ++oc) {
      float v = static_cast<float>(0.);
      for (unsigned s = 0; s < numSplines; s++) {
        const auto b = basis[e][s];
        const auto wi = weightIndex[e][s];
        for (unsigned ic = 0; ic < numInCh; ic++) {
          float tmp = weight[wi][ic][oc];
          tmp *= b * input[e][ic];
          v += tmp;
        }
      }
      output[e][oc] = v;
    }
  }

  return output;
}

static bool SplineWeightingTest(std::size_t numEdges, std::size_t numSplines,
                                std::size_t numInCh, std::size_t numOutCh,
                                std::size_t kernelSize, const Type &inputType) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto input = graph.addVariable(inputType, {numEdges, numInCh}, "input");
  poputil::mapTensorLinearly(graph, input);

  auto weight =
      graph.addVariable(inputType, {kernelSize, numInCh, numOutCh}, "weight");
  poputil::mapTensorLinearly(graph, weight);

  auto basis = graph.addVariable(inputType, {numEdges, numSplines}, "basis");
  poputil::mapTensorLinearly(graph, basis);

  auto weightIndex =
      graph.addVariable(INT, {numEdges, numSplines}, "weightIndex");
  poputil::mapTensorLinearly(graph, weightIndex);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;

  auto rawHostInput = allocateHostMemoryForTensor(
      input, "input", graph, uploadProg, downloadProg, tmap);
  auto rawHostWeight = allocateHostMemoryForTensor(
      weight, "weight", graph, uploadProg, downloadProg, tmap);
  auto rawHostBasis = allocateHostMemoryForTensor(
      basis, "basis", graph, uploadProg, downloadProg, tmap);
  auto rawHostWeightIndex = allocateHostMemoryForTensor(
      weightIndex, "weightIndex", graph, uploadProg, downloadProg, tmap);

  // Input data buffers
  boost::multi_array<float, 2> randInput(boost::extents[numEdges][numInCh]);
  boost::multi_array<float, 3> randWeight(
      boost::extents[kernelSize][numInCh][numOutCh]);
  boost::multi_array<float, 2> randBasis(boost::extents[numEdges][numSplines]);
  boost::multi_array<int, 2> randWeightIndex(
      boost::extents[numEdges][numSplines]);

  // Generate input data
  std::mt19937 randomEngine;
  boost::random::uniform_int_distribution<int> randDist(0, kernelSize - 1);
  boost::random::uniform_real_distribution<float> randDistFloat(0.0f, 1.0f);

  std::generate_n(randInput.data(), randInput.num_elements(),
                  [&] { return randDistFloat(randomEngine); });
  std::generate_n(randWeight.data(), randWeight.num_elements(),
                  [&] { return randDistFloat(randomEngine); });
  std::generate_n(randBasis.data(), randBasis.num_elements(),
                  [&] { return randDistFloat(randomEngine); });
  std::generate_n(randWeightIndex.data(), randWeightIndex.num_elements(),
                  [&] { return randDist(randomEngine); });

  // Copy and convert the data from the initialised buffers to the transfer
  // buffers (still on host)
  const auto copyBuffer = [&](auto &buf, std::unique_ptr<char[]> &rawBuf,
                              const Type dataType) {
    copy(target, buf.data(), buf.num_elements(), dataType, rawBuf.get());
    // For HALF, we copy and convert back into the (float) host buffers so
    // that the host buffers contain the exact HALF values (which are exactly
    // representable in float). This helps with the validation for the
    // comparison operators
    if (dataType == HALF)
      copy(target, dataType, rawBuf.get(), buf.data(), buf.num_elements());
  };

  copyBuffer(randInput, rawHostInput, inputType);
  copyBuffer(randWeight, rawHostWeight, inputType);
  copyBuffer(randBasis, rawHostBasis, inputType);
  copyBuffer(randWeightIndex, rawHostWeightIndex, INT);

  auto prog = Sequence();
  auto output = splineWeighting(graph, input, weight, basis, weightIndex, prog,
                                "/SplineWeighting");

  auto rawHostOutput = allocateHostMemoryForTensor(
      output, "output", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg});
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  boost::multi_array<float, 2> hostOutput(boost::extents[numEdges][numOutCh]);

  copy(target, inputType, rawHostOutput.get(), hostOutput);

  const auto refOutput =
      getSplineWeightingRef(randInput, randWeight, randBasis, randWeightIndex);
  const double rtol = inputType == HALF ? 10e-3 : 10e-5;
  const double atol = inputType == HALF ? 10e-4 : 10e-8;

  auto outputCheck = checkIsClose("output", hostOutput, refOutput, rtol, atol);

  return outputCheck;
}

#define TEST_NAME(name, edges, splines, in_ch, out_ch, kernelSize, iType)      \
  name##_##edges##x##in_ch##_##splines##_##out_ch##_##kernelSize##_##iType

#define TEST_TYPE(name, edges, splines, in_ch, out_ch, kernelSize, iType)      \
  BOOST_AUTO_TEST_CASE(                                                        \
      TEST_NAME(name, edges, splines, in_ch, out_ch, kernelSize, iType)) {     \
    auto result =                                                              \
        SplineWeightingTest(edges, splines, in_ch, out_ch, kernelSize, iType); \
    BOOST_CHECK(result);                                                       \
  }

#define ENUMERATE_VALID_TYPE_TESTS(name, edges, splines, in_ch, out_ch,        \
                                   kernelSize)                                 \
  TEST_TYPE(name, edges, splines, in_ch, out_ch, kernelSize, HALF)             \
  TEST_TYPE(name, edges, splines, in_ch, out_ch, kernelSize, FLOAT)

ENUMERATE_VALID_TYPE_TESTS(SplineWeighting, 6, 2, 2, 2, 5)
ENUMERATE_VALID_TYPE_TESTS(SplineWeighting, 6, 2, 3, 5, 5)
ENUMERATE_VALID_TYPE_TESTS(SplineWeighting, 512, 6, 3, 5, 5)
