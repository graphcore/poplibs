// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PaddingTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/Compiler.hpp>
#include <popops/Pad.hpp>
#include <popops/codelets.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

#define DIM_SIZE 4

void padWithTensor(const float in[DIM_SIZE], const float *constant,
                   const unsigned constantSize, float out[DIM_SIZE + 1],
                   const std::vector<ptrdiff_t> &pLows,
                   const std::vector<ptrdiff_t> &pUpps) {
  BOOST_CHECK(pLows.size() == pUpps.size());
  BOOST_CHECK(pLows.size() == 1);
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(FLOAT, {DIM_SIZE}, "t1");
  graph.setTileMapping(tIn, 0);
  Tensor tC = graph.addConstant(FLOAT, {constantSize}, constant);
  graph.setTileMapping(tC, 0);

  auto seq = Sequence();
  const auto tOut = popops::pad(graph, tIn, pLows, pUpps, tC);

  graph.createHostWrite("in", tIn);
  graph.createHostRead("out", tOut);

  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in, &in[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", out, &out[DIM_SIZE + 1]);
  });
}

// Function which iterates padding that has been applied to
// a tensor in each dimension starting from outermost dimension.
// Padding is stripped from each dimension as we iterate to unwrap
// the padding from the given tensor in the reverse operation to
// applying padding in the first place.
template <typename Fn>
static void
iteratePadding(const Graph &graph, const Tensor &tIn, const Tensor &tOut,
               const std::vector<ptrdiff_t> &pLows,
               const std::vector<ptrdiff_t> &pUpps, const Fn &predicate) {
  auto stripped = tOut;
  for (std::size_t d = 0; d < pLows.size(); ++d) {
    auto dimSize = stripped.dim(d);
    auto lower = stripped.slice(0, pLows[d], d);
    auto upper = stripped.slice(dimSize - pUpps[d], dimSize, d);
    stripped = stripped.slice(pLows[d], dimSize - pUpps[d], d);
    predicate(stripped, lower, upper, d);
  }
}

void checkMapping(const Graph &graph, const Tensor &tIn, const Tensor &tOut,
                  const std::vector<ptrdiff_t> &pLows,
                  const std::vector<ptrdiff_t> &pUpps,
                  padding::MappingMethod mappingMethod) {
  switch (mappingMethod) {
  case padding::MappingMethod::NONE:
    iteratePadding(graph, tIn, tOut, pLows, pUpps,
                   [&](const Tensor &tStripped, const Tensor &lower,
                       const Tensor &upper, unsigned dim) {
                     auto padding = concat(lower, upper, dim);
                     auto mapping = graph.getTileMapping(padding, false);
                     for (unsigned tile = 0; tile < mapping.size(); ++tile) {
                       BOOST_CHECK(mapping[tile].empty());
                     }
                   });
    break;
  case padding::MappingMethod::ZERO:
    iteratePadding(graph, tIn, tOut, pLows, pUpps,
                   [&](const Tensor &tStripped, const Tensor &lower,
                       const Tensor &upper, unsigned dim) {
                     auto padding = concat(lower, upper, dim);
                     auto mapping = graph.getTileMapping(padding, false);
                     auto t0MappedElems = std::accumulate(
                         mapping[0].begin(), mapping[0].end(), std::size_t(0),
                         [](std::size_t t, const poplar::Interval &i) {
                           return t + i.size();
                         });
                     // We expect all padding elements mapped to tile 0.
                     BOOST_CHECK(t0MappedElems == padding.numElements());
                     for (unsigned tile = 1; tile < mapping.size(); ++tile) {
                       BOOST_CHECK(mapping[tile].empty());
                     }
                   });
    break;
  case padding::MappingMethod::EDGE:
    iteratePadding(
        graph, tIn, tOut, pLows, pUpps,
        [&](const Tensor &tStripped, const Tensor &lower, const Tensor &upper,
            unsigned dim) {
          auto expectedLowerMapping = graph.getTileMapping(
              tStripped.slice(0, 1, dim).broadcast(lower.dim(dim), dim));
          auto expectedUpperMapping = graph.getTileMapping(
              tStripped.slice(tStripped.dim(dim) - 1, tStripped.dim(dim), dim)
                  .broadcast(upper.dim(dim), dim));
          BOOST_CHECK(expectedLowerMapping == graph.getTileMapping(lower));
          BOOST_CHECK(expectedUpperMapping == graph.getTileMapping(upper));
        });
    break;
  default:
    throw std::logic_error("Unknown mapping method");
  }
}

void padWithConstant(
    const float in[DIM_SIZE], const float constant, float out[DIM_SIZE + 1],
    const std::vector<ptrdiff_t> &pLows, const std::vector<ptrdiff_t> &pUpps,
    const std::vector<std::vector<poplar::Interval>> &prePadTileMapping,
    padding::MappingMethod mappingMethod) {
  BOOST_CHECK(pLows.size() == pUpps.size());
  BOOST_CHECK(pLows.size() == 1);
  auto numTiles = prePadTileMapping.size();
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(FLOAT, {DIM_SIZE}, "t1");
  graph.setTileMapping(tIn, prePadTileMapping);

  auto seq = Sequence();
  const auto tOut =
      popops::pad(graph, tIn, pLows, pUpps, constant, mappingMethod);
  checkMapping(graph, tIn, tOut, pLows, pUpps, mappingMethod);

  // In this case we need to provide a mapping for the padding so just
  // give the whole thing an arbitrary easy mapping.
  if (mappingMethod == padding::MappingMethod::NONE) {
    graph.setTileMapping(tOut, 0);
  }

  graph.createHostWrite("in", tIn);
  graph.createHostRead("out", tOut);

  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in, &in[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", out, &out[DIM_SIZE + 1]);
  });
}

BOOST_AUTO_TEST_CASE(PaddingWithTensor) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float c = 5.0f;
  float out[DIM_SIZE + 1];
  padWithTensor(in, &c, 1, out, {0}, {1});
  const float expect_out[DIM_SIZE + 1] = {1.0f, 2.0f, 3.0f, 4.0f, c};
  for (unsigned i = 0; i < DIM_SIZE + 1; ++i) {
    BOOST_CHECK_EQUAL(out[i], expect_out[i]);
  }
}

BOOST_AUTO_TEST_CASE(PaddingWithNonScalarTensor) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float c[2] = {5.0f, 6.0f};
  float out[DIM_SIZE + 1];
  BOOST_CHECK_THROW(padWithTensor(in, &c[0], 2, out, {0}, {1}),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(PaddingWithConstantNoMapping) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto inMapping = std::vector<std::vector<Interval>>{
      {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 4}}};
  const float c = 5.0f;
  float out[DIM_SIZE + 1];
  padWithConstant(in, c, out, {0}, {1}, inMapping,
                  padding::MappingMethod::NONE);
  const float expect_out[DIM_SIZE + 1] = {1.0f, 2.0f, 3.0f, 4.0f, c};
  for (unsigned i = 0; i < DIM_SIZE + 1; ++i) {
    BOOST_CHECK_EQUAL(out[i], expect_out[i]);
  }
}

BOOST_AUTO_TEST_CASE(PaddingWithConstantZeroMapping) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto inMapping = std::vector<std::vector<Interval>>{
      {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 4}}};
  const float c = 5.0f;
  float out[DIM_SIZE + 1];
  padWithConstant(in, c, out, {0}, {1}, inMapping,
                  padding::MappingMethod::ZERO);
  const float expect_out[DIM_SIZE + 1] = {1.0f, 2.0f, 3.0f, 4.0f, c};
  for (unsigned i = 0; i < DIM_SIZE + 1; ++i) {
    BOOST_CHECK_EQUAL(out[i], expect_out[i]);
  }
}

BOOST_AUTO_TEST_CASE(PaddingWithConstantEdgeMapping) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto inMapping = std::vector<std::vector<Interval>>{
      {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 4}}};
  const float c = 5.0f;
  float out[DIM_SIZE + 1];
  padWithConstant(in, c, out, {0}, {1}, inMapping,
                  padding::MappingMethod::EDGE);
  const float expect_out[DIM_SIZE + 1] = {1.0f, 2.0f, 3.0f, 4.0f, c};
  for (unsigned i = 0; i < DIM_SIZE + 1; ++i) {
    BOOST_CHECK_EQUAL(out[i], expect_out[i]);
  }
}
