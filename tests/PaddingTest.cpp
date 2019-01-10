#define BOOST_TEST_MODULE PaddingTest
#include <popops/Pad.hpp>
#include <poputil/exceptions.hpp>
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

#define DIM_SIZE 4

void padWithTensor(const float in[DIM_SIZE], const float* constant,
                   const unsigned constantSize, float* out,
                   const std::vector<ptrdiff_t> &pLows,
                   const std::vector<ptrdiff_t> &pUpps) {
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
    eng.writeTensor("in", in);
    eng.run();
    eng.readTensor("out", out);
  });
}

void padWithConstant(const float in[DIM_SIZE], const float constant, float* out,
                     const std::vector<ptrdiff_t> &pLows,
                     const std::vector<ptrdiff_t> &pUpps) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Tensor tIn = graph.addVariable(FLOAT, {DIM_SIZE}, "t1");
  graph.setTileMapping(tIn, 0);

  auto seq = Sequence();
  const auto tOut = popops::pad(graph, tIn, pLows, pUpps, constant);

  graph.createHostWrite("in", tIn);
  graph.createHostRead("out", tOut);

  Engine eng(graph, seq);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", in);
    eng.run();
    eng.readTensor("out", out);
  });
}

BOOST_AUTO_TEST_CASE(PaddingWithTensor) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float c = 5.0f;
  float out[DIM_SIZE + 1];
  padWithTensor(in, &c, 1, &out[0], {0}, {1});
  const float expect_out[DIM_SIZE + 1] = {1.0f, 2.0f, 3.0f, 4.0f, c};
  for (unsigned i = 0; i < DIM_SIZE + 1; ++i) {
    BOOST_CHECK_EQUAL(out[i], expect_out[i]);
  }
}

BOOST_AUTO_TEST_CASE(PaddingWithNonScalarTensor) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float c[2] = {5.0f, 6.0f};
  float out[DIM_SIZE + 1];
  BOOST_CHECK_THROW(padWithTensor(in, &c[0], 2, &out[0], {0}, {1}),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(PaddingWithConstant) {
  const float in[DIM_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float c = 5.0f;
  float out[DIM_SIZE + 1];
  padWithConstant(in, c, &out[0], {0}, {1});
  const float expect_out[DIM_SIZE + 1] = {1.0f, 2.0f, 3.0f, 4.0f, c};
  for (unsigned i = 0; i < DIM_SIZE + 1; ++i) {
    BOOST_CHECK_EQUAL(out[i], expect_out[i]);
  }
}
