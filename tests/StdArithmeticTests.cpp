#define BOOST_TEST_MODULE StdArithmeticTests

#include <boost/test/unit_test.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <popops/ScaledAdd.hpp>
#include <popops/Cast.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define DIM_SIZE  10



static std::tuple<Tensor, Tensor> mapBinaryOpTensors(Graph &graph,
                                                     const Type &type) {
  auto in1 = graph.addVariable(type, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in1);

  auto in2 = graph.addVariable(type, {DIM_SIZE, DIM_SIZE}, "in2");
  mapTensorLinearly(graph, in2);

  return std::make_pair(in1.dimShuffle({1, 0}), in2.dimShuffle({1, 0}));
}


static void setBinaryOpInputs(float hIn1[DIM_SIZE][DIM_SIZE],
                              float hIn2[DIM_SIZE][DIM_SIZE]) {
  float val1 = -100;
  float val2 = 50;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      float sign1 = (1.0 - 2.0 * ((c + 1) & 1));
      float sign2 = (1.0 - 2.0 * ((r + c) & 1));
      hIn1[r][c] = (val1 + (r * DIM_SIZE + c) * .1) * sign1;
      hIn2[r][c] = (val2 + (r * DIM_SIZE + c) * .1) * sign2;
    }
  }
}

static void setBinaryOpInputs(int hIn1[DIM_SIZE][DIM_SIZE],
                              int hIn2[DIM_SIZE][DIM_SIZE]) {
  int val1 = -100;
  int val2 = 50;
  for (auto r = 0; r != DIM_SIZE; ++r) {
    for (auto c = 0; c != DIM_SIZE; ++c) {
      int sign1 = (1 - 2 * ((c + 1) & 1));
      int sign2 = (1 - 2 * ((r + c) & 1));
      hIn1[r][c] = (val1 + (r * DIM_SIZE + c) * 1) * sign1;
      hIn2[r][c] = (val2 + (r * DIM_SIZE + c) * 1) * sign2;
    }
  }
}

BOOST_AUTO_TEST_CASE(StdAddTo_float,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  float k = 2;
  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  auto prog = Sequence();
  scaledAddTo(graph, in1, in2, k, prog);
  Engine eng(device, graph, prog);

  float hOut[DIM_SIZE][DIM_SIZE];

  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] + k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdSubFrom_int,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  int k = 2;
  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, INT);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  auto prog = Sequence();
  scaledSubtractFrom(graph, in1, in2, k, prog);
  Engine eng(device, graph, prog);

  int hOut[DIM_SIZE][DIM_SIZE];

  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] - k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdCast) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  float hIn[DIM_SIZE];
  for (auto i = 0U; i<DIM_SIZE; ++i) {
    hIn[i] = (float)i;
  }

  auto in = graph.addVariable(FLOAT, {DIM_SIZE}, "in");
  mapTensorLinearly(graph, in);
  graph.createHostWrite("in", in);

  auto prog = Sequence();

  poplar::Tensor out = cast(graph, in, INT, prog, "cast");
  graph.createHostRead("out", out);

  int hOut[DIM_SIZE];

  Engine eng(device, graph, prog);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    BOOST_TEST(hOut[i] == i);
  }
}
