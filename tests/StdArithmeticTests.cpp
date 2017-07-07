#define BOOST_TEST_MODULE StdArithmeticTests

#include <boost/test/unit_test.hpp>
#include <popstd/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <iostream>
#include <popstd/Add.hpp>
#include <popstd/SubtractFrom.hpp>
#include <popstd/Cast.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define DIM_SIZE  10



static std::tuple<Tensor, Tensor> mapBinaryOpTensors(Graph &graph,
                                                     const std::string &type) {
  auto in1 = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in1);

  auto in2 = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in2");
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
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "float");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));


  addTo(graph, in1, in2, 1.0, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(in1, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] + hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdAddTo_int) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  addTo(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(in1, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] + hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdSubtractFrom_float,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "float");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  subtractFrom(graph, in1, in2, 1.0, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(in1, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] - hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdSubtractFrom_int) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  subtractFrom(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(in1, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] - hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdCast) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE];
  for (auto i = 0U; i<DIM_SIZE; ++i) {
    hIn[i] = (float)i;
  }

  auto in = graph.addTensor("float", {DIM_SIZE}, "in");
  mapTensorLinearly(graph, in);

  auto prog = Sequence();

  prog.add(Copy(hIn, in));

  poplar::Tensor out = cast(graph, in, "int", prog, "cast");

  int hOut[DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    BOOST_TEST(hOut[i] == i);
  }
}
