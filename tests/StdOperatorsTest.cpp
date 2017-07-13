#define BOOST_TEST_MODULE StdOperatorTest
#include <popstd/AllTrue.hpp>
#include <popstd/exceptions.hpp>
#include <popstd/Operations.hpp>
#include <popstd/SubtractFrom.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <popstd/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <iostream>
#include <cmath>


using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define DIM_SIZE  10

static Tensor mapUnaryOpTensor(Graph &graph,
                               const std::string type) {
  auto in = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in0");
  mapTensorLinearly(graph, in);

  return in.dimShuffle({1, 0});
}

static std::pair<Tensor, Tensor> mapBinaryOpTensors(Graph &graph,
                                                     const std::string &type) {
  auto in1 = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in1);

  auto in2 = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in2");
  mapTensorLinearly(graph, in2);

  return std::make_pair(in1.dimShuffle({1, 0}), in2.dimShuffle({1, 0}));
}

/* Generates a 2D matrix of size DIM_SIZE x DIM_SIZE containing linearly
 * increasing absolute values with a fixed slope and sign alternating sign
 * for each element in a row.
 */
static void setUnaryOpInput(float hIn[DIM_SIZE][DIM_SIZE]) {
  float val = -100;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      float sign = 1.0 - 2.0 * ((r + c) & 1);
      hIn[r][c] = (val + (r * DIM_SIZE + c) * .1) * sign;
    }
  }
}

static void setUnaryOpInput(int hIn[DIM_SIZE][DIM_SIZE]) {
  int val = -100;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      int sign = 1 - 2 * ((r + c) & 1);
      hIn[r][c] = (val + (r * DIM_SIZE + c)) * sign;
    }
  }
}

/* Generates two 2D matrix of size DIM_SIZE x DIM_SIZE containing linearly
 * increasing absolute values with a fixed slope and sign alternating sign
 * for each element in a row. The start value at position [0][0] for each
 * of the matrices is different.
 */
static void setBinaryOpInputs(float hIn1[DIM_SIZE][DIM_SIZE],
                              float hIn2[DIM_SIZE][DIM_SIZE]) {
  float val1 = 1000;
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


static void setBinaryOpInputs(half hIn1[DIM_SIZE][DIM_SIZE],
                              half hIn2[DIM_SIZE][DIM_SIZE]) {
  float val1 = -100;
  float val2 = 50;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      float sign1 = (1.0 - 2.0 * ((c + 1) & 1));
      float sign2 = (1.0 - 2.0 * ((r + c) & 1));
      hIn1[r][c] = (val1 + (r * DIM_SIZE + c) *.1) * sign1;
      hIn2[r][c] = (val2 + (r * DIM_SIZE + c) *.1) * sign2;
    }
  }
}


/* Generates two 2D matrix of size DIM_SIZE x DIM_SIZE containing
 * boolean values. All combinations of boolean values are produced
 */
static void setBinaryOpInputs(bool hIn1[DIM_SIZE][DIM_SIZE],
                              bool hIn2[DIM_SIZE][DIM_SIZE]) {
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn1[r][c] = r & 1;
      hIn2[r][c] = (r + c) & 1;
    }
  }
}

static void setBinaryOpInputs(int hIn1[DIM_SIZE][DIM_SIZE],
                              int hIn2[DIM_SIZE][DIM_SIZE]) {
  int val1 = -100;
  int val2 = 59;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn1[r][c] = (1 - 2 * (r & 1)) * (r + val1);
      hIn2[r][c] = (1 - 2 * ((r + c) & 1)) * (r + c + val2);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationAbsFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = abs(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = fabs(static_cast<double>(hIn[i][j]));
      BOOST_TEST(res == hOut[i][j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationAbsInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "int");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = abs(graph, in, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = std::abs(hIn[i][j]);
      BOOST_TEST(res == hOut[i][j]);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationAddFloat,
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

  auto out = add(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

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

BOOST_AUTO_TEST_CASE(StdOperationAddInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = add(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn1[i][j] + hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationCeil,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = ceil(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::ceil(static_cast<double>(hIn[i][j]));
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationCos,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = cos(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::cos(static_cast<double>(hIn[i][j]));
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}



BOOST_AUTO_TEST_CASE(StdOperationDivideFloat,
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

  auto out = div(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] / hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationDivideInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = div(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn1[i][j] / hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationEqualFloat,
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

  auto out = eq(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] == hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationEqualBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  bool hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "bool");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = eq(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] == hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationExponent,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = exp(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::exp(static_cast<double>(hIn[i][j]));
      BOOST_TEST(hOut[i][j] == (float) res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationFloor,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = floor(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::floor(static_cast<double>(hIn[i][j]));
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationGreaterThanFloat,
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

  auto out = gt(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] > hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = gt(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] > hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanEqual,
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

  auto out = gteq(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] >= hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationLessThan,
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

  auto out = lt(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] < hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationLessThanEqual,
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

  auto out = lteq(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] <= hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationLogarithm,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn[r][c] = std::abs(hIn[r][c]);
    }
  }

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = log(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::log(static_cast<double>(hIn[i][j]));
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationLogicalAnd,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  bool hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];

  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      hIn1[i][j] = (i + 1) & 1;
      hIn2[i][j] = (i + j) & 1;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "bool");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = logicalAnd(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] && hIn2[i][j] ;
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationLogicalNot,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  bool hIn[DIM_SIZE][DIM_SIZE];

  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      hIn[i][j] = (i + 1) & 1;
    }
  }

  auto in = mapUnaryOpTensor(graph, "bool");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = logicalNot(graph, in, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = !hIn[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationLogicalOr,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  bool hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];

  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      hIn1[i][j] = (i + 1) & 1;
      hIn2[i][j] = (i + j) & 1;
    }
  }
  hIn1[0][0] = false;
  hIn2[0][0] = false;

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "bool");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = logicalOr(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] || hIn2[i][j] ;
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}



BOOST_AUTO_TEST_CASE(StdOperationMaximumFloat,
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

  auto out = max(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::max(hIn1[i][j], hIn2[i][j]);
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationMaximumInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = max(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = std::max(hIn1[i][j], hIn2[i][j]);
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationMinimumFloat,
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

  auto out = min(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::min(hIn1[i][j], hIn2[i][j]);
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationMinimumInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = min(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::min(hIn1[i][j], hIn2[i][j]);
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationMultiply,
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

  auto out = mul(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] * hIn2[i][j];
      BOOST_TEST((double)hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationNotEqualFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  hIn1[0][0] = 0;
  hIn2[0][0] = 0;

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "float");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = neq(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] != hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationNotEqualBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  bool hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  hIn1[0][0] = false;
  hIn2[0][0] = false;

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "bool");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = neq(graph, in1, in2, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool res = hIn1[i][j] != hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationNegateFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = neg(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = -hIn[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationNegateInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "int");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = neg(graph, in, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = -hIn[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationPower,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];

  float x = 10.23;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c, x += .1f) {
      hIn1[r][c] = x;
      hIn2[r][c] = 2;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "float");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = pow(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::pow(static_cast<double>(hIn1[i][j]),
                            static_cast<double>(hIn2[i][j]));
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}



BOOST_AUTO_TEST_CASE(StdOperationRemainderFloat,
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

  auto out = rem(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::fmod(static_cast<double>(hIn1[i][j]),
                             static_cast<double>(hIn2[i][j]));
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}



BOOST_AUTO_TEST_CASE(StdOperationRemainderInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = rem(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn1[i][j] % hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationSignum,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);
  hIn[0][0] = 0;

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = signum(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = (0 < hIn[i][j]) - (hIn[i][j] < 0);
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationTanh,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = tanh(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = tanh(static_cast<double>(hIn[i][j]));
      BOOST_TEST((float)hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationSquare,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = square(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double x = static_cast<double>(hIn[i][j]);
      double res = x * x;
      BOOST_TEST((float)hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationSqrt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn[r][c] = std::abs(hIn[r][c]);
    }
  }

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = sqrt(graph, in, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double x = static_cast<double>(hIn[i][j]);
      double res = std::sqrt(x);
      BOOST_TEST((float)hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationSubtractFloat,
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

  auto out = sub(graph, in1, in2, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

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

BOOST_AUTO_TEST_CASE(StdOperationSubtractHalf,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  half hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "half");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = sub(graph, in1, in2, prog);

  half hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = static_cast<double>(hIn1[i][j])
                   - static_cast<double>(hIn2[i][j]);
      BOOST_TEST((double)hOut[i][j] == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationSubtractInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));

  auto out = sub(graph, in1, in2, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn1[i][j] - hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationSelectFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);
  bool hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn3[r][c] = (c) & 0x1;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "float");
  Tensor in3 = mapUnaryOpTensor(graph, "bool");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));
  prog.add(Copy(hIn3, in3));

  auto out = select(graph, in1, in2, in3, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hIn3[i][j] ? hIn1[i][j] : hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationSelectInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);
  bool hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn3[r][c] = (c) & 0x1;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");
  Tensor in3 = mapUnaryOpTensor(graph, "bool");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));
  prog.add(Copy(hIn3, in3));

  auto out = select(graph, in1, in2, in3, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn3[i][j] ? hIn1[i][j] : hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationClampFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn1);
  float hIn2[DIM_SIZE][DIM_SIZE], hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn2[r][c] = -0.5;
      hIn3[r][c] = 0.5;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "float");
  Tensor in3 = mapUnaryOpTensor(graph, "float");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));
  prog.add(Copy(hIn3, in3));

  auto out = clamp(graph, in1, in2, in3, prog);

  float hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hIn1[i][j];
      if (res < hIn2[i][j])
        res = hIn2[i][j];
      if (res > hIn3[i][j])
        res = hIn3[i][j];

      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationClampInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  int hIn1[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn1);
  int hIn2[DIM_SIZE][DIM_SIZE], hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn2[r][c] = -10;
      hIn3[r][c] = 10;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, "int");
  Tensor in3 = mapUnaryOpTensor(graph, "int");

  auto prog = Sequence();

  prog.add(Copy(hIn1, in1));
  prog.add(Copy(hIn2, in2));
  prog.add(Copy(hIn3, in3));

  auto out = clamp(graph, in1, in2, in3, prog);

  int hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn1[i][j];
      if (res < hIn2[i][j])
        res = hIn2[i][j];
      if (res > hIn3[i][j])
        res = hIn3[i][j];

      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationBinaryOutputMapChoice) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  Tensor in1, in2;
  in1 = graph.addTensor("float", {2, 2}, "t1");
  in2 = graph.addTensor("float", {2, 2}, "t2");

  // Tensor in1 all on tile 0
  graph.setTileMapping(in1, 0);

  // Tensor in2 spread out
  graph.setTileMapping(in2.index({0, 0}), 0);
  graph.setTileMapping(in2.index({0, 1}), 1);
  graph.setTileMapping(in2.index({1, 0}), 2);
  graph.setTileMapping(in2.index({1, 1}), 3);

  auto prog = Sequence();
  auto out1 = add(graph, in1, in2, prog);
  auto out2 = add(graph, in2, in1, prog);

  const auto &tile1 = graph.getTileMapping(out1);
  BOOST_TEST(tile1[0].size() > 0);
  BOOST_TEST(tile1[1].size() > 0);
  BOOST_TEST(tile1[2].size() > 0);
  BOOST_TEST(tile1[3].size() > 0);

  const auto &tile2 = graph.getTileMapping(out2);
  BOOST_TEST(tile2[0].size() > 0);
  BOOST_TEST(tile2[1].size() > 0);
  BOOST_TEST(tile2[2].size() > 0);
  BOOST_TEST(tile2[3].size() > 0);
}

BOOST_AUTO_TEST_CASE(StdOperationTrinaryOutputMapChoice) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  Tensor in1, in2, in3, in4;
  in1 = graph.addTensor("float", {2, 2}, "t1");
  in2 = graph.addTensor("float", {2, 2}, "t2");
  in3 = graph.addTensor("bool", {2, 2}, "pred1");
  in4 = graph.addTensor("bool", {2, 2}, "pred2");

  // Tensor in1 all on tile 0
  graph.setTileMapping(in1, 0);

  // Tensor in2 spread out
  graph.setTileMapping(in2.index({0, 0}), 0);
  graph.setTileMapping(in2.index({0, 1}), 1);
  graph.setTileMapping(in2.index({1, 0}), 2);
  graph.setTileMapping(in2.index({1, 1}), 3);

  // Tensor pred1 all on tile 1
  graph.setTileMapping(in3, 1);

  // Tensor pred2 spread out
  graph.setTileMapping(in4.index({0, 0}), 0);
  graph.setTileMapping(in4.index({0, 1}), 1);
  graph.setTileMapping(in4.index({1, 0}), 2);
  graph.setTileMapping(in4.index({1, 1}), 3);


  auto prog = Sequence();
  auto out1 = select(graph, in1, in2, in3, prog);
  auto out2 = select(graph, in2, in1, in3, prog);
  auto out3 = select(graph, in1, in1, in4, prog);

  const auto &tile1 = graph.getTileMapping(out1);
  BOOST_TEST(tile1[0].size() > 0);
  BOOST_TEST(tile1[1].size() > 0);
  BOOST_TEST(tile1[2].size() > 0);
  BOOST_TEST(tile1[3].size() > 0);

  const auto &tile2 = graph.getTileMapping(out2);
  BOOST_TEST(tile2[0].size() > 0);
  BOOST_TEST(tile2[1].size() > 0);
  BOOST_TEST(tile2[2].size() > 0);
  BOOST_TEST(tile2[3].size() > 0);

  const auto &tile3 = graph.getTileMapping(out3);
  BOOST_TEST(tile3[0].size() > 0);
  BOOST_TEST(tile3[1].size() > 0);
  BOOST_TEST(tile3[2].size() > 0);
  BOOST_TEST(tile3[3].size() > 0);
}

BOOST_AUTO_TEST_CASE(StdOperationAllTrueBadType) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  Tensor in = graph.addTensor("float", {2, 2}, "t1");
  auto prog = Sequence();
  BOOST_CHECK_THROW(allTrue(graph, in, prog, "all_true"),
                    poplib_error);
}

BOOST_AUTO_TEST_CASE(StdOperationAllTrue) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  Tensor in = graph.addTensor("int", {2}, "t1");
  Tensor ones = graph.addConstantTensor("int", {2}, 1);
  Tensor zeros = graph.addConstantTensor("int", {2}, 0);

  graph.setTileMapping(in, 0);

  auto bodyProg = Sequence();
  subtractFrom(graph, in, ones, bodyProg);

  auto condProg = Sequence();
  Tensor neZero = neq(graph, in, zeros, condProg);
  allTrue(graph, neZero, condProg, "all_true");

  int init[2] = {10, 8};
  int output[2] = {0, 0};

  auto mainProg = Sequence();
  mainProg.add(Copy(init, in));
  mainProg.add(RepeatWhileTrue(condProg, bodyProg));
  mainProg.add(Copy(in, output));

  Engine eng(graph, mainProg);
  eng.run();

  BOOST_CHECK_EQUAL(output[0], 2);
  BOOST_CHECK_EQUAL(output[1], 0);
}

BOOST_AUTO_TEST_CASE(StdOperationIsFinite) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);
  hIn[0][0] = INFINITY;
  hIn[1][0] = -INFINITY;
  hIn[0][1] = NAN;
  hIn[1][1] = NAN;
  hIn[0][2] = 0.0f;

  auto in = mapUnaryOpTensor(graph, "float");
  auto prog = Sequence();

  prog.add(Copy(hIn, in));
  auto out = isFinite(graph, in, prog);

  bool hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool expected = !(i<=1 && j<=1);
      BOOST_TEST(hOut[i][j] == expected);
    }
  }
}
