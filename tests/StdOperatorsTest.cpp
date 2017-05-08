#define BOOST_TEST_MODULE StdOperatorTest
#include <popstd/Operations.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <popstd/ActivationMapping.hpp>
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
  auto in = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in1");
  mapActivations(graph, in);

  return in.dimShuffle({1, 0});
}

static std::tuple<Tensor, Tensor> mapBinaryOpTensors(Graph &graph,
                                                     const std::string &type) {
  auto in1 = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in1");
  mapActivations(graph, in1);

  auto in2 = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in2");
  mapActivations(graph, in2);

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


/* Generates two 2D matrix of size DIM_SIZE x DIM_SIZE containing linearly
 * increasing absolute values with a fixed slope and sign alternating sign
 * for each element in a row. The start value at position [0][0] for each
 * of the matrices is different.
 */
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

BOOST_AUTO_TEST_CASE(StdOperationAbs,
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

BOOST_AUTO_TEST_CASE(StdOperationAdd,
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

BOOST_AUTO_TEST_CASE(StdOperationDivide,
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

BOOST_AUTO_TEST_CASE(StdOperationEqual,
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


BOOST_AUTO_TEST_CASE(StdOperationGreaterThan,
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
      hIn[r][c] = abs(hIn[r][c]);
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



BOOST_AUTO_TEST_CASE(StdOperationMaximum,
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


BOOST_AUTO_TEST_CASE(StdOperationMinimum,
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


BOOST_AUTO_TEST_CASE(StdOperationNotEqual,
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


BOOST_AUTO_TEST_CASE(StdOperationNegate,
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



BOOST_AUTO_TEST_CASE(StdOperationRemainder,
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


BOOST_AUTO_TEST_CASE(StdOperationSubtract,
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
