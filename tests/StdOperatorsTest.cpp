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
#include <poplar/IPUModel.hpp>
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
                               const Type &type) {
  auto in = graph.addTensor(type, {DIM_SIZE, DIM_SIZE}, "in0");
  mapTensorLinearly(graph, in);

  return in.dimShuffle({1, 0});
}

static std::pair<Tensor, Tensor> mapBinaryOpTensors(Graph &graph,
                                                    const Type &type) {
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

static void setUnaryOpInput(bool hIn[DIM_SIZE][DIM_SIZE]) {
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

template<typename T>
void convertToPositive(T array[DIM_SIZE][DIM_SIZE]) {
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      array[i][j] = std::abs(array[i][j]);
    }
  }
}

template<>
void convertToPositive(bool array[DIM_SIZE][DIM_SIZE]) {}

template<>
void convertToPositive(float array[DIM_SIZE][DIM_SIZE]) {
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      array[i][j] = std::fabs(array[i][j]);
    }
  }
}

template <typename T, typename TestT>
void unaryOpTest(const std::function<Tensor(Graph &, Tensor, Sequence &,
                                            const std::string &)> &op,
                 const std::function<TestT(T)> &testFn,
                 bool positiveInputs = false) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  auto in = mapUnaryOpTensor(graph, equivalent_device_type<T>().value);
  auto prog = Sequence();

  auto out = op(graph, in, prog, "unaryOp");
  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  Engine eng(device, graph, prog);

  T hIn[DIM_SIZE][DIM_SIZE];
  T hOut[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);
  if (positiveInputs) {
    convertToPositive(hIn);
  }
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto res = testFn(hIn[i][j]);
      BOOST_TEST(res == hOut[i][j]);
    }
  }
}

template <typename T, typename TestT, typename OutT = T>
void binaryOpTest(const std::function<Tensor(Graph &, Tensor, Tensor,
                                             Sequence &,
                                             const std::string &)> &op,
                 const std::function<TestT(T, T)> &testFn) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph,
                                          equivalent_device_type<T>().value);

  auto prog = Sequence();

  auto out = op(graph, in1, in2, prog, "binaryOp");
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", out);

  Engine eng(device, graph, prog);
  T hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  OutT hOut[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto res = testFn(hIn1[i][j], hIn2[i][j]);
      BOOST_TEST(static_cast<TestT>(hOut[i][j]) == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(StdOperationAbsFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::abs,
                             [](float x) -> double {
                                double res = fabs(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationAbsInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<int, int>(popstd::abs, [](int x) -> int { return std::abs(x);});
}

BOOST_AUTO_TEST_CASE(StdOperationAddFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popstd::add,
                              [](float x, float y) -> double {
                                 double res = x + y;
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationAddInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popstd::add, [](int x, int y) -> int {return x + y;});
}

BOOST_AUTO_TEST_CASE(StdOperationBitwiseAndInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                     ) {
  binaryOpTest<int, int>(popstd::bitwiseAnd,
                         [](int x, int y) -> int {return x & y;});
}

BOOST_AUTO_TEST_CASE(StdOperationBitwiseOrInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                     ) {
  binaryOpTest<int, int>(popstd::bitwiseOr,
                         [](int x, int y) -> int {return x | y;});
}

BOOST_AUTO_TEST_CASE(StdOperationBitwiseNotInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                     ) {
  unaryOpTest<int, int>(popstd::bitwiseNot,
                        [](int x) -> int { return ~x;});
}

BOOST_AUTO_TEST_CASE(StdOperationCeil,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::ceil,
                             [](float x) -> double {
                                double res = std::ceil(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationCos,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
) {
  unaryOpTest<float, double>(popstd::cos,
                             [](float x) -> double {
                                double res = std::cos(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationDivideFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>([](Graph &g, Tensor x, Tensor y,
                                 Sequence &prog, const std::string &d) {
                                return popstd::div(g, x, y, prog, d);
                              },
                              [](float x, float y) -> double {
                                 double res = x / y;
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationDivideInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>([](Graph &g, Tensor x, Tensor y,
                            Sequence &prog, const std::string &d) {
                           return popstd::div(g, x, y, prog, d);
                         },
                         [](int x, int y) -> int {
                            int res = x / y;
                           return res;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationEqualFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popstd::eq,
    [](float x, float y) -> bool {
       return x == y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationEqualBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::eq,
    [](bool x, bool y) -> bool {
       return x == y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::gt,
    [](bool x, bool y) -> bool {
       return x > y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanEqualBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::gteq,
    [](bool x, bool y) -> bool {
       return x >= y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThanBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::lt,
    [](bool x, bool y) -> bool {
       return x < y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThanEqualBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::lteq,
    [](bool x, bool y) -> bool {
       return x <= y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationExponent,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, float>(popstd::exp,
                           [](float x) -> float {
                              double res = std::exp(static_cast<double>(x));
                              return res;
                           });
}

BOOST_AUTO_TEST_CASE(StdOperationFloor,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::floor,
                             [](float x) -> double {
                                double res = std::floor(static_cast<double>(x));
                                return res;
                             });
}


BOOST_AUTO_TEST_CASE(StdOperationGreaterThanFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popstd::gt,
    [](float x, float y) -> bool {
       return x > y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, bool, bool>(
    popstd::gt,
    [](int x, int y) -> bool {
       return x > y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanEqual,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popstd::gteq,
    [](float x, float y) -> bool {
       return x >= y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThan,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popstd::lt,
    [](float x, float y) -> bool {
       return x < y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThanEqual,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popstd::lteq,
    [](float x, float y) -> bool {
       return x <= y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationLogarithm,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::log,
                             [](float x) -> double {
                                double res = std::log(static_cast<double>(x));
                                return res;
                             },
                             true /* positive inputs */);
}

BOOST_AUTO_TEST_CASE(StdOperationLogicalAnd,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::logicalAnd,
    [](bool x, bool y) -> bool {
       return x && y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationLogicalNot,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<bool, bool>(popstd::logicalNot,
                         [](bool x) -> bool { return !x; });
}

BOOST_AUTO_TEST_CASE(StdOperationLogicalOr,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::logicalOr,
    [](bool x, bool y) -> bool {
       return x || y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationMaximumFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popstd::max,
                              [](float x, float y) -> double {
                                 double res = std::max(x, y);
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationMaximumInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popstd::max,
                         [](int x, int y) -> int {
                            auto res = std::max(x, y);
                            return res;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationMinimumFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popstd::min,
                              [](float x, float y) -> double {
                                 double res = std::min(x, y);
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationMinimumInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popstd::min,
                         [](int x, int y) -> int {
                            auto res = std::min(x, y);
                            return res;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationMultiply,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popstd::mul,
                              [](float x, float y) -> double {
                                 double res = x * y;
                                 return res;
                              });
}


BOOST_AUTO_TEST_CASE(StdOperationNotEqualFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popstd::neq,
    [](float x, float y) -> bool {
       return x != y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationNotEqualBool,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popstd::neq,
    [](bool x, bool y) -> bool {
       return x != y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationNegateFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::neg,
                             [](float x) -> double {
                                return -x;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationNegateInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<int, int>(popstd::neg,
                             [](int x) -> int {
                                return -x;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationPower,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
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
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);

  auto prog = Sequence();
  auto out = pow(graph, in1, in2, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", out);

  Engine eng(device, graph, prog);
  float hOut[DIM_SIZE][DIM_SIZE];
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);

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
  binaryOpTest<float, double>(popstd::rem,
                              [](float x, float y) -> double {
                                double res = std::fmod(static_cast<double>(x),
                                                       static_cast<double>(y));
                                return res;
                              });
}



BOOST_AUTO_TEST_CASE(StdOperationRemainderInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popstd::rem,
                         [](int x, int y) -> double {
                            return x % y;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationShiftLeftInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int, int>(popstd::shiftLeft,
                              [](int x, int y) -> int {
                                 return x << y;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationShiftRightInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int, int>(popstd::shiftRight,
                              [](int x, int y) -> int {
                                return (unsigned)x >> y;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationShiftRightSignExtendInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int, int>(popstd::shiftRightSignExtend,
                              [](int x, int y) -> int {
                                return x >> y;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationSignumFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::signum,
                             [](float x) -> double {
                                double res = (0 < x) - (x < 0);
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationSignumInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<int, int>(popstd::signum,
                             [](int x) -> int {
                                int res = (0 < x) - (x < 0);
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationSin,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::sin,
                             [](float x) -> double {
                                double res = std::sin(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationTanh,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::tanh,
                             [](float x) -> double {
                                double res = std::tanh(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationSquare,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::square,
                             [](float x) -> double {
                                double xd = static_cast<double>(x);
                                double res = xd * xd;
                                return res;
                             });
}


BOOST_AUTO_TEST_CASE(StdOperationSqrt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::sqrt,
                             [](float x) -> double {
                                double xd = static_cast<double>(x);
                                double res = std::sqrt(xd);
                                return res;
                             },
                             true /* positive inputs */);
}


BOOST_AUTO_TEST_CASE(StdOperationSubtractFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(
    popstd::sub,
    [](float x, float y) -> double {
       return static_cast<double>(x) - static_cast<double>(y);
    });
}

BOOST_AUTO_TEST_CASE(StdOperationSubtractHalf,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<half, double>(
    popstd::sub,
    [](half x, half y) -> double {
      return static_cast<double>(x) - static_cast<double>(y);
    });
}

BOOST_AUTO_TEST_CASE(StdOperationSubtractInt,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(
    popstd::sub,
    [](int x, int y) -> int {
      return (x - y);
    });
}

BOOST_AUTO_TEST_CASE(StdOperationRoundFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popstd::round,
    [](float x) -> double {
      return std::round(x);
    });
}


BOOST_AUTO_TEST_CASE(StdOperationSelectFloat,
                  *utf::tolerance<half>(fpc::percent_tolerance<half>(0.1))
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
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
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);
  Tensor in3 = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = select(graph, in1, in2, in3, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(device, graph, prog);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.run();
  eng.readTensor("out", hOut);

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
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
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
  std::tie(in1, in2) = mapBinaryOpTensors(graph, INT);
  Tensor in3 = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = select(graph, in1, in2, in3, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);
  graph.createHostRead("out", out);

  int hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(device, graph, prog);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.run();
  eng.readTensor("out", hOut);
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
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
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
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);
  Tensor in3 = mapUnaryOpTensor(graph, FLOAT);

  auto prog = Sequence();
  auto out = clamp(graph, in1, in2, in3, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(device, graph, prog);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.run();
  eng.readTensor("out", hOut);

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
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
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
  std::tie(in1, in2) = mapBinaryOpTensors(graph, INT);
  Tensor in3 = mapUnaryOpTensor(graph, INT);

  auto prog = Sequence();
  auto out = clamp(graph, in1, in2, in3, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);
  graph.createHostRead("out", out);

  int hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(device, graph, prog);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.run();
  eng.readTensor("out", hOut);

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
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  Tensor in1, in2;
  in1 = graph.addTensor(FLOAT, {2, 2}, "t1");
  in2 = graph.addTensor(FLOAT, {2, 2}, "t2");

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
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  Tensor in1, in2, in3, in4;
  in1 = graph.addTensor(FLOAT, {2, 2}, "t1");
  in2 = graph.addTensor(FLOAT, {2, 2}, "t2");
  in3 = graph.addTensor(BOOL, {2, 2}, "pred1");
  in4 = graph.addTensor(BOOL, {2, 2}, "pred2");

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
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  Tensor in = graph.addTensor(FLOAT, {2, 2}, "t1");
  auto prog = Sequence();
  BOOST_CHECK_THROW(allTrue(graph, in, prog, "all_true"),
                    poplib_error);
}

BOOST_AUTO_TEST_CASE(StdOperationAllTrue) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  Tensor in = graph.addTensor(INT, {2}, "t1");
  Tensor ones = graph.addConstantTensor(INT, {2}, 1);
  Tensor zeros = graph.addConstantTensor(INT, {2}, 0);

  graph.setTileMapping(in, 0);

  auto bodyProg = Sequence();
  subtractFrom(graph, in, ones, bodyProg);

  auto condProg = Sequence();
  Tensor neZero = neq(graph, in, zeros, condProg);
  allTrue(graph, neZero, condProg, "all_true");

  int init[2] = {10, 8};
  int output[2] = {0, 0};

  auto mainProg = Sequence();
  mainProg.add(RepeatWhileTrue(condProg, bodyProg));
  graph.createHostWrite("in", in);
  graph.createHostRead("out", in);

  Engine eng(device, graph, mainProg);
  eng.writeTensor("in", init);
  eng.run();
  eng.readTensor("out", output);

  BOOST_CHECK_EQUAL(output[0], 2);
  BOOST_CHECK_EQUAL(output[1], 0);
}

BOOST_AUTO_TEST_CASE(StdOperationIsFinite) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popstd::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);
  hIn[0][0] = INFINITY;
  hIn[1][0] = -INFINITY;
  hIn[0][1] = NAN;
  hIn[1][1] = NAN;
  hIn[0][2] = 0.0f;

  auto in = mapUnaryOpTensor(graph, FLOAT);
  auto prog = Sequence();
  auto out = isFinite(graph, in, prog);
  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  bool hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(device, graph, prog);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool expected = !(i<=1 && j<=1);
      BOOST_TEST(hOut[i][j] == expected);
    }
  }
}
