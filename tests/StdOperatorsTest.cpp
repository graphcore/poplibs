#define BOOST_TEST_MODULE StdOperatorTest
#include <popops/AllTrue.hpp>
#include <poputil/exceptions.hpp>
#include <popops/ElementWise.hpp>
#include <popops/SubtractFrom.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <cmath>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define DIM_SIZE  10

static Tensor mapUnaryOpTensor(Graph &graph,
                               const Type &type) {
  auto in = graph.addVariable(type, {DIM_SIZE, DIM_SIZE}, "in0");
  mapTensorLinearly(graph, in);

  return in.dimShuffle({1, 0});
}

static std::pair<Tensor, Tensor> mapBinaryOpTensors(Graph &graph,
                                                    const Type &type) {
  auto in1 = graph.addVariable(type, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in1);

  auto in2 = graph.addVariable(type, {DIM_SIZE, DIM_SIZE}, "in2");
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


static void setBinaryOpInputsHalf(float hIn1[DIM_SIZE][DIM_SIZE],
                                  float hIn2[DIM_SIZE][DIM_SIZE]) {
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

using UnaryOpFn = std::function<Tensor(Graph &, const Tensor &, Sequence &,
                                 const std::string &,
                                 const std::vector<std::string> &)>;

template <typename T, typename TestT>
void unaryOpTest(const UnaryOpFn &op,
                 const std::function<TestT(T)> &testFn,
                 bool positiveInputs = false) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  auto in = mapUnaryOpTensor(graph, equivalent_device_type<T>().value);
  auto prog = Sequence();

  auto out = op(graph, in, prog, "unaryOp", {});
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

using BinaryOpFn = std::function<Tensor(Graph &, const Tensor &,
                                 const Tensor &, Sequence &,
                                 const std::string &,
                                 const std::vector<std::string> &)>;

template <typename T, typename TestT, typename OutT = T>
void binaryOpTest(const BinaryOpFn &op,
                 const std::function<TestT(T, T)> &testFn) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  Tensor in1, in2;
  auto type = equivalent_device_type<T>().value;

  std::tie(in1, in2) = mapBinaryOpTensors(graph, type);

  auto prog = Sequence();

  auto out = op(graph, in1, in2, prog, "binaryOp", {});
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

void binaryOpTestHalf(const BinaryOpFn &op,
                      const std::function<float(float, float)> &testFn) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  const auto &target = device.getTarget();
  Graph graph(device);
  popops::addCodelets(graph);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, poplar::HALF);

  auto prog = Sequence();

  auto out = op(graph, in1, in2, prog, "binaryOp", {});
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", out);

  Engine eng(device, graph, prog);
  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputsHalf(hIn1, hIn2);
  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawIn1(rawBufSize), rawIn2(rawBufSize),
                    rawOut(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hIn1[0][0], rawIn1.data(),
                                DIM_SIZE * DIM_SIZE);
  poplar::copyFloatToDeviceHalf(target, &hIn2[0][0], rawIn2.data(),
                                DIM_SIZE * DIM_SIZE);
  eng.writeTensor("in1", rawIn1.data());
  eng.writeTensor("in2", rawIn2.data());
  eng.run();
  eng.readTensor("out", rawOut.data());
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                DIM_SIZE * DIM_SIZE);
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto res = testFn(hIn1[i][j], hIn2[i][j]);
      BOOST_TEST(static_cast<float>(hOut[i][j]) == res);
    }
  }
}


BOOST_AUTO_TEST_CASE(StdOperationAbsFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::abs,
                             [](float x) -> double {
                                double res = fabs(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationAbsInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<int, int>(popops::abs, [](int x) -> int { return std::abs(x);});
}

BOOST_AUTO_TEST_CASE(StdOperationAddFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::add,
                              [](float x, float y) -> double {
                                 double res = x + y;
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationAtan2Float,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::atan2,
                              [](float x, float y) -> double {
                                double res = std::atan2(x, y);
                                return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationAddInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popops::add, [](int x, int y) -> int {return x + y;});
}

BOOST_AUTO_TEST_CASE(StdOperationBitwiseAndInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                     ) {
  binaryOpTest<int, int>(popops::bitwiseAnd,
                         [](int x, int y) -> int {return x & y;});
}

BOOST_AUTO_TEST_CASE(StdOperationBitwiseOrInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                     ) {
  binaryOpTest<int, int>(popops::bitwiseOr,
                         [](int x, int y) -> int {return x | y;});
}

BOOST_AUTO_TEST_CASE(StdOperationBitwiseNotInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                     ) {
  unaryOpTest<int, int>(popops::bitwiseNot,
                        [](int x) -> int { return ~x;});
}

BOOST_AUTO_TEST_CASE(StdOperationCeil,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::ceil,
                             [](float x) -> double {
                                double res = std::ceil(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationCos,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
) {
  unaryOpTest<float, double>(popops::cos,
                             [](float x) -> double {
                                double res = std::cos(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationDivideFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::div,
                              [](float x, float y) -> double {
                                 double res = x / y;
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationDivideInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popops::div,
                         [](int x, int y) -> int {
                            int res = x / y;
                           return res;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationEqualFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popops::eq,
    [](float x, float y) -> bool {
       return x == y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationEqualBool,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::eq,
    [](bool x, bool y) -> bool {
       return x == y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanBool,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::gt,
    [](bool x, bool y) -> bool {
       return x > y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanEqualBool,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::gteq,
    [](bool x, bool y) -> bool {
       return x >= y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThanBool,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::lt,
    [](bool x, bool y) -> bool {
       return x < y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThanEqualBool,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::lteq,
    [](bool x, bool y) -> bool {
       return x <= y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationExponent,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, float>(popops::exp,
                           [](float x) -> float {
                              double res = std::exp(static_cast<double>(x));
                              return res;
                           });
}

BOOST_AUTO_TEST_CASE(StdOperationFloor,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::floor,
                             [](float x) -> double {
                                double res = std::floor(static_cast<double>(x));
                                return res;
                             });
}


BOOST_AUTO_TEST_CASE(StdOperationGreaterThanFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popops::gt,
    [](float x, float y) -> bool {
       return x > y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, bool, bool>(
    popops::gt,
    [](int x, int y) -> bool {
       return x > y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationGreaterThanEqual,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popops::gteq,
    [](float x, float y) -> bool {
       return x >= y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThan,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popops::lt,
    [](float x, float y) -> bool {
       return x < y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationLessThanEqual,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popops::lteq,
    [](float x, float y) -> bool {
       return x <= y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationLogarithm,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::log,
                             [](float x) -> double {
                                double res = std::log(static_cast<double>(x));
                                return res;
                             },
                             true /* positive inputs */);
}

BOOST_AUTO_TEST_CASE(StdOperationLogicalAnd,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::logicalAnd,
    [](bool x, bool y) -> bool {
       return x && y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationLogicalNot,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<bool, bool>(popops::logicalNot,
                         [](bool x) -> bool { return !x; });
}

BOOST_AUTO_TEST_CASE(StdOperationLogicalOr,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::logicalOr,
    [](bool x, bool y) -> bool {
       return x || y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationMaximumFloat,
                 *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::max,
                              [](float x, float y) -> double {
                                 double res = std::max(x, y);
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationMaximumInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popops::max,
                         [](int x, int y) -> int {
                            auto res = std::max(x, y);
                            return res;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationMinimumFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::min,
                              [](float x, float y) -> double {
                                 double res = std::min(x, y);
                                 return res;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationMinimumInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popops::min,
                         [](int x, int y) -> int {
                            auto res = std::min(x, y);
                            return res;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationMultiply,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::mul,
                              [](float x, float y) -> double {
                                 double res = x * y;
                                 return res;
                              });
}


BOOST_AUTO_TEST_CASE(StdOperationNotEqualFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, bool, bool>(
    popops::neq,
    [](float x, float y) -> bool {
       return x != y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationNotEqualBool,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<bool, bool, bool>(
    popops::neq,
    [](bool x, bool y) -> bool {
       return x != y;
    });
}


BOOST_AUTO_TEST_CASE(StdOperationNegateFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::neg,
                             [](float x) -> double {
                                return -x;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationNegateInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<int, int>(popops::neg,
                             [](int x) -> int {
                                return -x;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationPower,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

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
  auto out = popops::pow(graph, in1, in2, prog);
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
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(popops::rem,
                              [](float x, float y) -> double {
                                double res = std::fmod(static_cast<double>(x),
                                                       static_cast<double>(y));
                                return res;
                              });
}



BOOST_AUTO_TEST_CASE(StdOperationRemainderInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(popops::rem,
                         [](int x, int y) -> double {
                            return x % y;
                         });
}

BOOST_AUTO_TEST_CASE(StdOperationShiftLeftInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int, int>(popops::shiftLeft,
                              [](int x, int y) -> int {
                                 return x << y;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationShiftRightInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int, int>(popops::shiftRight,
                              [](int x, int y) -> int {
                                return (unsigned)x >> y;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationShiftRightSignExtendInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int, int>(popops::shiftRightSignExtend,
                              [](int x, int y) -> int {
                                return x >> y;
                              });
}

BOOST_AUTO_TEST_CASE(StdOperationSignumFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::signum,
                             [](float x) -> double {
                                double res = (0 < x) - (x < 0);
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationSignumInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<int, int>(popops::signum,
                             [](int x) -> int {
                                int res = (0 < x) - (x < 0);
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationSin,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::sin,
                             [](float x) -> double {
                                double res = std::sin(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationTanh,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::tanh,
                             [](float x) -> double {
                                double res = std::tanh(static_cast<double>(x));
                                return res;
                             });
}

BOOST_AUTO_TEST_CASE(StdOperationSquare,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::square,
                             [](float x) -> double {
                                double xd = static_cast<double>(x);
                                double res = xd * xd;
                                return res;
                             });
}


BOOST_AUTO_TEST_CASE(StdOperationSqrt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::sqrt,
                             [](float x) -> double {
                                double xd = static_cast<double>(x);
                                double res = std::sqrt(xd);
                                return res;
                             },
                             true /* positive inputs */);
}


BOOST_AUTO_TEST_CASE(StdOperationSubtractFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<float, double>(
    popops::sub,
    [](float x, float y) -> double {
       return static_cast<double>(x) - static_cast<double>(y);
    });
}

BOOST_AUTO_TEST_CASE(StdOperationSubtractHalf,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.14))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.14))
                  ) {
  binaryOpTestHalf(
    popops::sub,
    [](float x, float y) -> float {
      return x - y;
    });
}

BOOST_AUTO_TEST_CASE(StdOperationSubtractInt,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  binaryOpTest<int, int>(
    popops::sub,
    [](int x, int y) -> int {
      return (x - y);
    });
}

BOOST_AUTO_TEST_CASE(StdOperationRoundFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  unaryOpTest<float, double>(popops::round,
    [](float x) -> double {
      return std::round(x);
    });
}


BOOST_AUTO_TEST_CASE(StdOperationSelectFloat,
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

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
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

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
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

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
                  *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01))
                  *utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))
                  ) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

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
  popops::addCodelets(graph);

  Tensor in1, in2;
  in1 = graph.addVariable(FLOAT, {2, 2}, "t1");
  in2 = graph.addVariable(FLOAT, {2, 2}, "t2");

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
  popops::addCodelets(graph);

  Tensor in1, in2, in3, in4;
  in1 = graph.addVariable(FLOAT, {2, 2}, "t1");
  in2 = graph.addVariable(FLOAT, {2, 2}, "t2");
  in3 = graph.addVariable(BOOL, {2, 2}, "pred1");
  in4 = graph.addVariable(BOOL, {2, 2}, "pred2");

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
  popops::addCodelets(graph);

  Tensor in = graph.addVariable(FLOAT, {2, 2}, "t1");
  auto prog = Sequence();
  BOOST_CHECK_THROW(allTrue(graph, in, prog, "all_true"),
                    poplib_error);
}

BOOST_AUTO_TEST_CASE(StdOperationAllTrue) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  Tensor in = graph.addVariable(INT, {2}, "t1");
  Tensor ones = graph.addConstant(INT, {2}, 1);
  Tensor zeros = graph.addConstant(INT, {2}, 0);

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
  popops::addCodelets(graph);

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

using namespace popops::expr;

BOOST_AUTO_TEST_CASE(MapTest) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto in = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in");
  mapTensorLinearly(graph, in);
  auto out = map(graph, Add(Abs(_1), Const(3)), {in}, prog);

  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  float hIn[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn);

  Engine eng(device, graph, prog);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  Engine::ReportOptions opt;
  opt.doLayerWiseBreakdown = true;
  auto execReport = eng.getExecutionReport(opt);
  execReport.printSummary(std::cerr);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = fabs(hIn[i][j]) + 3;
      BOOST_TEST(hOut[i][j] == expected);
    }
  }
}


BOOST_AUTO_TEST_CASE(MapTestMultiTensor) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto a = graph.addVariable(FLOAT, {DIM_SIZE}, "a");
  mapTensorLinearly(graph, a);
  auto b = graph.addVariable(FLOAT, {DIM_SIZE}, "b");
  mapTensorLinearly(graph, b);
  auto c = graph.addVariable(FLOAT, {DIM_SIZE}, "c");
  mapTensorLinearly(graph, c);
  auto out = map(graph, Select(Add(_3, _2), Add(_1, _2), Gte(_1, _2)),
                 {a, b, c}, prog);

  graph.createHostWrite("a", a);
  graph.createHostWrite("b", b);
  graph.createHostWrite("c", c);
  graph.createHostRead("out", out);

  float aIn[DIM_SIZE],
        bIn[DIM_SIZE],
        cIn[DIM_SIZE];
  float hOut[DIM_SIZE];

  std::mt19937 randomEngine;
  std::uniform_real_distribution<> dist(-10.0, +10.0);
  for (unsigned i = 0; i < DIM_SIZE; ++i) {
    aIn[i] = dist(randomEngine);
    bIn[i] = dist(randomEngine);
    cIn[i] = dist(randomEngine);
  }

  Engine eng(device, graph, prog);
  eng.writeTensor("a", aIn);
  eng.writeTensor("b", bIn);
  eng.writeTensor("c", cIn);
  eng.run();
  eng.readTensor("out", hOut);

  Engine::ReportOptions opt;
  opt.doLayerWiseBreakdown = true;
  auto execReport = eng.getExecutionReport(opt);
  execReport.printSummary(std::cerr);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    auto expected = (aIn[i] >= bIn[i]) ? cIn[i] + bIn[i] : aIn[i] + bIn[i];
    BOOST_TEST(hOut[i] == expected);
  }
}

BOOST_AUTO_TEST_CASE(MapInPlaceTest) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in");
  mapTensorLinearly(graph, t);
  mapInPlace(graph, Add(Abs(_1), Const(3)), {t}, prog);

  graph.createHostWrite("in", t);
  graph.createHostRead("out", t);

  float hIn[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn);

  Engine eng(device, graph, prog);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  Engine::ReportOptions opt;
  opt.doLayerWiseBreakdown = true;
  auto execReport = eng.getExecutionReport(opt);
  execReport.printSummary(std::cerr);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = fabs(hIn[i][j]) + 3;
      BOOST_TEST(hOut[i][j] == expected);
    }
  }
}
