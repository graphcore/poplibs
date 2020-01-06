// Copyright (c) Graphcore Ltd, All rights reserved.
#include "TestDevice.hpp"
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/AllTrue.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <random>
#include <stdexcept>
#include <string>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_test::util;
namespace br = boost::random;
namespace pe = popops::expr;

static DeviceType deviceType;

const poplar::OptionFlags options{{"target.workerStackSizeInBytes", "0x1000"}};

#define DIM_SIZE 3

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CHECK(pred)                                                            \
  do {                                                                         \
    if (!(pred)) {                                                             \
      throw std::runtime_error("failed check '" #pred                          \
                               "' on line " TOSTRING(__LINE__));               \
    }                                                                          \
  } while (false)

#define CHECK_CLOSE(a, b)                                                      \
  do {                                                                         \
    if (!checkIsClose<                                                         \
            typename std::common_type<decltype(a), decltype(b)>::type>(        \
            a, b, 0.01)) {                                                     \
      throw std::runtime_error(                                                \
          "failed close check on line " TOSTRING(__LINE__));                   \
    }                                                                          \
  } while (false)

static Tensor mapUnaryOpTensor(Graph &graph, const Type &type) {
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
 * for each element in a row. The first two elements are +/-Infinity
 */
static void setUnaryOpInput(float hIn[DIM_SIZE][DIM_SIZE]) {
  float val = -100;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      float sign = 1.0 - 2.0 * ((r + c) & 1);
      hIn[r][c] = (val + (r * DIM_SIZE + c) * .1) * sign;
    }
  }
  hIn[0][0] = std::numeric_limits<float>::infinity();
  hIn[0][1] = -std::numeric_limits<float>::infinity();
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
  std::mt19937 randomEngine;
  br::uniform_int_distribution<> dist(0, 1);
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn[r][c] = dist(randomEngine);
    }
  }
}

static void setUnaryOpInputHalf(float hIn[DIM_SIZE][DIM_SIZE]) {
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
                              float hIn2[DIM_SIZE][DIM_SIZE],
                              bool isShift = false) {
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
                                  float hIn2[DIM_SIZE][DIM_SIZE],
                                  bool isShift = false) {
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

/* Generates two 2D matrix of size DIM_SIZE x DIM_SIZE containing
 * boolean values. All combinations of boolean values are produced
 */
static void setBinaryOpInputs(bool hIn1[DIM_SIZE][DIM_SIZE],
                              bool hIn2[DIM_SIZE][DIM_SIZE],
                              bool isShift = false) {
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn1[r][c] = r & 1;
      hIn2[r][c] = (r + c) & 1;
    }
  }
}

static void setBinaryOpInputs(int hIn1[DIM_SIZE][DIM_SIZE],
                              int hIn2[DIM_SIZE][DIM_SIZE],
                              bool isShift = false) {
  int val1 = -100;
  int val2 = 59;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn1[r][c] = (1 - 2 * (r & 1)) * (r + val1);
      hIn2[r][c] = (1 - 2 * ((r + c) & 1)) * (r + c + val2);

      // Shifting by more than 32 is undefined.
      if (isShift) {
        hIn2[r][c] = hIn2[r][c] % 32;
        if (hIn2[r][c] < 0)
          hIn2[r][c] = -hIn2[r][c];
      }
    }
  }
}

template <typename T> void convertToPositive(T array[DIM_SIZE][DIM_SIZE]) {
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      array[i][j] = std::abs(array[i][j]);
    }
  }
}

template <> void convertToPositive(bool array[DIM_SIZE][DIM_SIZE]) {}

template <> void convertToPositive(float array[DIM_SIZE][DIM_SIZE]) {
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      array[i][j] = std::fabs(array[i][j]);
    }
  }
}

using UnaryOpFn =
    std::function<Tensor(Graph &, const Tensor &, Sequence &,
                         const std::string &, const poplar::OptionFlags &)>;

template <typename T, typename TestT>
void unaryOpTest(const UnaryOpFn &op, const std::function<TestT(T)> &testFn,
                 bool positiveInputs = false) {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto in = mapUnaryOpTensor(graph, equivalent_device_type<T>().value);
  auto prog = Sequence();

  auto out = op(graph, in, prog, "unaryOp", {});
  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  Engine eng(graph, prog, options);
  T hIn[DIM_SIZE][DIM_SIZE];
  T hOut[DIM_SIZE][DIM_SIZE];
  device.bind([&](const Device &d) {
    eng.load(d);

    setUnaryOpInput(hIn);
    if (positiveInputs) {
      convertToPositive(hIn);
    }
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto res = testFn(hIn[i][j]);
      CHECK_CLOSE(res, hOut[i][j]);
    }
  }
}

using BinaryOpFn =
    std::function<Tensor(Graph &, const Tensor &, const Tensor &, Sequence &,
                         const std::string &, const poplar::OptionFlags &)>;

template <typename BinaryOpFn> struct BinaryOpFnPtr;

template <typename R, typename... Args>
struct BinaryOpFnPtr<std::function<R(Args...)>> {
  using type = R (*)(Args...);
};

using BinaryOpFnPtr_t = BinaryOpFnPtr<BinaryOpFn>::type;

template <typename T, typename TestT, typename OutT = T>
void binaryOpTest(const BinaryOpFn &op,
                  const std::function<TestT(T, T)> &testFn,
                  bool isShift = false) {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Tensor in1, in2;
  auto type = equivalent_device_type<T>().value;

  std::tie(in1, in2) = mapBinaryOpTensors(graph, type);

  auto prog = Sequence();

  auto out = op(graph, in1, in2, prog, "binaryOp", {});
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", out);

  Engine eng(graph, prog, options);
  T hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  OutT hOut[DIM_SIZE][DIM_SIZE];
  device.bind([&](const Device &d) {
    eng.load(d);
    setBinaryOpInputs(hIn1, hIn2, isShift);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      T res = testFn(hIn1[i][j], hIn2[i][j]);
      CHECK_CLOSE(static_cast<TestT>(hOut[i][j]), res);
    }
  }
}

void binaryOpTestHalf(const BinaryOpFn &op,
                      const std::function<float(float, float)> &testFn) {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  const auto &target = device.getTarget();
  popops::addCodelets(graph);

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, poplar::HALF);

  auto prog = Sequence();

  auto out = op(graph, in1, in2, prog, "binaryOp", {});
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", out);

  Engine eng(graph, prog, options);
  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];
  device.bind([&](const Device &d) {
    eng.load(d);
    setBinaryOpInputsHalf(hIn1, hIn2);
    auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
    std::vector<char> rawIn1(rawBufSize), rawIn2(rawBufSize),
        rawOut(rawBufSize);
    poplar::copyFloatToDeviceHalf(target, &hIn1[0][0], rawIn1.data(),
                                  DIM_SIZE * DIM_SIZE);
    poplar::copyFloatToDeviceHalf(target, &hIn2[0][0], rawIn2.data(),
                                  DIM_SIZE * DIM_SIZE);
    eng.writeTensor("in1", rawIn1.data(), rawIn1.data() + rawIn1.size());
    eng.writeTensor("in2", rawIn2.data(), rawIn2.data() + rawIn2.size());
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
    poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                  DIM_SIZE * DIM_SIZE);
  });
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto res = testFn(hIn1[i][j], hIn2[i][j]);
      CHECK_CLOSE(static_cast<float>(hOut[i][j]), res);
    }
  }
}

void powTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
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

  Engine eng(graph, prog, options);
  float hOut[DIM_SIZE][DIM_SIZE];
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = std::pow(static_cast<double>(hIn1[i][j]),
                            static_cast<double>(hIn2[i][j]));
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

template <typename InType> void selectTest() {
  auto type = equivalent_device_type<InType>().value;
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  InType hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);
  bool hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn3[r][c] = (c)&0x1;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, type);
  Tensor in3 = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = select(graph, in1, in2, in3, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);
  graph.createHostRead("out", out);

  InType hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.writeTensor("in3", hIn3, &hIn3[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      InType res = hIn3[i][j] ? hIn1[i][j] : hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void selectTestFloatLHSConst() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hLhs = 1.f;
  float hRhs[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hRhs);
  bool hPred[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hPred[r][c] = (c)&0x1;
    }
  }

  Tensor rhs = mapUnaryOpTensor(graph, FLOAT);
  Tensor pred = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = map(
      graph,
      pe::TernaryOp(pe::TernaryOpType::SELECT, pe::Const(hLhs), pe::_1, pe::_2),
      {rhs, pred}, prog);
  graph.createHostWrite("rhs", rhs);
  graph.createHostWrite("pred", pred);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("rhs", hRhs, &hRhs[DIM_SIZE]);
    eng.writeTensor("pred", hPred, &hPred[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hPred[i][j] ? hLhs : hRhs[i][j];
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void selectTestFloatRHSConst() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hRhs = 1.f;
  float hLhs[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hLhs);
  bool hPred[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hPred[r][c] = (c)&0x1;
    }
  }

  Tensor lhs = mapUnaryOpTensor(graph, FLOAT);
  Tensor pred = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = map(
      graph,
      pe::TernaryOp(pe::TernaryOpType::SELECT, pe::_1, pe::Const(hRhs), pe::_2),
      {lhs, pred}, prog);
  graph.createHostWrite("lhs", lhs);
  graph.createHostWrite("pred", pred);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("lhs", hLhs, &hLhs[DIM_SIZE]);
    eng.writeTensor("pred", hPred, &hPred[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hPred[i][j] ? hLhs[i][j] : hRhs;
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void selectTestFloatLHSAndRHSConst() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hLhs = 2.f;
  float hRhs = 1.f;
  bool hPred[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hPred[r][c] = (c)&0x1;
    }
  }

  Tensor pred = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = map(graph,
                 pe::TernaryOp(pe::TernaryOpType::SELECT, pe::Const(hLhs),
                               pe::Const(hRhs), pe::_1),
                 {pred}, prog);
  graph.createHostWrite("pred", pred);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("pred", hPred, &hPred[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hPred[i][j] ? hLhs : hRhs;
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void selectTestHalfLHSAndRHSConst() {
  auto device = createTestDevice(deviceType);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hLhs = 2.f;
  float hRhs = 1.f;
  bool hPred[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hPred[r][c] = (c)&0x1;
    }
  }

  Tensor pred = mapUnaryOpTensor(graph, BOOL);

  auto prog = Sequence();
  auto out = map(graph,
                 pe::TernaryOp(pe::TernaryOpType::SELECT, pe::ConstHalf(hLhs),
                               pe::ConstHalf(hRhs), pe::_1),
                 {pred}, prog);
  CHECK(out.elementType() == HALF);
  graph.createHostWrite("pred", pred);
  graph.createHostRead("out", out);

  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawOut(rawBufSize);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("pred", hPred, &hPred[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
  });

  float hOut[DIM_SIZE][DIM_SIZE];
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                DIM_SIZE * DIM_SIZE);
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hPred[i][j] ? hLhs : hRhs;
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

template <typename InType> void broadcastSelectorSelectTest(bool inPlace) {
  auto type = equivalent_device_type<InType>().value;
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  InType hIn1[DIM_SIZE][DIM_SIZE];
  InType hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);
  bool hIn3 = true;

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, type);

  Tensor in3 = graph.addVariable(BOOL, {}, "pred"); // selector
  mapTensorLinearly(graph, in3);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.setInitialValue(in3, hIn3);

  auto prog = Sequence();

  if (inPlace) {
    selectInPlace(graph, in1, in2, in3, prog);
    graph.createHostRead("out", in1);
  } else {
    auto out = select(graph, in1, in2, in3, prog);
    graph.createHostRead("out", out);
  }

  InType hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      InType res = hIn3 ? hIn1[i][j] : hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void clampTestFloatMinConst() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hMin = -0.5f;
  float hMax[DIM_SIZE][DIM_SIZE];
  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hMax[r][c] = 0.5;
    }
  }

  Tensor max = mapUnaryOpTensor(graph, FLOAT);
  Tensor in = mapUnaryOpTensor(graph, FLOAT);

  auto prog = Sequence();
  auto out = map(
      graph,
      pe::TernaryOp(pe::TernaryOpType::CLAMP, pe::_1, pe::Const(hMin), pe::_2),
      {in, max}, prog);
  graph.createHostWrite("max", max);
  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("max", hMax, &hMax[DIM_SIZE]);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hIn[i][j];
      if (res < hMin)
        res = hMin;
      if (res > hMax[i][j])
        res = hMax[i][j];

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void clampTestFloatMaxConst() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hMax = 0.5f;
  float hMin[DIM_SIZE][DIM_SIZE];
  float hIn[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn);

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hMin[r][c] = 0.5;
    }
  }

  Tensor min = mapUnaryOpTensor(graph, FLOAT);
  Tensor in = mapUnaryOpTensor(graph, FLOAT);

  auto prog = Sequence();
  auto out = map(
      graph,
      pe::TernaryOp(pe::TernaryOpType::CLAMP, pe::_1, pe::_2, pe::Const(hMax)),
      {in, min}, prog);
  graph.createHostWrite("min", min);
  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("min", hMin, &hMin[DIM_SIZE]);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hIn[i][j];
      if (res < hMin[i][j])
        res = hMin[i][j];
      if (res > hMax)
        res = hMax;

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

template <typename InType> void clampTest(bool inPlace) {
  auto type = equivalent_device_type<InType>().value;
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  InType hIn1[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn1);
  InType hIn2[DIM_SIZE][DIM_SIZE], hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn2[r][c] = static_cast<InType>(-5.5);
      hIn3[r][c] = static_cast<InType>(5.5);
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, type);
  Tensor in3 = mapUnaryOpTensor(graph, type);

  auto prog = Sequence();

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);

  if (inPlace) {
    clampInPlace(graph, in1, in2, in3, prog);
    graph.createHostRead("out", in1);
  } else {
    auto out = clamp(graph, in1, in2, in3, prog);
    graph.createHostRead("out", out);
  }

  InType hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.writeTensor("in3", hIn3, &hIn3[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      InType res = hIn1[i][j];
      if (res < hIn2[i][j])
        res = hIn2[i][j];
      if (res > hIn3[i][j])
        res = hIn3[i][j];

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

template <typename InType>
void broadcastClampTest(bool inPlace, size_t dim = DIM_SIZE) {
  auto type = equivalent_device_type<InType>().value;
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  InType hIn1[DIM_SIZE][DIM_SIZE];
  InType hIn2 = InType(-5.5);
  InType hIn3 = InType(5.5);

  if (dim == DIM_SIZE) {
    setUnaryOpInput(hIn1);
  } else if (dim == 1) {
    hIn1[0][0] = 10;
  } else {
    throw std::runtime_error("broadcastClampTest: Unsupported dimension size");
  }

  Tensor in1 = graph.addVariable(type, {dim, dim}, "input"); // source
  Tensor in2 = graph.addVariable(type, {}, "lower");         // lower
  Tensor in3 = graph.addVariable(type, {}, "upper");         // upper

  mapTensorLinearly(graph, in1);
  mapTensorLinearly(graph, in2);
  mapTensorLinearly(graph, in3);

  graph.createHostWrite("in1", in1);
  graph.setInitialValue(in2, hIn2);
  graph.setInitialValue(in3, hIn3);

  auto prog = Sequence();

  if (inPlace) {
    clampInPlace(graph, in1, in2, in3, prog);
    graph.createHostRead("out", in1);
  } else {
    auto out = clamp(graph, in1, in2, in3, prog);
    graph.createHostRead("out", out);
  }

  InType hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[dim]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[dim]);
  });

  /* Check result */
  for (auto i = 0U; i < dim; ++i) {
    for (auto j = 0U; j < dim; ++j) {
      InType res = hIn1[i][j];
      if (res < hIn2)
        res = hIn2;
      if (res > hIn3)
        res = hIn3;

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void binaryOutputMapChoiceTest() {
  auto device = createTestDevice(deviceType, 1, 4);
  Graph graph(device.getTarget());
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
  CHECK(tile1[0].size() > 0);
  CHECK(tile1[1].size() > 0);
  CHECK(tile1[2].size() > 0);
  CHECK(tile1[3].size() > 0);

  const auto &tile2 = graph.getTileMapping(out2);
  CHECK(tile2[0].size() > 0);
  CHECK(tile2[1].size() > 0);
  CHECK(tile2[2].size() > 0);
  CHECK(tile2[3].size() > 0);
}

void trinaryOutputMapChoiceTest() {
  auto device = createTestDevice(deviceType, 1, 4);
  Graph graph(device.getTarget());
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
  CHECK(tile1[0].size() > 0);
  CHECK(tile1[1].size() > 0);
  CHECK(tile1[2].size() > 0);
  CHECK(tile1[3].size() > 0);

  const auto &tile2 = graph.getTileMapping(out2);
  CHECK(tile2[0].size() > 0);
  CHECK(tile2[1].size() > 0);
  CHECK(tile2[2].size() > 0);
  CHECK(tile2[3].size() > 0);

  const auto &tile3 = graph.getTileMapping(out3);
  CHECK(tile3[0].size() > 0);
  CHECK(tile3[1].size() > 0);
  CHECK(tile3[2].size() > 0);
  CHECK(tile3[3].size() > 0);
}

void allTrueBadTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Tensor in = graph.addVariable(FLOAT, {2, 2}, "t1");
  auto prog = Sequence();

  bool throws = false;
  try {
    allTrue(graph, in, prog, "all_true");
  } catch (const poplibs_error &) {
    throws = true;
  }
  CHECK(throws);
}

void allTrueTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  constexpr auto SIZE = 2;
  Tensor in = graph.addVariable(INT, {SIZE}, "t1");
  Tensor ones = graph.addConstant(INT, {SIZE}, 1);
  Tensor zeros = graph.addConstant(INT, {SIZE}, 0);
  graph.setTileMapping(ones, 0);
  graph.setTileMapping(zeros, 0);
  graph.setTileMapping(in, 0);

  auto bodyProg = Sequence();
  subInPlace(graph, in, ones, bodyProg);

  auto condProg = Sequence();
  Tensor neZero = neq(graph, in, zeros, condProg);
  auto predicate = allTrue(graph, neZero, condProg, "all_true");

  int init[SIZE] = {10, 8};
  int output[SIZE] = {0, 0};

  auto mainProg = Sequence();
  mainProg.add(RepeatWhileTrue(condProg, predicate, bodyProg));
  graph.createHostWrite("in", in);
  graph.createHostRead("out", in);

  Engine eng(graph, mainProg, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", init, &init[SIZE]);
    eng.run();
    eng.readTensor("out", output, &output[SIZE]);
  });

  CHECK(output[0] == 2);
  CHECK(output[1] == 0);
}

void isFiniteTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
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

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool expected = !(i <= 1 && j <= 1);
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

using namespace popops::expr;

void mapTestCast(bool inPlace, poplar::Type in1Type, poplar::Type in2Type) {
  auto device = createTestDevice(deviceType);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto in1 = graph.addVariable(in1Type, {DIM_SIZE, DIM_SIZE}, "in1");
  auto in2 = graph.addVariable(in2Type, {DIM_SIZE, DIM_SIZE}, "in2");
  auto in3 = graph.addVariable(in1Type, {DIM_SIZE, DIM_SIZE}, "in3");
  mapTensorLinearly(graph, in1);
  mapTensorLinearly(graph, in2);
  mapTensorLinearly(graph, in3);
  Tensor out, out2;
  if (inPlace) {
    mapInPlace(graph, Add(_1, Cast(_2, in1Type)), {in1, in2}, prog);
    mapInPlace(graph, Add(Cast(_2, in1Type), _1), {in3, in2}, prog);
  } else {
    out = map(graph, Add(_1, Cast(_2, in1Type)), {in1, in2}, prog);
    // The type cast to determines the output type.
    out2 =
        map(graph, Add(Cast(_2, in1Type), Cast(_1, in1Type)), {in1, in2}, prog);
  }
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in3", in3);
  graph.createHostWrite("in2", in2);
  if (inPlace) {
    graph.createHostRead("out", in1);
    graph.createHostRead("out2", in3);
  } else {
    graph.createHostRead("out", out);
    graph.createHostRead("out2", out2);
  }
  float hIn1[DIM_SIZE][DIM_SIZE];
  float hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];
  float hOut2[DIM_SIZE][DIM_SIZE];

  setBinaryOpInputsHalf(hIn1, hIn2);
  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawIn1(rawBufSize), rawIn2(rawBufSize), rawOut(rawBufSize),
      rawOut2(rawBufSize);
  if (in1Type == HALF) {
    poplar::copyFloatToDeviceHalf(target, &hIn1[0][0], rawIn1.data(),
                                  DIM_SIZE * DIM_SIZE);
  }
  if (in2Type == HALF) {
    poplar::copyFloatToDeviceHalf(target, &hIn2[0][0], rawIn2.data(),
                                  DIM_SIZE * DIM_SIZE);
  }
  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    if (in1Type == HALF) {
      eng.writeTensor("in1", rawIn1.data(), rawIn1.data() + rawIn1.size());
      eng.writeTensor("in3", rawIn1.data(), rawIn1.data() + rawIn1.size());
    } else {
      eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
      eng.writeTensor("in3", hIn1, &hIn1[DIM_SIZE]);
    }
    if (in2Type == HALF) {
      eng.writeTensor("in2", rawIn2.data(), rawIn2.data() + rawIn2.size());
    } else {
      eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    }
    eng.run();
    if (in1Type == HALF) {
      eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
      eng.readTensor("out2", rawOut2.data(), rawOut2.data() + rawOut2.size());
    } else {
      eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
      eng.readTensor("out2", hOut2, &hOut2[DIM_SIZE]);
    }
  });
  if (in1Type == HALF) {
    poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                  DIM_SIZE * DIM_SIZE);
    poplar::copyDeviceHalfToFloat(target, rawOut2.data(), &hOut2[0][0],
                                  DIM_SIZE * DIM_SIZE);
  }
  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = hIn1[i][j] + hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], expected);
      CHECK_CLOSE(hOut2[i][j], expected);
    }
  }
}

template <typename T, typename TestT>
void unaryMapExprTest(const expr::Expr &expr,
                      const std::function<TestT(T)> &testFn,
                      bool positiveInputs = false) {
  auto op = [&](Graph &graph, const Tensor &t, Sequence &prog,
                const std::string &, const poplar::OptionFlags &) -> Tensor {
    return map(graph, expr, {t}, prog);
  };
  return unaryOpTest(op, testFn, positiveInputs);
}

template <typename T, typename TestT>
void binaryMapExprTest(const expr::Expr &expr,
                       const std::function<TestT(T, T)> &testFn,
                       bool positiveInputs = false) {
  auto op = [&](Graph &graph, const Tensor &t0, const Tensor &t1,
                Sequence &prog, const std::string &,
                const poplar::OptionFlags &) -> Tensor {
    return map(graph, expr, {t0, t1}, prog);
  };
  return binaryOpTest(op, testFn, positiveInputs);
}

void mapTestCastIntToFloat() {
  auto device = createTestDevice(deviceType);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto in1 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  auto in2 = graph.addVariable(INT, {DIM_SIZE, DIM_SIZE}, "in2");
  mapTensorLinearly(graph, in1);
  mapTensorLinearly(graph, in2);
  auto out = map(graph, Add(_1, Cast(_2, FLOAT)), {in1, in2}, prog);
  auto out2 = map(graph, Cast(_1, FLOAT), {in2}, prog);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", out);
  graph.createHostRead("out2", out2);

  float hIn1[DIM_SIZE][DIM_SIZE];
  int hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];
  float hOut2[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn1);
  setUnaryOpInput(hIn2);
  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
    eng.readTensor("out2", hOut2, &hOut2[DIM_SIZE]);
  });
  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = hIn1[i][j] + hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], expected);

      CHECK_CLOSE(hOut2[i][j], hIn2[i][j]);
    }
  }
}
void mapTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
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

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = fabs(hIn[i][j]) + 3;
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void mapTestMultiTensor() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
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

  float aIn[DIM_SIZE], bIn[DIM_SIZE], cIn[DIM_SIZE];
  float hOut[DIM_SIZE];

  std::mt19937 randomEngine;
  br::uniform_real_distribution<> dist(-10.0, +10.0);
  for (unsigned i = 0; i < DIM_SIZE; ++i) {
    aIn[i] = dist(randomEngine);
    bIn[i] = dist(randomEngine);
    cIn[i] = dist(randomEngine);
  }

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("a", aIn, &aIn[DIM_SIZE]);
    eng.writeTensor("b", bIn, &bIn[DIM_SIZE]);
    eng.writeTensor("c", cIn, &cIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    auto expected = (aIn[i] >= bIn[i]) ? cIn[i] + bIn[i] : aIn[i] + bIn[i];
    CHECK_CLOSE(hOut[i], expected);
  }
}

void mapInPlaceTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in");
  mapTensorLinearly(graph, t);
  mapInPlace(graph, Add(Abs(_1), _1), {t}, prog);

  graph.createHostWrite("in", t);
  graph.createHostRead("out", t);

  float hIn[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = fabs(hIn[i][j]) + hIn[i][j];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void mapInPlaceBroadcastTest() {
  auto device = createTestDevice(deviceType, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t = graph.addVariable(FLOAT, {2, 2, 2, 2}, "in");

  // force there to be more than one contiguous region
  graph.setTileMapping(t, 0);
  graph.setTileMapping(t[0][1][1], 3);

  const float scale = 0.5f;
  mapInPlace(graph, Mul(_1, Const(scale)), {t}, prog);

  Engine eng(graph, prog, options);
}

void mapInferTypeTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto a = graph.addVariable(FLOAT, {DIM_SIZE}, "a");
  mapTensorLinearly(graph, a);
  auto b = graph.addVariable(FLOAT, {DIM_SIZE}, "b");
  mapTensorLinearly(graph, b);
  auto c = graph.addVariable(BOOL, {DIM_SIZE}, "c");
  mapTensorLinearly(graph, c);
  auto out = map(graph, And(Equal(_1, _2), _3), {a, b, c}, prog);

  graph.createHostWrite("a", a);
  graph.createHostWrite("b", b);
  graph.createHostWrite("c", c);
  graph.createHostRead("out", out);

  float aIn[DIM_SIZE], bIn[DIM_SIZE];
  bool cIn[DIM_SIZE];
  bool hOut[DIM_SIZE];

  std::mt19937 randomEngine;
  br::uniform_real_distribution<> dist(-10.0, +10.0);
  for (unsigned i = 0; i < DIM_SIZE; ++i) {
    aIn[i] = dist(randomEngine);
    bIn[i] = dist(randomEngine);
    cIn[i] = dist(randomEngine) > 0.0;
  }

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("a", aIn, &aIn[DIM_SIZE]);
    eng.writeTensor("b", bIn, &bIn[DIM_SIZE]);
    eng.writeTensor("c", cIn, &cIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    auto expected = (aIn[i] == bIn[i]) && cIn[i];
    CHECK_CLOSE(hOut[i], expected);
  }
}

void addInPlaceTest() {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t1 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  auto t2 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in2");

  mapTensorLinearly(graph, t1);
  mapTensorLinearly(graph, t2);
  addInPlace(graph, t1, t2, prog);

  graph.createHostWrite("in1", t1);
  graph.createHostWrite("in2", t2);
  graph.createHostRead("out", t1);

  float hIn1[DIM_SIZE][DIM_SIZE];
  float hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn1);
  setUnaryOpInput(hIn2);

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
  eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
  eng.run();
  eng.readTensor("out", hOut, &hOut[DIM_SIZE]);

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = hIn1[i][j] + hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void multiplyFloatInPlaceConstScalarTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in");
  mapTensorLinearly(graph, t);
  const float b = 1.2;
  mulInPlace(graph, t, b, prog);

  graph.createHostWrite("in", t);
  graph.createHostRead("out", t);

  float hIn[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = hIn[i][j] * b;
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void addHalfConstScalarTest() {
  auto device = createTestDevice(deviceType);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto tIn = graph.addVariable(HALF, {DIM_SIZE, DIM_SIZE}, "in");
  mapTensorLinearly(graph, tIn);
  const float b = 1.0;
  auto tOut = add(graph, tIn, b, prog);

  graph.createHostWrite("in", tIn);
  graph.createHostRead("out", tOut);

  float hIn[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInputHalf(hIn);

  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawIn(rawBufSize), rawOut(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hIn[0][0], rawIn.data(),
                                DIM_SIZE * DIM_SIZE);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
  });

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                DIM_SIZE * DIM_SIZE);
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = hIn[i][j] + b;
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void binaryConcatTest() {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t1 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  auto t2 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in2");
  auto t3 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in3");
  auto t4 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in4");

  mapTensorLinearly(graph, t1);
  mapTensorLinearly(graph, t2);
  mapTensorLinearly(graph, t3);
  mapTensorLinearly(graph, t4);
  auto t5 = add(graph, concat(t1, t2, 1), concat(t3, t4, 1), prog);

  graph.createHostWrite("in1", t1);
  graph.createHostWrite("in2", t2);
  graph.createHostWrite("in3", t3);
  graph.createHostWrite("in4", t4);
  graph.createHostRead("out", t5);

  float hIn1[DIM_SIZE][DIM_SIZE];
  float hIn2[DIM_SIZE][DIM_SIZE];
  float hIn3[DIM_SIZE][DIM_SIZE];
  float hIn4[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][2 * DIM_SIZE];

  setUnaryOpInput(hIn1);
  setUnaryOpInput(hIn2);
  setUnaryOpInput(hIn3);
  setUnaryOpInput(hIn4);

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
  eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
  eng.writeTensor("in3", hIn3, &hIn3[DIM_SIZE]);
  eng.writeTensor("in4", hIn4, &hIn4[DIM_SIZE]);
  eng.run();
  eng.readTensor("out", hOut, &hOut[DIM_SIZE]);

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < 2 * DIM_SIZE; ++j) {
      auto expected = j < DIM_SIZE
                          ? hIn1[i][j] + hIn3[i][j]
                          : hIn2[i][j - DIM_SIZE] + hIn4[i][j - DIM_SIZE];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void unaryConcatTest() {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t1 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  auto t2 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in2");

  mapTensorLinearly(graph, t1);
  mapTensorLinearly(graph, t2);
  auto t3 = neg(graph, concat(t1, t2, 1), prog);

  graph.createHostWrite("in1", t1);
  graph.createHostWrite("in2", t2);
  graph.createHostRead("out", t3);

  float hIn1[DIM_SIZE][DIM_SIZE];
  float hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][2 * DIM_SIZE];

  setUnaryOpInput(hIn1);
  setUnaryOpInput(hIn2);

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
  eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
  eng.run();
  eng.readTensor("out", hOut, &hOut[DIM_SIZE]);

  if (deviceType == DeviceType::IpuModel) {
    eng.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < 2 * DIM_SIZE; ++j) {
      auto expected = j < DIM_SIZE ? -hIn1[i][j] : -hIn2[i][j - DIM_SIZE];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  std::string test;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("test",
     po::value<std::string>(&test)->required(),
     "The test to run");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (test == "AbsFloat") {
    unaryOpTest<float, double>(popops::abs, [](float x) -> double {
      double res = fabs(static_cast<double>(x));
      return res;
    });
  } else if (test == "AbsInt") {
    unaryOpTest<int, int>(popops::abs,
                          [](int x) -> int { return std::abs(x); });
  } else if (test == "AddFloat") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::add),
                                [](float x, float y) -> double {
                                  double res = x + y;
                                  return res;
                                });
  } else if (test == "Atan2Float") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::atan2),
                                [](float x, float y) -> double {
                                  double res = std::atan2(x, y);
                                  return res;
                                });
  } else if (test == "AddInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::add),
                           [](int x, int y) -> int { return x + y; });
  } else if (test == "BitwiseAndInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::bitwiseAnd),
                           [](int x, int y) -> int { return x & y; });
  } else if (test == "BitwiseOrInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::bitwiseOr),
                           [](int x, int y) -> int { return x | y; });
  } else if (test == "BitwiseXorInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::bitwiseXor),
                           [](int x, int y) -> int { return x ^ y; });
  } else if (test == "BitwiseXnorInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::bitwiseXnor),
                           [](int x, int y) -> int { return ~(x ^ y); });
  } else if (test == "BitwiseNotInt") {
    unaryOpTest<int, int>(popops::bitwiseNot, [](int x) -> int { return ~x; });
  } else if (test == "Ceil") {
    unaryOpTest<float, double>(popops::ceil, [](float x) -> double {
      double res = std::ceil(static_cast<double>(x));
      return res;
    });

  } else if (test == "Cos") {
    unaryOpTest<float, double>(popops::cos, [](float x) -> double {
      double res = std::cos(static_cast<double>(x));
      return res;
    });
  } else if (test == "CountLeadingZeros") {
    unaryOpTest<int, int>(popops::countLeadingZeros, [](int x) -> int {
      return x ? __builtin_clz(x) : 0;
    });
  } else if (test == "DivideFloat") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::div),
                                [](float x, float y) -> double {
                                  double res = x / y;
                                  return res;
                                });
  } else if (test == "DivideInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::div),
                           [](int x, int y) -> int {
                             int res = x / y;
                             return res;
                           });
  } else if (test == "EqualFloat") {
    binaryOpTest<float, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::eq),
        [](float x, float y) -> bool { return x == y; });
  } else if (test == "GreaterThanBool") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::gt),
        [](bool x, bool y) -> bool { return x > y; });
  } else if (test == "GreaterThanEqualBool") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::gteq),
        [](bool x, bool y) -> bool { return x >= y; });
  } else if (test == "LessThanBool") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::lt),
        [](bool x, bool y) -> bool { return x < y; });
  } else if (test == "LessThanEqualBool") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::lteq),
        [](bool x, bool y) -> bool { return x <= y; });
  } else if (test == "Exponent") {
    unaryOpTest<float, float>(popops::exp, [](float x) -> float {
      double res = std::exp(static_cast<double>(x));
      return res;
    });
  } else if (test == "ExponentMinus1") {
    unaryOpTest<float, float>(popops::expm1, [](float x) -> float {
      double res = std::expm1(static_cast<double>(x));
      return res;
    });
  } else if (test == "Floor") {
    unaryOpTest<float, double>(popops::floor, [](float x) -> double {
      double res = std::floor(static_cast<double>(x));
      return res;
    });
  } else if (test == "GreaterThanFloat") {
    binaryOpTest<float, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::gt),
        [](float x, float y) -> bool { return x > y; });
  } else if (test == "GreaterThanInt") {
    binaryOpTest<int, bool, bool>(static_cast<BinaryOpFnPtr_t>(popops::gt),
                                  [](int x, int y) -> bool { return x > y; });
  } else if (test == "GreaterThanEqualFloat") {
    binaryOpTest<float, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::gteq),
        [](float x, float y) -> bool { return x >= y; });
  } else if (test == "LessThanFloat") {
    binaryOpTest<float, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::lt),
        [](float x, float y) -> bool { return x < y; });
  } else if (test == "LessThanEqualFloat") {
    binaryOpTest<float, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::lteq),
        [](float x, float y) -> bool { return x <= y; });
  } else if (test == "Logarithm") {
    unaryOpTest<float, double>(
        popops::log,
        [](float x) -> double {
          double res = std::log(static_cast<double>(x));
          return res;
        },
        true /* positive inputs */);
  } else if (test == "Logarithm1Plus") {
    unaryOpTest<float, double>(
        popops::log1p,
        [](float x) -> double {
          double res = std::log1p(static_cast<double>(x));
          return res;
        },
        true /* positive inputs */);
  } else if (test == "LogicalAnd") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::logicalAnd),
        [](bool x, bool y) -> bool { return x && y; });
  } else if (test == "LogicalNot") {
    unaryOpTest<bool, bool>(popops::logicalNot,
                            [](bool x) -> bool { return !x; });
  } else if (test == "LogicalOr") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::logicalOr),
        [](bool x, bool y) -> bool { return x || y; });
  } else if (test == "MaxFloat") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::max),
                                [](float x, float y) -> double {
                                  double res = std::max(x, y);
                                  return res;
                                });
  } else if (test == "MaxInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::max),
                           [](int x, int y) -> int {
                             auto res = std::max(x, y);
                             return res;
                           });
  } else if (test == "MinFloat") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::min),
                                [](float x, float y) -> double {
                                  double res = std::min(x, y);
                                  return res;
                                });
  } else if (test == "MinInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::min),
                           [](int x, int y) -> int {
                             auto res = std::min(x, y);
                             return res;
                           });
  } else if (test == "Multiply") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::mul),
                                [](float x, float y) -> double {
                                  double res = x * y;
                                  return res;
                                });
  } else if (test == "NotEqualFloat") {
    binaryOpTest<float, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::neq),
        [](float x, float y) -> bool { return x != y; });
  } else if (test == "NotEqualBool") {
    binaryOpTest<bool, bool, bool>(
        static_cast<BinaryOpFnPtr_t>(popops::neq),
        [](bool x, bool y) -> bool { return x != y; });
  } else if (test == "NegateFloat") {
    unaryOpTest<float, double>(popops::neg,
                               [](float x) -> double { return -x; });
  } else if (test == "NegateInt") {
    unaryOpTest<int, int>(popops::neg, [](int x) -> int { return -x; });
  } else if (test == "Popcount") {
    unaryOpTest<int, int>(popops::popcount,
                          [](int x) -> int { return __builtin_popcount(x); });
  } else if (test == "Power") {
    powTest();
  } else if (test == "RemainderFloat") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::rem),
                                [](float x, float y) -> double {
                                  double res =
                                      std::fmod(static_cast<double>(x),
                                                static_cast<double>(y));
                                  return res;
                                });
  } else if (test == "RemainderInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::rem),
                           [](int x, int y) -> double { return x % y; });
  } else if (test == "ShiftLeftInt") {
    binaryOpTest<int, int, int>(
        static_cast<BinaryOpFnPtr_t>(popops::shiftLeft),
        [](int x, int y) -> int { return x << y; }, true);
  } else if (test == "ShiftRightInt") {
    binaryOpTest<int, int, int>(
        static_cast<BinaryOpFnPtr_t>(popops::shiftRight),
        [](int x, int y) -> int { return (unsigned)x >> y; }, true);
  } else if (test == "ShiftRightSignExtendInt") {
    binaryOpTest<int, int, int>(
        static_cast<BinaryOpFnPtr_t>(popops::shiftRightSignExtend),
        [](int x, int y) -> int { return x >> y; }, true);
  } else if (test == "SignumFloat") {
    unaryOpTest<float, double>(popops::signum, [](float x) -> double {
      double res = (0 < x) - (x < 0);
      return res;
    });
  } else if (test == "SignumInt") {
    unaryOpTest<int, int>(popops::signum, [](int x) -> int {
      int res = (0 < x) - (x < 0);
      return res;
    });
  } else if (test == "Sin") {
    unaryOpTest<float, double>(popops::sin, [](float x) -> double {
      double res = std::sin(static_cast<double>(x));
      return res;
    });
  } else if (test == "Asin") {
    unaryOpTest<float, double>(popops::asin, [](float x) -> double {
      double res = std::asin(static_cast<double>(x));
      return res;
    });
  } else if (test == "Tan") {
    unaryOpTest<float, double>(popops::tan, [](float x) -> double {
      double res = std::tan(static_cast<double>(x));
      return res;
    });
  } else if (test == "Tanh") {
    unaryOpTest<float, double>(popops::tanh, [](float x) -> double {
      double res = std::tanh(static_cast<double>(x));
      return res;
    });
  } else if (test == "Square") {
    unaryOpTest<float, double>(popops::square, [](float x) -> double {
      double xd = static_cast<double>(x);
      double res = xd * xd;
      return res;
    });
  } else if (test == "SquareRoot") {
    unaryOpTest<float, double>(
        popops::sqrt,
        [](float x) -> double {
          double xd = static_cast<double>(x);
          double res = std::sqrt(xd);
          return res;
        },
        true /* positive inputs */);
  } else if (test == "Sigmoid") {
    unaryOpTest<float, double>(popops::sigmoid, [](float x) -> double {
      double xd = static_cast<double>(x);
      double res = (1.0 / (1.0 + std::exp(-xd)));
      return res;
    });
  } else if (test == "Rsqrt") {
    unaryOpTest<float, double>(popops::rsqrt, [](float x) -> double {
      double xd = static_cast<double>(x);
      double res = 1.0 / std::sqrt(xd);
      return res;
    });
  } else if (test == "SubtractFloat") {
    binaryOpTest<float, double>(static_cast<BinaryOpFnPtr_t>(popops::sub),
                                [](float x, float y) -> double {
                                  return static_cast<double>(x) -
                                         static_cast<double>(y);
                                });
  } else if (test == "SubtractHalf") {
    binaryOpTestHalf(static_cast<BinaryOpFnPtr_t>(popops::sub),
                     [](float x, float y) -> float { return x - y; });
  } else if (test == "SubtractInt") {
    binaryOpTest<int, int>(static_cast<BinaryOpFnPtr_t>(popops::sub),
                           [](int x, int y) -> int { return (x - y); });
  } else if (test == "RoundFloat") {
    unaryOpTest<float, double>(popops::round,
                               [](float x) -> double { return std::round(x); });
  } else if (test == "SelectFloat") {
    selectTest<float>();
  } else if (test == "SelectFloatLHSConst") {
    selectTestFloatLHSConst();
  } else if (test == "SelectFloatRHSConst") {
    selectTestFloatRHSConst();
  } else if (test == "SelectFloatLHSAndRHSConst") {
    selectTestFloatLHSAndRHSConst();
  } else if (test == "SelectHalfLHSAndRHSConst") {
    selectTestHalfLHSAndRHSConst();
  } else if (test == "SelectInt") {
    selectTest<int>();
  } else if (test == "BroadcastSelectorSelectInt") {
    broadcastSelectorSelectTest<int>(false);
  } else if (test == "BroadcastSelectorSelectFloat") {
    broadcastSelectorSelectTest<float>(false);
  } else if (test == "BroadcastSelectorSelectInPlaceInt") {
    broadcastSelectorSelectTest<int>(true);
  } else if (test == "BroadcastSelectorSelectInPlaceFloat") {
    broadcastSelectorSelectTest<float>(true);
  } else if (test == "ClampFloat") {
    clampTest<float>(false);
  } else if (test == "ClampFloatMinConst") {
    clampTestFloatMinConst();
  } else if (test == "ClampFloatMaxConst") {
    clampTestFloatMaxConst();
  } else if (test == "ClampInt") {
    clampTest<int>(false);
  } else if (test == "ClampInPlaceFloat") {
    clampTest<float>(true);
  } else if (test == "BroadcastClampInt") {
    broadcastClampTest<int>(false);
  } else if (test == "BroadcastClampInPlaceInt") {
    broadcastClampTest<int>(true);
  } else if (test == "BroadcastClampFloat") {
    broadcastClampTest<float>(false);
  } else if (test == "BroadcastClampInPlaceFloat") {
    broadcastClampTest<float>(true);
  } else if (test == "BroadcastClampSingleElementSrcFloat") {
    broadcastClampTest<float>(false, 1);
  } else if (test == "BinaryOutputMapChoice") {
    binaryOutputMapChoiceTest();
  } else if (test == "TrinaryOutputMapChoice") {
    trinaryOutputMapChoiceTest();
  } else if (test == "AllTrueBad") {
    allTrueBadTest();
  } else if (test == "AllTrue") {
    allTrueTest();
  } else if (test == "IsFinite") {
    isFiniteTest();
  } else if (test == "Map") {
    mapTest();
  } else if (test == "MapCast") {
    mapTestCast(false, FLOAT, HALF);
  } else if (test == "MapCastIntToFloat") {
    mapTestCastIntToFloat();
  } else if (test == "MapCastInPlace") {
    mapTestCast(true, HALF, FLOAT);
  } else if (test == "MapMultiTensor") {
    mapTestMultiTensor();
  } else if (test == "MapInPlace") {
    mapInPlaceTest();
  } else if (test == "MapInPlaceBroadcast") {
    mapInPlaceBroadcastTest();
  } else if (test == "MapInferType") {
    mapInferTypeTest();
  } else if (test == "MapInferTypeNot") {
    unaryMapExprTest<bool, bool>(Equal(Const(0), Not(_1)),
                                 [](bool x) -> bool { return false == !x; });
  } else if (test == "MapInferTypeEqual") {
    binaryMapExprTest<bool, bool>(
        Equal(Const(0), Equal(_1, _2)),
        [](bool x, bool y) -> bool { return false == (x == y); });
  } else if (test == "MapInferTypeCast") {
    unaryMapExprTest<float, float>(Cast(Add(_1, Const(1)), FLOAT),
                                   [](float x) -> float { return x + 1; });
  } else if (test == "AddInPlace") {
    addInPlaceTest();
  } else if (test == "BinaryConcat") {
    binaryConcatTest();
  } else if (test == "UnaryConcat") {
    unaryConcatTest();
  } else if (test == "MultiplyFloatInPlaceConstScalarTest") {
    multiplyFloatInPlaceConstScalarTest();
  } else if (test == "AddHalfConstScalarTest") {
    addHalfConstScalarTest();
  } else {
    throw std::runtime_error("Unknown test '" + test + "'");
  }

  return 0;
}
