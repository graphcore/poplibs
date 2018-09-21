#include <popops/AllTrue.hpp>
#include <poputil/exceptions.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <limits>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <boost/random.hpp>
#include "TestDevice.hpp"
#include <poplibs_test/Util.hpp>
#include <stdexcept>
#include <string>
#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_test::util;
namespace br = boost::random;

static DeviceType deviceType;

const poplar::OptionFlags options {
  {"target.textSectionSizeInBytes", "0x9000"},
  {"target.workerStackSizeInBytes", "0x1000"}
};

#define DIM_SIZE  3

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CHECK(pred) \
  do { \
    if (!(pred)) { \
      throw std::runtime_error( \
        "failed check '" #pred "' on line " TOSTRING(__LINE__)); \
    } \
  } while (false)

#define CHECK_CLOSE(a, b) \
  do { \
    if (!checkIsClose<typename std::common_type<decltype(a), \
        decltype(b)>::type>(a, b, 0.01)) { \
      throw std::runtime_error("failed close check on line " \
        TOSTRING(__LINE__)); \
    } \
  } while (false)

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
  std::mt19937 randomEngine;
  br::uniform_int_distribution<> dist(0, 1);
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn[r][c] = dist(randomEngine);
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
  auto device = createTestDevice(deviceType);
  Graph graph(device);
  popops::addCodelets(graph);

  auto in = mapUnaryOpTensor(graph, equivalent_device_type<T>().value);
  auto prog = Sequence();

  auto out = op(graph, in, prog, "unaryOp", {});
  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  Engine eng(graph, prog, options);
  eng.load(device);

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
      CHECK_CLOSE(res, hOut[i][j]);
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
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
  T hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  OutT hOut[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);

  if (deviceType == DeviceType::IpuModel) {
    eng.printSummary(std::cout, {{"doLayerWiseBreakdown", "true"}});
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
  Graph graph(device);
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
  eng.load(device);
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
      CHECK_CLOSE(static_cast<float>(hOut[i][j]), res);
    }
  }
}

void powTest() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
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
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void selectTestFloat() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      float res = hIn3[i][j] ? hIn1[i][j] : hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void selectTestInt() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.run();
  eng.readTensor("out", hOut);
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      int res = hIn3[i][j] ? hIn1[i][j] : hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void clampTestFloat() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
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

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void clampTestInt() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
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

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void clampInPlaceTestFloat() {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE];
  setUnaryOpInput(hIn1);
  float hIn2[DIM_SIZE][DIM_SIZE], hIn3[DIM_SIZE][DIM_SIZE];

  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      hIn2[r][c] = -10;
      hIn3[r][c] = 10;
    }
  }

  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);
  Tensor in3 = mapUnaryOpTensor(graph, FLOAT);

  auto prog = Sequence();
  clampInPlace(graph, in1, in2, in3, prog);
  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);
  graph.createHostRead("out", in1);

  float hOut[DIM_SIZE][DIM_SIZE];

  Engine eng(graph, prog, options);
  eng.load(device);
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

      CHECK_CLOSE(hOut[i][j], res);
    }
  }
}

void binaryOutputMapChoiceTest() {
  auto device = createTestDevice(deviceType, 1, 4);
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
  Graph graph(device);
  popops::addCodelets(graph);

  Tensor in = graph.addVariable(FLOAT, {2, 2}, "t1");
  auto prog = Sequence();

  bool throws = false;
  try {
    allTrue(graph, in, prog, "all_true");
  } catch(const poplib_error &) {
    throws = true;
  }
  CHECK(throws);
}

void allTrueTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device);
  popops::addCodelets(graph);

  Tensor in = graph.addVariable(INT, {2}, "t1");
  Tensor ones = graph.addConstant(INT, {2}, 1);
  Tensor zeros = graph.addConstant(INT, {2}, 0);

  graph.setTileMapping(in, 0);

  auto bodyProg = Sequence();
  subInPlace(graph, in, ones, bodyProg);

  auto condProg = Sequence();
  Tensor neZero = neq(graph, in, zeros, condProg);
  auto predicate = allTrue(graph, neZero, condProg, "all_true");

  int init[2] = {10, 8};
  int output[2] = {0, 0};

  auto mainProg = Sequence();
  mainProg.add(RepeatWhileTrue(condProg, predicate, bodyProg));
  graph.createHostWrite("in", in);
  graph.createHostRead("out", in);


  Engine eng(graph, mainProg, options);
  eng.load(device);
  eng.writeTensor("in", init);
  eng.run();
  eng.readTensor("out", output);

  CHECK(output[0] == 2);
  CHECK(output[1] == 0);
}

void isFiniteTest() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      bool expected = !(i<=1 && j<=1);
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

using namespace popops::expr;

void mapTest() {
  auto device = createTestDevice(deviceType);
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

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);

  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
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
  br::uniform_real_distribution<> dist(-10.0, +10.0);
  for (unsigned i = 0; i < DIM_SIZE; ++i) {
    aIn[i] = dist(randomEngine);
    bIn[i] = dist(randomEngine);
    cIn[i] = dist(randomEngine);
  }

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("a", aIn);
  eng.writeTensor("b", bIn);
  eng.writeTensor("c", cIn);
  eng.run();
  eng.readTensor("out", hOut);


  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    auto expected = (aIn[i] >= bIn[i]) ? cIn[i] + bIn[i] : aIn[i] + bIn[i];
    CHECK_CLOSE(hOut[i], expected);
  }
}

void mapInPlaceTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device);
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
  eng.load(device);
  eng.writeTensor("in", hIn);
  eng.run();
  eng.readTensor("out", hOut);


  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = fabs(hIn[i][j]) + hIn[i][j];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void mapInferTypeTest() {
  auto device = createTestDevice(deviceType);
  Graph graph(device);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto a = graph.addVariable(FLOAT, {DIM_SIZE}, "a");
  mapTensorLinearly(graph, a);
  auto b = graph.addVariable(FLOAT, {DIM_SIZE}, "b");
  mapTensorLinearly(graph, b);
  auto c = graph.addVariable(BOOL, {DIM_SIZE}, "c");
  mapTensorLinearly(graph, c);
  auto out = map(graph, And(Equal(_1, _2), _3),
                 {a, b, c}, prog);

  graph.createHostWrite("a", a);
  graph.createHostWrite("b", b);
  graph.createHostWrite("c", c);
  graph.createHostRead("out", out);

  float aIn[DIM_SIZE],
        bIn[DIM_SIZE];
  bool  cIn[DIM_SIZE];
  bool  hOut[DIM_SIZE];

  std::mt19937 randomEngine;
  br::uniform_real_distribution<> dist(-10.0, +10.0);
  for (unsigned i = 0; i < DIM_SIZE; ++i) {
    aIn[i] = dist(randomEngine);
    bIn[i] = dist(randomEngine);
    cIn[i] = dist(randomEngine) > 0.0;
  }

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("a", aIn);
  eng.writeTensor("b", bIn);
  eng.writeTensor("c", cIn);
  eng.run();
  eng.readTensor("out", hOut);

  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
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
  Graph graph(device);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t1 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  auto t2 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in2");

  mapTensorLinearly(graph, t1);
  mapTensorLinearly(graph, t2);
  addInPlace(graph, t1, t2,  prog);

  graph.createHostWrite("in1", t1);
  graph.createHostWrite("in2", t2);
  graph.createHostRead("out",  t1);

  float hIn1[DIM_SIZE][DIM_SIZE];
  float hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][DIM_SIZE];

  setUnaryOpInput(hIn1);
  setUnaryOpInput(hIn2);

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);


  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
  }


  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      auto expected = hIn1[i][j] + hIn2[i][j];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void binaryConcatTest() {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
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
  auto t5 = add(graph, concat(t1, t2, 1), concat(t3, t4, 1),  prog);

  graph.createHostWrite("in1", t1);
  graph.createHostWrite("in2", t2);
  graph.createHostWrite("in3", t3);
  graph.createHostWrite("in4", t4);
  graph.createHostRead("out",  t5);

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
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.writeTensor("in3", hIn3);
  eng.writeTensor("in4", hIn4);
  eng.run();
  eng.readTensor("out", hOut);


  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < 2 * DIM_SIZE; ++j) {
      auto expected =
          j < DIM_SIZE ? hIn1[i][j] + hIn3[i][j] :
                         hIn2[i][j - DIM_SIZE] + hIn4[i][j - DIM_SIZE];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

void unaryConcatTest() {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  auto prog = Sequence();
  auto t1 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  auto t2 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in2");

  mapTensorLinearly(graph, t1);
  mapTensorLinearly(graph, t2);
  auto t3 = neg(graph, concat(t1, t2, 1), prog);

  graph.createHostWrite("in1", t1);
  graph.createHostWrite("in2", t2);
  graph.createHostRead("out",  t3);

  float hIn1[DIM_SIZE][DIM_SIZE];
  float hIn2[DIM_SIZE][DIM_SIZE];
  float hOut[DIM_SIZE][2 * DIM_SIZE];

  setUnaryOpInput(hIn1);
  setUnaryOpInput(hIn2);

  Engine eng(graph, prog, options);
  eng.load(device);
  eng.writeTensor("in1", hIn1);
  eng.writeTensor("in2", hIn2);
  eng.run();
  eng.readTensor("out", hOut);


  if (deviceType == DeviceType::IpuModel) {
    auto execReport = eng.getExecutionReport({
      { "doLayerWiseBreakdown", "true" }
    });
    execReport.printSummary(std::cerr);
  }

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < 2 * DIM_SIZE; ++j) {
      auto expected =
          j < DIM_SIZE ? -hIn1[i][j] : -hIn2[i][j - DIM_SIZE];
      CHECK_CLOSE(hOut[i][j], expected);
    }
  }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  std::string test;

  po::options_description desc("Options");
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("test",
     po::value<std::string>(&test)->required(),
     "The test to run");

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
    unaryOpTest<float, double>(popops::abs,
                           [](float x) -> double {
                              double res = fabs(static_cast<double>(x));
                              return res;
                           });
  } else if (test == "AbsInt") {
    unaryOpTest<int, int>(popops::abs, [](int x) -> int { return std::abs(x);});
  } else if (test == "AddFloat") {
    binaryOpTest<float, double>(popops::add,
                              [](float x, float y) -> double {
                                 double res = x + y;
                                 return res;
                              });
  } else if (test == "Atan2Float") {
    binaryOpTest<float, double>(popops::atan2,
                              [](float x, float y) -> double {
                                double res = std::atan2(x, y);
                                return res;
                              });
  } else if (test == "AddInt") {
    binaryOpTest<int,int>(popops::add, [](int x, int y) -> int {return x + y;});
  } else if (test == "BitwiseAndInt") {
    binaryOpTest<int, int>(popops::bitwiseAnd,
                         [](int x, int y) -> int {return x & y;});
  } else if (test == "BitwiseOrInt") {
    binaryOpTest<int, int>(popops::bitwiseOr,
                         [](int x, int y) -> int {return x | y;});
  } else if (test == "BitwiseNotInt") {
    unaryOpTest<int, int>(popops::bitwiseNot,
                        [](int x) -> int { return ~x;});
  } else if (test == "Ceil") {
    unaryOpTest<float, double>(popops::ceil,
                             [](float x) -> double {
                                double res = std::ceil(static_cast<double>(x));
                                return res;
                             });

  } else if (test == "Cos") {
    unaryOpTest<float, double>(popops::cos,
                             [](float x) -> double {
                                double res = std::cos(static_cast<double>(x));
                                return res;
                             });
  } else if (test == "CountLeadingZeros") {
    unaryOpTest<int, int>(popops::countLeadingZeros,
                        [](int x) -> int { return x ? __builtin_clz(x) : 0; });
  } else if (test == "DivideInt") {
    binaryOpTest<float, double>(popops::div,
                              [](float x, float y) -> double {
                                 double res = x / y;
                                 return res;
                              });
  } else if (test == "EqualFloat") {
    binaryOpTest<int, int>(popops::div,
                         [](int x, int y) -> int {
                            int res = x / y;
                           return res;
                         });
  } else if (test == "EqualBool") {
    binaryOpTest<float, bool, bool>(
      popops::eq,
      [](float x, float y) -> bool {
         return x == y;
      });
  } else if (test == "GreaterThanBool") {
    binaryOpTest<bool, bool, bool>(
      popops::gt,
      [](bool x, bool y) -> bool {
         return x > y;
      });
  } else if (test == "GreaterThanEqualBool") {
    binaryOpTest<bool, bool, bool>(
      popops::gteq,
      [](bool x, bool y) -> bool {
         return x >= y;
      });
  } else if (test == "LessThanBool") {
    binaryOpTest<bool, bool, bool>(
      popops::lt,
      [](bool x, bool y) -> bool {
         return x < y;
      });
  } else if (test == "LessThanEqualBool") {
    binaryOpTest<bool, bool, bool>(
      popops::lteq,
      [](bool x, bool y) -> bool {
         return x <= y;
      });
  } else if (test == "Exponent") {
    unaryOpTest<float, float>(popops::exp,
                           [](float x) -> float {
                              double res = std::exp(static_cast<double>(x));
                              return res;
                           });
  } else if (test == "ExponentMinus1") {
    unaryOpTest<float, float>(popops::expm1,
                           [](float x) -> float {
                              double res = std::expm1(static_cast<double>(x));
                              return res;
                           });
  } else if (test == "Floor") {
    unaryOpTest<float, double>(popops::floor,
                             [](float x) -> double {
                                double res = std::floor(static_cast<double>(x));
                                return res;
                             });
  } else if (test == "GreaterThanFloat") {
    binaryOpTest<float, bool, bool>(
      popops::gt,
      [](float x, float y) -> bool {
         return x > y;
      });
  } else if (test == "GreaterThanInt") {
    binaryOpTest<int, bool, bool>(
      popops::gt,
      [](int x, int y) -> bool {
         return x > y;
      });
  } else if (test == "GreaterThanEqualFloat") {
    binaryOpTest<float, bool, bool>(
      popops::gteq,
      [](float x, float y) -> bool {
         return x >= y;
      });
  } else if (test == "LessThanFloat") {
    binaryOpTest<float, bool, bool>(
      popops::lt,
      [](float x, float y) -> bool {
         return x < y;
      });
  } else if (test == "LessThanEqualFloat") {
    binaryOpTest<float, bool, bool>(
      popops::lteq,
      [](float x, float y) -> bool {
         return x <= y;
      });
  } else if (test == "Logarithm") {
    unaryOpTest<float, double>(popops::log,
                             [](float x) -> double {
                                double res = std::log(static_cast<double>(x));
                                return res;
                             },
                             true /* positive inputs */);
  } else if (test == "Logarithm1Plus") {
    unaryOpTest<float, double>(popops::log1p,
                             [](float x) -> double {
                                double res = std::log1p(static_cast<double>(x));
                                return res;
                             },
                             true /* positive inputs */);
  } else if (test == "LogicalAnd") {
    binaryOpTest<bool, bool, bool>(
      popops::logicalAnd,
      [](bool x, bool y) -> bool {
         return x && y;
      });
  } else if (test == "LogicalNot") {
    unaryOpTest<bool, bool>(popops::logicalNot,
                         [](bool x) -> bool { return !x; });
  } else if (test == "LogicalOr") {
    binaryOpTest<bool, bool, bool>(
      popops::logicalOr,
      [](bool x, bool y) -> bool {
         return x || y;
      });
  } else if (test == "MaxFloat") {
    binaryOpTest<float, double>(popops::max,
                              [](float x, float y) -> double {
                                 double res = std::max(x, y);
                                 return res;
                              });
  } else if (test == "MaxInt") {
    binaryOpTest<int, int>(popops::max,
                         [](int x, int y) -> int {
                            auto res = std::max(x, y);
                            return res;
                         });
  } else if (test == "MinFloat") {
    binaryOpTest<float, double>(popops::min,
                              [](float x, float y) -> double {
                                 double res = std::min(x, y);
                                 return res;
                              });
  } else if (test == "MinInt") {
    binaryOpTest<int, int>(popops::min,
                         [](int x, int y) -> int {
                            auto res = std::min(x, y);
                            return res;
                         });
  } else if (test == "Multiply") {
    binaryOpTest<float, double>(popops::mul,
                              [](float x, float y) -> double {
                                 double res = x * y;
                                 return res;
                              });
  } else if (test == "NotEqualFloat") {
    binaryOpTest<float, bool, bool>(
      popops::neq,
      [](float x, float y) -> bool {
         return x != y;
      });
  } else if (test == "NotEqualBool") {
    binaryOpTest<bool, bool, bool>(
      popops::neq,
      [](bool x, bool y) -> bool {
         return x != y;
      });
  } else if (test == "NegateFloat") {
    unaryOpTest<float, double>(popops::neg,
                             [](float x) -> double {
                                return -x;
                             });
  } else if (test == "NegateInt") {
    unaryOpTest<int, int>(popops::neg,
                             [](int x) -> int {
                                return -x;
                             });
  } else if (test == "Popcount") {
    unaryOpTest<int, int>(popops::popcount,
                        [](int x) -> int { return __builtin_popcount(x); });
  } else if (test == "Power") {
    powTest();
  } else if (test == "RemainderFloat") {
    binaryOpTest<float, double>(popops::rem,
                              [](float x, float y) -> double {
                                double res = std::fmod(static_cast<double>(x),
                                                       static_cast<double>(y));
                                return res;
                              });
  } else if (test == "RemainderInt") {
    binaryOpTest<int, int>(popops::rem,
                         [](int x, int y) -> double {
                            return x % y;
                         });
  } else if (test == "ShiftLeftInt") {
    binaryOpTest<int, int, int>(popops::shiftLeft,
                              [](int x, int y) -> int {
                                 return x << y;
                              });
  } else if (test == "ShiftRightInt") {
    binaryOpTest<int, int, int>(popops::shiftRight,
                              [](int x, int y) -> int {
                                return (unsigned)x >> y;
                              });
  } else if (test == "ShiftRightSignExtendInt") {
    binaryOpTest<int, int, int>(popops::shiftRightSignExtend,
                              [](int x, int y) -> int {
                                return x >> y;
                              });
  } else if (test == "SignumFloat") {
    unaryOpTest<float, double>(popops::signum,
                             [](float x) -> double {
                                double res = (0 < x) - (x < 0);
                                return res;
                             });
  } else if (test == "SignumInt") {
    unaryOpTest<int, int>(popops::signum,
                             [](int x) -> int {
                                int res = (0 < x) - (x < 0);
                                return res;
                             });
  } else if (test == "Sin") {
    unaryOpTest<float, double>(popops::sin,
                             [](float x) -> double {
                                double res = std::sin(static_cast<double>(x));
                                return res;
                             });
  } else if (test == "Tanh") {
    unaryOpTest<float, double>(popops::tanh,
                             [](float x) -> double {
                                double res = std::tanh(static_cast<double>(x));
                                return res;
                             });
  } else if (test == "Square") {
    unaryOpTest<float, double>(popops::square,
                             [](float x) -> double {
                                double xd = static_cast<double>(x);
                                double res = xd * xd;
                                return res;
                             });
  } else if (test == "SquareRoot") {
    unaryOpTest<float, double>(popops::sqrt,
                             [](float x) -> double {
                                double xd = static_cast<double>(x);
                                double res = std::sqrt(xd);
                                return res;
                             },
                             true /* positive inputs */);
  } else if (test == "SubtractFloat") {
    binaryOpTest<float, double>(
      popops::sub,
      [](float x, float y) -> double {
         return static_cast<double>(x) - static_cast<double>(y);
      });
  } else if (test == "SubtractHalf") {
    binaryOpTestHalf(
      popops::sub,
      [](float x, float y) -> float {
        return x - y;
      });
  } else if (test == "SubtractInt") {
    binaryOpTest<int, int>(
      popops::sub,
      [](int x, int y) -> int {
        return (x - y);
      });
  } else if (test == "RoundFloat") {
    unaryOpTest<float, double>(popops::round,
      [](float x) -> double {
        return std::round(x);
      });
  } else if (test == "SelectFloat") {
    selectTestFloat();
  } else if (test == "SelectInt") {
    selectTestInt();
  } else if (test == "ClampFloat") {
    clampTestFloat();
  } else if (test == "ClampInt") {
    clampTestInt();
  } else if (test == "ClampInPlaceFloat") {
    clampInPlaceTestFloat();
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
  } else if (test == "MapMultiTensor") {
    mapTestMultiTensor();
  } else if (test == "MapInPlace") {
    mapInPlaceTest();
  } else if (test == "MapInferType") {
    mapInferTypeTest();
  } else if (test == "AddInPlace") {
    addInPlaceTest();
  } else if (test == "BinaryConcat") {
    binaryConcatTest();
  } else if (test == "UnaryConcat") {
    unaryConcatTest();
  } else {
    throw std::runtime_error("Unknown test '" + test + "'");
  }

  return 0;
}
