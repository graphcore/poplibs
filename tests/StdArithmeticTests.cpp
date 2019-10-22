#define BOOST_TEST_MODULE StdArithmeticTests

#include "TestDevice.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "popops/ElementWise.hpp"
#include <boost/test/unit_test.hpp>
#include <cstring>
#include <iostream>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/Cast.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define DIM_SIZE 10

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
static void setBroadcastOpInputs(float hIn1[DIM_SIZE][DIM_SIZE]) {
  float val1 = -100;
  for (auto r = 0; r != DIM_SIZE; ++r) {
    for (auto c = 0; c != DIM_SIZE; ++c) {
      int sign1 = (1 - 2 * ((c + 1) & 1));
      hIn1[r][c] = (val1 + (r * DIM_SIZE + c) * 1) * sign1;
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdBroadcastAdd_float,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setBroadcastOpInputs(hIn);

  float k = 2;
  auto B = graph.addVariable(FLOAT, {});
  graph.setInitialValue(B, k);
  auto in = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in);
  mapTensorLinearly(graph, B);

  graph.createHostWrite("in", in);
  graph.createHostRead("out", in);
  auto prog = Sequence();

  addInPlace(graph, in, B, prog);
  Engine eng(graph, prog);
  float hOut[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn[i][j] + k;
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdBroadcastMultiply_float,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setBroadcastOpInputs(hIn);

  float k = 2;
  auto B = graph.addVariable(FLOAT, {});
  graph.setInitialValue(B, k);
  auto in = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in);
  mapTensorLinearly(graph, B);

  graph.createHostWrite("in", in);
  graph.createHostRead("out", in);
  auto prog = Sequence();

  mulInPlace(graph, in, B, prog);
  Engine eng(graph, prog);
  float hOut[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn[i][j] * k;
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdBroadcastSubtract_half,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hIn[DIM_SIZE][DIM_SIZE];
  setBroadcastOpInputs(hIn);

  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawIn(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hIn[0][0], rawIn.data(),
                                DIM_SIZE * DIM_SIZE);

  float k = 2;
  auto B = graph.addVariable(HALF, {});
  graph.setInitialValue(B, k);
  auto in = graph.addVariable(HALF, {DIM_SIZE, DIM_SIZE}, "in1");
  mapTensorLinearly(graph, in);
  mapTensorLinearly(graph, B);

  std::vector<char> rawOut(rawBufSize);
  graph.createHostWrite("in", in);
  graph.createHostRead("out", in);
  auto prog = Sequence();

  subInPlace(graph, in, B, prog);
  Engine eng(graph, prog);

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
  });

  float hOut[DIM_SIZE][DIM_SIZE];
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                DIM_SIZE * DIM_SIZE);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn[i][j] - k;
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdAddTo_half_float_tensor,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(1.4)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(1.4))) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  float k = 2;
  auto factor = graph.addVariable(HALF, {});
  graph.setInitialValue(factor, k);

  auto in1 = graph.addVariable(HALF, {DIM_SIZE, DIM_SIZE}, "in1");
  auto in2 = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in2");
  mapTensorLinearly(graph, in1);
  mapTensorLinearly(graph, in2);
  mapTensorLinearly(graph, factor);

  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawIn1(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hIn1[0][0], rawIn1.data(),
                                DIM_SIZE * DIM_SIZE);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  auto prog = Sequence();
  scaledAddTo(graph, in1, in2, factor, prog);
  Engine eng(graph, prog);

  std::vector<char> rawOut(rawBufSize);
  float hOut[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", rawIn1.data(), rawIn1.data() + rawIn1.size());
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
  });

  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                DIM_SIZE * DIM_SIZE);

  // Check result
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] + k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdAddTo_half_scale_float_tensor_const,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.1)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.1))) {
  // Avoid testing this properly in IPUModel, as half isn't accurate
  const bool isIpuModel = TEST_TARGET == DeviceType::IpuModel;
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE];
  float hIn2[DIM_SIZE];

  // Large values for the 2nd operand which, when multiplied by a very small
  // scale should have a sensible result
  for (unsigned i = 0; i < DIM_SIZE; i++) {
    hIn1[i] = 0.0;
    hIn2[i] = 10000 * (i % 4);
  }
  // Very small k to make the selection of codelets work to solve the issue
  // that it can't be represented as a half
  float k = (3.0e-9);
  auto factor = graph.addVariable(FLOAT, {});
  graph.setInitialValue(factor, k);

  auto in1 = graph.addVariable(HALF, {DIM_SIZE}, "in1");
  auto in1ConstTest = graph.addVariable(HALF, {DIM_SIZE}, "in1");
  auto in1Fails = graph.addVariable(HALF, {DIM_SIZE}, "in1");
  auto in1ConstTestFails = graph.addVariable(HALF, {DIM_SIZE}, "in1");
  auto in2 = graph.addVariable(HALF, {DIM_SIZE}, "in2");
  mapTensorLinearly(graph, in1);
  mapTensorLinearly(graph, in1ConstTest);
  mapTensorLinearly(graph, in1Fails);
  mapTensorLinearly(graph, in1ConstTestFails);
  mapTensorLinearly(graph, in2);
  mapTensorLinearly(graph, factor);

  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE;
  std::vector<char> rawIn1(rawBufSize), rawIn2(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hIn1[0], rawIn1.data(), DIM_SIZE);
  poplar::copyFloatToDeviceHalf(target, &hIn2[0], rawIn2.data(), DIM_SIZE);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in1ConstTest", in1ConstTest);
  graph.createHostWrite("in1Fails", in1Fails);
  graph.createHostWrite("in1ConstTestFails", in1ConstTestFails);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  graph.createHostRead("outConstTest", in1ConstTest);
  graph.createHostRead("outFails", in1Fails);
  graph.createHostRead("outConstTestFails", in1ConstTestFails);

  auto prog = Sequence();
  // These tests should produce a reasonable answer as the scale and multiply
  // is implemented in full precision
  scaledAddTo(graph, in1, in2, factor, prog, "Tensor test",
              {{"scaleFloatToHalfTolerance", "1e-6"}});
  scaledAddTo(graph, in1ConstTest, in2, k, prog, "Const test",
              {{"scaleFloatToHalfTolerance", "1e-6"}});

  // These tests should "fail", producing 0.0 exactly as the scale becomes
  // 0.0 in half precision.  We use the tolerance option to switch
  // automatic selection of full precision arithmetic off.
  scaledAddTo(graph, in1Fails, in2, factor, prog, "Tensor test fail",
              {{"scaleFloatToHalfTolerance", "2.0"}});
  scaledAddTo(graph, in1ConstTestFails, in2, k, prog, "Const test fail",
              {{"scaleFloatToHalfTolerance", "2.0"}});
  Engine eng(graph, prog);

  std::vector<char> rawOut(rawBufSize), rawOutConstTest(rawBufSize);
  std::vector<char> rawOutFails(rawBufSize), rawOutConstTestFails(rawBufSize);

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in1", rawIn1.data(), rawIn1.data() + rawIn1.size());
    eng.writeTensor("in1ConstTest", rawIn1.data(),
                    rawIn1.data() + rawIn1.size());
    eng.writeTensor("in1Fails", rawIn1.data(), rawIn1.data() + rawIn1.size());
    eng.writeTensor("in1ConstTestFails", rawIn1.data(),
                    rawIn1.data() + rawIn1.size());
    eng.writeTensor("in2", rawIn2.data(), rawIn2.data() + rawIn2.size());

    eng.run();

    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
    eng.readTensor("outConstTest", rawOutConstTest.data(),
                   rawOutConstTest.data() + rawOutConstTest.size());
    eng.readTensor("outFails", rawOutFails.data(),
                   rawOutFails.data() + rawOutFails.size());
    eng.readTensor("outConstTestFails", rawOutConstTestFails.data(),
                   rawOutConstTestFails.data() + rawOutConstTestFails.size());
  });

  float hOut[DIM_SIZE], hOutConstTest[DIM_SIZE];
  float hOutFails[DIM_SIZE], hOutConstTestFails[DIM_SIZE];
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0], DIM_SIZE);
  poplar::copyDeviceHalfToFloat(target, rawOutConstTest.data(),
                                &hOutConstTest[0], DIM_SIZE);
  poplar::copyDeviceHalfToFloat(target, rawOutFails.data(), &hOutFails[0],
                                DIM_SIZE);
  poplar::copyDeviceHalfToFloat(target, rawOutConstTestFails.data(),
                                &hOutConstTestFails[0], DIM_SIZE);

  // Check result
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    double res = hIn1[i] + k * hIn2[i];
    BOOST_TEST(hOut[i] == res);
    BOOST_TEST(hOutConstTest[i] == res);
    // These tests should "fail", producing 0.0 exactly.
    if (!isIpuModel) {
      BOOST_TEST(static_cast<bool>(hOutFails[i] == 0.0f));
      BOOST_TEST(static_cast<bool>(hOutConstTestFails[i] == 0.0f));
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdAddTo_float_constant,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
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
  Engine eng(graph, prog);
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
      double res = hIn1[i][j] + k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdAddTo_float_runtime_fast_path,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  float k = 2;
  // Creating a larger tensor to force a gap between allocations of the two
  // operands will result in the fast path being chosen at runtime.
  // Note TMEM_ELEMSIZE is in bytes
  const unsigned padSize = 16 + (TMEM_ELEMSIZE * 2) / 4;
  const unsigned regionSize = 2 * DIM_SIZE * DIM_SIZE + padSize;
  auto in = graph.addVariable(FLOAT, {regionSize}, "Whole input");
  graph.setTileMapping(in, 0);

  auto in1 = in.slice(0, DIM_SIZE * DIM_SIZE);
  auto in2 = in.slice(padSize, padSize + DIM_SIZE * DIM_SIZE);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  auto prog = Sequence();
  scaledAddTo(graph, in1, in2, k, prog);
  Engine eng(graph, prog);
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
      double res = hIn1[i][j] + k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdAddTo_float_tensor,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  float k = 2;
  auto factor = graph.addVariable(FLOAT, {});
  graph.setInitialValue(factor, k);
  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  auto prog = Sequence();
  scaledAddTo(graph, in1, in2, factor, prog);
  Engine eng(graph, prog);
  float hOut[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  // Check result
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] + k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdSubtractFrom_float_tensor,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hIn1[DIM_SIZE][DIM_SIZE], hIn2[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn1, hIn2);

  float k = 2;
  auto factor = graph.addVariable(FLOAT, {});
  graph.setInitialValue(factor, k);
  Tensor in1, in2;
  std::tie(in1, in2) = mapBinaryOpTensors(graph, FLOAT);

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in2", in2);
  graph.createHostRead("out", in1);
  auto prog = Sequence();
  scaledSubtractFrom(graph, in1, in2, factor, prog);
  Engine eng(graph, prog);
  float hOut[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in1", hIn1, &hIn1[DIM_SIZE]);
    eng.writeTensor("in2", hIn2, &hIn2[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  // Check result
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = hIn1[i][j] - k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdSubFrom_int,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.01)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.01))) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
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
  Engine eng(graph, prog);
  int hOut[DIM_SIZE][DIM_SIZE];

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
      double res = hIn1[i][j] - k * hIn2[i][j];
      BOOST_TEST(hOut[i][j] == res);
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdaXPlusbY_halfin_tensor_and_const,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(1)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(1))) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hInOut[DIM_SIZE][DIM_SIZE];
  float hIn[DIM_SIZE][DIM_SIZE];
  float hInOutFloat[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn, hInOut);
  std::memcpy(hInOutFloat, hInOut, sizeof(hInOutFloat));

  auto rawBufSize = target.getTypeSize(HALF) * DIM_SIZE * DIM_SIZE;
  std::vector<char> rawIn(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hIn[0][0], rawIn.data(),
                                DIM_SIZE * DIM_SIZE);
  std::vector<char> rawInOut(rawBufSize);
  poplar::copyFloatToDeviceHalf(target, &hInOut[0][0], rawInOut.data(),
                                DIM_SIZE * DIM_SIZE);

  float k = 2, k2 = 3;
  auto A = graph.addVariable(HALF, {});
  graph.setInitialValue(A, k);
  auto B = graph.addVariable(HALF, {});
  graph.setInitialValue(B, k2);
  auto inOut = graph.addVariable(HALF, {DIM_SIZE, DIM_SIZE}, "inOut");
  auto inOutFloat =
      graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "inOutFloat");
  auto inOutConstTest =
      graph.addVariable(HALF, {DIM_SIZE, DIM_SIZE}, "inOutConstTest");
  auto inOutFloatConstTest =
      graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "inOutFloatConstTest");
  auto in = graph.addVariable(HALF, {DIM_SIZE, DIM_SIZE}, "in");
  mapTensorLinearly(graph, A);
  mapTensorLinearly(graph, B);
  mapTensorLinearly(graph, inOut);
  mapTensorLinearly(graph, inOutFloat);
  mapTensorLinearly(graph, inOutConstTest);
  mapTensorLinearly(graph, inOutFloatConstTest);
  mapTensorLinearly(graph, in);

  std::vector<char> rawOut(rawBufSize);
  std::vector<char> rawOutConstTest(rawBufSize);
  graph.createHostWrite("in", in);
  graph.createHostWrite("inOut", inOut);
  graph.createHostWrite("inOutFloat", inOutFloat);
  graph.createHostRead("out", inOut);
  graph.createHostRead("outFloat", inOutFloat);
  graph.createHostRead("outConstTest", inOutConstTest);
  graph.createHostRead("outFloatConstTest", inOutFloatConstTest);

  auto prog = Sequence();

  prog.add(Copy(inOut, inOutConstTest));
  prog.add(Copy(inOutFloat, inOutFloatConstTest));
  scaledAddTo(graph, inOut, A, in, B, prog, "Debug - optimized",
              {{"optimizeForSpeed", "true"}});
  scaledAddTo(graph, inOutConstTest, -k, in, -k2, prog, "Debug - optimized",
              {{"optimizeForSpeed", "true"}});
  scaledAddTo(graph, inOutFloat, A, in, B, prog, "Float out Debug - optimized",
              {{"optimizeForSpeed", "true"}});
  scaledAddTo(graph, inOutFloatConstTest, -k, in, -k2, prog,
              "Float out Debug - optimized", {{"optimizeForSpeed", "true"}});

  Engine eng(graph, prog);

  float hOutFloat[DIM_SIZE][DIM_SIZE];
  float hOutFloatConst[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in", rawIn.data(), rawIn.data() + rawIn.size());
    eng.writeTensor("inOut", rawInOut.data(),
                    rawInOut.data() + rawInOut.size());
    eng.writeTensor("inOutFloat", hInOutFloat, &hInOutFloat[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
    eng.readTensor("outConstTest", rawOutConstTest.data(),
                   rawOutConstTest.data() + rawOutConstTest.size());
    eng.readTensor("outFloat", hOutFloat, &hOutFloat[DIM_SIZE]);
    eng.readTensor("outFloatConstTest", hOutFloatConst,
                   &hOutFloatConst[DIM_SIZE]);
  });

  float hOut[DIM_SIZE][DIM_SIZE];
  poplar::copyDeviceHalfToFloat(target, rawOut.data(), &hOut[0][0],
                                DIM_SIZE * DIM_SIZE);
  float hOutConstTest[DIM_SIZE][DIM_SIZE];
  poplar::copyDeviceHalfToFloat(target, rawOutConstTest.data(),
                                &hOutConstTest[0][0], DIM_SIZE * DIM_SIZE);

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = k * hInOut[i][j] + k2 * hIn[i][j];
      double resFloat = k * hInOutFloat[i][j] + k2 * hIn[i][j];
      BOOST_TEST(hOut[i][j] == res, "Tensor scale test");
      BOOST_TEST(hOutConstTest[i][j] == -res, "Constant scale test");
      BOOST_TEST(hOutFloat[i][j] == resFloat, "Tensor scale float out test");
      BOOST_TEST(hOutFloatConst[i][j] == -res, "Constant float out scale test");
    }
  }
}

BOOST_AUTO_TEST_CASE(
    StdaXPlusbY_float_tensor_and_const,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(1)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(1))) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hInOut[DIM_SIZE][DIM_SIZE];
  float hIn[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn, hInOut);

  float k = 2, k2 = 3;
  auto A = graph.addVariable(FLOAT, {});
  graph.setInitialValue(A, k);
  auto B = graph.addVariable(FLOAT, {});
  graph.setInitialValue(B, k2);

  auto inOut = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "inOut");
  auto inOutConstTest =
      graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "inOutConstTest");
  auto in = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in");

  mapTensorLinearly(graph, A);
  mapTensorLinearly(graph, B);
  mapTensorLinearly(graph, inOut);
  mapTensorLinearly(graph, inOutConstTest);
  mapTensorLinearly(graph, in);

  graph.createHostWrite("in", in);
  graph.createHostWrite("inOut", inOut);
  graph.createHostRead("out", inOut);
  graph.createHostRead("outConstTest", inOutConstTest);

  auto prog = Sequence();
  prog.add(Copy(inOut, inOutConstTest));

  scaledAddTo(graph, inOut, A, in, B, prog, "Debug - optimized",
              {{"optimizeForSpeed", "true"}});
  scaledAddTo(graph, inOutConstTest, -k, in, -k2, prog, "Debug - optimized",
              {{"optimizeForSpeed", "true"}});

  Engine eng(graph, prog);

  float hResult[DIM_SIZE][DIM_SIZE];
  float hResultConst[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.writeTensor("inOut", hInOut, &hInOut[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hResult, &hResult[DIM_SIZE]);
    eng.readTensor("outConstTest", hResultConst, &hResultConst[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = k * hInOut[i][j] + k2 * hIn[i][j];
      BOOST_TEST(hResult[i][j] == res, "Tensor scale test");
      BOOST_TEST(hResultConst[i][j] == -res, "Constant scale test");
    }
  }
}

BOOST_AUTO_TEST_CASE(StdCast) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  float hIn[DIM_SIZE];
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    hIn[i] = (float)i;
  }

  auto in = graph.addVariable(FLOAT, {DIM_SIZE}, "in");
  mapTensorLinearly(graph, in);
  graph.createHostWrite("in", in);

  auto prog = Sequence();

  poplar::Tensor out = cast(graph, in, INT, prog, "cast");
  graph.createHostRead("out", out);

  int hOut[DIM_SIZE];

  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hOut, &hOut[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    BOOST_TEST(hOut[i] == i);
  }
}

BOOST_AUTO_TEST_CASE(
    StdaXMinusbY_float_tensor_and_const,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(1)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(1))) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  float hInOut[DIM_SIZE][DIM_SIZE];
  float hIn[DIM_SIZE][DIM_SIZE];
  setBinaryOpInputs(hIn, hInOut);

  float k = 2, k2 = 3;
  auto A = graph.addVariable(FLOAT, {});
  graph.setInitialValue(A, k);
  auto B = graph.addVariable(FLOAT, {});
  graph.setInitialValue(B, k2);

  auto inOut = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "inOut");
  auto inOutConstTest =
      graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "inOutConstTest");
  auto in = graph.addVariable(FLOAT, {DIM_SIZE, DIM_SIZE}, "in");

  mapTensorLinearly(graph, A);
  mapTensorLinearly(graph, B);
  mapTensorLinearly(graph, inOut);
  mapTensorLinearly(graph, inOutConstTest);
  mapTensorLinearly(graph, in);

  graph.createHostWrite("in", in);
  graph.createHostWrite("inOut", inOut);
  graph.createHostRead("out", inOut);
  graph.createHostRead("outConstTest", inOutConstTest);

  auto prog = Sequence();
  prog.add(Copy(inOut, inOutConstTest));

  scaledSubtractFrom(graph, inOut, A, in, B, prog, "Debug - optimized",
                     {{"optimizeForSpeed", "true"}});
  scaledSubtractFrom(graph, inOutConstTest, k, in, k2, prog,
                     "Debug - optimized", {{"optimizeForSpeed", "true"}});

  Engine eng(graph, prog);

  float hResult[DIM_SIZE][DIM_SIZE];
  float hResultConst[DIM_SIZE][DIM_SIZE];

  device.bind([&](const Device &d) {
    eng.load(d);

    eng.writeTensor("in", hIn, &hIn[DIM_SIZE]);
    eng.writeTensor("inOut", hInOut, &hInOut[DIM_SIZE]);
    eng.run();
    eng.readTensor("out", hResult, &hResult[DIM_SIZE]);
    eng.readTensor("outConstTest", hResultConst, &hResultConst[DIM_SIZE]);
  });

  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    for (auto j = 0U; j < DIM_SIZE; ++j) {
      double res = k * hInOut[i][j] - k2 * hIn[i][j];
      BOOST_TEST(hResult[i][j] == res, "Tensor scale test");
      BOOST_TEST(hResultConst[i][j] == res, "Constant scale test");
    }
  }
}

BOOST_AUTO_TEST_CASE(
    checkAccuracyFloatHalf,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.1)) *
        utf::tolerance<double>(fpc::percent_tolerance<double>(0.1))) {
  // Avoid testing this properly in IPUModel, as half isn't accurate
  const bool isIpuModel = TEST_TARGET == DeviceType::IpuModel;
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  // Values chosen because:
  // 1.0 - simple, exact, expect 1
  // 65500 - in range, expect 1
  // 80000 - Not in range of half, expect 0
  // 3e-7 - Uses denorms so expect 0, but on IPUModel we get 1 as it is not
  //        doing half precision correctly
  // (1.0f/32768.0f) - Precise as, although denorm it is a power of 2
  float hIn[DIM_SIZE] = {1.0,  65500,  80000,  3e-7,  (1.0f / 32768.0f),
                         -1.0, -65500, -80000, -3e-7, (-1.0f / 32768.0f)};
  double tolerance[DIM_SIZE] = {1e-6, 1e-4, 1e-6, 1e-6, 1e-6,
                                1e-6, 1e-4, 1e-6, 1e-6, 1e-6};
  auto input = graph.addVariable(FLOAT, {DIM_SIZE}, "in");
  graph.setTileMapping(input, 0);

  auto prog = Sequence();
  // Some casts can cause exceptions, if the float is unrepresentable as a half.
  // The codelet should disable exceptions.  Setting them on here means that
  // we are checking that it does so.
  FloatingPointBehaviour behaviour;
  setFloatingPointBehaviour(graph, prog, behaviour, "Set Exceptions");

  auto castResult =
      checkAccuracyWhenCast(graph, input[0], HALF, tolerance[0], prog);
  auto isAccurate = castResult.first.reshape({1});
  auto out = castResult.second.reshape({1});
  for (unsigned i = 1; i < DIM_SIZE; i++) {
    auto castResult =
        checkAccuracyWhenCast(graph, input[i], HALF, tolerance[i], prog);
    isAccurate = concat(isAccurate, castResult.first.reshape({1}));
    out = concat(out, castResult.second.reshape({1}));
  }
  graph.createHostWrite("input", input);
  graph.createHostRead("out", out);
  graph.createHostRead("isAccurate", isAccurate);

  bool hIsAccurate[DIM_SIZE];
  std::vector<char> outputHalf(target.getTypeSize(HALF) * DIM_SIZE);
  float hOut[DIM_SIZE];
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("input", hIn, &hIn[DIM_SIZE]);
    eng.run();
    eng.readTensor("isAccurate", hIsAccurate, &hIsAccurate[DIM_SIZE]);
    eng.readTensor("out", outputHalf.data(),
                   outputHalf.data() + outputHalf.size());
  });
  poplar::copyDeviceHalfToFloat(target, outputHalf.data(), &hOut[0], DIM_SIZE);

  bool expected[DIM_SIZE] = {1, 1, 0, 0, 1, 1, 1, 0, 0, 1};
  bool expectedIpuModel[DIM_SIZE] = {1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
  float expectedOut[DIM_SIZE] = {
      1.0,  65504,  80000,  3e-7,  (1.0f / 32768.0f),
      -1.0, -65504, -80000, -3e-7, (-1.0f / 32768.0f)};
  float expectedOutIpuModel[DIM_SIZE] = {
      1.0,  65472,  80000,  2.98e-7,  (1.0f / 32768.0f),
      -1.0, -65472, -80000, -2.98e-7, (-1.0f / 32768.0f)};
  /* Check result */
  for (auto i = 0U; i < DIM_SIZE; ++i) {
    BOOST_TEST(hIsAccurate[i] ==
               (isIpuModel ? expectedIpuModel[i] : expected[i]));
    if (hIsAccurate[i] == 1) {
      BOOST_TEST(hOut[i] ==
                 (isIpuModel ? expectedOutIpuModel[i] : expectedOut[i]));
    }
  }
}
