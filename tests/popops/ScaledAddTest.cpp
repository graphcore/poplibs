// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ScaledAddTest

#include <boost/test/unit_test.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>

#include <vector>
using namespace poplibs_support;

BOOST_AUTO_TEST_CASE(ScaledAddTestCheckaXPlusbYInt) {
  auto device = createTestDevice(TEST_TARGET);
  const auto target = device.getTarget();
  poplar::Graph g(target);
  popops::addCodelets(g);

  auto X = g.addVariable(poplar::INT, {4}, "X");
  auto Y = g.addVariable(poplar::INT, {4}, "Y");
  g.setTileMapping(X, 0);
  g.setTileMapping(Y, 0);
  g.createHostWrite("X_write", X);
  g.createHostWrite("Y_write", Y);
  g.createHostRead("X_read", X);

  poplar::program::Sequence prog;
  poplar::setStochasticRounding(g, prog, false);
  popops::scaledAddTo(g, X, 2, Y, 1, prog, "scaledAdd");

  const std::vector<int> rawX = {1, 2, 3, 4};
  const std::vector<int> rawY = {1, 2, 3, 4};
  std::vector<int> rawOut(4);
  const std::vector<int> rawTarget = {3, 6, 9, 12};

  poplar::Engine e(g, prog);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.writeTensor("X_write", rawX.data(), rawX.data() + rawX.size());
    e.writeTensor("Y_write", rawY.data(), rawY.data() + rawY.size());
    e.run(0);
    e.readTensor("X_read", rawOut.data(), rawOut.data() + rawOut.size());
  });

  BOOST_CHECK_EQUAL_COLLECTIONS(rawOut.begin(), rawOut.end(), rawTarget.begin(),
                                rawTarget.end());
}

BOOST_AUTO_TEST_CASE(ScaledAddTestCheckEquivalenceHalf) {
  auto device = createTestDevice(TEST_TARGET);
  const auto target = device.getTarget();
  poplar::Graph g(target);
  popops::addCodelets(g);

  const unsigned size = 16U;
  auto input = g.addVariable(poplar::HALF, {size}, "input");
  auto outScaledAdd = g.addVariable(poplar::HALF, {size}, "outScaledAdd");
  auto outScaledSub = g.addVariable(poplar::HALF, {size}, "outScaledSub");
  g.setTileMapping(input, 0);
  g.setTileMapping(outScaledAdd, 0);
  g.setTileMapping(outScaledSub, 0);

  // use low enough scale compared to tolerance threshold
  const float scale = 1.33e-7;
  poplar::program::Sequence prog;
  poplar::setStochasticRounding(g, prog, false);
  popops::zero(g, outScaledAdd, prog, "scaledAdd");
  popops::zero(g, outScaledSub, prog, "scaledSub");
  poplar::OptionFlags options = {{"scaleFloatToHalfTolerance", "1e-5"}};
  popops::scaledAddTo(g, outScaledAdd, input, scale, prog, "scaledAdd",
                      options);
  popops::scaledSubtractFrom(g, outScaledSub, input, -scale, prog, "scaledSub",
                             options);

  auto rawBufSize = target.getTypeSize(poplar::HALF) * size;
  std::vector<char> rawIn(rawBufSize);
  std::vector<char> rawOutAdd(rawBufSize);
  std::vector<char> rawOutSub(rawBufSize);

  const std::vector<float> hostIn = {1e3,  2e3,  3e3,  4e3,  5e3,  6e3,
                                     7e3,  8e3,  -1e3, -2e3, -3e3, -4e3,
                                     -5e3, -6e3, -7e3, -8e3};
  poplar::copyFloatToDeviceHalf(target, hostIn.data(), rawIn.data(), size);

  g.createHostWrite("input", input);
  g.createHostRead("outScaledAddRd", outScaledAdd);
  g.createHostRead("outScaledSubRd", outScaledSub);

  poplar::Engine e(g, prog);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.writeTensor("input", rawIn.data(), rawIn.data() + rawIn.size());
    e.run();
    e.readTensor("outScaledAddRd", rawOutAdd.data(),
                 rawOutAdd.data() + rawOutAdd.size());
    e.readTensor("outScaledSubRd", rawOutSub.data(),
                 rawOutSub.data() + rawOutSub.size());
  });

  std::vector<float> hostOutAdd(size);
  poplar::copyDeviceHalfToFloat(target, rawOutAdd.data(), hostOutAdd.data(),
                                size);
  std::vector<float> hostOutSub(size);
  poplar::copyDeviceHalfToFloat(target, rawOutSub.data(), hostOutSub.data(),
                                size);
  BOOST_CHECK_EQUAL_COLLECTIONS(hostOutAdd.begin(), hostOutAdd.end(),
                                hostOutSub.begin(), hostOutSub.end());
}

void callScaledAddConst(poplar::Graph &g, poplar::Type aType,
                        poplar::Type bType, float scale, float aScale,
                        bool subtract, bool is2D, bool testSpeciality,
                        poplar::program::Sequence &prog,
                        const std::string &debugName) {
  auto A = g.addVariable(aType, {12}, "A");
  auto B = g.addVariable(bType, {12}, "B");
  g.setTileMapping(A, 0);
  g.setTileMapping(B, 0);
  auto ASlice = is2D ? concat(A.slice(0, 4), A.slice(8, 12)) : A;
  auto BSlice = is2D ? concat(B.slice(0, 4), B.slice(8, 12)) : B;
  if (aScale == 1.0f) {
    if (subtract) {
      popops::scaledSubtractFrom(g, ASlice, BSlice, scale, prog, debugName);
    } else {
      popops::scaledAddTo(g, ASlice, BSlice, scale, prog, debugName);
    }
  } else {
    if (testSpeciality) {
      popops::scaledAddTo(g, ASlice, aScale, BSlice, scale, prog,
                          popops::ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY,
                          debugName);
    } else if (subtract) {
      popops::scaledSubtractFrom(g, ASlice, aScale, BSlice, scale, prog,
                                 debugName);
    } else {
      popops::scaledAddTo(g, ASlice, aScale, BSlice, scale, prog, debugName);
    }
  }
}

BOOST_AUTO_TEST_CASE(ScaledAddGeneration) {
  // Verify that all cases of scaled add, subtract etc with const scale can be
  // created using vertices with a tensor scale
  auto device = createTestDevice(TEST_TARGET);
  const auto target = device.getTarget();
  poplar::Graph g(target);
  popops::addCodelets(g);

  poplar::program::Sequence prog;
  std::vector<poplar::Type> types = {poplar::FLOAT, poplar::HALF};
  std::vector<float> scales = {1.0, 2.0, 1.0e-6};

  // Basic scaled add and subtract support all combinations of types
  for (const auto &aType : types) {
    for (const auto &bType : types) {
      for (const auto &scale : scales) {
        callScaledAddConst(g, aType, bType, scale, 1.0f, false, false, false,
                           prog, "scaledAdd");
        callScaledAddConst(g, aType, bType, scale, 1.0f, true, false, false,
                           prog, "scaledSubtract");

        callScaledAddConst(g, aType, bType, scale, 1.0f, false, true, false,
                           prog, "scaledAdd_2D");
        callScaledAddConst(g, aType, bType, scale, 1.0f, true, true, false,
                           prog, "scaledSubtract_2D");
      }
    }
  }

  // aX + bY support all half, or all float tensors
  for (const auto &type : types) {
    for (const auto &scale : scales) {
      callScaledAddConst(g, type, type, scale, 2.0f, false, false, false, prog,
                         "aX+bY");
      callScaledAddConst(g, type, type, scale, 2.0f, true, false, false, prog,
                         "aX-bY");
      callScaledAddConst(g, type, type, scale, 2.0f, false, true, false, prog,
                         "aX+bY_2D");
      callScaledAddConst(g, type, type, scale, 2.0f, true, true, false, prog,
                         "aX-bY_2D");
    }
  }

  // The speciality X - aX + bY supports half only
  for (const auto &scale : scales) {
    callScaledAddConst(g, poplar::HALF, poplar::HALF, scale, 2.0f, false, false,
                       true, prog, "X-aX+bY");
    callScaledAddConst(g, poplar::HALF, poplar::HALF, scale, 2.0f, false, true,
                       true, prog, "X-aX+bY_2D");
  }

  poplar::Engine e(g, prog);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.run();
  });
}
