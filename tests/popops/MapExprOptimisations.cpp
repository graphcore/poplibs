// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MapExprOptimisations

#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <sstream>
#include <utility>
#include <vector>

using namespace poplibs_support;
using namespace popops;
using namespace popops::expr;

#define HALF_TOLERANCE .1
#define FLOAT_TOLERANCE .001

std::pair<std::vector<float>, std::vector<float>>
executeExpr(const Expr &expression, const poplar::Type &dType, bool inPlace,
            bool enableFusedCodelets = true) {
  auto device = createTestDevice(TEST_TARGET);
  const auto &target = device.getTarget();
  poplar::Graph g(device.getTarget());
  popops::addCodelets(g);
  unsigned size = 16;

  auto a = g.addVariable(dType, {size}, "inputa");
  g.setTileMapping(a, 0);
  auto b = g.addVariable(dType, {size}, "inputb");
  g.setTileMapping(b, 0);
  auto c = g.addVariable(dType, {size}, "inputc");
  g.setTileMapping(c, 0);

  auto rawBufSize = target.getTypeSize(dType) * size;

  std::vector<char> rawInA(rawBufSize);
  std::vector<char> rawInB(rawBufSize);
  std::vector<char> rawInC(rawBufSize);
  std::vector<char> rawOutOpt(rawBufSize);
  std::vector<char> rawOutNoOpt(rawBufSize);

  const std::vector<float> hostInA = {1,    2,    3,   4,   5,   6,    7,    8,
                                      1.25, 2.25, 3.5, 4.5, 5.5, 6.75, 6.75, 8};

  const std::vector<float> hostInC = hostInA;

  const std::vector<float> hostInB = {
      8.25, 7.25, 6.5, 5.5, 4.5, 3.5, 2.75, 1.75, 8, 7, 6, 5, 4, 3, 2, 1};

  poplibs_test::util::copy(target, hostInA.data(), size, dType,
                           reinterpret_cast<void *>(rawInA.data()));
  poplibs_test::util::copy(target, hostInC.data(), size, dType,
                           reinterpret_cast<void *>(rawInC.data()));

  poplar::OptionFlags enableOptims = {
      {"enableExpressionOptimizations", "true"}};
  poplar::OptionFlags disableOptims = {
      {"enableExpressionOptimizations", "false"}};

  enableOptims.set("enableGenerateCodelet",
                   enableFusedCodelets ? "true" : "false");

  poplar::program::Sequence progOpt;
  poplar::Tensor tOpt;
  if (!inPlace) {
    tOpt =
        popops::map(g, expression, {a, b}, progOpt, "test1Opt", enableOptims);
  } else {
    popops::mapInPlace(g, expression, {a, b}, progOpt, "test1Opt",
                       enableOptims);
  }

  poplar::program::Sequence progNoOpt;
  poplar::Tensor tNoOpt;
  if (!inPlace) {
    tNoOpt = popops::map(g, expression, {c, b}, progNoOpt, "test1NoOpt",
                         disableOptims);
  } else {
    popops::mapInPlace(g, expression, {c, b}, progNoOpt, "test1NoOpt",
                       disableOptims);
  }
  g.createHostWrite("inA", a);
  g.createHostWrite("inC", c);
  g.createHostWrite("inB", b);
  if (!inPlace) {
    g.createHostRead("tOpt", tOpt);
    g.createHostRead("tNoOpt", tNoOpt);
  } else {
    g.createHostRead("tOpt", a);
    g.createHostRead("tNoOpt", c);
  }

  poplar::program::Sequence controlProg(std::move(progOpt),
                                        std::move(progNoOpt));

  poplar::Engine e(g, controlProg);
  device.bind([&](const poplar::Device &d) {
    e.load(d);
    e.writeTensor("inA", rawInA.data(), rawInA.data() + rawInA.size());
    e.writeTensor("inC", rawInC.data(), rawInC.data() + rawInC.size());
    e.writeTensor("inB", rawInB.data(), rawInB.data() + rawInB.size());
    e.run();
    e.readTensor("tOpt", rawOutOpt.data(), rawOutOpt.data() + rawOutOpt.size());
    e.readTensor("tNoOpt", rawOutNoOpt.data(),
                 rawOutNoOpt.data() + rawOutNoOpt.size());
  });

  std::vector<float> hostOutOpt(size);
  poplibs_test::util::copy(target, dType,
                           reinterpret_cast<void *>(rawOutOpt.data()),
                           hostOutOpt.data(), size);
  std::vector<float> hostOutNoOpt(size);
  poplibs_test::util::copy(target, dType,
                           reinterpret_cast<void *>(rawOutNoOpt.data()),
                           hostOutNoOpt.data(), size);

  return std::make_pair(hostOutOpt, hostOutNoOpt);
}

BOOST_AUTO_TEST_CASE(Pow_cast_half) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(1.0f));
  auto p = executeExpr(e1, poplar::HALF, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_cast_float) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(1.0f));
  auto p = executeExpr(e1, poplar::FLOAT, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_float) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::FLOAT, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_isqrt_half) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(-0.5f));
  auto p = executeExpr(e1, poplar::HALF, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_isqrt_float) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(-0.5f));
  auto p = executeExpr(e1, poplar::FLOAT, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_inv_half) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(-1.0f));
  auto p = executeExpr(e1, poplar::HALF, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_inv_float) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(-1.0f));
  auto p = executeExpr(e1, poplar::FLOAT, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sq_half) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(2.0f));
  auto p = executeExpr(e1, poplar::HALF, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sq_float) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(2.0f));
  auto p = executeExpr(e1, poplar::FLOAT, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sq_float_1) {
  auto e1 = popops::expr::Add(
      Add(Add(Pow(_1, Const(2)), _2), Sub(_2, Const(-1))), _2);
  auto p = executeExpr(e1, poplar::FLOAT, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sq_float_1_inplace) {
  auto e1 = popops::expr::Add(
      Add(Add(Pow(_1, Const(2)), _2), Sub(_2, Const(-1))), _2);
  auto p = executeExpr(e1, poplar::FLOAT, true);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_cast_float_inplace) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(1.0f));
  auto p = executeExpr(e1, poplar::FLOAT, true);
  BOOST_CHECK_EQUAL_COLLECTIONS(p.first.begin(), p.first.end(),
                                p.second.begin(), p.second.end());
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half_inplace) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, true);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half_disable_fusedcodelets) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, false, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}

BOOST_AUTO_TEST_CASE(Pow_sqrt_half_inplace_disable_fusedcodelets) {
  auto e1 = popops::expr::Pow(Add(Add(Add(_1, _2), _2), Sub(_2, Const(-1))),
                              Const(0.5f));
  auto p = executeExpr(e1, poplar::HALF, true, false);
  for (unsigned i = 0; i != p.first.size(); ++i) {
    BOOST_CHECK_CLOSE(p.first[i], p.second[i], HALF_TOLERANCE);
  }
}
