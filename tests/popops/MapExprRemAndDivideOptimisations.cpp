// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <boost/program_options.hpp>
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

static DeviceType deviceType;

std::pair<std::vector<float>, std::vector<float>>
executeExpr(const Expr &expression, const poplar::Type &dType, bool inPlace,
            bool enableFusedCodelets = true) {
  auto device = createTestDevice(TEST_TARGET);
  const auto &target = device.getTarget();
  poplar::Graph g(device.getTarget());
  popops::addCodelets(g);

  const std::vector<unsigned> hostInA = {1, 2, 3, 4, 5, 6, 7, 8};

  const std::vector<unsigned> hostInC = hostInA;

  const std::vector<unsigned> hostInB = {8, 7, 6, 5, 4, 3, 2, 1};

  unsigned size = hostInA.size();

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

  poplar::program::Sequence controlProg(
      {std::move(progOpt), std::move(progNoOpt)});

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

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  std::string test;
  poplar::Type dataType;
  bool rhsIsSigned, powerOfTwo, inPlace;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("rhs-is-signed",
     po::value<bool>(&rhsIsSigned)->required(),
     "Right hand side is signed")     
    ("power-of-two",
     po::value<bool>(&powerOfTwo)->required(),
     "right hand side is power of two")
    ("data-type",
     po::value<poplar::Type>(&dataType)->required(),
     "Data Type: unsigned, int")
    ("in-place",
     po::value<bool>(&inPlace)->required(),
     "operation is inplace")
    ("test-type",
     po::value<std::string>(&test)->required(),
     "The test to run: REMAINDER | DIVIDE ");
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

  // Only allow tests for types for which codelets are instantiated
  if (dataType != poplar::UNSIGNED_INT && dataType != poplar::INT) {
    std::cerr << "Unsupported data type\n";
    return 1;
  }

  std::unique_ptr<popops::expr::Expr> subExpr;

  if (test == "REMAINDER") {
    if (rhsIsSigned) {
      int rhsValue = powerOfTwo ? 4 : 3;
      auto e = Rem(_2, Const(rhsValue));
      subExpr = e.clone();
    } else {
      unsigned rhsValue = powerOfTwo ? 4 : 3;
      auto e = Rem(_2, Const(rhsValue));
      subExpr = e.clone();
    }
  } else if (test == "DIVIDE") {
    if (rhsIsSigned) {
      int rhsValue = powerOfTwo ? 4 : 3;
      auto e = Divide(_2, Const(rhsValue));
      subExpr = e.clone();
    } else {
      unsigned rhsValue = powerOfTwo ? 4 : 3;
      auto e = Divide(_2, Const(rhsValue));
      subExpr = e.clone();
    }
  } else {
    std::cerr << "Unsupported test type \n";
    return 1;
  }
  auto mainExpr = Add(Add(Add(_1, _2), *subExpr), Const(3));
  auto [opt, noOpt] = executeExpr(mainExpr, dataType, inPlace);
  for (unsigned i = 0; i != opt.size(); ++i) {
    if (opt[i] != noOpt[i]) {
      std::cerr << "mismatch at position " << i << " expected " << noOpt[i];
      std::cerr << " actual " << opt[i];
      return 1;
    }
  }
  return 0;
}