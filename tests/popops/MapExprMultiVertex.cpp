// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <boost/program_options.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/TempDir.hpp>
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

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popops;
using namespace popops::expr;
using namespace poplibs_test::util;
using namespace poplar_test;

std::pair<std::vector<double>, std::vector<double>>
executeExpr(const DeviceType &deviceType, const Expr &expression,
            const Type &dType, bool inPlace, unsigned length, unsigned slices,
            bool profile) {
  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph g(device.getTarget());
  addCodelets(g);

  // Padding to use to either check for overwrites or create a gap between
  // slices. Padding will result in later slices being aligned to 4 element
  // (At least 8 byte) boundaries to avoid copies and leave in place ops where
  // they are allocated
  const auto paddingSize = (1 << ceilLog2(std::max(length + 1, 3u))) - length;
  const auto dataLength = (length + paddingSize) * slices;
  std::vector<double> hostInA(dataLength);
  std::vector<double> hostInC(dataLength);
  std::vector<double> hostInB(dataLength);
  std::vector<double> hostOutOpt(inPlace ? dataLength : length * slices);
  std::vector<double> hostOutNoOpt(inPlace ? dataLength : length * slices);

  for (unsigned i = 0; i < hostInA.size(); i++) {
    double value = dType != UNSIGNED_INT && (i % 2) ? -1.0 * i : i;
    hostInA[i] = value;
    hostInB[i] = 2 * (hostInA.size() - i);
    hostInC[i] = value;
  }

  auto a = g.addVariable(dType, {dataLength}, "inputa");
  g.setTileMapping(a, 0);
  auto b = g.addVariable(dType, {dataLength}, "inputb");
  g.setTileMapping(b, 0);
  auto c = g.addVariable(dType, {dataLength}, "inputc");
  g.setTileMapping(c, 0);

  std::vector<Interval> sliceIntervals(slices);
  for (unsigned i = 0; i < slices; i++) {
    sliceIntervals[i] = {i * (length + paddingSize),
                         i * (length + paddingSize) + length};
  }
  auto aSlices = concat(a.slices(sliceIntervals));
  auto bSlices = concat(b.slices(sliceIntervals));
  auto cSlices = concat(c.slices(sliceIntervals));

  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawA, rawB, rawC, rawOutOpt, rawOutNoOpt;
  program::Sequence uploadProg, downloadProg;
  rawA =
      allocateHostMemoryForTensor(a, "rawA", g, uploadProg, downloadProg, tmap);
  rawB =
      allocateHostMemoryForTensor(b, "rawB", g, uploadProg, downloadProg, tmap);
  rawC =
      allocateHostMemoryForTensor(c, "rawC", g, uploadProg, downloadProg, tmap);

  copy(target, hostInA, dType, rawA.get());
  copy(target, hostInB, dType, rawB.get());
  copy(target, hostInC, dType, rawC.get());
  OptionFlags enableGen = {{"enableGenerateCodelet", "true"}};
  OptionFlags disableGen = {{"enableGenerateCodelet", "false"}};

  Sequence prog;
  Tensor tOpt;
  Tensor tNoOpt;
  if (!inPlace) {
    tOpt = popops::map(g, expression, {aSlices, bSlices}, prog, "testEnable",
                       enableGen);
    tNoOpt = popops::map(g, expression, {cSlices, bSlices}, prog, "testDisable",
                         disableGen);
  } else {
    popops::mapInPlace(g, expression, {aSlices, bSlices}, prog, "testEnable",
                       enableGen);
    popops::mapInPlace(g, expression, {cSlices, bSlices}, prog, "testDisable",
                       disableGen);
  }
  if (!inPlace) {
    rawOutOpt = allocateHostMemoryForTensor(tOpt, "rawOpt", g, uploadProg,
                                            downloadProg, tmap);
    rawOutNoOpt = allocateHostMemoryForTensor(tNoOpt, "rawNoOpt", g, uploadProg,
                                              downloadProg, tmap);
  }
  Sequence controlProg({uploadProg, prog, downloadProg});

  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Engine e(g, controlProg, engineOptions);
  attachStreams(e, tmap);
  device.bind([&](const Device &d) {
    e.load(d);
    e.run();
  });

  if (deviceType != DeviceType::Cpu && profile) {
    e.printProfileSummary(std::cout,
                          OptionFlags{{"showExecutionSteps", "true"}});
  }
  if (inPlace) {
    copy(target, dType, rawA.get(), hostOutOpt);
    copy(target, dType, rawC.get(), hostOutNoOpt);
  } else {
    copy(target, dType, rawOutOpt.get(), hostOutOpt);
    copy(target, dType, rawOutNoOpt.get(), hostOutNoOpt);
  }
  return std::make_pair(hostOutOpt, hostOutNoOpt);
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  poplar::Type dataType;
  bool inPlace;
  bool profile = false;
  unsigned length, slices = 1;
  bool vectorizableExpression = true;
  DeviceType deviceType;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("length",
     po::value<unsigned>(&length)->required(),
     "Data length")
    ("slices",
     po::value<unsigned>(&slices)->default_value(slices),
     "Number of slices to operate on")
    ("data-type",
     po::value<poplar::Type>(&dataType)->required(),
     "Data Type: unsigned, int")
    ("in-place",
     po::value<bool>(&inPlace)->required(),
     "operation is inplace")
    ("vectorizable-expression",
     po::value<bool>(&vectorizableExpression)
        ->default_value(vectorizableExpression),
     "Use a vectorizable expression that should produce a MultiVertex")
   ("profile",
     po::value<bool>(&profile)->default_value(profile),
     "provide profile output")
    ;
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

  const auto vExpr = Add(Sub(Add(_1, _2), _1), _2);
  // Due to bool as a
  const auto expr = Select(Mul(Const(0.0f), _1), _1, Lt(_1, Const(0.0f)));

  auto [opt, noOpt] = vectorizableExpression
                          ? executeExpr(deviceType, vExpr, dataType, inPlace,
                                        length, slices, profile)
                          : executeExpr(deviceType, expr, dataType, inPlace,
                                        length, slices, profile);
  for (unsigned i = 0; i != opt.size(); ++i) {
    if (opt[i] != noOpt[i]) {
      std::cerr << "mismatch at position " << i << " expected " << noOpt[i];
      std::cerr << " actual " << opt[i];
      return 1;
    }
  }
  return 0;
}
