// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include "popops/NaN.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/ElementWise.hpp"
#include "popops/codelets.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

static constexpr std::size_t D0 = 6;
static constexpr std::size_t D1 = 5;
static constexpr std::size_t D2 = 4;
static constexpr std::size_t D3 = 3;
static constexpr std::size_t totalSize = D0 * D1 * D2 * D3;

static bool hasNaNTest(DeviceType deviceType, const bool introduceNaN,
                       const bool introduceInf, const Type &type,
                       std::size_t testSize, unsigned numTiles,
                       bool twoDimensionalVertices, bool checkNaNOnly) {
  std::mt19937 randomEngine;
  boost::random::uniform_real_distribution<double> dist(0., 10.);

  boost::multi_array<double, 4> input(boost::extents[D0][D1][D2][D3]);

  for (unsigned i = 0; i < input.num_elements(); ++i) {
    *(input.data() + i) = dist(randomEngine);
  }

  // Fill last element
  const std::vector<std::size_t> shape = {D0, D1, D2, D3};

  if (introduceNaN) {
    auto indices = poputil::unflattenIndex(shape, testSize - 1);
    input[indices[0]][indices[1]][indices[2]][indices[3]] = NAN;
  }

  if ((testSize > 1 || !introduceNaN) && introduceInf) {
    auto off = 1 + (testSize > 1);
    auto indices = poputil::unflattenIndex(shape, testSize - off);
    input[indices[0]][indices[1]][indices[2]][indices[3]] =
        std::numeric_limits<double>::infinity();
  }

  // fill NANs outside
  std::fill(input.data() + testSize, input.data() + input.num_elements(), NAN);

  auto device = createTestDevice(deviceType, 1, numTiles, true);
  const auto &target = device.getTarget();
  Graph graph(target);

  popops::addCodelets(graph);

  Tensor inputT;
  if (twoDimensionalVertices) {
    static_assert(D0 % 2 == 0);
    auto var1 = graph.addVariable(type, {D0 / 2, D1, D2, D3}, "input1");
    auto var2 = graph.addVariable(type, {D0 / 2, D1, D2, D3}, "input2");
    poputil::mapTensorLinearly(graph, var1);
    poputil::mapTensorLinearly(graph, var2);

    inputT = concat(var1, var2);
  } else {
    inputT = graph.addVariable(type, {D0, D1, D2, D3}, "input");
    poputil::mapTensorLinearly(graph, inputT);
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  auto rawHostInput = allocateHostMemoryForTensor(
      inputT, "input", graph, uploadProg, downloadProg, tmap);
  copy(target, input, type, rawHostInput.get());

  Sequence prog;
  const auto out =
      checkNaNOnly
          ? popops::hasNaN(graph, inputT.flatten().slice(0, testSize), prog)
          : popops::hasNaNOrInf(graph, inputT.flatten().slice(0, testSize),
                                prog);
  auto rawHostOutput = allocateHostMemoryForTensor(
      out, "out", graph, uploadProg, downloadProg, tmap);

  const poplar::OptionFlags options{{"debug.floatPointOpException", "false"}};
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  boost::multi_array<bool, 1> result(boost::extents[1]);
  copy(target, BOOL, rawHostOutput.get(), result);
  return checkNaNOnly ? result[0] == introduceNaN
                      : result[0] == (introduceNaN || introduceInf);
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  DeviceType deviceType;

  std::size_t numElements;
  bool fullSize, addInf, addNaN, twoD = false, checkNaNOnly;
  Type dType;
  unsigned numTiles = 1;

  const std::string sizeString = "Number of elements in tensor (max is " +
                                 std::to_string(totalSize) +
                                 ") "
                                 " ignored if fullSize is set to true";

  po::options_description desc("Options");
  // clang-format off
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("full-size",
     po::value<bool>(&fullSize)->required(),
     "Test full size tensor(if set tests full tensor size else only part of it "
     "as given by numElements)")
    ("size",
     po::value<std::size_t>(&numElements)->default_value(totalSize),
     sizeString.data())
    ("two-d",
      po::value<bool>(&twoD)->default_value(twoD),
      "Use 2D vertex (only allowed with fullSize)")
    ("add-inf",
     po::value<bool>(&addInf)->required(),
     "Add inf to the tensor")
    ("add-nan",
     po::value<bool>(&addNaN)->required(),
     "Add NaN to the tensor")
    ("check-nan-only",
     po::value<bool>(&checkNaNOnly)->required(),
     "Check for NaNs only even if there may be infs, else checks for both")
    ("num-tiles",
     po::value<unsigned>(&numTiles)->default_value(numTiles),
     "Number of tiles to use")
    ("data-type",
     po::value<Type>(&dType)->required(),
     "data type used in the test (half/float)");

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
  if (numElements > totalSize) {
    std::cerr << "size exceed maximum allowed\n";
    return 1;
  }
  if (!fullSize && twoD) {
    std::cerr << "2D vertices are only restricted with full sized tensors\n";
    return 1;
  }

  auto result = hasNaNTest(deviceType, addNaN, addInf, dType, numElements,
                           numTiles, twoD, checkNaNOnly);

  return !result;
}
