// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include "popops/NormaliseImage.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/codelets.hpp"
#include "poputil/Util.hpp"
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <poplar/Program.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

static constexpr float halfTolerance = .005;
static constexpr float floatTolerance = 1e-6;

// Normalise the 3 input channels and pad a 4th zero channel
void refNormalise(const boost::multi_array<float, 4> &input,
                  boost::multi_array<float, 4> &output, float inScale,
                  std::vector<float> &offsets, std::vector<float> &scales) {
  for (std::size_t i = 0; i != input.num_elements() / 3; ++i) {
    for (unsigned c = 0; c != 3; ++c) {
      output.data()[i * 4 + c] =
          (input.data()[i * 3 + c] * inScale - offsets[c]) * scales[c];
    }
    output.data()[i * 4 + 3] = 0;
  }
}

static bool imageNormaliseTest(DeviceType deviceType, unsigned numTiles,
                               Type dType, std::size_t batchSize,
                               std::size_t fieldSize) {
  std::mt19937 randomEngine;
  boost::random::uniform_real_distribution<float> dist(
      0., dType == UNSIGNED_CHAR ? 256 : 10.);

  boost::multi_array<float, 4> input(
      boost::extents[batchSize][fieldSize][fieldSize][3]);
  boost::multi_array<float, 4> output(
      boost::extents[batchSize][fieldSize][fieldSize][4]);
  boost::multi_array<float, 4> refOutput(
      boost::extents[batchSize][fieldSize][fieldSize][4]);

  for (unsigned i = 0; i < input.num_elements(); ++i) {
    auto &val = *(input.data() + i);
    val = dist(randomEngine);
    if (dType == UNSIGNED_CHAR)
      val = floor(val);
  }

  const std::vector<std::size_t> shape = {batchSize, fieldSize, fieldSize, 3};

  auto device = createTestDevice(deviceType, 1, numTiles, true);
  const auto &target = device.getTarget();
  Graph graph(target);

  popops::addCodelets(graph);

  Type outType = dType == FLOAT ? FLOAT : HALF;
  Tensor tOffsets = graph.addVariable(outType, {3}, {"offsets"});
  Tensor tScales = graph.addVariable(outType, {3}, {"scales"});
  graph.setTileMapping(tOffsets, 0);
  graph.setTileMapping(tScales, 0);
  float inScale = 1. / 255;
  std::vector<float> offsets{0.485, 0.456, 0.406};
  std::vector<float> scales{1 / 0.229, 1 / 0.224, 1 / 0.225};
  Sequence seq;
  Tensor tIn =
      popops::createNormaliseImageInput(graph, dType, shape, {"TestIn"});
  Tensor tOut = popops::normaliseImage(graph, seq, tIn, inScale, tOffsets,
                                       tScales, {"Normalise"});

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;

  auto rawHostInput = allocateHostMemoryForTensor(
      tIn, "input", graph, uploadProg, downloadProg, tmap);
  copy(target, input, dType, rawHostInput.get());
  auto rawHostScales = allocateHostMemoryForTensor(
      tScales, "scales", graph, uploadProg, downloadProg, tmap);
  copy(target, scales, outType, rawHostScales.get());
  auto rawHostOffsets = allocateHostMemoryForTensor(
      tOffsets, "offsets", graph, uploadProg, downloadProg, tmap);
  copy(target, offsets, outType, rawHostOffsets.get());

  auto rawHostOutput = allocateHostMemoryForTensor(
      tOut, "out", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence{uploadProg, seq, downloadProg});
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  copy(target, outType, rawHostOutput.get(), output);

  refNormalise(input, refOutput, inScale, offsets, scales);
  bool debugging = false;
  auto tol = outType == FLOAT ? floatTolerance : halfTolerance;
  for (unsigned i = 0; i != output.num_elements(); ++i) {
    auto inVal = i % 4 == 3 ? 0 : input.data()[i / 4 * 3 + i % 4];
    auto outVal = output.data()[i];
    auto refVal = refOutput.data()[i];
    if (debugging)
      std::cerr << "i=" << i << ", in=" << inVal << ", ref=" << refVal
                << ", out=" << outVal << "\n";

    if (fabs(refVal - outVal) > tol &&
        (fabs(refVal) > tol && fabs(outVal) > tol &&
         fabs(refVal / outVal - 1.0) > tol)) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  DeviceType deviceType;

  std::size_t batchSize = 1, fieldSize = 224;
  Type dType = FLOAT;
  unsigned numTiles = 4;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("batch-size",
     po::value<std::size_t>(&batchSize)->default_value(batchSize))
    ("field-size",
     po::value<std::size_t>(&fieldSize)->default_value(fieldSize))
    ("num-tiles",
     po::value<unsigned>(&numTiles),
     "Number of tiles to use (default chosen by heuristics)")
    ("data-type",
     po::value<Type>(&dType)->required(),
     "data type used in the test (uint8/half/float)");

  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
    if (dType != UNSIGNED_CHAR && dType != HALF && dType != FLOAT) {
      std::cout << "Only uchar, half and float are permitted\n";
      return 1;
    }
    if (!vm.count("num-tiles")) {
      // Use enough tiles to keep the footprint below 128K/tile.
      auto numElem = batchSize * fieldSize * fieldSize;
      auto bytes = (3 + 4) * sizeof(float) * numElem; // allow for FLOAT->FLOAT
      numTiles = ceil(bytes + (1024 * 128 - 1)) / (1024 * 128);
      std::cout << "Testing with " << numTiles << " tiles.\n";
    }
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  auto result =
      imageNormaliseTest(deviceType, numTiles, dType, batchSize, fieldSize);

  return !result;
}
