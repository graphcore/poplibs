// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "poplar/Graph.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/TestDevice.hpp"
#include "popsparse/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_test/Util.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <numeric>

using namespace poplibs_support;
using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;

static constexpr int maxRepresentableIntegerInHalf = 1024;

boost::multi_array<double, 2> createInput(unsigned numXOrY, unsigned numZ,
                                          int startNum) {
  boost::multi_array<double, 2> in(boost::extents[numZ][numXOrY]);
  for (unsigned z = 0; z != numZ; ++z) {
    for (unsigned xy = 0; xy != numXOrY; ++xy, ++startNum) {
      in[z][xy] = startNum;
    }
  }
  return in;
}

static unsigned zBlockSize(const Type &type) { return type == FLOAT ? 8 : 16; }

// Return input value for a given output position
static double getOutputVal(const boost::multi_array_ref<double, 2> in,
                           unsigned blockSizeXY, const Type &type, unsigned xy,
                           unsigned z) {
  const auto numZ = in.shape()[0];
  const unsigned blockSizeZ = zBlockSize(type);
  const auto outFlattenedIndex = xy * numZ + z;
  const auto blockXY = outFlattenedIndex / (numZ * blockSizeXY);
  const auto xyBlockIndex = outFlattenedIndex % (numZ * blockSizeXY);
  const auto blockZ = xyBlockIndex / (blockSizeZ * blockSizeXY);
  const auto zBlockIndex = xyBlockIndex % (blockSizeZ * blockSizeXY);
  const auto zWithinZBlock = zBlockIndex % blockSizeZ;
  const auto xyWithinZBlock = zBlockIndex / blockSizeZ;
  const auto xyInIndex = blockXY * blockSizeXY + xyWithinZBlock;
  const auto zInIndex = blockZ * blockSizeZ + zWithinZBlock;
  return in[zInIndex][xyInIndex];
}

static bool supportedBlockSize(unsigned blockSize) {
  return blockSize == 4 || blockSize == 8 || blockSize == 16;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  Type dataType = FLOAT;
  unsigned numXOrY, numZ, blockSize;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("data-type",
     po::value<Type>(&dataType)->default_value(dataType),
     "Data type")
    ("x-y-size",
     po::value<unsigned>(&numXOrY)->required(),
     "X/Y size such that product of X/Y size and Z size is <= 2048")
    ("z-size",
     po::value<unsigned>(&numZ)->required(),
     "Z size such that product of X/Y size and Z size is <= 2048")
    ("block-size",
     po::value<unsigned>(&blockSize)->required(),
     "Block size (only 4, 8 and 16 are supported)")
  ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (dataType != HALF && dataType != FLOAT) {
    std::cerr << "Only half and float data types supported"
              << "\n";
    return 1;
  }

  if (!supportedBlockSize(blockSize)) {
    std::cerr << "Block size not supported"
              << "\n";
    return 1;
  }

  if (numXOrY % blockSize) {
    std::cerr << "X/Y size must be a multiple of block size"
              << "\n";
    return 1;
  }

  if (numZ % zBlockSize(dataType)) {
    std::cerr << "Z size must be a multiple of block size"
              << "\n";
    return 1;
  }

  if (numZ * numXOrY > 2 * maxRepresentableIntegerInHalf) {
    std::cerr << "Product of X/Y and Z must be less than 2048";
    return 1;
  }

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popsparse::addCodelets(graph);

  // Allocate operands
  const auto xyIn = graph.addVariable(dataType, {numZ, numXOrY}, "xyIn");
  const auto xyOut = graph.addVariable(dataType, {numXOrY, numZ}, "xyOut");
  graph.setTileMapping(xyIn, 0);
  graph.setTileMapping(xyOut, 0);

  const auto numXOrYBlocks = numXOrY / blockSize;
  const auto cs = graph.addComputeSet("cs0");
  const auto vertexClass =
      templateVertex("popsparse::BlockTransposeGradW", dataType);
  const auto v = graph.addVertex(
      cs, vertexClass, {{"in", xyIn.flatten()}, {"out", xyOut.flatten()}});
  graph.setInitialValue(v["blockSizeXOrY"], blockSize);
  graph.setInitialValue(v["numXOrYBlocks"], numXOrYBlocks);
  graph.setInitialValue(v["numZ"], numZ);
  graph.setInitialValue(v["maxXOrYBlocksPerWorker"],
                        ceildiv(numXOrYBlocks, target.getNumWorkerContexts()));
  graph.setTileMapping(v, 0);

  Sequence prog;
  prog.add(Execute(cs));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostXYIn, rawHostXYOut;
  rawHostXYIn = allocateHostMemoryForTensor(xyIn, "xyIn", graph, uploadProg,
                                            downloadProg, tmap);
  rawHostXYOut = allocateHostMemoryForTensor(xyOut, "xyOut", graph, uploadProg,
                                             downloadProg, tmap);

  std::cout << "Building engine...\n";
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  std::cout << "Built\n";
  attachStreams(engine, tmap);

  auto hostXYIn = createInput(numXOrY, numZ, 0);
  boost::multi_array<double, 2> hostXYOut(boost::extents[numXOrY][numZ]);

  copy(target, hostXYIn, dataType, rawHostXYIn.get());

  std::cout << "Attaching to device and running...\n";
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.setPrintStream(std::cerr);
    engine.run();
  });

  copy(target, dataType, rawHostXYOut.get(), hostXYOut);
  std::cout << "Finished running\n";

  bool matchesModel = true;
  for (unsigned xy = 0; xy != numXOrY; ++xy) {
    for (unsigned z = 0; z != numZ; ++z) {
      const auto modelOut = getOutputVal(hostXYIn, blockSize, dataType, xy, z);
      if (modelOut != hostXYOut[xy][z]) {
        std::cerr << "Mismatch at [" << xy << "," << z << "]";
        std::cerr << "expected " << modelOut << " actual " << hostXYOut[xy][z];
        std::cerr << "\n";
        matchesModel = false;
      }
    }
  }
  return !matchesModel;
}
