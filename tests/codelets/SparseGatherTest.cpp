// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseGatherTest
#include <poplibs_support/TestDevice.hpp>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <poplar/Graph.hpp>
#include <poplibs_test/Util.hpp>
#include <popsparse/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace poplibs_support;

std::pair<unsigned short, unsigned short>
workDivision(unsigned short numIndices, Type type, unsigned numWorkers) {
  assert(type == HALF || type == FLOAT);
  assert(numWorkers <= 6);
  unsigned elemsInVector = type == HALF ? 4 : 2;
  auto numVectors = (numIndices / elemsInVector) / numWorkers;
  int remainder = (numIndices / elemsInVector) - numVectors * numWorkers;

  // distribute remainder equally amonst workers. The last worker index is first
  unsigned offsetDistribution = 0;
  for (unsigned w = 0; w != numWorkers; ++w, --remainder) {
    offsetDistribution = (offsetDistribution << 1) | (remainder > 0);
  }

  return std::make_pair((numVectors * elemsInVector) |
                            (numIndices % elemsInVector),
                        offsetDistribution);
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  Type dataType = HALF;
  std::size_t numIndices;

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
    ("num-indices",
     po::value<std::size_t>(&numIndices)->required(),
     "Number of indices (must be less than 32768 (HALF) and 16384 (FLOAT")
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

  if (numIndices >= 65536 / (dataType == HALF ? 2 : 4)) {
    std::cerr << "error: number of indices exceed limit for data type"
              << "\n";
  }

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popsparse::addCodelets(graph);

  // Allocate operands
  const auto rIn = graph.addVariable(dataType, {numIndices}, "rIn");
  const auto rOut = graph.addVariable(dataType, {numIndices}, "rOut");
  const auto indicesVar =
      graph.addVariable(UNSIGNED_SHORT, {numIndices}, "indices");

  graph.setTileMapping(rIn, 0);
  graph.setTileMapping(rOut, 0);
  graph.setTileMapping(indicesVar, 0);

  const auto cs = graph.addComputeSet("cs0");
  const auto vertexClass =
      templateVertex("popsparse::SparseGatherElementWise", dataType);
  const auto v = graph.addVertex(
      cs, vertexClass, {{"rIn", rIn}, {"rOut", rOut}, {"indices", indicesVar}});

  auto work = workDivision(numIndices, dataType,
                           graph.getTarget().getNumWorkerContexts());
  graph.setInitialValue(v["numIndices"], work.first);
  graph.setInitialValue(v["workerOffsets"], work.second);
  graph.setTileMapping(v, 0);

  Sequence prog;
  prog.add(Execute(cs));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostRIn, rawHostROut, rawHostIndices;

  rawHostRIn = allocateHostMemoryForTensor(rIn, "rIn", graph, uploadProg,
                                           downloadProg, tmap);
  rawHostROut = allocateHostMemoryForTensor(rOut, "rOut", graph, uploadProg,
                                            downloadProg, tmap);
  rawHostIndices = allocateHostMemoryForTensor(indicesVar, "indicesVar", graph,
                                               uploadProg, downloadProg, tmap);

  std::cout << "Building engine...\n";
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  std::cout << "Built\n";
  attachStreams(engine, tmap);

  std::vector<unsigned short> hostIndices;
  hostIndices.resize(numIndices);
  std::iota(hostIndices.begin(), hostIndices.end(), 0);
  std::mt19937 rng;
  const auto randomGen = [&](auto max) {
    boost::random::uniform_int_distribution<decltype(max)> dist(0, max - 1);
    return dist(rng);
  };
  boost::range::random_shuffle(hostIndices, randomGen);

  const auto scale = dataType == HALF ? 2 : 4;

  for (unsigned i = 0; i != hostIndices.size(); ++i) {
    hostIndices[i] *= scale;
  }

  std::vector<float> hostRIn;
  hostRIn.resize(numIndices);

  std::vector<float> hostROut;
  hostROut.resize(numIndices);

  for (unsigned i = 0; i != numIndices; ++i) {
    hostROut[i] = numIndices + 1;
    // just to keep within fp16 integer range
    hostRIn[i] = static_cast<float>(i % 1024);
  }

  copy(target, hostRIn, dataType, rawHostRIn.get());
  copy(target, hostROut, dataType, rawHostROut.get());
  copy(target, hostIndices, UNSIGNED_SHORT, rawHostIndices.get());

  std::cout << "Attaching to device and running...\n";
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.setPrintStream(std::cerr);
    engine.run();
  });

  copy(target, dataType, rawHostROut.get(), hostROut.data(), numIndices);
  std::cout << "Finished running\n";

  // check result
  for (std::size_t i = 0; i != numIndices; ++i) {
    if (hostROut[i] != hostRIn[hostIndices[i] / scale]) {
      std::cerr << "Validation failed at index " << i;
      std::cerr << " actual " << hostROut[i];
      std::cerr << " exp " << hostRIn[hostIndices[i] / scale] << "\n";
      return 1;
    }
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
