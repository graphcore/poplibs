// Copyright (c) Graphcore Ltd, All rights reserved.
#include "TestDevice.hpp"
#include <poplar/Engine.hpp>

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_test/Util.hpp"
#include "popnn/codelets.hpp"
#include "poputil/VertexTemplates.hpp"

#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;

namespace {

static void modelVertex(const std::vector<double> &activations, unsigned index,
                        std::vector<double> &maxActs,
                        std::vector<std::uint64_t> &maxIndices, unsigned size,
                        unsigned divisor) {

  const auto nOutputs = (size + divisor - 1) / divisor;
  for (std::size_t i = 0; i < nOutputs; ++i) {
    unsigned maxLabel = i * divisor;
    double max = activations[maxLabel];
    const auto end = std::min(size, maxLabel + divisor);
    for (std::size_t j = maxLabel + 1; j < end; ++j) {
      if (activations[j] > max) {
        max = activations[j];
        maxLabel = j;
      }
    }
    maxActs[i] = max;
    maxIndices[i] = maxLabel + index;
  }
}

static bool doTest(const DeviceType &deviceType, const Type &activationsType,
                   const Type &labelType, unsigned divisor, unsigned size) {
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();

  if ((divisor & (divisor - 1)) != 0) {
    throw std::logic_error("divisor is not a power of 2");
  }
  const auto nOutputs = (size + divisor - 1) / divisor;
  if (nOutputs > target.getNumWorkerContexts()) {
    throw std::logic_error(
        "Divisor is not large enough for the vertex to process all inputs");
  }

  Graph graph(target);
  popnn::addCodelets(graph);
  auto partialsType = (activationsType == HALF || activationsType == FLOAT)
                          ? FLOAT
                          : activationsType;

  auto activations = graph.addVariable(activationsType, {size}, "activations");
  auto maxActs =
      graph.addVariable(partialsType, {nOutputs}, "maxValuePartials");
  auto maxIndices =
      graph.addVariable(labelType, {nOutputs}, "maxIndexPartials");
  graph.setTileMapping(activations, 0);
  graph.setTileMapping(maxActs, 0);
  graph.setTileMapping(maxIndices, 0);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);
  auto rawHostMaxActs = allocateHostMemoryForTensor(
      maxActs, "maxValuePartials", graph, uploadProg, downloadProg, tmap);
  auto rawHostMaxIndices = allocateHostMemoryForTensor(
      maxIndices, "maxIndexPartials", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  std::vector<double> hostActivations(size);
  const bool isFpType = activationsType == FLOAT || activationsType == HALF;
  const bool isInt = activationsType == INT;
  writeRandomValues(target, activationsType, hostActivations.data(),
                    hostActivations.data() + hostActivations.size(),
                    isInt ? std::numeric_limits<int>::min() : 0.0,
                    isFpType ? 1.0 : std::numeric_limits<int>::max(),
                    randomEngine);
  copy(target, hostActivations.data(), size, activationsType,
       rawHostActivations.get());

  auto cs = graph.addComputeSet();
  auto v = graph.addVertex(cs, templateVertex("popnn::ReduceMaxClassGather",
                                              activationsType, labelType));
  graph.setTileMapping(v, 0);

  // TODO: T12987 Add test case for when index != 0.
  unsigned index = 0;
  graph.connect(v["activations"], activations);
  graph.setInitialValue(v["index"], index);
  graph.connect(v["maxValue"], maxActs);
  graph.connect(v["maxIndex"], maxIndices);
  graph.setInitialValue(v["size"], size);
  graph.setInitialValue(v["workerSize"], divisor);

  Engine e(std::move(graph), Sequence(uploadProg, Execute(cs), downloadProg));
  attachStreams(e, tmap);
  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<double> modelMaxActs(nOutputs);
  std::vector<std::uint64_t> modelMaxIndices(nOutputs);
  modelVertex(hostActivations, index, modelMaxActs, modelMaxIndices, size,
              divisor);

  std::vector<double> hostMaxActs(nOutputs);
  std::vector<std::uint64_t> hostMaxIndices(nOutputs);
  copy(target, partialsType, rawHostMaxActs.get(), hostMaxActs.data(),
       nOutputs);
  copy(target, labelType, rawHostMaxIndices.get(), hostMaxIndices.data(),
       nOutputs);

  bool success = true;
  success &= checkIsClose("maxValue", hostMaxActs.data(), {nOutputs},
                          modelMaxActs.data(), nOutputs, 0.1,
                          activationsType == HALF ? 1e-7 : 1e-20);
  success &= checkEqual("maxIndex", hostMaxIndices.data(), {nOutputs},
                        modelMaxIndices.data(), nOutputs);
  return success;
}

} // end anonymous namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type activationType;
  Type labelType = UNSIGNED_INT;
  unsigned divisor, size;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("activation-type",
     po::value<Type>(&activationType)->required(),
     "Element type for input activations")
    ("divisor",
     po::value<unsigned>(&divisor)->required(),
     "Factor by which to reduce max class")
    ("size",
     po::value<unsigned>(&size)->required(),
     "Total size to process with vertex");
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

  if (!doTest(deviceType, activationType, labelType, divisor, size))
    return 1;
  return 0;
}
