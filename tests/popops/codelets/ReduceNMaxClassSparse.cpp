// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ReduceNMaxClassSparse
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>

#include "poplibs_test/Util.hpp"
#include "popnn/codelets.hpp"
#include "poputil/VertexTemplates.hpp"

#include <gccs/Algorithm.hpp>

#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poputil;
using namespace poplibs_support;

namespace {

template <typename DataType>
std::vector<DataType> cpp_model(std::vector<DataType> acts, int numK) {
  std::make_heap(acts.begin(), acts.end());
  std::vector<DataType> output(numK);

  for (int i = 0; i < numK; ++i) {
    std::pop_heap(acts.begin(), acts.end());
    output[i] = acts.back();
    acts.pop_back();
  }

  return output;
}

} // namespace

template <typename DataType>
static bool doTest(const DeviceType &deviceType, const Type &activationType,
                   unsigned size, unsigned numK, bool sort) {
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);

  auto activations = graph.addVariable(activationType, {size}, "activations");
  auto labels = graph.addVariable(UNSIGNED_INT, {size}, "labels");

  auto maxAct = graph.addVariable(activationType, {numK}, "maxValues");
  auto maxValueIndices =
      graph.addVariable(UNSIGNED_INT, {numK}, "maxValuesIndices");

  graph.setTileMapping(activations, 0);
  graph.setTileMapping(maxAct, 0);
  graph.setTileMapping(maxValueIndices, 0);
  graph.setTileMapping(labels, 0);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);

  auto rawHostMaxAct = allocateHostMemoryForTensor(
      maxAct, "maxValues", graph, uploadProg, downloadProg, tmap);

  auto rawHostLabels = allocateHostMemoryForTensor(
      labels, "labels", graph, uploadProg, downloadProg, tmap);

  auto rawHostMaxValueIndices =
      allocateHostMemoryForTensor(maxValueIndices, "maxValuesIndices", graph,
                                  uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  std::vector<DataType> hostActivations(size);

  {
    boost::random::uniform_int_distribution<unsigned> randDist(
        std::numeric_limits<unsigned>::lowest(),
        std::numeric_limits<unsigned>::max());
    for (auto &a : hostActivations) {
      unsigned tmp = randDist(randomEngine);
      std::memcpy(&a, &tmp, sizeof(DataType));

      // Remove NANs.
      if (std::isnan(a)) {
        tmp = tmp >> 2;
        std::memcpy(&a, &tmp, sizeof(DataType));
      }

      // Flush denormals to zero.
      if (std::is_floating_point<DataType>::value &&
          std::fabs(a) < std::numeric_limits<DataType>::min()) {
        a = 0;
      }
    }
  }
  copy(target, hostActivations.data(), size, activationType,
       rawHostActivations.get());

  // Create the index vector for the indices. Should just be sequential 0 to
  // size.
  std::vector<unsigned> hostLabels(size);
  {
    for (unsigned i = 0; i < size; ++i) {
      hostLabels[i] = i;
    }
  }
  copy(target, hostLabels.data(), size, UNSIGNED_INT, rawHostLabels.get());

  auto cs = graph.addComputeSet();

  std::string vertexName =
      templateVertex("popnn::ReduceMaxNClassSparse", activationType, sort);

  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, 0);

  graph.connect(v["activations"], activations);
  graph.connect(v["maxValuesIndices"], maxValueIndices);
  graph.connect(v["maxValues"], maxAct);
  graph.connect(v["labels"], labels);
  graph.setInitialValue(v["numK"], numK);
  graph.setInitialValue(v["shouldSort"], sort);

  Engine e(std::move(graph), Sequence{uploadProg, Execute(cs), downloadProg});

  attachStreams(e, tmap);
  device.bind([&](const Device &d) { e.loadAndRun(d); });

  std::vector<DataType> deviceOut(numK);
  copy(target, activationType, rawHostMaxAct.get(), deviceOut.data(), numK);

  std::vector<unsigned> deviceOutIndices(numK);
  copy(target, UNSIGNED_INT, rawHostMaxValueIndices.get(),
       deviceOutIndices.data(), numK);

  std::vector<DataType> cppOut = cpp_model(hostActivations, numK);
  bool success = true;

  // Check that the indices returned match up to the activations. We do this
  // first so we don't have to deal with the sort later.
  for (unsigned i = 0; i < numK; ++i) {
    int ind = deviceOutIndices[i];

    success &= deviceOut[i] == hostActivations[ind];
  }

  // If we didn't expect the output to be sorted originally we have to sort it
  // as the raw array view of a binary heap is not sorted.
  if (!sort) {
    std::sort(deviceOut.begin(), deviceOut.end(), std::greater<DataType>());
  }

  // Check that it matches the C++ version.
  for (unsigned i = 0; i < numK; ++i) {
    success &= deviceOut[i] == cppOut[i];
  }

  return success;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type activationType;
  unsigned topK;
  unsigned size;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()("help", "Print help")(
      "device-type", po::value<DeviceType>(&deviceType)->required(),
      "Device Type")("size", po::value<unsigned>(&size)->required(),
                     "Total size to process with vertex")(
      "k", po::value<unsigned>(&topK)->required(),
      "Return 'k' number of elements.")(
      "activation-type", po::value<Type>(&activationType)->required(),
      "Element type for input activations");
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

  if (activationType == FLOAT) {
    // Check against unsorted.
    if (!doTest<float>(deviceType, activationType, size, topK, false))
      return 1;

    // Check against sorted.
    if (!doTest<float>(deviceType, activationType, size, topK, true))
      return 1;
  } else if (activationType == INT) {
    if (!doTest<int>(deviceType, activationType, size, topK, false) ||
        !doTest<int>(deviceType, activationType, size, topK, true)) {
      return 1;
    }
  } else if (activationType == UNSIGNED_INT) {
    if (!doTest<unsigned>(deviceType, activationType, size, topK, false) ||
        !doTest<unsigned>(deviceType, activationType, size, topK, true)) {
      return 1;
    }
  }
  return 0;
}
