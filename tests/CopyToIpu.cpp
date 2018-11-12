#define BOOST_TEST_MODULE CopyToIpu

#include "poplibs_test/Util.hpp"
#include "poputil/TileMapping.hpp"
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

BOOST_AUTO_TEST_CASE(CopyToIpuTest) {
  const auto numIpus = 4;
  const auto tilesPerIpu = 16;
  const auto numElements = 1000;
  auto device = createTestDevice(TEST_TARGET, numIpus, tilesPerIpu);

  Graph graph(device);
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  Tensor testInput = graph.addVariable(INT, {numElements},
                                       VariableMappingMethod::LINEAR);
  Tensor testResult = graph.clone(testInput);
  auto rawHostTestInput =
    allocateHostMemoryForTensor(testInput, "testInput", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostTestResult =
    allocateHostMemoryForTensor(testResult, "testResult", graph, uploadProg,
                                downloadProg, tmap);
  std::vector<Tensor> srcTensors;
  // Create a source tensor that is spread across IPUs
  srcTensors.push_back(graph.addVariable(INT, {numElements},
                                         VariableMappingMethod::LINEAR));
  // Create source tensors that are on a single IPU.
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    auto subGraph = graph.createVirtualGraph(ipu * tilesPerIpu,
                                             (ipu + 1) * tilesPerIpu);
    srcTensors.push_back(subGraph.addVariable(INT, {numElements},
                                              VariableMappingMethod::LINEAR));
  }

  std::vector<Program> progs;
  // Try to copy each source tensor to each IPU
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    for (const auto &srcTensor : srcTensors) {
      Sequence prog;
      prog.add(Copy(testInput, srcTensor));
      auto ipuCopy = poputil::copyToIpu(graph, srcTensor, prog, ipu);
      // Check the copy is mapped to the right IPU.
      auto mapping = graph.getTileMapping(ipuCopy, true);
      for (unsigned tile = 0; tile != mapping.size(); ++tile) {
        if (mapping[tile].empty())
          continue;
        BOOST_CHECK_EQUAL(tile / tilesPerIpu, ipu);
      }
      prog.add(Copy(ipuCopy, testResult));
      progs.push_back(Sequence(uploadProg, prog, downloadProg));
    }
  }
  Engine engine(graph, progs);
  engine.load(device);
  attachStreams(engine, tmap);

  std::vector<int> hostTestInput(numElements);
  std::vector<int> hostTestResult(numElements);
  auto rawHostTestInputPtr = reinterpret_cast<int*>(rawHostTestInput.get());
  auto rawHostTestResultPtr = reinterpret_cast<int*>(rawHostTestResult.get());

  std::mt19937 randomEngine;
  for (unsigned progNum = 0; progNum != progs.size(); ++progNum) {
    // Run each program and check the data is actually copied.
    std::generate(hostTestInput.begin(), hostTestInput.end(),
                  randomEngine);
    std::copy(hostTestInput.begin(), hostTestInput.end(), rawHostTestInputPtr);
    std::fill(hostTestResult.begin(), hostTestResult.end(), 0);
    std::copy(hostTestResult.begin(), hostTestResult.end(),
              rawHostTestResultPtr);
    engine.run(progNum);
    std::copy(rawHostTestResultPtr, rawHostTestResultPtr + numElements,
              hostTestResult.begin());
    BOOST_CHECK_EQUAL_COLLECTIONS(hostTestInput.begin(),
                                  hostTestInput.end(),
                                  hostTestResult.begin(),
                                  hostTestResult.end());
  }
}
