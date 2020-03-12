// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DuplicateTensorTest

#include "TestDevice.hpp"
#include "poplibs_test/Util.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

static void TestFunc(poplar::TensorCloneMethod cloneMethod) {
  const auto numIpus = 1;
  const auto tilesPerIpu = 4;
  const auto numElements = 4;
  auto device = createTestDevice(TEST_TARGET, numIpus, tilesPerIpu);

  Graph graph(device.getTarget());
  Sequence uploadProg, prog, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  Tensor testInput =
      graph.addVariable(INT, {numElements}, VariableMappingMethod::LINEAR);

  // create a tensor view with aliases when checking for alias preservation
  const unsigned broadcastFactor = 2;
  auto srcTensor =
      cloneMethod == poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES
          ? testInput.broadcast(broadcastFactor, 0)
          : testInput;

  // test funtion: duplicate (clone+copy) the tensor
  Tensor testResult =
      poputil::duplicate(graph, srcTensor, prog, "testResult", cloneMethod);

  // create unaliased view of the testResult
  Tensor testResultFlat = testResult.flatten();
  auto testResultFlatRegions = graph.getSortedContiguousRegions(
      testResultFlat, {{0, testResultFlat.numElements()}}, true);
  Tensor testResultNA =
      poplar::concat(testResultFlat.slices(testResultFlatRegions));

  // setup Host FIFO
  auto rawHostTestInput = allocateHostMemoryForTensor(
      testInput, "testInput", graph, uploadProg, downloadProg, tmap);
  auto rawHostTestResult = allocateHostMemoryForTensor(
      testResultNA, "testResultNA", graph, uploadProg, downloadProg, tmap);

  // Verify aliases and intervals are identical
  using TyVecA = std::vector<std::size_t>;
  using TyVecVecI = std::vector<std::vector<Interval>>;

  auto GetVecVecIntervals = [&graph](const Tensor &t,
                                     TyVecA &aliases) -> TyVecVecI {
    return graph.getSortedContiguousRegions(t.flatten(), {{0, t.numElements()}},
                                            false, &aliases);
  };

  TyVecA srcAliases, resAliases;
  TyVecVecI srcVVIntervals = GetVecVecIntervals(srcTensor, srcAliases);
  TyVecVecI resVVIntervals = GetVecVecIntervals(testResult, resAliases);

  BOOST_CHECK(srcVVIntervals == resVVIntervals);
  BOOST_CHECK(srcAliases == resAliases);

  // verify duplicated tensor has expected number of elements
  if (cloneMethod == poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES)
    BOOST_CHECK_EQUAL(broadcastFactor * testInput.numElements(),
                      testResult.numElements());
  else
    BOOST_CHECK_EQUAL(testInput.numElements(), testResult.numElements());

  // run Engine and verify copying
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    std::vector<int> hostTestInput(numElements);
    std::vector<int> hostTestResult(numElements);
    auto rawHostTestInputPtr = reinterpret_cast<int *>(rawHostTestInput.get());
    auto rawHostTestResultPtr =
        reinterpret_cast<int *>(rawHostTestResult.get());

    std::mt19937 randomEngine;
    // Run program and check the data is copied correctly to duplicate
    std::generate(hostTestInput.begin(), hostTestInput.end(),
                  randomEngine); // fill random in test input
    std::copy(hostTestInput.begin(), hostTestInput.end(),
              rawHostTestInputPtr); // copy to device
    std::fill(hostTestResult.begin(), hostTestResult.end(), 0); // 0s in result
    std::copy(hostTestResult.begin(), hostTestResult.end(),
              rawHostTestResultPtr); // copy  to device
    engine.run(0);
    std::copy(rawHostTestResultPtr, rawHostTestResultPtr + numElements,
              hostTestResult.begin()); // copy result to host
    BOOST_CHECK_EQUAL_COLLECTIONS(hostTestInput.begin(), hostTestInput.end(),
                                  hostTestResult.begin(), hostTestResult.end());
  });
}

BOOST_AUTO_TEST_CASE(CopyToIpuPreserveOrderUnlessAliasesTest) {
  TestFunc(poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
}

BOOST_AUTO_TEST_CASE(CopyToIpuCreateNewOrderTest) {
  TestFunc(poplar::TensorCloneMethod::CREATE_NEW_ORDER);
}

BOOST_AUTO_TEST_CASE(CopyToIpuPreserveOrderAndAliases) {
  TestFunc(poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
}
