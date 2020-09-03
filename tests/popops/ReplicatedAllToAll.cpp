// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_test/Util.hpp"
#include <iostream>
#include <math.h>
#include <poplar/Engine.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#define BOOST_TEST_MODULE ReplicatedAllToAll
#include <poplibs_support/TestDevice.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
namespace pe = popops::expr;

void RunAllToAll() {
  const unsigned numIpus = 8;

  auto device = createTestDevice(TEST_TARGET, numIpus, 1);
  Target target = device.getTarget();

  Graph graph(target, poplar::replication_factor(numIpus));
  popops::addCodelets(graph);

  Sequence sequence;

  std::vector<unsigned> v;
  for (unsigned i = 0; i < numIpus; ++i) {
    v.push_back(i);
  }

  auto c0 = graph.addConstant<unsigned>(UNSIGNED_INT, {numIpus}, v, "c0");
  graph.setTileMapping(c0, 0);
  auto t0 = graph.addVariable(UNSIGNED_INT, {numIpus},
                              VariableMappingMethod::LINEAR, "t0");

  std::vector<std::pair<std::string, char *>> tmap;

  Sequence uploadProg, downloadProg;
  auto inputHost = poplibs_test::util::allocateHostMemoryForTensor(
      t0, "input", graph, uploadProg, downloadProg, tmap);

  sequence.add(poplar::program::Copy(c0, t0));

  Tensor replicationFactorTensor = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationFactorTensor, 0);

  poplar::Tensor mul =
      popops::mul(graph, replicationFactorTensor, numIpus, sequence);
  popops::addInPlace(graph, t0, mul, sequence);

  poplar::Tensor output =
      popops::allToAllPersonalizedExchange(graph, t0, sequence, "AllToAll");

  auto outputHost = poplibs_test::util::allocateHostMemoryForTensor(
      output, "output", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, sequence, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    poplibs_test::util::attachStreams(engine, tmap);

    engine.run();
  });

  // Check the inputs.
  for (unsigned i = 0; i < numIpus * numIpus; ++i) {
    std::uint32_t *ptr = (std::uint32_t *)(inputHost.get() + i * 4);
    std::uint32_t value = *ptr;
    BOOST_TEST(i == value);
  }

  // Check the outputs.
  std::vector<std::uint32_t> groundTruthOutput{
      0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
      2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
      4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
      6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};

  for (unsigned i = 0; i < numIpus * numIpus; ++i) {
    std::uint32_t *ptr = (std::uint32_t *)(outputHost.get() + i * 4);
    std::uint32_t value = *ptr;
    BOOST_TEST(groundTruthOutput[i] == value);
  }
}

void RunAllGather(const OptionFlags &options) {
  const unsigned numIpus = 8;
  const unsigned cubeRootIpus = 2;

  auto device = createTestDevice(TEST_TARGET, numIpus, 1);
  Target target = device.getTarget();

  Graph graph(target, poplar::replication_factor(numIpus));
  popops::addCodelets(graph);

  Sequence sequence;

  std::vector<unsigned> v;
  for (unsigned i = 0; i < numIpus; ++i) {
    v.push_back(i);
  }

  auto c0 = graph.addConstant<unsigned>(
      UNSIGNED_INT, {cubeRootIpus, cubeRootIpus, cubeRootIpus}, v, "c0");
  graph.setTileMapping(c0, 0);
  auto t0 = graph.addVariable(UNSIGNED_INT,
                              {cubeRootIpus, cubeRootIpus, cubeRootIpus},
                              VariableMappingMethod::LINEAR, "t0");

  std::vector<std::pair<std::string, char *>> tmap;

  Sequence uploadProg, downloadProg;
  auto inputHost = poplibs_test::util::allocateHostMemoryForTensor(
      t0, "input", graph, uploadProg, downloadProg, tmap);

  sequence.add(poplar::program::Copy(c0, t0));

  Tensor replicationFactorTensor = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationFactorTensor, 0);

  poplar::Tensor mul =
      popops::mul(graph, replicationFactorTensor, numIpus, sequence);
  popops::addInPlace(graph, t0, mul, sequence);

  poplar::Tensor output =
      popops::replicatedAllGather(graph, t0, sequence, "", options);

  // Output is expected to have shape [numReplicas, 2, 2, 2]
  BOOST_TEST(output.shape().size() == 4);
  BOOST_TEST(output.shape()[0] == 8);
  BOOST_TEST(output.shape()[1] == 2);
  BOOST_TEST(output.shape()[2] == 2);
  BOOST_TEST(output.shape()[3] == 2);

  auto outputHost = poplibs_test::util::allocateHostMemoryForTensor(
      output, "output", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, sequence, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    poplibs_test::util::attachStreams(engine, tmap);

    engine.run();
  });

  // Check the inputs
  for (unsigned i = 0; i < numIpus * numIpus; ++i) {
    std::uint32_t *ptr = (std::uint32_t *)(inputHost.get() + i * 4);
    std::uint32_t value = *ptr;
    BOOST_TEST(i == value);
  }

  // Check the outputs, we expect the output to be sized
  // [numIpus][sizeof(input)] and we have one per IPU. So in this case
  // numIpus*numIpus 0-N patterns numIpus times.
  std::uint32_t *ptr = (std::uint32_t *)outputHost.get();
  for (unsigned j = 0; j < numIpus; ++j) {
    for (unsigned i = 0; i < numIpus * numIpus; ++i) {
      std::uint32_t value = *ptr;
      BOOST_TEST(i == value);
      ptr++;
    }
  }
}

BOOST_AUTO_TEST_CASE(RunAllToAllTest) { RunAllToAll(); }
BOOST_AUTO_TEST_CASE(RunAllGatherClockwiseRingTest) {
  RunAllGather({{"method", "clockwise_ring"}});
}
BOOST_AUTO_TEST_CASE(RunAllGatherBidirectionalRingPairTest) {
  RunAllGather({{"method", "bidirectional_ring_pair"}});
}
BOOST_AUTO_TEST_CASE(RunAllGatherMeetInMiddleRingTest) {
  RunAllGather({{"method", "meet_in_middle_ring"}});
}
