// Copyright (c) 2020 Graphcore Ltd, All rights reserved.
#include "poplibs_test/Util.hpp"
#include <poplar/Engine.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include "TestDevice.hpp"
#define BOOST_TEST_MODULE ReplicatedReduceScatter
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;

BOOST_AUTO_TEST_CASE(ReduceScatterOneIPUPerReplica) {
  const unsigned numIpus = 8;
  const unsigned numReplicas = 8;

  auto device = createTestDevice(TEST_TARGET, numIpus, 1);
  Target target = device.getTarget();

  Graph graph(target, 0, replication_factor(numReplicas));
  popops::addCodelets(graph);

  // Add one element to verify padding behaviour
  const unsigned numElements = numReplicas + 1;
  const unsigned numElementsPerReplica = 2; // ceil(numElements / numReplicas)

  std::vector<unsigned> rangeVector(numElements);
  std::iota(rangeVector.begin(), rangeVector.end(), 0);

  auto rangeConst = graph.addConstant<unsigned>(UNSIGNED_INT, {numElements},
                                                rangeVector, "rangeConst");
  graph.setTileMapping(rangeConst, 0);
  auto rangeVar = graph.addVariable(UNSIGNED_INT, {numElements},
                                    VariableMappingMethod::LINEAR, "rangeVar");

  Sequence sequence;
  sequence.add(Copy(rangeConst, rangeVar));

  Tensor replicationIndex = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationIndex, 0);

  Tensor input = popops::mul(graph, replicationIndex, rangeVar, sequence);

  Tensor output = popops::replicatedReduceScatter(
      graph, input, popops::Operation::ADD, sequence);

  BOOST_REQUIRE_EQUAL(output.rank(), 1);
  BOOST_REQUIRE_EQUAL(output.dim(0), numElementsPerReplica);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  auto outputHost = poplibs_test::util::allocateHostMemoryForTensor(
      output, "output", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, sequence, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    poplibs_test::util::attachStreams(engine, tmap);
    engine.run();
  });

  auto actualOutput = reinterpret_cast<const std::uint32_t *>(outputHost.get());
  for (std::uint32_t i = 0; i < numElements; ++i) {
    // The output should be a sum of an arithmetic series:
    // sum([0, i * 1, i * 2, ..., i * (numIpus - 1)]) =
    // i * numIpus * (numIpus - 1) / 2
    const auto expectedOutput = i * numReplicas * (numReplicas - 1) / 2;
    BOOST_REQUIRE_EQUAL(actualOutput[i], expectedOutput);
  }

  // The rest should be zero-padding
  const auto numElementsAfterPadding = numElementsPerReplica * numReplicas;
  for (std::uint32_t i = numElements; i < numElementsAfterPadding; ++i) {
    BOOST_REQUIRE_EQUAL(actualOutput[i], 0);
  }
}

BOOST_AUTO_TEST_CASE(ReduceScatterTwoIPUsPerReplica) {
  const unsigned numIPUs = 8;
  const unsigned numReplicas = 4;
  const unsigned tilesPerIPU = 1;

  auto device = createTestDevice(TEST_TARGET, numIPUs, tilesPerIPU);
  Target target = device.getTarget();

  Graph graph(target, 0, replication_factor(numReplicas));
  popops::addCodelets(graph);

  const unsigned numIPUsPerReplica = numIPUs / numReplicas;
  const unsigned numElements = 4;

  std::vector<unsigned> rangeVector(numElements);
  std::iota(rangeVector.begin(), rangeVector.end(), 0);

  auto range = graph.addConstant<unsigned>(UNSIGNED_INT, {numElements},
                                           rangeVector, "range");
  graph.setTileMapping(range, 0);
  auto input = graph.addVariable(UNSIGNED_INT, {numElements},
                                 VariableMappingMethod::NONE, "input");

  auto ipuToTensorMapping = Graph::TileToTensorMapping(numIPUsPerReplica);
  ipuToTensorMapping[0].push_back(Interval(0, 3)); // IPU0
  ipuToTensorMapping[1].push_back(Interval(3, 4)); // IPU1
  graph.setTileMapping(input, ipuToTensorMapping);

  // ceil(numElementsIPU0 / numReplicas) + ceil(numElementsIPU1 / numReplicas):
  const unsigned numElementsPerReplica = 2;

  Sequence sequence;
  sequence.add(Copy(range, input));

  Tensor output = popops::replicatedReduceScatter(
      graph, input, popops::Operation::ADD, sequence);

  BOOST_REQUIRE_EQUAL(output.rank(), 1);
  BOOST_REQUIRE_EQUAL(output.dim(0), numElementsPerReplica);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  auto outputHost = poplibs_test::util::allocateHostMemoryForTensor(
      output, "output", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, sequence, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    poplibs_test::util::attachStreams(engine, tmap);
    engine.run();
  });

  // Replica0: input[0, 1, 2, 3]
  // Replica1: input[0, 1, 2, 3]
  // Replica2: input[0, 1, 2, 3]
  // Replica3: input[0, 1, 2, 3]
  // Mapping:  input[IPU0, IPU0, IPU0, IPU1]

  // Replica0: output[0, 12]
  // Replica1: output[4, 0]
  // Replica2: output[8, 0]
  // Replica3: output[0, 0]
  // Mapping:  output[IPU0, IPU1]

  auto actualOutputPtr =
      reinterpret_cast<const std::uint32_t *>(outputHost.get());
  unsigned expectedOutput = 0;
  for (unsigned i = 0; i < numIPUsPerReplica; ++i) {
    BOOST_REQUIRE_EQUAL(ipuToTensorMapping[i].size(), 1);
    const unsigned numElementsOnThisIPU = ipuToTensorMapping[i][0].size();

    // We are testing maximum one element per IPU per replica
    BOOST_REQUIRE_LE(numElementsOnThisIPU, numReplicas);
    const unsigned numElementsPadded = numReplicas;

    // The result should be scattered across the replicas
    for (unsigned j = 0; j < numElementsOnThisIPU; ++j) {
      const auto actualOutput = actualOutputPtr[j * numElementsPerReplica + i];
      BOOST_CHECK_EQUAL(actualOutput, expectedOutput);
      expectedOutput += numReplicas;
    }

    // The rest should be zero-padding
    for (unsigned j = numElementsOnThisIPU; j < numElementsPadded; ++j) {
      const auto actualOutput = actualOutputPtr[j * numElementsPerReplica + i];
      BOOST_CHECK_EQUAL(actualOutput, 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReduceScatterWithRank2InputIsInvalid) {
  const unsigned numReplicas = 8;

  auto device = createTestDevice(TEST_TARGET, numReplicas, 1);
  Target target = device.getTarget();

  Graph graph(target, 0, replication_factor(numReplicas));
  popops::addCodelets(graph);

  auto inputOfRank2 = graph.addVariable(UNSIGNED_INT, {2, 2},
                                        VariableMappingMethod::LINEAR, "input");

  Sequence sequence;

  BOOST_CHECK_THROW(popops::replicatedReduceScatter(
                        graph, inputOfRank2, popops::Operation::ADD, sequence),
                    poputil::poplibs_error);
}
