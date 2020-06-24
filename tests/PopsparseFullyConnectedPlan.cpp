// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"

#define BOOST_TEST_MODULE PopsparseFullyConnectedPlan
#include <boost/test/unit_test.hpp>

#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>

#include "popsparse/FullyConnectedOptions.hpp"
#include "popsparse/FullyConnectedParams.hpp"
#include "popsparse/FullyConnectedPlan.hpp"
#include "popsparse/SparsityParams.hpp"

using namespace poplar;

using namespace popsparse;
using namespace popsparse::dynamic;
using namespace popsparse::fullyconnected;

BOOST_AUTO_TEST_CASE(getAPlan) {
  auto device = createTestDevice(TEST_TARGET, 1, 64);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 128;
  constexpr std::size_t numGroups = 2;
  constexpr std::size_t inputChannels = 4096;
  constexpr std::size_t outputChannels = 256;
  poplar::OptionFlags options;
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.1, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, HALF, params, options);
}

BOOST_AUTO_TEST_CASE(getAPlanWithGradA) {
  auto device = createTestDevice(TEST_TARGET, 1, 64);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 128;
  constexpr std::size_t numGroups = 2;
  constexpr std::size_t inputChannels = 4096;
  constexpr std::size_t outputChannels = 256;
  OptionFlags options{{"doGradAPass", "true"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.1, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, HALF, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase1) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 4;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 1024;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, HALF, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase1BS8) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 8;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 1024;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, HALF, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase1Float) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 4;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 1024;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, FLOAT, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase1FloatBS8) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 8;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 1024;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, FLOAT, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase2) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 4;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 32768;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, HALF, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase2BS8) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 8;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 32768;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, HALF, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase2Float) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 4;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 32768;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, FLOAT, params, options);
}

BOOST_AUTO_TEST_CASE(InterestingCase2FloatBS8) {
  auto device = createTestDevice(TEST_TARGET, 1, IPUModel("ipu1").tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  constexpr std::size_t batchSize = 8;
  constexpr std::size_t numGroups = 1;
  constexpr std::size_t inputChannels = 32768;
  constexpr std::size_t outputChannels = 32768;
  poplar::OptionFlags options{{"availableMemoryProportion", "100"}};
  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);
  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), 0.01, batchSize, numGroups, inputChannels,
      outputChannels);
  getPlan(target, FLOAT, params, options);
}
