// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphBlockZoltan.hpp"
#include "poplibs_support/logging.hpp"
#include <algorithm>
#include <vector>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphBlockZoltan::HyperGraphBlockZoltan(
    BlockMatrix &A, BlockMatrix &B, poplar::Type inDataTypeIn,
    poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn, int nTileIn,
    float memoryCycleRatioIn, int nTargetNodesVPerTileIn)
    : HyperGraphBlock(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn,
                      nTileIn, nTargetNodesVPerTileIn),
      memoryCycleRatio(memoryCycleRatioIn) {

  partitioner = std::make_unique<ZoltanPartitioner>(
      ZoltanPartitioner::PartitionType::HYPERGRAPH);

  logging::popsparse::info("HyperGraphBlockZoltan is created");
}

HyperGraphData HyperGraphBlockZoltan::getDataForPartitioner() {
  logging::popsparse::info("Number of nodes in A: {}", nodeA.size());
  logging::popsparse::info("Number of nodes in B: {}", nodeB.size());
  logging::popsparse::info("Number of nodes in V: {}", nodeV.size());

  HyperGraphData graph;

  graph.nodes = nodeA.size() + nodeB.size() + nodeC.size() + nodeV.size();

  // The pins vector stores the indices of the vertices in each of the edges.
  std::vector<unsigned int> pins;

  // The hyperedge vector stores the offsets into the pins vector.
  std::vector<unsigned int> hyperEdges;
  std::vector<float> weights(graph.nodes, 0.0F);

  for (const auto &n : nodeA) {
    weights[n.id] = n.w;
  }

  for (const auto &n : nodeB) {
    weights[n.id] = n.w;
  }

  for (const auto &n : nodeV) {
    weights[n.id] = n.w;
  }

  hyperEdges.reserve(edgeA.size() + edgeB.size());

  for (const auto &e : edgeA) {
    std::vector<unsigned int> v(e.in);

    if (!e.out.empty()) {
      v.insert(v.end(), e.out.begin(), e.out.end());
    }

    std::sort(v.begin(), v.end());
    hyperEdges.push_back(pins.size());
    pins.insert(pins.end(), v.begin(), v.end());
  }

  for (const auto &e : edgeB) {
    std::vector<unsigned int> v(e.in);

    if (!e.out.empty()) {
      v.insert(v.end(), e.out.begin(), e.out.end());
    }

    std::sort(v.begin(), v.end());
    hyperEdges.push_back(pins.size());
    pins.insert(pins.end(), v.begin(), v.end());
  }

  logging::popsparse::info("Number of pins is {}", pins.size());
  logging::popsparse::info("Number of edges is {}", hyperEdges.size());

  graph.pins = std::move(pins);
  graph.hyperEdges = std::move(hyperEdges);
  graph.weights = std::move(weights);

  return graph;
}

void HyperGraphBlockZoltan::setupWeights(const poplar::Graph &graph) {
  float memA = matA.getBlockRow() * matA.getBlockCol() *
               graph.getTarget().getTypeSize(inDataType);
  float memB = matB.getBlockRow() * matB.getBlockCol() *
               graph.getTarget().getTypeSize(inDataType);
  float memV = matC->getBlockRow() * matC->getBlockCol() *
               graph.getTarget().getTypeSize(partialDataType);
  // set the weight for node
  float memWeight =
      1.0f / (memA * nodeA.size() + memB * nodeB.size() + memV * nodeV.size());
  for (auto &n : nodeA) {
    n.w = memA * memWeight * memoryCycleRatio;
  }
  for (auto &n : nodeB) {
    n.w = memB * memWeight * memoryCycleRatio;
  }
  // We allocate nodes C manually later in the code.

  for (auto &n : nodeV) {
    float cycleWeight = static_cast<float>(n.idxA.size()) / numMuls;
    n.w = memoryCycleRatio * memV * memWeight +
          (1.0 - memoryCycleRatio) * cycleWeight;
  }
}

void HyperGraphBlockZoltan::partitionGraph() {
  HyperGraphData data = getDataForPartitioner();
  partitioner->partitionGraph(data, nTile, tileAssignment);
}

} // namespace experimental
} // namespace popsparse
