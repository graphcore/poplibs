// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "BalancedPartitioner.hpp"
#include <algorithm>
#include <float.h>
#include <poplibs_support/logging.hpp>
#include <poputil/exceptions.hpp>

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

void BalancedPartitioner::partition(const std::vector<float> &nodeW,
                                    int nPartition,
                                    std::vector<int> &nodeAssignment) {
  std::vector<std::pair<float, int>> nodes(nodeW.size());
  for (unsigned i = 0; i < nodeW.size(); i++) {
    nodes[i].first = nodeW[i];
    nodes[i].second = i;
  }
  std::sort(nodes.begin(), nodes.end(), std::greater<std::pair<float, int>>());

  std::vector<std::vector<int>> partition(nPartition);
  std::vector<float> partitionW(nPartition, 0.0);

  for (unsigned i = 0; i < nodes.size(); i++) {
    // find the partition that has less weight
    float min = FLT_MAX;
    unsigned minIndex = -1;
    for (unsigned k = 0; k < partition.size(); k++) {
      if (partitionW[k] < min) {
        min = partitionW[k];
        minIndex = k;
      }
    }
    partition[minIndex].push_back(nodes[i].second);
    partitionW[minIndex] += nodes[i].first;
  }

  nodeAssignment.resize(nodes.size());
  for (unsigned k = 0; k < partition.size(); k++) {
    for (unsigned n = 0; n < partition[k].size(); n++) {
      nodeAssignment[partition[k][n]] = k;
    }
  }
}

float BalancedPartitioner::partitionGraph(const HyperGraphData &graphData,
                                          int nPartition,
                                          std::vector<int> &nodeAssignment) {

  partition(graphData.weights, nPartition, nodeAssignment);

  float minWeight, maxWeight, avgWeight, balance;
  int minTileId, maxTileId, zeroTiles;
  computeLoadBalance(graphData.weights, nPartition, nodeAssignment, minWeight,
                     minTileId, maxWeight, maxTileId, avgWeight, balance,
                     zeroTiles);

  logging::info("Min weight {} on tile {}", minWeight, minTileId);
  logging::info("Max weight {} on tile {}", maxWeight, maxTileId);
  logging::info("Average weight {}", avgWeight);
  logging::info("partition load balance {}, number of tile that has no "
                "assignment {}",
                balance, zeroTiles);

  return balance + zeroTiles;
}

} // namespace experimental
} // namespace popsparse