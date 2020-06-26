// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "HyperGraphPartitioner.hpp"
#include <float.h>
#include <poplibs_support/logging.hpp>

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

void HyperGraphPartitioner::computeLoadBalance(
    const std::vector<float> &nodeWeight, int nPartition,
    std::vector<int> &nodeAssignment, float &minWeight, int &minTileId,
    float &maxWeight, int &maxTileId, float &avgWeight, float &balance,
    int &zeroTiles) {
  // compute load balance
  std::vector<float> tileWeight(nPartition, 0.0f);

  float totalWeight = 0.0f;
  for (unsigned n = 0; n < nodeWeight.size(); n++) {
    tileWeight[nodeAssignment[n]] += nodeWeight[n];
    totalWeight += nodeWeight[n];
  }

  minWeight = FLT_MAX;
  maxWeight = -FLT_MAX;
  avgWeight = totalWeight / nPartition;
  balance = -FLT_MAX;
  zeroTiles = 0;
  for (int i = 0; i < nPartition; i++) {
    if (tileWeight[i] == 0.0) {
      zeroTiles++;
      continue;
    }

    if (tileWeight[i] < minWeight) {
      minWeight = tileWeight[i];
      minTileId = i;
    }
    if (tileWeight[i] > maxWeight) {
      maxWeight = tileWeight[i];
      maxTileId = i;
    }

    float b = 1.0f;
    if (tileWeight[i] > avgWeight) {
      b = tileWeight[i] / avgWeight;
    } else {
      b = avgWeight / tileWeight[i];
    }
    balance = std::max(b, balance);
  }
}

} // namespace experimental
} // namespace popsparse