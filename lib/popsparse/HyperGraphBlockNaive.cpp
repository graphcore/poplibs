// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphBlockNaive.hpp"
#include <algorithm>
#include <poplibs_support/logging.hpp>
#include <poputil/exceptions.hpp>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphBlockNaive::HyperGraphBlockNaive(
    const BlockMatrix &A, const BlockMatrix &B, poplar::Type inDataTypeIn,
    poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn, int nTileIn,
    float memoryCycleRatioIn, int nMulsOnVNodeIn)
    : HyperGraphBlock(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn,
                      nTileIn, memoryCycleRatioIn, nMulsOnVNodeIn) {

  logging::popsparse::info("HyperGraphBlockNaive is created");
}

// This algorithm tries to balance computational work equally.
// Then it tries to minimize tiles exchange weight.
//
// The algorithm:
// 1. Distribute V nodes
// 1.1 Assign each V node a compute weight which is proportional
// to number of cycles to run on its vertex.
// 1.2. Distribute V nodes across tiles as evenly as possible
// with regards to compute weight.
// loop through V nodes
//  take the heaviest node
//  put a node on least occupied tile
//  increase total tile's compute weight
//  repeat
// 2. Distribute A,B nodes needed for matmul
// 2.1. Compute total memory for each tile.
// Assign exchange memory to a total memory -
// this is the case when all A and B variables needed to perform
// partial matmul are not present on a tile and must be copied.
// 2.2.
// loop through A,B nodes
//  put a node on a tile with largest exchange memory
//  decrease tile's exchange memory
//  repeat
// 3. Distribute unmapped A,B nodes (not participating oin matmul)
// loop through unmapped A,B nodes
//  put a node on a tile with smallest total memory
//  increase tile's total memory
//  repeat
//
void HyperGraphBlockNaive::partitionGraph() {
  const unsigned inDataTypeSize = (inDataType == poplar::FLOAT) ? 4 : 2;
  const unsigned partialDataTypeSize =
      (partialDataType == poplar::FLOAT) ? 4 : 2;

  unsigned bytesNodeA =
      matA.getBlockRow() * matA.getBlockCol() * inDataTypeSize;
  unsigned bytesNodeB =
      matB.getBlockRow() * matB.getBlockCol() * inDataTypeSize;
  unsigned bytesNodeV =
      matA.getBlockRow() * matB.getBlockCol() * partialDataTypeSize;

  enum NodeType { A, B, C, V };

  //________________<node id, <node type, node index>>
  std::unordered_map<unsigned int, std::pair<NodeType, unsigned int>>
      nodeIdMapping;
  //________________<node V id, node A,B id>
  std::unordered_map<unsigned int, std::unordered_set<unsigned int>>
      nodeVInputMapping;
  //________________<node id>
  std::unordered_set<unsigned> nodesABUnmapped;

#ifndef NDEBUG
  std::set<int> tileAssignmentSet;
  std::pair<std::set<int>::iterator, bool> tileInsertRes;
#endif

  for (std::size_t i = 0; i < nodeA.size(); ++i) {
    const auto &n = nodeA[i];
    nodeIdMapping[n.id] = std::make_pair(NodeType::A, i);
    nodesABUnmapped.insert(n.id);
#ifndef NDEBUG
    tileInsertRes = tileAssignmentSet.emplace(n.id);
    assert(tileInsertRes.second);
#endif
  }
  for (std::size_t i = 0; i < nodeB.size(); ++i) {
    const auto &n = nodeB[i];
    nodeIdMapping[n.id] = std::make_pair(NodeType::B, i);
    nodesABUnmapped.insert(n.id);
#ifndef NDEBUG
    tileInsertRes = tileAssignmentSet.emplace(n.id);
    assert(tileInsertRes.second);
#endif
  }
  for (std::size_t i = 0; i < nodeC.size(); ++i) {
    const auto &n = nodeC[i];
    nodeIdMapping[n.id] = std::make_pair(NodeType::C, i);
#ifndef NDEBUG
    tileInsertRes = tileAssignmentSet.emplace(n.id);
    assert(tileInsertRes.second);
#endif
  }
  for (std::size_t i = 0; i < nodeV.size(); ++i) {
    const auto &n = nodeV[i];
    nodeIdMapping[n.id] = std::make_pair(NodeType::V, i);
#ifndef NDEBUG
    tileInsertRes = tileAssignmentSet.emplace(n.id);
    assert(tileInsertRes.second);
#endif
  }
  for (std::size_t i = 0; i < edgeA.size(); ++i) {
    const auto &e = edgeA[i];
    assert(e.in.size() == 1);
    unsigned int idNodeA = e.in[0];
    assert(nodeIdMapping.at(idNodeA).first == NodeType::A);
    for (std::size_t i_out = 0; i_out < e.out.size(); ++i_out) {
      unsigned int idNodeV = e.out[i_out];
      nodeVInputMapping[idNodeV].insert(idNodeA);
    }
  }
  for (std::size_t i = 0; i < edgeB.size(); ++i) {
    const auto &e = edgeB[i];
    assert(e.in.size() == 1);
    unsigned int idNodeB = e.in[0];
    assert(nodeIdMapping.at(idNodeB).first == NodeType::B);
    for (std::size_t i_out = 0; i_out < e.out.size(); ++i_out) {
      unsigned int idNodeV = e.out[i_out];
      nodeVInputMapping[idNodeV].insert(idNodeB);
    }
  }
  // Prepare output
  std::size_t numNodes =
      nodeA.size() + nodeB.size() + nodeC.size() + nodeV.size();
#ifndef NDEBUG
  assert(numNodes == tileAssignmentSet.size());
  assert(*tileAssignmentSet.begin() == 0);
  assert(*(--tileAssignmentSet.end()) == static_cast<int>(numNodes) - 1);
#endif
  tileAssignment.resize(numNodes, -1);

  //___________________<node id, <num muls, weight>>
  std::vector<std::pair<unsigned int, std::pair<std::size_t, std::size_t>>>
      nodeVWeights;

  // Compute weights
  for (std::size_t i = 0; i < nodeV.size(); ++i) {
    const auto &n = nodeV[i];
    std::size_t compWeight = n.idxA.size();
    std::size_t memWeight =
        n.idxA.size() * (bytesNodeA + bytesNodeB) + bytesNodeV;
    nodeVWeights.emplace_back(n.id, std::make_pair(compWeight, memWeight));
  }

  // Sorting by computational weight
  std::sort(nodeVWeights.begin(), nodeVWeights.end(),
            [&](const auto &elem0, const auto &elem1) {
              return elem0.second.first > elem1.second.first;
            });

  // Creating pool of tiles by minimum compute weight
  //___________________________<muls, tile id>
  std::priority_queue<std::pair<std::size_t, int>,
                      std::vector<std::pair<std::size_t, int>>,
                      // Min priority queue
                      std::greater<std::pair<std::size_t, int>>>
      tilesByCompWeightMinQueue;

  for (int idTile = 0; idTile < nTile; ++idTile) {
    tilesByCompWeightMinQueue.push(std::make_pair(0, idTile));
  }

  //________________<tile id, bytes>
  std::unordered_map<int, std::size_t> tilesByMemWeight;
  //________________<tile id, nodes A,B ids>
  std::unordered_map<int, std::vector<unsigned int>> tilesAndABNodes;

  // Mapping V nodes
  // Take the heaviest to compute node V
  // and put it to the least occupied tile
  for (std::size_t i = 0; i < nodeVWeights.size(); ++i) {
    const auto &nodeInfo = nodeVWeights[i];
    unsigned int idNodeV = nodeInfo.first;
    std::size_t nodeCompWeight = nodeInfo.second.first;
    std::size_t nodeMemWeight = nodeInfo.second.second;

    std::pair<int, std::size_t> tile = tilesByCompWeightMinQueue.top();
    int idTile = tile.second;
    std::size_t tileCompWeight = tile.first;

    tileAssignment[idNodeV] = idTile;
    tileCompWeight += nodeCompWeight;

    tilesByMemWeight[idTile] += nodeMemWeight;

    const auto &inputMapping = nodeVInputMapping.at(idNodeV);
    for (unsigned int idInputNode : inputMapping) {
      tilesAndABNodes[idTile].push_back(idInputNode);
    }
    tilesByCompWeightMinQueue.pop();
    tilesByCompWeightMinQueue.push(std::make_pair(tileCompWeight, idTile));
  }

  // Creating pool of tiles by maximum memory weight
  //___________________________<bytes occupied, tile id>
  std::priority_queue<std::pair<std::size_t, int>,
                      std::vector<std::pair<std::size_t, int>>>
      tilesByMemWeightMaxQueue;
  // Creating pool of tiles by minimum memory weight
  //___________________________<bytes occupied, tile id>
  std::priority_queue<std::pair<std::size_t, int>,
                      std::vector<std::pair<std::size_t, int>>,
                      // Min priority queue
                      std::greater<std::pair<std::size_t, int>>>
      tilesByMemWeightMinQueue;

  for (auto iter = tilesByMemWeight.begin(); iter != tilesByMemWeight.end();
       ++iter) {
    tilesByMemWeightMaxQueue.push(std::make_pair(iter->second, iter->first));
    tilesByMemWeightMinQueue.push(std::make_pair(iter->second, iter->first));
  }

  // Map A,B nodes
  // Try to distribute exchange weight as even as possible
  // At the beginning all weights on tiles are exchange weights
  // When we put variable A or B on tile, tile's exchange weight got reduced for
  // the weight on this variable, because we don't need to copy this variable to
  // this tile at runtime as a part of exchange message
  while (!tilesAndABNodes.empty()) {
    assert(!tilesByMemWeightMaxQueue.empty());
    std::pair<int, std::size_t> tile = tilesByMemWeightMaxQueue.top();
    tilesByMemWeightMaxQueue.pop();
    int idTile = tile.second;
    std::size_t tileMemWeight = tile.first;

    auto iter = tilesAndABNodes.find(idTile);
    if (iter != tilesAndABNodes.end()) {
      auto &nodesAB = iter->second;
      assert(!nodesAB.empty());
      while (!nodesAB.empty()) {
        unsigned int idABNode = nodesAB.back();
        nodesAB.pop_back();
        auto iterUnmapped = nodesABUnmapped.find(idABNode);
        if (iterUnmapped != nodesABUnmapped.end()) {
          nodesABUnmapped.erase(iterUnmapped);

          tileAssignment[idABNode] = idTile;

          const auto &nodeInfo = nodeIdMapping.at(idABNode);
          std::size_t bytes =
              nodeInfo.first == NodeType::A ? bytesNodeA : bytesNodeB;
          if (tileMemWeight >= bytes) {
            std::size_t tileMemNewWeight = tileMemWeight - bytes;
            tilesByMemWeightMaxQueue.push(
                std::make_pair(tileMemNewWeight, idTile));
          }
          break;
        }
      }
      if (nodesAB.empty()) {
        tilesAndABNodes.erase(idTile);
      }
    }
  }

  // Map unmapped A,B nodes
  // Take a node and put it on less occupied tile
  while (!nodesABUnmapped.empty()) {
    auto iter = nodesABUnmapped.begin();
    unsigned int idABNode = *iter;
    nodesABUnmapped.erase(iter);

    std::pair<int, std::size_t> tile = tilesByMemWeightMinQueue.top();
    tilesByMemWeightMinQueue.pop();
    int idTile = tile.second;
    std::size_t tileMemWeight = tile.first;

    tileAssignment[idABNode] = idTile;

    const auto &nodeInfo = nodeIdMapping.at(idABNode);
    std::size_t bytes = nodeInfo.first == NodeType::A ? bytesNodeA : bytesNodeB;
    std::size_t tileMemNewWeight = tileMemWeight + bytes;
    tilesByMemWeightMinQueue.push(std::make_pair(tileMemNewWeight, idTile));
  }

  for (const auto &n : nodeA) {
    if (tileAssignment[n.id] < 0) {
      throw poputil::poplibs_error("invalid tile for node A " +
                                   std::to_string(n.id));
    }
  }

  for (const auto &n : nodeB) {
    if (tileAssignment[n.id] < 0) {
      throw poputil::poplibs_error("invalid tile for node B " +
                                   std::to_string(n.id));
    }
  }

  for (const auto &n : nodeV) {
    if (tileAssignment[n.id] < 0) {
      throw poputil::poplibs_error("invalid tile for node V " +
                                   std::to_string(n.id));
    }
  }
}

} // namespace experimental
} // namespace popsparse
