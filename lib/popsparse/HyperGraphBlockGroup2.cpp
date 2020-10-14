// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphBlockGroup2.hpp"
#include <algorithm>
#include <cmath>
#include <poplibs_support/logging.hpp>
#include <poputil/exceptions.hpp>
#include <stdlib.h>
#include <unordered_set>
#include <vector>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphBlockGroup2::HyperGraphBlockGroup2(BlockMatrix &A, BlockMatrix &B,
                                             poplar::Type inDataTypeIn,
                                             poplar::Type outDataTypeIn,
                                             poplar::Type partialDataTypeIn,
                                             int nTileIn)
    : HyperGraphBlockGroup(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn,
                           nTileIn) {

  logging::popsparse::info("HyperGraphBlockGroup2 is created");
}

void HyperGraphBlockGroup2::populateNodesV(
    int nRowC, int nColC, const std::vector<std::vector<int>> &) {

  nodeVLayout.resize(
      nRowC, std::vector<std::unordered_map<unsigned, std::size_t>>(nColC));
  logging::popsparse::trace("Muls total: {}", numMuls);
  // Nodes are during graph partitioning
}

HyperGraphBlock::ComputeNode HyperGraphBlockGroup2::populateNodeV(
    unsigned row, unsigned col, unsigned p, int &nodeId,
    const std::vector<std::pair<unsigned int, unsigned int>> &aList,
    const std::vector<std::pair<unsigned int, unsigned int>> &bList) {
  std::vector<unsigned int> nodeAList, nodeBList;
  for (unsigned m = 0; m < aList.size(); m++) {
    unsigned int rmul = aList[m].first;
    unsigned int cmul = aList[m].second;
    nodeAList.push_back(blockIdMatrixA[rmul][cmul]);
    edgeA[hyperEdgeIdA[rmul][cmul]].out.push_back(nodeId);

    rmul = bList[m].first;
    cmul = bList[m].second;
    nodeVIdxById[nodeId] = nodeV.size();
    nodeBList.push_back(blockIdMatrixB[rmul][cmul]);
    edgeB[hyperEdgeIdB[rmul][cmul]].out.push_back(nodeId);
  }
  ComputeNode n(nodeId, nodeAList, nodeBList, row, col, p);
  nodeV.push_back(n);
  edgeC[hyperEdgeIdC[row][col]].in.push_back(nodeId);

  ++nodeId;
  return n;
}

void HyperGraphBlockGroup2::partitionGraph() {
  unsigned numP0 = P;
  bool doRowSplitBest = doRowSplit;
  const int P_SEARCH_SPACE = 5;

  // Note - for row split we have 2 models describing extreme cases of
  // cumulative sparsity. Because we don't know which one is more correct, we
  // simply pick the best and search for the minimum weight. Because the minimum
  // is global, this approach works.
  std::size_t maxBytesOnTile = 0;
  // Start from the best estimation
  partitionGraphSearch(maxBytesOnTile, false);
  unsigned numPBest = P;
  std::size_t maxBytesOnTileBest = maxBytesOnTile;
  for (int i = 0; i < P_SEARCH_SPACE; ++i) {
    ++P;
    // Increase P until weight is improving
    partitionGraphSearch(maxBytesOnTile, false);
    if (maxBytesOnTile < maxBytesOnTileBest) {
      numPBest = P;
      maxBytesOnTileBest = maxBytesOnTile;
    } else {
      break;
    }
  }
  P = numP0;
  for (int i = 0; i < P_SEARCH_SPACE && P > 1; ++i) {
    --P;
    // Decrease P until weight is improving
    partitionGraphSearch(maxBytesOnTile, false);
    if (maxBytesOnTile < maxBytesOnTileBest) {
      numPBest = P;
      maxBytesOnTileBest = maxBytesOnTile;
    } else {
      break;
    }
  }

  float wAlt;
  unsigned numPAlt;
  if (doRowSplit) {
    wAlt = wTileEstColSplit;
    numPAlt = estPColSplit;
  } else {
    wAlt = wTileEstRowSplit;
    numPAlt = estPRowSplit[0];
  }

  if (wAlt < maxBytesOnTileBest * 1.3) {
    // Try alternative row/column split
    doRowSplit = !doRowSplit;
    P = numPAlt;
    partitionGraphSearch(maxBytesOnTile, false);
    if (maxBytesOnTile < maxBytesOnTileBest) {
      numPBest = P;
      maxBytesOnTileBest = maxBytesOnTile;
      doRowSplitBest = doRowSplit;
    }
    for (int i = 0; i < P_SEARCH_SPACE; ++i) {
      ++P;
      partitionGraphSearch(maxBytesOnTile, false);
      if (maxBytesOnTile < maxBytesOnTileBest) {
        numPBest = P;
        maxBytesOnTileBest = maxBytesOnTile;
        doRowSplitBest = doRowSplit;
      } else {
        break;
      }
    }
    P = numPAlt;
    ;
    for (int i = 0; i < P_SEARCH_SPACE && P > 1; ++i) {
      --P;
      partitionGraphSearch(maxBytesOnTile, false);
      if (maxBytesOnTile < maxBytesOnTileBest) {
        numPBest = P;
        maxBytesOnTileBest = maxBytesOnTile;
        doRowSplitBest = doRowSplit;
      } else {
        break;
      }
    }
  }
  // Performing final partitioning to populate nodes in most optimal way
  P = numPBest;
  doRowSplit = doRowSplitBest;
  partitionGraphSearch(maxBytesOnTile, true);
}

void HyperGraphBlockGroup2::partitionGraphSearch(std::size_t &maxBytesOnTile,
                                                 bool final) {
  logging::popsparse::debug("{} {} split with k parts = {}",
                            final ? "Performing" : "Trying",
                            doRowSplit ? "row" : "column", P);

  nodeV.clear();
  nodeVIdxById.clear();
  nodeVLayout.clear();
  for (auto &e : edgeA) {
    e.out.clear();
  }
  for (auto &e : edgeB) {
    e.out.clear();
  }
  for (auto &e : edgeC) {
    e.in.clear();
  }
  nodeAPartitions.clear();
  nodeBPartitions.clear();
  nodeCInputMapping.clear();

  unsigned blockRowsC = static_cast<unsigned>(matC->getBlockRowCount());
  unsigned blockColsC = static_cast<unsigned>(matC->getBlockColCount());
  unsigned blockColsA = static_cast<unsigned>(matA.getBlockColCount());

  std::vector<std::vector<std::unordered_set<int>>> tilesByRK(
      blockRowsC, std::vector<std::unordered_set<int>>(P));
  std::vector<std::vector<std::unordered_set<int>>> tilesByCK(
      blockColsC, std::vector<std::unordered_set<int>>(P));

  std::vector<W> tilesWeightsMatMul(nTile);
  std::vector<int> tilesWeightsReduce(nTile);

  std::size_t numNodes = nodeA.size() + nodeB.size() + nodeC.size();
  tileAssignment.resize(numNodes, -1);

  std::size_t mulsOnTileMaxDbg = 0;
  int tileWithMaxMulsDbg = -1;
  std::size_t mulsOnTileMinDbg = 0xFFFFFFFF;
  int tileWithMinMulsDbg = -1;

  auto blockIdMatrixC = matC->getBlockIdMatrix();

  int nodeId = gNodeId;

  std::vector<std::vector<unsigned>> kOffsets(
      blockRowsC, std::vector<unsigned>(blockColsC, 0));

  const unsigned partitionSize = static_cast<unsigned>(blockColsA) / P;
  const unsigned partitionSizeLfto = static_cast<unsigned>(blockColsA) % P;

  const std::size_t mulsPerTile = numMuls / nTile;
  const std::size_t mulsPerTileLfto = numMuls % nTile;
  if (final) {
    if (mulsPerTile > partitionSize) {
      logging::popsparse::trace("Muls per tile: {}-{}", mulsPerTile,
                                (mulsPerTileLfto == 0) ? mulsPerTile
                                                       : (mulsPerTile + 1));
    } else {
      logging::popsparse::trace(
          "Muls per tile: ({}){}-{}", mulsPerTile, partitionSize,
          (partitionSizeLfto == 0) ? partitionSize : (partitionSize + 1));
    }
  }

  TaDbg taDbg(P, blockRowsC, blockColsC, blockColsA);

  int idTile = 0;
  std::size_t mulsOnTile = 0;

  std::function<void(unsigned, unsigned, unsigned, const ComputeNode &)>
      updateTilePlacement =
          [&](unsigned p, unsigned c, unsigned r, const ComputeNode &nv) {
            std::size_t wMatMul = nv.idxA.size() * (wA + wB) + wV;
            tilesWeightsMatMul[idTile].wTmp += wMatMul;
            tilesWeightsMatMul[idTile].wTotal += wMatMul;

            tilesWeightsReduce[idTile] += wP;

            assert(tileAssignment.size() == nv.id);
            assert(nv.blockRow == r);
            assert(nv.blockCol == c);
            assert(nv.partition == p);
            tileAssignment.push_back(idTile);

            tilesByRK[r][p].insert(idTile);
            tilesByCK[c][p].insert(idTile);

            taDbg.setV(r, c, p, idTile);
          };

  std::function<void(unsigned, unsigned, unsigned, bool)> placeNodeV =
      [&](unsigned p, unsigned c, unsigned r, bool changeTile) {
        if (blockIdMatrixC[r][c] == -1) {
          return;
        }

        std::size_t mulsPerTileCur;
        if (mulsPerTile > partitionSize) {
          mulsPerTileCur = static_cast<std::size_t>(idTile) < mulsPerTileLfto
                               ? (mulsPerTile + 1)
                               : mulsPerTile;
        } else {
          mulsPerTileCur =
              p < partitionSizeLfto ? (partitionSize + 1) : partitionSize;
        }
        unsigned kMin = p <= partitionSizeLfto
                            ? (partitionSize + 1) * p
                            : (partitionSize + 1) * partitionSizeLfto +
                                  partitionSize * (p - partitionSizeLfto);
        unsigned kMax = p < partitionSizeLfto
                            ? (partitionSize + 1) * (p + 1)
                            : (partitionSize + 1) * partitionSizeLfto +
                                  partitionSize * (p + 1 - partitionSizeLfto);
        kMax = std::min(kMax, blockColsA);

        std::vector<std::pair<unsigned int, unsigned int>> aList, bList;
        for (unsigned k = kMin; k < kMax; ++k) {
          if (blockIdMatrixA[r][k] == -1 || blockIdMatrixB[k][c] == -1) {
            continue;
          }

          aList.push_back(std::make_pair(r, k));
          bList.push_back(std::make_pair(k, c));
          ++mulsOnTile;

          if (mulsOnTile >= mulsPerTileCur || changeTile) {
            if (logging::popsparse::shouldLog(logging::Level::Trace)) {
              if (mulsOnTileMaxDbg < mulsOnTile) {
                mulsOnTileMaxDbg = mulsOnTile;
                tileWithMaxMulsDbg = idTile;
              }
              if (mulsOnTileMinDbg > mulsOnTile) {
                mulsOnTileMinDbg = mulsOnTile;
                tileWithMinMulsDbg = idTile;
              }
            }

            auto nv = populateNodeV(r, c, p, nodeId, aList, bList);
            updateTilePlacement(p, c, r, nv);
            aList.clear();
            bList.clear();

            idTile = (idTile + 1) % nTile;
            mulsOnTile = 0;
          }
        }
        if (!aList.empty()) {
          auto nv = populateNodeV(r, c, p, nodeId, aList, bList);
          updateTilePlacement(p, c, r, nv);
        }
      };

  bool changeTile = false;
  if (!doRowSplit) {
    for (unsigned p = 0; p < P; ++p) {
      for (unsigned c = 0; c < blockColsC; ++c) {
        for (unsigned r1 = 0; r1 < blockRowsC; ++r1) {
          unsigned r = (c % 2 == 0) ? r1 : blockRowsC - 1 - r1;
          placeNodeV(p, c, r, changeTile);
          changeTile = false;
        }
      }
    }
  } else {
    for (unsigned p = 0; p < P; ++p) {
      for (unsigned r = 0; r < blockRowsC; ++r) {
        for (unsigned c1 = 0; c1 < blockColsC; ++c1) {
          unsigned c = (r % 2 == 0) ? c1 : blockColsC - 1 - c1;
          placeNodeV(p, c, r, changeTile);
          changeTile = false;
        }
      }
    }
  }
  if (final) {
    logging::popsparse::trace("muls on tile {} max: {}", tileWithMaxMulsDbg,
                              mulsOnTileMaxDbg);
    logging::popsparse::trace("muls on tile {} min: {}", tileWithMinMulsDbg,
                              mulsOnTileMinDbg);
  }

  fillNodeAB2VMapping();

  placeABNodes(tilesByCK, tilesByRK, tilesWeightsMatMul, taDbg);

  placeCNodes(tilesWeightsReduce, taDbg);

  if (final) {
    logTilesMapping(taDbg);
  } else {
    MemoryStatistics stats;
    computeBytesByTile(stats);
    maxBytesOnTile = stats.maxBytesOnTile;
    logging::popsparse::debug(
        "Max Kb on a tile = {}. Matmul = {}; Reduce = {}. Min Kb"
        " on a tile = {}",
        stats.maxBytesOnTile / KBf, stats.maxBytesOnTileMatmul / KBf,
        stats.maxBytesOnTileReduce / KBf, stats.minBytesOnTile / KBf);
  }
}

void HyperGraphBlockGroup2::mapCNodes(poplar::Graph &graph) {
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  for (std::size_t i = 0; i < nodeC.size(); i++) {
    unsigned int blockId = nodeC[i].blockId;
    graph.setTileMapping(blockDataC[blockId], nodeCTileId[i]);
  }
}

} // namespace experimental
} // namespace popsparse