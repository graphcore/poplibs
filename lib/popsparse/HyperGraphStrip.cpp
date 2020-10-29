// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphStrip.hpp"
#include "BalancedPartitioner.hpp"
#include <algorithm>
#include <cfloat>
#include <numeric>
#include <poplibs_support/logging.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <poputil/exceptions.hpp>
#include <unordered_set>
#include <zoltan_cpp.h>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphStrip::HyperGraphStrip(BlockMatrix &A, BlockMatrix &B,
                                 poplar::Type inDataTypeIn,
                                 poplar::Type outDataTypeIn,
                                 poplar::Type partialDataTypeIn, int nTileIn,
                                 int nPassIn)
    : HyperGraph(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn, nTileIn),
      gNodeId(0), gEdgeId(0), nPass(nPassIn), isResultSparse(false) {

  partitioner = std::make_unique<ZoltanPartitioner>(
      ZoltanPartitioner::PartitionType::BLOCK);
  // TODO: BalancedPartitioner does not work very well for strip-256x4.txt,
  //       Need to look into it
  // partitioner = std::make_unique<BalancedPartitioner>();

  if (A.isDense() && !B.isDense()) {
    isResultSparse = false;
  } else if (A.isDense() && B.isDense()) {
    isResultSparse = true;
  } else {
    assert(0);
  }

  logging::popsparse::info("HyperGraphStrip is created");
}

void HyperGraphStrip::createGraphMatMul(poplar::Graph &graph,
                                        const std::string &debugPrefix) {
  assert(matA.isDense() && !matB.isDense());

  assert(matC == nullptr);
  assert(matA.getColCount() == matB.getRowCount());
  assert(matA.getBlockCol() == matB.getBlockRow());

  // Populate matrix C
  const int blockRowC = matA.getBlockRow();
  const int blockColC = matB.getBlockCol();
  const int rowC = matA.getRowCount();
  const int colC = matB.getColCount();

  matC = std::make_unique<BlockDenseMatrix>(rowC, colC, blockRowC, blockColC,
                                            false);
  if (matA.getBlockRowCount() % nPass != 0) {
    throw poputil::poplibs_error(
        "The number of block rows" + std::to_string(matA.getBlockRowCount()) +
        "of LHS is not divisible by the number of pass " +
        std::to_string(nPass));
  }
  nGroup = matA.getBlockRowCount() / nPass;
  // TODO: handle the case nTile is not divisible by nGroup
  nTilePerGroup = nTile / nGroup;

  logging::popsparse::info("number of pass = {} number of group = {} "
                           "number of tile per group = {}",
                           nPass, nGroup, nTilePerGroup);

  nSplitFactor = 2 * nTilePerGroup / matB.getBlockColCount();
  if (nSplitFactor == 0) {
    nSplitFactor = 1;
  }
  if (nSplitFactor > nTilePerGroup) {
    nSplitFactor = nTilePerGroup;
  }
  float loadBalance =
      createPartitionPlan(graph, matB, partitionPlan, nTilePerGroup,
                          nSplitFactor, true, debugPrefix);

  logging::popsparse::info("load balance: {}, nSplitFactor: {}", loadBalance,
                           nSplitFactor);

  lhsPartitionDSD(partitionPlan, lhsPartitionPlan, nTilePerGroup);
}

void HyperGraphStrip::createGraphMatMulSparsifyResult(
    poplar::Graph &graph, const unsigned char *sparsity,
    const std::string &debugPrefix) {
  assert(matA.isDense() && matB.isDense());
  assert(matC == nullptr);
  assert(matA.getColCount() == matB.getRowCount());
  assert(matA.getBlockCol() == matB.getBlockRow());

  // Populate matrix C
  const int blockRowC = matA.getBlockRow();
  const int blockColC = matB.getBlockCol();
  const int rowC = matA.getRowCount();
  const int colC = matB.getColCount();

  matC = std::make_unique<BlockSparseMatrix>(rowC, colC, blockRowC, blockColC,
                                             false, sparsity);

  if (matA.getBlockColCount() % nPass != 0) {
    throw poputil::poplibs_error(
        "The number of block columns " +
        std::to_string(matA.getBlockColCount()) +
        " of LHS is not divisible by the number of pass " +
        std::to_string(nPass));
    return;
  }

  nGroup = matA.getBlockColCount() / nPass;
  nTilePerGroup = nTile / nGroup;

  logging::popsparse::info("number of pass = {} number of group = {} "
                           "number of tile per group = {}",
                           nPass, nGroup, nTilePerGroup);

  nSplitFactor = 2 * nTilePerGroup / matC->getBlockColCount();

  if (nSplitFactor == 0) {
    nSplitFactor = 1;
  }
  if (nSplitFactor > matC->getBlockRowCount()) {
    nSplitFactor = matC->getBlockRowCount();
  }
  float loadBalance =
      createPartitionPlan(graph, *matC, partitionPlan, nTilePerGroup,
                          nSplitFactor, false, debugPrefix);

  lhsPartitionDDS(partitionPlan, lhsPartitionPlan, nTilePerGroup);

  logging::popsparse::info("load balance: {}, nSplitFactor: {}", loadBalance,
                           nSplitFactor);
}

void HyperGraphStrip::setTileMappingLHS(poplar::Graph &graph,
                                        poplar::Tensor &lhsTensor) {
  matA.setBlockTensor(lhsTensor);
  std::vector<int> lhsBlockTileId;
  if (!isResultSparse) {
    setLHSTileMapDSD(graph, lhsBlockTileId, true);
  } else {
    setLHSTileMapDDS(graph, lhsBlockTileId, true);
  }
}

void HyperGraphStrip::setTileMappingRHS(poplar::Graph &graph,
                                        poplar::Tensor &rhsTensor) {
  matB.setBlockTensor(rhsTensor);
  std::vector<int> rhsBlockTileId;
  if (!isResultSparse) {
    setRHSTileMapDSD(graph, rhsBlockTileId, true);
  } else {
    setRHSTileMapDDS(graph, rhsBlockTileId, true);
  }
}

void HyperGraphStrip::createProgramMatMul(poplar::Graph &graph,
                                          SubBlockMask subBlockMask,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugPrefix) {
  if (!isResultSparse) {
    createComputeSetDSD(graph, prog, debugPrefix);
  } else {
    createComputeSetDDS(graph, prog, debugPrefix);
    if (subBlockMask != SubBlockMask::None) {
      applySubBlockMask(graph, subBlockMask, prog, debugPrefix);
    }
  }
}

void HyperGraphStrip::createProgramMatMul(poplar::Graph &graph,
                                          poplar::ComputeSet *transposeCS,
                                          poplar::ComputeSet &mulCS,
                                          poplar::ComputeSet &reduceCS,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugPrefix) {
  assert(nPass == 1);
  std::vector<poplar::ComputeSet> transposeCSVec;
  std::vector<poplar::ComputeSet> mulCSVec;
  std::vector<poplar::ComputeSet> reduceCSVec;
  if (transposeCS != nullptr) {
    transposeCSVec.push_back(*transposeCS);
  }
  mulCSVec.push_back(mulCS);
  reduceCSVec.push_back(reduceCS);

  if (!isResultSparse) {
    createComputeSetDSD(graph, transposeCSVec, mulCSVec, reduceCSVec, prog,
                        debugPrefix);
  } else {
    createComputeSetDDS(graph, transposeCSVec, mulCSVec, reduceCSVec, prog,
                        debugPrefix);
  }
}

HyperGraphData HyperGraphStrip::getDataForPartitioner() {
  logging::popsparse::info("Number of hyper graph nodes: {}", nodes.size());

  HyperGraphData graphData;

  graphData.nodes = nodes.size();

  // The pins vector stores the indices of the vertices in each of the edges.
  std::vector<unsigned int> pins;

  // The hyperedge vector stores the offsets into the pins vector.
  std::vector<unsigned int> hyperEdges;
  std::vector<float> weights(graphData.nodes, 0.0F);

  for (const auto &n : nodes) {
    weights[n.id] = n.w;
  }

  hyperEdges.reserve(edges.size());

  for (const auto &e : edges) {
    hyperEdges.push_back(pins.size());
    pins.insert(pins.end(), e.pins.begin(), e.pins.end());
  }

  logging::popsparse::info("Number of pins is {}", pins.size());
  logging::popsparse::info("Number of edges is {}", hyperEdges.size());

  graphData.pins = std::move(pins);
  graphData.hyperEdges = std::move(hyperEdges);
  graphData.weights = std::move(weights);

  return graphData;
}

float HyperGraphStrip::partitionGraph(std::vector<int> &tileAssignment,
                                      int nTilePerGroup) {
  HyperGraphData graphData = getDataForPartitioner();
  float balance =
      partitioner->partitionGraph(graphData, nTilePerGroup, tileAssignment);
  if (balance < 0.0) {
    throw poputil::poplibs_error("Hyper graph partition failed");
  }

  return balance;
}

////////////////////////////////////////////////////////////////////////////////
// Functions for dense x sparse = dense
////////////////////////////////////////////////////////////////////////////////
float HyperGraphStrip::createPartitionPlan(
    poplar::Graph &graph, const BlockMatrix &mat,
    std::vector<std::vector<partition>> &partitionPlan, int nTilePerGroup,
    int nSplitFactor, bool allocatePartials, const std::string &debugPrefix) {
  assert(mat.isDense() == false);

  // Clean up hyper graph
  nodes.clear();
  edges.clear();
  gNodeId = gEdgeId = 0;

  const int nRow = mat.getBlockRowCount();
  const int nCol = mat.getBlockColCount();

  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrix = mat.getBlockIdMatrix();

  // populate HyperGraph node, each column of matrix B is a node
  for (int c = 0; c < nCol; c++) {
    float nonZeroBlock = 0.0;
    for (int r = 0; r < nRow; r++) {
      if (blockIdMatrix[r][c] != -1) {
        nonZeroBlock++;
      }
    }
    nodes.emplace_back(gNodeId++, nonZeroBlock);
  }

  // populate HyperGraph edge, each row of matrix is an edge, which connects
  // to non zero blocks (a HyperGraph node) in that row.
  for (int r = 0; r < nRow; r++) {
    std::vector<unsigned int> nonZeroBlocks;
    for (int c = 0; c < nCol; c++) {
      if (blockIdMatrix[r][c] != -1) {
        nonZeroBlocks.push_back(c);
      }
    }
    if (!nonZeroBlocks.empty()) {
      edges.emplace_back(gEdgeId++);
      edges.back().pins = nonZeroBlocks;
    }
  }

  std::vector<int> tileAssignment;
  int nPart = nTilePerGroup / nSplitFactor;
  if (nPart > nCol) {
    nPart = nCol;
  }
  float balance = partitionGraph(tileAssignment, nPart);

  // each sub array is the list of columns assigned to the same partition
  std::vector<std::vector<int>> partColumnMap(nPart);
  std::vector<std::pair<float, int>> partWeight(nPart);
  for (auto &p : partWeight) {
    p.first = 0.0f;
  }
  for (unsigned i = 0; i < tileAssignment.size(); i++) {
    int partId = tileAssignment[i];
    partColumnMap[partId].push_back(i);
    partWeight[partId].second = partId;
    partWeight[partId].first += nodes[i].w;
  }
  // Sort the part based on its weight from high to low
  std::sort(partWeight.begin(), partWeight.end(),
            std::greater<std::pair<float, int>>());

  std::vector<std::vector<int>> partTileIds(nPart);
  for (int i = 0; i < nPart; i++) {
    for (int j = 0; j < nSplitFactor; j++) {
      partTileIds[i].push_back(i * nSplitFactor + j);
    }
  }

  // hand extra tiles to the partition that has high weight
  int nLeftTile = nPart * nSplitFactor;
  int count = 0;
  while (count < nPart && nLeftTile < nTilePerGroup) {
    int partId = partWeight[count].second;
    partTileIds[partId].push_back(nLeftTile);
    count++;
    nLeftTile++;
  }

  // Now distribute the tiles to each partition
  partitionPlan.resize(nCol);
  for (int p = 0; p < nPart; p++) {
    std::vector<int> &columns = partColumnMap[p];
    int nColumn = static_cast<int>(columns.size());
    if (nColumn == 0) {
      continue;
    }
    std::vector<float> weight(nColumn);
    float totalWeight = 0.0f;
    for (int j = 0; j < nColumn; j++) {
      weight[j] = nodes[columns[j]].w;
      totalWeight += weight[j];
    }

    std::vector<int> &tiles = partTileIds[p];
    int nTotalTile = static_cast<int>(tiles.size());

    std::vector<std::vector<int>> columnTileId(nColumn);
    if (nColumn < nTotalTile) {
      int currentTileIdex = 0;
      for (int i = 0; i < nColumn; i++) {
        int tileCount = 0;
        if (i != nColumn - 1) {
          tileCount = static_cast<int>(nTotalTile * weight[i] / totalWeight);

          if (tileCount < 1 && weight[i] > 0.0) {
            tileCount = 1;
          }

          if (tileCount + currentTileIdex >= nTotalTile) {
            tileCount = nTotalTile - currentTileIdex;
          }
        } else {
          tileCount = nTotalTile - currentTileIdex;
        }

        for (int j = 0; j < tileCount; j++) {
          columnTileId[i].push_back(tiles[currentTileIdex + j]);
        }
        currentTileIdex += tileCount;
      }
    } else {
      std::vector<int> nodeAssignment;
      BalancedPartitioner::partition(weight, nTotalTile, nodeAssignment);
      for (unsigned n = 0; n < nodeAssignment.size(); n++) {
        columnTileId[n].push_back(tiles[nodeAssignment[n]]);
      }
    }
    for (int j = 0; j < nColumn; j++) {
      int c = columns[j];
      if (weight[j] == 0.0f) {
        // This is the case the column does not have any block
        continue;
      }
      int tileCount = static_cast<int>(columnTileId[j].size());

      std::vector<unsigned int> nonZeroBlocks;
      for (int r = 0; r < nRow; r++) {
        if (blockIdMatrix[r][c] != -1) {
          nonZeroBlocks.push_back(r);
        }
      }

      int nBlock = nonZeroBlocks.size() / tileCount;
      int nLeft = nonZeroBlocks.size() % tileCount;
      int start = 0;
      for (int k = 0; k < tileCount; k++) {
        int end = start + nBlock + (k < nLeft ? 1 : 0);
        assert(end <= static_cast<int>(nonZeroBlocks.size()));

        if (start == end) {
          continue;
        }

        struct partition p;
        p.tileId = columnTileId[j][k];
        if (allocatePartials) {
          p.partials.resize(nGroup);
          for (int g = 0; g < nGroup; g++) {
            p.partials[g] = graph.addVariable(
                partialDataType,
                {static_cast<unsigned long>(matC->getBlockRow() *
                                            matC->getBlockCol())},
                debugPrefix + "/paritials");
            graph.setTileMapping(p.partials[g], p.tileId);
          }
        }
        for (int b = start; b < end; b++) {
          p.rows.push_back(nonZeroBlocks[b]);
        }

        partitionPlan[c].push_back(p);

        start = end;
      }
    }
  }

  std::vector<int> tileBlocks(nTilePerGroup, 0);
  unsigned totalPartition = 0;
  for (int c = 0; c < nCol; c++) {
    totalPartition += partitionPlan[c].size();

    for (auto &p : partitionPlan[c]) {
      // printf("column = %d, tileId = %d, {", c, p.tileId);
      tileBlocks[p.tileId] += p.rows.size();
      /*
      for(auto &b : p.rows){
        printf("%d ", b);
      }
      printf("}\n");
      */
    }
  }

  int maxTileId, maxBlock, minTileId, minBlock;
  minBlock = INT_MAX;
  maxBlock = -INT_MAX;
  float avgBlock = static_cast<float>(mat.getNonZeroBlockCount()) /
                   static_cast<float>(nTilePerGroup);
  balance = -FLT_MAX;
  int zeroTiles = 0;
  for (int i = 0; i < nTilePerGroup; i++) {
    // printf("tile %d, blocks %d\n", i, tileBlocks[i]);
    if (tileBlocks[i] == 0) {
      zeroTiles++;
      continue;
    }
    if (minBlock > tileBlocks[i]) {
      minBlock = tileBlocks[i];
      minTileId = i;
    }
    if (maxBlock < tileBlocks[i]) {
      maxBlock = tileBlocks[i];
      maxTileId = i;
    }

    float b;
    if (tileBlocks[i] > avgBlock) {
      b = tileBlocks[i] / avgBlock;
    } else {
      b = avgBlock / tileBlocks[i];
    }

    balance = std::max(b, balance);
  }

  logging::popsparse::info("min tile id = {} blocks = {} ", minTileId,
                           minBlock);
  logging::popsparse::info("max tile id = {} blocks = {} ", maxTileId,
                           maxBlock);
  logging::popsparse::info("avg block = {} load balance = {}, zero tiles = {}",
                           avgBlock, balance, zeroTiles);

  return balance + zeroTiles;
}

void HyperGraphStrip::lhsPartitionDSD(
    std::vector<std::vector<partition>> &partitionPlan,
    std::vector<std::vector<int>> &lhsPartitionPlan, int nTilePerGroup) {
  const int nColA = matA.getBlockColCount();
  const int nColB = matB.getBlockColCount();

  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  std::vector<std::unordered_set<int>> lhsTileList;
  lhsTileList.resize(nColA);
  for (int c = 0; c < nColB; c++) {
    for (auto &p : partitionPlan[c]) {
      for (auto b : p.rows) {
        lhsTileList[b].insert(p.tileId);
      }
    }
  }

  std::vector<int> tileCount(nTilePerGroup, 0);
  lhsPartitionPlan.resize(nColA);
  // assign lhs block to tiles
  if (nTilePerGroup > nColA) {
    int nTileCount = nTilePerGroup / nColA;
    int leftTile = nTilePerGroup % nColA;
    for (int c = 0; c < nColA; c++) {
      int currentTileCount = nTileCount + (c < leftTile ? 1 : 0);
      if (currentTileCount < 0) {
        currentTileCount = 1;
      }

      for (int i = 0; i < currentTileCount; i++) {
        // find out which tile has less blocks
        int min = INT_MAX;
        int tileId;
        for (auto &e : lhsTileList[c]) {
          if (tileCount[e] < min) {
            min = tileCount[e];
            tileId = e;
          }
        }

        if (min == INT_MAX) {
          // if lhsTileList is empty, just put it onto the
          // tile which has less blocks
          for (int i = 0; i < nTilePerGroup; i++) {
            if (tileCount[i] < min) {
              min = tileCount[i];
              tileId = i;
            }
          }
        } else {
          lhsTileList[c].erase(tileId);
        }

        tileCount[tileId]++;
        lhsPartitionPlan[c].push_back(tileId);
      }
    }
  } else {
    int avgBlockPerTile = matA.getNonZeroBlockCount() / nTilePerGroup;
    for (int c = 0; c < nColA; c++) {
      // find out which tile has less blocks
      int min = INT_MAX;
      int tileId;
      for (auto &e : lhsTileList[c]) {
        if (tileCount[e] < min) {
          min = tileCount[e];
          tileId = e;
        }
      }

      if (min == INT_MAX || min + 1 > avgBlockPerTile) {
        // if lhsTileList is empty, just put it onto the
        // tile which has less blocks
        for (int i = 0; i < nTilePerGroup; i++) {
          if (tileCount[i] < min) {
            min = tileCount[i];
            tileId = i;
          }
        }
      }

      tileCount[tileId]++;
      lhsPartitionPlan[c].push_back(tileId);
    }
  }
}

void HyperGraphStrip::setLHSTileMapDSD(poplar::Graph &graph,
                                       std::vector<int> &lhsBlockTileId,
                                       bool setTileMap) {
  const std::vector<poplar::Tensor> &blockData = matA.getBlockTensor();

  std::vector<std::vector<int>> blockIdMatrix = matA.getBlockIdMatrix();
  const int nRow = matA.getBlockRowCount();
  const int nCol = matA.getBlockColCount();

  lhsBlockTileId.resize(matA.getNonZeroBlockCount());
  int startRow = 0;
  for (int p = 0; p < nPass; p++) {
    int endRow = startRow + nRow / nPass;
    unsigned int tileOffset = 0;
    for (int r = startRow; r < endRow; r++) {
      for (int c = 0; c < nCol; c++) {
        int blockId = blockIdMatrix[r][c];
        // TODO: lhsBlockTileId is used in preprocessBlocks to copy the block
        //       to make it contiguous. Not sure how to handle it since one
        //       block may be on multiple tiles. set to the first tile for now.
        lhsBlockTileId[blockId] = lhsPartitionPlan[c][0] + tileOffset;
        poplar::Tensor block = blockData[blockId].reshape(
            {static_cast<unsigned long>(matA.getBlockRow()),
             static_cast<unsigned long>(matA.getBlockCol())});
        int tileCount = static_cast<int>(lhsPartitionPlan[c].size());
        int sliceSize = matA.getBlockRow() / tileCount;
        int leftOver = matA.getBlockRow() % tileCount;
        std::size_t start = 0;
        for (int t = 0; t < tileCount; t++) {
          unsigned int tileId = lhsPartitionPlan[c][t] + tileOffset;
          std::size_t end = start + sliceSize + (t < leftOver ? 1 : 0);
          poplar::Tensor oneSlice = block.slice(
              {start, 0}, {end, static_cast<std::size_t>(matA.getBlockCol())});
          if (setTileMap) {
            graph.setTileMapping(oneSlice, tileId);
          }
          start = end;
        }
      }
      tileOffset += nTilePerGroup;
    }
    startRow = endRow;
  }
}

// need to evenly distribute to groups of tiles.
void HyperGraphStrip::setRHSTileMapDSD(poplar::Graph &graph,
                                       std::vector<int> &blockTileId,
                                       bool setTileMap) {
  const std::vector<poplar::Tensor> &blockData = matB.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matB.getBlockIdMatrix();
  const int nCol = matB.getBlockColCount();

  blockTileId.resize(matB.getNonZeroBlockCount());

  for (int c = 0; c < nCol; c++) {
    for (auto &p : partitionPlan[c]) {
      int tileId = p.tileId;

      int nBlockSize = p.rows.size() / nGroup;
      int leftOver = p.rows.size() % nGroup;

      int currentTileOffset = 0;
      int start = 0;
      for (int i = 0; i < nGroup; i++) {
        int end = start + (i < leftOver ? nBlockSize + 1 : nBlockSize);
        for (int j = start; j < end; j++) {
          int blockId = blockIdMatrix[p.rows[j]][c];
          blockTileId[blockId] = tileId + currentTileOffset;
          if (setTileMap) {
            graph.setTileMapping(blockData[blockId], blockTileId[blockId]);
          }
        }
        currentTileOffset += nTilePerGroup;
        start = end;
      }
      assert(start == static_cast<int>(p.rows.size()));
    }
  }
}

void HyperGraphStrip::setResultTileMapDSD(poplar::Graph &graph,
                                          std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matC->getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matC->getBlockIdMatrix();
  const int nRow = matC->getBlockRowCount();
  const int nCol = matC->getBlockColCount();

  blockTileId.resize(matC->getNonZeroBlockCount());

  int startRow = 0, endRow = 0;
  for (int p = 0; p < nPass; p++) {
    endRow += nRow / nPass;
    unsigned int tileOffset = 0;
    for (int r = startRow; r < endRow; r++) {
      for (int c = 0; c < nCol; c++) {
        int blockId = blockIdMatrix[r][c];
        poplar::Tensor block = blockData[blockId].reshape(
            {static_cast<unsigned long>(matC->getBlockRow()),
             static_cast<unsigned long>(matC->getBlockCol())});
        int tileCount = partitionPlan[c].size();
        if (tileCount == 0) {
          // this is the zero block
          int tileId = getRandomTile(nTilePerGroup) + tileOffset;
          blockTileId[blockId] = tileId;
          graph.setTileMapping(block, tileId);
          continue;
        }
        // TODO: This is for the tile to put zero vertex, for now, put on the
        // first tile
        blockTileId[blockId] = partitionPlan[c][0].tileId;
        int sliceSize = matC->getBlockRow() / tileCount;
        int leftOver = matC->getBlockRow() % tileCount;
        std::size_t start = 0;
        for (int t = 0; t < tileCount; t++) {
          int tileId = partitionPlan[c][t].tileId + tileOffset;
          std::size_t end = start + sliceSize + (t < leftOver ? 1 : 0);
          graph.setTileMapping(
              block.slice({start, 0},
                          {end, static_cast<std::size_t>(matC->getBlockCol())}),
              tileId);
          start = end;
        }
      }
      tileOffset += nTilePerGroup;
    }
    startRow = endRow;
  }
}

void HyperGraphStrip::createComputeSetDSD(poplar::Graph &graph,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugPrefix) {

  genSeq3(graph, &HyperGraphStrip::createComputeSetDSD, prog, debugPrefix);
}

void HyperGraphStrip::createComputeSetDSD(
    poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
    std::vector<poplar::ComputeSet> &mulCSVec,
    std::vector<poplar::ComputeSet> &reduceCSVec,
    poplar::program::Sequence &prog, const std::string &debugPrefix) {

  poplar::Tensor matCTensor =
      matC->createTensor(graph, inDataType, debugPrefix + "/matrix_c");
  matC->setBlockTensor(matCTensor);

  std::vector<int> lhsBlockTileId;
  setLHSTileMapDSD(graph, lhsBlockTileId, false);
  std::vector<int> rhsBlockTileId;
  setRHSTileMapDSD(graph, rhsBlockTileId, false);
  std::vector<int> outputBlockTileId;
  setResultTileMapDSD(graph, outputBlockTileId);

  std::vector<poplar::Tensor> blockDataA, blockDataB;

  poplar::ComputeSet *transposeCS = nullptr;
  if (!transposeCSVec.empty()) {
    transposeCS = &transposeCSVec[0];
  }
  preprocessBlocks(graph, matA, matB, blockDataA, blockDataB, lhsBlockTileId,
                   rhsBlockTileId, transposeCS, prog, debugPrefix);

  const int nRowC = matC->getBlockRowCount();
  const int nColC = matC->getBlockColCount();
  const std::vector<std::vector<int>> blockIdMatrixA = matA.getBlockIdMatrix();
  const std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  const std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  // TODO: handle zero output blocks
  int nConvVertex = 0, nReduceVertex = 0;
  int startRow = 0, endRow = 0;
  assert(static_cast<int>(mulCSVec.size()) == nPass);
  for (int p = 0; p < nPass; p++) {
    poplar::ComputeSet &mulCS = mulCSVec[p];
    poplar::ComputeSet &reduceCS = reduceCSVec[p];

    endRow += nRowC / nPass;
    unsigned int tileOffset = 0;
    for (int r = startRow; r < endRow; r++) {
      for (int c = 0; c < nColC; c++) {
        if (partitionPlan[c].size() == 0) {
          int blockId = blockIdMatrixC[r][c];
          // this is the zero block
          popops::zero(graph, blockDataC[blockId], outputBlockTileId[blockId],
                       reduceCS);
          continue;
        }
        for (auto &p : partitionPlan[c]) {
          if (p.rows.size() == 0) {
            continue;
          }
          unsigned int tileId = p.tileId + tileOffset;

          std::vector<poplar::Tensor> inputA, inputB;
          for (auto k : p.rows) {
            inputA.push_back(blockDataA[blockIdMatrixA[r][k]]);
            inputB.push_back(blockDataB[blockIdMatrixB[k][c]]);
          }

          if (partitionPlan[c].size() == 1 && partialDataType == outDataType) {
            addConv1x1Vertex(graph, inputA, inputB,
                             blockDataC[blockIdMatrixC[r][c]], tileId, mulCS,
                             debugPrefix);
          } else {
            addConv1x1Vertex(graph, inputA, inputB, p.partials[r - startRow],
                             tileId, mulCS, debugPrefix);
          }
          nConvVertex++;
        }

        // sum partials
        int nPartition = partitionPlan[c].size();
        if (nPartition <= 1 && partialDataType == outDataType)
          continue;

        int sliceSize = matC->getBlockRow() / nPartition;
        int leftOver = matC->getBlockRow() % nPartition;

        std::size_t start = 0;
        std::size_t blockColC = static_cast<std::size_t>(matC->getBlockCol());
        for (int p = 0; p < nPartition; p++) {
          unsigned int tileId = partitionPlan[c][p].tileId + tileOffset;
          std::size_t end = start + sliceSize + (p < leftOver ? 1 : 0);

          std::vector<poplar::Tensor> input;
          for (int s = 0; s < nPartition; s++) {
            poplar::Tensor t = partitionPlan[c][s].partials[r - startRow];
            t = t.reshape({static_cast<unsigned long>(matC->getBlockRow()),
                           static_cast<unsigned long>(matC->getBlockCol())});
            input.push_back(t.slice({start, 0}, {end, blockColC}).flatten());
          }
          poplar::Tensor outputTensor = blockDataC[blockIdMatrixC[r][c]];
          outputTensor = outputTensor.reshape(
              {static_cast<unsigned long>(matC->getBlockRow()),
               static_cast<unsigned long>(matC->getBlockCol())});
          outputTensor =
              outputTensor.slice({start, 0}, {end, blockColC}).flatten();
          addReduceVertex(graph, input, outputTensor, tileId, reduceCS);
          nReduceVertex++;
          start = end;
        }
      }
      tileOffset += nTilePerGroup;
    }
    startRow = endRow;
  }
}

void HyperGraphStrip::genSeq3(poplar::Graph &graph, GenCs3 genCs3,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix) {

  std::vector<poplar::ComputeSet> transposeCSVec;
  std::vector<poplar::ComputeSet> mulCSVec;
  std::vector<poplar::ComputeSet> reduceCSVec;

  if (!matB.getNeedTranspose()) {
    transposeCSVec.push_back(graph.addComputeSet(debugPrefix + "/transposeCS"));
  }

  for (int p = 0; p < nPass; p++) {
    mulCSVec.push_back(
        graph.addComputeSet(debugPrefix + "/mulCS" + std::to_string(p)));
    reduceCSVec.push_back(
        graph.addComputeSet(debugPrefix + "/reduceCS" + std::to_string(p)));
  }

  if (!matB.getNeedTranspose()) {
    prog.add(poplar::program::Execute(transposeCSVec[0]));
  }

  (this->*genCs3)(graph, transposeCSVec, mulCSVec, reduceCSVec, prog,
                  debugPrefix);

  for (int p = 0; p < nPass; p++) {
    prog.add(poplar::program::Execute(mulCSVec[p]));
    prog.add(poplar::program::Execute(reduceCSVec[p]));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functions for dense x dense = sparse
////////////////////////////////////////////////////////////////////////////////

void HyperGraphStrip::lhsPartitionDDS(
    std::vector<std::vector<partition>> &partitionPlan,
    std::vector<std::vector<int>> &lhsPartitionPlan, int nTilePerGroup) {
  const int nRowA = matA.getBlockRowCount();

  // block id look up matrix for C
  std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();
  std::vector<std::unordered_set<int>> lhsTileList;
  lhsTileList.resize(nRowA);
  for (int c = 0; c < matC->getBlockColCount(); c++) {
    for (auto &p : partitionPlan[c]) {
      for (auto b : p.rows) {
        lhsTileList[b].insert(p.tileId);
      }
    }
  }

  std::vector<int> tileCount(nTilePerGroup, 0);
  lhsPartitionPlan.resize(nRowA);
  // assign lhs block to tiles
  if (nTilePerGroup > nRowA) {
    int nTileCount = nTilePerGroup / nRowA;
    int leftTile = nTilePerGroup % nRowA;
    for (int r = 0; r < nRowA; r++) {
      int currentTileCount = nTileCount + (r < leftTile ? 1 : 0);

      for (int i = 0; i < currentTileCount; i++) {
        // find out which tile has less blocks
        int min = INT_MAX;
        int tileId;
        for (auto &e : lhsTileList[r]) {
          if (tileCount[e] < min) {
            min = tileCount[e];
            tileId = e;
          }
        }

        if (min == INT_MAX) {
          // if lhsTileList is empty, just put it onto the
          // tile which has less blocks
          for (int i = 0; i < nTilePerGroup; i++) {
            if (tileCount[i] < min) {
              min = tileCount[i];
              tileId = i;
            }
          }
        } else {
          lhsTileList[r].erase(tileId);
        }

        tileCount[tileId]++;
        lhsPartitionPlan[r].push_back(tileId);
      }
    }
  } else {
    int avgBlockPerTile = matA.getNonZeroBlockCount() / nTilePerGroup;
    for (int r = 0; r < nRowA; r++) {
      // find out which tile has less blocks
      int min = INT_MAX;
      int tileId;
      for (auto &e : lhsTileList[r]) {
        if (tileCount[e] < min) {
          min = tileCount[e];
          tileId = e;
        }
      }

      if (min == INT_MAX || min + 1 > avgBlockPerTile) {
        // if lhsTileList is empty, just put it onto the
        // tile which has less blocks
        for (int i = 0; i < nTilePerGroup; i++) {
          if (tileCount[i] < min) {
            min = tileCount[i];
            tileId = i;
          }
        }
      }

      tileCount[tileId]++;
      lhsPartitionPlan[r].push_back(tileId);
    }
  }
}

void HyperGraphStrip::setLHSTileMapDDS(poplar::Graph &graph,
                                       std::vector<int> &lhsBlockTileId,
                                       bool setTileMap) {

  const std::vector<poplar::Tensor> &blockData = matA.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matA.getBlockIdMatrix();
  const int nRow = matA.getBlockRowCount();
  const int nCol = matA.getBlockColCount();

  lhsBlockTileId.resize(matA.getNonZeroBlockCount());
  int startCol = 0;
  for (int p = 0; p < nPass; p++) {
    int endCol = startCol + nCol / nPass;
    for (int r = 0; r < nRow; r++) {
      int tileOffset = 0;
      int nBlock = nCol / nPass;
      int tileCount = lhsPartitionPlan[r].size();
      if (tileCount > nBlock) {
        // distribut the blocks to the first few tiles
        int start = startCol;
        for (int g = 0; g < nGroup; g++) {
          int end = start + nBlock / nGroup;
          for (int c = start; c < end; c++) {
            int blockId = blockIdMatrix[r][c];
            unsigned int tileId = lhsPartitionPlan[r][c - start] + tileOffset;
            if (setTileMap)
              graph.setTileMapping(blockData[blockId], tileId);
            lhsBlockTileId[blockId] = tileId;
          }
          start = end;
          tileOffset += nTilePerGroup;
        }
      } else {
        // evenly distribute blocks to all available tiles
        int blocksPerTile = (nBlock + tileCount) / tileCount;
        int start = startCol;
        for (int g = 0; g < nGroup; g++) {
          int end = start + nBlock / nGroup;
          for (int c = start; c < end; c++) {
            int blockId = blockIdMatrix[r][c];
            int partitionIndex = (c - start) / blocksPerTile;
            assert(partitionIndex <
                   static_cast<int>(lhsPartitionPlan[r].size()));
            unsigned int tileId =
                lhsPartitionPlan[r][partitionIndex] + tileOffset;
            if (setTileMap) {
              graph.setTileMapping(blockData[blockId], tileId);
            }
            lhsBlockTileId[blockId] = tileId;
          }
          start = end;
          tileOffset += nTilePerGroup;
        }
      }
    }

    startCol = endCol;
  }
}

void HyperGraphStrip::setRHSTileMapDDS(poplar::Graph &graph,
                                       std::vector<int> &blockTileId,
                                       bool setTileMap) {
  const std::vector<poplar::Tensor> &blockData = matB.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matB.getBlockIdMatrix();
  const int nRow = matB.getBlockRowCount();
  const int nCol = matB.getBlockColCount();

  blockTileId.resize(matB.getNonZeroBlockCount());
  int startRow = 0;
  for (int p = 0; p < nPass; p++) {
    int endRow = startRow + nRow / nPass;

    for (int c = 0; c < nCol; c++) {
      int tileCount = partitionPlan[c].size();
      int sliceSize = 0;
      int leftOver = 0;
      if (tileCount != 0) {
        sliceSize = matB.getBlockRow() / tileCount;
        leftOver = matB.getBlockRow() % tileCount;
      }
      unsigned int tileOffset = 0;
      int start = startRow;
      for (int g = 0; g < nGroup; g++) {
        int end = start + nRow / nPass / nGroup;
        for (int r = start; r < end; r++) {
          int blockId = blockIdMatrix[r][c];
          if (tileCount == 0) {
            blockTileId[blockId] = getRandomTile(nTilePerGroup) + tileOffset;
            if (setTileMap) {
              graph.setTileMapping(blockData[blockId], blockTileId[blockId]);
            }
            continue;
          }
          // TODO: blockTileId is used for transpose the block, use the first
          // tile for now. May need to improve it if transpose becomes the
          // bottle neck
          blockTileId[blockId] = partitionPlan[c][0].tileId + tileOffset;
          poplar::Tensor block = blockData[blockId].reshape(
              {static_cast<unsigned long>(matB.getBlockRow()),
               static_cast<unsigned long>(matB.getBlockCol())});

          std::size_t startSlice = 0;
          for (int t = 0; t < tileCount; t++) {
            int tileId = partitionPlan[c][t].tileId + tileOffset;
            std::size_t endSlice =
                startSlice + sliceSize + (t < leftOver ? 1 : 0);
            if (setTileMap) {
              graph.setTileMapping(
                  block.slice(
                      {startSlice, 0},
                      {endSlice, static_cast<std::size_t>(matB.getBlockCol())}),
                  tileId);
            }
            startSlice = endSlice;
          }
        }
        start = end;
        tileOffset += nTilePerGroup;
      }
    }
    startRow = endRow;
  }
}

void HyperGraphStrip::setOutputTileMapDDS(poplar::Graph &graph,
                                          std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matC->getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matC->getBlockIdMatrix();
  const int nCol = matC->getBlockColCount();

  blockTileId.resize(matC->getNonZeroBlockCount());

  for (int c = 0; c < nCol; c++) {
    for (auto &p : partitionPlan[c]) {
      int tileId = p.tileId;

      int nBlockSize = p.rows.size() / nGroup;
      int leftOver = p.rows.size() % nGroup;
      int currentTileOffset = 0;
      int start = 0;
      for (int i = 0; i < nGroup; i++) {
        int end = start + (i < leftOver ? nBlockSize + 1 : nBlockSize);
        for (int j = start; j < end; j++) {
          int blockId = blockIdMatrix[p.rows[j]][c];
          blockTileId[blockId] = tileId + currentTileOffset;
          graph.setTileMapping(blockData[blockId], blockTileId[blockId]);
        }
        currentTileOffset += nTilePerGroup;
        start = end;
      }
      assert(start == static_cast<int>(p.rows.size()));
    }
  }

  // Check if all blocks are mapped to tiles
  for (int i = 0; i < matC->getNonZeroBlockCount(); i++) {
    try {
      graph.getTileMapping(blockData[i]);
    } catch (const std::exception &) {
      printf("block %d is not mapped to tiles\n", i);
      break;
    }
  }
}

void HyperGraphStrip::createComputeSetDDS(poplar::Graph &graph,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugPrefix) {

  genSeq3(graph, &HyperGraphStrip::createComputeSetDDS, prog, debugPrefix);
}

void HyperGraphStrip::createComputeSetDDS(
    poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
    std::vector<poplar::ComputeSet> &mulCSVec,
    std::vector<poplar::ComputeSet> &reduceCSVec,
    poplar::program::Sequence &prog, const std::string &debugPrefix) {

  poplar::Tensor matCTensor =
      matC->createTensor(graph, inDataType, debugPrefix + "/matrix_c");
  matC->setBlockTensor(matCTensor);

  std::vector<int> lhsBlockTileId;
  setLHSTileMapDDS(graph, lhsBlockTileId, false);
  std::vector<int> rhsBlockTileId;
  setRHSTileMapDDS(graph, rhsBlockTileId, false);
  std::vector<int> outputBlockTileId;
  setOutputTileMapDDS(graph, outputBlockTileId);

  std::vector<poplar::Tensor> blockDataA, blockDataB;

  poplar::ComputeSet *transposeCS = nullptr;
  if (!transposeCSVec.empty()) {
    transposeCS = &transposeCSVec[0];
  }
  preprocessBlocks(graph, matA, matB, blockDataA, blockDataB, lhsBlockTileId,
                   rhsBlockTileId, transposeCS, prog, debugPrefix);

  const unsigned nColC = matC->getBlockColCount();
  const std::vector<std::vector<int>> blockIdMatrixA = matA.getBlockIdMatrix();
  const std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  const std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  std::vector<std::vector<poplar::Tensor>> partialSum;
  if (nGroup > 1 || (partialDataType != outDataType)) {
    partialSum.resize(nGroup);
    int tileOffset = 0;
    for (int i = 0; i < nGroup; i++) {
      partialSum[i].resize(blockDataC.size());
      for (unsigned c = 0; c < nColC; c++) {
        for (auto &p : partitionPlan[c]) {
          unsigned int tileId = p.tileId + tileOffset;
          for (unsigned b = 0; b < p.rows.size(); b++) {
            int blockId = blockIdMatrixC[p.rows[b]][c];
            partialSum[i][blockId] = graph.addVariable(
                partialDataType,
                {static_cast<unsigned long>(matC->getBlockRow() *
                                            matC->getBlockCol())},
                debugPrefix + "/partial_matC_block_" + std::to_string(blockId));
            graph.setTileMapping(partialSum[i][blockId], tileId);
          }
        }
      }

      tileOffset += nTilePerGroup;
    }
  }

  int nBlockPerPass = matA.getBlockColCount() / nPass;
  int nBlockPerGroup = matA.getBlockColCount() / nGroup / nPass;
  assert(static_cast<int>(mulCSVec.size()) == nPass);
  assert(static_cast<int>(reduceCSVec.size()) == nPass);
  const unsigned convInChannels = (inDataType == poplar::FLOAT ? 8 : 16);
  for (int p = 0; p < nPass; p++) {
    poplar::ComputeSet &mulCS = mulCSVec[p];
    poplar::ComputeSet &reduceCS = reduceCSVec[p];

    int tileOffset = 0;
    int colStart = nBlockPerPass * p;
    for (int g = 0; g < nGroup; g++) {
      int colEnd = colStart + nBlockPerGroup;
      for (unsigned c = 0; c < nColC; c++) {
        for (auto &p : partitionPlan[c]) {
          int tileId = p.tileId + tileOffset;

          // add vertex for the rows in the buffer
          std::vector<poplar::Tensor> inputA, inputB;
          for (int k = colStart; k < colEnd; k++) {
            std::vector<poplar::Tensor> combinedA;
            for (unsigned n = 0; n < p.rows.size(); n++) {
              poplar::Tensor t = blockDataA[blockIdMatrixA[p.rows[n]][k]];
              t = t.reshape({t.dim(0),
                             static_cast<unsigned long>(matA.getBlockRow()),
                             convInChannels});
              combinedA.push_back(t);
            }
            poplar::Tensor tmp = concat(combinedA, 1);
            tmp = tmp.reshape({tmp.dim(0), tmp.dim(1) * tmp.dim(2)});
            inputA.push_back(tmp);
            inputB.push_back(blockDataB[blockIdMatrixB[k][c]]);
          }
          std::vector<poplar::Tensor> outputBlocks;
          for (unsigned n = 0; n < p.rows.size(); n++) {
            poplar::Tensor t;
            if (nGroup == 1 && partialDataType == outDataType) {
              t = blockDataC[blockIdMatrixC[p.rows[n]][c]];
            } else {
              t = partialSum[g][blockIdMatrixC[p.rows[n]][c]];
            }
            t = t.reshape({static_cast<unsigned long>(matC->getBlockRow()),
                           static_cast<unsigned long>(matC->getBlockCol())});
            outputBlocks.push_back(t);
          }

          addConv1x1Vertex(graph, inputA, inputB,
                           concat(outputBlocks).flatten(), tileId, mulCS,
                           debugPrefix);
        }
      }
      colStart = colEnd;
      tileOffset += nTilePerGroup;
    }

    if (nGroup > 1) {
      for (unsigned c = 0; c < nColC; c++) {
        for (auto &p : partitionPlan[c]) {
          for (auto r : p.rows) {
            int blockId = blockIdMatrixC[r][c];
            std::vector<poplar::Tensor> inputBlocks;
            for (int i = 0; i < nGroup; i++) {
              inputBlocks.push_back(partialSum[i][blockId]);
            }
            poplar::Tensor out = blockDataC[blockId];
            addReduceVertex(graph, inputBlocks, out, outputBlockTileId[blockId],
                            reduceCS);
          }
        }
      }
    } else if (partialDataType != outDataType) {
      // cast parital data type to output data type
      for (unsigned c = 0; c < nColC; c++) {
        for (auto &p : partitionPlan[c]) {
          for (auto r : p.rows) {
            int blockId = blockIdMatrixC[r][c];
            popops::cast(graph, partialSum[0][blockId], blockDataC[blockId],
                         reduceCS);
          }
        }
      }
    }
  }
}

} // namespace experimental
} // namespace popsparse
