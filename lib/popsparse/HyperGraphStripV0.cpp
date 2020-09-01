// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphStripV0.hpp"
#include <cfloat>
#include <poplibs_support/logging.hpp>
#include <popops/Zero.hpp>
#include <poputil/exceptions.hpp>
#include <unordered_set>
#include <zoltan_cpp.h>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphStripV0::HyperGraphStripV0(const BlockMatrix &A, const BlockMatrix &B,
                                     poplar::Type inDataTypeIn,
                                     poplar::Type outDataTypeIn,
                                     poplar::Type partialDataTypeIn,
                                     int nTileIn, int nPassIn)
    : HyperGraph(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn, nTileIn),
      gNodeId(0), gEdgeId(0), nPass(nPassIn), isResultSparse(false) {

  partitioner = std::make_unique<ZoltanPartitioner>(
      ZoltanPartitioner::PartitionType::BLOCK);

  if (A.isDense() && !B.isDense()) {
    isResultSparse = false;
  } else if (A.isDense() && B.isDense()) {
    isResultSparse = true;
  } else {
    assert(0);
  }

  logging::popsparse::info("HyperGraphStripV0 is created");
}

void HyperGraphStripV0::createGraphMatMul(poplar::Graph &graph,
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
  poplar::Tensor matCTensor =
      matC->createTensor(graph, inDataType, debugPrefix + "/matrix_c");
  matC->setBlockTensor(matCTensor);

  if (matA.getBlockRowCount() % nPass != 0) {
    throw poputil::poplibs_error(
        "The number of block rows" + std::to_string(matA.getBlockRowCount()) +
        "of LHS is not divisible by the number of pass " +
        std::to_string(nPass));
  }
  nGroup = matA.getBlockRowCount() / nPass;

  if (nTile % nGroup != 0) {
    throw poputil::poplibs_error(
        "The number of tiles " + std::to_string(nTile) +
        " is not divisible by the number of group " + std::to_string(nGroup));
  }
  nTilePerGroup = nTile / nGroup;

  logging::popsparse::info("number of pass = {} number of group = {} "
                           "number of tile per group = {}",
                           nPass, nGroup, nTilePerGroup);

  std::vector<int> rowSplitTileAssignment, colSplitTileAssignment;
  float rowSplitBalance =
      partitionRowSplitDSD(rowSplitTileAssignment, nTilePerGroup);
  float colSplitBalance =
      partitionColSplitDSD(colSplitTileAssignment, nTilePerGroup);
  float loadBalance =
      rowSplitBalance < colSplitBalance ? rowSplitBalance : colSplitBalance;

  if (colSplitBalance <= rowSplitBalance) {
    doRowSplit = false;
    rhsTileAssignment = colSplitTileAssignment;
    logging::popsparse::info("column split, load balance: {}", loadBalance);
  } else {
    doRowSplit = true;
    rhsTileAssignment = rowSplitTileAssignment;
    logging::popsparse::info("row split, load balance: {}", loadBalance);
  }

  if (doRowSplit) {
    lhsTileAssignment = rhsTileAssignment;
    resultPartitionDSD(resultTileAssignment, nTilePerGroup);
  } else {
    if (matB.getRowCount() == matB.getColCount() &&
        matB.getBlockRow() == matB.getBlockCol()) {
      // the matrix is square, lhs and rhs has the same tile assignment
      lhsTileAssignment = rhsTileAssignment;
    } else {
      lhsPartitionDSD(rhsTileAssignment, lhsTileAssignment, nTilePerGroup);
    }
    resultTileAssignment = rhsTileAssignment;
  }
}

void HyperGraphStripV0::createGraphMatMulSparsifyResult(
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
  poplar::Tensor matCTensor =
      matC->createTensor(graph, inDataType, debugPrefix + "/matrix_c");
  matC->setBlockTensor(matCTensor);

  if (matA.getBlockColCount() % nPass != 0) {
    throw poputil::poplibs_error(
        "The number of block columns " +
        std::to_string(matA.getBlockColCount()) +
        " of LHS is not divisible by the number of pass " +
        std::to_string(nPass));
    return;
  }
  nGroup = matA.getBlockColCount() / nPass;

  if (nTile % nGroup != 0) {
    throw poputil::poplibs_error(
        "The number of tiles " + std::to_string(nTile) +
        " is not divisible by the number of group " + std::to_string(nGroup));
    return;
  }
  nTilePerGroup = nTile / nGroup;

  logging::popsparse::info("number of pass = {} number of group = {} "
                           "number of tile per group = {}",
                           nPass, nGroup, nTilePerGroup);

  std::vector<int> rowSplitTileAssignment, colSplitTileAssignment;
  float rowSplitBalance =
      partitionRowSplitDDS(rowSplitTileAssignment, nTilePerGroup);
  float colSplitBalance =
      partitionColSplitDDS(colSplitTileAssignment, nTilePerGroup);
  float loadBalance =
      rowSplitBalance < colSplitBalance ? rowSplitBalance : colSplitBalance;

  if (colSplitBalance <= rowSplitBalance) {
    doRowSplit = false;
    resultTileAssignment = colSplitTileAssignment;
    logging::popsparse::info("column split, load balance: {}", loadBalance);
  } else {
    doRowSplit = true;
    resultTileAssignment = rowSplitTileAssignment;
    logging::popsparse::info("row split, load balance: {}", loadBalance);
  }

  if (doRowSplit) {
    lhsTileAssignment = resultTileAssignment;
    if (matA.getRowCount() == matB.getColCount() &&
        matA.getBlockRow() == matB.getBlockCol()) {
      rhsTileAssignment = resultTileAssignment;
    } else {
      rhsPartitionDDS(lhsTileAssignment, rhsTileAssignment, nTilePerGroup);
    }
  } else {
    rhsTileAssignment = resultTileAssignment;
    if (matA.getRowCount() == matB.getColCount() &&
        matA.getBlockRow() == matB.getBlockCol()) {
      lhsTileAssignment = rhsTileAssignment;
    } else {
      lhsPartitionDDS(rhsTileAssignment, lhsTileAssignment, nTilePerGroup);
    }
  }
}

void HyperGraphStripV0::createProgramMatMul(poplar::Graph &graph,
                                            SubBlockMask subBlockMask,
                                            poplar::program::Sequence &prog,
                                            const std::string &debugPrefix) {
  if (!isResultSparse) {
    if (doRowSplit) {
      createComputeSetRowSplitDSD(graph, prog, debugPrefix);
    } else {
      createComputeSetColSplitDSD(graph, prog, debugPrefix);
    }
  } else {
    if (doRowSplit) {
      createComputeSetRowSplitDDS(graph, prog, debugPrefix);
    } else {
      createComputeSetColSplitDDS(graph, prog, debugPrefix);
    }
    if (subBlockMask != SubBlockMask::None) {
      applySubBlockMask(graph, subBlockMask, prog, debugPrefix);
    }
  }
}

void HyperGraphStripV0::createProgramMatMul(poplar::Graph &graph,
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
    if (doRowSplit) {
      createComputeSetRowSplitDSD(graph, transposeCSVec, mulCSVec, reduceCSVec,
                                  prog, debugPrefix);
    } else {
      createComputeSetColSplitDSD(graph, transposeCSVec, mulCSVec, prog,
                                  debugPrefix);
    }
  } else {
    if (doRowSplit) {
      createComputeSetRowSplitDDS(graph, transposeCSVec, mulCSVec, reduceCSVec,
                                  prog, debugPrefix);
    } else {
      createComputeSetColSplitDDS(graph, transposeCSVec, mulCSVec, reduceCSVec,
                                  prog, debugPrefix);
    }
  }
}

HyperGraphData HyperGraphStripV0::getDataForPartitioner() {
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

float HyperGraphStripV0::partitionGraph(std::vector<int> &tileAssignment,
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
float HyperGraphStripV0::partitionColSplitDSD(std::vector<int> &tileAssignment,
                                              int nTilePerGroup) {
  // Clean up hyper graph
  nodes.clear();
  edges.clear();
  gNodeId = gEdgeId = 0;

  const int nRowB = matB.getBlockRowCount();
  const int nColB = matB.getBlockColCount();

  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();

  // populate HyperGraph node, each column of matrix B is a node
  for (int c = 0; c < nColB; c++) {
    float nonZeroBlock = 0.0;
    for (int r = 0; r < nRowB; r++) {
      if (blockIdMatrixB[r][c] != -1) {
        nonZeroBlock++;
      }
    }
    nodes.emplace_back(gNodeId++, nonZeroBlock);
  }

  // populate HyperGraph edge, each row of matrix B is an edge, which connects
  // to non zero blocks (a HyperGraph node) in that row.
  for (int r = 0; r < nRowB; r++) {
    std::vector<unsigned int> nonZeroBlocks;
    for (int c = 0; c < nColB; c++) {
      if (blockIdMatrixB[r][c] != -1) {
        nonZeroBlocks.push_back(c);
      }
    }
    if (!nonZeroBlocks.empty()) {
      edges.emplace_back(gEdgeId++);
      edges.back().pins = nonZeroBlocks;
    }
  }

  return partitionGraph(tileAssignment, nTilePerGroup);
}

float HyperGraphStripV0::partitionRowSplitDSD(std::vector<int> &tileAssignment,
                                              int nTilePerGroup) {
  // Clean up hyper graph
  nodes.clear();
  edges.clear();
  gNodeId = gEdgeId = 0;

  const int nRowB = matB.getBlockRowCount();
  const int nColB = matB.getBlockColCount();

  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();

  // populate HyperGraph node, each row of matrix B is a node
  for (int r = 0; r < nRowB; r++) {
    float nonZeroBlock = 0.0;
    for (int c = 0; c < nColB; c++) {
      if (blockIdMatrixB[r][c] != -1) {
        nonZeroBlock++;
      }
    }
    nodes.emplace_back(gNodeId++, nonZeroBlock);
  }

  // populate HyperGraph edge, each column of matrix B is an edge, which
  // connects to non zero blocks (a HyperGraph node) in that column.
  for (int c = 0; c < nColB; c++) {
    std::vector<unsigned int> nonZeroBlocks;
    for (int r = 0; r < nRowB; r++) {
      if (blockIdMatrixB[r][c] != -1) {
        nonZeroBlocks.push_back(c);
      }
    }
    if (!nonZeroBlocks.empty()) {
      edges.emplace_back(gEdgeId++);
      edges.back().pins = nonZeroBlocks;
    }
  }

  return partitionGraph(tileAssignment, nTilePerGroup);
}

void HyperGraphStripV0::lhsPartitionDSD(std::vector<int> &rhsTileAssignment,
                                        std::vector<int> &lhsTileAssignment,
                                        int nTilePerGroup) {

  std::vector<int> tileCount(nTilePerGroup, 0);
  std::vector<std::unordered_set<int>> lhsTileList;
  const int nColA = matA.getBlockColCount();
  lhsTileList.resize(nColA);
  lhsTileAssignment.resize(nColA);

  const int nRowB = matB.getBlockRowCount();
  const int nColB = matB.getBlockColCount();

  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  for (int c = 0; c < nColB; c++) {
    int tileId = rhsTileAssignment[c];
    for (int r = 0; r < nRowB; r++) {
      if (blockIdMatrixB[r][c] == -1) {
        continue;
      }

      lhsTileList[r].insert(tileId);
    }
  }

  // assign lhs block to tiles
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

    if (min == INT_MAX) {
      // if this column is not found in any tile, just put it onto the
      // tile which has less blocks
      for (int i = 0; i < nTilePerGroup; i++) {
        if (tileCount[i] < min) {
          min = tileCount[i];
          tileId = i;
        }
      }
    }
    tileCount[tileId]++;
    lhsTileAssignment[c] = tileId;
  }
}

void HyperGraphStripV0::resultPartitionDSD(
    std::vector<int> &resultTileAssignment, int nTilePerGroup) {
  const int nColC = matC->getBlockColCount();
  assert(doRowSplit == true);

  std::vector<int> tileCount(nTilePerGroup, 0);
  std::vector<std::unordered_set<int>> resultTileList;
  resultTileList.resize(nColC);
  resultTileAssignment.resize(nColC);

  const int nRowB = matB.getBlockRowCount();
  const int nColB = matB.getBlockColCount();

  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  for (int r = 0; r < nRowB; r++) {
    int tileId = rhsTileAssignment[r];
    for (int c = 0; c < nColB; c++) {
      if (blockIdMatrixB[r][c] == -1) {
        continue;
      }

      resultTileList[c].insert(tileId);
    }
  }

  // assign result block to tiles
  float avgBlockPerTile =
      static_cast<float>(nColC) / static_cast<float>(nTilePerGroup);
  for (int c = 0; c < nColC; c++) {
    // find out which tile has less blocks
    int min = INT_MAX;
    int tileId;
    for (auto &e : resultTileList[c]) {
      if (tileCount[e] < min) {
        min = tileCount[e];
        tileId = e;
      }
    }

    if (min == INT_MAX || min + 1 > avgBlockPerTile) {
      // if this tile already has more than average number of blocks, just put
      // it onto the tile which has less blocks
      for (int i = 0; i < nTilePerGroup; i++) {
        if (tileCount[i] < min) {
          min = tileCount[i];
          tileId = i;
        }
      }
    }
    tileCount[tileId]++;
    resultTileAssignment[c] = tileId;
  }
}

void HyperGraphStripV0::setLHSTileMapDSD(poplar::Graph &graph,
                                         std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matA.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matA.getBlockIdMatrix();
  const int nRow = matA.getBlockRowCount();
  const int nCol = matA.getBlockColCount();

  blockTileId.resize(matA.getNonZeroBlockCount());

  int start = 0, end = 0;
  for (int p = 0; p < nPass; p++) {
    end += nRow / nPass;
    unsigned int tileOffset = 0;
    for (int r = start; r < end; r++) {
      for (int c = 0; c < nCol; c++) {
        unsigned int tileId = lhsTileAssignment[c] + tileOffset;
        if (tileId < 0) {
          throw poputil::poplibs_error(
              "Invalid tile id: " + std::to_string(tileId) + " For column " +
              std::to_string(c));
        }

        int blockId = blockIdMatrix[r][c];
        if (blockId == -1) {
          continue;
        }
        blockTileId[blockId] = tileId;
        graph.setTileMapping(blockData[blockId], tileId);
      }

      tileOffset += nTilePerGroup;
    }
    start = end;
  }
}

// need to evenly distribute to groups of tiles.
void HyperGraphStripV0::setRHSTileMapDSD(poplar::Graph &graph,
                                         std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matB.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matB.getBlockIdMatrix();
  const int nRow = matB.getBlockRowCount();
  const int nCol = matB.getBlockColCount();

  blockTileId.resize(matB.getNonZeroBlockCount());

  if (doRowSplit) {
    for (int r = 0; r < nRow; r++) {
      if (rhsTileAssignment[r] < 0) {
        throw poputil::poplibs_error(
            "Invalid tile id: " + std::to_string(rhsTileAssignment[r]) +
            " For row " + std::to_string(r));
        return;
      }
      unsigned int tileId = static_cast<unsigned int>(rhsTileAssignment[r]);

      std::vector<int> nonZeroBlockIds;
      for (int c = 0; c < nCol; c++) {
        if (blockIdMatrix[r][c] != -1)
          nonZeroBlockIds.push_back(c);
      }
      int nBlock = nonZeroBlockIds.size() / nGroup;
      int leftOver = nonZeroBlockIds.size() % nGroup;

      int currentTileOffset = 0;
      int start = 0;
      for (int i = 0; i < nGroup; i++) {
        int end = start + (i < leftOver ? nBlock + 1 : nBlock);
        for (int j = start; j < end; j++) {
          int blockId = blockIdMatrix[r][nonZeroBlockIds[j]];
          blockTileId[blockId] = tileId + currentTileOffset;
          graph.setTileMapping(blockData[blockId], tileId + currentTileOffset);
        }
        currentTileOffset += nTilePerGroup;
        start = end;
      }
      assert(start == static_cast<int>(nonZeroBlockIds.size()));
    }
  } else {
    for (int c = 0; c < nCol; c++) {
      if (rhsTileAssignment[c] < 0) {
        throw poputil::poplibs_error(
            "Invalid tile id: " + std::to_string(rhsTileAssignment[c]) +
            " For column " + std::to_string(c));
      }
      unsigned int tileId = static_cast<unsigned int>(rhsTileAssignment[c]);

      std::vector<int> nonZeroBlockIds;
      for (int r = 0; r < nRow; r++) {
        if (blockIdMatrix[r][c] != -1)
          nonZeroBlockIds.push_back(r);
      }
      int nBlockSize = nonZeroBlockIds.size() / nGroup;
      int leftOver = nonZeroBlockIds.size() % nGroup;

      int currentTileOffset = 0;
      int start = 0;
      for (int i = 0; i < nGroup; i++) {
        int end = start + (i < leftOver ? nBlockSize + 1 : nBlockSize);
        for (int j = start; j < end; j++) {
          int blockId = blockIdMatrix[nonZeroBlockIds[j]][c];
          blockTileId[blockId] = tileId + currentTileOffset;
          graph.setTileMapping(blockData[blockId], blockTileId[blockId]);
        }
        currentTileOffset += nTilePerGroup;
        start = end;
      }
      assert(start == static_cast<int>(nonZeroBlockIds.size()));
    }
  }
}

// TODO: share the common code in setResultTileMapDSD and setLHSTileMapDSD
void HyperGraphStripV0::setResultTileMapDSD(poplar::Graph &graph) {
  const std::vector<poplar::Tensor> &blockData = matC->getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matC->getBlockIdMatrix();
  const int nRow = matC->getBlockRowCount();
  const int nCol = matC->getBlockColCount();

  int start = 0, end = 0;
  for (int p = 0; p < nPass; p++) {
    end += nRow / nPass;
    unsigned int tileOffset = 0;
    for (int r = start; r < end; r++) {
      for (int c = 0; c < nCol; c++) {
        unsigned int tileId = resultTileAssignment[c] + tileOffset;
        if (tileId < 0) {
          throw poputil::poplibs_error(
              "Invalid tile id: " + std::to_string(tileId) + " For column " +
              std::to_string(c));
          return;
        }

        int blockId = blockIdMatrix[r][c];
        if (blockId == -1) {
          continue;
        }

        graph.setTileMapping(blockData[blockId], tileId);
      }

      tileOffset += nTilePerGroup;
    }
    start = end;
  }
}

void HyperGraphStripV0::createComputeSetColSplitDSD(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    const std::string &debugPrefix) {

  genSeq2(graph, &HyperGraphStripV0::createComputeSetColSplitDSD, prog,
          debugPrefix);
}

void HyperGraphStripV0::createComputeSetColSplitDSD(
    poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
    std::vector<poplar::ComputeSet> &mulCSVec, poplar::program::Sequence &prog,
    const std::string &debugPrefix) {

  std::vector<int> lhsBlockTileId;
  setLHSTileMapDSD(graph, lhsBlockTileId);
  std::vector<int> rhsBlockTileId;
  setRHSTileMapDSD(graph, rhsBlockTileId);
  setResultTileMapDSD(graph);

  std::vector<poplar::Tensor> blockDataA, blockDataB;

  poplar::ComputeSet *transposeCS = nullptr;
  if (!transposeCSVec.empty()) {
    transposeCS = &transposeCSVec[0];
  }
  preprocessBlocks(graph, matA, matB, blockDataA, blockDataB, lhsBlockTileId,
                   rhsBlockTileId, transposeCS, prog, debugPrefix);

  const int nColA = matA.getBlockColCount();
  const int nRowC = matC->getBlockRowCount();
  const int nColC = matC->getBlockColCount();
  const std::vector<std::vector<int>> blockIdMatrixA = matA.getBlockIdMatrix();
  const std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  const std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  int startR = 0, endR = 0;
  assert(static_cast<int>(mulCSVec.size()) == nPass);
  for (int p = 0; p < nPass; p++) {
    poplar::ComputeSet &mulCS = mulCSVec[p];

    endR += nRowC / nPass;
    unsigned int tileOffset = 0;
    for (int r = startR; r < endR; r++) {
      for (int c = 0; c < nColC; c++) {
        unsigned int tileId = rhsTileAssignment[c] + tileOffset;

        std::vector<poplar::Tensor> inputA, inputB;
        bool isNonZero = false;
        for (int k = 0; k < nColA; k++) {
          if (blockIdMatrixA[r][k] != -1 && blockIdMatrixB[k][c] != -1) {
            isNonZero = true;
            inputA.push_back(blockDataA[blockIdMatrixA[r][k]]);
            inputB.push_back(blockDataB[blockIdMatrixB[k][c]]);
          }
        }

        if (!isNonZero) {
          popops::zero(graph, blockDataC[blockIdMatrixC[r][c]], tileId, mulCS);
        } else {
          addConv1x1Vertex(graph, inputA, inputB,
                           blockDataC[blockIdMatrixC[r][c]], tileId, mulCS,
                           debugPrefix);
        }
      }

      tileOffset += nTilePerGroup;
    }
    startR = endR;
  }
}

void HyperGraphStripV0::genSeq2(poplar::Graph &graph, GenCs2 genCs2,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix) {

  std::vector<poplar::ComputeSet> transposeCSVec;
  std::vector<poplar::ComputeSet> mulCSVec;

  if (!matB.getNeedTranspose()) {
    transposeCSVec.push_back(graph.addComputeSet(debugPrefix + "/transposeCS"));
  }

  for (int p = 0; p < nPass; p++) {
    mulCSVec.push_back(
        graph.addComputeSet(debugPrefix + "/mulCS" + std::to_string(p)));
  }

  if (!matB.getNeedTranspose()) {
    prog.add(poplar::program::Execute(transposeCSVec[0]));
  }

  (this->*genCs2)(graph, transposeCSVec, mulCSVec, prog, debugPrefix);

  for (int p = 0; p < nPass; p++) {
    prog.add(poplar::program::Execute(mulCSVec[p]));
  }
}

void HyperGraphStripV0::genSeq3(poplar::Graph &graph, GenCs3 genCs3,
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

void HyperGraphStripV0::createComputeSetRowSplitDSD(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    const std::string &debugPrefix) {

  genSeq3(graph, &HyperGraphStripV0::createComputeSetRowSplitDSD, prog,
          debugPrefix);
}

void HyperGraphStripV0::createComputeSetRowSplitDSD(
    poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
    std::vector<poplar::ComputeSet> &mulCSVec,
    std::vector<poplar::ComputeSet> &reduceCSVec,
    poplar::program::Sequence &prog, const std::string &debugPrefix) {

  const unsigned nColB = matB.getBlockColCount();

  std::vector<int> lhsBlockTileId;
  setLHSTileMapDSD(graph, lhsBlockTileId);
  std::vector<int> rhsBlockTileId;
  setRHSTileMapDSD(graph, rhsBlockTileId);
  setResultTileMapDSD(graph);

  std::vector<poplar::Tensor> blockDataA, blockDataB;

  poplar::ComputeSet *transposeCS = nullptr;
  if (!transposeCSVec.empty()) {
    transposeCS = &transposeCSVec[0];
  }
  preprocessBlocks(graph, matA, matB, blockDataA, blockDataB, lhsBlockTileId,
                   rhsBlockTileId, transposeCS, prog, debugPrefix);

  const std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  std::vector<Strip> tileStrips;
  tileStrips.resize(nTilePerGroup);
  for (int i = 0; i < nTilePerGroup; i++) {
    tileStrips[i].tensors.resize(nGroup);
    for (int g = 0; g < nGroup; g++)
      tileStrips[i].tensors[g].resize(nColB);
  }
  for (unsigned r = 0; r < rhsTileAssignment.size(); r++) {
    int tileId = rhsTileAssignment[r];
    tileStrips[tileId].rows.push_back(r);

    for (int g = 0; g < nGroup; g++) {
      for (unsigned c = 0; c < nColB; c++) {
        if (blockIdMatrixB[r][c] == -1) {
          continue;
        }

        if (!tileStrips[tileId].tensors[g][c].valid()) {
          tileStrips[tileId].tensors[g][c] = graph.addVariable(
              partialDataType,
              {static_cast<unsigned long>(matC->getBlockRow() *
                                          matC->getBlockCol())},
              debugPrefix + "/paritial_block_group_" + std::to_string(g) +
                  "_tile_" + std::to_string(tileId) + "_col_" +
                  std::to_string(c));
          graph.setTileMapping(tileStrips[tileId].tensors[g][c],
                               tileId + g * nTilePerGroup);
        }
      }
    }
  }

  const std::vector<std::vector<int>> blockIdMatrixA = matA.getBlockIdMatrix();
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  const std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  const unsigned nRowC = matC->getBlockRowCount();
  int startR = 0, endR = 0;

  assert(static_cast<int>(mulCSVec.size()) == nPass);
  assert(static_cast<int>(reduceCSVec.size()) == nPass);
  for (int p = 0; p < nPass; p++) {
    endR += nRowC / nPass;
    poplar::ComputeSet &mulCS = mulCSVec[p];
    poplar::ComputeSet &reduceCS = reduceCSVec[p];

    unsigned int tileOffset = 0;
    for (int g = startR; g < endR; g++) {
      for (int i = 0; i < nTilePerGroup; i++) {
        unsigned int tileId = i + tileOffset;
        for (unsigned c = 0; c < nColB; c++) {
          bool isNonZero = false;
          std::vector<poplar::Tensor> inputA, inputB;
          for (auto r : tileStrips[i].rows) {
            if (blockIdMatrixA[g][r] != -1 && blockIdMatrixB[r][c] != -1) {
              isNonZero = true;
              inputA.push_back(blockDataA[blockIdMatrixA[g][r]]);
              inputB.push_back(blockDataB[blockIdMatrixB[r][c]]);
            }
          }

          if (isNonZero) {
            addConv1x1Vertex(graph, inputA, inputB,
                             tileStrips[i].tensors[g - startR][c], tileId,
                             mulCS, debugPrefix);
          }
        }
      }
      tileOffset += nTilePerGroup;
    }

    // partial reduction
    tileOffset = 0;
    for (int g = startR; g < endR; g++) {
      for (unsigned c = 0; c < nColB; c++) {
        unsigned int tileId = resultTileAssignment[c] + tileOffset;
        poplar::Tensor outputTensor = blockDataC[blockIdMatrixC[g][c]];
        std::vector<poplar::Tensor> inputBlocks;
        for (int i = 0; i < nTilePerGroup; i++) {
          if (tileStrips[i].tensors[g - startR][c].valid()) {
            inputBlocks.push_back(tileStrips[i].tensors[g - startR][c]);
          }
        }

        if (inputBlocks.empty()) {
          popops::zero(graph, outputTensor, tileId, reduceCS);
        } else {
          addReduceVertex(graph, inputBlocks, outputTensor, tileId, reduceCS);
        }
      }

      tileOffset += nTilePerGroup;
    }

    startR = endR;
  } // end of pass loop
}

////////////////////////////////////////////////////////////////////////////////
// Functions for dense x dense = sparse
////////////////////////////////////////////////////////////////////////////////
float HyperGraphStripV0::partitionColSplitDDS(std::vector<int> &tileAssignment,
                                              int nTilePerGroup) {
  // Clean up hyper graph
  nodes.clear();
  edges.clear();
  gNodeId = gEdgeId = 0;

  const int nRowC = matC->getBlockRowCount();
  const int nColC = matC->getBlockColCount();

  // block id look up matrix for C
  std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  // populate HyperGraph node, each column of matrix B is a node
  for (int c = 0; c < nColC; c++) {
    float nonZeroBlock = 0.0;
    for (int r = 0; r < nRowC; r++) {
      if (blockIdMatrixC[r][c] != -1) {
        nonZeroBlock++;
      }
    }
    nodes.emplace_back(gNodeId++, nonZeroBlock);
  }

  // populate HyperGraph edge, each row of matrix C is an edge, which connects
  // to non zero blocks (a HyperGraph node) in that row.
  for (int r = 0; r < nRowC; r++) {
    std::vector<unsigned int> nonZeroBlocks;
    for (int c = 0; c < nColC; c++) {
      if (blockIdMatrixC[r][c] != -1) {
        nonZeroBlocks.push_back(c);
      }
    }
    if (!nonZeroBlocks.empty()) {
      edges.emplace_back(gEdgeId++);
      edges.back().pins = nonZeroBlocks;
    }
  }

  return partitionGraph(tileAssignment, nTilePerGroup);
}

float HyperGraphStripV0::partitionRowSplitDDS(std::vector<int> &tileAssignment,
                                              int nTilePerGroup) {
  // Clean up hyper graph
  nodes.clear();
  edges.clear();
  gNodeId = gEdgeId = 0;

  const int nRowC = matC->getBlockRowCount();
  const int nColC = matC->getBlockColCount();

  // block id look up matrix for C
  std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  // populate HyperGraph node, each row of matrix C is a node
  for (int r = 0; r < nRowC; r++) {
    float nonZeroBlock = 0.0;
    for (int c = 0; c < nColC; c++) {
      if (blockIdMatrixC[r][c] != -1) {
        nonZeroBlock++;
      }
    }
    nodes.emplace_back(gNodeId++, nonZeroBlock);
  }

  // populate HyperGraph edge, each column of matrix C is an edge, which
  // connects to non zero blocks (a HyperGraph node) in that column.
  for (int c = 0; c < nColC; c++) {
    std::vector<unsigned int> nonZeroBlocks;
    for (int r = 0; r < nRowC; r++) {
      if (blockIdMatrixC[r][c] != -1) {
        nonZeroBlocks.push_back(r);
      }
    }
    if (!nonZeroBlocks.empty()) {
      edges.emplace_back(gEdgeId++);
      edges.back().pins = nonZeroBlocks;
    }
  }

  return partitionGraph(tileAssignment, nTilePerGroup);
}

void HyperGraphStripV0::rhsPartitionDDS(
    const std::vector<int> &lhsTileAssignment,
    std::vector<int> &rhsTileAssignment, int nTilePerGroup) {
  assert(doRowSplit == true);

  std::vector<int> tileCount(nTilePerGroup, 0);
  std::vector<std::unordered_set<int>> rhsTileList;
  const int nColB = matB.getBlockColCount();
  rhsTileList.resize(nColB);
  rhsTileAssignment.resize(nColB);

  const int nRowC = matC->getBlockRowCount();
  const int nColC = matC->getBlockColCount();

  // block id look up matrix for C
  std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();
  for (int r = 0; r < nRowC; r++) {
    int tileId = lhsTileAssignment[r];
    for (int c = 0; c < nColC; c++) {
      if (blockIdMatrixC[r][c] == -1) {
        continue;
      }

      rhsTileList[c].insert(tileId);
    }
  }

  // assign rhs block to tiles
  float avgBlockPerTile =
      static_cast<float>(nColB) / static_cast<float>(nTilePerGroup);
  for (int c = 0; c < nColB; c++) {
    // find out which tile has less blocks
    int min = INT_MAX;
    int tileId;
    for (auto &e : rhsTileList[c]) {
      if (tileCount[e] < min) {
        min = tileCount[e];
        tileId = e;
      }
    }

    if (min == INT_MAX || min + 1 > avgBlockPerTile) {
      for (int i = 0; i < nTilePerGroup; i++) {
        if (tileCount[i] < min) {
          min = tileCount[i];
          tileId = i;
        }
      }
    }
    tileCount[tileId]++;
    rhsTileAssignment[c] = tileId;
  }
}

void HyperGraphStripV0::lhsPartitionDDS(
    const std::vector<int> &rhsTileAssignment,
    std::vector<int> &lhsTileAssignment, int nTilePerGroup) {
  const int nRowA = matA.getBlockRowCount();
  assert(doRowSplit == false);

  std::vector<int> tileCount(nTilePerGroup, 0);
  std::vector<std::unordered_set<int>> lhsTileList;
  lhsTileList.resize(nRowA);
  lhsTileAssignment.resize(nRowA);

  const int nRowC = matC->getBlockRowCount();
  const int nColC = matC->getBlockColCount();

  // block id look up matrix for C
  std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();
  for (int c = 0; c < nColC; c++) {
    int tileId = rhsTileAssignment[c];
    for (int r = 0; r < nRowC; r++) {
      if (blockIdMatrixC[r][c] == -1) {
        continue;
      }

      lhsTileList[r].insert(tileId);
    }
  }

  // assign result block to tiles
  float avgBlockPerTile =
      static_cast<float>(nRowA) / static_cast<float>(nTilePerGroup);
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
      // if this tile already has more than average number of blocks, just put
      // it onto the tile which has less blocks
      for (int i = 0; i < nTilePerGroup; i++) {
        if (tileCount[i] < min) {
          min = tileCount[i];
          tileId = i;
        }
      }
    }
    tileCount[tileId]++;
    lhsTileAssignment[r] = tileId;
  }
}

void HyperGraphStripV0::setLHSTileMapDDS(poplar::Graph &graph,
                                         std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matA.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matA.getBlockIdMatrix();
  const int nRow = matA.getBlockRowCount();
  const int nCol = matA.getBlockColCount();

  blockTileId.resize(matA.getNonZeroBlockCount());
  for (int r = 0; r < nRow; r++) {
    unsigned int tileId = lhsTileAssignment[r];
    if (tileId < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " + std::to_string(tileId) + " For row " +
          std::to_string(r));
      return;
    }

    int start = 0, end = 0;
    for (int p = 0; p < nPass; p++) {
      end = start + nCol / nPass;
      int tileOffset = 0;
      for (int c = start; c < end; c++) {
        int blockId = blockIdMatrix[r][c];
        if (blockId == -1) {
          continue;
        }
        blockTileId[blockId] = tileId + tileOffset;
        graph.setTileMapping(blockData[blockId], tileId + tileOffset);
        tileOffset += nTilePerGroup;
      }

      start = end;
    }
  }
}

// need to evenly distribute to groups of tiles.
void HyperGraphStripV0::setRHSTileMapDDS(poplar::Graph &graph,
                                         std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matB.getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matB.getBlockIdMatrix();
  const int nRow = matB.getBlockRowCount();
  const int nCol = matB.getBlockColCount();

  blockTileId.resize(matB.getNonZeroBlockCount());

  for (int c = 0; c < nCol; c++) {
    unsigned int tileId = rhsTileAssignment[c];
    if (tileId < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " + std::to_string(tileId) + " For column " +
          std::to_string(c));
      return;
    }

    int start = 0, end = 0;
    for (int p = 0; p < nPass; p++) {
      end = start + nRow / nPass;
      int tileOffset = 0;
      for (int r = start; r < end; r++) {
        int blockId = blockIdMatrix[r][c];
        if (blockId == -1) {
          continue;
        }
        blockTileId[blockId] = tileId + tileOffset;
        graph.setTileMapping(blockData[blockId], tileId + tileOffset);
        tileOffset += nTilePerGroup;
      }
      start = end;
    }
  }
}

void HyperGraphStripV0::setOutputTileMapDDS(poplar::Graph &graph,
                                            std::vector<int> &blockTileId) {
  const std::vector<poplar::Tensor> &blockData = matC->getBlockTensor();
  std::vector<std::vector<int>> blockIdMatrix = matC->getBlockIdMatrix();
  const int nRow = matC->getBlockRowCount();
  const int nCol = matC->getBlockColCount();

  blockTileId.resize(matC->getNonZeroBlockCount());

  if (doRowSplit) {
    for (int r = 0; r < nRow; r++) {
      if (resultTileAssignment[r] < 0) {
        throw poputil::poplibs_error(
            "Invalid tile id: " + std::to_string(resultTileAssignment[r]) +
            " For row " + std::to_string(r));
        return;
      }
      unsigned int tileId = static_cast<unsigned int>(resultTileAssignment[r]);

      std::vector<int> nonZeroBlocks;
      for (int c = 0; c < nCol; c++) {
        if (blockIdMatrix[r][c] != -1) {
          nonZeroBlocks.push_back(c);
        }
      }
      int nBlock = nonZeroBlocks.size() / nGroup;
      int leftOver = nonZeroBlocks.size() % nGroup;

      int currentTileOffset = 0;
      int start = 0;
      for (int i = 0; i < nGroup; i++) {
        int end = start + (i < leftOver ? nBlock + 1 : nBlock);
        for (int j = start; j < end; j++) {
          int blockId = blockIdMatrix[r][nonZeroBlocks[j]];
          graph.setTileMapping(blockData[blockId], tileId + currentTileOffset);
          blockTileId[blockId] = tileId + currentTileOffset;
        }
        currentTileOffset += nTilePerGroup;
        start = end;
      }
      assert(start == static_cast<int>(nonZeroBlocks.size()));
    }
  } else {
    for (int c = 0; c < nCol; c++) {
      if (resultTileAssignment[c] < 0) {
        throw poputil::poplibs_error(
            "Invalid tile id: " + std::to_string(resultTileAssignment[c]) +
            " For column " + std::to_string(c));
      }
      unsigned int tileId = static_cast<unsigned int>(resultTileAssignment[c]);

      std::vector<int> nonZeroBlocks;
      for (int r = 0; r < nRow; r++) {
        if (blockIdMatrix[r][c] != -1)
          nonZeroBlocks.push_back(r);
      }
      int nBlock = nonZeroBlocks.size() / nGroup;
      int leftOver = nonZeroBlocks.size() % nGroup;

      int currentTileOffset = 0;
      int start = 0;
      for (int i = 0; i < nGroup; i++) {
        int end = start + (i < leftOver ? nBlock + 1 : nBlock);
        for (int j = start; j < end; j++) {
          int blockId = blockIdMatrix[nonZeroBlocks[j]][c];
          graph.setTileMapping(blockData[blockId], tileId + currentTileOffset);
          blockTileId[blockId] = tileId + currentTileOffset;
        }
        currentTileOffset += nTilePerGroup;
        start = end;
      }
      assert(start == static_cast<int>(nonZeroBlocks.size()));
    }
  }
}

void HyperGraphStripV0::createComputeSetColSplitDDS(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    const std::string &debugPrefix) {

  genSeq3(graph, &HyperGraphStripV0::createComputeSetColSplitDDS, prog,
          debugPrefix);
}

void HyperGraphStripV0::createComputeSetColSplitDDS(
    poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
    std::vector<poplar::ComputeSet> &mulCSVec,
    std::vector<poplar::ComputeSet> &reduceCSVec,
    poplar::program::Sequence &prog, const std::string &debugPrefix) {

  std::vector<int> lhsBlockTileId;
  setLHSTileMapDDS(graph, lhsBlockTileId);
  std::vector<int> rhsBlockTileId;
  setRHSTileMapDDS(graph, rhsBlockTileId);
  std::vector<int> blockTileId;
  setOutputTileMapDDS(graph, blockTileId);

  std::vector<poplar::Tensor> blockDataA, blockDataB;

  poplar::ComputeSet *transposeCS = nullptr;
  if (!transposeCSVec.empty()) {
    transposeCS = &transposeCSVec[0];
  }
  preprocessBlocks(graph, matA, matB, blockDataA, blockDataB, lhsBlockTileId,
                   rhsBlockTileId, transposeCS, prog, debugPrefix);

  const unsigned nRowC = matC->getBlockRowCount();
  const unsigned nColC = matC->getBlockColCount();
  const std::vector<std::vector<int>> blockIdMatrixA = matA.getBlockIdMatrix();
  const std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  const std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  std::vector<std::vector<poplar::Tensor>> partialSum;
  partialSum.resize(nGroup);
  int tileOffset = 0;
  for (int i = 0; i < nGroup; i++) {
    partialSum[i].resize(blockDataC.size());
    for (unsigned c = 0; c < nColC; c++) {
      unsigned int tileId = resultTileAssignment[c] + tileOffset;
      for (unsigned r = 0; r < nRowC; r++) {
        int blockId = blockIdMatrixC[r][c];
        if (blockId == -1) {
          continue;
        }
        partialSum[i][blockId] = graph.addVariable(
            partialDataType,
            {static_cast<unsigned long>(matC->getBlockRow() *
                                        matC->getBlockCol())},
            debugPrefix + "/partial_matC_block_" + std::to_string(blockId));
        graph.setTileMapping(partialSum[i][blockId], tileId);
      }
    }

    tileOffset += nTilePerGroup;
  }

  int gStart = 0;
  int gEnd = 0;

  assert(static_cast<int>(mulCSVec.size()) == nPass);
  assert(static_cast<int>(reduceCSVec.size()) == nPass);
  for (int p = 0; p < nPass; p++) {
    tileOffset = 0;
    gEnd += nGroup;

    poplar::ComputeSet &mulCS = mulCSVec[p];
    poplar::ComputeSet &reduceCS = reduceCSVec[p];

    for (int g = gStart; g < gEnd; g++) {
      for (unsigned c = 0; c < nColC; c++) {
        unsigned int tileId = resultTileAssignment[c] + tileOffset;
        for (unsigned r = 0; r < nRowC; r++) {
          if (blockIdMatrixC[r][c] == -1) {
            continue;
          }
          std::vector<poplar::Tensor> inputA, inputB;
          inputA.push_back(blockDataA[blockIdMatrixA[r][g]]);
          inputB.push_back(blockDataB[blockIdMatrixB[g][c]]);

          addConv1x1Vertex(graph, inputA, inputB,
                           partialSum[g - gStart][blockIdMatrixC[r][c]], tileId,
                           mulCS, debugPrefix);
        }
      }
      tileOffset += nTilePerGroup;
    }

    for (unsigned c = 0; c < nColC; c++) {
      for (unsigned r = 0; r < nRowC; r++) {
        int blockId = blockIdMatrixC[r][c];
        if (blockId == -1) {
          continue;
        }

        unsigned tileId = blockTileId[blockId];

        std::vector<poplar::Tensor> inputBlocks;
        for (int i = 0; i < nGroup; i++) {
          inputBlocks.push_back(partialSum[i][blockId]);
        }
        poplar::Tensor out = blockDataC[blockId];
        addReduceVertex(graph, inputBlocks, out, tileId, reduceCS);
      }
    }

    gStart = gEnd;
  }
}

void HyperGraphStripV0::createComputeSetRowSplitDDS(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    const std::string &debugPrefix) {

  genSeq3(graph, &HyperGraphStripV0::createComputeSetRowSplitDDS, prog,
          debugPrefix);
}

void HyperGraphStripV0::createComputeSetRowSplitDDS(
    poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
    std::vector<poplar::ComputeSet> &mulCSVec,
    std::vector<poplar::ComputeSet> &reduceCSVec,
    poplar::program::Sequence &prog, const std::string &debugPrefix) {

  std::vector<int> lhsBlockTileId;
  setLHSTileMapDDS(graph, lhsBlockTileId);
  std::vector<int> rhsBlockTileId;
  setRHSTileMapDDS(graph, rhsBlockTileId);
  std::vector<int> blockTileId;
  setOutputTileMapDDS(graph, blockTileId);

  std::vector<poplar::Tensor> blockDataA, blockDataB;

  poplar::ComputeSet *transposeCS = nullptr;
  if (!transposeCSVec.empty()) {
    transposeCS = &transposeCSVec[0];
  }
  preprocessBlocks(graph, matA, matB, blockDataA, blockDataB, lhsBlockTileId,
                   rhsBlockTileId, transposeCS, prog, debugPrefix);

  const unsigned nRowC = matC->getBlockRowCount();
  const unsigned nColC = matC->getBlockColCount();
  const std::vector<std::vector<int>> blockIdMatrixA = matA.getBlockIdMatrix();
  const std::vector<std::vector<int>> blockIdMatrixB = matB.getBlockIdMatrix();
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  const std::vector<std::vector<int>> blockIdMatrixC = matC->getBlockIdMatrix();

  std::vector<std::vector<poplar::Tensor>> partialSum;
  partialSum.resize(nGroup);
  int tileOffset = 0;
  for (int i = 0; i < nGroup; i++) {
    partialSum[i].resize(blockDataC.size());
    for (unsigned r = 0; r < nRowC; r++) {
      unsigned int tileId = resultTileAssignment[r] + tileOffset;
      for (unsigned c = 0; c < nColC; c++) {
        int blockId = blockIdMatrixC[r][c];
        if (blockId == -1) {
          continue;
        }
        partialSum[i][blockId] = graph.addVariable(
            partialDataType,
            {static_cast<unsigned long>(matC->getBlockRow() *
                                        matC->getBlockCol())},
            debugPrefix + "/partial_matC_block_" + std::to_string(blockId));
        graph.setTileMapping(partialSum[i][blockId], tileId);
      }
    }

    tileOffset += nTilePerGroup;
  }

  int gStart = 0;
  int gEnd = 0;

  assert(static_cast<int>(mulCSVec.size()) == nPass);
  assert(static_cast<int>(reduceCSVec.size()) == nPass);
  for (int p = 0; p < nPass; p++) {
    tileOffset = 0;
    gEnd += nGroup;

    poplar::ComputeSet &mulCS = mulCSVec[p];
    poplar::ComputeSet &reduceCS = reduceCSVec[p];

    for (int g = gStart; g < gEnd; g++) {
      for (unsigned r = 0; r < nRowC; r++) {
        unsigned int tileId = resultTileAssignment[r] + tileOffset;
        for (unsigned c = 0; c < nColC; c++) {
          if (blockIdMatrixC[r][c] == -1) {
            continue;
          }

          std::vector<poplar::Tensor> inputA, inputB;
          inputA.push_back(blockDataA[blockIdMatrixA[r][g]]);
          inputB.push_back(blockDataB[blockIdMatrixB[g][c]]);

          addConv1x1Vertex(graph, inputA, inputB,
                           partialSum[g - gStart][blockIdMatrixC[r][c]], tileId,
                           mulCS, debugPrefix);
        }
      }
      tileOffset += nTilePerGroup;
    }

    for (unsigned r = 0; r < nRowC; r++) {
      for (unsigned c = 0; c < nColC; c++) {
        int blockId = blockIdMatrixC[r][c];
        if (blockId == -1) {
          continue;
        }
        unsigned int tileId = blockTileId[blockId];

        std::vector<poplar::Tensor> inputBlocks;
        for (int i = 0; i < nGroup; i++) {
          inputBlocks.push_back(partialSum[i][blockId]);
        }

        poplar::Tensor out = blockDataC[blockId];
        addReduceVertex(graph, inputBlocks, out, tileId, reduceCS);
      }
    }

    gStart = gEnd;
  }
}

} // namespace experimental
} // namespace popsparse
