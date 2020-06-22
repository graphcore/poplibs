// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphBlock.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Zero.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <poplin/codelets.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphBlock::HyperGraphBlock(const BlockMatrix &A, const BlockMatrix &B,
                                 poplar::Type inDataTypeIn,
                                 poplar::Type outDataTypeIn,
                                 poplar::Type partialDataTypeIn, int nTileIn,
                                 float memoryCycleRatioIn,
                                 int nMulNodesSplitFactorIn)
    : HyperGraph(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn, nTileIn),
      gNodeId(0), gEdgeId(0), memoryCycleRatio(memoryCycleRatioIn),
      nMulsOnVNode(nMulNodesSplitFactorIn) {}

void HyperGraphBlock::createGraphMatMul(poplar::Graph &graph,
                                        const std::string &debugPrefix) {
  if (matA.isDense()) {
    createGraphMatMulDSD(graph, debugPrefix);
  } else {
    throw poputil::poplibs_error("Not implemented");
  }
  partitionGraph();
}

void HyperGraphBlock::createGraphMatMulSparsifyResult(
    poplar::Graph &graph, const unsigned char *sparsity,
    const std::string &debugPrefix) {
  if (matA.isDense() && matB.isDense()) {
    createGraphMatMulDDSSparsiryResult(graph, sparsity, debugPrefix);
  } else {
    throw poputil::poplibs_error("Not implemented");
  }
  partitionGraph();
}

void HyperGraphBlock::createGraphMatMulDSD(poplar::Graph &graph,
                                           const std::string &debugPrefix) {
  assert(!matC);
  assert(matA.getColCount() == matB.getRowCount());
  assert(matA.getBlockCol() == matB.getBlockRow());

  // block id look up matrix for A
  auto blockIdMatrixA = matA.getBlockIdMatrix();

  const int nRowA = matA.getBlockRowCount();
  const int nColA = matA.getBlockColCount();

  // block id look up matrix for B
  auto blockIdMatrixB = matB.getBlockIdMatrix();

  const int nRowB = matB.getBlockRowCount();
  const int nColB = matB.getBlockColCount();

  // hyper edge id look up matrix for A
  auto hyperEdgeIdA = populateNodesA(nRowA, nColA, blockIdMatrixA);
  // hyper edge id look up matrix for B
  auto hyperEdgeIdB = populateNodesB(nRowB, nColB, blockIdMatrixB);

  // Populate node and hyper edge for matrix C
  const int blockRowC = matA.getBlockRow();
  const int blockColC = matB.getBlockCol();
  const int rowC = matA.getRowCount();
  const int colC = matB.getColCount();
  const int nRowC = rowC / blockRowC;
  const int nColC = colC / blockColC;

  std::unique_ptr<BlockDenseMatrix> matCDense(
      new BlockDenseMatrix(rowC, colC, blockRowC, blockColC, false));

  // block id look up matrix for C
  auto blockIdMatrixC = matCDense->getBlockIdMatrix();

  int numMuls = 0;
  for (int i = 0; i < nRowC; i++) {
    for (int j = 0; j < nColC; j++) {
      for (int k = 0; k < nColA; k++) {
        // matrix A is dense, only need to check matrix B
        if (blockIdMatrixB[k][j] != -1) {
          numMuls++;
        }
      }
    }
  }

  poplar::Tensor matCTensor =
      matCDense->createTensor(graph, outDataType, "/matC");
  matCDense->setBlockTensor(matCTensor);

  auto hyperEdgeIdC = populateNodesC(nRowC, nColC, blockIdMatrixC);

  // nArgV is the desired average number of muls per node V (vertex)
  // We use this number to group the block pairs that contribute to the same
  // block in the matrix C. If this number is too small, we need more time to
  // sum up the partials. If this number is too big, we may have load balance
  // issue.
  unsigned int nAvgV = numMuls / nTile / nMulsOnVNode;

  logging::info("Total number of muls: {}", numMuls);
  logging::info("Average number of muls per node V: {}", nAvgV);

  // populate multiply nodes V
  populateNodesV(nRowC, nColC, nColA, nAvgV, blockIdMatrixA, blockIdMatrixB,
                 blockIdMatrixC, hyperEdgeIdA, hyperEdgeIdB, hyperEdgeIdC);

  // save the pointer to matrix C
  matC = std::move(matCDense);
  setupWeights(graph, numMuls);
}

void HyperGraphBlock::createGraphMatMulDDSSparsiryResult(
    poplar::Graph &graph, const unsigned char *sparsity,
    const std::string &debugPrefix) {
  assert(!matC);
  assert(matA.getColCount() == matB.getRowCount());
  assert(matA.getBlockCol() == matB.getBlockRow());

  // block id look up matrix for A
  auto blockIdMatrixA = matA.getBlockIdMatrix();

  const int nRowA = matA.getBlockRowCount();
  const int nColA = matA.getBlockColCount();

  // block id look up matrix for B
  auto blockIdMatrixB = matB.getBlockIdMatrix();

  const int nRowB = matB.getBlockRowCount();
  const int nColB = matB.getBlockColCount();

  // hyper edge id look up matrix for A
  auto hyperEdgeIdA = populateNodesA(nRowA, nColA, blockIdMatrixA);
  // hyper edge id look up matrix for B
  auto hyperEdgeIdB = populateNodesB(nRowB, nColB, blockIdMatrixB);

  // Populate node and hyper edge for matrix C
  const int blockRowC = matA.getBlockRow();
  const int blockColC = matB.getBlockCol();
  const int rowC = matA.getRowCount();
  const int colC = matB.getColCount();
  const int nRowC = rowC / blockRowC;
  const int nColC = colC / blockColC;
  std::unique_ptr<BlockSparseMatrix> matCSparse(
      new BlockSparseMatrix(rowC, colC, blockRowC, blockColC, false, sparsity));

  // block id look up matrix for C
  auto blockIdMatrixC = matCSparse->getBlockIdMatrix();

  poplar::Tensor matCTensor =
      matCSparse->createTensor(graph, outDataType, debugPrefix + "/matC");
  matCSparse->setBlockTensor(matCTensor);

  auto hyperEdgeIdC = populateNodesC(nRowC, nColC, blockIdMatrixC);

  int numMuls = 0;
  for (int i = 0; i < nRowC; i++) {
    for (int j = 0; j < nColC; j++) {
      if (blockIdMatrixC[i][j] != -1) {
        numMuls += nColA;
      }
    }
  }

  unsigned int nAvgV = numMuls / nTile / nMulsOnVNode;

  logging::info("Total number of muls: {}", numMuls);
  logging::info("Average number of muls per node V: {}", nAvgV);

  // populate multiply nodes V
  populateNodesV(nRowC, nColC, nColA, nAvgV, blockIdMatrixA, blockIdMatrixB,
                 blockIdMatrixC, hyperEdgeIdA, hyperEdgeIdB, hyperEdgeIdC);

  // save the pointer to matrix C
  matC = std::move(matCSparse);
  setupWeights(graph, numMuls);
}

void HyperGraphBlock::setupWeights(const poplar::Graph &graph, int numMuls) {
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

std::vector<std::vector<unsigned int>> HyperGraphBlock::populateDataNodes(
    int nRow, int nCol, const std::vector<std::vector<int>> &blockIdMatrix,
    std::vector<DataNode> &nodeGroup, std::vector<HyperEdge> &edgeGroup) {
  std::vector<std::vector<unsigned int>> hyperEdgeId(
      nRow, std::vector<unsigned int>(nCol));
  unsigned int edgeId = 0;
  for (int i = 0; i < nRow; i++) {
    for (int j = 0; j < nCol; j++) {
      if (blockIdMatrix[i][j] != -1) {
        nodeGroup.emplace_back(gNodeId, blockIdMatrix[i][j]);

        edgeGroup.emplace_back(gEdgeId);
        HyperEdge &e = edgeGroup.back();
        e.in.push_back(gNodeId);

        hyperEdgeId[i][j] = edgeId++;

        gNodeId++;
        gEdgeId++;
      }
    }
  }
  return hyperEdgeId;
}

std::vector<std::vector<unsigned int>> HyperGraphBlock::populateNodesA(
    int nRowA, int nColA, const std::vector<std::vector<int>> &blockIdMatrixA) {
  return populateDataNodes(nRowA, nColA, blockIdMatrixA, nodeA, edgeA);
}

std::vector<std::vector<unsigned int>> HyperGraphBlock::populateNodesB(
    int nRowB, int nColB, const std::vector<std::vector<int>> &blockIdMatrixB) {
  return populateDataNodes(nRowB, nColB, blockIdMatrixB, nodeB, edgeB);
}

std::vector<std::vector<unsigned int>> HyperGraphBlock::populateNodesC(
    int nRowC, int nColC, const std::vector<std::vector<int>> &blockIdMatrixC) {
  std::vector<std::vector<unsigned int>> hyperEdgeIdC(
      nRowC, std::vector<unsigned int>(nColC));
  unsigned int edgeId = 0;
  for (int i = 0; i < nRowC; i++) {
    for (int j = 0; j < nColC; j++) {
      if (blockIdMatrixC[i][j] != -1) {
        nodeC.emplace_back(gNodeId, blockIdMatrixC[i][j]);
        edgeC.emplace_back(gEdgeId);
        HyperEdge &e = edgeC.back();
        e.out.push_back(gNodeId);
        hyperEdgeIdC[i][j] = edgeId++;
        gNodeId++;
        gEdgeId++;
      }
    }
  }
  return hyperEdgeIdC;
}

void HyperGraphBlock::populateNodesV(
    int nRowC, int nColC, int nColA, unsigned int nAvgV,
    const std::vector<std::vector<int>> &blockIdMatrixA,
    const std::vector<std::vector<int>> &blockIdMatrixB,
    const std::vector<std::vector<int>> &blockIdMatrixC,
    const std::vector<std::vector<unsigned int>> &hyperEdgeIdA,
    const std::vector<std::vector<unsigned int>> &hyperEdgeIdB,
    const std::vector<std::vector<unsigned int>> &hyperEdgeIdC) {
  // populate multiply nodes V
  for (int i = 0; i < nRowC; i++) {
    for (int j = 0; j < nColC; j++) {
      if (blockIdMatrixC[i][j] == -1) {
        continue;
      }

      std::vector<std::pair<unsigned int, unsigned int>> aList, bList;
      for (int k = 0; k < nColA; k++) {
        if (blockIdMatrixA[i][k] == -1 || blockIdMatrixB[k][j] == -1) {
          continue;
        }

        aList.push_back(std::make_pair(i, k));
        bList.push_back(std::make_pair(k, j));

        if (aList.size() >= nAvgV) {
          populateNodeV(i, j, aList, bList, blockIdMatrixA, blockIdMatrixB,
                        hyperEdgeIdA, hyperEdgeIdB, hyperEdgeIdC);
          aList.clear();
          bList.clear();
        }
      }

      // handle the blocks less than vAvgN
      if (aList.size() != 0) {
        populateNodeV(i, j, aList, bList, blockIdMatrixA, blockIdMatrixB,
                      hyperEdgeIdA, hyperEdgeIdB, hyperEdgeIdC);
        aList.clear();
        bList.clear();
      }
    }
  }
}

void HyperGraphBlock::populateNodeV(
    int row, int col,
    const std::vector<std::pair<unsigned int, unsigned int>> &aList,
    const std::vector<std::pair<unsigned int, unsigned int>> &bList,
    const std::vector<std::vector<int>> &blockIdMatrixA,
    const std::vector<std::vector<int>> &blockIdMatrixB,
    const std::vector<std::vector<unsigned int>> &hyperEdgeIdA,
    const std::vector<std::vector<unsigned int>> &hyperEdgeIdB,
    const std::vector<std::vector<unsigned int>> &hyperEdgeIdC) {
  std::vector<unsigned int> nodeAList, nodeBList;
  for (unsigned m = 0; m < aList.size(); m++) {
    unsigned int rmul = aList[m].first;
    unsigned int cmul = aList[m].second;
    nodeAList.push_back(blockIdMatrixA[rmul][cmul]);
    edgeA[hyperEdgeIdA[rmul][cmul]].out.push_back(gNodeId);

    rmul = bList[m].first;
    cmul = bList[m].second;
    nodeBList.push_back(blockIdMatrixB[rmul][cmul]);
    edgeB[hyperEdgeIdB[rmul][cmul]].out.push_back(gNodeId);
  }
  nodeV.emplace_back(gNodeId, nodeAList, nodeBList);
  edgeC[hyperEdgeIdC[row][col]].in.push_back(gNodeId);

  gNodeId++;
}

void HyperGraphBlock::addCodelets(poplar::Graph &graph) {
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
}

void HyperGraphBlock::createProgramMatMul(poplar::Graph &graph,
                                          SubBlockMask subBlockMask,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugPrefix) {
  std::map<unsigned int, poplar::Tensor> partialData;
  std::vector<unsigned int> nodeCTileId;

  createComputeSetMatMul(graph, partialData, nodeCTileId, prog, debugPrefix);

  createComputeSetReduce(graph, partialData, nodeCTileId, prog, debugPrefix);

  if (subBlockMask != SubBlockMask::None) {
    applySubBlockMask(graph, subBlockMask, prog, debugPrefix);
  }
}

void HyperGraphBlock::createComputeSetMatMul(
    poplar::Graph &graph, std::map<unsigned int, poplar::Tensor> &partialData,
    std::vector<unsigned int> &nodeCTileId, poplar::program::Sequence &prog,
    const std::string &debugPrefix) {

  poplar::ComputeSet mulCS = graph.addComputeSet(debugPrefix + "/mulCS");
  if (!matB.getNeedTranspose()) {
    poplar::ComputeSet transposeCS =
        graph.addComputeSet(debugPrefix + "/transposeCS");
    createComputeSetMatMul(graph, partialData, nodeCTileId, mulCS, &transposeCS,
                           debugPrefix);
    prog.add(poplar::program::Execute(transposeCS));
  } else {
    createComputeSetMatMul(graph, partialData, nodeCTileId, mulCS, nullptr,
                           debugPrefix);
  }
  prog.add(poplar::program::Execute(mulCS));
}

void HyperGraphBlock::createComputeSetMatMul(
    poplar::Graph &graph, std::map<unsigned int, poplar::Tensor> &partialData,
    std::vector<unsigned int> &nodeCTileId, poplar::ComputeSet &mulCS,
    poplar::ComputeSet *transposeCS, const std::string &debugPrefix) {

  // set tile mapping for tensors in all nodes
  const std::vector<poplar::Tensor> &blockDataA = matA.getBlockTensor();
  for (const auto &n : nodeA) {
    unsigned int blockId = n.blockId;
    unsigned int nodeId = n.id;
    if (tileAssignment[nodeId] < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " + std::to_string(tileAssignment[nodeId]) +
          " For node " + std::to_string(nodeId));
    }
    unsigned int tileId = static_cast<unsigned int>(tileAssignment[nodeId]);
    graph.setTileMapping(blockDataA[blockId], tileId);
  }

  const std::vector<poplar::Tensor> &blockDataB = matB.getBlockTensor();
  std::vector<int> rhsBlockTileId(blockDataB.size());
  const std::string debugPrefix1 = debugPrefix + "transposed_matBblock_";
  for (const auto &n : nodeB) {
    unsigned int blockId = n.blockId;
    unsigned int nodeId = n.id;
    if (tileAssignment[nodeId] < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " + std::to_string(tileAssignment[nodeId]) +
          " For node " + std::to_string(nodeId));
    }
    unsigned int tileId = static_cast<unsigned int>(tileAssignment[nodeId]);
    graph.setTileMapping(blockDataB[blockId], tileId);
    rhsBlockTileId[blockId] = tileId;
  }

  std::vector<poplar::Tensor> processedBlockDataA, processedBlockDataB;
  preprocessBlocks(graph, matA, matB, processedBlockDataA, processedBlockDataB,
                   rhsBlockTileId, transposeCS, debugPrefix);

  unsigned int vNodeCount = 0;
  std::vector<int> tileNodes(nTile, 0);

  const std::string &debugPrefix2 = debugPrefix + "partial_block_";
  for (const auto &n : nodeV) {
    unsigned int nodeId = n.id;
    poplar::Tensor t = graph.addVariable(
        partialDataType,
        {static_cast<unsigned long>(matC->getBlockRow() * matC->getBlockCol())},
        debugPrefix2 + std::to_string(nodeId));
    if (tileAssignment[nodeId] < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " + std::to_string(tileAssignment[nodeId]) +
          " For node " + std::to_string(nodeId));
    }
    unsigned int tileId = static_cast<unsigned int>(tileAssignment[nodeId]);
    graph.setTileMapping(t, tileId);
    tileNodes[tileId] += n.idxB.size();

    partialData[nodeId] = t;

    vNodeCount++;
  }

  // compute the load balance
  int minNode = INT_MAX, maxNode = 0, totalNode = 0;
  for (int i = 0; i < nTile; i++) {
    minNode = std::min(tileNodes[i], minNode);
    maxNode = std::max(tileNodes[i], maxNode);
    totalNode += tileNodes[i];
  }

  if (logging::shouldLog(logging::Level::Info)) {
    float avgNode = static_cast<float>(totalNode) / nTile;
    float varNode = 0.0f;
    float maxBalance = 0;
    for (int i = 0; i < nTile; i++) {
      varNode += std::sqrt((tileNodes[i] - avgNode) * (tileNodes[i] - avgNode));
      float b = tileNodes[i] / avgNode;
      maxBalance = std::max(b, maxBalance);
    }

    varNode /= nTile;

    logging::info("min node: {} max node: {}", minNode, maxNode);
    logging::info("total node: {} avg node: {}", totalNode, avgNode);
    logging::info("varaiance {} load balance: {}", varNode, maxBalance);
  }

  nodeCTileId.resize(nodeC.size());
  const std::vector<poplar::Tensor> &blockDataC = matC->getBlockTensor();
  for (std::size_t i = 0; i < nodeC.size(); i++) {
    // Put node C in the tile that most of children are located
    std::map<int, int> tileMap;
    for (std::size_t v = 0; v < edgeC[i].in.size(); v++) {
      unsigned int idV = edgeC[i].in[v];
      if (tileAssignment[idV] < 0) {
        throw poputil::poplibs_error(
            "Invalid tile id: " + std::to_string(tileAssignment[idV]) +
            "For nodeV");
      }
      unsigned int tileId = static_cast<unsigned int>(tileAssignment[idV]);
      ++tileMap[tileId];
    }

    unsigned int tileId;
    int max = -1;
    for (const auto &t : tileMap) {
      if (max < t.second) {
        max = t.second;
        tileId = t.first;
      }
    }

    if (max == -1) {
      // This node does not have any input, so it is zero block
      // put it on a random tile
      tileId = getRandomTile();
    }

    assert(tileId >= 0);
    unsigned int blockId = nodeC[i].blockId;
    graph.setTileMapping(blockDataC[blockId], tileId);
    nodeCTileId[i] = tileId;
  }

  logTileAssignment(graph, tileAssignment, nodeCTileId);

  vNodeCount = 0;
  for (const auto &n : nodeV) {
    unsigned int nodeId = n.id;
    std::vector<poplar::Tensor> inputA, inputB;
    for (unsigned int i = 0; i < n.idxA.size(); i++) {
      inputA.push_back(processedBlockDataA[n.idxA[i]]);
      inputB.push_back(processedBlockDataB[n.idxB[i]]);
    }

    std::vector<poplar::Tensor> out;
    out.push_back(partialData[nodeId]);

    if (tileAssignment[nodeV[vNodeCount].id] < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " +
          std::to_string(tileAssignment[nodeV[vNodeCount].id]) + "For nodeV");
    }
    unsigned int tileId =
        static_cast<unsigned int>(tileAssignment[nodeV[vNodeCount].id]);

    vNodeCount++;

    addConv1x1Vertex(graph, inputA, inputB, partialData[nodeId], tileId, mulCS,
                     debugPrefix);
  }
}

void HyperGraphBlock::createComputeSetReduce(
    poplar::Graph &graph,
    const std::map<unsigned int, poplar::Tensor> &partialDataIn,
    const std::vector<unsigned int> &nodeCTileId,
    poplar::program::Sequence &prog, const std::string &debugPrefix) {

  poplar::ComputeSet reduceCS = graph.addComputeSet(debugPrefix + "/reduceCS");
  createComputeSetReduce(graph, partialDataIn, nodeCTileId, reduceCS,
                         debugPrefix);
  prog.add(poplar::program::Execute(reduceCS));
}

void HyperGraphBlock::createComputeSetReduce(
    poplar::Graph &graph,
    const std::map<unsigned int, poplar::Tensor> &partialDataIn,
    const std::vector<unsigned int> &nodeCTileId, poplar::ComputeSet &reduceCS,
    const std::string &debugPrefix) {
  unsigned int minC = INT_MAX, maxC = 0;

  for (unsigned int i = 0; i < edgeC.size(); i++) {
    poplar::Tensor output = matC->getBlockTensor()[nodeC[i].blockId];

    HyperEdge &e = edgeC[i];
    if (e.in.empty()) {
      popops::zero(graph, output, nodeCTileId[i], reduceCS);
      continue;
    }

    minC = std::min(static_cast<uint>(e.in.size()), minC);
    maxC = std::max(static_cast<uint>(e.in.size()), maxC);

    std::vector<poplar::Tensor> partialBlocks;
    for (const auto &in : e.in) {
      auto iter = partialDataIn.find(in);
      if (iter == partialDataIn.end()) {
        throw poputil::poplibs_error("Error: incomplete input partial data");
        return;
      }
      partialBlocks.push_back(iter->second);
    }

    addReduceVertex(graph, partialBlocks, output, nodeCTileId[i], reduceCS);
  }

  logging::info("Total reduction node = {}, min = {}, max = {} ", edgeC.size(),
                minC, maxC);
}

static const float KBf = 1024.0f;
static const std::size_t KBu = 1024;

// Knowing the variables and vertices to tile assignment,
// calculates the number of nodes and amount of memory for every tile.
// Builds nodes number and memory histogram and outputs the largest and smallest
// tile statistics. Memory histogram is given by 2 compute sets: matmul and
// reduce.
//
// Assumptions:
// 1. Variables A,B,C are always live
// 2. Copies of variables A and B for matmul are only live on matmul compute set
// 3. Copies of partial results are only live on reduce compute set
//
// Not taken into account yet:
// 1. Possible transposed copies of variable B
// 2. Vertices memory
//
// The following notation is used:
// A, B, C - permanent variables A, B, C
// V - vertices
// P - partials
// a, b - copied variables A, B
// p - copied partials
//
// For Nodes number Histogram:
// Each bar length is proportional to the number of tiles
// that contain a particular nodes number range
// This is how normally histograms are built
//
// For memory Histogram:
// Each bar length is proportional to the total memory on
// those tiles that contain a particular Kb range.
// This is different from how normally histograms are built.
// It is done to better visualise memory allocation.
// For memory histogram 2 bars are given for eact range:
// for matmul compute set and reduce compute set.
// Maximum tile statistics is given independently for
// both matmul and reduce.
// Minimum tile statistics is given only for one tile -
// minimum matmul or minimum reduce whichever is larger.
//
// For both histograms, bars are broken into subbars,
// labeled by variables type so that each subbar's length is
// proportional to that variables share.
// A mix of leftover variables is labeled as X
// Output example:
/*
 Tiles assignment by number of nodes:
 Range    	Histogram (num tiles)
 0 - 1	AAAAAAX
 2 - 3	X
 4 - 5	AAAABVCX
 6 - 7
 AAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCX
 8 - 9	X
 10 - 11	X
 12 - 13	VVVVVVVVVCCCCCCCCCXX
 14 - 15	VVVVVVCCCCCC
 16 - 17
 18 - 19
 Maximum nodes on tile 626=14: A=0, B=0, V=7, C=7
 Minimum nodes on tile 777=6: A=0, B=0, V=6, C=0

 Legend:
 A, B, C - permanent variables
 P - partials
 a, b - A, B copied variables
 p - copied partials
 X - mix of variables
 Tiles assignment by memory:
 Range (Kb) Histogram (Kb share)
 0   - 5   mm AAAAAAAABBBBXX
           rd AAAAAAAABBBBX
 5   - 11  mm PPPPPPPPaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbXX
           rd PPPPPPPX
 11  - 16  mm aabbX
           rd
 16  - 22  mm abX
           rd
 22  - 27  mm X
           rd
 27  - 33  mm
           rd
 33  - 38  mm
 CCCPPPaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbXXX
           rd CCCPPPpppXX
 38  - 44  mm
 CCCCPPPPaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbXX
           rd CCCCPPPPppppXX
 44  - 49  mm aaaaaabbbbbbXX
           rd X
 49  - 55  mm abXX
           rd
 Maximum matmul Kb on tile 56=55.5: A=0, B=0, C=1.75, P=1.75, a=19.5, b=32.5
 Maximum reduce Kb on tile 1=5.25: B=0, P=0, C=1.75, p=1.75
 Minimum Kb on tile 1058=0: A=0, B=0, C=0, P=0, a=0, b=0, p=0
*/
void HyperGraphBlock::logTileAssignment(
    const poplar::Graph &graph, const std::vector<int> &tileAssignment,
    const std::vector<unsigned int> &nodeCTileId) {
  if (!logging::shouldLog(logging::Level::Debug)) {
    return;
  }

  assert(nodeC.size() == nodeCTileId.size());
  std::unordered_map<unsigned int, int> tileAssignmentC;
  for (std::size_t i = 0; i < nodeC.size(); ++i) {
    const auto &n = nodeC[i];
    tileAssignmentC[n.id] = nodeCTileId[i];
  }

  // Histogram by node numbers
  {
    const std::size_t MAX_BUCKETS = 10;
    const std::size_t MAX_BAR_SIZE = 120;

    std::size_t maxNodesOnTile = 0;
    std::size_t minNodesOnTile = 256 * 1024;
    int idxMaxOccupiedTile = 0;
    int idxMinOccupiedTile = 0;
    std::vector<std::size_t> nodesAByTiles(nTile, 0);
    std::vector<std::size_t> nodesBByTiles(nTile, 0);
    std::vector<std::size_t> nodesVByTiles(nTile, 0);
    std::vector<std::size_t> nodesCByTiles(nTile, 0);
    std::vector<std::size_t> nodesTotalByTiles(nTile, 0);

    std::function<void(int, std::size_t)> updateMaxNodesTile =
        [&](int idxTile, std::size_t nodesOnTile) {
          if (nodesOnTile > maxNodesOnTile) {
            maxNodesOnTile = nodesOnTile;
            idxMaxOccupiedTile = idxTile;
          }
          if (nodesOnTile <= minNodesOnTile) {
            minNodesOnTile = nodesOnTile;
            idxMinOccupiedTile = idxTile;
          }
        };
    for (std::size_t i = 0; i < nodeA.size(); ++i) {
      const auto &n = nodeA[i];
      int idxTile = tileAssignment[n.id];
      assert(idxTile >= 0);
      ++nodesAByTiles[idxTile];
      std::size_t nodesOnTile = ++nodesTotalByTiles[idxTile];
      updateMaxNodesTile(idxTile, nodesOnTile);
    }
    for (std::size_t i = 0; i < nodeB.size(); ++i) {
      const auto &n = nodeB[i];
      int idxTile = tileAssignment[n.id];
      assert(idxTile >= 0);
      ++nodesBByTiles[idxTile];
      std::size_t nodesOnTile = ++nodesTotalByTiles[idxTile];
      updateMaxNodesTile(idxTile, nodesOnTile);
    }
    for (std::size_t i = 0; i < nodeV.size(); ++i) {
      const auto &n = nodeV[i];
      int idxTile = tileAssignment[n.id];
      assert(idxTile >= 0);
      ++nodesVByTiles[idxTile];
      std::size_t nodesOnTile = ++nodesTotalByTiles[idxTile];
      updateMaxNodesTile(idxTile, nodesOnTile);
    }
    for (std::size_t i = 0; i < nodeC.size(); ++i) {
      const auto &n = nodeC[i];
      int idxTile = tileAssignmentC.at(n.id);
      assert(idxTile >= 0);
      ++nodesCByTiles[idxTile];
      std::size_t nodesOnTile = ++nodesTotalByTiles[idxTile];
      updateMaxNodesTile(idxTile, nodesOnTile);
    }

    const std::size_t numBuckets = std::min(MAX_BUCKETS, maxNodesOnTile + 1);
    const std::size_t bucketSizeNodes =
        (maxNodesOnTile + numBuckets) / numBuckets;
    std::vector<std::size_t> tilesPerBucket(numBuckets, 0);
    std::vector<std::size_t> nodesPerBucket(numBuckets, 0);
    std::vector<std::size_t> nodesAPerBucket(numBuckets, 0);
    std::vector<std::size_t> nodesBPerBucket(numBuckets, 0);
    std::vector<std::size_t> nodesVPerBucket(numBuckets, 0);
    std::vector<std::size_t> nodesCPerBucket(numBuckets, 0);

    for (std::size_t idxTile = 0; idxTile < nodesTotalByTiles.size();
         ++idxTile) {
      std::size_t idxBucket = nodesTotalByTiles[idxTile] / bucketSizeNodes;
      assert(idxBucket < numBuckets);
      ++tilesPerBucket[idxBucket];
      nodesPerBucket[idxBucket] += nodesTotalByTiles[idxTile];
      nodesAPerBucket[idxBucket] += nodesAByTiles[idxTile];
      nodesBPerBucket[idxBucket] += nodesBByTiles[idxTile];
      nodesVPerBucket[idxBucket] += nodesVByTiles[idxTile];
      nodesCPerBucket[idxBucket] += nodesCByTiles[idxTile];
    }

    const std::size_t tilesPerCell =
        std::max(nTile / MAX_BAR_SIZE, std::size_t(1));

    logging::info("Number of nodes in A: {}", nodeA.size());
    logging::info("Number of nodes in B: {}", nodeB.size());
    logging::info("Number of nodes in C: {}", nodeC.size());
    logging::info("Number of nodes in V: {}", nodeV.size());

    // Nodes number Histogram:
    logging::debug("");
    logging::debug("Tiles assignment by number of nodes:");
    logging::debug("Range    \tHistogram (num tiles)");
    for (std::size_t idxBucket = 0, numNodessLow = 0; idxBucket < numBuckets;
         ++idxBucket) {
      std::size_t numNodesHigh = numNodessLow + bucketSizeNodes - 1;
      std::size_t barLen =
          (tilesPerBucket[idxBucket] + tilesPerCell - 1) / tilesPerCell;
      std::string bar;
      if (nodesPerBucket[idxBucket] > 0) {
        float nodesShareA = static_cast<float>(nodesAPerBucket[idxBucket]) /
                            nodesPerBucket[idxBucket];
        float nodesShareB = static_cast<float>(nodesBPerBucket[idxBucket]) /
                            nodesPerBucket[idxBucket];
        float nodesShareV = static_cast<float>(nodesVPerBucket[idxBucket]) /
                            nodesPerBucket[idxBucket];
        float nodesShareC = static_cast<float>(nodesCPerBucket[idxBucket]) /
                            nodesPerBucket[idxBucket];
        std::size_t barLenA = static_cast<std::size_t>(barLen * nodesShareA);
        std::size_t barLenB = static_cast<std::size_t>(barLen * nodesShareB);
        std::size_t barLenV = static_cast<std::size_t>(barLen * nodesShareV);
        std::size_t barLenC = static_cast<std::size_t>(barLen * nodesShareC);
        assert(barLen >= barLenA + barLenB + barLenV + barLenC);
        std::size_t barLenX = barLen - barLenA - barLenB - barLenV - barLenC;

        bar.append(barLenA, 'A');
        bar.append(barLenB, 'B');
        bar.append(barLenV, 'V');
        bar.append(barLenV, 'C');
        bar.append(barLenX, 'X');
      } else {
        bar.append(barLen, '0');
      }

      logging::debug("{} - {}\t{}", numNodessLow, numNodesHigh, bar.c_str());
      numNodessLow += bucketSizeNodes;
    }
    logging::debug(
        "Maximum nodes on tile {}={}: A={}, B={}, V={}, C={}",
        idxMaxOccupiedTile, nodesTotalByTiles[idxMaxOccupiedTile],
        nodesAByTiles[idxMaxOccupiedTile], nodesBByTiles[idxMaxOccupiedTile],
        nodesVByTiles[idxMaxOccupiedTile], nodesCByTiles[idxMaxOccupiedTile]);
    logging::debug(
        "Minimum nodes on tile {}={}: A={}, B={}, V={}, C={}",
        idxMinOccupiedTile, nodesTotalByTiles[idxMinOccupiedTile],
        nodesAByTiles[idxMinOccupiedTile], nodesBByTiles[idxMinOccupiedTile],
        nodesVByTiles[idxMinOccupiedTile], nodesCByTiles[idxMinOccupiedTile]);
  }

  // Histogram by memory
  {
    const std::size_t MAX_BUCKETS = 10;
    const std::size_t MAX_BAR_SIZE = 120;

    const int memWeightA = matA.getBlockRow() * matA.getBlockCol() *
                           graph.getTarget().getTypeSize(inDataType);
    const int memWeightB = matB.getBlockRow() * matB.getBlockCol() *
                           graph.getTarget().getTypeSize(inDataType);
    const int memWeightC = matC->getBlockRow() * matC->getBlockCol() *
                           graph.getTarget().getTypeSize(outDataType);
    const int memWeightV = matC->getBlockRow() * matC->getBlockCol() *
                           graph.getTarget().getTypeSize(partialDataType);

    std::vector<std::size_t> bytesAByTiles(nTile, 0);
    std::vector<std::size_t> bytesBByTiles(nTile, 0);
    std::vector<std::size_t> bytesVaByTiles(nTile, 0);
    std::vector<std::size_t> bytesVbByTiles(nTile, 0);
    std::vector<std::size_t> bytesVpByTiles(nTile, 0);
    std::vector<std::size_t> bytesCByTiles(nTile, 0);
    std::vector<std::size_t> bytesCpByTiles(nTile, 0);
    // For matmul we count A,B,C,P,a,b
    std::vector<std::size_t> bytesByTilesMatmul(nTile, 0);
    // For reduce we count A,B,C,P,p,B
    std::vector<std::size_t> bytesByTilesReduce(nTile, 0);

    for (std::size_t i = 0; i < nodeA.size(); ++i) {
      const auto &n = nodeA[i];
      int idxTile = tileAssignment[n.id];
      assert(idxTile >= 0 && idxTile < nTile);
      bytesAByTiles[idxTile] += memWeightA;
      bytesByTilesMatmul[idxTile] += memWeightA;
      bytesByTilesReduce[idxTile] += memWeightA;
    }
    for (std::size_t i = 0; i < nodeB.size(); ++i) {
      const auto &n = nodeB[i];
      int idxTile = tileAssignment[n.id];
      assert(idxTile >= 0 && idxTile < nTile);
      bytesBByTiles[idxTile] += memWeightB;
      bytesByTilesMatmul[idxTile] += memWeightB;
      bytesByTilesReduce[idxTile] += memWeightB;
    }
    for (std::size_t i = 0; i < nodeV.size(); ++i) {
      const auto &n = nodeV[i];
      int idxTile = tileAssignment[n.id];
      assert(idxTile >= 0 && idxTile < nTile);
      bytesVpByTiles[idxTile] += memWeightV;
      bytesByTilesMatmul[idxTile] += memWeightV;
      bytesByTilesReduce[idxTile] += memWeightV;
    }
    for (std::size_t i = 0; i < edgeA.size(); ++i) {
      const auto &e = edgeA[i];
      assert(e.in.size() == 1);
      unsigned int idNodeA = e.in[0];
      int idxTileA = tileAssignment[idNodeA];
      assert(idxTileA >= 0 && idxTileA < nTile);
      std::unordered_set<int> tilesToCopy;
      for (std::size_t i_out = 0; i_out < e.out.size(); ++i_out) {
        unsigned int idNodeV = e.out[i_out];
        int idxTileV = tileAssignment[idNodeV];
        assert(idxTileV >= 0 && idxTileV < nTile);
        if (idxTileA != idxTileV) {
          if (tilesToCopy.find(idxTileV) == tilesToCopy.end()) {
            bytesVaByTiles[idxTileV] += memWeightA;
            bytesByTilesMatmul[idxTileV] += memWeightA;
            tilesToCopy.insert(idxTileV);
          }
        }
      }
    }
    for (std::size_t i = 0; i < edgeB.size(); ++i) {
      const auto &e = edgeB[i];
      assert(e.in.size() == 1);
      unsigned int idNodeB = e.in[0];
      int idxTileB = tileAssignment[idNodeB];
      assert(idxTileB >= 0 && idxTileB < nTile);
      std::unordered_set<int> tilesToCopy;
      for (std::size_t i_out = 0; i_out < e.out.size(); ++i_out) {
        unsigned int idNodeV = e.out[i_out];
        int idxTileV = tileAssignment[idNodeV];
        assert(idxTileV >= 0 && idxTileV < nTile);
        if (idxTileB != idxTileV) {
          if (tilesToCopy.find(idxTileV) == tilesToCopy.end()) {
            bytesVbByTiles[idxTileV] += memWeightB;
            bytesByTilesMatmul[idxTileV] += memWeightB;
            tilesToCopy.insert(idxTileV);
          }
        }
      }
    }

    for (std::size_t i = 0; i < nodeC.size(); ++i) {
      const auto &n = nodeC[i];
      int idxTile = tileAssignmentC.at(n.id);
      assert(idxTile >= 0 && idxTile < nTile);
      bytesCByTiles[idxTile] += memWeightC;
      bytesByTilesMatmul[idxTile] += memWeightC;
      bytesByTilesReduce[idxTile] += memWeightC;
    }
    for (std::size_t i = 0; i < edgeC.size(); ++i) {
      const auto &e = edgeC[i];
      assert(e.out.size() == 1);
      unsigned int idNodeC = e.out[0];
      int idxTileC = tileAssignmentC.at(idNodeC);
      assert(idxTileC >= 0 && idxTileC < nTile);
      for (std::size_t i_in = 0; i_in < e.in.size(); ++i_in) {
        unsigned int idNodeV = e.in[i_in];
        int idxTileV = tileAssignment[idNodeV];
        assert(idxTileV >= 0 && idxTileV < nTile);
        if (idxTileV != idxTileC) {
          bytesCpByTiles[idxTileC] += memWeightV;
          bytesByTilesReduce[idxTileC] += memWeightV;
        }
      }
    }

    std::size_t maxBytesOnTileMatmul = 0;
    std::size_t idxMaxOccupiedTileMatmul = 0;
    std::size_t maxBytesOnTileReduce = 0;
    std::size_t idxMaxOccupiedTileReduce = 0;

    std::size_t minBytesOnTile = 256 * KBu;
    std::size_t idxMinOccupiedTile = 0;
    for (int i = 0; i < nTile; ++i) {
      if (bytesByTilesMatmul[i] > maxBytesOnTileMatmul) {
        maxBytesOnTileMatmul = bytesByTilesMatmul[i];
        idxMaxOccupiedTileMatmul = i;
      }
      if (bytesByTilesReduce[i] > maxBytesOnTileReduce) {
        maxBytesOnTileReduce = bytesByTilesReduce[i];
        idxMaxOccupiedTileReduce = i;
      }
      std::size_t maxBytes =
          std::max(bytesByTilesMatmul[i], bytesByTilesReduce[i]);
      if (maxBytes < minBytesOnTile) {
        minBytesOnTile = maxBytes;
        idxMinOccupiedTile = i;
      }
    }
    std::size_t maxBytesOnTile =
        std::max(maxBytesOnTileMatmul, maxBytesOnTileReduce);

    const std::size_t numBuckets =
        std::min(MAX_BUCKETS, maxBytesOnTile / KBu + 1);
    const std::size_t bucketSizeBytes =
        std::max((maxBytesOnTile + numBuckets) / numBuckets, KBu);
    std::vector<std::size_t> bytesAPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesBPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesVpPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesVaPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesVbPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesCPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesCpPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesMatmulPerBucket(numBuckets, 0);
    std::vector<std::size_t> bytesReducePerBucket(numBuckets, 0);

    std::size_t maxBytesPerBucket = 0;
    for (int idxTile = 0; idxTile < nTile; ++idxTile) {
      std::size_t bytesOnTileMax =
          std::max(bytesByTilesMatmul[idxTile], bytesByTilesReduce[idxTile]);
      std::size_t idxBucket = bytesOnTileMax / bucketSizeBytes;
      assert(idxBucket < numBuckets);
      bytesAPerBucket[idxBucket] += bytesAByTiles[idxTile];
      bytesBPerBucket[idxBucket] += bytesBByTiles[idxTile];
      bytesVpPerBucket[idxBucket] += bytesVpByTiles[idxTile];
      bytesVaPerBucket[idxBucket] += bytesVaByTiles[idxTile];
      bytesVbPerBucket[idxBucket] += bytesVbByTiles[idxTile];
      bytesCPerBucket[idxBucket] += bytesCByTiles[idxTile];
      bytesCpPerBucket[idxBucket] += bytesCpByTiles[idxTile];
      bytesMatmulPerBucket[idxBucket] =
          bytesAPerBucket[idxBucket] + bytesBPerBucket[idxBucket] +
          bytesCPerBucket[idxBucket] + bytesVpPerBucket[idxBucket] +
          bytesVaPerBucket[idxBucket] + bytesVbPerBucket[idxBucket];
      bytesReducePerBucket[idxBucket] =
          bytesAPerBucket[idxBucket] + bytesBPerBucket[idxBucket] +
          bytesCPerBucket[idxBucket] + bytesCpPerBucket[idxBucket] +
          bytesVpPerBucket[idxBucket];
      maxBytesPerBucket = std::max(maxBytesPerBucket,
                                   std::max(bytesMatmulPerBucket[idxBucket],
                                            bytesReducePerBucket[idxBucket]));
    }
    const std::size_t bytesPerCell = std::max(
        (maxBytesPerBucket + MAX_BAR_SIZE - 1) / MAX_BAR_SIZE, std::size_t(1));

    // Memory Histogram:
    logging::debug("");
    logging::debug("Legend:");
    logging::debug("A, B, C - permanent variables");
    logging::debug("P - partials");
    logging::debug("a, b - A, B copied variables");
    logging::debug("p - copied partials");
    logging::debug("X - mix of variables");
    logging::debug("Tiles assignment by memory:");
    logging::debug("Range (Kb) Histogram (Kb share)");
    for (std::size_t idxBucket = 0, bytesLow = 0; idxBucket < numBuckets;
         ++idxBucket) {
      std::size_t bytesHigh = bytesLow + bucketSizeBytes - 1;
      std::size_t kBLow = bytesLow / KBu;
      std::size_t kBHigh = bytesHigh / KBu;

      std::size_t bytesPerBucket = std::max(bytesMatmulPerBucket[idxBucket],
                                            bytesReducePerBucket[idxBucket]);
      std::size_t barLen = (bytesPerBucket + bytesPerCell - 1) / bytesPerCell;
      assert(barLen <= MAX_BAR_SIZE + 1);
      {
        // Matmul bar
        std::string bar;

        std::size_t bytesMatmul = bytesMatmulPerBucket[idxBucket];
        if (bytesMatmul > 0) {
          float bytesShareA =
              static_cast<float>(bytesAPerBucket[idxBucket]) / bytesMatmul;
          float bytesShareB =
              static_cast<float>(bytesBPerBucket[idxBucket]) / bytesMatmul;
          float bytesShareC =
              static_cast<float>(bytesCPerBucket[idxBucket]) / bytesMatmul;
          float bytesShareVp =
              static_cast<float>(bytesVpPerBucket[idxBucket]) / bytesMatmul;
          float bytesShareVa =
              static_cast<float>(bytesVaPerBucket[idxBucket]) / bytesMatmul;
          float bytesShareVb =
              static_cast<float>(bytesVbPerBucket[idxBucket]) / bytesMatmul;

          float matmulRatio = static_cast<float>(bytesMatmul) / bytesPerBucket;
          std::size_t barLenMatmul =
              static_cast<std::size_t>(barLen * matmulRatio);
          assert(barLenMatmul <= MAX_BAR_SIZE + 1);

          std::size_t barLenA =
              static_cast<std::size_t>(barLenMatmul * bytesShareA);
          std::size_t barLenB =
              static_cast<std::size_t>(barLenMatmul * bytesShareB);
          std::size_t barLenC =
              static_cast<std::size_t>(barLenMatmul * bytesShareC);
          std::size_t barLenVp =
              static_cast<std::size_t>(barLenMatmul * bytesShareVp);
          std::size_t barLenVa =
              static_cast<std::size_t>(barLenMatmul * bytesShareVa);
          std::size_t barLenVb =
              static_cast<std::size_t>(barLenMatmul * bytesShareVb);
          assert(barLenMatmul >=
                 barLenA + barLenB + barLenC + barLenVp + barLenVa + barLenVb);
          std::size_t barLenX = barLenMatmul - barLenA - barLenB - barLenC -
                                barLenVp - barLenVa - barLenVb;

          bar.append(barLenA, 'A');
          bar.append(barLenB, 'B');
          bar.append(barLenC, 'C');
          bar.append(barLenVp, 'P');
          bar.append(barLenVa, 'a');
          bar.append(barLenVb, 'b');
          bar.append(barLenX, 'X');
        } else {
          bar.append(barLen, '0');
        }
        assert(bar.size() <= MAX_BAR_SIZE);
        logging::debug("{:<3} - {:<3} mm {}", kBLow, kBHigh, bar.c_str());
      }
      {
        // Reduce bar
        std::string bar;
        std::size_t bytesReduce = bytesReducePerBucket[idxBucket];
        if (bytesReduce > 0) {
          float bytesShareA =
              static_cast<float>(bytesAPerBucket[idxBucket]) / bytesReduce;
          float bytesShareB =
              static_cast<float>(bytesBPerBucket[idxBucket]) / bytesReduce;
          float bytesShareC =
              static_cast<float>(bytesCPerBucket[idxBucket]) / bytesReduce;
          float bytesShareVp =
              static_cast<float>(bytesVpPerBucket[idxBucket]) / bytesReduce;
          float bytesShareCp =
              static_cast<float>(bytesCpPerBucket[idxBucket]) / bytesReduce;
          float reduceRatio = static_cast<float>(bytesReduce) / bytesPerBucket;
          std::size_t barLenReduce =
              static_cast<std::size_t>(barLen * reduceRatio);

          std::size_t barLenA =
              static_cast<std::size_t>(barLenReduce * bytesShareA);
          std::size_t barLenB =
              static_cast<std::size_t>(barLenReduce * bytesShareB);
          std::size_t barLenC =
              static_cast<std::size_t>(barLenReduce * bytesShareC);
          std::size_t barLenVp =
              static_cast<std::size_t>(barLenReduce * bytesShareVp);
          std::size_t barLenCp =
              static_cast<std::size_t>(barLenReduce * bytesShareCp);

          assert(barLenReduce >=
                 barLenA + barLenB + barLenC + barLenVp + barLenCp);
          std::size_t barLenX =
              barLenReduce - barLenA - barLenB - barLenC - barLenVp - barLenCp;

          bar.append(barLenA, 'A');
          bar.append(barLenB, 'B');
          bar.append(barLenC, 'C');
          bar.append(barLenVp, 'P');
          bar.append(barLenCp, 'p');
          bar.append(barLenX, 'X');
        } else {
          bar.append(barLen, '0');
        }
        logging::debug("          rd {}", bar.c_str());
      }

      bytesLow += bucketSizeBytes;
    }
    logging::debug(
        "Maximum matmul Kb on tile {}={}: A={}, B={}, C={}, P={}, a={}, b={}",
        idxMaxOccupiedTileMatmul, maxBytesOnTileMatmul / KBf,
        bytesAByTiles[idxMaxOccupiedTileMatmul] / KBf,
        bytesBByTiles[idxMaxOccupiedTileMatmul] / KBf,
        bytesCByTiles[idxMaxOccupiedTileMatmul] / KBf,
        bytesVpByTiles[idxMaxOccupiedTileMatmul] / KBf,
        bytesVaByTiles[idxMaxOccupiedTileMatmul] / KBf,
        bytesVbByTiles[idxMaxOccupiedTileMatmul] / KBf);
    logging::debug("Maximum reduce Kb on tile {}={}: B={}, P={}, C={}, p={}",
                   idxMaxOccupiedTileReduce, maxBytesOnTileReduce / KBf,
                   bytesAByTiles[idxMaxOccupiedTileReduce] / KBf,
                   bytesBByTiles[idxMaxOccupiedTileReduce] / KBf,
                   bytesCByTiles[idxMaxOccupiedTileReduce] / KBf,
                   bytesVpByTiles[idxMaxOccupiedTileReduce] / KBf,
                   bytesCpByTiles[idxMaxOccupiedTileReduce] / KBf);
    logging::debug(
        "Minimum Kb on tile {}={}: A={}, B={}, C={}, P={}, a={}, b={}, p={}",
        idxMinOccupiedTile, minBytesOnTile / KBf,
        bytesAByTiles[idxMinOccupiedTile] / KBf,
        bytesBByTiles[idxMinOccupiedTile] / KBf,
        bytesCByTiles[idxMinOccupiedTile] / KBf,
        bytesVpByTiles[idxMinOccupiedTile] / KBf,
        bytesVaByTiles[idxMinOccupiedTile] / KBf,
        bytesVbByTiles[idxMinOccupiedTile] / KBf,
        bytesCpByTiles[idxMinOccupiedTile] / KBf);
  }
}

} // namespace experimental
} // namespace popsparse
