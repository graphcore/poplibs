// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "HyperGraph.hpp"
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
#include <random>
#include <zoltan_cpp.h>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

void HyperGraph::createGraphMatMul(float memoryCycleRatio, poplar::Graph &graph,
                                   const std::string &debugPrefix) {
  if (matA.isDense()) {
    createGraphMatMulDSD(memoryCycleRatio, graph, debugPrefix);
  } else {
    throw poputil::poplibs_error("Not implemented");
  }
}

void HyperGraph::createGraphMatMulSparsifyResult(
    const unsigned char *sparsity, float memoryCycleRatio, poplar::Graph &graph,
    const std::string &debugPrefix) {
  if (matA.isDense() && matB.isDense()) {
    createGraphMatMulDDSSparsiryResult(sparsity, memoryCycleRatio, graph,
                                       debugPrefix);
  } else {
    throw poputil::poplibs_error("Not implemented");
  }
}

void HyperGraph::createGraphMatMulDSD(float memoryCycleRatio,
                                      poplar::Graph &graph,
                                      const std::string &debugPrefix) {
  assert(!matC);
  assert(matA.getColCount() == matB.getRowCount());
  assert(matA.getBlockCol() == matB.getBlockRow());

  // block id look up matrix for A
  auto blockIdMatrixA = matA.getBlockIdMatrix();

  const int nRowA = matA.getRowCount() / matA.getBlockRow();
  const int nColA = matA.getColCount() / matA.getBlockCol();

  // block id look up matrix for B
  auto blockIdMatrixB = matB.getBlockIdMatrix();

  const int nRowB = matB.getRowCount() / matB.getBlockRow();
  const int nColB = matB.getColCount() / matB.getBlockCol();

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

  poplar::Tensor matCTensor = graph.addVariable(
      outDataType,
      {static_cast<unsigned long>(rowC), static_cast<unsigned long>(colC)},
      debugPrefix + "/matC");
  matCDense->setBlockTensor(matCTensor);

  auto hyperEdgeIdC = populateNodesC(nRowC, nColC, blockIdMatrixC);

  // nArgV is the desired average number of muls per node V (vertex)
  // We use this number to group the block pairs that contribute to the same
  // block in the matrix C. If this number is too small, we need more time to
  // sum up the partials. If this number is too big, we may have load balance
  // issue.
  unsigned int nAvgV = numMuls / nTile / nMulNodesSplitFactor;

  logging::info("Total number of muls: {}", numMuls);
  logging::info("Average number of muls per node V: {}", nAvgV);

  // populate multiply nodes V
  populateNodesV(nRowC, nColC, nColA, nAvgV, blockIdMatrixA, blockIdMatrixB,
                 blockIdMatrixC, hyperEdgeIdA, hyperEdgeIdB, hyperEdgeIdC);

  // save the pointer to matrix C
  matC = std::move(matCDense);
  setupWeights(graph, memoryCycleRatio, numMuls);
}

void HyperGraph::createGraphMatMulDDSSparsiryResult(
    const unsigned char *sparsity, float memoryCycleRatio, poplar::Graph &graph,
    const std::string &debugPrefix) {
  assert(!matC);
  assert(matA.getColCount() == matB.getRowCount());
  assert(matA.getBlockCol() == matB.getBlockRow());

  // block id look up matrix for A
  auto blockIdMatrixA = matA.getBlockIdMatrix();

  const int nRowA = matA.getRowCount() / matA.getBlockRow();
  const int nColA = matA.getColCount() / matA.getBlockCol();

  // block id look up matrix for B
  auto blockIdMatrixB = matB.getBlockIdMatrix();

  const int nRowB = matB.getRowCount() / matB.getBlockRow();
  const int nColB = matB.getColCount() / matB.getBlockCol();

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

  unsigned int nAvgV = numMuls / nTile / nMulNodesSplitFactor;

  logging::info("Total number of muls: {}", numMuls);
  logging::info("Average number of muls per node V: {}", nAvgV);

  // populate multiply nodes V
  populateNodesV(nRowC, nColC, nColA, nAvgV, blockIdMatrixA, blockIdMatrixB,
                 blockIdMatrixC, hyperEdgeIdA, hyperEdgeIdB, hyperEdgeIdC);

  // save the pointer to matrix C
  matC = std::move(matCSparse);
  setupWeights(graph, memoryCycleRatio, numMuls);
}

void HyperGraph::setupWeights(const poplar::Graph &graph,
                              float memoryCycleRatio, int numMuls) {
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

std::vector<std::vector<unsigned int>> HyperGraph::populateDataNodes(
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

std::vector<std::vector<unsigned int>> HyperGraph::populateNodesA(
    int nRowA, int nColA, const std::vector<std::vector<int>> &blockIdMatrixA) {
  return populateDataNodes(nRowA, nColA, blockIdMatrixA, nodeA, edgeA);
}

std::vector<std::vector<unsigned int>> HyperGraph::populateNodesB(
    int nRowB, int nColB, const std::vector<std::vector<int>> &blockIdMatrixB) {
  return populateDataNodes(nRowB, nColB, blockIdMatrixB, nodeB, edgeB);
}

std::vector<std::vector<unsigned int>> HyperGraph::populateNodesC(
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

void HyperGraph::populateNodesV(
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

void HyperGraph::populateNodeV(
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

ZoltanGraph HyperGraph::getDataForZoltan() {
  logging::info("Number of nodes in A: {}", nodeA.size());
  logging::info("Number of nodes in B: {}", nodeB.size());
  logging::info("Number of nodes in V: {}", nodeV.size());

  ZoltanGraph graph;

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

  logging::info("Number of pins is {}", pins.size());
  logging::info("Number of edges is {}", hyperEdges.size());

  graph.pins = std::move(pins);
  graph.hyperEdges = std::move(hyperEdges);
  graph.weights = std::move(weights);

  return graph;
}

static int objectNumber(void *data, int *ierr) {
  // Returns the number of vertices in the hypergraph.

  auto &graph = *static_cast<ZoltanGraph *>(data);
  *ierr = ZOLTAN_OK;
  return static_cast<int>(graph.nodes);
}

static void objectList(void *data, int globalIDEntries, int localIDEntries,
                       ZOLTAN_ID_PTR globalIDs, ZOLTAN_ID_PTR localIDs,
                       int weightDimension, float *objectWeights, int *ierr) {
  // Fills in the global IDs and weights for the vertices in the hypergraph.

  auto &graph = *static_cast<ZoltanGraph *>(data);
  *ierr = ZOLTAN_OK;

  for (unsigned i = 0; i < graph.nodes; ++i) {
    globalIDs[i] = i;
  }

  if (weightDimension > 0) {
    for (unsigned i = 0; i < graph.weights.size(); ++i) {
      objectWeights[i] = graph.weights[i];
    }
  }
}

static void hypergraphSize(void *data, int *hyperGraphSize, int *pinCount,
                           int *format, int *ierr) {
  // Fills in the number of hyperedges and the number of vertices in those
  // edges.

  auto &graph = *static_cast<ZoltanGraph *>(data);
  *ierr = ZOLTAN_OK;

  *hyperGraphSize = static_cast<int>(graph.hyperEdges.size());
  *pinCount = static_cast<int>(graph.pins.size());
  *format = ZOLTAN_COMPRESSED_EDGE;
}

static void hypergraphList(void *data, int globalIDEntries, int hyperGraphSize,
                           int pinCount, int format, ZOLTAN_ID_PTR hyperEdgeGID,
                           int *hyperEdgeOffset, ZOLTAN_ID_PTR pinGID,
                           int *ierr) {
  // Fills in the the global IDs of the hyperedges and the global IDs of the
  // vertices in each edge.

  auto &graph = *static_cast<ZoltanGraph *>(data);
  *ierr = ZOLTAN_OK;

  for (int i = 0; i < hyperGraphSize; ++i) {
    hyperEdgeGID[i] = i;
    hyperEdgeOffset[i] = static_cast<int>(graph.hyperEdges[i]);
  }

  for (int i = 0; i < pinCount; ++i) {
    pinGID[i] = graph.pins[i];
  }
}

void HyperGraph::partitionGraph(std::vector<int> &tileAssignment) {
  float zoltanVersion;
  ZoltanGraph zoltanGraph = getDataForZoltan();
  char *debugLevel = getenv("POPLIBS_LOG_LEVEL");

  if (Zoltan_Initialize(0, nullptr, &zoltanVersion) != ZOLTAN_OK) {
    throw poputil::poplibs_error("Partitioning of the graph failed");
  }

  std::unique_ptr<Zoltan> zz(new Zoltan);

  logging::info("Zoltan version: {}", zoltanVersion);

  // Register query functions.
  zz->Set_Num_Obj_Fn(objectNumber, &zoltanGraph);
  zz->Set_Obj_List_Fn(objectList, &zoltanGraph);
  zz->Set_HG_Size_CS_Fn(hypergraphSize, &zoltanGraph);
  zz->Set_HG_CS_Fn(hypergraphList, &zoltanGraph);

  // Set parameters
  // Set debug level to same level as POPLIBS_LOG_LEVEL.
  zz->Set_Param("DEBUG_LEVEL", debugLevel != NULL ? debugLevel : "0");

  // We want to use hypergraphs to model our connections.
  zz->Set_Param("LB_METHOD", "HYPERGRAPH");

  // PHG is the Zoltan hypergraph partitioning package.
  zz->Set_Param("HYPERGRAPH_PACKAGE", "PHG");

  // We have one and only one list of global ID entries.
  zz->Set_Param("NUM_GID_ENTRIES", "1");

  // We do not have any local ID entries.
  zz->Set_Param("NUM_LID_ENTRIES", "0");

  // We want to partition the hypergraph.
  zz->Set_Param("LB_APPROACH", "PARTITION");

  // We want to partition it into a predermined number of subgraphs.
  zz->Set_Param("NUM_GLOBAL_PARTS", std::to_string(nTile));

  // Return the parts into which the vertices are partitioned.
  zz->Set_Param("RETURN_LISTS", "PARTS");

  // We have a one-dimensional list of object weights.
  zz->Set_Param("OBJ_WEIGHT_DIM", "1");

  // We do not have any edge weights.
  zz->Set_Param("EDGE_WEIGHT_DIM", "0");

  // TODO:
  // Would it be beneficial to set edge weights,
  // in particular if the partial type is larger than the input type
  // then edge that connects the output to the calculation should have a larger
  // weight than the edges that connect the inputs to the calculation?

  // Perform the partitioning.

  struct PartitionData {
    int changes = 0;
    int globalIDEntries = 1;
    int localIDEntries = 0;
    int imports = 1;
    ZOLTAN_ID_PTR importGlobalIDs = nullptr;
    ZOLTAN_ID_PTR importLocalIDs = nullptr;
    int *importProcs = nullptr;
    int *importToPart = nullptr;
    int exports = 1;
    ZOLTAN_ID_PTR exportGlobalIDs = nullptr;
    ZOLTAN_ID_PTR exportLocalIDs = nullptr;
    int *exportProcs = nullptr;
    int *exportToPart = nullptr;

    ~PartitionData() {
      Zoltan::LB_Free_Part(&importGlobalIDs, &importLocalIDs, &importProcs,
                           &importToPart);
      Zoltan::LB_Free_Part(&exportGlobalIDs, &exportLocalIDs, &exportProcs,
                           &exportToPart);
    }
  } data;

  auto result = zz->LB_Partition(
      data.changes, data.globalIDEntries, data.localIDEntries, data.imports,
      data.importGlobalIDs, data.importLocalIDs, data.importProcs,
      data.importToPart, data.exports, data.exportGlobalIDs,
      data.exportLocalIDs, data.exportProcs, data.exportToPart);

  switch (result) {
  case ZOLTAN_OK:
    break;
  case ZOLTAN_WARN:
    logging::warn("Hypergraph partitioning returned with warnings");
    break;
  case ZOLTAN_FATAL:
    throw poputil::poplibs_error("Partitioning of the hypergraph failed");
  case ZOLTAN_MEMERR:
    throw poputil::poplibs_error(
        "Memory allocation failure in hypergraph partitioning");
  }

  // Translate the partition back into a tile mapping.
  tileAssignment.resize(data.exports);

  for (int i = 0; i < data.exports; ++i) {
    tileAssignment[data.exportGlobalIDs[i]] = data.exportToPart[i];
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

void HyperGraph::addCodelets(poplar::Graph &graph) {
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
}

void HyperGraph::createProgramMatMul(std::vector<int> &tileAssignment,
                                     poplar::Graph &graph,
                                     poplar::program::Sequence &prog,
                                     const std::string &debugPrefix) {
  std::map<unsigned int, poplar::Tensor> partialData;
  std::vector<unsigned int> nodeCTileId;
  createComputeSetMatMul(tileAssignment, partialData, nodeCTileId, graph,
                         debugPrefix, prog);

  createComputeSetReduce(partialData, nodeCTileId, graph, debugPrefix, prog);
}

void HyperGraph::createComputeSetMatMul(
    std::vector<int> &tileAssignment,
    std::map<unsigned int, poplar::Tensor> &partialData,
    std::vector<unsigned int> &nodeCTileId, poplar::Graph &graph,
    const std::string &debugPrefix, poplar::program::Sequence &prog) {

#if DEBUG_INFO == 5
  std::vector<poplar::Tensor> weightsReport;
  std::vector<poplar::Tensor> inReport;
  std::vector<poplar::Tensor> outReport;
#endif
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

  poplar::ComputeSet transposeCS =
      graph.addComputeSet(debugPrefix + "transposeCS");

  const std::vector<poplar::Tensor> &blockDataB = matB.getBlockTensor();
  std::vector<poplar::Tensor> transposedBlocks;
  transposedBlocks.resize(blockDataB.size());
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

    // TODO:
    // [richard]
    // This creates one transpose vertex per block which is inefficient both in
    // terms of memory and cycles
    // - it would be better to group the blocks that need to be transposed by
    // tile and split the work between just enough vertices to keep all workers
    // busy. This logic already exists in addTransposeVertices() in ConvUtil.cpp
    // - would it be possible to use this function instead of adding the
    // vertices here?
    if (!matB.getNeedTranspose()) {
      transposedBlocks[blockId] = graph.addVariable(
          inDataType,
          {static_cast<unsigned long>(matB.getBlockRow() * matB.getBlockCol())},
          debugPrefix1 + std::to_string(nodeId));
      // May need a new vertex to do in memory transpose to avoid memory copy
      std::vector<poplar::Tensor> src, dst;
      src.push_back(blockDataB[blockId]);
      dst.push_back(transposedBlocks[blockId]);
      auto v = graph.addVertex(
          transposeCS,
          poputil::templateVertex("popops::Transpose2d", inDataType));

      graph.connect(v["src"], src);
      graph.connect(v["dst"], dst);
      graph.setInitialValue(v["numSrcRows"], matB.getBlockRow());
      graph.setInitialValue(v["numSrcColumns"], matB.getBlockCol());
      graph.setTileMapping(v, tileId);
      graph.setTileMapping(transposedBlocks[blockId], tileId);
    } else {
      // If matrix B should be transposed, the data layout of the
      // block is exactly the layout ConvPartial1x1Out vertex want, so no
      // transpose is needed.
      transposedBlocks[blockId] = blockDataB[blockId];
    }
  }

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

  // TODO
  // Move this logic out to partitioner
  // Also try to let zoltan do the partitioning and compare the results
  std::mt19937 randomEngine;

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

    // TODO:
    // [richard]
    // This creates a new compute set for every block that needs to be zeroed
    // which is going to be inefficient. Could you concat all blocks that need
    // to be zeroed and call popops::zero once with this tensor (if it is non
    // empty). Also it would be better to zero the data after the multiplication
    // in the same compute set as the reduce vertices for a couple of reasons:
    // Writing them after means the tensor isn't live while the compute is
    // happening which gives the variable allocator more freedom to save memory
    // by overlapping variables. Poplar only track the liveness of tensors as a
    // whole and it is conservative if a compute set partially writes a tensor.
    // If you use two compute sets to write them and don't insert a WriteUndef
    // program then the tensor will be treated as always live.

    if (max == -1) {
      // This node does not have any input, so it is zero block
      // put it on a random tile
      tileId = randomEngine() % nTile;
    }

    assert(tileId >= 0);
    unsigned int blockId = nodeC[i].blockId;
    graph.setTileMapping(blockDataC[blockId], tileId);
    nodeCTileId[i] = tileId;

    if (max == -1) {
      popops::zero(graph, blockDataC[blockId], prog,
                   debugPrefix + "/zero_block");
    }
  }

  poplar::ComputeSet mulCS = graph.addComputeSet(debugPrefix + "mulCS");

  // add vertex for block matmul
  int nWorker = graph.getTarget().getNumWorkerContexts();
  assert(nWorker == 6);

  vNodeCount = 0;
  const unsigned convInChannels = (inDataType == poplar::FLOAT ? 8 : 16);
  assert(convInChannels == static_cast<unsigned int>(matB.getBlockRow()));

  // Here we use term "batchSize" more general just for the lack of better term
  // Number of rows in matrix A == batchSize if we use matmut to multiply input
  // by weights for forward pass
  int batchSize = matA.getBlockRow();

  int workerSize = batchSize / nWorker;
  int leftover = batchSize % nWorker;
  std::array<int, 18> worklist;
  int offset = 0;
  // Distrubute batchSize between workers as even as possible
  for (int i = 0; i < 6; ++i) {
    worklist[i * 3] = offset;
    worklist[i * 3 + 2] = offset;
    int curWorkerSize = (i < leftover ? workerSize + 1 : workerSize);
    // The ConvPartial1x1Out vertex expects worklists[3i+1] to be the number of
    // elements minus 3.
    worklist[i * 3 + 1] = curWorkerSize - 3;
    offset += curWorkerSize;
  }

  auto t = graph.addConstant(poplar::UNSIGNED_SHORT,
                             {static_cast<unsigned long>(nWorker * 3)},
                             worklist.data(), debugPrefix + "/worklists");
  graph.setTileMapping(t, 0);

  for (const auto &n : nodeV) {
    unsigned int nodeId = n.id;
    std::vector<poplar::Tensor> inputA, inputB;
    for (unsigned int i = 0; i < n.idxA.size(); i++) {
      poplar::Tensor inputAShaped = blockDataA[n.idxA[i]];
      poplar::Tensor inputBShaped = transposedBlocks[n.idxB[i]];
      inputA.push_back(inputAShaped);
      inputB.push_back(inputBShaped);
#if DEBUG_INFO == 5
      weightsReport.push_back(inputBShaped);
      inReport.push_back(inputAShaped);
#endif
    }

    poplar::VertexRef v;

    // use128bit load seems crashed the application
    // TODO: Try to reproduce and file a bug if confirmed
    v = graph.addVertex(
        mulCS, poputil::templateVertex("poplin::ConvPartial1x1Out", inDataType,
                                       partialDataType, "true", "true", 8));
    std::vector<poplar::Tensor> out;
    out.push_back(partialData[nodeId]);
#if DEBUG_INFO == 5
    outReport.push_back(partialData[nodeId]);
#endif
    graph.connect(v["in"], inputA);
    graph.connect(v["out"], out);
    graph.connect(v["weights"], inputB);
    // TODO: test for all supported shapes.
    // Figure out parameters for the best performance.
    graph.setInitialValue(v["outChansPerGroup"], matB.getBlockCol());
    graph.setInitialValue(v["inChansPerGroup"], convInChannels);
    graph.setInitialValue(v["numOutGroupsM1"], 0);
    graph.setInitialValue(v["numInGroups"], inputA.size());
    graph.setInitialValue(v["transformedInStride"], 1);
    graph.setInitialValue(v["numConvGroupsM1"], 0);
    if (partialDataType == poplar::FLOAT) {
      graph.setInitialValue(v["transformedOutStride"], matB.getBlockCol() - 6);
    } else {
      graph.setInitialValue(v["transformedOutStride"], matB.getBlockCol() - 4);
    }

    graph.connect(v["worklists"], t);

    if (tileAssignment[nodeV[vNodeCount].id] < 0) {
      throw poputil::poplibs_error(
          "Invalid tile id: " +
          std::to_string(tileAssignment[nodeV[vNodeCount].id]) + "For nodeV");
    }
    unsigned int tileId =
        static_cast<unsigned int>(tileAssignment[nodeV[vNodeCount].id]);
    graph.setTileMapping(v, tileId);

    vNodeCount++;
  }

  if (!matB.getNeedTranspose()) {
    prog.add(poplar::program::Execute(transposeCS));
  }
  prog.add(poplar::program::Execute(mulCS));
#if DEBUG_INFO == 5
  for (unsigned i = 0; i < weightsReport.size(); ++i) {
    prog.add(poplar::program::PrintTensor(
        std::string("weights_") + std::to_string(i), weightsReport[i]));
  }
  for (unsigned i = 0; i < inReport.size(); ++i) {
    prog.add(poplar::program::PrintTensor(
        std::string("in_") + std::to_string(i), inReport[i]));
  }
  for (unsigned i = 0; i < outReport.size(); ++i) {
    prog.add(poplar::program::PrintTensor(
        std::string("out_") + std::to_string(i), outReport[i]));
  }
#endif
}

void HyperGraph::createComputeSetReduce(
    const std::map<unsigned int, poplar::Tensor> &partialDataIn,
    const std::vector<unsigned int> &nodeCTileId, poplar::Graph &graph,
    const std::string &debugPrefix, poplar::program::Sequence &prog) {
  unsigned int minC = INT_MAX, maxC = 0;

  int nWorker = graph.getTarget().getNumWorkerContexts();
  assert(nWorker == 6);

  poplar::ComputeSet reduceCS = graph.addComputeSet(debugPrefix + "reduceCS");
  // how many partials we can fit in 128-bits: 4 is for FLOAT, 8 for HALF
  const unsigned numValsIn128 = outDataType == poplar::FLOAT ? 4 : 8;
  int blockSize = matC->getBlockRow() * matC->getBlockCol();

  if (blockSize % numValsIn128 != 0) {
    throw poputil::poplibs_error(
        "Error: the size of block in matrix should be divisible by " +
        std::to_string(numValsIn128));
  }

  int grains = blockSize / numValsIn128;
  int grainsPerWorker = grains / nWorker;
  int grainsLeftover = grains % nWorker;
  for (unsigned int i = 0; i < edgeC.size(); i++) {
    HyperEdge &e = edgeC[i];
    if (e.in.empty()) {
      continue;
    }

    minC = std::min(static_cast<uint>(e.in.size()), minC);
    maxC = std::max(static_cast<uint>(e.in.size()), maxC);

    poplar::Tensor output = matC->getBlockTensor()[nodeC[i].blockId];
    std::vector<poplar::Tensor> inputBlocks;
    for (const auto &in : e.in) {
      auto iter = partialDataIn.find(in);
      if (iter == partialDataIn.end()) {
        throw poputil::poplibs_error("Error: incomplete input partial data");
        return;
      }
      inputBlocks.push_back(iter->second);
    }

    for (int w = 0, offset = 0; w < nWorker; w++) {
      int curGrains =
          (w < grainsLeftover) ? (grainsPerWorker + 1) : grainsPerWorker;
      int curElements = curGrains * numValsIn128;
      std::vector<poplar::Tensor> inputOneWorker;
      for (const auto &b : inputBlocks) {
        inputOneWorker.push_back(b.slice(offset, offset + curElements));
      }
      auto v = graph.addVertex(
          reduceCS, poputil::templateVertex(
                        "popops::ReducePartialsEqualSize", "popops::ReduceAdd",
                        partialDataType, outDataType, "false"));
      graph.connect(v["out"], output.slice(offset, offset + curElements));
      graph.setInitialValue(v["outCount"], curGrains);
      graph.connect(v["partials"], inputOneWorker);
      graph.setInitialValue(v["partialsSizeM1"], 0);
      graph.setTileMapping(v, nodeCTileId[i]);

      offset += curElements;
    }
  }

  logging::info("Total reduction node = {}, min = {}, max = {} ", edgeC.size(),
                minC, maxC);

  prog.add(poplar::program::Execute(reduceCS));
}

} // namespace experimental
} // namespace popsparse
