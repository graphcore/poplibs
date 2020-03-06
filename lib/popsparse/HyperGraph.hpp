// Copyright (c) 2020 Graphcore Ltd, All rights reserved.

#ifndef popsparse_HyperGraph_hpp
#define popsparse_HyperGraph_hpp

#include <memory>
#include <vector>

#include "BSMatrix.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

// empirical constant
const int MUL_NODES_SPLIT_FACTOR = 2;

namespace popsparse {
namespace experimental {

struct ZoltanGraph {
  std::vector<unsigned int> pins;
  std::vector<unsigned int> hyperEdges;
  std::vector<float> weights;
  unsigned int nodes;
};

class HyperGraph {

public:
  HyperGraph(const BlockMatrix &A, const BlockMatrix &B,
             poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
             poplar::Type partialDataTypeIn, int nTileIn,
             int nMulNodesSplitFactorIn = MUL_NODES_SPLIT_FACTOR)
      : gNodeId(0), gEdgeId(0), inDataType(inDataTypeIn),
        outDataType(outDataTypeIn), partialDataType(partialDataTypeIn), matA(A),
        matB(B), nTile(nTileIn), nMulNodesSplitFactor(nMulNodesSplitFactorIn) {}

private:
  /*
  DataNode class describes an node in a Hypergraph
  that references input or output values.
  In our case the reference is blockId field.
  */
  class DataNode {
  public:
    unsigned int id;
    unsigned int blockId;
    float w;

    DataNode(unsigned int idIn, unsigned int blockIdIn)
        : id(idIn), blockId(blockIdIn), w(1.0f) {}
  };

  /*
  ComputeNode class describes an node in a Hypergraph where ops happen.
  In our case it is mul op and (maybe) several add ops.
  Fields idxA, inxB refer to block Ids.
  Blocks of idxA are pairwise multiplied with blocks of idxB.
  Multiplication results are summed as partials to be further reduced.
  */
  class ComputeNode {
  public:
    unsigned int id;
    std::vector<unsigned int> idxA; // block Ids in matrix A
    std::vector<unsigned int> idxB; // block Ids in matrix B
    float w;

    ComputeNode(unsigned int idIn, std::vector<unsigned int> &idxAIn,
                std::vector<unsigned int> &idxBIn)
        : id(idIn), idxA(std::move(idxAIn)), idxB(std::move(idxBIn)), w(1.0f) {}
  };

  /*
  HyperEdge class describes directional connections to or from a node.
  Each node can have multiple input and output connections.
  In our design we take several connections flowing to/from a compute node
  from/to data nodes of the same type (A, B or C) and put them into a single
  Edge class.
  */
  class HyperEdge {
  public:
    unsigned int id;
    float w;
    std::vector<unsigned int> in;  // Input (source) node Id
    std::vector<unsigned int> out; // Output (target) node Id

    HyperEdge(unsigned int idIn) : id(idIn), w(1.0f) {}
  };

  // These 4 nodes describe a partial expression:
  // A * B = C
  // Partial of matrix A
  // Corresponds to a tensor
  std::vector<DataNode> nodeA;
  // Partial of matrix B
  // Corresponds to a tensor
  std::vector<DataNode> nodeB;
  // Partial of matrix C
  // Corresponds to a tensor
  std::vector<DataNode> nodeC;
  // Partial of A * B
  // Corresponds to a vertex
  std::vector<ComputeNode> nodeV;

  std::vector<HyperEdge> edgeA;
  std::vector<HyperEdge> edgeB;
  std::vector<HyperEdge> edgeC;

  unsigned int gNodeId;
  unsigned int gEdgeId;

  poplar::Type inDataType;
  poplar::Type outDataType;
  poplar::Type partialDataType;

public:
  const BlockMatrix &matA;
  const BlockMatrix &matB;
  std::unique_ptr<BlockMatrix> matC;

private:
  int nTile;
  int nMulNodesSplitFactor;

public:
  const std::vector<DataNode> &getNodeA() const { return nodeA; }
  const std::vector<DataNode> &getNodeB() const { return nodeB; }
  const std::vector<DataNode> &getNodeC() const { return nodeC; }
  const std::vector<ComputeNode> &getNodeV() const { return nodeV; }

  const std::vector<HyperEdge> &getEdgeA() const { return edgeA; }
  const std::vector<HyperEdge> &getEdgeB() const { return edgeB; }
  const std::vector<HyperEdge> &getEdgeC() const { return edgeC; }

  // Creates a graph for (sparse) matrix multiplication
  void createGraphMatMul(float memoryCycleRatio, poplar::Graph &graph,
                         const std::string &debugPrefix);

  // Creates a graph for matrix multiplication and sparsifies the result
  void createGraphMatMulSparsifyResult(const unsigned char *sparsity,
                                       float memoryCycleRatio,
                                       poplar::Graph &graph,
                                       const std::string &debugPrefix);

  // Creates a program to perform matmul
  void createProgramMatMul(std::vector<int> &tileAssignment,
                           poplar::Graph &graph,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix);

  // Creates a compute set to perform a partial matmul
  void
  createComputeSetMatMul(std::vector<int> &tileAssignment,
                         std::map<unsigned int, poplar::Tensor> &partialData,
                         std::vector<unsigned int> &nodeCTileId,
                         poplar::Graph &graph, const std::string &debugPrefix,
                         poplar::program::Sequence &prog);

  // Creates a compute set to perform reduce
  void createComputeSetReduce(
      const std::map<unsigned int, poplar::Tensor> &partialDataIn,
      const std::vector<unsigned int> &nodeCTileId, poplar::Graph &graph,
      const std::string &debugPrefix, poplar::program::Sequence &prog);

  void partitionGraph(std::vector<int> &tileAssignment);

  unsigned int getTotalNodes() const { return gNodeId; }

  // Adds necessary codelets
  // This method must be called once per graph
  static void addCodelets(poplar::Graph &graph);

private:
  // Creates a graph for dense * sparse = dense matrix multiplication
  void createGraphMatMulDSD(float memoryCycleRatio, poplar::Graph &graph,
                            const std::string &debugPrefix);

  // Creates a graph for dense * dense -> sparse matrix multiplication
  void createGraphMatMulDDSSparsiryResult(const unsigned char *sparsity,
                                          float memoryCycleRatio,
                                          poplar::Graph &graph,
                                          const std::string &debugPrefix);

  // Helper methods
  // Populates nodes and edges for a node group A, B or C
  std::vector<std::vector<unsigned int>> populateDataNodes(
      int nRow, int nCol, const std::vector<std::vector<int>> &blockIdMatrix,
      std::vector<DataNode> &nodeGroup, std::vector<HyperEdge> &edgeGroup);

  // Populates data nodes and edges for non zero blocks of matrix A
  std::vector<std::vector<unsigned int>>
  populateNodesA(int nRowA, int nColA,
                 const std::vector<std::vector<int>> &blockIdMatrixA);

  // Populates data nodes and edges for non zero blocks of matrix B
  std::vector<std::vector<unsigned int>>
  populateNodesB(int nRowB, int nColB,
                 const std::vector<std::vector<int>> &blockIdMatrixB);

  // Populates data nodes and edges for non zero blocks of matrix C
  std::vector<std::vector<unsigned int>>
  populateNodesC(int nRowC, int nColC,
                 const std::vector<std::vector<int>> &blockIdMatrixC);

  // Populates compute nodes V
  void
  populateNodesV(int nRowC, int nColC, int nColA, unsigned int nAvgV,
                 const std::vector<std::vector<int>> &blockIdMatrixA,
                 const std::vector<std::vector<int>> &blockIdMatrixB,
                 const std::vector<std::vector<int>> &blockIdMatrixC,
                 const std::vector<std::vector<unsigned int>> &hyperEdgeIdA,
                 const std::vector<std::vector<unsigned int>> &hyperEdgeIdB,
                 const std::vector<std::vector<unsigned int>> &hyperEdgeIdC);

  // Adds node V and all corresponding connections
  void
  populateNodeV(int row, int col,
                const std::vector<std::pair<unsigned int, unsigned int>> &aList,
                const std::vector<std::pair<unsigned int, unsigned int>> &bList,
                const std::vector<std::vector<int>> &blockIdMatrixA,
                const std::vector<std::vector<int>> &blockIdMatrixB,
                const std::vector<std::vector<unsigned int>> &hyperEdgeIdA,
                const std::vector<std::vector<unsigned int>> &hyperEdgeIdB,
                const std::vector<std::vector<unsigned int>> &hyperEdgeIdC);

  // Set up weights for a graph
  void setupWeights(const poplar::Graph &graph, float memoryCycleRatio,
                    int numMuls);

  // Represents graph in a zoltan format
  ZoltanGraph getDataForZoltan();
};

} // namespace experimental
} // namespace popsparse

#endif
