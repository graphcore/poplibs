// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphBlock_hpp
#define popsparse_HyperGraphBlock_hpp

#include "HyperGraph.hpp"
#include "ZoltanPartitioner.hpp"

// empirical constant
const int MUL_ON_NODE_V = 2;

namespace popsparse {
namespace experimental {

/*
This class is a base class for block-based matrix multiplication
partitioning algoriothms.
Derived classes uses different partitioning algorithms.
*/
class HyperGraphBlock : public HyperGraph {

public:
  HyperGraphBlock(BlockMatrix &A, BlockMatrix &B, poplar::Type inDataTypeIn,
                  poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn,
                  int nTileIn, float memoryCycleRatioIn,
                  int nMulsOnVNodeIn = MUL_ON_NODE_V);

  virtual ~HyperGraphBlock() = default;

protected:
  /*
  DataNode class describes an node in a Hypergraph
  that references input or output values.
  In our case the reference is blockId field.
  One DataNode corresponds to one input/output block tensor.
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
  In our case it is muliply and add operation that computes partial result of
  matrix A block-row by matrix B block-column multiplicatoin.
  Fields idxA, inxB refer to block Ids of matrices A and B accordingly.
  Blocks of idxA are pairwise multiplied with blocks of idxB.
  Multiplication results are summed as partials to be further reduced.
  One ComputeNode corresponds to one 1x1 convolution vertex.
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
  // Corresponds to a block tensor
  std::vector<DataNode> nodeA;
  // Partial of matrix B
  // Corresponds to a block tensor
  std::vector<DataNode> nodeB;
  // Partial of matrix C
  // Corresponds to a block tensor
  std::vector<DataNode> nodeC;
  // Partial of A * B
  // Corresponds to a vertex
  std::vector<ComputeNode> nodeV;

  // HyperEdges connected to nodes A
  std::vector<HyperEdge> edgeA;
  // HyperEdges connected to nodes B
  std::vector<HyperEdge> edgeB;
  // HyperEdges connected to nodes C
  std::vector<HyperEdge> edgeC;

  // Global node Id counter
  unsigned int gNodeId;
  // Global edge Id counter
  unsigned int gEdgeId;

  // Used in derived classes that use Zoltan partitioner to tune partitioning
  float memoryCycleRatio;
  // Optimization parameter
  // This is the desired average number of mul operations on a compute node
  int nMulsOnVNode;

  // Tile assignment map for all nodes sorted by their id
  std::vector<int> tileAssignment;

public:
  const std::vector<DataNode> &getNodeA() const { return nodeA; }
  const std::vector<DataNode> &getNodeB() const { return nodeB; }
  const std::vector<DataNode> &getNodeC() const { return nodeC; }
  const std::vector<ComputeNode> &getNodeV() const { return nodeV; }

  const std::vector<HyperEdge> &getEdgeA() const { return edgeA; }
  const std::vector<HyperEdge> &getEdgeB() const { return edgeB; }
  const std::vector<HyperEdge> &getEdgeC() const { return edgeC; }

  // Creates a graph for (sparse) matrix multiplication
  void createGraphMatMul(poplar::Graph &graph,
                         const std::string &debugPrefix) override;

  // Creates a graph for matrix multiplication and sparsifies the result
  void createGraphMatMulSparsifyResult(poplar::Graph &graph,
                                       const unsigned char *sparsity,
                                       const std::string &debugPrefix) override;

  // Creates a program to perform matmul
  void createProgramMatMul(poplar::Graph &graph, SubBlockMask subBlockMask,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix) override;

  // Adds a logic to perform matmul to existing compusets
  void createProgramMatMul(poplar::Graph &graph,
                           poplar::ComputeSet *transposeCS,
                           poplar::ComputeSet &mulCS,
                           poplar::ComputeSet &reduceCS,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix) override;
  // Creates a compute set to perform a partial matmul
  void createComputeSetMatMul(
      poplar::Graph &graph, std::map<unsigned int, poplar::Tensor> &partialData,
      std::vector<unsigned int> &nodeCTileId, poplar::program::Sequence &prog,
      const std::string &debugPrefix);

  // Creates a compute set to perform a partial matmul
  void createComputeSetMatMul(
      poplar::Graph &graph, std::map<unsigned int, poplar::Tensor> &partialData,
      std::vector<unsigned int> &nodeCTileId, poplar::ComputeSet &mulCS,
      poplar::ComputeSet *transposeCS, poplar::program::Sequence &prog,
      const std::string &debugPrefix);

  // Creates a compute set to perform reduce
  void createComputeSetReduce(
      poplar::Graph &graph,
      const std::map<unsigned int, poplar::Tensor> &partialDataIn,
      const std::vector<unsigned int> &nodeCTileId,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  // Creates a compute set to perform reduce
  void createComputeSetReduce(
      poplar::Graph &graph,
      const std::map<unsigned int, poplar::Tensor> &partialDataIn,
      const std::vector<unsigned int> &nodeCTileId,
      poplar::ComputeSet &reduceCS, const std::string &debugPrefix);

  // Set the tile mapping for left hand matrix
  void setTileMappingLHS(poplar::Graph &graph,
                         poplar::Tensor &lhsTensor) override;

  // Set the tile mapping for right hand matrix
  void setTileMappingRHS(poplar::Graph &graph,
                         poplar::Tensor &rhsTensor) override;

  unsigned int getTotalNodes() const { return gNodeId; }

  void setTileAssignment(std::vector<int> &tileAssignmentIn) {
    tileAssignment = tileAssignmentIn;
  }

  // Adds necessary codelets
  // This method must be called once per graph
  static void addCodelets(poplar::Graph &graph);

protected:
  virtual void partitionGraph() = 0;

private:
  // Creates a graph for dense * sparse = dense matrix multiplication
  void createGraphMatMulDSD(poplar::Graph &graph,
                            const std::string &debugPrefix);

  // Creates a graph for dense * dense -> sparse matrix multiplication
  void createGraphMatMulDDSSparsiryResult(poplar::Graph &graph,
                                          const unsigned char *sparsity,
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
  void setupWeights(const poplar::Graph &graph, int numMuls);

  // Logs tile assignment
  virtual void logTileAssignment(const poplar::Graph &graph,
                                 const std::vector<int> &tileAssignment,
                                 const std::vector<unsigned int> &nodeCTileId);
};

} // namespace experimental
} // namespace popsparse

#endif
