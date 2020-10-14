// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphBlock_hpp
#define popsparse_HyperGraphBlock_hpp

#include "HyperGraph.hpp"
#include "ZoltanPartitioner.hpp"
#include <unordered_map>
#include <unordered_set>

// empirical constant
const int TARGET_V_NODES_PER_TILE = 2;

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
                  int nTileIn,
                  int nTargetNodesVPerTileIn = TARGET_V_NODES_PER_TILE);

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
    unsigned int blockRow;
    unsigned int blockCol;
    float w;

    DataNode(unsigned int idIn, unsigned int blockIdIn, unsigned int blockRowIn,
             unsigned int blockColIn)
        : id(idIn), blockId(blockIdIn), blockRow(blockRowIn),
          blockCol(blockColIn), w(1.0f) {}
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
    unsigned int blockRow;
    unsigned int blockCol;
    unsigned int partition;
    float w;

    ComputeNode(unsigned int idIn, std::vector<unsigned int> &idxAIn,
                std::vector<unsigned int> &idxBIn, unsigned int blockRowIn,
                unsigned int blockColIn, unsigned int partitionIn)
        : id(idIn), idxA(std::move(idxAIn)), idxB(std::move(idxBIn)),
          blockRow(blockRowIn), blockCol(blockColIn), partition(partitionIn),
          w(1.0f) {}
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
  // Node V inverted index
  std::unordered_map<unsigned int, std::size_t> nodeVIdxById;
  // Node V layout
  std::vector<std::vector<std::unordered_map<unsigned, std::size_t>>>
      nodeVLayout;

  // HyperEdges connected to nodes A
  std::vector<HyperEdge> edgeA;
  // HyperEdges connected to nodes B
  std::vector<HyperEdge> edgeB;
  // HyperEdges connected to nodes C
  std::vector<HyperEdge> edgeC;

  // Edgs A id layout by block row, block column
  std::vector<std::vector<unsigned int>> hyperEdgeIdA;
  // Edgs B id layout by block row, block column
  std::vector<std::vector<unsigned int>> hyperEdgeIdB;
  // Edgs C id layout by block row, block column
  std::vector<std::vector<unsigned int>> hyperEdgeIdC;

  // block id look up matrix for A
  std::vector<std::vector<int>> blockIdMatrixA;
  // block id look up matrix for B
  std::vector<std::vector<int>> blockIdMatrixB;

  // Global node Id counter
  unsigned int gNodeId;
  // Global edge Id counter
  unsigned int gEdgeId;

  // Number of block matrix multiplications
  std::size_t numMuls;

  // Optimization parameter
  // This is the desired number of V nodes per tile
  int nTargetNodesVPerTile;

  // Tile assignment map for all nodes sorted by their id
  std::vector<int> tileAssignment;
  // Tile assignment map for nodes C sorted by node index
  std::vector<unsigned int> nodeCTileId;

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
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  // Creates a compute set to perform a partial matmul
  void createComputeSetMatMul(
      poplar::Graph &graph, std::map<unsigned int, poplar::Tensor> &partialData,
      poplar::ComputeSet &mulCS, poplar::ComputeSet *transposeCS,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  // Creates a compute set to perform reduce
  void createComputeSetReduce(
      poplar::Graph &graph,
      const std::map<unsigned int, poplar::Tensor> &partialDataIn,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  // Creates a compute set to perform reduce
  void createComputeSetReduce(
      poplar::Graph &graph,
      const std::map<unsigned int, poplar::Tensor> &partialDataIn,
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
  // Performs graph patitioning
  virtual void partitionGraph() = 0;

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
  void populateNodesA();

  // Populates data nodes and edges for non zero blocks of matrix B
  void populateNodesB();

  // Populates data nodes and edges for non zero blocks of matrix C
  void populateNodesC(int nRowC, int nColC,
                      const std::vector<std::vector<int>> &blockIdMatrixC);

  // Populates compute nodes V
  virtual void
  populateNodesV(int nRowC, int nColC,
                 const std::vector<std::vector<int>> &blockIdMatrixC);

  // Adds node V and all corresponding connections
  void populateNodeV(
      int row, int col, unsigned int p,
      const std::vector<std::pair<unsigned int, unsigned int>> &aList,
      const std::vector<std::pair<unsigned int, unsigned int>> &bList);

  // Helper fuinction. Gets number of muls per V node
  // Used in some partitioning algorithms
  unsigned int getMulsPerVNode() const;

  virtual void setupWeights(const poplar::Graph &graph) {}

  // Map C nodes to tiles after partitioning
  virtual void mapCNodes(poplar::Graph &graph);

  // Logs tile assignment
  virtual void logTileAssignment(const std::vector<int> &tileAssignment);

  struct MemoryStatistics {
    std::size_t maxBytesOnTile = 0;
    std::size_t minBytesOnTile = 0;
    std::size_t idxMinOccupiedTile = 0;
    std::size_t maxBytesOnTileMatmul = 0;
    std::size_t idxMaxOccupiedTileMatmul = 0;
    std::size_t maxBytesOnTileReduce = 0;
    std::size_t idxMaxOccupiedTileReduce = 0;
    std::vector<std::size_t> bytesAByTiles;
    std::vector<std::size_t> bytesBByTiles;
    std::vector<std::size_t> bytesVaByTiles;
    std::vector<std::size_t> bytesVbByTiles;
    std::vector<std::size_t> bytesVpByTiles;
    std::vector<std::size_t> bytesCByTiles;
    std::vector<std::size_t> bytesCpByTiles;
    // For matmul we count A,B,C,P,a,b
    std::vector<std::size_t> bytesByTilesMatmul;
    // For reduce we count A,B,C,P,p,B
    std::vector<std::size_t> bytesByTilesReduce;
  };

  // Computes estimated data breakdowh per each tile
  void computeBytesByTile(MemoryStatistics &stats);
};

static const float KBf = 1024.0f;

} // namespace experimental
} // namespace popsparse

#endif
