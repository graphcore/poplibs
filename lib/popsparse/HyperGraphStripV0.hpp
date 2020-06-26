// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphStripV0_hpp
#define popsparse_HyperGraphStripV0_hpp

#include "HyperGraph.hpp"
#include "ZoltanPartitioner.hpp"

namespace popsparse {
namespace experimental {

class HyperGraphStripV0 : public HyperGraph {

public:
  HyperGraphStripV0(const BlockMatrix &A, const BlockMatrix &B,
                    poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                    poplar::Type partialDataTypeIn, int nTileIn,
                    int nPassIn = 1);

  virtual ~HyperGraphStripV0() = default;

private:
  class Node {
  public:
    unsigned int id;
    float w;

    Node(unsigned int idIn, float wIn) : id(idIn), w(wIn) {}
  };

  class HyperEdge {
  public:
    unsigned int id;
    float w;
    std::vector<unsigned int> pins;

    HyperEdge(unsigned int idIn) : id(idIn), w(1.0f) {}
  };

  std::vector<Node> nodes;

  std::vector<HyperEdge> edges;

  unsigned int gNodeId;
  unsigned int gEdgeId;

private:
  // The dense matrix is split along one dimension, in dense x sparse = dense
  // case, it is split along rows of the dense matrix; in dense x dense = sparse
  // case, it is split along columns of the dense matrix.
  // First that dimension is split into "nPass", this is for the case that a
  // very large matmul can not fit into IPU memory
  // Second in each pass, that dimension is continuously split into "nGroup",
  // each group is processed by "nTilePerGroup" (nTilePerGroup = nTile / nGroup)
  // tiles, this design is to minimize the data exchange between tiles.
  int nGroup;
  int nTilePerGroup;
  int nPass;

  std::vector<int> rhsTileAssignment;
  std::vector<int> lhsTileAssignment;
  std::vector<int> resultTileAssignment;
  bool doRowSplit;
  bool isResultSparse;

  struct Strip {
    std::vector<int> rows;
    std::vector<std::vector<poplar::Tensor>> tensors;
  };

public:
  // Creates a graph for (sparse) matrix multiplication
  virtual void createGraphMatMul(poplar::Graph &graph,
                                 const std::string &debugPrefix) override;

  // Creates a graph for matrix multiplication and sparsifies the result
  virtual void
  createGraphMatMulSparsifyResult(poplar::Graph &graph,
                                  const unsigned char *sparsity,
                                  const std::string &debugPrefix) override;

  // Creates a program to perform matmul
  virtual void createProgramMatMul(poplar::Graph &graph,
                                   SubBlockMask subBlockMask,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix) override;

  // Adds a logic to perform matmul to existing compusets
  virtual void createProgramMatMul(poplar::Graph &graph,
                                   poplar::ComputeSet *transposeCS,
                                   poplar::ComputeSet &mulCS,
                                   poplar::ComputeSet &reduceCS,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix) override;

private:
  HyperGraphData getDataForPartitioner();

  float partitionGraph(std::vector<int> &tileAssignment, int nPartition);

  /* name convention:  DSD --> dense x sparse = dense
                       DDS --> dense x dense  = sparse
   */

  //////////////////////////////////////////////////////////////////////////////
  // Functions for dense x sparse = dense
  /////////////////////////////////////////////////////////////////////////////
  void createComputeSetRowSplitDSD(poplar::Graph &graph,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix);

  void createComputeSetRowSplitDSD(
      poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
      std::vector<poplar::ComputeSet> &mulCSVec,
      std::vector<poplar::ComputeSet> &reduceCSVec,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  void createComputeSetColSplitDSD(poplar::Graph &graph,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix);

  void createComputeSetColSplitDSD(
      poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
      std::vector<poplar::ComputeSet> &mulCSVec,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  void setLHSTileMapDSD(poplar::Graph &graph, std::vector<int> &blockTileId);

  void setRHSTileMapDSD(poplar::Graph &graph, std::vector<int> &blockTileId);

  void setResultTileMapDSD(poplar::Graph &graph);

  float partitionRowSplitDSD(std::vector<int> &tileAssignment, int nPartition);

  float partitionColSplitDSD(std::vector<int> &tileAssignment, int nPartition);

  void lhsPartitionDSD(std::vector<int> &rhsTileAssignment,
                       std::vector<int> &lhsTileAssignment, int nPartition);

  void resultPartitionDSD(std::vector<int> &resultTileAssignment,
                          int nTilePerGroup);

  //////////////////////////////////////////////////////////////////////////////
  // Functions for dense x dense = sparse
  //////////////////////////////////////////////////////////////////////////////

  void createComputeSetRowSplitDDS(poplar::Graph &graph,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix);

  void createComputeSetRowSplitDDS(
      poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
      std::vector<poplar::ComputeSet> &mulCSVec,
      std::vector<poplar::ComputeSet> &reduceCSVec,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  void createComputeSetColSplitDDS(poplar::Graph &graph,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix);

  void createComputeSetColSplitDDS(
      poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
      std::vector<poplar::ComputeSet> &mulCSVec,
      std::vector<poplar::ComputeSet> &reduceCSVec,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  typedef void (HyperGraphStripV0::*GenCs2)(
      poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
      std::vector<poplar::ComputeSet> &mulCSVec,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  // Generates program sequence with 2 computeset groups:
  // Transpose, Matmul
  void genSeq2(poplar::Graph &graph, GenCs2 genCs2,
               poplar::program::Sequence &prog, const std::string &debugPrefix);

  typedef void (HyperGraphStripV0::*GenCs3)(
      poplar::Graph &graph, std::vector<poplar::ComputeSet> &transposeCSVec,
      std::vector<poplar::ComputeSet> &mulCSVec,
      std::vector<poplar::ComputeSet> &reduceCSVec,
      poplar::program::Sequence &prog, const std::string &debugPrefix);

  // Generates program sequence with 3 computeset groups:
  // Transpose, Matmul, Reduce
  void genSeq3(poplar::Graph &graph, GenCs3 genCs3,
               poplar::program::Sequence &prog, const std::string &debugPrefix);

  void setLHSTileMapDDS(poplar::Graph &graph, std::vector<int> &blockTileId);

  void setRHSTileMapDDS(poplar::Graph &graph, std::vector<int> &blockTileId);

  void setOutputTileMapDDS(poplar::Graph &graph, std::vector<int> &blockTileId);

  float partitionRowSplitDDS(std::vector<int> &tileAssignment, int nPartition);

  float partitionColSplitDDS(std::vector<int> &tileAssignment, int nPartition);

  void lhsPartitionDDS(const std::vector<int> &rhsTileAssignment,
                       std::vector<int> &lhsTileAssignment, int nPartition);

  void rhsPartitionDDS(const std::vector<int> &lhsTileAssignment,
                       std::vector<int> &rhsTileAssignment, int nPartition);
};

} // namespace experimental
} // namespace popsparse

#endif
