// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraph_hpp
#define popsparse_HyperGraph_hpp

#include <random>
#include <unordered_map>

#include "BSMatrix.hpp"
#include "HyperGraphPartitioner.hpp"
#include "popsparse/experimental/BlockSparse.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <random>

namespace popsparse {
namespace experimental {

/*
This is a base class for block-sparse matrix multiplication engine.
*/
class HyperGraph {
public:
  HyperGraph(const BlockMatrix &A, const BlockMatrix &B,
             poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
             poplar::Type partialDataTypeIn, int nTileIn)
      : matA(A), matB(B), inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn), nTile(nTileIn) {}

  virtual ~HyperGraph() = default;

  // Creates a graph for dense x sparse matrix multiplication
  virtual void createGraphMatMul(poplar::Graph &graph,
                                 const std::string &debugPrefix) = 0;

  // Creates a graph for dense x dense matrix multiplication and sparsifies the
  // result
  virtual void
  createGraphMatMulSparsifyResult(poplar::Graph &graph,
                                  const unsigned char *sparsity,
                                  const std::string &debugPrefix) = 0;
  // Creates a program to perform matmul
  virtual void createProgramMatMul(poplar::Graph &graph,
                                   SubBlockMask subBlockMask,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix) = 0;

  // Adds a logic to perform matmul to existing compute sets
  virtual void createProgramMatMul(poplar::Graph &graph,
                                   poplar::ComputeSet *transposeCS,
                                   poplar::ComputeSet &mulCS,
                                   poplar::ComputeSet &reduceCS,
                                   poplar::program::Sequence &prog,
                                   const std::string &debugPrefix) = 0;

  // Gets output matmul tensor result
  poplar::Tensor getResultTensor() const;

  // Gets output matmul tensor result block size
  void getResultBlockSize(int &blockRow, int &blockCol) const;

  // Gets output matmul tensor result size in blocks
  void getResultBlockCount(int &blockRowCount, int &blockColCount) const;

  // Applies sub-block mask to the result tensor
  void applySubBlockMask(poplar::Graph &graph, SubBlockMask subBlockMask,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix);

  // Util functions:
  void addConv1x1Vertex(poplar::Graph &graph,
                        const std::vector<poplar::Tensor> &lhs,
                        const std::vector<poplar::Tensor> &rhs,
                        const poplar::Tensor &output, unsigned int tileId,
                        poplar::ComputeSet &mulCS,
                        const std::string &debugPrefix);

  void addReduceVertex(poplar::Graph &graph,
                       const std::vector<poplar::Tensor> &partialBlocks,
                       poplar::Tensor &output, unsigned int tileId,
                       poplar::ComputeSet &reduceCS);

  // Optionally splits and/or transposes blocks before feed them to convolution.
  void preprocessBlocks(poplar::Graph &graph, const BlockMatrix &lhs,
                        const BlockMatrix &rhs,
                        std::vector<poplar::Tensor> &lhsblocks,
                        std::vector<poplar::Tensor> &rhsBlocks,
                        const std::vector<int> &lhsTileAssignment,
                        const std::vector<int> &rhsTileAssignment,
                        poplar::ComputeSet *transposeCS,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix);

public:
  const BlockMatrix &matA;
  const BlockMatrix &matB;
  std::unique_ptr<BlockMatrix> matC;

  poplar::Type inDataType;
  poplar::Type outDataType;
  poplar::Type partialDataType;

  int nTile;

  // This is an utility abstract hypergraph partitioner.
  // Derived classes could use it to partition hypergraph
  // or parts of hypergraph presented in abstract format.
  std::unique_ptr<HyperGraphPartitioner> partitioner;

  std::mt19937 randomEngine;

protected:
  // Shared worklist Tensor for different vertices
  // Since the blocks maybe grouped, the batch size may be different.
  // This is a look up table to use batch size to find the worklist tensor.
  std::unordered_map<int, poplar::Tensor> worklistTensorMap;

protected:
  unsigned int getRandomTile(int totalTile) {
    std::uniform_int_distribution<> distrib(0, totalTile - 1);
    return distrib(randomEngine);
  }
};

} // namespace experimental
} // namespace popsparse

#endif
