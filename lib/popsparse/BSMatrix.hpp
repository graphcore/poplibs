// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popsparse_BSMatMul_hpp
#define popsparse_BSMatMul_hpp

#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

namespace popsparse {
namespace experimental {

class BlockMatrix {

public:
  enum class Order { ROW_MAJOR, COL_MAJOR };

  BlockMatrix(int rowIn, int colIn, int blockRowIn, int blockColIn,
              bool needTransposeIn)
      : row(rowIn), col(colIn), blockRow(blockRowIn), blockCol(blockColIn),
        order(Order::ROW_MAJOR), needTranspose(needTransposeIn) {
    assert(row % blockRow == 0);
    assert(col % blockCol == 0);
  }

  virtual ~BlockMatrix() = default;

  virtual bool isDense() = 0;

  virtual int getNonZeroBlockCount() = 0;

  // This function returns a 2D matrix which has index to blockData for each
  // block
  virtual std::vector<std::vector<int>> getBlockIdMatrix() = 0;

  // For dense block matrix, the input tensor is a regular 2D matrix
  // For sparse block matrix, the input tensor is an array of non zero blocks.
  virtual void setBlockTensor(const poplar::Tensor &matrixData) = 0;

  const std::vector<poplar::Tensor> &getBlockTensor() { return blockData; }

  int getRowCount() {
    if (needTranspose) {
      return col;
    } else {
      return row;
    }
  }

  int getColCount() {
    if (needTranspose) {
      return row;
    } else {
      return col;
    }
  }

  int getBlockRow() {
    if (needTranspose) {
      return blockCol;
    } else {
      return blockRow;
    }
  }

  int getBlockCol() {
    if (needTranspose) {
      return blockRow;
    } else {
      return blockCol;
    }
  }

  bool getNeedTranspose() { return needTranspose; }

  // This function is a utility function to get the regular matrix from block
  // matrix, it fills the non zero blocks with blockData and zeros for others
  poplar::Tensor getDenseMatrix(poplar::Graph &graph,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix);

protected:
  int row;
  int col;
  int blockRow;
  int blockCol;
  // order is hard coded to ROW_MAJOR for now.
  Order order;
  bool needTranspose;
  std::vector<poplar::Tensor> blockData;
};

class BlockSparseMatrix : public BlockMatrix {

public:
  std::vector<unsigned> indices;
  std::vector<unsigned> indexPtr;
  int nNonZeroBlock;

  BlockSparseMatrix(int rowIn, int colIn, int blockRowIn, int blockColIn,
                    bool needTransposeIn, const unsigned char *sparsity)
      : BlockMatrix(rowIn, colIn, blockRowIn, blockColIn, needTransposeIn) {
    init(sparsity);
  }

  virtual bool isDense() override { return false; }

  virtual int getNonZeroBlockCount() override { return nNonZeroBlock; }

  virtual std::vector<std::vector<int>> getBlockIdMatrix() override;

  virtual void setBlockTensor(const poplar::Tensor &matrixData) override;

private:
  void init(const unsigned char *sparsity);
};

class BlockDenseMatrix : public BlockMatrix {

public:
  BlockDenseMatrix(int rowIn, int colIn, int blockRowIn, int blockColIn,
                   bool needTransposeIn)
      : BlockMatrix(rowIn, colIn, blockRowIn, blockColIn, needTransposeIn) {}

  virtual bool isDense() override { return true; }

  virtual int getNonZeroBlockCount() override {
    return row * col / blockRow / blockCol;
  }

  virtual std::vector<std::vector<int>> getBlockIdMatrix() override;

  virtual void setBlockTensor(const poplar::Tensor &matrixData) override;

  poplar::Tensor denseMatrix;
};

} // namespace experimental
} // namespace popsparse

#endif