// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "BSMatrix.hpp"
#include "popops/Zero.hpp"
#include <poputil/exceptions.hpp>

namespace popsparse {
namespace experimental {

poplar::Tensor
BlockMatrix::getDenseMatrix(poplar::Graph &graph,
                            poplar::program::Sequence &prog,
                            const poplar::DebugNameAndId &dnai) const {
  poplar::Tensor dense = graph.addVariable(
      blockData[0].elementType(),
      {static_cast<unsigned long>(row), static_cast<unsigned long>(col)},
      {dnai, "dense_matrix"});

  std::vector<std::vector<int>> blockIdMatrix = getBlockIdMatrix();

  const int nrow = row / blockRow;
  const int ncol = col / blockCol;
  const int nTile = graph.getTarget().getTilesPerIPU();

  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      std::size_t blockRowStart = i * blockRow;
      std::size_t blockColStart = j * blockCol;
      std::size_t blockRowEnd = blockRowStart + blockRow;
      std::size_t blockColEnd = blockColStart + blockCol;
      poplar::Tensor t = dense.slice({blockRowStart, blockColStart},
                                     {blockRowEnd, blockColEnd});
      if (blockIdMatrix[i][j] == -1) {
        // put zero block uniformly accross all tiles
        graph.setTileMapping(t, (i * ncol + j) % nTile);
        popops::zero(graph, t, prog, {dnai});
      } else {
        poplar::Tensor oneBlock = blockData.at(blockIdMatrix[i][j]);
        graph.setTileMapping(t, graph.getTileMapping(oneBlock));
        prog.add(poplar::program::Copy(t, oneBlock));
      }
    }
  }

  return dense;
}

///////////////////////////////////////////////////////////////////////////////
// Sparse Block Matrix
///////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::init(const unsigned char *sparsity) {
  assert(sparsity != nullptr);

  const int nrow = row / blockRow;
  const int ncol = col / blockCol;

  indexPtr.reserve(nrow + 1);

  int indexForRow = 0;
  for (int i = 0; i < nrow; i++) {
    indexPtr.push_back(indexForRow);
    for (int j = 0; j < ncol; j++) {
      if (sparsity[i * ncol + j]) {
        indices.push_back(j * blockCol);
        indexForRow++;
      }
    }
  }
  indexPtr.push_back(indexForRow);
}

std::vector<std::vector<int>> BlockSparseMatrix::getBlockIdMatrix() const {
  std::vector<std::vector<int>> blockIdMatrix;

  const int nRow = row / blockRow;
  const int nCol = col / blockCol;
  if (needTranspose) {
    blockIdMatrix.resize(nCol);
    for (int i = 0; i < nCol; i++) {
      blockIdMatrix[i].resize(nRow, -1);
    }
  } else {
    blockIdMatrix.resize(nRow);
    for (int i = 0; i < nRow; i++) {
      blockIdMatrix[i].resize(nCol, -1);
    }
  }

  int blockCount = 0;
  for (uint m = 0; m < indexPtr.size() - 1; m++) {
    int start = indexPtr[m];
    int end = indexPtr[m + 1];
    for (int n = start; n < end; n++) {
      int colIndex = indices[n] / blockCol;
      if (needTranspose) {
        blockIdMatrix[colIndex][m] = blockCount++;
      } else {
        blockIdMatrix[m][colIndex] = blockCount++;
      }
    }
  }

  return blockIdMatrix;
}

void BlockSparseMatrix::setBlockTensor(const poplar::Tensor &matrixData) {
  assert(matrixData.rank() == 2);
  const int nBlocks = getNonZeroBlockCount();
  const int blockArea = blockRow * blockCol;
  if (nBlocks != static_cast<int>(matrixData.dim(0)) ||
      blockArea != static_cast<int>(matrixData.dim(1))) {
    throw poputil::poplibs_error("Sparse tensor must "
                                 "have shape [" +
                                 std::to_string(nBlocks) + " x " +
                                 std::to_string(blockArea) + "]");
  }
  blockData.resize(nBlocks);
  for (int i = 0; i < nBlocks; i++) {
    blockData[i] = matrixData[i];
  }
}

poplar::Tensor
BlockSparseMatrix::createTensor(poplar::Graph &graph,
                                const poplar::Type &dataType,
                                const poplar::DebugNameAndId &dnai) const {
  int nonZeroBlocks = getNonZeroBlockCount();
  std::vector<poplar::Tensor> blocks(nonZeroBlocks);
  for (int i = 0; i < nonZeroBlocks; i++) {
    blocks[i] =
        graph
            .addVariable(dataType,
                         {static_cast<unsigned long>(blockRow * blockCol)},
                         {dnai, std::to_string(i)})
            .expand({0});
  }
  return concat(blocks);
}

std::array<int, 2> BlockSparseMatrix::getDimensions() const {
  std::array<int, 2> dims;
  dims[0] = getNonZeroBlockCount();
  dims[1] = blockRow * blockCol;
  return dims;
}

///////////////////////////////////////////////////////////////////////////////
// Dense Block Matrix
///////////////////////////////////////////////////////////////////////////////
void BlockDenseMatrix::setBlockTensor(const poplar::Tensor &matrixData) {
  assert(matrixData.rank() == 2);
  if (row != static_cast<int>(matrixData.dim(0)) ||
      col != static_cast<int>(matrixData.dim(1))) {
    throw poputil::poplibs_error("Dense tensor must "
                                 "have shape [" +
                                 std::to_string(row) + " x " +
                                 std::to_string(col) + "]");
  }

  denseMatrix = matrixData;

  blockData.resize(row * col / blockRow / blockCol);
  poplar::Tensor t = matrixData
                         .reshape({static_cast<std::size_t>(row / blockRow),
                                   static_cast<std::size_t>(blockRow),
                                   static_cast<std::size_t>(col / blockCol),
                                   static_cast<std::size_t>(blockCol)})
                         .dimShuffle({0, 2, 1, 3});
  int blockCount = 0;
  for (int i = 0; i < row / blockRow; i++) {
    for (int j = 0; j < col / blockCol; j++) {
      blockData[blockCount] = t[i][j].flatten();
      blockCount++;
    }
  }
}

poplar::Tensor
BlockDenseMatrix::createTensor(poplar::Graph &graph,
                               const poplar::Type &dataType,
                               const poplar::DebugNameAndId &dnai) const {
  // Create variable with the memory layout we want
  auto t = graph.addVariable(dataType,
                             {static_cast<std::size_t>(row / blockRow),
                              static_cast<std::size_t>(col / blockCol),
                              static_cast<std::size_t>(blockRow),
                              static_cast<std::size_t>(blockCol)},
                             {dnai});
  // Dimshuffle / reshape to 2D tensor with the correct dimensions for the
  // matrix.
  return t.dimShuffle({0, 2, 1, 3})
      .reshape({static_cast<std::size_t>(row), static_cast<std::size_t>(col)});
}

std::vector<std::vector<int>> BlockDenseMatrix::getBlockIdMatrix() const {
  std::vector<std::vector<int>> blockIdMatrix;
  const int blockRows = row / blockRow;
  const int blockCols = col / blockCol;
  if (needTranspose) {
    blockIdMatrix = std::vector<std::vector<int>>(
        blockCols, std::vector<int>(blockRows, -1));
    for (int br = 0, blockCount = 0; br < blockRows; br++) {
      for (int bc = 0; bc < blockCols; bc++) {
        blockIdMatrix[bc][br] = blockCount++;
      }
    }
  } else {
    blockIdMatrix = std::vector<std::vector<int>>(
        blockRows, std::vector<int>(blockCols, -1));
    for (int br = 0, blockCount = 0; br < blockRows; br++) {
      for (int bc = 0; bc < blockCols; bc++) {
        blockIdMatrix[br][bc] = blockCount++;
      }
    }
  }

  return blockIdMatrix;
}

std::array<int, 2> BlockDenseMatrix::getDimensions() const {
  std::array<int, 2> dims;
  dims[0] = row;
  dims[1] = col;
  return dims;
}

} // namespace experimental
} // namespace popsparse