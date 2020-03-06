// Copyright (c) 2020 Graphcore Ltd, All rights reserved.
#include "BSMatrix.hpp"
#include "popops/Zero.hpp"

namespace popsparse {
namespace experimental {

poplar::Tensor
BlockMatrix::getDenseMatrix(poplar::Graph &graph,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix) const {
  poplar::Tensor dense = graph.addVariable(
      blockData[0].elementType(),
      {static_cast<unsigned long>(row), static_cast<unsigned long>(col)},
      debugPrefix + "/dense_matrix");

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
        popops::zero(graph, t, prog, debugPrefix);
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
  const int nBlocks = getNonZeroBlockCount();
  assert(nBlocks == static_cast<int>(matrixData.dim(0)));
  blockData.resize(nBlocks);
  for (int i = 0; i < nBlocks; i++) {
    blockData[i] = matrixData[i];
  }
}

poplar::Tensor BlockSparseMatrix::createTensor(poplar::Graph &graph,
                                               const poplar::Type &dataType,
                                               const std::string &name) const {
  int nonZeroBlocks = getNonZeroBlockCount();
  std::vector<poplar::Tensor> blocks(nonZeroBlocks);
  for (int i = 0; i < nonZeroBlocks; i++) {
    blocks[i] =
        graph
            .addVariable(dataType,
                         {static_cast<unsigned long>(blockRow * blockCol)},
                         name + "_" + std::to_string(i))
            .expand({0});
  }
  return concat(blocks);
}

///////////////////////////////////////////////////////////////////////////////
// Dense Block Matrix
///////////////////////////////////////////////////////////////////////////////
void BlockDenseMatrix::setBlockTensor(const poplar::Tensor &matrixData) {
  denseMatrix = matrixData;

  const int blockRows = row / blockRow;
  const int blockCols = col / blockCol;

  blockData.resize(blockRows * blockCols);
  for (int br = 0, blockCount = 0; br < blockRows; br++) {
    for (int bc = 0; bc < blockCols; bc++) {
      std::size_t blockRowStart = br * blockRow;
      std::size_t blockColStart = bc * blockCol;
      std::size_t blockRowEnd = blockRowStart + blockRow;
      std::size_t blockColEnd = blockColStart + blockCol;
      blockData[blockCount++] =
          matrixData
              .slice({blockRowStart, blockColStart}, {blockRowEnd, blockColEnd})
              .flatten();
    }
  }
}

poplar::Tensor BlockDenseMatrix::createTensor(poplar::Graph &graph,
                                              const poplar::Type &dataType,
                                              const std::string &name) const {
  poplar::Tensor t;
  t = graph.addVariable(
      dataType,
      {static_cast<unsigned long>(row), static_cast<unsigned long>(col)}, name);
  return t;
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

} // namespace experimental
} // namespace popsparse