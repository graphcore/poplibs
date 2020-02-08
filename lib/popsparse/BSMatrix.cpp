
#include "BSMatrix.hpp"
#include "popops/Zero.hpp"

namespace popsparse {
namespace experimental {

poplar::Tensor BlockMatrix::getDenseMatrix(poplar::Graph &graph,
                                           poplar::program::Sequence &prog,
                                           const std::string &debugPrefix) {
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

  if (order == Order::ROW_MAJOR) {
    int indexForRow = 0;
    nNonZeroBlock = 0;
    for (int i = 0; i < nrow; i++) {
      indexPtr.push_back(indexForRow);
      for (int j = 0; j < ncol; j++) {
        if (sparsity[i * ncol + j]) {
          indices.push_back(j * blockCol);
          indexForRow++;
          nNonZeroBlock++;
        }
      }
    }
    indexPtr.push_back(indexForRow);
  } else {
    // TODO: not implemented
    assert(0);
  }
}

std::vector<std::vector<int>> BlockSparseMatrix::getBlockIdMatrix() {
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

///////////////////////////////////////////////////////////////////////////////
// Dense Block Matrix
///////////////////////////////////////////////////////////////////////////////
void BlockDenseMatrix::setBlockTensor(const poplar::Tensor &matrixData) {
  denseMatrix = matrixData;

  const int nrow = row / blockRow;
  const int ncol = col / blockCol;

  if (order == Order::ROW_MAJOR) {
    blockData.resize(nrow * ncol);
    for (int i = 0, blockCount = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        std::size_t blockRowStart = i * blockRow;
        std::size_t blockColStart = j * blockCol;
        std::size_t blockRowEnd = blockRowStart + blockRow;
        std::size_t blockColEnd = blockColStart + blockCol;
        blockData[blockCount++] = matrixData
                                      .slice({blockRowStart, blockColStart},
                                             {blockRowEnd, blockColEnd})
                                      .flatten();
      }
    }
  } else {
    // TODO: not implemented
    assert(0);
  }
}

std::vector<std::vector<int>> BlockDenseMatrix::getBlockIdMatrix() {
  std::vector<std::vector<int>> blockIdMatrix;

  const int nRow = row / blockRow;
  const int nCol = col / blockCol;
  if (needTranspose) {
    blockIdMatrix.resize(nCol);
    for (int i = 0, blockCount = 0; i < nCol; i++) {
      blockIdMatrix[i].resize(nRow);
      for (int j = 0; j < nRow; j++) {
        blockIdMatrix[j][i] = blockCount++;
      }
    }
  } else {
    blockIdMatrix.resize(nRow);
    for (int i = 0, blockCount = 0; i < nRow; i++) {
      blockIdMatrix[i].resize(nCol);
      for (int j = 0; j < nCol; j++) {
        blockIdMatrix[i][j] = blockCount++;
      }
    }
  }

  return blockIdMatrix;
}

} // namespace experimental
} // namespace popsparse
