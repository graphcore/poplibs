// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "BSOps.hpp"
#include "BSUtils.hpp"
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>

#include <algorithm>
#include <vector>

using namespace poplar;
using namespace poputil;

namespace popsparse {
namespace experimental {

Tensor slice(const Tensor &sparseTensor, std::size_t coord, unsigned dimension,
             unsigned blockRow, unsigned blockCol, unsigned blockRows,
             unsigned blockCols, bool columnMajorBlock,
             const unsigned char *sparsity) {
  if (sparseTensor.rank() != 2) {
    throw poplibs_error("The rank of sparse tensor must be 2");
  }
  if (dimension >= 2) {
    throw poplibs_error("The slicing dimension must be < 2");
  }
  if (blockRow == 0 || blockCol == 0 || blockRows == 0 || blockCols == 0) {
    throw poplibs_error("Block dimension cannot be zero");
  }
  unsigned rows = blockRow * blockRows;
  unsigned cols = blockCol * blockCols;
  unsigned blockArea = blockRow * blockCol;
  if (sparseTensor.dim(1) != blockArea) {
    throw poplibs_error("The second dimension of sparse tensor must be equal "
                        "to the block area");
  }
  if (dimension == 0 && coord >= rows) {
    throw poplibs_error(
        "coord parameter must be less than the number of dense rows");
  }
  if (dimension == 1 && coord >= cols) {
    throw poplibs_error(
        "coord parameter must be less than the number of dense columns");
  }
  std::vector<Tensor> sliceParts;
  unsigned idxBlockSparse = 0;
  unsigned idxBlockDense = 0;
  unsigned row = 0;
  for (unsigned int br = 0; br < blockRows; ++br, row += blockRow) {
    unsigned col = 0;
    for (unsigned int bc = 0; bc < blockCols;
         ++bc, ++idxBlockDense, col += blockCol) {
      if (sparsity[idxBlockDense]) {
        if (idxBlockSparse >= sparseTensor.dim(0)) {
          throw poplibs_error(
              "The number of blocks in sparse tensor is less "
              "than the number of non-zero elements in sparsity mask");
        }
        if (dimension == 0) {
          if (row <= coord && row + blockRow > coord) {
            unsigned rowSparse = coord - row;
            if (!columnMajorBlock) {
              unsigned idxInBlock = rowSparse * blockCol;
              assert(idxInBlock + blockCol <= sparseTensor.dim(1));
              Tensor t = sparseTensor[idxBlockSparse].slice(
                  idxInBlock, idxInBlock + blockCol);
              sliceParts.push_back(t);
            } else {
              Tensor transposedBlock = sparseTensor[idxBlockSparse]
                                           .reshape({blockRow, blockCol})
                                           .transpose()
                                           .reshape({blockArea});
              unsigned idxInBlock = rowSparse * blockCol;
              Tensor t =
                  transposedBlock.slice(idxInBlock, idxInBlock + blockCol);
              sliceParts.push_back(t);
            }
          }
        } else {
          if (col <= coord && col + blockCol > coord) {
            unsigned colSparse = coord - col;
            if (columnMajorBlock) {
              unsigned idxInBlock = colSparse * blockRow;
              assert(idxInBlock + blockRow <= sparseTensor.dim(1));
              Tensor t = sparseTensor[idxBlockSparse].slice(
                  idxInBlock, idxInBlock + blockRow);
              sliceParts.push_back(t);
            } else {
              Tensor transposedBlock = sparseTensor[idxBlockSparse]
                                           .reshape({blockCol, blockRow})
                                           .transpose()
                                           .reshape({blockArea});
              unsigned idxInBlock = colSparse * blockRow;
              Tensor t =
                  transposedBlock.slice(idxInBlock, idxInBlock + blockRow);
              sliceParts.push_back(t);
            }
          }
        }
        ++idxBlockSparse;
      }
    }
  }
  Tensor sliceTensor;
  if (!sliceParts.empty()) {
    sliceTensor = concat(sliceParts).flatten();
  } else {
    sliceTensor = sparseTensor[0].slice(0, 0);
  }
  return sliceTensor;
}

void applySubBlockMask(poplar::Graph &graph, const poplar::Tensor &sparseTensor,
                       SubBlockMask subBlockMask, unsigned blockRow,
                       unsigned blockCol, unsigned blockRows,
                       unsigned blockCols, const unsigned char *sparsity,
                       unsigned numGroups, poplar::program::Sequence &prog,
                       const std::string &debugPrefix) {

  if (blockRow == 0 || blockCol == 0 || blockRows == 0 || blockCols == 0) {
    throw poplibs_error("Block dimension cannot be zero");
  }
  if ((sparseTensor.rank() != 2) && (sparseTensor.rank() != 3)) {
    throw poplibs_error("The rank of sparse tensor must be 2 or 3");
  }
  std::vector<bool> emptyRowsMask; // Not used here, only for sparse softmax
  poplar::Type dataType = sparseTensor.elementType();
  std::vector<poplar::Tensor> diagonalBlocks;
  std::vector<poplar::Tensor> maskBlocks;

  unsigned blockArea = blockRow * blockCol;
  if (sparseTensor.dim(sparseTensor.rank() - 1) != blockArea) {
    throw poplibs_error("The last dimension of sparse tensor must be equal "
                        "to the block area");
  }
  if (numGroups == 1) {
    std::vector<unsigned> diagBlockIdxs;
    bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                       sparsity, subBlockMask, 0.0f, 1.0f, dataType, maskBlocks,
                       diagBlockIdxs, emptyRowsMask, debugPrefix);
    for (unsigned idxDiagBlock : diagBlockIdxs) {
      if (idxDiagBlock >= sparseTensor.dim(0)) {
        throw poplibs_error(
            "The number of blocks in sparse tensor is less "
            "than the number of non-zero elements in sparsity mask");
      }
      diagonalBlocks.push_back(sparseTensor[idxDiagBlock].expand({0}));
    }
  } else {
    std::size_t denseSize = blockRows * blockCols;
    const unsigned char *curSparsity = sparsity;
    std::size_t dim0Start = 0;
    for (unsigned g = 0; g < numGroups; ++g) {
      unsigned nonZeroBlock =
          std::accumulate(curSparsity, curSparsity + denseSize, 0);
      std::size_t dim0End = dim0Start + nonZeroBlock;
      if (dim0End > sparseTensor.dim(0)) {
        throw poplibs_error(
            "The number of blocks in sparse tensor is less "
            "than the number of non-zero elements in sparsity mask");
      }
      poplar::Tensor sparseSubTensor =
          sparseTensor.slice(dim0Start, dim0End, 0);

      std::vector<unsigned> diagBlockIdxs;
      std::vector<poplar::Tensor> curMaskBlocks;
      bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                         curSparsity, subBlockMask, 0.0f, 1.0f, dataType,
                         curMaskBlocks, diagBlockIdxs, emptyRowsMask,
                         debugPrefix);
      maskBlocks.insert(maskBlocks.end(), curMaskBlocks.begin(),
                        curMaskBlocks.end());
      for (unsigned idxDiagBlock : diagBlockIdxs) {
        assert(idxDiagBlock < sparseSubTensor.dim(0));
        diagonalBlocks.push_back(sparseSubTensor[idxDiagBlock].expand({0}));
      }

      curSparsity += denseSize;
      dim0Start = dim0End;
    }
  }

  if (!diagonalBlocks.empty()) {
    popops::mulInPlace(graph, concat(diagonalBlocks), concat(maskBlocks), prog,
                       debugPrefix + "/subBlockMasked");
  }
}

} // namespace experimental
} // namespace popsparse