// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Contains public headers for Sparse Storage formats

#ifndef _poplibs_popsparse_SparseStorageFormats_hpp_
#define _poplibs_popsparse_SparseStorageFormats_hpp_

#include <vector>

namespace popsparse {

struct Block {
  // Number of columns in a block
  std::size_t getNumColumnsInBlock() const { return blockDimensions[1]; }
  // Number of rows in a block
  std::size_t getNumRowsInBlock() const {
    return blockDimensions[0];
    ;
  }
  // Block size
  std::size_t getBlockSize() const {
    return blockDimensions[0] * blockDimensions[1];
  }

  // Block dimensions
  std::array<std::size_t, 2> getBlockDimensions() const {
    return blockDimensions;
  }

protected:
  std::array<std::size_t, 2> blockDimensions;
};

/// Block Sparse matrix stored as coordinate (COO) or triplets format. The
/// case of element sparsity is treated as a special case with block size equal
/// to  {number of rows in block, number of columns in block} = {1, 1}.
template <typename T> struct COOMatrix : Block {
  /// The non-zero values of the sparse matrix.
  std::vector<T> nzValues;

  /// Corresponding column indices for the non-zero values.
  std::vector<std::size_t> columnIndices;

  /// Corresponding row indices for the non-zero values.
  std::vector<std::size_t> rowIndices;

  COOMatrix(const std::vector<T> &nzValues,
            const std::vector<std::size_t> &columnIndices,
            const std::vector<std::size_t> &rowIndices,
            const std::array<std::size_t, 2> &blockDimensions = {1, 1})
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {
    Block::blockDimensions = blockDimensions;
  }

  COOMatrix(std::vector<T> &&nzValues, std::vector<std::size_t> &&columnIndices,
            std::vector<std::size_t> &&rowIndices,
            const std::array<std::size_t, 2> &blockDimensions = {1, 1})
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {
    Block::blockDimensions = blockDimensions;
  }

  /// Constructor to allocate memory
  COOMatrix(std::size_t numNZValues,
            const std::array<std::size_t, 2> &blockDimensions_ = {1, 1}) {
    nzValues.resize(numNZValues);
    rowIndices.resize(numNZValues);
    columnIndices.resize(numNZValues);
    Block::blockDimensions = blockDimensions_;
  }

  COOMatrix(const std::array<std::size_t, 2> &blockDimensions = {1, 1}) {
    Block::blockDimensions = blockDimensions;
  }
  COOMatrix(const COOMatrix &) = default;
};

/// Sparse matrix stored in compressed sparse columns (CSC) format for a matrix
/// of size [M x N]. There is no explicit encoding of M in the storage. The
/// number of column indices is equal to (N/number of columns in block) + 1.
/// The case of element sparsity is treated as a special case with block size
/// equal to {number of rows in block, number of columns in block} = {1, 1}.
template <typename T> struct CSCMatrix : Block {
  /// The non-zero values of the sparse matrix. The number of values is always
  /// an integer multiple of the block size.
  std::vector<T> nzValues;

  /// Indices where non-zero values for each column block starts. There are a
  /// total of N/block size + 1 entries with the last entry equal to
  /// \c nzValues.
  std::vector<std::size_t> columnIndices;

  /// The row index of each block in \c nzValues. There are as many entries
  /// as the number of blocks in \c nzValues.
  std::vector<std::size_t> rowIndices;

  CSCMatrix(const std::vector<T> &nzValues,
            const std::vector<std::size_t> &columnIndices,
            const std::vector<std::size_t> &rowIndices,
            const std::array<std::size_t, 2> &blockDimensions = {1, 1})
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {
    Block::blockDimensions = blockDimensions;
  }

  CSCMatrix(std::vector<T> &&nzValues, std::vector<std::size_t> &&columnIndices,
            std::vector<std::size_t> &&rowIndices,
            const std::array<std::size_t, 2> &blockDimensions = {1, 1})
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {
    Block::blockDimensions = blockDimensions;
  }

  /// Constructor to allocate memory.
  CSCMatrix(std::size_t numNZValues, std::size_t numColumns,
            const std::array<std::size_t, 2> &blockDimensions_ = {1, 1}) {
    nzValues.resize(numNZValues);
    rowIndices.resize(numNZValues);
    columnIndices.resize(numColumns + 1);
    Block::blockDimensions = blockDimensions_;
  }

  CSCMatrix(const std::array<std::size_t, 2> &blockDimensions_ = {1, 1}) {
    Block::blockDimensions = blockDimensions_;
  }
  CSCMatrix(const CSCMatrix &) = default;
};

/// Sparse matrix stored in compressed sparse rows (CSR) format for a matrix
/// of size [M x N]. There is no explicit encoding of N in the storage. The
/// number of row indices is equal to (M / number of rows in block) + 1.
/// The case of element sparsity is treated as a special case with block size
/// equal to  {number of rows in block, number of columns in block} = {1, 1}.
template <typename T> struct CSRMatrix : Block {
  /// The non-zero values of the sparse matrix.
  std::vector<T> nzValues;

  /// The column index of each block in nzValues. There are as many as blocks
  /// in \c nzValues.
  std::vector<std::size_t> columnIndices;

  /// Indices where non-zero blocks of each row start. There are a total of M+1
  /// entries with the last entry equal to the number of entries in \c nzValues.
  std::vector<std::size_t> rowIndices;

  CSRMatrix(const std::vector<T> &nzValues,
            const std::vector<std::size_t> &columnIndices,
            const std::vector<std::size_t> &rowIndices,
            const std::array<std::size_t, 2> &blockDimensions = {1, 1})
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {
    Block::blockDimensions = blockDimensions;
  }

  CSRMatrix(std::vector<T> &&nzValues, std::vector<std::size_t> &&columnIndices,
            std::vector<std::size_t> &&rowIndices,
            const std::array<std::size_t, 2> &blockDimensions = {1, 1})
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {
    Block::blockDimensions = blockDimensions;
  }

  // Constructor to allocate memory
  CSRMatrix(std::size_t numNZValues, std::size_t numRows,
            const std::array<std::size_t, 2> &blockDimensions_ = {1, 1}) {
    nzValues.resize(numNZValues);
    columnIndices.resize(numNZValues);
    rowIndices.resize(numRows + 1);
    Block::blockDimensions = blockDimensions_;
  }

  CSRMatrix(const std::array<std::size_t, 2> &blockDimensions_ = {1, 1}) {
    Block::blockDimensions = blockDimensions_;
  }
  CSRMatrix(const CSRMatrix &) = default;
};

} // namespace popsparse
#endif // _poplibs_popsparse_SparseStorageFormats_hpp_
