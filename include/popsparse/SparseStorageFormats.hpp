// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Contains public headers for Sparse Storage formats

#ifndef _poplibs_popsparse_SparseStorageFormats_hpp_
#define _poplibs_popsparse_SparseStorageFormats_hpp_

#include <vector>

namespace popsparse {

/// Sparse matrix stored as coordinate (COO) or triplets format.
template <typename T> struct COOMatrix {
  /// The non-zero values of the sparse matrix.
  std::vector<T> nzValues;

  /// Corresponding column indices for the non-zero values.
  std::vector<std::size_t> columnIndices;

  /// Corresponding row indices for the non-zero values.
  std::vector<std::size_t> rowIndices;

  COOMatrix(const std::vector<T> &nzValues,
            const std::vector<std::size_t> &columnIndices,
            const std::vector<std::size_t> &rowIndices)
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {}

  COOMatrix(std::vector<T> &&nzValues, std::vector<std::size_t> &&columnIndices,
            std::vector<std::size_t> &&rowIndices)
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {}

  /// Constructor to allocate memory
  COOMatrix(std::size_t numNZValues) {
    nzValues.resize(numNZValues);
    rowIndices.resize(numNZValues);
    columnIndices.resize(numNZValues);
  }

  COOMatrix() = default;
  COOMatrix(const COOMatrix &) = default;
};

/// Sparse matrix stored in compressed sparse columns (CSC) format for a matrix
/// of size [M x N]. There is no explicit encoding of M in the storage. The
/// number of column indices is equal to N + 1.
template <typename T> struct CSCMatrix {
  /// The non-zero values of the sparse matrix.
  std::vector<T> nzValues;

  /// Indices where non-zero values for each column start. There are a total of
  /// N+1 entries with the last entry equal to the number of entries in
  /// \c nzValues.
  std::vector<std::size_t> columnIndices;

  /// The row index of each element in \c nzValues. There are as many entries
  /// as \c nzValues.
  std::vector<std::size_t> rowIndices;

  CSCMatrix(const std::vector<T> &nzValues,
            const std::vector<std::size_t> &columnIndices,
            const std::vector<std::size_t> &rowIndices)
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {}

  CSCMatrix(std::vector<T> &&nzValues, std::vector<std::size_t> &&columnIndices,
            std::vector<std::size_t> &&rowIndices)
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {}

  /// Constructor to allocate memory.
  CSCMatrix(std::size_t numNZValues, std::size_t numColumns) {
    nzValues.resize(numNZValues);
    rowIndices.resize(numNZValues);
    columnIndices.resize(numColumns + 1);
  }

  CSCMatrix() = default;
  CSCMatrix(const CSCMatrix &) = default;
};

/// Sparse matrix stored in compressed sparse rows (CSR) format for a matrix
/// of size [M x N]. There is no explicit encoding of N in the storage. The
/// number of row indices is equal to M + 1.
template <typename T> struct CSRMatrix {
  /// The non-zero values of the sparse matrix.
  std::vector<T> nzValues;

  /// The column index of each element in nzValues. There are as many entries
  /// as \c nzValues.
  std::vector<std::size_t> columnIndices;

  /// Indices where non-zero values of each row start. There are a total of M+1
  /// entries with the last entry equal to the number of entries in \c nzValues.
  std::vector<std::size_t> rowIndices;

  CSRMatrix(const std::vector<T> &nzValues,
            const std::vector<std::size_t> &columnIndices,
            const std::vector<std::size_t> &rowIndices)
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {}

  CSRMatrix(std::vector<T> &&nzValues, std::vector<std::size_t> &&columnIndices,
            std::vector<std::size_t> &&rowIndices)
      : nzValues(nzValues), columnIndices(columnIndices),
        rowIndices(rowIndices) {}

  // Constructor to allocate memory
  CSRMatrix(std::size_t numNZValues, std::size_t numRows) {
    nzValues.resize(numNZValues);
    columnIndices.resize(numNZValues);
    rowIndices.resize(numRows + 1);
  }

  CSRMatrix() = default;
  CSRMatrix(const CSRMatrix &) = default;
};

} // namespace popsparse
#endif // _poplibs_popsparse_SparseStorageFormats_hpp_
