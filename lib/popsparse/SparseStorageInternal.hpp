// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Internally used functions for sparse storage representation

#ifndef _poplibs_popsparse_SparseStorageInternal_hpp_
#define _poplibs_popsparse_SparseStorageInternal_hpp_

#include "poplar/Interval.hpp"
#include "popsparse/SparseStorageFormats.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <numeric>

namespace popsparse {

using ValueType = std::size_t;
using CSCInternal = CSCMatrix<ValueType>;
using CSRInternal = CSRMatrix<ValueType>;
using COOInternal = COOMatrix<ValueType>;

// Gives the row index, column index and the z index
using TileIndex = std::tuple<std::size_t, std::size_t, std::size_t>;

// Represents a tile in a matrix with right open intervals for rows and columns
class Tile {
private:
  poplar::Interval rows;
  poplar::Interval columns;

public:
  Tile(const poplar::Interval &rows, const poplar::Interval &columns)
      : rows(rows), columns(columns){};

  Tile() = default;
  Tile(const Tile &) = default;

  poplar::Interval getRows() const { return rows; }

  poplar::Interval getColumns() const { return columns; }

  std::size_t size() const { return rows.size() * columns.size(); }

  friend std::ostream &operator<<(std::ostream &os, const Tile &t) {
    os << "rows:[ " << t.getRows().begin() << " " << t.getRows().end();
    os << " ] cols:[ " << t.getColumns().begin() << " " << t.getColumns().end();
    os << "]";
    return os;
  }
};

// Non zero information in a row. Contains the row number and a vector of
// non zero column value pairs in that row.
struct RowPositionValues {
  std::size_t rowNumber;
  std::vector<std::pair<std::size_t, ValueType>> positionValues;

  RowPositionValues(
      std::size_t rowNumber,
      const std::vector<std::pair<std::size_t, ValueType>> &positionValues)
      : rowNumber(rowNumber), positionValues(positionValues) {}

  friend bool operator>(const RowPositionValues &a,
                        const RowPositionValues &b) {
    return a.positionValues.size() > b.positionValues.size();
  }
};

// The row and column dimensions are divided by straight lines creating
// rectangular tiles. A tile is then given by a right open interval for
// the row and column dimension.
struct TilePartition {
  // indices of the tile in the X, Y, Z splits
  TileIndex tileIndex;
  // actual row and column intervals
  Tile tile;
  std::vector<RowPositionValues> tileInfo;

  TilePartition() = default;
  TilePartition(const TilePartition &) = default;

  TilePartition(const TileIndex &tileIndex, const Tile &tile,
                const std::vector<RowPositionValues> &tileInfo_)
      : tileIndex(tileIndex), tile(tile) {
    tileInfo = tileInfo_;

    // keep sorted so that it is easy to remove rows which are the smallest
    // first
    std::sort(tileInfo.begin(), tileInfo.end(),
              [](const RowPositionValues &a, const RowPositionValues &b) {
                return a > b;
              });
  }

  std::size_t numNzValues() const {
    std::size_t num = 0;
    for (const auto &t : tileInfo) {
      num += t.positionValues.size();
    }
    return num;
  }

  bool empty() const { return tileInfo.size() == 0; }
};

// Information kept in a bucket. Two buckets are kept. One for meta information
// and the other for non-zero values.
struct PNBucket {
  std::size_t metaInfoElements;
  std::size_t numNzElements;

  std::vector<TilePartition> subGroups;
  std::size_t numSubgroups() const { return subGroups.size(); }
  bool empty() const { return metaInfoElements == 0 && numNzElements == 0; }

  friend bool operator>(const PNBucket &a, const PNBucket &b) {
    return std::tie(a.metaInfoElements, a.numNzElements) >
           std::tie(b.metaInfoElements, b.numNzElements);
  }
  friend bool operator<(const PNBucket &a, const PNBucket &b) {
    return std::tie(a.metaInfoElements, a.numNzElements) <
           std::tie(b.metaInfoElements, b.numNzElements);
  }

  void move(PNBucket &other) {
    metaInfoElements += other.metaInfoElements;
    numNzElements += other.numNzElements;
    for (auto &sg : other.subGroups) {
      subGroups.push_back(std::move(sg));
    }
    other.metaInfoElements = 0;
    other.numNzElements = 0;
  }
};

static inline void validateBlockSizes(std::size_t numRows,
                                      std::size_t numColumns,
                                      std::size_t rowBlockSize,
                                      std::size_t columnBlockSize) {
  if (numRows % rowBlockSize || rowBlockSize > numRows) {
    throw poputil::poplibs_error("Invalid row block size");
  }
  if (numColumns % columnBlockSize || columnBlockSize > numColumns) {
    throw poputil::poplibs_error("Invalid row block size");
  }
}

// Convert from a CSR representation of a matrix of dimension
// [numRows x numColumns] to a CSC representation.
//
// \param  numRows    Number of rows in the matrix.
// \param  numColumns Number of columns in the matrix.
// \param  csr        CSR representation of matrix .
//
// \return CSC representation of the matrix.
// Position and value pairs for each row
template <class T>
CSCMatrix<T> csrToCSC(std::size_t numRows, std::size_t numColumns,
                      const CSRMatrix<T> &csr) {

  // number of NZ blocks are always equal to the number of column indices.
  const std::size_t numNZBlocks = csr.columnIndices.size();
  const auto blockSize = csr.getBlockSize();
  const auto numNZValues = numNZBlocks * blockSize;

  validateBlockSizes(numRows, numColumns, csr.getNumRowsInBlock(),
                     csr.getNumColumnsInBlock());
  if (csr.rowIndices.back() != numNZValues) {
    throw poputil::poplibs_error("Number of non-zero values do not match last "
                                 "entry on rowIndices");
  }
  const auto numColumnBlocks = numColumns / csr.getNumColumnsInBlock();
  std::vector<std::size_t> columnIndices(numColumnBlocks + 1);
  for (std::size_t nz = 0; nz != numNZBlocks; ++nz) {
    columnIndices[csr.columnIndices[nz] / csr.getNumColumnsInBlock() + 1]++;
  }
  std::partial_sum(columnIndices.begin(), columnIndices.end(),
                   columnIndices.begin());
  // The last entry in the CSC columns is always the number of non zero entries
  columnIndices[numColumnBlocks] = numNZBlocks;

  std::vector<std::size_t> rowIndices;
  rowIndices.resize(numNZBlocks);
  std::vector<T> nzValues;
  nzValues.resize(numNZValues);

  // Fill in the non zero entries and the row indices
  for (std::size_t row = 0; row != numRows / csr.getNumRowsInBlock(); ++row) {
    for (std::size_t csrRow = csr.rowIndices[row] / blockSize;
         csrRow != csr.rowIndices[row + 1] / blockSize; ++csrRow) {
      auto column = csr.columnIndices[csrRow] / csr.getNumColumnsInBlock();
      auto dstRow = columnIndices[column];
      rowIndices[dstRow] = row;
      // transpose block and store
      for (std::size_t r = 0; r != csr.getNumRowsInBlock(); ++r) {
        for (std::size_t c = 0; c != csr.getNumColumnsInBlock(); ++c) {
          const auto srcIndex =
              dstRow * blockSize + c * csr.getNumRowsInBlock() + r;
          const auto dstIndex =
              csrRow * blockSize + r * csr.getNumColumnsInBlock() + c;
          nzValues[srcIndex] = csr.nzValues[dstIndex];
        }
      }
      ++columnIndices[column];
    }
  }

  std::size_t startIndex = 0;
  for (std::size_t column = 0; column != numColumnBlocks; ++column) {
    std::swap(columnIndices[column], startIndex);
  }

  // scale to block dimensions
  std::for_each(rowIndices.begin(), rowIndices.end(),
                [=](std::size_t &x) { x *= csr.getNumRowsInBlock(); });
  std::for_each(columnIndices.begin(), columnIndices.end(),
                [=](std::size_t &x) { x *= blockSize; });

  return CSCMatrix<T>(std::move(nzValues), std::move(columnIndices),
                      std::move(rowIndices),
                      {csr.getNumColumnsInBlock(), csr.getNumRowsInBlock()});
}

// Sort the columns of a CSR matrix in a given row to be in increasing order.
// The operation is done inplace.
//
// \param  csr  The matrix in csr representation to be canonicalized
template <class T> void canonicalizeCSR(CSRMatrix<T> &matrix) {
  const auto numRowBlocks = matrix.rowIndices.size() - 1;
  const auto blockSize = matrix.getBlockSize();

  for (std::size_t row = 0; row != numRowBlocks; ++row) {
    const auto startIndex = matrix.rowIndices[row] / blockSize;
    const auto endIndex = matrix.rowIndices[row + 1] / blockSize;
    const auto numElems = endIndex - startIndex;

    std::vector<std::vector<T>> columnNzValues;
    std::vector<std::size_t> columnIndices;
    columnNzValues.resize(numElems);
    columnIndices.resize(numElems);

    for (std::size_t index = startIndex; index != endIndex; ++index) {
      std::vector<T> nzBlock;
      columnNzValues[index - startIndex].resize(blockSize);
      std::copy(matrix.nzValues.begin() + index * blockSize,
                matrix.nzValues.begin() + (index + 1) * blockSize,
                columnNzValues[index - startIndex].begin());
      columnIndices[index - startIndex] = matrix.columnIndices[index];
    }

    std::vector<std::size_t> columnOrder;
    columnOrder.resize(numElems);
    std::iota(columnOrder.begin(), columnOrder.end(), 0);

    std::sort(columnOrder.begin(), columnOrder.end(),
              [&](std::size_t a, std::size_t b) {
                return columnIndices[a] < columnIndices[b];
              });

    for (std::size_t index = startIndex; index != endIndex; ++index) {
      auto thisIndex = columnOrder[index - startIndex];
      matrix.columnIndices[index] = columnIndices[thisIndex];
      std::move(columnNzValues[thisIndex].begin(),
                columnNzValues[thisIndex].end(),
                matrix.nzValues.begin() + index * blockSize);
    }
  }
}

// Convert a CSC representation of a matrix of dimension [numRows x numColumns]
// to a CSR representation.
template <class T>
CSRMatrix<T> cscToCSR(std::size_t numRows, std::size_t numColumns,
                      const CSCMatrix<T> &input) {
  auto csrMatrix =
      CSRMatrix<T>(input.nzValues, input.rowIndices, input.columnIndices,
                   {input.getNumColumnsInBlock(), input.getNumRowsInBlock()});
  auto cscMatrix = csrToCSC<T>(numColumns, numRows, csrMatrix);
  return CSRMatrix<T>(
      std::move(cscMatrix.nzValues), std::move(cscMatrix.rowIndices),
      std::move(cscMatrix.columnIndices),
      {cscMatrix.getNumRowsInBlock(), cscMatrix.getNumColumnsInBlock()});
}

// Transpose a CSR representation of a matrix of dimension
// [numRows x numColumns] to a CSR representation of it's transpose
//
// \param  numRows    Number of rows of the input matrix.
// \param  numColumns Number of columns of the input matrix.
// \param  input      CSR representation of the input.
template <class T>
CSRMatrix<T> csrTranspose(std::size_t numRows, std::size_t numColumns,
                          const CSRMatrix<T> &input) {
  auto output = csrToCSC(numRows, numColumns, input);
  // convert to CSC matrix
  return CSRMatrix<T>(
      output.nzValues, output.rowIndices, output.columnIndices,
      {output.getNumRowsInBlock(), output.getNumColumnsInBlock()});
}

// Given a tile, return the information for every row containing non-zero values
// in it.
// Each row is a vector of <column index, non-zero value> which meets the
// tile boundaries. The rows and columns within the tile are offsets within
// the tile and their absolute positions can be obtained from the tile
// information stored as part of the partition.
//
// \param  csr        The internal CSR representation of the sparse matrix.
// \param  blockSizeX The block size in the X dimension for the matrix the
//                    internal CSR representation refers to.
// \param  blockSizeY The block size in the Y dimension for the matrix the
//                    internal CSR representation refers to.
// \param  tile       The row and column intervals for the tile
//
// A vector for each row in the row interval range containing (column, value)
// pairs.
std::vector<RowPositionValues>
getPositionValuePairsPerRow(const CSRInternal &csr, std::size_t blockSizeX,
                            std::size_t blockSizeY, const Tile &tile);

// Convert from a COO representation of a matrix of dimension
// [numRows x numColumns] to a CSR representation. Duplicate entries are not
// handled by the function.
//
// \return CSR representation of the matrix.
// Position and value pairs for each row
template <class T>
CSRMatrix<T> cooToCSR(std::size_t numRows, std::size_t numColumns,
                      const COOMatrix<T> &coo) {

  // number of NZ values are always equal to the number of column indices.
  const std::size_t numNZValues = coo.nzValues.size();
  const auto blockSize = coo.getBlockSize();
  const auto rowBlockSize = coo.getNumRowsInBlock();
  const auto numNZBlocks = numNZValues / blockSize;
  const auto numRowBlocks = numRows / rowBlockSize;

  if (numNZValues > numRows * numColumns) {
    throw poputil::poplibs_error("Number of non-zero blocks in COO exceed the "
                                 "size of the matrix");
  }

  if (coo.rowIndices.size() != numNZBlocks) {
    throw poputil::poplibs_error("Number of non-zero blocks does not match "
                                 "number of row elements in the COO");
  }

  if (coo.columnIndices.size() != numNZBlocks) {
    throw poputil::poplibs_error("Number of non-zero blocks does not match "
                                 "number of column elements in the COO");
  }

  std::vector<std::size_t> rowIndices(numRowBlocks + 1);
  for (const auto &row : coo.rowIndices) {
    const auto rowBlock = row / rowBlockSize;
    rowIndices.at(rowBlock + 1) += blockSize;
  }
  std::partial_sum(std::next(rowIndices.begin()), rowIndices.end(),
                   std::next(rowIndices.begin()));

  std::vector<std::size_t> perRowBlockIndex(numRowBlocks);
  std::vector<std::size_t> columnIndices(numNZBlocks);
  std::vector<T> nzValues(numNZValues);

  for (std::size_t block = 0; block < numNZBlocks; ++block) {
    const auto rowBlock = coo.rowIndices[block] / rowBlockSize;
    const auto dstIndex =
        rowIndices[rowBlock] / blockSize + perRowBlockIndex[rowBlock];
    columnIndices[dstIndex] = coo.columnIndices[block];
    std::copy(coo.nzValues.begin() + block * blockSize,
              coo.nzValues.begin() + (block + 1) * blockSize,
              nzValues.begin() + dstIndex * blockSize);
    ++perRowBlockIndex[rowBlock];
  }

  auto csrMatrix =
      CSRMatrix<T>(nzValues, columnIndices, rowIndices,
                   {coo.getNumRowsInBlock(), coo.getNumColumnsInBlock()});
  canonicalizeCSR(csrMatrix);
  return csrMatrix;
}

} // namespace popsparse

#endif // _poplibs_popsparse_SparseStorageInternal_hpp_
