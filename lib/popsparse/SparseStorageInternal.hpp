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
template <typename T> struct RowPositionValues {
  std::size_t rowNumber;
  std::vector<std::pair<std::size_t, T>> positionValues;

  RowPositionValues(
      std::size_t rowNumber,
      const std::vector<std::pair<std::size_t, T>> &positionValues)
      : rowNumber(rowNumber), positionValues(positionValues) {}

  friend bool operator>(const RowPositionValues &a,
                        const RowPositionValues &b) {
    return a.positionValues.size() > b.positionValues.size();
  }
};

// The row and column dimensions are divided by straight lines creating
// rectangular tiles. A tile is then given by a right open interval for
// the row and column dimension.
template <typename T> struct TilePartition {
  // indices of the tile in the X, Y, Z splits
  TileIndex tileIndex;
  // actual row and column intervals
  Tile tile;
  std::vector<RowPositionValues<T>> tileInfo;

  TilePartition() = default;
  TilePartition(const TilePartition &) = default;

  TilePartition(const TileIndex &tileIndex, const Tile &tile,
                const std::vector<RowPositionValues<T>> &tileInfo_)
      : tileIndex(tileIndex), tile(tile) {
    tileInfo = tileInfo_;

    // keep sorted so that it is easy to remove rows which are the smallest
    // first
    std::sort(tileInfo.begin(), tileInfo.end(),
              [](const RowPositionValues<T> &a, const RowPositionValues<T> &b) {
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
template <typename T> struct PNBucket {
  std::size_t metaInfoElements;
  std::size_t numNzElements;

  std::vector<TilePartition<T>> subGroups;
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

// Convert from a CSR representation of a matrix of dimension
// [numRows x numColumns] to a CSC representation.
// x§
// \param  numRows    Number of rows in the matrix.
// \param  numColumns Number of columns in the matrix.
// \param  csr        CSR representation of matrix .
//
// \return CSC representation of the matrix.
// Position and value pairs for each row
template <class T>
CSCMatrix<T> csrToCSC(std::size_t numRows, std::size_t numColumns,
                      const CSRMatrix<T> &csr) {

  // number of NZ values are always equal to the number of column indices.
  const std::size_t numNZValues = csr.columnIndices.size();

  if (csr.rowIndices.back() != numNZValues) {
    throw poputil::poplibs_error("Number of non-zero values do not match last "
                                 "entry on rowIndices");
  }

  std::vector<std::size_t> columnIndices(numColumns + 1);
  for (std::size_t nz = 0; nz != numNZValues; ++nz) {
    columnIndices[csr.columnIndices[nz] + 1]++;
  }
  std::partial_sum(columnIndices.begin(), columnIndices.end(),
                   columnIndices.begin());
  // The last entry in the CSC columns is always the number of non zero entries
  columnIndices[numColumns] = numNZValues;

  std::vector<std::size_t> rowIndices;
  rowIndices.resize(numNZValues);
  std::vector<T> nzValues;
  nzValues.resize(numNZValues);

  // Fill in the non zero entries and the row indices
  for (std::size_t row = 0; row != numRows; ++row) {
    for (std::size_t csrRow = csr.rowIndices[row];
         csrRow != csr.rowIndices[row + 1]; ++csrRow) {
      auto column = csr.columnIndices[csrRow];
      auto dstRow = columnIndices[column];
      rowIndices[dstRow] = row;
      nzValues[dstRow] = csr.nzValues[csrRow];
      ++columnIndices[column];
    }
  }

  std::size_t startIndex = 0;
  for (std::size_t column = 0; column != numColumns; ++column) {
    std::swap(columnIndices[column], startIndex);
  }

  return CSCMatrix<T>(std::move(nzValues), std::move(columnIndices),
                      std::move(rowIndices));
}

// Sort the columns of a CSR matrix in a given row to be in increasing order.
// The operation is done inplace.
//
// \param  csr  The matrix in csr representation to be canonicalized
template <class T> void canonicalizeCSR(CSRMatrix<T> &matrix) {
  const auto numRows = matrix.rowIndices.size() - 1;
  using TP = std::pair<std::size_t, T>;
  std::vector<TP> columns;

  for (std::size_t row = 0; row != numRows; ++row) {
    const auto startIndex = matrix.rowIndices[row];
    const auto endIndex = matrix.rowIndices[row + 1];
    const auto numElems = endIndex - startIndex;

    columns.resize(numElems);

    for (std::size_t index = startIndex; index != endIndex; ++index) {
      columns[index - startIndex] =
          std::make_pair(matrix.columnIndices[index], matrix.nzValues[index]);
    }

    std::sort(columns.begin(), columns.end(),
              [](const TP &a, const TP &b) { return a.first < b.first; });

    for (std::size_t index = startIndex; index != endIndex; ++index) {
      matrix.columnIndices[index] = columns[index - startIndex].first;
      matrix.nzValues[index] = columns[index - startIndex].second;
    }
  }
}

// Convert a CSC representation of a matrix of dimension [numRows x numColumns]
// to a CSR representation.
template <class T>
CSRMatrix<T> cscToCSR(std::size_t numRows, std::size_t numColumns,
                      const CSCMatrix<T> &input) {

  auto csrMatrix =
      CSRMatrix<T>(input.nzValues, input.rowIndices, input.columnIndices);

  auto cscMatrix = csrToCSC<T>(numColumns, numRows, csrMatrix);

  return CSRMatrix<T>(std::move(cscMatrix.nzValues),
                      std::move(cscMatrix.rowIndices),
                      std::move(cscMatrix.columnIndices));
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
  return CSRMatrix<T>(output.nzValues, output.rowIndices, output.columnIndices);
}

// Given a tile, return the information for every row containing non-zero values
// in it.
// Each row is a vector of <column index, non-zero value> which meets the
// tile boundaries. The rows and columns within the tile are offsets within
// the tile and their absolute positions can be obtained from the tile
// information stored as part of the partition.
//
// \param  csr       The CSR representation of the sparse matrix
// \param  tile      The row and column intervals for the tile
//
// A vector for each row in the row interval range containing (column, value)
// pairs.
template <typename T>
std::vector<RowPositionValues<T>>
getPositionValuePairsPerRow(const CSRMatrix<T> &csr, const Tile &tile) {
  const auto startRow = tile.getRows().begin();
  const auto endRow = tile.getRows().end();
  const auto startColumn = tile.getColumns().begin();
  const auto endColumn = tile.getColumns().end();

  if (startRow >= csr.rowIndices.size()) {
    throw poputil::poplibs_error("Start row in tile doesn't match information "
                                 "in CSR");
  }

  if (endRow >= csr.rowIndices.size()) {
    throw poputil::poplibs_error("End row in tile doesn't match information "
                                 "in CSR");
  }

  std::vector<RowPositionValues<T>> rowValuePairs;

  for (auto row = startRow; row != endRow; ++row) {
    std::vector<std::pair<std::size_t, T>> valuePairs;
    for (auto column = csr.rowIndices[row]; column != csr.rowIndices[row + 1];
         ++column) {
      // This can be optimised if the columns are always sorted in increasing
      // order.
      if (csr.columnIndices[column] >= startColumn &&
          csr.columnIndices[column] < endColumn) {
        valuePairs.emplace_back(csr.columnIndices[column] - startColumn,
                                csr.nzValues[column]);
      }
    }
    if (!valuePairs.empty()) {
      rowValuePairs.emplace_back(
          RowPositionValues<T>(row - startRow, std::move(valuePairs)));
    }
  }
  return rowValuePairs;
}

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

  if (numNZValues > numRows * numColumns) {
    throw poputil::poplibs_error("Number of non-zero values in COO exceed the "
                                 "size of the matrix");
  }

  if (coo.rowIndices.size() != numNZValues) {
    throw poputil::poplibs_error("Number of non-zero values does not match "
                                 "number of row elements in the COO");
  }

  if (coo.columnIndices.size() != numNZValues) {
    throw poputil::poplibs_error("Number of non-zero values does not match "
                                 "number of column elements in the COO");
  }

  std::vector<std::size_t> rowIndices(numRows + 1);
  for (const auto &row : coo.rowIndices) {
    ++rowIndices.at(row + 1);
  }

  std::partial_sum(std::next(rowIndices.begin()), rowIndices.end(),
                   std::next(rowIndices.begin()));

  std::vector<std::size_t> countPerRow(numRows);

  std::vector<std::size_t> columnIndices;
  columnIndices.resize(numNZValues);
  std::vector<T> nzValues;
  nzValues.resize(numNZValues);

  for (std::size_t elem = 0; elem < numNZValues; ++elem) {
    // The bounds on row elements is already done. Directly index into vector.
    const auto row = coo.rowIndices[elem];
    const auto dstIndex = rowIndices[row] + countPerRow[row];
    columnIndices[dstIndex] = coo.columnIndices[elem];
    nzValues[dstIndex] = coo.nzValues[elem];
    ++countPerRow[row];
  }

  auto csrMatrix = CSRMatrix<T>(nzValues, columnIndices, rowIndices);
  canonicalizeCSR(csrMatrix);
  return csrMatrix;
}

} // namespace popsparse

#endif // _poplibs_popsparse_SparseStorageInternal_hpp_
