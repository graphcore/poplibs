// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Internally used functions for sparse storage representation

#ifndef _poplibs_popsparse_SparseStorageInternal_hpp_
#define _poplibs_popsparse_SparseStorageInternal_hpp_

#include "poplar/Interval.hpp"
#include "poplibs_support/logging.hpp"
#include "popsparse/SparseStorageFormats.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_map>

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

namespace dynamic {

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

} // namespace dynamic

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

  return CSCMatrix<T>(numRows, numColumns, std::move(nzValues),
                      std::move(columnIndices), std::move(rowIndices),
                      {csr.getNumRowsInBlock(), csr.getNumColumnsInBlock()});
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
      numRows, numColumns, std::move(cscMatrix.nzValues),
      std::move(cscMatrix.rowIndices), std::move(cscMatrix.columnIndices),
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
      numColumns, numRows, std::move(output.nzValues),
      std::move(output.rowIndices), std::move(output.columnIndices),
      {output.getNumColumnsInBlock(), output.getNumRowsInBlock()});
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
      CSRMatrix<T>(numRows, numColumns, nzValues, columnIndices, rowIndices,
                   {coo.getNumRowsInBlock(), coo.getNumColumnsInBlock()});
  canonicalizeCSR(csrMatrix);
  return csrMatrix;
}

static inline void
validateBlockChange(const std::array<std::size_t, 2> &oldBlockDimensions,
                    const std::array<std::size_t, 2> &newBlockDimensions) {

  auto validateChange = [](std::size_t oldDim, std::size_t newDim,
                           const std::string &str) {
    std::string errString = "Cannot change " + str + " from " +
                            std::to_string(oldDim) + " to " +
                            std::to_string(newDim);
    if (newDim > oldDim || oldDim % newDim) {
      throw poputil::poplibs_error(errString);
    }
  };
  validateChange(oldBlockDimensions[0], newBlockDimensions[0], "rows");
  validateChange(oldBlockDimensions[1], newBlockDimensions[1], "columns");
}

// Form a new CSR matrix with smaller block sizes given by newBlockDimensions
// from the passed CSR matrix.
template <class T>
CSRMatrix<T>
changeCSRBlockSize(const CSRMatrix<T> &csr,
                   const std::array<std::size_t, 2> &newBlockDimensions) {
  if (csr.getBlockDimensions() == newBlockDimensions) {
    return csr;
  }
  validateBlockChange(csr.getBlockDimensions(), newBlockDimensions);
  poplibs_support::logging::popsparse::debug(
      "Changing CSR block size "
      "[{},{}] ->[{},{}]",
      csr.getNumRowsInBlock(), csr.getNumColumnsInBlock(),
      newBlockDimensions.at(0), newBlockDimensions.at(1));
  const std::size_t subRowsPerRow =
      csr.getNumRowsInBlock() / newBlockDimensions[0];
  const std::size_t subColsPerCol =
      csr.getNumColumnsInBlock() / newBlockDimensions[1];
  const std::size_t oldBlockSize = csr.getBlockSize();

  std::vector<std::size_t> rowIndices;
  std::vector<std::size_t> colIndices;
  std::vector<T> nzValues;

  rowIndices.resize((csr.rowIndices.size() - 1) * subRowsPerRow + 1);
  colIndices.resize(csr.columnIndices.size() * subRowsPerRow * subColsPerCol);
  nzValues.resize(csr.nzValues.size());

  auto newRowIndicesIt = rowIndices.begin();
  auto newColIndicesIt = colIndices.begin();
  auto oldColIndicesIt = csr.columnIndices.begin();
  auto newNzValuesIt = nzValues.begin();
  for (auto oldRowIt = csr.rowIndices.begin();
       oldRowIt != csr.rowIndices.end() - 1; ++oldRowIt) {
    const auto numOldCols = (*std::next(oldRowIt) - *oldRowIt) / oldBlockSize;
    auto thisRowColBegin = oldColIndicesIt;
    auto thisRowColEnd = oldColIndicesIt + numOldCols;

    for (std::size_t subR = 0; subR != csr.getNumRowsInBlock();
         subR += newBlockDimensions[0]) {
      *newRowIndicesIt++ =
          *oldRowIt + subR * csr.getNumColumnsInBlock() * numOldCols;
      for (auto colIt = thisRowColBegin; colIt != thisRowColEnd; ++colIt) {
        const auto blockStartIndex =
            *oldRowIt + std::distance(thisRowColBegin, colIt) * oldBlockSize;
        for (std::size_t subC = 0; subC != csr.getNumColumnsInBlock();
             subC += newBlockDimensions[1]) {
          *newColIndicesIt++ = *colIt + subC;

          for (std::size_t r = 0; r != newBlockDimensions[0]; ++r) {
            auto src = csr.nzValues.begin() + blockStartIndex +
                       (subR + r) * csr.getNumColumnsInBlock() + subC;
            std::copy(src, src + newBlockDimensions[1], newNzValuesIt);
            newNzValuesIt += newBlockDimensions[1];
          }
        }
      }
    }
    oldColIndicesIt = thisRowColEnd;
  }
  assert(static_cast<std::size_t>(std::distance(
             colIndices.begin(), newColIndicesIt)) == colIndices.size());
  assert(static_cast<std::size_t>(std::distance(
             nzValues.begin(), newNzValuesIt)) == nzValues.size());
  rowIndices.back() = csr.rowIndices.back();
  return CSRMatrix<T>(nzValues, colIndices, rowIndices, newBlockDimensions);
}

// Form a new CSC matrix with smaller block sizes given by newBlockDimensions
// from the passed CSC matrix.
template <class T>
CSCMatrix<T>
changeCSCBlockSize(const CSCMatrix<T> &csc,
                   const std::array<std::size_t, 2> &newBlockDimensions) {
  validateBlockChange(csc.getBlockDimensions(), newBlockDimensions);
  if (csc.getBlockDimensions() == newBlockDimensions) {
    return csc;
  }

  poplibs_support::logging::popsparse::debug(
      "Changing CSC block size "
      "[{},{}] ->[{},{}]",
      csc.getNumRowsInBlock(), csc.getNumColumnsInBlock(),
      newBlockDimensions.at(0), newBlockDimensions.at(1));

  const std::size_t subRowsPerRow =
      csc.getNumRowsInBlock() / newBlockDimensions[0];
  const std::size_t subColsPerCol =
      csc.getNumColumnsInBlock() / newBlockDimensions[1];
  const std::size_t oldBlockSize = csc.getBlockSize();

  std::vector<std::size_t> rowIndices;
  std::vector<std::size_t> colIndices;
  std::vector<T> nzValues;

  colIndices.resize((csc.columnIndices.size() - 1) * subColsPerCol + 1);
  rowIndices.resize(csc.rowIndices.size() * subRowsPerRow * subColsPerCol);
  nzValues.resize(csc.nzValues.size());

  auto newColIndicesIt = colIndices.begin();
  auto newRowIndicesIt = rowIndices.begin();
  auto oldRowIndicesIt = csc.rowIndices.begin();
  auto newNzValuesIt = nzValues.begin();

  for (auto oldColIt = csc.columnIndices.begin();
       oldColIt != csc.columnIndices.end() - 1; ++oldColIt) {
    const auto numOldRows = (*std::next(oldColIt) - *oldColIt) / oldBlockSize;
    auto thisColRowBegin = oldRowIndicesIt;
    auto thisColRowEnd = oldRowIndicesIt + numOldRows;
    for (std::size_t subC = 0; subC != csc.getNumColumnsInBlock();
         subC += newBlockDimensions[1]) {
      *newColIndicesIt++ =
          *oldColIt + subC * csc.getNumRowsInBlock() * numOldRows;
      for (auto rowIt = thisColRowBegin; rowIt != thisColRowEnd; ++rowIt) {
        const auto blockStartIndex =
            *oldColIt + std::distance(thisColRowBegin, rowIt) * oldBlockSize;
        for (std::size_t subR = 0; subR != csc.getNumRowsInBlock();
             subR += newBlockDimensions[0]) {
          *newRowIndicesIt++ = *rowIt + subR;
          for (std::size_t c = 0; c != newBlockDimensions[1]; ++c) {
            auto src = csc.nzValues.begin() + blockStartIndex +
                       (subC + c) * csc.getNumRowsInBlock() + subR;
            std::copy(src, src + newBlockDimensions[0], newNzValuesIt);
            newNzValuesIt += newBlockDimensions[0];
          }
        }
      }
    }
    oldRowIndicesIt = thisColRowEnd;
  }

  assert(static_cast<std::size_t>(std::distance(
             rowIndices.begin(), newRowIndicesIt)) == rowIndices.size());
  assert(static_cast<std::size_t>(std::distance(
             nzValues.begin(), newNzValuesIt)) == nzValues.size());
  colIndices.back() = csc.columnIndices.back();
  return CSCMatrix<T>(nzValues, colIndices, rowIndices, newBlockDimensions);
}

// Form a new COO matrix with either smaller or larger block sizes given by
// newBlockDimensions.
template <class T>
COOMatrix<T>
changeCOOBlockSize(const COOMatrix<T> &coo,
                   const std::array<std::size_t, 2> &newBlockDimensions) {
  if (coo.getBlockDimensions() == newBlockDimensions) {
    return coo;
  }

  // For COO we allow stiching smaller blocks and making them into bigger
  if (coo.getNumRowsInBlock() <= newBlockDimensions.at(0) &&
      coo.getNumColumnsInBlock() <= newBlockDimensions.at(1)) {
    validateBlockChange(newBlockDimensions, coo.getBlockDimensions());

    poplibs_support::logging::popsparse::debug(
        "Changing COO block sizes down "
        "[{},{}] ->[{},{}]",
        newBlockDimensions.at(0), newBlockDimensions.at(1),
        coo.getNumRowsInBlock(), coo.getNumColumnsInBlock());

    const std::size_t subRowsPerRow =
        newBlockDimensions.at(0) / coo.getNumRowsInBlock();
    const std::size_t subColsPerCol =
        newBlockDimensions.at(1) / coo.getNumColumnsInBlock();
    const auto numNewIndices =
        coo.columnIndices.size() / (subRowsPerRow * subColsPerCol);

    assert(coo.columnIndices.size() % (subRowsPerRow * subColsPerCol) == 0);
    const std::size_t newBlockSize =
        newBlockDimensions.at(0) * newBlockDimensions.at(1);
    std::vector<T> nzValues;
    nzValues.resize(coo.nzValues.size());
    std::vector<std::size_t> subblockValidity(numNewIndices);
    std::size_t indexOfNewBlock = 0;

    // guarantee that the mapping of (row,col) <-> indices is bijective.
    // The other option is to use an unordered_map of a pair -> index.
    // We also use the mapping to check if number of sub-blocks in a block
    // are present.
    std::size_t maxColElement =
        *std::max_element(coo.columnIndices.begin(), coo.columnIndices.end());
    maxColElement = (maxColElement / newBlockDimensions.at(1) + 1) *
                    newBlockDimensions.at(1);

    auto rowColToMapIndex = [&](std::size_t row, std::size_t col) {
      return row * maxColElement + col;
    };
    auto mapIndexToRowCol = [&](std::size_t index) {
      return std::make_pair(index / maxColElement, index % maxColElement);
    };

    // keeps track of where the blocks of the new blocks are mapped to in the
    // new NZ values vector.
    std::unordered_map<std::size_t, std::size_t> blockCoordToNzIndex;
    auto oldNzIt = coo.nzValues.begin();

    for (auto oldRowIt = coo.rowIndices.begin(),
              oldColIt = coo.columnIndices.begin();
         oldRowIt != coo.rowIndices.end(); ++oldRowIt, ++oldColIt) {
      const auto newRow =
          *oldRowIt / newBlockDimensions.at(0) * newBlockDimensions.at(0);
      const auto newCol =
          *oldColIt / newBlockDimensions.at(1) * newBlockDimensions.at(1);
      const auto entry = blockCoordToNzIndex.insert(
          {rowColToMapIndex(newRow, newCol), indexOfNewBlock});
      std::size_t blockStart = entry.first->second * newBlockSize;
      indexOfNewBlock += entry.second;
      auto rowOffset = *oldRowIt % newBlockDimensions.at(0);
      auto colOffset = *oldColIt % newBlockDimensions.at(1);

      subblockValidity.at(entry.first->second) +=
          rowColToMapIndex(rowOffset, colOffset);

      auto newNzIt = nzValues.begin() + blockStart + colOffset;
      for (std::size_t subR = 0; subR != coo.getNumRowsInBlock();
           ++subR, oldNzIt += coo.getNumColumnsInBlock()) {
        auto dst = newNzIt + (rowOffset + subR) * newBlockDimensions.at(1);
        std::copy(oldNzIt, oldNzIt + coo.getNumColumnsInBlock(), dst);
      }
    }

    const auto numCoords = blockCoordToNzIndex.size();
    std::vector<std::pair<std::size_t, std::size_t>> blockCoordPair;
    blockCoordPair.resize(numCoords);
    std::size_t index = 0;
    for (const auto &p : blockCoordToNzIndex) {
      blockCoordPair.at(index++) = p;
    }

    assert(numCoords == numNewIndices);
    // Check that all subblocks with the new bigger blocks are present
    const auto subblockValidityValue =
        subRowsPerRow * subColsPerCol *
        ((subRowsPerRow - 1) * maxColElement * coo.getNumRowsInBlock() +
         (subColsPerCol - 1) * coo.getNumColumnsInBlock()) /
        2;
    const auto validNumSubBlocks =
        std::all_of(subblockValidity.begin(), subblockValidity.end(),
                    [&](std::size_t a) { return a == subblockValidityValue; });
    if (!validNumSubBlocks) {
      throw poputil::poplibs_error("All sublocks not present in COO block "
                                   "change");
    }

    // we need row columns as stored in the new NZ values, hence sort based
    // on the start index into it.
    std::sort(blockCoordPair.begin(), blockCoordPair.end(),
              [](const std::pair<std::size_t, std::size_t> &a,
                 const std::pair<std::size_t, std::size_t> &b) {
                return a.second < b.second;
              });

    std::vector<std::size_t> rowIndices, columnIndices;
    rowIndices.resize(numCoords);
    columnIndices.resize(numCoords);

    for (auto it = blockCoordPair.begin(); it != blockCoordPair.end(); ++it) {
      auto p = mapIndexToRowCol(it->first);
      auto index = std::distance(blockCoordPair.begin(), it);
      rowIndices[index] = p.first;
      columnIndices[index] = p.second;
    }
    return COOMatrix<T>(nzValues, columnIndices, rowIndices,
                        newBlockDimensions);
  } else {
    validateBlockChange(coo.getBlockDimensions(), newBlockDimensions);
    poplibs_support::logging::popsparse::debug(
        "Changing COO block sizes up "
        "[{},{}] ->[{},{}]",
        coo.getNumRowsInBlock(), coo.getNumColumnsInBlock(),
        newBlockDimensions.at(0), newBlockDimensions.at(1));

    const std::size_t subRowsPerRow =
        coo.getNumRowsInBlock() / newBlockDimensions.at(0);
    const std::size_t subColsPerCol =
        coo.getNumColumnsInBlock() / newBlockDimensions.at(1);
    const std::size_t oldBlockSize = coo.getBlockSize();

    std::vector<std::size_t> rowIndices;
    std::vector<std::size_t> colIndices;
    std::vector<T> nzValues;

    rowIndices.resize(coo.rowIndices.size() * subRowsPerRow * subColsPerCol);
    colIndices.resize(coo.columnIndices.size() * subRowsPerRow * subColsPerCol);
    nzValues.resize(coo.nzValues.size());

    auto newRowIndicesIt = rowIndices.begin();
    auto newColIndicesIt = colIndices.begin();
    auto newNzValuesIt = nzValues.begin();
    assert(coo.rowIndices.size() == coo.columnIndices.size());
    for (auto oldRowIt = coo.rowIndices.begin(),
              oldColIt = coo.columnIndices.begin();
         oldRowIt != coo.rowIndices.end(); ++oldRowIt, ++oldColIt) {
      const auto blockStart =
          std::distance(coo.rowIndices.begin(), oldRowIt) * oldBlockSize;
      for (std::size_t subR = 0; subR != coo.getNumRowsInBlock();
           subR += newBlockDimensions.at(0)) {
        for (std::size_t subC = 0; subC != coo.getNumColumnsInBlock();
             subC += newBlockDimensions.at(1)) {
          *newRowIndicesIt++ = *oldRowIt + subR;
          *newColIndicesIt++ = *oldColIt + subC;
          for (std::size_t r = 0; r != newBlockDimensions.at(0); ++r) {
            auto src = coo.nzValues.begin() + blockStart +
                       (subR + r) * coo.getNumColumnsInBlock() + subC;
            std::copy(src, src + newBlockDimensions.at(1), newNzValuesIt);
            newNzValuesIt += newBlockDimensions.at(1);
          }
        }
      }
    }
    assert(static_cast<std::size_t>(std::distance(
               rowIndices.begin(), newRowIndicesIt)) == rowIndices.size());
    assert(static_cast<std::size_t>(std::distance(
               colIndices.begin(), newColIndicesIt)) == colIndices.size());
    assert(static_cast<std::size_t>(std::distance(
               nzValues.begin(), newNzValuesIt)) == coo.nzValues.size());
    return COOMatrix<T>(nzValues, colIndices, rowIndices, newBlockDimensions);
  }
}

} // namespace popsparse

#endif // _poplibs_popsparse_SparseStorageInternal_hpp_
