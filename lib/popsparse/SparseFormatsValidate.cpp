// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "SparseFormatsValidate.hpp"
#include "poputil/exceptions.hpp"

static void isMultipleOf(const std::vector<std::size_t> &indices,
                         std::size_t blockSize, const std::string &str) {
  for (auto it = indices.begin(); it != indices.end(); ++it) {
    if (*it % blockSize) {
      const auto index = std::distance(indices.begin(), it);
      throw poputil::poplibs_error(str + std::to_string(*it) + " at index " +
                                   std::to_string(index) +
                                   " is not a multiple of block size");
    }
  }
}

static void isLessThan(const std::vector<std::size_t> &indices,
                       std::size_t bound, const std::string &str) {
  for (auto it = indices.begin(); it != indices.end(); ++it) {
    if (*it >= bound) {
      const auto index = std::distance(indices.begin(), it);
      throw poputil::poplibs_error(str + std::to_string(*it) + " at index " +
                                   std::to_string(index) + " exceeds bound");
    }
  }
}

static void commonChecks(unsigned numRows, unsigned numColumns,
                         const std::array<std::size_t, 2> &blockDimensions,
                         const std::vector<std::size_t> &rowIndices,
                         const std::vector<std::size_t> &columnIndices,
                         const std::string &baseString) {
  if (numRows % blockDimensions.at(0)) {
    throw poputil::poplibs_error(baseString +
                                 " number of rows must be "
                                 "an integral multiple of the rows in a block");
  }

  if (numColumns % blockDimensions.at(1)) {
    throw poputil::poplibs_error(
        baseString + "number of columns must be "
                     "an integral multiple of the columns in a block");
  }
}
namespace popsparse {

void validateCSR(unsigned numRows, unsigned numColumns,
                 const std::array<std::size_t, 2> &blockDimensions,
                 const std::size_t numNzValues,
                 const std::vector<std::size_t> &rowIndices,
                 const std::vector<std::size_t> &columnIndices) {
  const std::string &baseString = "CSR Validation failed: ";
  commonChecks(numRows, numColumns, blockDimensions, rowIndices, columnIndices,
               baseString);

  if (rowIndices.size() != numRows / blockDimensions.at(0) + 1) {
    throw poputil::poplibs_error(baseString +
                                 "number of row indices must be "
                                 "equal to 1 more than number of block rows");
  }

  if (numNzValues != rowIndices.back()) {
    throw poputil::poplibs_error(baseString + "number of NZ values must match "
                                              "last entry in the row indices");
  }
  const std::size_t elemsPerBlock =
      blockDimensions.at(0) * blockDimensions.at(1);
  isMultipleOf(columnIndices, blockDimensions.at(1),
               baseString + " column indices :");
  isMultipleOf(rowIndices, elemsPerBlock, baseString + " row indices :");
  isLessThan(columnIndices, numColumns, baseString + " column indices :");
}

void validateCSC(unsigned numRows, unsigned numColumns,
                 const std::array<std::size_t, 2> &blockDimensions,
                 const std::size_t numNzValues,
                 const std::vector<std::size_t> &rowIndices,
                 const std::vector<std::size_t> &columnIndices) {
  const std::string &baseString = "CSC Validation failed: ";
  commonChecks(numRows, numColumns, blockDimensions, rowIndices, columnIndices,
               baseString);

  if (columnIndices.size() != numColumns / blockDimensions.at(0) + 1) {
    throw poputil::poplibs_error(baseString +
                                 "number of column indices must "
                                 "be equal to 1 more than number of block "
                                 "columns");
  }

  if (numNzValues != columnIndices.back()) {
    throw poputil::poplibs_error(baseString +
                                 "number of NZ values must match "
                                 "last entry in the column indices");
  }
  isLessThan(rowIndices, numRows, baseString + " row indices :");
  const std::size_t elemsPerBlock =
      blockDimensions.at(0) * blockDimensions.at(1);
  isMultipleOf(rowIndices, blockDimensions.at(0),
               baseString + " row indices :");
  isMultipleOf(columnIndices, elemsPerBlock, baseString + " column indices :");
  isLessThan(rowIndices, numRows, baseString + " row indices :");
}

void validateCOO(unsigned numRows, unsigned numColumns,
                 const std::array<std::size_t, 2> &blockDimensions,
                 const std::size_t numNzValues,
                 const std::vector<std::size_t> &rowIndices,
                 const std::vector<std::size_t> &columnIndices) {
  const std::string &baseString = "COO Validation failed: ";
  commonChecks(numRows, numColumns, blockDimensions, rowIndices, columnIndices,
               baseString);
  const std::size_t elemsPerBlock =
      blockDimensions.at(0) * blockDimensions.at(1);

  if (columnIndices.size() != rowIndices.size()) {
    throw poputil::poplibs_error(baseString +
                                 "number of column indices must "
                                 "be equal to number of row entries");
  }

  if (columnIndices.size() * elemsPerBlock != numNzValues) {
    throw poputil::poplibs_error(baseString +
                                 "product of number of indices "
                                 "and block size must be equal to number of "
                                 "nz values");
  }
  isMultipleOf(rowIndices, blockDimensions.at(0),
               baseString + " row indices :");
  isMultipleOf(columnIndices, blockDimensions.at(1),
               baseString + " column indices :");
  isLessThan(rowIndices, numRows, baseString + " row indices :");
  isLessThan(columnIndices, numColumns, baseString + " row indices :");
}

} // namespace popsparse
