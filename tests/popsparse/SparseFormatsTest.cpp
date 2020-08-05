// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseFormatsTest
#include "../lib/popsparse/SparseStorageInternal.hpp"
#include <boost/test/unit_test.hpp>

// CSR/CSC representation of matrix for element sparsity
//    10   20    0    0    0    0
//     0   30    0   40    0    0
//     0    0   50   60   70    0
//     0    0    0    0    0   80

// CSR/CSC representation of matrix for block sparsity with 2x3 blocks
//    10 11 12    20 21 22    0  0  0     0  0  0     0   0  0     0  0  0
//    13 14 15    23 24 25    0  0  0     0  0  0     0   0  0     0  0  0
//
//     0  0  0    30 31 32    0  0  0    40 41 42     0   0  0     0  0  0
//     0  0  0    33 34 35    0  0  0    43 44 45     0   0  0     0  0  0
//
//     0  0  0    0  0  0    50 51 52    60 61 62     70 71 72     0  0  0
//     0  0  0    0  0  0    53 54 55    63 64 65     73 74 75     0  0  0
//
//     0  0  0    0  0  0    0   0  0     0  0  0      0  0  0    80 81 82
//     0  0  0    0  0  0    0   0  0     0  0  0      0  0  0    83 84 85

const std::size_t numRows = 4;
const std::size_t numColumns = 6;
const std::size_t numRowsInBlock = 2;
const std::size_t numColumnsInBlock = 3;

static const popsparse::CSRMatrix<double>
    csrRef({10.0, 20, 30, 40, 50, 60, 70, 80.0}, {0, 1, 1, 3, 2, 3, 4, 5},
           {0, 2, 4, 7, 8});

const popsparse::CSRMatrix<double> csrRef2x3(
    {10.0, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24,   25, 30, 31, 32, 33,
     34,   35, 40, 41, 42, 43, 44, 45, 50, 51, 52,   53, 54, 55, 60, 61,
     62,   63, 64, 65, 70, 71, 72, 73, 74, 75, 80.0, 81, 82, 83, 84, 85},
    {0, 3, 3, 9, 6, 9, 12, 15}, {0, 12, 24, 42, 48}, {2, 3});

static const popsparse::CSRMatrix<std::size_t>
    csrRefS({10, 20, 30, 40, 50, 60, 70, 80}, {0, 1, 1, 3, 2, 3, 4, 5},
            {0, 2, 4, 7, 8});

static const popsparse::CSCMatrix<double>
    cscRef({10, 20, 30, 50, 40, 60, 70, 80}, {0, 1, 3, 4, 6, 7, 8},
           {0, 0, 1, 2, 1, 2, 2, 3});

static const popsparse::CSCMatrix<double>
    cscRef2x3({10, 13, 11, 14, 12, 15, 20, 23, 21, 24, 22, 25, 30, 33, 31, 34,
               32, 35, 50, 53, 51, 54, 52, 55, 40, 43, 41, 44, 42, 45, 60, 63,
               61, 64, 62, 65, 70, 73, 71, 74, 72, 75, 80, 83, 81, 84, 82, 85},
              {0, 6, 18, 24, 36, 42, 48}, {0, 0, 2, 4, 2, 4, 4, 6}, {2, 3});

static const popsparse::CSRMatrix<double>
    csrRefUnsorted({10.0, 20, 40, 30, 70, 50, 60, 80.0},
                   {0, 1, 3, 1, 4, 2, 3, 5}, {0, 2, 4, 7, 8});

static const popsparse::CSRMatrix<double> csrRefUnsorted2x3(
    {10.0, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24,   25, 40, 41, 42, 43,
     44,   45, 30, 31, 32, 33, 34, 35, 70, 71, 72,   73, 74, 75, 50, 51,
     52,   53, 54, 55, 60, 61, 62, 63, 64, 65, 80.0, 81, 82, 83, 84, 85},
    {0, 3, 9, 3, 12, 6, 9, 15}, {0, 12, 24, 42, 48}, {2, 3});

static const popsparse::COOMatrix<double>
    cooUnsorted({80, 70, 60, 50, 40, 30, 20, 10}, {5, 4, 3, 2, 3, 1, 1, 0},
                {3, 2, 2, 2, 1, 1, 0, 0});

static const popsparse::COOMatrix<double> cooUnsorted2x3(
    {80, 81, 82, 83, 84, 85, 70, 71, 72, 73, 74, 75, 60, 61, 62, 63,
     64, 65, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 30, 31,
     32, 33, 34, 35, 20, 21, 22, 23, 24, 25, 10, 11, 12, 13, 14, 15},
    {15, 12, 9, 6, 9, 3, 3, 0}, {6, 4, 4, 4, 2, 2, 0, 0});

BOOST_AUTO_TEST_CASE(ValidateCsrToCsc) {
  auto cscResult = popsparse::csrToCSC(numRows, numColumns, csrRef);
  BOOST_CHECK_EQUAL_COLLECTIONS(cscRef.nzValues.begin(), cscRef.nzValues.end(),
                                cscResult.nzValues.begin(),
                                cscResult.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cscRef.rowIndices.begin(), cscRef.rowIndices.end(),
      cscResult.rowIndices.begin(), cscResult.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cscRef.columnIndices.begin(), cscRef.columnIndices.end(),
      cscResult.columnIndices.begin(), cscResult.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateCsrToCscBlock) {
  auto cscResult = popsparse::csrToCSC(
      numRows * numRowsInBlock, numColumns * numColumnsInBlock, csrRef2x3);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      cscRef2x3.nzValues.begin(), cscRef2x3.nzValues.end(),
      cscResult.nzValues.begin(), cscResult.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cscRef2x3.rowIndices.begin(), cscRef2x3.rowIndices.end(),
      cscResult.rowIndices.begin(), cscResult.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cscRef2x3.columnIndices.begin(), cscRef2x3.columnIndices.end(),
      cscResult.columnIndices.begin(), cscResult.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateCscToCsr) {
  auto csrResult = popsparse::cscToCSR(numRows, numColumns, cscRef);
  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef.nzValues.begin(), csrRef.nzValues.end(),
                                csrResult.nzValues.begin(),
                                csrResult.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef.rowIndices.begin(), csrRef.rowIndices.end(),
      csrResult.rowIndices.begin(), csrResult.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef.columnIndices.begin(), csrRef.columnIndices.end(),
      csrResult.columnIndices.begin(), csrResult.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateCscToCsrBlock) {
  auto csrResult = popsparse::cscToCSR(
      numRows * numRowsInBlock, numColumns * numColumnsInBlock, cscRef2x3);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.nzValues.begin(), csrRef2x3.nzValues.end(),
      csrResult.nzValues.begin(), csrResult.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.rowIndices.begin(), csrRef2x3.rowIndices.end(),
      csrResult.rowIndices.begin(), csrResult.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.columnIndices.begin(), csrRef2x3.columnIndices.end(),
      csrResult.columnIndices.begin(), csrResult.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateTransposeCSR) {
  auto csrRes1 = popsparse::csrTranspose(numRows, numColumns, csrRef);
  auto csrRes2 = popsparse::csrTranspose(numColumns, numRows, csrRes1);
  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef.nzValues.begin(), csrRef.nzValues.end(),
                                csrRes2.nzValues.begin(),
                                csrRes2.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef.rowIndices.begin(), csrRef.rowIndices.end(),
      csrRes2.rowIndices.begin(), csrRes2.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef.columnIndices.begin(), csrRef.columnIndices.end(),
      csrRes2.columnIndices.begin(), csrRes2.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateTransposeCSRBlock) {
  auto csrRes1 = popsparse::csrTranspose(
      numRows * numRowsInBlock, numColumns * numColumnsInBlock, csrRef2x3);
  auto csrRes2 = popsparse::csrTranspose(numColumns * numColumnsInBlock,
                                         numRows * numRowsInBlock, csrRes1);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.nzValues.begin(), csrRef2x3.nzValues.end(),
      csrRes2.nzValues.begin(), csrRes2.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.rowIndices.begin(), csrRef2x3.rowIndices.end(),
      csrRes2.rowIndices.begin(), csrRes2.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.columnIndices.begin(), csrRef2x3.columnIndices.end(),
      csrRes2.columnIndices.begin(), csrRes2.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateCSRCanonicalize) {
  auto csr = csrRefUnsorted;
  popsparse::canonicalizeCSR(csr);
  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef.nzValues.begin(), csrRef.nzValues.end(),
                                csr.nzValues.begin(), csr.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef.rowIndices.begin(),
                                csrRef.rowIndices.end(), csr.rowIndices.begin(),
                                csr.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef.columnIndices.begin(), csrRef.columnIndices.end(),
      csr.columnIndices.begin(), csr.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ValidateCSRCanonicalizeBlock) {
  auto csr = csrRefUnsorted2x3;
  popsparse::canonicalizeCSR(csr);
  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef2x3.nzValues.begin(),
                                csrRef2x3.nzValues.end(), csr.nzValues.begin(),
                                csr.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef2x3.rowIndices.begin(),
                                csrRef2x3.rowIndices.end(),
                                csr.rowIndices.begin(), csr.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x3.columnIndices.begin(), csrRef2x3.columnIndices.end(),
      csr.columnIndices.begin(), csr.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(GetRowPositionTest) {

  const auto tile =
      popsparse::Tile(poplar::Interval(0, 2), poplar::Interval(0, 4));
  const std::size_t blockSizeX = 1, blockSizeY = 1;
  const auto rowInfo = popsparse::getPositionValuePairsPerRow(
      csrRefS, blockSizeX, blockSizeY, tile);

  const std::vector<std::vector<std::pair<double, std::size_t>>> expectedInfo =
      {{{0, 10}, {1, 20}}, {{1, 30}, {3, 40}}};

  BOOST_CHECK_EQUAL(rowInfo.size(), expectedInfo.size());

  for (unsigned row = 0; row != expectedInfo.size(); ++row) {
    BOOST_CHECK_EQUAL(rowInfo[row].positionValues.size(),
                      expectedInfo[row].size());
    const auto expt = expectedInfo[row];
    const auto real = rowInfo[row].positionValues;
    for (unsigned column = 0; column != expectedInfo[row].size(); ++column) {
      BOOST_CHECK_EQUAL(real[column].first, expt[column].first);
      BOOST_CHECK_EQUAL(real[column].second, expt[column].second);
    }
    BOOST_CHECK_EQUAL(rowInfo[row].rowNumber, row);
  }
}

BOOST_AUTO_TEST_CASE(ConvertCooToCsr) {
  auto csr = popsparse::cooToCSR(numRows, numColumns, cooUnsorted);

  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef.nzValues.begin(), csrRef.nzValues.end(),
                                csr.nzValues.begin(), csr.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef.rowIndices.begin(),
                                csrRef.rowIndices.end(), csr.rowIndices.begin(),
                                csr.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef.columnIndices.begin(), csrRef.columnIndices.end(),
      csr.columnIndices.begin(), csr.columnIndices.end());
}
