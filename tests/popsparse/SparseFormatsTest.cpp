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

// CSR/CSC/COO representation of matrix for block sparsity with 2x6 blocks
//    0  1  2  3  4  5     0  0  0  0  0  0   20 21 22 23 24 25
//    6  7  8  9 10 11     0  0  0  0  0  0   26 27 28 29 30 31
//
//    0  0  0  0  0  0    40 41 42 43 44 45    0  0  0  0  0  0
//    0  0  0  0  0  0    46 47 48 49 50 51    0  0  0  0  0  0
//
//   50 51 52 53 54 55    70 71 72 73 74 75   80 81 82 83 84 85
//   56 57 58 59 60 61    76 77 78 79 80 81   86 87 88 89 90 91
static const popsparse::CSRMatrix<std::size_t> csrRef2x6(
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
     50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
     76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
    {0, 12, 6, 0, 6, 12}, {0, 24, 36, 72}, {2, 6});

static const popsparse::COOMatrix<std::size_t> cooRef2x6(
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
     50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
     76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
    {0, 12, 6, 0, 6, 12}, {0, 0, 2, 4, 4, 4}, {2, 6});

static const popsparse::CSRMatrix<std::size_t> csrRef2x6_1x2(
    {0,  1,  2,  3,  4,  5,  20, 21, 22, 23, 24, 25, 6,  7,  8,  9,  10, 11,
     26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
     50, 51, 52, 53, 54, 55, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 84, 85,
     56, 57, 58, 59, 60, 61, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 90, 91},
    {0, 2, 4, 12, 14, 16, 0,  2,  4,  12, 14, 16, 6, 8, 10, 6,  8,  10,
     0, 2, 4, 6,  8,  10, 12, 14, 16, 0,  2,  4,  6, 8, 10, 12, 14, 16},
    {0, 12, 24, 30, 36, 54, 72}, {1, 2});

static const popsparse::COOMatrix<std::size_t> cooRef2x6_1x2(
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
     50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
     76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
    {0, 2, 4, 0, 2, 4, 12, 14, 16, 12, 14, 16, 6,  8,  10, 6,  8,  10,
     0, 2, 4, 0, 2, 4, 6,  8,  10, 6,  8,  10, 12, 14, 16, 12, 14, 16},
    {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
     4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5},
    {1, 2});

// CSR/CSC with block size of 4x4
//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
// 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
// 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
// 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
static const popsparse::CSRMatrix<std::size_t>
    csrRef4x4({0,  1,  2,  3,  16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51,
               4,  5,  6,  7,  20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55,
               8,  9,  10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59,
               12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63},
              {0, 4, 8, 12}, {0, 64}, {4, 4});

static const popsparse::CSRMatrix<std::size_t> csrRef4x4_2x2(
    {0,  1,  16, 17, 2,  3,  18, 19, 4,  5,  20, 21, 6,  7,  22, 23,
     8,  9,  24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31,
     32, 33, 48, 49, 34, 35, 50, 51, 36, 37, 52, 53, 38, 39, 54, 55,
     40, 41, 56, 57, 42, 43, 58, 59, 44, 45, 60, 61, 46, 47, 62, 63},
    {0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14}, {0, 32, 64},
    {2, 2});

static const popsparse::COOMatrix<std::size_t>
    cooRef4x4({0,  1,  2,  3,  16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51,
               4,  5,  6,  7,  20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55,
               8,  9,  10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59,
               12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63},
              {0, 4, 8, 12}, {0, 0, 0, 0}, {4, 4});

static const popsparse::COOMatrix<std::size_t> cooRef4x4_2x2(

    {0,  1,  16, 17, 2,  3,  18, 19, 32, 33, 48, 49, 34, 35, 50, 51,
     4,  5,  20, 21, 6,  7,  22, 23, 36, 37, 52, 53, 38, 39, 54, 55,
     8,  9,  24, 25, 10, 11, 26, 27, 40, 41, 56, 57, 42, 43, 58, 59,
     12, 13, 28, 29, 14, 15, 30, 31, 44, 45, 60, 61, 46, 47, 62, 63},
    {0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14, 12, 14},
    {0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2}, {2, 2});

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

BOOST_AUTO_TEST_CASE(ChangeCsrBlockSize2x6) {
  auto csr = popsparse::changeCSRBlockSize(csrRef2x6, {1, 2});
  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef2x6_1x2.nzValues.begin(),
                                csrRef2x6_1x2.nzValues.end(),
                                csr.nzValues.begin(), csr.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef2x6_1x2.rowIndices.begin(),
                                csrRef2x6_1x2.rowIndices.end(),
                                csr.rowIndices.begin(), csr.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef2x6_1x2.columnIndices.begin(), csrRef2x6_1x2.columnIndices.end(),
      csr.columnIndices.begin(), csr.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCsrBlockSize4x4) {
  auto csr = popsparse::changeCSRBlockSize(csrRef4x4, {2, 2});
  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef4x4_2x2.nzValues.begin(),
                                csrRef4x4_2x2.nzValues.end(),
                                csr.nzValues.begin(), csr.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csrRef4x4_2x2.rowIndices.begin(),
                                csrRef4x4_2x2.rowIndices.end(),
                                csr.rowIndices.begin(), csr.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csrRef4x4_2x2.columnIndices.begin(), csrRef4x4_2x2.columnIndices.end(),
      csr.columnIndices.begin(), csr.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCscBlockSize4x4) {
  auto csc4x4 = popsparse::csrToCSC(4, 16, csrRef4x4);
  auto csc4x4_2x2 = popsparse::csrToCSC(4, 16, csrRef4x4_2x2);
  auto csc = popsparse::changeCSCBlockSize(csc4x4, {2, 2});
  BOOST_CHECK_EQUAL_COLLECTIONS(csc4x4_2x2.nzValues.begin(),
                                csc4x4_2x2.nzValues.end(), csc.nzValues.begin(),
                                csc.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csc4x4_2x2.rowIndices.begin(),
                                csc4x4_2x2.rowIndices.end(),
                                csc.rowIndices.begin(), csc.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csc4x4_2x2.columnIndices.begin(), csc4x4_2x2.columnIndices.end(),
      csc.columnIndices.begin(), csc.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCscBlockSize2x6) {
  auto csc2x6 = popsparse::csrToCSC(6, 18, csrRef2x6);
  auto csc2x6_1x2 = popsparse::csrToCSC(6, 18, csrRef2x6_1x2);
  auto csc = popsparse::changeCSCBlockSize(csc2x6, {1, 2});
  BOOST_CHECK_EQUAL_COLLECTIONS(csc2x6_1x2.nzValues.begin(),
                                csc2x6_1x2.nzValues.end(), csc.nzValues.begin(),
                                csc.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(csc2x6_1x2.rowIndices.begin(),
                                csc2x6_1x2.rowIndices.end(),
                                csc.rowIndices.begin(), csc.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      csc2x6_1x2.columnIndices.begin(), csc2x6_1x2.columnIndices.end(),
      csc.columnIndices.begin(), csc.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCooBlockSize4x4) {
  auto coo = popsparse::changeCOOBlockSize(cooRef4x4, {2, 2});
  BOOST_CHECK_EQUAL_COLLECTIONS(cooRef4x4_2x2.nzValues.begin(),
                                cooRef4x4_2x2.nzValues.end(),
                                coo.nzValues.begin(), coo.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(cooRef4x4_2x2.rowIndices.begin(),
                                cooRef4x4_2x2.rowIndices.end(),
                                coo.rowIndices.begin(), coo.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef4x4_2x2.columnIndices.begin(), cooRef4x4_2x2.columnIndices.end(),
      coo.columnIndices.begin(), coo.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCooBlockSize2x6) {
  auto coo = popsparse::changeCOOBlockSize(cooRef2x6, {1, 2});
  BOOST_CHECK_EQUAL_COLLECTIONS(cooRef2x6_1x2.nzValues.begin(),
                                cooRef2x6_1x2.nzValues.end(),
                                coo.nzValues.begin(), coo.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(cooRef2x6_1x2.rowIndices.begin(),
                                cooRef2x6_1x2.rowIndices.end(),
                                coo.rowIndices.begin(), coo.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef2x6_1x2.columnIndices.begin(), cooRef2x6_1x2.columnIndices.end(),
      coo.columnIndices.begin(), coo.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCooBlockSize2x6DoubleChange) {
  auto coo = popsparse::changeCOOBlockSize(cooRef2x6, {1, 2});
  auto cooOrig = popsparse::changeCOOBlockSize(coo, {2, 6});
  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef2x6.nzValues.begin(), cooRef2x6.nzValues.end(),
      cooOrig.nzValues.begin(), cooOrig.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef2x6.rowIndices.begin(), cooRef2x6.rowIndices.end(),
      cooOrig.rowIndices.begin(), cooOrig.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef2x6.columnIndices.begin(), cooRef2x6.columnIndices.end(),
      cooOrig.columnIndices.begin(), cooOrig.columnIndices.end());
}

BOOST_AUTO_TEST_CASE(ChangeCooBlockSize4x4DoubleChange) {
  auto coo = popsparse::changeCOOBlockSize(cooRef4x4, {2, 2});
  auto cooOrig = popsparse::changeCOOBlockSize(coo, {4, 4});

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef4x4.nzValues.begin(), cooRef4x4.nzValues.end(),
      cooOrig.nzValues.begin(), cooOrig.nzValues.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef4x4.rowIndices.begin(), cooRef4x4.rowIndices.end(),
      cooOrig.rowIndices.begin(), cooOrig.rowIndices.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(
      cooRef4x4.columnIndices.begin(), cooRef4x4.columnIndices.end(),
      cooOrig.columnIndices.begin(), cooOrig.columnIndices.end());
}