// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseFormatsTest
#include "../lib/popsparse/SparseStorageInternal.hpp"
#include <boost/test/unit_test.hpp>

// CSR/CSC representation of matrix
//    10   20    0    0    0    0
//     0   30    0   40    0    0
//     0    0   50   60   70    0
//     0    0    0    0    0   80

const std::size_t numRows = 4;
const std::size_t numColumns = 6;
static const popsparse::CSRMatrix<double>
    csrRef({10.0, 20, 30, 40, 50, 60, 70, 80.0}, {0, 1, 1, 3, 2, 3, 4, 5},
           {0, 2, 4, 7, 8});

static const popsparse::CSCMatrix<double>
    cscRef({10, 20, 30, 50, 40, 60, 70, 80}, {0, 1, 3, 4, 6, 7, 8},
           {0, 0, 1, 2, 1, 2, 2, 3});

static const popsparse::CSRMatrix<double>
    csrRefUnsorted({10.0, 20, 40, 30, 70, 50, 60, 80.0},
                   {0, 1, 3, 1, 4, 2, 3, 5}, {0, 2, 4, 7, 8});

static const popsparse::COOMatrix<double>
    cooUnsorted({80, 70, 60, 50, 40, 30, 20, 10}, {5, 4, 3, 2, 3, 1, 1, 0},
                {3, 2, 2, 2, 1, 1, 0, 0});

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

BOOST_AUTO_TEST_CASE(GetRowPositionTest) {

  const auto tile =
      popsparse::Tile(poplar::Interval(0, 2), poplar::Interval(0, 4));
  const auto rowInfo =
      popsparse::getPositionValuePairsPerRow<double>(csrRef, tile);

  const std::vector<std::vector<std::pair<double, std::size_t>>> expectedInfo =
      {{{0, 10.0}, {1, 20.0}}, {{1, 30.0}, {3, 40.0}}};

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
