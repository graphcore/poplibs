// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseFormatsValidateTest
#include "../lib/popsparse/SparseFormatsValidate.hpp"
#include "popsparse/SparseStorageFormats.hpp"
#include "poputil/exceptions.hpp"
#include <boost/test/unit_test.hpp>

using namespace popsparse;

BOOST_AUTO_TEST_CASE(ValidateCsrElemWrongRowIndicesElems) {
  static const popsparse::CSRMatrix<double> csrRef(
      {10.0, 20, 30, 40, 50, 60, 70, 80.0}, {0, 1, 1, 3, 2, 3, 4, 5},
      {0, 2, 4, 7, 8, 9});
  BOOST_CHECK_THROW(validateCSR(4, 6, {1, 1}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrElemWrongColumnIndex) {
  static const popsparse::CSRMatrix<double> csrRef(
      {10.0, 20, 30, 40, 50, 60, 70, 80.0}, {0, 1, 1, 3, 2, 3, 4, 5, 6},
      {0, 2, 4, 7, 8, 9});
  BOOST_CHECK_THROW(validateCSR(4, 6, {1, 1}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrElemWrongNzValues) {
  static const popsparse::CSRMatrix<double> csrRef(
      {10.0, 20, 30, 40, 50, 60, 70, 80.0, 90.0}, {0, 1, 1, 3, 2, 3, 4, 5},
      {0, 2, 4, 7, 8});
  BOOST_CHECK_THROW(validateCSR(4, 6, {1, 1}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrBlockWrongRowIndices) {
  const popsparse::CSRMatrix<double> csrRef(
      {10.0, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24,   25, 30, 31, 32, 33,
       34,   35, 40, 41, 42, 43, 44, 45, 50, 51, 52,   53, 54, 55, 60, 61,
       62,   63, 64, 65, 70, 71, 72, 73, 74, 75, 80.0, 81, 82, 83, 84, 85},
      {0, 3, 3, 9, 6, 9, 12, 15}, {0, 12, 24, 42, 48, 52}, {2, 3});
  BOOST_CHECK_THROW(validateCSR(8, 18, {2, 3}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrBlockWrongColumnIndex) {
  const popsparse::CSRMatrix<double> csrRef(
      {10.0, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24,   25, 30, 31, 32, 33,
       34,   35, 40, 41, 42, 43, 44, 45, 50, 51, 52,   53, 54, 55, 60, 61,
       62,   63, 64, 65, 70, 71, 72, 73, 74, 75, 80.0, 81, 82, 83, 84, 85},
      {0, 3, 3, 9, 6, 9, 12, 15, 18}, {0, 12, 24, 42, 48}, {2, 3});

  BOOST_CHECK_THROW(validateCSR(8, 18, {2, 3}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrBlockWrongNzValues) {
  const popsparse::CSRMatrix<double> csrRef(
      {10.0, 11, 12, 13, 14, 15, 20, 21, 22,   23, 24, 25, 30, 31, 32, 33, 34,
       35,   40, 41, 42, 43, 44, 45, 50, 51,   52, 53, 54, 55, 60, 61, 62, 63,
       64,   65, 70, 71, 72, 73, 74, 75, 80.0, 81, 82, 83, 84, 85, 89},
      {0, 3, 3, 9, 6, 9, 12, 15}, {0, 12, 24, 42, 48}, {2, 3});

  BOOST_CHECK_THROW(validateCSR(8, 18, {2, 3}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrBlockRowNotMultipleOfBlock) {
  const popsparse::CSRMatrix<double> csrRef(
      {10.0, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24,   25, 30, 31, 32, 33,
       34,   35, 40, 41, 42, 43, 44, 45, 50, 51, 52,   53, 54, 55, 60, 61,
       62,   63, 64, 65, 70, 71, 72, 73, 74, 75, 80.0, 81, 82, 83, 84, 85},
      {0, 3, 3, 9, 6, 9, 12, 15}, {0, 12, 24, 42, 48, 53}, {2, 3});

  BOOST_CHECK_THROW(validateCSR(8, 18, {2, 3}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCsrBlockColNotMultipleOfBlock) {
  const popsparse::CSRMatrix<double> csrRef(
      {10.0, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24,   25, 30, 31, 32, 33,
       34,   35, 40, 41, 42, 43, 44, 45, 50, 51, 52,   53, 54, 55, 60, 61,
       62,   63, 64, 65, 70, 71, 72, 73, 74, 75, 80.0, 81, 82, 83, 84, 85},
      {0, 3, 3, 9, 6, 9, 12, 15, 17}, {0, 12, 24, 42, 48}, {2, 3});

  BOOST_CHECK_THROW(validateCSR(8, 18, {2, 3}, csrRef.nzValues.size(),
                                csrRef.rowIndices, csrRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCscBlockWrongColIndices) {
  static const popsparse::CSCMatrix<double> cscRef(
      {10, 13, 11, 14, 12, 15, 20, 23, 21, 24, 22, 25, 30, 33, 31, 34,
       32, 35, 50, 53, 51, 54, 52, 55, 40, 43, 41, 44, 42, 45, 60, 63,
       61, 64, 62, 65, 70, 73, 71, 74, 72, 75, 80, 83, 81, 84, 82, 85},
      {0, 6, 18, 24, 36, 42, 48, 53}, {0, 0, 2, 4, 2, 4, 4, 6}, {2, 3});
  BOOST_CHECK_THROW(validateCSC(8, 18, {2, 3}, cscRef.nzValues.size(),
                                cscRef.rowIndices, cscRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCscBlockWrongRowIndex) {
  static const popsparse::CSCMatrix<double> cscRef(
      {10, 13, 11, 14, 12, 15, 20, 23, 21, 24, 22, 25, 30, 33, 31, 34,
       32, 35, 50, 53, 51, 54, 52, 55, 40, 43, 41, 44, 42, 45, 60, 63,
       61, 64, 62, 65, 70, 73, 71, 74, 72, 75, 80, 83, 81, 84, 82, 85},
      {0, 6, 18, 24, 36, 42, 48}, {0, 0, 2, 4, 2, 4, 4, 6, 8}, {2, 3});
  BOOST_CHECK_THROW(validateCSC(8, 18, {2, 3}, cscRef.nzValues.size(),
                                cscRef.rowIndices, cscRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCscBlockWrongNzValues) {
  static const popsparse::CSCMatrix<double> cscRef(
      {10, 13, 11, 14, 12, 15, 20, 23, 21, 24, 22, 25, 30, 33, 31, 34, 32,
       35, 50, 53, 51, 54, 52, 55, 40, 43, 41, 44, 42, 45, 60, 63, 61, 64,
       62, 65, 70, 73, 71, 74, 72, 75, 80, 83, 81, 84, 82, 85, 1},
      {0, 6, 18, 24, 36, 42, 48}, {0, 0, 2, 4, 2, 4, 4, 6}, {2, 3});
  BOOST_CHECK_THROW(validateCSC(8, 18, {2, 3}, cscRef.nzValues.size(),
                                cscRef.rowIndices, cscRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCscBlockRowNotMultipleOfBlock) {
  static const popsparse::CSCMatrix<double> cscRef(
      {10, 13, 11, 14, 12, 15, 20, 23, 21, 24, 22, 25, 30, 33, 31, 34,
       32, 35, 50, 53, 51, 54, 52, 55, 40, 43, 41, 44, 42, 45, 60, 63,
       61, 64, 62, 65, 70, 73, 71, 74, 72, 75, 80, 83, 81, 84, 82, 85},
      {0, 6, 18, 24, 36, 42, 48}, {0, 0, 2, 4, 2, 4, 4, 6, 7}, {2, 3});
  BOOST_CHECK_THROW(validateCSC(8, 18, {2, 3}, cscRef.nzValues.size(),
                                cscRef.rowIndices, cscRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCscBlockColNotMultipleOfBlock) {
  static const popsparse::CSCMatrix<double> cscRef(
      {10, 13, 11, 14, 12, 15, 20, 23, 21, 24, 22, 25, 30, 33, 31, 34,
       32, 35, 50, 53, 51, 54, 52, 55, 40, 43, 41, 44, 42, 45, 60, 63,
       61, 64, 62, 65, 70, 73, 71, 74, 72, 75, 80, 83, 81, 84, 82, 85},
      {0, 6, 18, 24, 36, 42, 48, 52}, {0, 0, 2, 4, 2, 4, 4, 6}, {2, 3});
  BOOST_CHECK_THROW(validateCSC(8, 18, {2, 3}, cscRef.nzValues.size(),
                                cscRef.rowIndices, cscRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCooBlockWrongColIndices) {
  static const popsparse::COOMatrix<std::size_t> cooRef(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
       76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
      {0, 12, 6, 0, 6, 12, 15}, {0, 0, 2, 4, 4, 4}, {2, 6});
  BOOST_CHECK_THROW(validateCOO(6, 18, {2, 6}, cooRef.nzValues.size(),
                                cooRef.rowIndices, cooRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCooBlockWrongRowIndex) {
  static const popsparse::COOMatrix<std::size_t> cooRef(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
       76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
      {0, 12, 6, 0, 6, 12}, {0, 0, 2, 4, 4, 4, 4}, {2, 6});
  BOOST_CHECK_THROW(validateCOO(6, 18, {2, 6}, cooRef.nzValues.size(),
                                cooRef.rowIndices, cooRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCooBlockWrongColumnIndex) {
  static const popsparse::COOMatrix<std::size_t> cooRef(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
       76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
      {0, 12, 6, 0, 6, 12, 6}, {0, 0, 2, 4, 4, 4}, {2, 6});
  BOOST_CHECK_THROW(validateCOO(6, 18, {2, 6}, cooRef.nzValues.size(),
                                cooRef.rowIndices, cooRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCooBlockWrongNzValues) {
  static const popsparse::COOMatrix<std::size_t> cooRef(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22,
       23, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45,
       46, 47, 48, 49, 50, 51, 50, 51, 52, 53, 54, 55, 56, 57, 58,
       59, 60, 61, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
       80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 1},
      {0, 12, 6, 0, 6, 12}, {0, 0, 2, 4, 4, 4}, {2, 6});
  BOOST_CHECK_THROW(validateCOO(6, 18, {2, 6}, cooRef.nzValues.size(),
                                cooRef.rowIndices, cooRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCooBlockRowNotMultipleOfBlock) {
  static const popsparse::COOMatrix<std::size_t> cooRef(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
       76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
      {0, 12, 6, 0, 6, 12}, {0, 0, 2, 4, 4, 3}, {2, 6});
  BOOST_CHECK_THROW(validateCOO(6, 18, {2, 6}, cooRef.nzValues.size(),
                                cooRef.rowIndices, cooRef.columnIndices),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ValidateCooBlockColNotMultipleOfBlock) {
  static const popsparse::COOMatrix<std::size_t> cooRef(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75,
       76, 77, 78, 79, 80, 81, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91},
      {0, 12, 6, 0, 6, 8}, {0, 0, 2, 4, 4, 4}, {2, 6});
  BOOST_CHECK_THROW(validateCOO(6, 18, {2, 6}, cooRef.nzValues.size(),
                                cooRef.rowIndices, cooRef.columnIndices),
                    poputil::poplibs_error);
}
