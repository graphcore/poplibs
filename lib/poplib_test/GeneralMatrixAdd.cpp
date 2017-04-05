#include <poplib_test/GeneralMatrixAdd.hpp>

void poplib_test::axpby::add(const boost::multi_array_ref<double, 1> matA,
                             const boost::multi_array_ref<double, 1> matB,
                             boost::multi_array_ref<double, 1> matC,
                             float alpha,
                             float beta) {
#ifndef NDEBUG
  unsigned matACols = matA.shape()[0];
  unsigned matBCols = matB.shape()[0];
#endif
  unsigned matCCols = matC.shape()[1];

  assert(matACols == matCCols && matBCols == matCCols);


  for (auto c = 0U; c != matCCols; ++c) {
    matC[c] = alpha * matA[c] + beta * matB[c];
  }
}

void poplib_test::axpby::add(const boost::multi_array_ref<double, 2> matA,
                             const boost::multi_array_ref<double, 2> matB,
                             boost::multi_array_ref<double, 2> matC,
                             float alpha,
                             float beta,
                             bool  transposeA,
                             bool  transposeB) {
#ifndef NDEBUG
  unsigned matARows = matA.shape()[0];
  unsigned matACols = matA.shape()[1];
  unsigned matBRows = matB.shape()[0];
  unsigned matBCols = matB.shape()[1];
#endif
  unsigned matCRows = matB.shape()[0];
  unsigned matCCols = matB.shape()[1];

  if (transposeA) {
    assert(matARows == matCCols && matACols == matCRows);
  } else {
    assert(matARows == matCRows && matACols == matCCols);
  }

  if (transposeB) {
    assert(matBRows == matCCols && matBCols == matCRows);
  } else {
    assert(matBRows == matCRows && matBCols == matCCols);
  }

  for (auto r = 0U; r != matCRows; ++r) {
    for (auto c = 0U; c != matCCols; ++c) {
      matC[r][c] = alpha * (transposeA ? matA[c][r] : matA[r][c])
                   + beta  * (transposeB ? matB[c][r] : matB[r][c]);
    }
  }
}
