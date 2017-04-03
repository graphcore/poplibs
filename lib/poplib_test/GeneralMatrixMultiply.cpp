#include <poplib_test/GeneralMatrixMultiply.hpp>
#include <poplib_test/exceptions.hpp>

void poplib_test::gemm::generalMatrixMultiply(
            const boost::multi_array_ref<double, 2> matA,
            const boost::multi_array_ref<double, 1> vecB,
            const boost::multi_array_ref<double, 1> vecC,
            boost::multi_array_ref<double, 1> vecD,
            float alpha,
            float beta,
            bool  transposeA) {
#ifndef NDEBUG
  const auto matACols = matA.shape()[1];
  const auto matARows = matA.shape()[0];
#endif
  const auto vecBCols = vecB.shape()[0];
  const auto vecCCols = vecC.shape()[0];

  const auto m = vecCCols;
  const auto n = vecBCols;

  if (transposeA) {
    assert(matACols == m);
    assert(matARows == n);
  } else {
    assert(matACols == n);
    assert(matARows == m);
  }

  for (unsigned mIdx = 0; mIdx != m; ++mIdx) {
    double acc = 0;
    for (unsigned nIdx = 0; nIdx != n; ++nIdx) {
      acc += (transposeA ? matA[nIdx][mIdx] : matA[mIdx][nIdx]) * vecB[nIdx];
    }
    vecD[mIdx] = beta * vecC[mIdx] + alpha * acc;
  }
}

void poplib_test::gemm::generalMatrixMultiply(
            const boost::multi_array_ref<double, 2> matA,
            const boost::multi_array_ref<double, 2> matB,
            const boost::multi_array_ref<double, 2> matC,
            boost::multi_array_ref<double, 2> matD,
            float alpha,
            float beta,
            bool  transposeA,
            bool  transposeB) {

  const auto matACols = matA.shape()[1];
  const auto matARows = matA.shape()[0];
#ifndef NDEBUG
  const auto matBCols = matB.shape()[1];
  const auto matBRows = matB.shape()[0];
#endif
  const auto matCCols = matC.shape()[1];
  const auto matCRows = matC.shape()[0];

  const auto m = matCRows;
  const auto n = matCCols;
  const auto k = transposeA ? matARows : matACols;

  if (transposeA) {
    assert(matACols == m);
  } else {
    assert(matARows == m);
  }

  if (transposeB) {
    assert(matBRows == n);
    assert(matBCols == k);
  } else {
    assert(matBRows == k);
    assert(matBCols == n);
  }

  for (unsigned mIdx = 0; mIdx != m; ++mIdx) {
    for (unsigned nIdx = 0; nIdx != n; ++nIdx) {
      double acc = 0;
      for (unsigned kIdx = 0; kIdx != k; ++kIdx) {
        acc += (transposeA ? matA[kIdx][mIdx] : matA[mIdx][kIdx])
               * (transposeB ? matB[nIdx][kIdx] : matB[kIdx][nIdx]);
      }
      matD[mIdx][nIdx] = beta * matC[mIdx][nIdx] + alpha * acc;
    }
  }
}
