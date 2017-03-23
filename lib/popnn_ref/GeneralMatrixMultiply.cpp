#include <popnn_ref/GeneralMatrixMultiply.hpp>
#include <popnn_ref/exceptions.hpp>

void ref::gemm::generalMatrixMultiply(
            const boost::multi_array<double, 2> &matA,
            const boost::multi_array<double, 2> &matB,
            const boost::multi_array<double, 2> &matC,
            boost::multi_array<double, 2> &matD,
            float alpha,
            float beta,
            bool  transposeA,
            bool  transposeB) {

  const auto matACols = matA.shape()[1];
  const auto matARows = matA.shape()[0];
  const auto matBCols = matB.shape()[1];
  const auto matBRows = matB.shape()[0];
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
