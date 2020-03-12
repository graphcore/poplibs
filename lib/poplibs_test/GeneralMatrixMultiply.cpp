// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/exceptions.hpp>

void poplibs_test::gemm::hadamardProduct(
    const boost::multi_array_ref<double, 1> matA,
    const boost::multi_array_ref<double, 1> matB,
    boost::multi_array_ref<double, 1> matC, float alpha) {
#ifndef NDEBUG
  unsigned matACols = matA.shape()[0];
  unsigned matBCols = matB.shape()[0];
#endif
  unsigned matCCols = matC.shape()[1];

  assert(matACols == matCCols && matBCols == matCCols);

  for (auto c = 0U; c != matCCols; ++c) {
    matC[c] = alpha * matA[c] * matB[c];
  }
}

void poplibs_test::gemm::hadamardProduct(
    const boost::multi_array_ref<double, 2> matA,
    const boost::multi_array_ref<double, 2> matB,
    boost::multi_array_ref<double, 2> matC, float alpha, bool transposeA,
    bool transposeB) {
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
      matC[r][c] = alpha * (transposeA ? matA[c][r] : matA[r][c]) *
                   (transposeB ? matB[c][r] : matB[r][c]);
    }
  }
}

void poplibs_test::gemm::generalMatrixMultiply(
    const boost::multi_array_ref<double, 2> matA,
    const boost::multi_array_ref<double, 1> vecB,
    const boost::multi_array_ref<double, 1> vecC,
    boost::multi_array_ref<double, 1> vecD, float alpha, float beta,
    bool transposeA) {
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

void poplibs_test::gemm::generalMatrixMultiply(
    const boost::multi_array_ref<double, 2> matA,
    const boost::multi_array_ref<double, 2> matB,
    const boost::multi_array_ref<double, 2> matC,
    boost::multi_array_ref<double, 2> matD, float alpha, float beta,
    bool transposeA, bool transposeB) {

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
        acc += (transposeA ? matA[kIdx][mIdx] : matA[mIdx][kIdx]) *
               (transposeB ? matB[nIdx][kIdx] : matB[kIdx][nIdx]);
      }
      matD[mIdx][nIdx] = beta * matC[mIdx][nIdx] + alpha * acc;
    }
  }
}

void poplibs_test::gemm::generalGroupedMatrixMultiply(
    const boost::multi_array_ref<double, 3> matA,
    const boost::multi_array_ref<double, 3> matB,
    const boost::multi_array_ref<double, 3> matC,
    boost::multi_array_ref<double, 3> matD, float alpha, float beta,
    bool transposeA, bool transposeB) {

  const auto matAGroups = matA.shape()[0];
  const auto matACols = matA.shape()[2];
  const auto matARows = matA.shape()[1];
#ifndef NDEBUG
  const auto matBGroups = matB.shape()[0];
  const auto matBCols = matB.shape()[2];
  const auto matBRows = matB.shape()[1];
  const auto matCGroups = matC.shape()[0];
#endif
  const auto matCCols = matC.shape()[2];
  const auto matCRows = matC.shape()[1];

  const auto g = matAGroups;
  const auto m = matCRows;
  const auto n = matCCols;
  const auto k = transposeA ? matARows : matACols;

  assert(matAGroups == matBGroups);
  assert(matAGroups == matCGroups);
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

  for (unsigned gIdx = 0; gIdx != g; ++gIdx) {
    for (unsigned mIdx = 0; mIdx != m; ++mIdx) {
      for (unsigned nIdx = 0; nIdx != n; ++nIdx) {
        double acc = 0;
        for (unsigned kIdx = 0; kIdx != k; ++kIdx) {
          acc +=
              (transposeA ? matA[gIdx][kIdx][mIdx] : matA[gIdx][mIdx][kIdx]) *
              (transposeB ? matB[gIdx][nIdx][kIdx] : matB[gIdx][kIdx][nIdx]);
        }
        matD[gIdx][mIdx][nIdx] = beta * matC[gIdx][mIdx][nIdx] + alpha * acc;
      }
    }
  }
}

void poplibs_test::gemm::generalMatrixMultiply(
    const boost::multi_array_ref<double, 2> matA,
    const boost::multi_array_ref<double, 2> matB,
    boost::multi_array_ref<double, 2> matC, bool transposeA, bool transposeB) {

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
        acc += (transposeA ? matA[kIdx][mIdx] : matA[mIdx][kIdx]) *
               (transposeB ? matB[nIdx][kIdx] : matB[kIdx][nIdx]);
      }
      matC[mIdx][nIdx] = acc;
    }
  }
}

void poplibs_test::gemm::generalGroupedMatrixMultiply(
    const boost::multi_array_ref<double, 3> matA,
    const boost::multi_array_ref<double, 3> matB,
    boost::multi_array_ref<double, 3> matC, bool transposeA, bool transposeB) {
  const auto matACols = matA.shape()[2];
  const auto matARows = matA.shape()[1];
#ifndef NDEBUG
  const auto matAGroups = matA.shape()[0];
  const auto matBGroups = matB.shape()[0];
  const auto matBCols = matB.shape()[2];
  const auto matBRows = matB.shape()[1];
  const auto matCGroups = matC.shape()[0];
#endif
  const auto matCCols = matC.shape()[2];
  const auto matCRows = matC.shape()[1];

  const auto g = matA.shape()[0];
  const auto m = matCRows;
  const auto n = matCCols;
  const auto k = transposeA ? matARows : matACols;

  assert(matAGroups == matBGroups);
  assert(matAGroups == matCGroups);
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

  for (unsigned gIdx = 0; gIdx != g; ++gIdx) {
    for (unsigned mIdx = 0; mIdx != m; ++mIdx) {
      for (unsigned nIdx = 0; nIdx != n; ++nIdx) {
        double acc = 0;
        for (unsigned kIdx = 0; kIdx != k; ++kIdx) {
          acc +=
              (transposeA ? matA[gIdx][kIdx][mIdx] : matA[gIdx][mIdx][kIdx]) *
              (transposeB ? matB[gIdx][nIdx][kIdx] : matB[gIdx][kIdx][nIdx]);
        }
        matC[gIdx][mIdx][nIdx] = acc;
      }
    }
  }
}
