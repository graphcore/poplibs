// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popsparse/MatMulParams.hpp"

#include "poplibs_support/StructHelper.hpp"
#include "poputil/exceptions.hpp"

#include "FullyConnectedUtils.hpp"

namespace popsparse {

using namespace fullyconnected;

namespace dynamic {

MatMulParams
MatMulParams::createWithNzRatio(const SparsityParams &sparsityParams,
                                double nzRatio, std::size_t groups,
                                std::size_t m, std::size_t k, std::size_t n) {
  if (nzRatio < 0.0 || nzRatio > 1.0) {
    throw poputil::poplibs_error("Non-zero ratio (" + std::to_string(nzRatio) +
                                 ") must be in range [0.0,1.0] but is not");
  }
  MatMulParams p;
  p.sparsityParams = sparsityParams;
  p.nzRatio = nzRatio;
  p.groups = groups;
  p.m = m;
  p.k = k;
  p.n = n;
  return p;
}

MatMulParams MatMulParams::createWithNumNonZeroValues(
    const SparsityParams &sparsityParams, std::size_t numNonZeroElems,
    std::size_t groups, std::size_t m, std::size_t k, std::size_t n) {
  const std::size_t totalDenseElems = groups * m * k;
  if (numNonZeroElems > totalDenseElems) {
    throw poputil::poplibs_error(
        "Number of non-zero elements in lhs (sparse) matrix (" +
        std::to_string(numNonZeroElems) +
        ") exceeds maximum possible for given dense matrix dimensions (" +
        (groups > 1 ? std::to_string(groups) + "x" : "") + std::to_string(m) +
        "x" + std::to_string(k) + ")");
  }
  const auto nzRatio =
      convertAbsoluteNzElemsToRatio(groups, k, m, numNonZeroElems);
  return createWithNzRatio(sparsityParams, nzRatio, groups, m, k, n);
}

double MatMulParams::getNzRatio() const { return nzRatio; }

std::size_t MatMulParams::getNumNonZeroValues() const {
  return convertRatioNzElemsToAbsolute(groups, k, m, nzRatio);
}

bool operator<(const MatMulParams &a, const MatMulParams &b) {
  static constexpr auto comparisonHelper = poplibs_support::makeStructHelper(
      &MatMulParams::sparsityParams, &MatMulParams::nzRatio,
      &MatMulParams::groups, &MatMulParams::m, &MatMulParams::k,
      &MatMulParams::n);
  return comparisonHelper.lt(a, b);
}

bool operator==(const MatMulParams &a, const MatMulParams &b) {
  static constexpr auto comparisonHelper = poplibs_support::makeStructHelper(
      &MatMulParams::sparsityParams, &MatMulParams::nzRatio,
      &MatMulParams::groups, &MatMulParams::m, &MatMulParams::k,
      &MatMulParams::n);
  return comparisonHelper.eq(a, b);
}

bool operator!=(const MatMulParams &a, const MatMulParams &b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const MatMulParams &p) {
  os << "{sparsity: " << p.getSparsityParams()
     << ",\n sparsity ratio: " << p.getNzRatio() << ", m: " << p.getM()
     << ", k: " << p.getK() << ", n: " << p.getN() << "}";
  return os;
}

} // end namespace dynamic
} // end namespace popsparse
