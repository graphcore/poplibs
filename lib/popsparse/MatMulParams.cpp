// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popsparse/MatMulParams.hpp"

#include "poputil/DebugInfo.hpp"
#include "poputil/exceptions.hpp"

#include "FullyConnectedUtils.hpp"

#include <gccs/StructHelper.hpp>

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popsparse::dynamic::MatMulParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"sparsityParams", toProfileValue(t.getSparsityParams())});
  v.insert({"nzRatio", toProfileValue(t.getNzRatio())});
  v.insert({"groups", toProfileValue(t.getNumGroups())});
  v.insert({"m", toProfileValue(t.getM())});
  v.insert({"k", toProfileValue(t.getK())});
  v.insert({"n", toProfileValue(t.getN())});
  return v;
}

template <>
poplar::ProfileValue toProfileValue(const popsparse::static_::MatMulParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"groups", toProfileValue(t.getNumGroups())});
  v.insert({"m", toProfileValue(t.getM())});
  v.insert({"k", toProfileValue(t.getK())});
  v.insert({"n", toProfileValue(t.getN())});
  v.insert({"transposed", toProfileValue(t.isTransposed())});
  return v;
}

} // namespace poputil

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
  static constexpr auto comparisonHelper = gccs::makeStructHelper(
      &MatMulParams::sparsityParams, &MatMulParams::nzRatio,
      &MatMulParams::groups, &MatMulParams::m, &MatMulParams::k,
      &MatMulParams::n);
  return comparisonHelper.lt(a, b);
}

bool operator==(const MatMulParams &a, const MatMulParams &b) {
  static constexpr auto comparisonHelper = gccs::makeStructHelper(
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

namespace static_ {

MatMulParams MatMulParams::createForSparseDense(std::size_t groups,
                                                std::size_t m, std::size_t k,
                                                std::size_t n) {
  MatMulParams p;
  p.groups = groups;
  p.m = m;
  p.k = k;
  p.n = n;
  p.transposed = false;
  return p;
}

MatMulParams MatMulParams::createForDenseSparse(std::size_t groups,
                                                std::size_t n, std::size_t k,
                                                std::size_t m) {
  MatMulParams p;
  p.groups = groups;
  p.m = m;
  p.k = k;
  p.n = n;
  p.transposed = true;
  return p;
}

bool operator<(const MatMulParams &a, const MatMulParams &b) {
  static constexpr auto comparisonHelper = gccs::makeStructHelper(
      &MatMulParams::groups, &MatMulParams::m, &MatMulParams::k,
      &MatMulParams::n, &MatMulParams::transposed);
  return comparisonHelper.lt(a, b);
}

bool operator==(const MatMulParams &a, const MatMulParams &b) {
  static constexpr auto comparisonHelper = gccs::makeStructHelper(
      &MatMulParams::groups, &MatMulParams::m, &MatMulParams::k,
      &MatMulParams::n, &MatMulParams::transposed);
  return comparisonHelper.eq(a, b);
}

bool operator!=(const MatMulParams &a, const MatMulParams &b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const MatMulParams &p) {
  if (p.isTransposed()) {
    os << "{ dense [" << p.getN() << "," << p.getK() << "] * [" << p.getK()
       << "," << p.getM() << "] sparse }";
  } else {
    os << "{ sparse [" << p.getM() << "," << p.getK() << "] * [" << p.getK()
       << "," << p.getN() << "] dense }";
  }
  return os;
}

} // end namespace static_

} // end namespace popsparse
