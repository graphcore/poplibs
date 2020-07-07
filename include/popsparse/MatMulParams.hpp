// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_MatMulParams_hpp
#define popsparse_MatMulParams_hpp

#include "popsparse/SparsityParams.hpp"

#include <ostream>

namespace popsparse {
namespace dynamic {

class MatMulParams {
  SparsityParams sparsityParams;
  double nzRatio;
  std::size_t groups;
  std::size_t m;
  std::size_t k;
  std::size_t n;

public:
  /** @name Matrix multiplication parameters
   *  These are the parameters which define a matrix multiplication
   *  with one sparse operand (always the left-hand operand) and one
   *  dense operand.
   *
   *  Equivalent dense multiplication for given parameters is as follows:
   *
   *   [groups][m][k] * [groups][k][n] = [groups][m][n]
   */
  ///@{
  static MatMulParams createWithNzRatio(const SparsityParams &sparsityParams,
                                        double nzRatio, std::size_t groups,
                                        std::size_t m, std::size_t k,
                                        std::size_t n);
  static MatMulParams
  createWithNumNonZeroValues(const SparsityParams &sparsityParams,
                             std::size_t numNonZeroElems, std::size_t groups,
                             std::size_t m, std::size_t k, std::size_t n);
  const SparsityParams &getSparsityParams() const { return sparsityParams; }
  std::size_t getNumGroups() const { return groups; }
  std::size_t getM() const { return m; }
  std::size_t getK() const { return k; }
  std::size_t getN() const { return n; }
  double getNzRatio() const;
  std::size_t getNumNonZeroValues() const;

  friend bool operator<(const MatMulParams &a, const MatMulParams &b);
  friend bool operator==(const MatMulParams &a, const MatMulParams &b);
  friend bool operator!=(const MatMulParams &a, const MatMulParams &b);
};

std::ostream &operator<<(std::ostream &, const MatMulParams &);

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_MatMulParams_hpp
