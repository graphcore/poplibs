// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Definitions for sparse matrix multiply operations.
 */

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

namespace static_ {
class MatMulParams {
  std::size_t groups;
  std::size_t m;
  std::size_t k;
  std::size_t n;
  bool transposed;

public:
  /** @name Matrix multiplication parameters
   *  These are the parameters which define a matrix multiplication
   *  with one sparse operand and one dense operand.
   *
   *  Equivalent dense multiplication for given parameters is as follows when
   *  created as sparse-dense using static_::MatMulParams::createForSparseDense
   *
   *   [groups][m][k] * [groups][k][n] = [groups][m][n]
   *
   *  Equivalent dense multiplication for given parameters is as follows when
   *  created as a dense-sparse using
   static_::MatMulParams::createForDenseSparse
   *
   *   [groups][n][k] * [groups][k][m] = [groups][n][m]

   */
  ///@{

  /** Create matrix multiplication parameters for sparse * dense  yielding
   *  a dense result.
   *
   *     [groups][m][k] * [groups][k][n] = [groups][m][n]
   *
   *        sparse      *    dense       = dense
   */
  static MatMulParams createForSparseDense(std::size_t groups, std::size_t m,
                                           std::size_t k, std::size_t n);

  /** Create matrix multiplication parameters for sparse * dense  yielding
   *  a dense result.
   *
   *     [groups][n][k] * [groups][k][m] = [groups][n][m]
   *
   *        dense      *    sparse       = dense
   */

  static MatMulParams createForDenseSparse(std::size_t groups, std::size_t n,
                                           std::size_t k, std::size_t m);

  std::size_t getNumGroups() const { return groups; }
  std::size_t getM() const { return m; }
  std::size_t getK() const { return k; }
  std::size_t getN() const { return n; }
  bool isTransposed() const { return transposed; }

  friend bool operator<(const MatMulParams &a, const MatMulParams &b);
  friend bool operator==(const MatMulParams &a, const MatMulParams &b);
  friend bool operator!=(const MatMulParams &a, const MatMulParams &b);
};

std::ostream &operator<<(std::ostream &, const MatMulParams &);

} // end namespace static_

} // end namespace popsparse

#endif // popsparse_MatMulParams_hpp
