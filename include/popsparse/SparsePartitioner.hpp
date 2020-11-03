// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Contains public headers for Sparse Partitioner

#ifndef _poplibs_popsparse_SparsePartitioner_hpp_
#define _poplibs_popsparse_SparsePartitioner_hpp_

#include <poplar/Interval.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>
#include <popsparse/FullyConnected.hpp>
#include <popsparse/FullyConnectedParams.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include <string>
#include <vector>

namespace popsparse {

class PartitionerImpl;

namespace dynamic {

/// Encoding of sparsity representation.
template <typename T> struct SparsityDataImpl {
  /// Meta information representing sparsity for each tile.
  std::vector<std::size_t> metaInfo;

  /// The non-zero values of the sparse matrix.
  std::vector<T> nzValues;
};

/** Class to translate and encode  sparsity information for a fully connected
 * layer.
 *
 * See createFullyConnectedWeights() for details of the options.
 *
 */
template <typename T> class Partitioner {
  std::string name;

public:
  const PartitionerImpl &getImpl() const { return *impl; }

  Partitioner(const FullyConnectedParams &params, const poplar::Type &dataType,
              const poplar::Target &target, const poplar::OptionFlags &options,
              PlanningCache *cache = {}, std::string name = "");

  Partitioner(const MatMulParams &params, const poplar::Type &dataType,
              const poplar::Target &target, const poplar::OptionFlags &options,
              PlanningCache *cache = {}, std::string name = "");

  ~Partitioner();

  /// Create implementation sparsity representation for a compressed sparse
  /// columns (CSC) matrix.
  SparsityDataImpl<T> createSparsityDataImpl(const CSCMatrix<T> &matrix_) const;

  /// Creates implementation sparsity representation for a compressed sparse
  /// rows (CSR) matrix.
  SparsityDataImpl<T> createSparsityDataImpl(const CSRMatrix<T> &matrix_) const;

  /// Creates implementation sparsity representation for a coordinate (COO)
  /// format matrix.
  SparsityDataImpl<T> createSparsityDataImpl(const COOMatrix<T> &matrix_) const;

  /// Create a coordinate (COO) representation matrix from implementation
  /// sparsity representation. The COO entries are ordered by row first, and
  /// then columns.
  COOMatrix<T> sparsityDataImplToCOOMatrix(
      const SparsityDataImpl<T> &sparsityDataImpl) const;

  /// Create compressed sparse rows (CSR) representation from implementation
  /// sparsity representation.
  CSRMatrix<T> sparsityDataImplToCSRMatrix(
      const SparsityDataImpl<T> &sparsityDataImpl) const;

  /// Create compressed sparse columns (CSC) representation from implementation
  /// sparsity representation.
  CSCMatrix<T> sparsityDataImplToCSCMatrix(
      const SparsityDataImpl<T> &sparsityDataImpl) const;

  /// Fetch the partitions in X, Y and Z to reveal the plan
  std::array<std::vector<std::size_t>, 3> getPlanPartitions(void) const;

  /// Fetch the number of elements in a meta info bucket.
  std::size_t getmetaInfoBucketElements(void) const;

private:
  std::unique_ptr<PartitionerImpl> impl;
};

} // namespace dynamic
} // namespace popsparse
#endif // _poplibs_popsparse_SparsePartitioner_hpp_
