// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Translation and encoding of sparsity information for a fully connected
 *  layer.
 */

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

namespace dynamic {

class PartitionerImpl;

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

namespace static_ {

/// Encoding of sparsity representation.
template <typename T> struct SparsityDataImpl {
  /// The non-zero values of the sparse matrix in an order required by the
  /// device compute graph.
  std::vector<T> nzValues;

  SparsityDataImpl(std::vector<T> nzValues) : nzValues(std::move(nzValues)) {}
};

class PartitionerImpl;

// A partitioner object is needed to create host side data manipulation for a
// given sparse * dense or a dense * sparse matrix multiplication. An object of
// type Partitioner is first created. Method \ref createSparsityDataImpl is
// then called with a sparse representation (COO, CSR and CSC are supported) to
// create a host representation of sparsity used by the device implementation.
// The returned \ref SparsityDataImpl can be then be used to copy the NZ values
// to the device.
// The same compute graph can be used as long as the positions of the sparse
// elements are the same regardless of the representation used (i.e. COO, CSR,
// or CSC). Thus multiple calls to \ref createSparsityDataImpl may be used
// if the NZ values changed on the host.
//
// A COO, CSR or CSC representation of given NZ values can be re-created by
// reading the NZ values created by \ref createSparsityDataImpl from the device
// and calling the appropriate conversion function.
//   \ref sparsityDataImplToCOOMatrix
//   \ref sparsityDataImplToCSRMatrix
//   \ref sparsityDataImplToCSCMatrix
template <typename T> class Partitioner {
  std::string name;

public:
  const PartitionerImpl &getImpl() const { return *impl; }

  /// Construct Partitioner for a matrix multiplication.
  ///
  /// params       The matrix multiplication params. The parameters must be
  ///              correctly created for the operation. i.e. sparse * dense,
  ///              or dense * sparse.
  /// dataType     The imput data type
  /// target       A reference to the target for which the matmul is created.
  ///              The  target must outlive any use of the class object.
  /// options      Implementation options for the matrix multiplication.
  /// cache        Optional pointer to planning cache to use. Must outlive the
  ///              use of the class object.
  Partitioner(const MatMulParams &params, const poplar::Type &dataType,
              const poplar::Target &target, const poplar::OptionFlags &options,
              PlanningCache *cache = {}, std::string name = "");

  ~Partitioner();

  /// Create implementation sparsity representation for a compressed sparse
  /// columns (CSC) matrix.
  SparsityDataImpl<T> createSparsityDataImpl(const CSCMatrix<T> &matrix) const;

  /// Creates implementation sparsity representation for a compressed sparse
  /// rows (CSR) matrix.
  SparsityDataImpl<T> createSparsityDataImpl(const CSRMatrix<T> &matrix) const;

  /// Creates implementation sparsity representation for a coordinate (COO)
  /// format matrix.
  SparsityDataImpl<T> createSparsityDataImpl(const COOMatrix<T> &matrix) const;

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

private:
  std::unique_ptr<PartitionerImpl> impl;
};

} // namespace static_

} // namespace popsparse
#endif // _poplibs_popsparse_SparsePartitioner_hpp_
