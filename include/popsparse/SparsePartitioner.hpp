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
#include <popsparse/SparseStorageFormats.hpp>
#include <string>
#include <vector>

namespace popsparse {

template <typename T> class PartitionerImpl;

namespace dynamic {

// Encoding of sparsity representation
template <typename T> struct SparsityDataImpl {
  // Meta information representing sparsity for each tile
  std::vector<std::size_t> metaInfo;

  // NZ values
  std::vector<T> nzValues;
};

// Class to translate and encode  sparsity information for a Fully Connected
// layer.
template <typename T> class Partitioner {
public:
  const PartitionerImpl<T> &getImpl() const { return *impl; }

  Partitioner(const FullyConnectedParams &params, const poplar::Type &dataType,
              const poplar::Target &target, const poplar::OptionFlags &options,
              PlanningCache *cache = {});

  ~Partitioner();

  // Create implementation sparsity representation for a CSC matrix
  SparsityDataImpl<T> createSparsityDataImpl(const CSCMatrix<T> &matrix_) const;

  // Creates implementation sparsity representation for a CSR matrix
  SparsityDataImpl<T> createSparsityDataImpl(const CSRMatrix<T> &matrix_) const;

  // Creates implementation sparsity representation for a COO matrix
  SparsityDataImpl<T> createSparsityDataImpl(const COOMatrix<T> &matrix_) const;

  // Create COO representation matrix from implementation sparsity
  // representation. The COO entries are ordered by row first and then columns.
  COOMatrix<T> sparsityDataImplToCOOMatrix(
      const SparsityDataImpl<T> &sparsityDataImpl) const;

  // Create CSR representation from implementation sparsity representation.
  CSRMatrix<T> sparsityDataImplToCSRMatrix(
      const SparsityDataImpl<T> &sparsityDataImpl) const;

  // create CSC representation from implementation sparsity representation.
  CSCMatrix<T> sparsityDataImplToCSCMatrix(
      const SparsityDataImpl<T> &sparsityDataImpl) const;

private:
  std::unique_ptr<PartitionerImpl<T>> impl;
};

} // namespace dynamic
} // namespace popsparse
#endif // _poplibs_popsparse_SparsePartitioner_hpp_
