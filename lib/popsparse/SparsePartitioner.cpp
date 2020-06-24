// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/SparsePartitioner.hpp"
#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include "SparsePartitionerImpl.hpp"

namespace popsparse {
namespace dynamic {

template <typename T>
Partitioner<T>::Partitioner(const FullyConnectedParams &params,
                            const poplar::Type &dataType,
                            const poplar::Target &target,
                            const poplar::OptionFlags &options,
                            PlanningCache *cache) {
  impl.reset(new PartitionerImpl<T>(params, dataType, target, options, cache));
}

template <typename T> Partitioner<T>::~Partitioner() {}

template <typename T>
SparsityDataImpl<T>
Partitioner<T>::createSparsityDataImpl(const CSCMatrix<T> &matrix_) const {
  auto pnBuckets = impl->createBuckets(matrix_);
  auto info = impl->bucketImplAllPasses(pnBuckets);
  SparsityDataImpl<T> bucketImpl;
  bucketImpl.metaInfo = std::move(std::get<0>(info));
  bucketImpl.nzValues = std::move(std::get<1>(info));
  return bucketImpl;
}

template <typename T>
SparsityDataImpl<T>
Partitioner<T>::createSparsityDataImpl(const CSRMatrix<T> &matrix_) const {
  auto pnBuckets = impl->createBuckets(matrix_);
  auto info = impl->bucketImplAllPasses(pnBuckets);
  SparsityDataImpl<T> bucketImpl;
  bucketImpl.metaInfo = std::move(std::get<0>(info));
  bucketImpl.nzValues = std::move(std::get<1>(info));
  return bucketImpl;
}

template <typename T>
SparsityDataImpl<T>
Partitioner<T>::createSparsityDataImpl(const COOMatrix<T> &matrix_) const {
  auto pnBuckets = impl->createBuckets(matrix_);
  auto info = impl->bucketImplAllPasses(pnBuckets);
  SparsityDataImpl<T> bucketImpl;
  bucketImpl.metaInfo = std::move(std::get<0>(info));
  bucketImpl.nzValues = std::move(std::get<1>(info));
  return bucketImpl;
}

template <typename T>
COOMatrix<T> Partitioner<T>::sparsityDataImplToCOOMatrix(
    const SparsityDataImpl<T> &buckets) const {
  return impl->bucketsToCOOMatrix(buckets.metaInfo, buckets.nzValues);
}

template <typename T>
CSRMatrix<T> Partitioner<T>::sparsityDataImplToCSRMatrix(
    const SparsityDataImpl<T> &buckets) const {
  return impl->bucketsToCSRMatrix(buckets.metaInfo, buckets.nzValues);
}

template <typename T>
CSCMatrix<T> Partitioner<T>::sparsityDataImplToCSCMatrix(
    const SparsityDataImpl<T> &buckets) const {
  return impl->bucketsToCSCMatrix(buckets.metaInfo, buckets.nzValues);
}

// instantiation of supported types
template class Partitioner<double>;
template class Partitioner<float>;

} // namespace dynamic
} // namespace popsparse
