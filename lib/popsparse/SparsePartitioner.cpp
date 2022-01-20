// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/SparsePartitioner.hpp"
#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include "MatMulUtils.hpp"
#include "SparsePartitionerImpl.hpp"
#include "poplibs_support/logging.hpp"

using namespace poplibs_support;

namespace popsparse {

using namespace fullyconnected;

namespace dynamic {

template <typename T>
Partitioner<T>::Partitioner(const FullyConnectedParams &params,
                            const poplar::Type &dataType,
                            const poplar::Target &target,
                            const poplar::OptionFlags &optionFlags,
                            PlanningCache *cache, std::string name_) {
  Plan plan;
  Cost planCost;
  std::tie(plan, planCost) =
      getPlan(target, dataType, params, optionFlags, cache);
  const auto partitionIndices = getPartitionStartIndices(params, plan);
  const auto options = fullyconnected::parseOptionFlags(optionFlags);
  name = std::move(name_);

  logging::popsparse::info("Creating partitioner for:{}", name);

  // TODO: Perhaps we should represent the meta-info format more explicitly in
  // both partitioner and plan. For now base it purely on sparsity type.
  const bool useBlockMetaInfoFormat =
      params.getSparsityParams().type == SparsityType::Block;
  impl.reset(new PartitionerImpl(
      {params.getOutputChannelsPerGroup(), params.getInputChannelsPerGroup(),
       params.getBatchSize()},
      {plan.method.grouping.x, plan.method.grouping.y, plan.method.grouping.z},
      params.getSparsityParams().blockDimensions, partitionIndices.at(0),
      partitionIndices.at(1), partitionIndices.at(2),
      plan.fwdMetaInfoElemsPerBucket, plan.gradAMetaInfoElemsPerBucket,
      plan.nzElemsPerBucket, target.getNumWorkerContexts(), 1,
      useBlockMetaInfoFormat, options.doGradAPass, options.doGradWPass,
      options.sharedBuckets, dataType, options.partialsType,
      options.partitioner, plan.useDense));
}

template <typename T>
Partitioner<T>::Partitioner(const MatMulParams &params,
                            const poplar::Type &dataType,
                            const poplar::Target &target,
                            const poplar::OptionFlags &optionFlags,
                            PlanningCache *cache, std::string name)
    : Partitioner(
          getFullyConnectedParams(params), dataType, target,
          getFullyConnectedOptions(validateOptions(
              dataType, target, params, parseMatMulOptionFlags(optionFlags))),
          cache, name) {}

template <typename T> Partitioner<T>::~Partitioner() {}

template <typename T>
SparsityDataImpl<T>
Partitioner<T>::createSparsityDataImpl(const CSCMatrix<T> &matrix_) const {
  logging::popsparse::info("Creating sparsity implementation for CSC matrix:{}",
                           name);
  auto info = impl->bucketImplAllPasses(impl->createBuckets(matrix_), name);
  return {std::get<0>(info), std::get<1>(info)};
}

template <typename T>
SparsityDataImpl<T>
Partitioner<T>::createSparsityDataImpl(const CSRMatrix<T> &matrix_) const {
  logging::popsparse::info("Creating sparsity implementation for CSR matrix:{}",
                           name);
  auto info = impl->bucketImplAllPasses(impl->createBuckets(matrix_), name);
  return {std::get<0>(info), std::get<1>(info)};
}

template <typename T>
SparsityDataImpl<T>
Partitioner<T>::createSparsityDataImpl(const COOMatrix<T> &matrix_) const {
  logging::popsparse::info("Creating sparsity implementation for COO matrix:{}",
                           name);
  auto info = impl->bucketImplAllPasses(impl->createBuckets(matrix_), name);
  return {std::get<0>(info), std::get<1>(info)};
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
