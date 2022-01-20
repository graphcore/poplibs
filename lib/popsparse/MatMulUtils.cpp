// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "MatMulUtils.hpp"
#include "MatMulTensorMetaData.hpp"
#include "poplibs_support/logging.hpp"
#include "popsparse/MatMulParams.hpp"
#include "popsparse/SparseTensor.hpp"

using namespace poplar;

namespace popsparse {
namespace dynamic {

FullyConnectedParams getFullyConnectedParams(const MatMulParams &params) {
  return FullyConnectedParams::createWithNzRatio(
      params.getSparsityParams(), params.getNzRatio(), params.getN(),
      params.getNumGroups(), params.getK(), params.getM());
}

poplar::OptionFlags getFullyConnectedOptions(const MatMulOptions &options) {
  return OptionFlags{
      {"availableMemoryProportion",
       std::to_string(options.availableMemoryProportion)},
      {"metaInfoBucketOversizeProportion",
       std::to_string(options.metaInfoBucketOversizeProportion)},
      {"doGradAPass", "true"},
      {"doGradWPass", "false"},
      {"partialsType", options.partialsType.toString()},
      {"sharedBuckets", (options.sharedBuckets ? "true" : "false")},
      {"partitioner.optimiseForSpeed",
       (options.partitioner.optimiseForSpeed ? "true" : "false")},
      {"partitioner.forceBucketSpills",
       (options.partitioner.forceBucketSpills ? "true" : "false")},
      {"partitioner.useActualWorkerSplitCosts",
       (options.partitioner.useActualWorkerSplitCosts ? "true" : "false")}};
}

SparseTensor sparseMatrixToFCWeights(const SparseTensor &t) {
  assert(
      dynamic_cast<const MatMulTensorMetaData *>(t.getOpMetaData().getData()));
  const auto &mmMetaData =
      static_cast<const MatMulTensorMetaData *>(t.getOpMetaData().getData());

  return SparseTensor(t.getMetaInfoTensor(), t.getNzValuesTensor(),
                      mmMetaData->fc.clone());
}

// Validate the matmul options
MatMulOptions validateOptions(const poplar::Type &inOutType,
                              const poplar::Target &target,
                              const MatMulParams &params,
                              MatMulOptions options) {
  if (target.getTypeSize(options.partialsType) <
      target.getTypeSize(inOutType)) {
    poplibs_support::logging::popsparse::warn(
        "Ignoring sparse partialsType option ({}) "
        "which is smaller than the input/output type ({})",
        options.partialsType, inOutType);
    options.partialsType = inOutType;
  }
  if (options.partialsType != FLOAT &&
      (params.getSparsityParams().blockDimensions[0] *
       params.getSparsityParams().blockDimensions[0]) == 1) {
    poplibs_support::logging::popsparse::warn(
        "Ignoring sparse partialsType option ({}) "
        "which must be FLOAT for element wise sparse operations",
        options.partialsType, inOutType);
    options.partialsType = FLOAT;
  }
  return options;
}

} // end namespace dynamic
} // end namespace popsparse
