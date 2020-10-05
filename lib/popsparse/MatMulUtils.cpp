// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "MatMulUtils.hpp"
#include "MatMulTensorMetaData.hpp"
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

} // end namespace dynamic
} // end namespace popsparse
