// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparseCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popsparse {

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulElementWise)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulGradAElementWise)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulElementWiseTranspose)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulGradWElementWise)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulBlock)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, unsigned BlockRows, unsigned BlockCols) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulBlockGradA)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, unsigned BlockRows, unsigned BlockCols) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulBlockGradW)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, unsigned BlockRows, unsigned BlockCols) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMatMulBlockAmpGradW)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, unsigned BlockRows, unsigned BlockCols) {
  return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BlockTransposeGradW)(const VertexIntrospector &vertex,
                                               const Target &target,
                                               const Type &fpType) {
  const auto numWorkers = target.getNumWorkerContexts();
  CODELET_SCALAR_VAL(blockSizeXOrY, unsigned);
  CODELET_SCALAR_VAL(numXOrYBlocks, unsigned);
  CODELET_SCALAR_VAL(numZ, unsigned);

  return getBlockTransposeGradWCycles(fpType == FLOAT, blockSizeXOrY,
                                      numXOrYBlocks, numZ, numWorkers);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMultiSliceElementWise)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fpType) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMultiUpdateAddElementWise)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fpType) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMultiSliceBlock)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const unsigned vectorWidthInBytes) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseDenseMultiUpdateAddBlock)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const bool vectorise) {
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SparseGatherElementWise)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fpType) {
  const auto numWorkers = target.getNumWorkerContexts();
  CODELET_SCALAR_VAL(numIndices, unsigned);
  CODELET_SCALAR_VAL(workerOffsets, unsigned);
  const unsigned numBits = fpType == HALF ? 2 : 1;
  auto remainder = numIndices & ((1 << numBits) - 1);
  auto numVectors = (numIndices >> numBits) * numWorkers;

  // auto offsets = workerOffsets;
  for (unsigned i = 0, offsets = workerOffsets; i != numWorkers;
       ++i, offsets >>= 1) {
    numVectors += (offsets & 0x1);
  }

  return sparseGatherElementWiseCycles((numVectors << numBits) + remainder,
                                       numWorkers, fpType == FLOAT);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BufferIndexUpdate)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  return 6 * target.getNumWorkerContexts();
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BitIsSet)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &storageType, const Type &indexType) {
  // ld bit storage pointer
  // ld index pointer
  // ld index
  // index >> log2(sizeof(StorageType))
  // ld storage element
  // index & (sizeof(StorageType) - 1)
  // and storage element
  // st32
  return 8 * target.getNumWorkerContexts();
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return {
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulElementWise, HALF,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulElementWise, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulElementWiseTranspose,
                            HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulElementWiseTranspose,
                            FLOAT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulGradWElementWise, HALF,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulGradWElementWise, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulGradAElementWise, HALF,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulGradAElementWise, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseGatherElementWise, HALF),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseGatherElementWise, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, BufferIndexUpdate, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, BitIsSet, UNSIGNED_SHORT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, HALF, FLOAT, 4,
                            4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, HALF, HALF, 4,
                            4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, FLOAT, FLOAT, 4,
                            4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, HALF, FLOAT, 8,
                            8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, HALF, HALF, 8,
                            8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, FLOAT, FLOAT, 8,
                            8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, HALF, HALF, 16,
                            16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, HALF, FLOAT, 16,
                            16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlock, FLOAT, FLOAT, 16,
                            16),

      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, HALF, FLOAT,
                            4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, HALF, HALF,
                            4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, FLOAT,
                            FLOAT, 4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, HALF, FLOAT,
                            8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, HALF, HALF,
                            8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, FLOAT,
                            FLOAT, 8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, HALF, HALF,
                            16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, HALF, FLOAT,
                            16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradA, FLOAT,
                            FLOAT, 16, 16),

      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, HALF, FLOAT,
                            4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, HALF, HALF,
                            4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, FLOAT,
                            FLOAT, 4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, HALF, FLOAT,
                            8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, HALF, HALF,
                            8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, FLOAT,
                            FLOAT, 8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, HALF, HALF,
                            16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, HALF, FLOAT,
                            16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockGradW, FLOAT,
                            FLOAT, 16, 16),

      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, HALF,
                            FLOAT, 4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, HALF,
                            HALF, 4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, FLOAT,
                            FLOAT, 4, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, HALF,
                            FLOAT, 8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, HALF,
                            HALF, 8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, FLOAT,
                            FLOAT, 8, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, HALF,
                            HALF, 16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, HALF,
                            FLOAT, 16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMatMulBlockAmpGradW, FLOAT,
                            FLOAT, 16, 16),
      CYCLE_ESTIMATOR_ENTRY(popsparse, BlockTransposeGradW, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, BlockTransposeGradW, HALF),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceElementWise, HALF),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceElementWise, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiUpdateAddElementWise,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiUpdateAddElementWise,
                            FLOAT),

      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceBlock, HALF, 2),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceBlock, HALF, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceBlock, HALF, 8),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceBlock, FLOAT, 4),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiSliceBlock, FLOAT, 8),

      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiUpdateAddBlock, HALF,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiUpdateAddBlock, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiUpdateAddBlock, HALF,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popsparse, SparseDenseMultiUpdateAddBlock, FLOAT,
                            false),

  };
}

} // end namespace popsparse
