// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file PerformanceEstimation.hpp
 *
 * This file is for internal use only and should not be used.
 *
 */

#ifndef _popops_performance_estimation_h_
#define _popops_performance_estimation_h_

#include "popops/Expr.hpp"
#include <poplar/PerfEstimateFunc.hpp>
#include <poplar/Target.hpp>
#include <poplar/VectorLayout.hpp>
#include <popops/OperationDef.hpp>

namespace popops {

// All functions within the internal namespace are liable to be changed at any
// point without prior notice.
namespace internal {

/** Supervisor context cycle estimate for BinaryOp and UnaryOp Supervisor/
 * MultiVertex codelets.
 *
 * \param isScaledPtr64Type  Is true if vector vertex field is ScaledPtr64
 *                           and false otherwise.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t basicOpSupervisorOverhead(const bool isScaledPtr64Type = false);

/** Cycle cost for processing an arbitrary number of elements given the cycle
 * cost for processing a vector of elements simultaneously.
 *
 * \param numElems        Number of elements in the operand.
 * \param vectorSize      Size of vector of elements that can be processed
 *                        simultaneously.
 * \param cyclesPerVector Cycle cost per vector of elements.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t basicOpLoopCycles(const unsigned numElems,
                                const unsigned vectorSize,
                                const unsigned cyclesPerVector);

/** Cycle cost for processing an arbitrary number of elements given the cycle
 * cost for processing a vector of elements simultaneously.
 *
 * \param target          The target on which the operation should be estimated
 * \param type            Data type of elements.
 * \param cyclesPerVector Cycle cost per vector of elements.
 * \param vectorSize      Size of vector of elements that be processed
 *                        simultaneously.
 * \param numElems        Number of elements in the operand.
 * \param overheadPerLoop Cycles overhead per loop.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t binaryOpInnerLoopCycles(const poplar::Target &target,
                                      const poplar::Type &type,
                                      const unsigned cyclesPerVector,
                                      const bool vectorize,
                                      const unsigned numElems,
                                      const std::uint64_t overheadPerLoop);

/** Cycle estimate for Dynamic Slice 1D vertex.
 *
 * \param target         The target on which the operation should be estimated.
 * \param type           Data type of elements.
 * \param regionSize     Number of elements in tensor to be sliced.
 * \param numSubElements Number of elements per slice.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t getDynamicSlice1DEstimate(const poplar::Target &target,
                                        const poplar::Type &type,
                                        const unsigned regionSize,
                                        const unsigned numSubElements);

/** Cycle estimate for Binary-1D In-Place MultiVertex.
 *
 * \param target   The target on which the operation should be estimated.
 * \param type     Data type of elements.
 * \param op       Binary operator type.
 * \param numElems Total number of output elements.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t getBinaryOp1DInPlaceEstimate(const poplar::Target &target,
                                           const poplar::Type &type,
                                           const popops::expr::BinaryOpType op,
                                           const unsigned numElems);

/** Cycle estimate for MultiSlice vertex.
 */
struct MultiSliceTargetParameters {
  MultiSliceTargetParameters(const poplar::Target &target,
                             const poplar::Type &type)
      : atomicWriteSize(target.getAtomicStoreGranularity()),
        dataPathWidth(target.getDataPathWidth()),
        bytesPerElem(target.getTypeSize(type)),
        numWorkerContexts(target.getNumWorkerContexts()) {}
  unsigned atomicWriteSize;
  unsigned dataPathWidth;
  unsigned bytesPerElem;
  unsigned numWorkerContexts;
};

std::uint64_t getMultiSliceCycleEstimate(
    const MultiSliceTargetParameters &targetParams,
    const unsigned elemsPerSlice, const unsigned numOffsets,
    const unsigned numOffsetsInRangePerWorker,
    const unsigned offsetsPerDictEntry, const bool isUpdate = false,
    const bool indicesAreSorted = false, const bool splitSingleRegion = false);

/** Cycle estimate for MultiUpdateAdd vertex.
 */
struct MultiUpdateOpTargetParameters {
  MultiUpdateOpTargetParameters(const poplar::Target &target,
                                const poplar::Type &type)
      : atomicWriteSize(target.getAtomicStoreGranularity()),
        numWorkerContexts(target.getNumWorkerContexts()),
        bytesPerElem(target.getTypeSize(type)) {}
  unsigned atomicWriteSize;
  unsigned numWorkerContexts;
  unsigned bytesPerElem;
};

std::uint64_t getMultiUpdateOpCycleEstimate(
    const MultiUpdateOpTargetParameters &targetParams,
    bool subWordWritesRequired, const unsigned elemsPerSlice,
    const unsigned numOffsets, const unsigned numOffsetsInRangePerWorker,
    const unsigned offsetsPerDictEntry, const Operation op, const bool scaled,
    const bool scaleHigherPrecisionThanData = false,
    const bool indicesAreSorted = false);

/// Target parameters used in cast estimation
struct CastTargetParameters {
  CastTargetParameters(const poplar::Target &target,
                       const poplar::Type &fromType, const poplar::Type &toType)
      : numWorkerContexts(target.getNumWorkerContexts()),
        dataPathWidth(target.getDataPathWidth()),
        fromTypeSize(target.getTypeSize(fromType)),
        toTypeSize(target.getTypeSize(toType)) {}
  unsigned numWorkerContexts;
  unsigned dataPathWidth;
  unsigned fromTypeSize;
  unsigned toTypeSize;
};

std::uint64_t getCast2DCycleEstimate(const CastTargetParameters &targetParams,
                                     const poplar::Type &fromType,
                                     const poplar::Type &toType,
                                     std::vector<unsigned> &elemCounts);

std::uint64_t getCast1DSingleWorkerCycleEstimate(
    const CastTargetParameters &targetParams, const poplar::Type &fromType,
    const poplar::Type &toType, const unsigned numElems);

std::uint64_t getCast1DCycleEstimate(const CastTargetParameters &targetParams,
                                     const poplar::Type &fromType,
                                     const poplar::Type &toType,
                                     const unsigned numElems);

struct FillTargetParameters {
  FillTargetParameters(const poplar::Target &target)
      : dataPathWidth(target.getDataPathWidth()) {}
  unsigned dataPathWidth;
};

std::uint64_t getFill1DCycleEstimate(const FillTargetParameters &targetParams,
                                     const poplar::Type &type,
                                     const unsigned numElems);

std::uint64_t getFill2DCycleEstimate(const FillTargetParameters &targetParams,
                                     const poplar::Type &type,
                                     const std::vector<unsigned> &numElems);

enum class ScaledArithmeticOp { ADD, SUBTRACT, AXPLUSBY, AXMINUSBY };

struct ScaledArithmeticTargetParameters {
  ScaledArithmeticTargetParameters(const poplar::Target &target,
                                   const poplar::Type &dataType)
      : numWorkerContexts(target.getNumWorkerContexts()),
        vectorWidth(target.getVectorWidth(dataType)) {}
  unsigned numWorkerContexts;
  unsigned vectorWidth;
};

std::uint64_t getScaledArithmeticSupervisorCycleEstimate(
    const ScaledArithmeticTargetParameters &targetParams,
    const poplar::Type &dataType, const poplar::Type &dataBType,
    const bool memConstrained, const ScaledArithmeticOp operation,
    const poplar::layout::Vector &aLayout,
    const poplar::layout::Vector &bLayout, const unsigned numElems);

// Computes the cycles used by the scalar broadcast 1D codelet
poplar::VertexPerfEstimate broadcastArithmetic1DCycleEstimate(
    const poplar::Target &target, popops::expr::BinaryOpType op,
    const poplar::Type &inType, const poplar::Type &outType, bool inPlace,
    std::size_t dataSize);

// Computes the cycles used by the scalar broadcast 2D codelet
poplar::VertexPerfEstimate broadcastArithmeticCycleEstimate(
    const poplar::Target &target, popops::expr::BinaryOpType op,
    const poplar::Type &inType, const poplar::Type &outType, bool inPlace,
    bool uniformScalar, const std::vector<std::size_t> &data);

} // namespace internal
} // namespace popops

#endif // _popops_performance_estimation_h_
