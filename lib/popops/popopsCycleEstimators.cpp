// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popopsCycleEstimators.hpp"
#include "ExprOpUtil.hpp"
#include "HistogramPerformanceEstimation.hpp"
#include "poplibs_support/FlopEstimation.hpp"
#include "poplibs_support/forceInterleavedEstimates.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Expr.hpp"
#include "popops/OperationDefUtil.hpp"
#include "popops/PerformanceEstimation.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

using namespace poplar;
using namespace poplibs_support;
using namespace popops::internal;

namespace popops {

namespace {

// ceiling of x/y
unsigned iceil(unsigned x, unsigned y) { return x / y + (x % y > 0); }

} // unnamed namespace

using BinaryOpType = popops::expr::BinaryOpType;

static bool hasExternalCodelet(expr::BinaryOpType op, Type type) {
  return (type == FLOAT || type == HALF) &&
         (op == expr::BinaryOpType::ADD || op == expr::BinaryOpType::SUBTRACT ||
          op == expr::BinaryOpType::MULTIPLY);
}

bool isComparisonOp(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::EQUAL:
  case BinaryOpType::GREATER_THAN_EQUAL:
  case BinaryOpType::GREATER_THAN:
  case BinaryOpType::LESS_THAN_EQUAL:
  case BinaryOpType::LESS_THAN:
  case BinaryOpType::LOGICAL_AND:
  case BinaryOpType::LOGICAL_OR:
  case BinaryOpType::NOT_EQUAL:
    return true;
  default:
    return false;
  }
}

namespace internal {
// Computes the cycles used by the inner loop in one of the binary op codelets,
// both for 1D MultiVertex and 2D version, in place or not.
std::uint64_t binaryOpInnerLoopCycles(const Target &target,
                                      const BinaryOpType op, const Type &type,
                                      const OpPerformanceInfo &perfInfo,
                                      const unsigned numElems,
                                      const bool inPlace = false) {
  unsigned elemsPerLoop;
  unsigned cyclesPerLoop = perfInfo.cyclesPerLoop;
  if (perfInfo.naturalVectorWidth) {
    elemsPerLoop = target.getVectorWidth(type);
    // This is an overhead for each cycle of the inner loop that processes
    // 1 element (or 1 vector). It accounts for load/store and loop decision.
    // If we force the use of interleaved memory, if inplace, we can always
    // utilise the fast path to reduce the overhead for each cycle by 1
    cyclesPerLoop += hasExternalCodelet(op, type)
                         ? (getForceInterleavedEstimates() && inPlace ? 1 : 2)
                         : 5;
  } else {
    elemsPerLoop = perfInfo.loopUnrollFactor;
  }
  unsigned numLoops = iceil(numElems, elemsPerLoop);
  return numLoops * cyclesPerLoop;
}
} // namespace internal

static unsigned flopsPerBinaryOpElement(BinaryOpType op) {
  if (op == BinaryOpType::BITWISE_AND || op == BinaryOpType::BITWISE_OR ||
      op == BinaryOpType::BITWISE_XOR || op == BinaryOpType::BITWISE_XNOR ||
      op == BinaryOpType::SHIFT_LEFT || op == BinaryOpType::SHIFT_RIGHT ||
      op == BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND) {
    return 0;
  }
  // treat flops per divide as single flop even though it is attributed with
  // more flops in some contexts.
  // And remainder as 3 flops.
  if (op == BinaryOpType::REMAINDER) {
    return 3;
  }
  if (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV ||
      op == BinaryOpType::INV_STD_DEV_TO_VARIANCE) {
    return 2;
  }
  if (op == BinaryOpType::ADD) {
    return flopsForAdd();
  }
  if (op == BinaryOpType::MULTIPLY) {
    return flopsForMultiply();
  }
  return 1;
}

static unsigned flopsPerUnaryOpElement(UnaryOpType op) {
  if (op == UnaryOpType::BITWISE_NOT ||
      op == UnaryOpType::COUNT_LEADING_ZEROS ||
      op == UnaryOpType::LOGICAL_NOT || op == UnaryOpType::POPCOUNT) {
    return 0;
  }
  return 1;
}

namespace internal {

// Computes the cycles used by one of the scalar broadcast MultiVertex codelets
VertexPerfEstimate
broadcastArithmetic1DCycleEstimate(const Target &target, BinaryOpType op,
                                   const Type &inType, const Type &outType,
                                   bool inPlace, std::size_t dataSize) {

  auto isVarianceConversionBinaryOp = [](BinaryOpType &op) {
    return (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) ||
           (op == BinaryOpType::INV_STD_DEV_TO_VARIANCE);
  };
  bool use2TypesVertex =
      isVarianceConversionBinaryOp(op) && (inType != outType);

  std::uint64_t overheadPerLoop;
  Type type = inType;
  if (inPlace) {
    overheadPerLoop = hasExternalCodelet(op, type)
                          ? (getForceInterleavedEstimates() ? 0 : 1)
                          : 4;
  } else if (use2TypesVertex) {
    // For vectorisation purposes, treat this as if it always processes float,
    // as it casts internally.  An extra cycle to cast to half output
    type = FLOAT;
    overheadPerLoop = outType == FLOAT ? 0 : 1;
  } else {
    overheadPerLoop = hasExternalCodelet(op, type) ? 1 : 4;
  }

  auto numWorkers = target.getNumWorkerContexts();

  const OpPerformanceInfo &perfInfo =
      inPlace ? broadcastOpInPlacePerfInfo.at({op, type})
              : broadcastOpPerfInfo.at({op, type});

  std::uint64_t cycles = 20;

  unsigned elemsPerLoop;
  unsigned cyclesPerLoop = perfInfo.cyclesPerLoop;
  if (perfInfo.naturalVectorWidth) {
    elemsPerLoop = target.getVectorWidth(type);
    // This is an overhead for each cycle of the inner loop that processes
    // 1 element (or 1 vector). It accounts for load/store and loop decision
    cyclesPerLoop += overheadPerLoop;
  } else {
    elemsPerLoop = perfInfo.loopUnrollFactor;
  }

  auto numElems = iceil(dataSize, numWorkers);

  unsigned numLoops = iceil(numElems, elemsPerLoop);
  cycles += numLoops * cyclesPerLoop;
  std::uint64_t flops =
      static_cast<std::uint64_t>(dataSize) * flopsPerBinaryOpElement(op);
  std::uint64_t totalCycles = cycles * numWorkers + basicOpSupervisorOverhead();
  return {totalCycles, convertToTypeFlops(flops, type)};
}

} // namespace internal

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar1DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(data);
  // In the inplace case, if forcing use of interleaved memory, the fast
  // path can always be utilized to reduce the overhead by 1 cycle, making the
  // inner loop one cycle for ADD, SUB and MULTIPLY.
  return broadcastArithmetic1DCycleEstimate(target, op, type, type, true,
                                            data.size());
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar1D)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            BinaryOpType op, const Type &type) {
  CODELET_FIELD(data);
  return broadcastArithmetic1DCycleEstimate(target, op, type, type, false,
                                            data.size());
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2Types1D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const Type &outType) {
  CODELET_FIELD(data);

  return broadcastArithmetic1DCycleEstimate(target, op, type, outType, false,
                                            data.size());
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(
    BroadcastScalar1DRelationalOpDualOutput)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             BinaryOpType op, const Type &type,
                                             const Type &outType) {
  CODELET_FIELD(data);

  // The C++ codelets use only 32 bit store, half vector width.
  unsigned elemsPerLoop = target.getVectorWidth(outType) / 2;
  auto numWorkers = target.getNumWorkerContexts();
  unsigned overhead = outType == FLOAT ? 15 : 36;
  unsigned cyclesPerLoop = outType == FLOAT ? 8 : 23;

  auto numElems = iceil(data.size(), numWorkers);
  unsigned numLoops = iceil(numElems, elemsPerLoop);
  std::uint64_t cycles = 20 + overhead + (numLoops * cyclesPerLoop);
  std::uint64_t flops = static_cast<std::uint64_t>(data.size()) *
                        (flopsPerBinaryOpElement(op) +
                         flopsPerBinaryOpElement(BinaryOpType::SUBTRACT));
  std::uint64_t totalCycles = cycles * numWorkers + basicOpSupervisorOverhead();
  return {totalCycles, convertToTypeFlops(flops, type)};
}

static VertexPerfEstimate BroadcastVectorOuterCycleEstimate(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, std::uint64_t overheadPerInnerLoop,
    std::uint64_t overheadPerOuterLoop, bool byRow) {
  CODELET_SCALAR_VAL(columns, uint16_t);
  CODELET_SCALAR_VAL(rows, uint16_t);

  CODELET_FIELD(data);
  assert(isFPType(type));
  auto numWorkers = target.getNumWorkerContexts();
  auto perfInfo = broadcastOpPerfInfo.at({op, type});

  std::uint64_t cycles = overheadPerOuterLoop;

  std::uint64_t supervisorCycles = basicOpSupervisorOverhead();
  const auto cyclesPerLoop = perfInfo.cyclesPerLoop + overheadPerInnerLoop;
  auto numElems = byRow ? columns : (columns + numWorkers - 1) / numWorkers;
  if (perfInfo.naturalVectorWidth) {
    cycles +=
        basicOpLoopCycles(numElems, target.getVectorWidth(type), cyclesPerLoop);
  } else {
    cycles += cyclesPerLoop * numElems;
  }
  auto numOuterLoops = byRow ? (rows + numWorkers - 1) / numWorkers : rows;
  std::uint64_t flops =
      static_cast<std::uint64_t>(rows) * columns * flopsPerBinaryOpElement(op);
  std::uint64_t totalCycles =
      (15 + numOuterLoops * cycles) * numWorkers + supervisorCycles;
  return {totalCycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(
    BroadcastVectorOuterByColumn1DInPlace)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           BinaryOpType op, const Type &type,
                                           bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  // If forcing use of interleaved memory, and in place, the overhead here
  // can be reduced as we can utilise a ldst64pace in the inner loop.
  return BroadcastVectorOuterCycleEstimate(
      vertex, target, op, type, getForceInterleavedEstimates() ? 0 : 1,
      allowMisaligned ? 25 : 7, false);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorOuterByColumn1D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, false);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorOuterByRow1DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  // If forcing use of interleaved memory, and in place, the overhead here
  // can be reduced as we can utilise a ldst64pace in the inner loop.
  return BroadcastVectorOuterCycleEstimate(
      vertex, target, op, type, getForceInterleavedEstimates() ? 0 : 1,
      allowMisaligned ? 25 : 7, true);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorOuterByRow1D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, true);
}

namespace internal {

VertexPerfEstimate
broadcastArithmeticCycleEstimate(const Target &target, BinaryOpType op,
                                 const Type &inType, const Type &outType,
                                 bool inPlace, bool uniformScalar,
                                 const std::vector<std::size_t> &data) {
  std::uint64_t cycles = 20;
  auto isVarianceConversionBinaryOp = [](BinaryOpType &op) {
    return (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) ||
           (op == BinaryOpType::INV_STD_DEV_TO_VARIANCE);
  };
  bool use2TypesVertex =
      isVarianceConversionBinaryOp(op) && (inType != outType);
  std::uint64_t overheadPerLoop;
  Type type = inType;
  if (inPlace) {
    if (uniformScalar) {
      overheadPerLoop = hasExternalCodelet(op, type)
                            ? (getForceInterleavedEstimates() ? 0 : 1)
                            : 4;
    } else {
      overheadPerLoop = 4;
    }
  } else {
    if (uniformScalar) {
      if (use2TypesVertex) {
        // For vectorisation purposes, treat this as if it always processes
        // float as casting makes this so. An extra cycle to cast the output to
        // half.
        type = FLOAT;
        overheadPerLoop = outType == FLOAT ? 0 : 1;
      } else {
        overheadPerLoop = hasExternalCodelet(op, type) ? 1 : 4;
      }
    } else {
      if (use2TypesVertex) {
        throw poputil::poplibs_error("No broadcast scalar vertex available "
                                     "which supports mixed types and "
                                     "non-uniform-scalar broadcasted input");
      } else {
        overheadPerLoop = 4;
      }
    }
  }

  const auto perfInfo = inPlace ? broadcastOpInPlacePerfInfo.at({op, type})
                                : broadcastOpPerfInfo.at({op, type});
  unsigned elemsPerLoop;
  unsigned cyclesPerLoop = perfInfo.cyclesPerLoop;
  if (perfInfo.naturalVectorWidth) {
    elemsPerLoop = target.getVectorWidth(type);
    // This is an overhead for each cycle of the inner loop that processes
    // 1 element (or 1 vector). It accounts for load/store and loop decision
    cyclesPerLoop += overheadPerLoop;
  } else {
    elemsPerLoop = perfInfo.loopUnrollFactor;
  }

  std::uint64_t totalElems = 0;
  for (unsigned i = 0; i < data.size(); i++) {
    auto numElems = data[i];
    totalElems += numElems;
    unsigned numCycles = iceil(numElems, elemsPerLoop);
    cycles += numCycles * cyclesPerLoop;
    cycles += 28;
  }
  std::uint64_t flops = totalElems * flopsPerBinaryOpElement(op);
  return {cycles, convertToTypeFlops(flops, type)};
}

} // namespace internal

static VertexPerfEstimate
broadcastArithmeticCycleEstimate(const Target &target, BinaryOpType op,
                                 const Type &inType, const Type &outType,
                                 bool inPlace, bool uniformScalar,
                                 const FieldData &data) {
  std::vector<std::size_t> sizes(data.size());
  for (unsigned i = 0; i < data.size(); ++i) {
    sizes[i] = data[i].size();
  }
  return popops::internal::broadcastArithmeticCycleEstimate(
      target, op, inType, outType, inPlace, uniformScalar, sizes);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2DDataInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(data);
  // In the inplace case, if forcing use of interleaved memory, the fast
  // path can always be utilized to reduce the overhead by 1 cycle, making the
  // inner loop one cycle for ADD, SUB and MULTIPLY.
  return broadcastArithmeticCycleEstimate(target, op, type, type, true, true,
                                          data);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2DData)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(data);
  return popops::broadcastArithmeticCycleEstimate(target, op, type, type, false,
                                                  true, data);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2Types2DData)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &inType, const Type &outType) {
  CODELET_FIELD(data);
  return popops::broadcastArithmeticCycleEstimate(target, op, inType, outType,
                                                  false, true, data);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(data);
  return popops::broadcastArithmeticCycleEstimate(target, op, type, type, true,
                                                  false, data);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2D)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            BinaryOpType op, const Type &type) {
  CODELET_FIELD(data);
  return popops::broadcastArithmeticCycleEstimate(target, op, type, type, false,
                                                  false, data);
}

VertexPerfEstimate scaledArithmeticSupervisorCycleEstimate(
    const VertexIntrospector &vertex, const Target &target,
    const Type &dataType, const Type &dataBType, const bool isConstant,
    const bool memConstrained, const ScaledArithmeticOp operation) {
  CODELET_FIELD(A);
  CODELET_FIELD(B);
  ScaledArithmeticTargetParameters targetParams(target, dataType);
  const auto cycles = getScaledArithmeticSupervisorCycleEstimate(
      targetParams, dataType, dataBType, isConstant, memConstrained, operation,
      A.getProfilerVectorLayout(0), B.getProfilerVectorLayout(0), A.size());
  std::uint64_t flops =
      A.size() * (flopsPerBinaryOpElement(BinaryOpType::ADD) +
                  flopsPerBinaryOpElement(BinaryOpType::MULTIPLY));
  return {cycles, convertToTypeFlops(flops, dataType)};
}

// Cycles used to do one vector in the Mixed (data=half/scale=float) aXPlusbY
std::uint64_t aXPlusbYMixedCoreCycleEstimate(unsigned count) {
  std::uint64_t cycles = 0;

  cycles = 4;
  unsigned countM4 = count >= 4 ? count - 4 : 0;
  if (countM4) {
    unsigned rptCount = countM4 / 2 - 1;
    cycles += 11 + (rptCount * 5) + 4;

    if (countM4 & 1) {
      cycles += 9;
    }
  } else {
    // less than 4
    cycles += 1; // brz
    if (count == 1) {
      cycles += 4 + 10;
    } else if (count == 2) {
      cycles += 12 + 1;
    } else if (count == 3) {
      cycles += 12 + 10;
    }
  }
  cycles += 1; // final bri
  return cycles;
}

// aX Plus BY vertices where the data is half and the scale coeffs are float
VertexPerfEstimate aXPlusbYMixedSupervisorCycleEstimate(
    const VertexIntrospector &vertex, const Target &target,
    const bool isConstant, const bool memConstrained) {
  CODELET_FIELD(A);
  std::uint64_t supervisorCycles = 0;
  const bool scaledPtr64 =
      A.getProfilerVectorLayout(0) == layout::Vector::ScaledPtr64;

  if (isConstant) {
    supervisorCycles += 9 + 5;
  } else {
    supervisorCycles += memConstrained ? 2 + 5 : 1;
    supervisorCycles += scaledPtr64 ? 12 : 6;
    supervisorCycles += 10;
    supervisorCycles += 15 * 6; // checkAccuracy thread
    supervisorCycles += 9 + 5;
  }

  // common 'VERTEX(supervisor)' code
  const auto numWorkers = target.getNumWorkerContexts();
  const unsigned atomSize = 2;
  const unsigned count = (A.size() / numWorkers / atomSize) * atomSize;
  const unsigned final = A.size() % numWorkers;
  const unsigned rem =
      (A.size() / numWorkers) % numWorkers + iceil(final, atomSize);

  supervisorCycles += 28 + (scaledPtr64 ? 2 : 0);
  if (final == 0)
    supervisorCycles += (6 - 1); // brz $final, 1f

  std::vector<unsigned> workerCycles(numWorkers);
  for (unsigned wid = 0; wid < numWorkers; ++wid) {
    unsigned workerCount =
        count + ((wid <= rem) ? atomSize : 0) + ((wid == rem) ? final : 0);

    workerCycles[wid] = 19 + aXPlusbYMixedCoreCycleEstimate(workerCount);
    if (wid == rem)
      workerCycles[wid] += 1; // brz $mscratch, 1f
  }

  auto maxWorkerCycles =
      *std::max_element(std::begin(workerCycles), std::end(workerCycles));
  std::uint64_t flops = static_cast<std::uint64_t>(A.size()) *
                        (2 * flopsPerBinaryOpElement(BinaryOpType::MULTIPLY) +
                         flopsPerBinaryOpElement(BinaryOpType::ADD));
  const auto cycles = supervisorCycles + maxWorkerCycles * numWorkers;
  return {cycles, flops};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScaledAddSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const Type &BType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, BType,
                                                 isConstant, memConstrained,
                                                 ScaledArithmeticOp::ADD);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScaledSubtractSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const Type &BType, const Type &ScaleType, const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, BType,
                                                 false, memConstrained,
                                                 ScaledArithmeticOp::SUBTRACT);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(aXPlusbYSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &DataType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  if (DataType == HALF && ScaleType == FLOAT)
    return aXPlusbYMixedSupervisorCycleEstimate(vertex, target, isConstant,
                                                memConstrained);
  else
    return scaledArithmeticSupervisorCycleEstimate(
        vertex, target, DataType, DataType, isConstant, memConstrained,
        ScaledArithmeticOp::AXPLUSBY);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(aXMinusbYSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &DataType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  if (DataType == HALF && ScaleType == FLOAT)
    // Other than for the minus, the plus and minus mixed supervisors are the
    // same.
    return aXPlusbYMixedSupervisorCycleEstimate(vertex, target, isConstant,
                                                memConstrained);
  else
    return scaledArithmeticSupervisorCycleEstimate(
        vertex, target, DataType, ScaleType, isConstant, memConstrained,
        ScaledArithmeticOp::AXMINUSBY);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(XMinusaXPlusbYSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const bool isConstant, const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, AType,
                                                 isConstant, memConstrained,
                                                 ScaledArithmeticOp::AXPLUSBY);
}

VertexPerfEstimate ScaledArithmetic2DCycleEstimate(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isConstant, const bool memConstrained,
    const ScaledArithmeticOp operation) {
  CODELET_FIELD(A);
  CODELET_FIELD(B);

  const auto aLayout = A.getProfilerVectorLayout(0);
  const auto bLayout = B.getProfilerVectorLayout(0);

  if (type == INT || type == UNSIGNED_INT) {
    std::uint64_t cycles = 8; // prologue and epilogue overhead.
    for (unsigned i = 0; i < A.size(); ++i) {
      cycles += 7                    // outer loop constant overhead
                + (A[i].size() * 5); // inner loop
    }
    if (!isConstant)
      cycles += 1;
    if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant)
      cycles += 1;
    return cycles;
  } else {
    assert(type == HALF || type == FLOAT);
  }

  const bool isFloataXPbY = (operation == ScaledArithmeticOp::AXPLUSBY ||
                             operation == ScaledArithmeticOp::AXMINUSBY) &&
                            type == FLOAT;

  unsigned innerLoopCycles = memConstrained && !isFloataXPbY ? 2 : 3;
  if (getForceInterleavedEstimates() && !isFloataXPbY) {
    innerLoopCycles -= 1;
  }
  const auto grain = target.getVectorWidth(type);
  std::uint64_t cycles = 9; // prologue and epilogue overhead.
  if (!isConstant)
    cycles += 1;
  if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant)
    cycles += 2;
  if (operation == ScaledArithmeticOp::AXPLUSBY && !isConstant)
    cycles += 6;
  if (operation == ScaledArithmeticOp::AXPLUSBY && isConstant)
    cycles += 4;

  std::uint64_t totalElems = 0;
  for (unsigned i = 0; i < A.size(); ++i) {
    // outer loop constant overhead
    cycles += 15;
    if (aLayout == layout::Vector::ShortSpan) {
      cycles += poputil::internal::getUnpackCost(bLayout);
    }

    cycles += (A[i].size() / grain != 0 ? 5 : 0)         // inner loop overhead
              + (A[i].size() / grain * innerLoopCycles); // inner loop

    if (type == FLOAT) {
      cycles += (A[i].size() % grain != 0 ? 7 : 0); // last element.
    } else {
      auto rem = A[i].size() % grain;
      cycles += (rem > 0 ? 4 : 0)         // remainder overhead
                + (rem >= 2 ? 6 : 0)      // process 2 more at end.
                + (rem % 2 == 1 ? 7 : 0); // process last element.
    }
    totalElems += A[i].size();
  }
  std::uint64_t flops =
      totalElems * (flopsPerBinaryOpElement(BinaryOpType::MULTIPLY) +
                    flopsPerBinaryOpElement(BinaryOpType::ADD));
  return {cycles, convertToTypeFlops(flops, type)};
}

// aX Plus BY vertices where the data is half and the scale coeffs are float
VertexPerfEstimate
aXPlusbYMixed2DCycleEstimate(const VertexIntrospector &vertex,
                             const Target &target, const bool isConstant,
                             const bool memConstrained) {
  CODELET_FIELD(A);
  CODELET_FIELD(B);
  std::uint64_t cycles = 0;
  const auto layoutA = A.getProfilerVectorLayout(1);
  const auto layoutB = B.getProfilerVectorLayout(1);
  const bool shortSpan = (layoutA == layout::Vector::ShortSpan);
  const bool scaledPtr64 = (layoutB == layout::Vector::ScaledPtr64);

  if (!isConstant) {
    cycles += memConstrained ? 2 : 1;
    cycles += 15;
  } else {
    cycles += 2;
  }
  cycles += 6;
  unsigned rowLoopCycles = 2 + (shortSpan ? 4 : 2) + (scaledPtr64 ? 2 : 1);
  for (unsigned i = 0; i < A.size(); i++) {
    cycles += rowLoopCycles * aXPlusbYMixedCoreCycleEstimate(A[i].size());
  }
  std::uint64_t flops = static_cast<std::uint64_t>(A.size()) *
                        (2 * flopsPerBinaryOpElement(BinaryOpType::MULTIPLY) +
                         flopsPerBinaryOpElement(BinaryOpType::ADD));
  return {cycles, flops};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScaledAdd2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const Type &BType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, AType, memConstrained,
                                         isConstant, ScaledArithmeticOp::ADD);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScaledSubtract2D)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &DataType, const Type &ScaleType, const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, DataType, false,
                                         memConstrained,
                                         ScaledArithmeticOp::SUBTRACT);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(aXPlusbY2D)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &DataType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  if (DataType == HALF && ScaleType == FLOAT)
    return aXPlusbYMixed2DCycleEstimate(vertex, target, isConstant,
                                        memConstrained);
  else
    return ScaledArithmetic2DCycleEstimate(vertex, target, DataType,
                                           memConstrained, isConstant,
                                           ScaledArithmeticOp::AXPLUSBY);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(aXMinusbY2D)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &DataType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  if (DataType == HALF && ScaleType == FLOAT)
    // Other than for the minus, the plus and minus workers are the same.
    return aXPlusbYMixed2DCycleEstimate(vertex, target, isConstant,
                                        memConstrained);
  else
    return ScaledArithmetic2DCycleEstimate(vertex, target, DataType,
                                           memConstrained, isConstant,
                                           ScaledArithmeticOp::AXMINUSBY);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(XMinusaXPlusbY2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isConstant, const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, memConstrained,
                                         isConstant,
                                         ScaledArithmeticOp::AXPLUSBY);
}

// Exact worker cycle count for VectorInnerAdd_core_float
std::uint64_t vectorInnerAddCoreCycles_float(unsigned vectorWidth,
                                             unsigned addendLen,
                                             unsigned blockCount) {
  std::uint64_t cycles = 1; // brz .Lreturn

  if (blockCount != 0) {
    cycles += 5; // after brz, before loop

    for (unsigned i = 0; i < addendLen; ++i) {
      cycles += 3;              // start of loop
      cycles += 2 * blockCount; // rpt loop
      cycles += 5;              // end of loop
    }
  }
  return cycles + 1; // return
}

std::uint64_t vectorInnerAddCoreCycles_half_scalar(unsigned addendLen,
                                                   unsigned blockCount) {
  std::uint64_t cycles = 5; // pre-loop
  // Aligned loop bodies take 8 cycles, misaligned take 10, but they are
  // equally numerous so it averages to 9.
  cycles += addendLen * (2 + blockCount * 9 + 3);
  return cycles + 1; // return
}
std::uint64_t vectorInnerAddCoreCycles_half_multiple_of_8(unsigned vectorWidth,
                                                          unsigned addendLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 2; // add, brneg

  // Due to retreiving outputs from accumulators we have a pipeline length
  // that means we process 2 vectors at a time.
  unsigned grainSize = vectorWidth * 2;
  unsigned addendLoops = addendLen / grainSize;
  const unsigned remainder = addendLen % grainSize;
  for (unsigned i = gccs::ceilLog2(8); i < gccs::ceilLog2(grainSize); ++i) {
    if (remainder & (1u << i)) {
      ++addendLoops;
    }
  }

  if (blockCount == 1) {
    cycles += 3 + 7 * addendLoops + 1;
  } else {
    cycles += 4;                                    // after brneg, pre-loop
    cycles += addendLoops * (8 +                    // pre-rpt
                             2 * (blockCount - 1) + // rpt body
                             // post-rpt
                             7) +
              1; // return
  }
  return cycles;
}
std::uint64_t vectorInnerAddCoreCycles_half_multiple_of_4(unsigned vectorWidth,
                                                          unsigned addendLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 5; // pre-loop

  unsigned addendLoops = addendLen / vectorWidth;
  const unsigned remainder = addendLen % vectorWidth;
  for (unsigned i = gccs::ceilLog2(4); i < gccs::ceilLog2(vectorWidth); ++i) {
    if (remainder & (1u << i)) {
      ++addendLoops;
    }
  }
  cycles += addendLoops *
            (7 +                        // pre-rpt
             2 * (blockCount / 2 - 1) + // rpt body
             // post-rpt. The code depends on whether or not blockCount was odd
             1 + (blockCount % 2) + 5);
  return cycles + 1; // return
}

// Exact worker cycle count for VectorInnerMul_core_half
std::uint64_t vectorInnerAddCoreCycles_half(unsigned vectorWidth,
                                            unsigned addendLen,
                                            unsigned blockCount) {
  std::uint64_t cycles = 1; // brz
  if (blockCount == 0)
    return cycles;

  cycles += 2; // cmpult > 2048, brz
  if (addendLen > 2048) {
    return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 8 == 0) {
    return cycles + vectorInnerAddCoreCycles_half_multiple_of_8(
                        vectorWidth, addendLen, blockCount);
  }

  cycles += 2; // cmpult, brnz
  if (blockCount < 2) {
    return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 4 == 0) {
    return cycles + vectorInnerAddCoreCycles_half_multiple_of_4(
                        vectorWidth, addendLen, blockCount);
  }
  return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
}

// Cycle count for the common part of all the VectorInner2D ADD and
// SUBTRACT codelets (from the .Lworker2d label)
std::uint64_t vectorInner2DAddCycles(
    unsigned vectorWidth, uint32_t n, const std::vector<uint32_t> &BLen,
    const std::vector<uint32_t> &dataBlockCount, const Type &type) {

  if (BLen.size() != n || dataBlockCount.size() != n) {
    throw poputil::poplibs_error("n (" + std::to_string(n) +
                                 ") does not "
                                 "match BLen or dataBlockCount "
                                 "length (" +
                                 std::to_string(BLen.size()) + " & " +
                                 std::to_string(dataBlockCount.size()) +
                                 " respectively) in Broadcast ADD vertex");
  }

  std::uint64_t numCycles = 3; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    // loop overhead. A bit more for halves
    if (type == HALF)
      numCycles += 15;
    else
      numCycles += 9;

    auto coreFunc = type == HALF ? vectorInnerAddCoreCycles_half
                                 : vectorInnerAddCoreCycles_float;

    numCycles += coreFunc(vectorWidth, BLen[i], dataBlockCount[i]);
  }

  return numCycles + 1; // exitnz
}

// Cycle count for the common part of all the VectorInnerMultiVertex ADD and
// SUBTRACT codelets
std::uint64_t vectorInnerMultiVertexAddCycles(unsigned numWorkerContexts,
                                              unsigned vectorWidth,
                                              uint32_t BLen,
                                              uint16_t dataBlockCountPacked,
                                              const Type &type) {

  // Need to get the max number of blocks that a worker will do.
  // Extract quotient and remainder from dataBlockCountPacked. The workers
  // will do 'quotient' blocks, but if the remainder is nonzero, 'remainder'
  // workers will do one extra block, so that will be the max block count.
  auto quotient = dataBlockCountPacked >> 3;
  auto remainder = dataBlockCountPacked & 0x3;

  auto maxBlocksPerWorker = quotient + (remainder != 0);

  // Common supervisor overhead:
  // * setzi for worker entry
  // * runall
  // * br $lr
  std::uint64_t numCycles = 1 + 6 * 2;

  // Worker cycles in common part (from the .Lworker label).
  std::uint64_t maxWorkerCycles = 0;
  if (type == HALF) {
    maxWorkerCycles = 19;
    if (BLen & 1) {
      maxWorkerCycles += 4;
      if (quotient == 0) {
        maxWorkerCycles += 1;
      }
      // 3 max as one worker will probably have to round down
      // its number of blocks.
      maxWorkerCycles += 3;
    }
  } else {
    maxWorkerCycles = 17;
  }

  auto coreFunc = type == HALF ? vectorInnerAddCoreCycles_half
                               : vectorInnerAddCoreCycles_float;

  maxWorkerCycles += coreFunc(vectorWidth, BLen, maxBlocksPerWorker);

  return numCycles + maxWorkerCycles * numWorkerContexts;
}

std::uint64_t vectorInnerDivCoreCycles_float(unsigned BLen,
                                             unsigned blockCount) {
  // NOTE: We use 3 cycles for each 'f32div' instruction (which takes
  // 1 or 3 cycles depending on input)
  const unsigned f32divCycles = 3;

  std::uint64_t cycles = 1; // brz .Lreturn
  if (blockCount != 0) {
    cycles += 5; // down to brz .Lfast_path
    if (BLen & 1) {
      // odd BLen
      cycles += 2 + BLen * (3 + (blockCount - 1) * (f32divCycles + 1) +
                            f32divCycles + 3);
    } else {
      // even BLen
      const unsigned pairs = BLen / 2;
      cycles += 3 + pairs * (2 + f32divCycles +
                             (blockCount - 1) * (2 * f32divCycles) +
                             f32divCycles + 3);
    }
  }
  return cycles + 1; // return
}

std::uint64_t vectorInnerDivCoreCycles_half(unsigned BLen,
                                            unsigned blockCount) {
  // NOTE: We use 3 cycles for each 'f32div' instruction (which takes
  // 1 or 3 cycles depending on input)
  const unsigned f32divCycles = 3;

  std::uint64_t cycles = 1; // brz $data_block_count, .Lreturn
  if (blockCount != 0) {
    if (BLen == 1) {
      // BLen==1 (like a BroadcastScalar!)
      cycles += 7;
      if (blockCount == 1) {
        cycles += 7 + 2 + f32divCycles + 3;
      } else {
        const unsigned pairs = blockCount / 2;
        const unsigned innerCycles = 2 + 2 * f32divCycles;
        cycles += 10 + f32divCycles * 2 + (pairs - 1) * innerCycles + 3;
        if (blockCount & 1) {
          cycles += 2 + f32divCycles + 3;
        }
      }
    } else if (BLen & 1) {
      // odd BLen
      cycles += 8;
      const unsigned rptCount = BLen / 2 - 1;
      const unsigned inner1 =
          3 + f32divCycles * 2 + rptCount * (3 + f32divCycles * 2) + 3;
      const unsigned inner2 = 5 + f32divCycles * 2 + 1 +
                              rptCount * (4 + f32divCycles * 2) + 3 +
                              f32divCycles * 2 + 3;
      cycles += blockCount / 2 * (inner1 + inner2) +
                ((blockCount & 1) ? (inner1 + 2 + f32divCycles + 3) : 0);
    } else {
      // even BLen, we process in pairs of values
      const unsigned pairs = BLen / 2;
      const unsigned innerCycles = 2 + 2 * f32divCycles;
      cycles +=
          8 +
          pairs * (3 + f32divCycles * 2 + (blockCount - 1) * innerCycles + 4) +
          4;
    }
  }
  return cycles + 1; // return
}

// Cycle count for the common part of all the VectorInner2D DIVIDE codelets
// (from the .Lworker2d label)
std::uint64_t
vectorInner2DDivCycles(uint32_t n, const std::vector<uint32_t> &BLen,
                       const std::vector<uint32_t> &dataBlockCount,
                       const Type &type) {

  if (BLen.size() != n || dataBlockCount.size() != n) {
    throw poputil::poplibs_error("n (" + std::to_string(n) +
                                 ") does not "
                                 "match BLen or dataBlockCount "
                                 "length (" +
                                 std::to_string(BLen.size()) + " & " +
                                 std::to_string(dataBlockCount.size()) +
                                 " respectively) in Broadcast ADD vertex");
  }

  std::uint64_t numCycles = 6; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    // loop overhead. A bit more for halves
    if (type == HALF)
      numCycles += 15;
    else
      numCycles += 9;

    auto coreFunc = type == HALF ? vectorInnerDivCoreCycles_half
                                 : vectorInnerDivCoreCycles_float;

    numCycles += coreFunc(BLen[i], dataBlockCount[i]);
  }

  return numCycles + 1; // exitnz
}

// Cycle count for the common part of all the VectorInnerMultiVertex DIV
// codelets.
std::uint64_t vectorInnerMultiVertexDivCycles(unsigned numWorkerContexts,
                                              uint32_t BLen,
                                              uint16_t dataBlockCountPacked,
                                              const Type &type) {
  // These numbers may not be exact (e.g. the remainder of
  // dataBlockCountPacked is ignored).

  // Supervisor overhead.
  std::uint64_t numCycles = 1 + 6 + 6;

  // We need to count the *maximum* block per worker.
  auto remainder = dataBlockCountPacked & 0x7;
  auto blocksPerWorker = (dataBlockCountPacked >> 3) + (remainder ? 1 : 0);

  // Worker cycles (from the .Lworker label)
  numCycles += numWorkerContexts * (type == HALF ? 24 : 19);

  auto coreFunc = type == HALF ? vectorInnerDivCoreCycles_half
                               : vectorInnerDivCoreCycles_float;

  numCycles += numWorkerContexts * coreFunc(BLen, blocksPerWorker);

  // Exit
  numCycles += 1;

  return numCycles;
}
// Exact worker cycle count for VectorInnerMul_core_float
std::uint64_t vectorInnerMulCoreCycles_float(unsigned vectorWidth,
                                             unsigned scaleLen,
                                             unsigned blockCount,
                                             bool inPlace) {
  std::uint64_t cycles = 1; // return

  ++cycles; // brz

  if (blockCount == 0)
    return cycles;

  cycles += 5; // before loop

  for (unsigned i = 0; i < scaleLen; ++i) {
    cycles += 3;              // start of loop
    cycles += 2 * blockCount; // rpt loop
    cycles += 5;              // end of loop
  }
  return cycles;
}

std::uint64_t vectorInnerMulCoreCycles_half_scalar(unsigned scaleLen,
                                                   unsigned blockCount) {
  std::uint64_t cycles = 4; // pre-loop
  // Aligned loop bodies take 8 cycles, misaligned take 10, but they are
  // equally numerous so it averages to 9.
  cycles += scaleLen * (5 + blockCount * 9);
  cycles += 1; // return
  return cycles;
}

std::uint64_t vectorInnerMulCoreCycles_half_multiple_of_4(unsigned vectorWidth,
                                                          unsigned scaleLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 3; // pre-loop
  unsigned scaleLoops = scaleLen / vectorWidth;
  unsigned remainder = scaleLen % vectorWidth;
  for (unsigned i = gccs::ceilLog2(4); i < gccs::ceilLog2(vectorWidth); ++i) {
    if (remainder & (1u << i)) {
      ++scaleLoops;
    }
  }

  // 2 cycles per-block because we do not load and store simultaneously
  // in this path.
  cycles += scaleLoops * (4 + 2 * blockCount + 3) + 1; // return
  return cycles;
}

std::uint64_t vectorInnerMulCoreCycles_half_multiple_of_4_pipeline(
    unsigned vectorWidth, unsigned scaleLen, unsigned blockCount) {
  std::uint64_t cycles = 3; // pre-loop

  unsigned scaleLoops = scaleLen / vectorWidth;
  unsigned remainder = scaleLen % vectorWidth;
  for (unsigned i = gccs::ceilLog2(4); i < gccs::ceilLog2(vectorWidth); ++i) {
    if (remainder & (1u << i)) {
      ++scaleLoops;
    }
  }
  // 1 cycle per-block because we can load/store simultaneously.
  cycles +=
      scaleLoops * ((blockCount == 1) ? 7 : 6 + blockCount + 3) + 1; // return
  return cycles;
}

// Exact worker cycle count for VectorInnerMul_core_half
std::uint64_t vectorInnerMulCoreCycles_half(unsigned vectorWidth,
                                            unsigned scaleLen,
                                            unsigned blockCount, bool inPlace) {

  std::uint64_t cycles = 1; // initial check for 0

  cycles += 2; // check for multiple of four
  if (scaleLen % 4 != 0) {
    return cycles + vectorInnerMulCoreCycles_half_scalar(scaleLen, blockCount);
  }

  cycles += 2; // check for in place
  if (inPlace) {
    return cycles + vectorInnerMulCoreCycles_half_multiple_of_4(
                        vectorWidth, scaleLen, blockCount);
  }

  cycles += 2; // check for > 2044
  if (scaleLen > 2044) {
    return cycles + vectorInnerMulCoreCycles_half_multiple_of_4(
                        vectorWidth, scaleLen, blockCount);
  }

  cycles += 2; // Check for > 1
  if (blockCount < 2) {
    return cycles + vectorInnerMulCoreCycles_half_multiple_of_4(
                        vectorWidth, scaleLen, blockCount);
  }

  return cycles + vectorInnerMulCoreCycles_half_multiple_of_4_pipeline(
                      vectorWidth, scaleLen, blockCount);
}

// Cycle count for the common part of all the VectorInner2D MUL
// codelets (from the .Lworker2d label)
std::uint64_t vectorInner2DMulCycles(
    unsigned vectorWidth, uint32_t n, const std::vector<uint32_t> &BLen,
    const std::vector<uint32_t> &dataBlockCount, const Type &type) {

  if (BLen.size() != n || dataBlockCount.size() != n) {
    throw poputil::poplibs_error("n (" + std::to_string(n) +
                                 ") does not "
                                 "match BLen or dataBlockCount "
                                 "length (" +
                                 std::to_string(BLen.size()) + " & " +
                                 std::to_string(dataBlockCount.size()) +
                                 " respectively) in Broadcast MUL vertex");
  }

  std::uint64_t numCycles = 3; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    numCycles += type == HALF ? 13 : 9; // loop overhead.

    auto coreFunc = type == HALF ? vectorInnerMulCoreCycles_half
                                 : vectorInnerMulCoreCycles_float;

    numCycles += coreFunc(vectorWidth, BLen[i], dataBlockCount[i], false);
  }

  // Exit
  numCycles += 1;

  return numCycles;
}

// Cycle count for the common part of all the VectorInnerMultiVertex MUL
// codelets.
std::uint64_t vectorInnerMultiVertexMulCycles(unsigned numWorkerContexts,
                                              unsigned vectorWidth,
                                              uint32_t BLen,
                                              uint16_t dataBlockCountPacked,
                                              const Type &type, bool inPlace) {
  // These numbers may not be exact (e.g. the remainder of
  // dataBlockCountPacked is ignored).

  // Common supervisor overhead:
  // * setzi for worker entry
  // * runall
  // * br $lr
  std::uint64_t numCycles = 1 + 6 * 2;

  auto quotient = dataBlockCountPacked >> 3;
  auto remainder = dataBlockCountPacked & 3;
  const auto maxBlocksPerWorker = quotient + (remainder != 0);

  // Worker cycles (from the .Lworker label)
  std::uint64_t maxWorkerCycles = 0;
  if (type == HALF) {
    maxWorkerCycles = 19;
    if (BLen & 1) {
      maxWorkerCycles += 4;
      if (quotient == 0) {
        maxWorkerCycles += 1;
      }
      maxWorkerCycles += 3;
    }
  } else {
    maxWorkerCycles = 17;
  }
  numCycles += numWorkerContexts * (type == HALF ? 24 : 17);

  auto coreFunc = type == HALF ? vectorInnerMulCoreCycles_half
                               : vectorInnerMulCoreCycles_float;

  maxWorkerCycles += coreFunc(vectorWidth, BLen, maxBlocksPerWorker, inPlace);

  return numCycles + maxWorkerCycles * numWorkerContexts;
}

static std::uint64_t blockLengthFromPacked(unsigned packedBlockLength,
                                           unsigned numWorkers) {
  unsigned quotient = packedBlockLength >> 3;
  unsigned remainder = packedBlockLength & 0x3;
  return static_cast<std::uint64_t>(quotient) * numWorkers + remainder;
}

static std::uint64_t flopsForBinaryOp2D(unsigned numElems, const Type &type,
                                        BinaryOpType op) {
  return convertToTypeFlops(numElems * flopsPerBinaryOpElement(op), type);
}

static std::uint64_t flopsForUnaryOp2D(unsigned numElems, const Type &type,
                                       UnaryOpType op) {
  return convertToTypeFlops(
      static_cast<std::uint64_t>(numElems) * flopsPerUnaryOpElement(op), type);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorInner1D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  unsigned numWorkerContexts = target.getNumWorkerContexts();
  unsigned vectorWidth = target.getVectorWidth(type);
  auto flops = flopsForBinaryOp2D(
      BLen * blockLengthFromPacked(dataBlockCountPacked, numWorkerContexts),
      type, op);
  // Additional branch in the supervisor, and preamble instructions in the
  // worker part.
  switch (op) {
  case BinaryOpType::ADD: {
    const unsigned addedSuperOverhead = 6;
    const unsigned addedWorkerOverhead = 3;
    return {vectorInnerMultiVertexAddCycles(numWorkerContexts, vectorWidth,
                                            BLen, dataBlockCountPacked, type) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::DIVIDE: {
    return {vectorInnerMultiVertexDivCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) +
                1 + 3,
            flops};
  }
  case BinaryOpType::SUBTRACT: {
    const unsigned addedSuperOverhead = 6;
    const unsigned addedWorkerOverhead = 3;
    return {vectorInnerMultiVertexAddCycles(numWorkerContexts, vectorWidth,
                                            BLen, dataBlockCountPacked, type) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::MULTIPLY: {
    const unsigned addedSuperOverhead = 0;
    const unsigned addedWorkerOverhead = 2;
    return {vectorInnerMultiVertexMulCycles(numWorkerContexts, vectorWidth,
                                            BLen, dataBlockCountPacked, type,
                                            false) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorInner1DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  const unsigned numWorkerContexts = target.getNumWorkerContexts();
  const unsigned vectorWidth = target.getVectorWidth(type);
  auto flops = flopsForBinaryOp2D(
      BLen * blockLengthFromPacked(dataBlockCountPacked, numWorkerContexts),
      type, op);

  switch (op) {
  case BinaryOpType::ADD: {
    const auto addedWorkerOverhead = 2;
    return {vectorInnerMultiVertexAddCycles(numWorkerContexts, vectorWidth,
                                            BLen, dataBlockCountPacked, type) +
                addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::DIVIDE: {
    return {vectorInnerMultiVertexDivCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) +
                2,
            flops};
  }
  case BinaryOpType::SUBTRACT: {
    const auto addedSuperOverhead = 6;
    const auto addedWorkerOverhead = 3;
    // Additional branches in the supervisor and worker part.
    return {vectorInnerMultiVertexAddCycles(numWorkerContexts, vectorWidth,
                                            BLen, dataBlockCountPacked, type) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::MULTIPLY: {
    const unsigned addedSuperOverhead = 6;
    const unsigned addedWorkerOverhead = 3;
    return {vectorInnerMultiVertexMulCycles(numWorkerContexts, vectorWidth,
                                            BLen, dataBlockCountPacked, type,
                                            true) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

// Broadcast inner vector worklist desconstruction
struct BroadcastInnerVec2DSizes {
  uint32_t n;
  std::vector<uint32_t> BLen;
  std::vector<uint32_t> dataBlockCount;
};

static BroadcastInnerVec2DSizes
deconstructWorkList(const std::vector<uint32_t> &workList) {
  BroadcastInnerVec2DSizes sizes;
  // encoded as 1 less
  sizes.n = 1 + workList.at(0);
  sizes.BLen.reserve(sizes.n);
  sizes.dataBlockCount.reserve(sizes.n);
  assert(workList.size() == 1 + 2 * sizes.n);
  for (unsigned i = 0; i != sizes.n; ++i) {
    sizes.BLen.push_back(workList[1 + 2 * i]);
    sizes.dataBlockCount.push_back(workList[2 * (i + 1)]);
  }
  return sizes;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorInner2D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_VECTOR_VALS(workList, uint32_t);
  const unsigned vectorWidth = target.getVectorWidth(type);
  const auto sizes = deconstructWorkList(workList);
  const auto &n = sizes.n;
  const auto &BLen = sizes.BLen;
  const auto &dataBlockCount = sizes.dataBlockCount;
  std::uint64_t totalElems = 0;
  for (unsigned i = 0; i != sizes.BLen.size(); ++i) {
    totalElems +=
        BLen[i] * blockLengthFromPacked(sizes.dataBlockCount[i],
                                        target.getNumWorkerContexts());
  }
  auto flops = flopsForBinaryOp2D(totalElems, type, op);

  switch (op) {
  case BinaryOpType::SUBTRACT:
    return {vectorInner2DAddCycles(vectorWidth, n, BLen, dataBlockCount, type) +
                4,
            flops};
  case BinaryOpType::ADD:
    // an additional branch at the start.
    return {vectorInner2DAddCycles(vectorWidth, n, BLen, dataBlockCount, type) +
                3,
            flops};
  case BinaryOpType::DIVIDE:
    // an additional branch at the start.
    return {vectorInner2DDivCycles(n, BLen, dataBlockCount, type) + 3, flops};
  case BinaryOpType::MULTIPLY:
    return {vectorInner2DMulCycles(vectorWidth, n, BLen, dataBlockCount, type) +
                2,
            flops};
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorInner2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_VECTOR_VALS(workList, uint32_t);
  const unsigned vectorWidth = target.getVectorWidth(type);
  std::uint64_t totalElems = 0;
  const auto sizes = deconstructWorkList(workList);
  const auto &n = sizes.n;
  const auto &BLen = sizes.BLen;
  const auto &dataBlockCount = sizes.dataBlockCount;

  auto flops = flopsForBinaryOp2D(totalElems, type, op);

  switch (op) {
  case BinaryOpType::SUBTRACT:
    return {vectorInner2DAddCycles(vectorWidth, n, BLen, dataBlockCount, type) +
                4,
            flops};
  case BinaryOpType::ADD:
    // an additional branch at the start.
    return {vectorInner2DAddCycles(vectorWidth, n, BLen, dataBlockCount, type) +
                2,
            flops};
  case BinaryOpType::DIVIDE:
    // an additional branch at the start.
    return {vectorInner2DDivCycles(n, BLen, dataBlockCount, type) + 2, flops};
  case BinaryOpType::MULTIPLY:
    return {vectorInner2DMulCycles(vectorWidth, n, BLen, dataBlockCount, type) +
                3,
            flops};
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(HadamardProd)(const VertexIntrospector &vertex,
                                       const Target &target, const Type &type) {
  uint64_t cycles = 5;
  const auto A = vertex.getFieldInfo("A");
  CODELET_FIELD(B);
  assert(A.size() == B.size());
  unsigned totalElems = 0;
  for (unsigned i = 0; i < A.size(); ++i) {
    assert(A[i].size() == B[i].size());
    unsigned numElem = A[i].size();
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;
    cycles += 5 + (1 + numVectors * 2);
    totalElems += numElem;
  }
  auto flops = flopsForBinaryOp2D(totalElems, type, BinaryOpType::MULTIPLY);
  return {cycles, flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Fill)(const VertexIntrospector &vertex,
                               const Target &target, const Type &type) {
  CODELET_FIELD(out);
  return getFill1DCycleEstimate(FillTargetParameters(target), type, out.size());
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Fill2d)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &type) {
  // UNSIGNED_INT and INT use the same cycle estimator as FLOAT
  // see: T24721
  CODELET_FIELD(out);
  std::vector<unsigned> numElems;
  numElems.reserve(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    numElems.emplace_back(out[i].size());
  }
  return getFill2DCycleEstimate(FillTargetParameters(target), type, numElems);
}

static std::uint64_t castFlops(const Type &fromType, const Type &toType,
                               unsigned numElems) {

  return isFPType(fromType) || isFPType(toType)
             ? static_cast<std::uint64_t>(numElems) * 1
             : 0;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(Cast1DSingleWorker)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fromType, const Type &toType) {
  CODELET_SCALAR_VAL(numElems, unsigned);

  const CastTargetParameters targetParams{target, fromType, toType};
  return {getCast1DSingleWorkerCycleEstimate(targetParams, fromType, toType,
                                             numElems),
          castFlops(fromType, toType, numElems)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Cast1D)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &fromType,
                                 const Type &toType) {
  CODELET_SCALAR_VAL(numElems, unsigned);
  const CastTargetParameters castTargetParams{target, fromType, toType};

  return {getCast1DCycleEstimate(castTargetParams, fromType, toType, numElems),
          castFlops(fromType, toType, numElems)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Cast2D)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &fromType,
                                 const Type &toType) {
  std::vector<unsigned> elemCounts;
  const auto dst = vertex.getFieldInfo("dst");
  CODELET_FIELD(src);
  assert(src.size() == dst.size());
  elemCounts.reserve(src.size());
  unsigned totalElems = 0;
  for (unsigned i = 0; i != dst.size(); ++i) {
    elemCounts.emplace_back(src[i].size());
    totalElems += src[i].size();
  }
  const CastTargetParameters targetParams{target, fromType, toType};
  return {getCast2DCycleEstimate(targetParams, fromType, toType, elemCounts),
          castFlops(fromType, toType, totalElems)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CheckAccuracyWhenCast)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType) {
  std::uint64_t cycles = 30;
  return cycles;
}

static std::uint64_t unaryOpInnerLoopCycles(const Target &target,
                                            const Type &type,
                                            const OpPerformanceInfo &perfInfo,
                                            unsigned numElems) {
  if (perfInfo.naturalVectorWidth == false && perfInfo.loopUnrollFactor > 0) {
    return iceil(numElems, perfInfo.loopUnrollFactor) * perfInfo.cyclesPerLoop;
  }
  unsigned vectorWidth = 1;
  if (perfInfo.naturalVectorWidth) {
    vectorWidth = target.getVectorWidth(type);
  }
  // Estimate loop cycles, including a constant loop overhead added to the
  // cycles per vector.  This accounts for load/store and loop decision.
  return basicOpLoopCycles(numElems, vectorWidth, perfInfo.cyclesPerLoop + 4);
}

namespace internal {
std::uint64_t getBinaryOp1DInPlaceEstimate(const poplar::Target &target,
                                           const Type &type,
                                           const popops::expr::BinaryOpType op,
                                           const unsigned numElems) {
  auto superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 58;
  const auto &info = binaryOpInPlacePerfInfo.at({op, type});

  const auto numWorkers = target.getNumWorkerContexts();
  auto numElemsPerWorker = iceil(numElems, numWorkers);
  workerCycles +=
      binaryOpInnerLoopCycles(target, op, type, info, numElemsPerWorker, true);
  return numWorkers * workerCycles + superviserOverhead;
}
} // namespace internal

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(UnaryOp2D)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t cycles = 20;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  assert(in.size() == out.size());
  const auto &info = unaryOpPerfInfo.at({op, type});
  unsigned totalElems = 0;
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    cycles += unaryOpInnerLoopCycles(target, type, info, in[i].size());
    totalElems += in[i].size();
  }
  return {cycles, flopsForUnaryOp2D(totalElems, type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(UnaryOp1D)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = op == expr::UnaryOpType::EXPONENT ? 33 : 17;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  const auto &info = unaryOpPerfInfo.at({op, type});
  assert(in.size() == out.size());
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (in.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // Unary op is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 9;
  return {cycles + superviserOverhead,
          flopsForUnaryOp2D(static_cast<unsigned>(in.size()), type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(UnaryOp2DInPlace)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t cycles = 20;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  unsigned totalElems = 0;
  for (unsigned i = 0; i < inOut.size(); ++i) {
    cycles += unaryOpInnerLoopCycles(target, type, info, inOut[i].size());
    totalElems += inOut[i].size();
  }
  return {cycles, flopsForUnaryOp2D(totalElems, type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(UnaryOp1DInPlace)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 55;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (inOut.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // UnaryOpInPlace is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 9;
  return {cycles + superviserOverhead,
          flopsForUnaryOp2D(static_cast<unsigned>(inOut.size()), type, op)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BinaryOp2D)(const VertexIntrospector &vertex,
                                     const Target &target, BinaryOpType op,
                                     const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());

  const auto &info = binaryOpPerfInfo.at({op, type});

  unsigned totalElems = 0;
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    cycles += binaryOpInnerLoopCycles(target, op, type, info, in1[i].size());
    totalElems += in1[i].size();
  }
  return {cycles, flopsForBinaryOp2D(totalElems, type, op)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BinaryOp1D)(const VertexIntrospector &vertex,
                                     const Target &target, BinaryOpType op,
                                     const Type &type) {
  uint64_t supervisorOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = (type == FLOAT) ? 29 : 26;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  const auto &info = binaryOpPerfInfo.at({op, type});

  const auto numWorkers = target.getNumWorkerContexts();
  unsigned numElems = iceil(in1.size(), numWorkers);
  workerCycles += binaryOpInnerLoopCycles(target, op, type, info, numElems);
  return {numWorkers * workerCycles + supervisorOverhead,
          flopsForBinaryOp2D(static_cast<unsigned>(in1.size()), type, op)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BinaryOp2DInPlace)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            BinaryOpType op, const Type &type) {
  uint64_t cycles = 20;
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  const auto &info = binaryOpInPlacePerfInfo.at({op, type});
  unsigned totalElems = 0;
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    assert(in1Out[i].size() == in2[i].size());
    cycles +=
        binaryOpInnerLoopCycles(target, op, type, info, in1Out[i].size(), true);
    totalElems += in1Out[i].size();
  }
  return {cycles, flopsForBinaryOp2D(totalElems, type, op)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BinaryOp1DInPlace)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            BinaryOpType op, const Type &type) {
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  return {getBinaryOp1DInPlaceEstimate(target, type, op, in1Out.size()),
          flopsForBinaryOp2D(static_cast<unsigned>(in1Out.size()), type, op)};
}

static std::uint64_t selectCycles(const Target &target, const Type &type,
                                  unsigned numElems) {
  unsigned cyclesPerVector = 5;
  unsigned overhead = 6;
  unsigned vectorWidth = 1;
  // ld in1, ld in2, ld in3, movz, st
  // it may be possible to load on the Aux side but then would
  // depend on bool size. If Aux side is used masks must be created after
  // expanding bools to match the input datum size
  return overhead + basicOpLoopCycles(numElems, vectorWidth, cyclesPerVector);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Select)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &type) {
  uint64_t cycles = 5;
  CODELET_FIELD(in1);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  assert(in3.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    assert(in3[i].size() == in1[i].size());
    cycles += selectCycles(target, type, in1[i].size());
  }
  // Assume zero flops
  return {cycles, 0};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastSelect)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  uint64_t cycles = 9 + 1;

  unsigned typeLen = target.getTypeSize(type);

  CODELET_FIELD(in1);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == 1);
  assert(in2.size() == 1);
  assert(in3.size() == out.size());
  for (unsigned i = 0; i < in3.size(); ++i) {
    unsigned n = in3[i].size();
    assert(n == out[i].size());

    switch (typeLen) {
    case 4: // INT, FLOAT
      cycles += 5 + 4 * n + 3;
      break;
    case 2: // HALF
      if (n & 1) {
        cycles += 23 + n * 4;
      } else {
        cycles += 30 + n * 4; // Worst case: pointer misaligned
      }
      break;
    case 1:                             // BOOL
      cycles += 40 + (n / 4) * 17 + 26; // Worst case
      break;
    default:
      throw poputil::poplibs_error("Cycle estimator for BroadcastSelect: "
                                   "invalid type:" +
                                   type.toString());
    }
  }
  // Assume zero flops
  return {cycles, 0};
}

// Estimation of cycles for the BroadcastSelectorSelect. This codelet calls
// LongMemcpy to copy rows into the output tensor and the execution cycles of
// that code can vary a lot, depending on length and alignment of data, so this
// is an estimate, based on being able to use ld64/st64
std::uint64_t BroadcastSelectorSelectCycles(const Type &type,
                                            const unsigned typeLen,
                                            std::vector<unsigned> rowSizes) {
  uint64_t cycles = 11 + 1;
  for (unsigned n : rowSizes) {
    unsigned bytes = n * typeLen;
    // When using ld64/st64 it takes 1 cycles for 8 bytes: 1 cycle/4 bytes
    cycles += 12 + 23 + (bytes / 4) + (bytes % 4) * 5;
  }
  return cycles;
}

namespace internal {
std::uint64_t getDynamicSlice1DEstimate(const poplar::Target &target,
                                        const Type &type,
                                        const unsigned regionSize,
                                        const unsigned numSubElements) {
  bool is8bit = target.getTypeSize(type) == 1;
  const auto is64Bit = type == UNSIGNED_LONGLONG || type == LONGLONG;
  const unsigned numWorkers = target.getNumWorkerContexts();
  const unsigned elementsPerWorker =
      ((1 + is64Bit) * regionSize + numWorkers - 1) / numWorkers;
  unsigned vectorWidth = 0;
  if (is8bit) {
    vectorWidth = 4;
  } else {
    // treat 64-bit as just twice 32-bit data
    vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  }
  //  Supervisor overhead.
  auto superCycles = basicOpSupervisorOverhead() + 1 + 6 + 1 + 6;

  // This is the more optimistic path - where the inner loop is copying
  // aligned data at 64bits/2cycles in the non-8bit case
  unsigned nCopies = elementsPerWorker / vectorWidth;

  auto workerCycles = is8bit ? 72 + (70 + 2 * nCopies) * numSubElements
                             : 35 + (24 + 2 * nCopies) * numSubElements;
  auto cycles = superCycles + workerCycles * numWorkers;

  return cycles;
}
} // namespace internal

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastSelectorSelect)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(in1);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  assert(in3.size() == 1);
  std::vector<unsigned> rowSizes;
  for (unsigned i = 0; i < in1.size(); ++i) {
    rowSizes.push_back(in1[i].size());
  }
  return BroadcastSelectorSelectCycles(type, target.getTypeSize(type),
                                       rowSizes);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(SelectInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == in1.size());
  assert(in3.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in2[i].size() == in1[i].size());
    assert(in3[i].size() == in1[i].size());
    cycles += selectCycles(target, type, in1.size());
  }
  // Assume zero flops
  return {cycles, 0};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastSelectorSelectInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(in1Out);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == in1Out.size());
  assert(in3.size() == 1);
  std::vector<unsigned> rowSizes;
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    rowSizes.push_back(in1Out[i].size());
  }
  // Assume zero flops
  return {
      BroadcastSelectorSelectCycles(type, target.getTypeSize(type), rowSizes),
      0};
}

static std::uint64_t histogramFlops(unsigned numElems, bool isAbsolute) {
  return static_cast<std::uint64_t>(numElems) * (1 + isAbsolute);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Histogram2D)(const VertexIntrospector &vertex,
                                      const Target &target, const Type &type,
                                      const bool isAbsolute) {
  CODELET_FIELD(data);
  CODELET_FIELD(histogram);
  CODELET_FIELD(limits);
  CODELET_SCALAR_VAL(histogramCount, unsigned);
  const auto unpackCostHistogram =
      poputil::internal::getUnpackCost(histogram.getProfilerVectorLayout(0));
  const auto unpackCostLimits =
      poputil::internal::getUnpackCost(limits.getProfilerVectorLayout(0));

  const auto vectorWidth =
      target.getDataPathWidth() / (type == FLOAT ? 32 : 16);

  uint64_t cycles = 7 + unpackCostHistogram + unpackCostLimits;
  unsigned totalElems = 0;
  if (type == HALF) {
    cycles += 5;                  // Pre-loop overhead
    uint64_t dataLoopCycles = 16; // Data-loop overhead
    for (unsigned i = 0; i < data.size(); i++) {
      unsigned elements = data[i].size();
      totalElems += data[i].size();
      if (elements & 1) {
        dataLoopCycles += 3 + isAbsolute;
      }
      if (elements & 2) {
        dataLoopCycles += 3 + isAbsolute;
      }
      dataLoopCycles += (3 + isAbsolute) * (elements / vectorWidth);
    }

    cycles += (dataLoopCycles + 6) * (histogramCount - 1);
  } else {
    cycles += 3;                  // Pre-loop overhead
    uint64_t dataLoopCycles = 10; // Data-loop overhead
    for (unsigned i = 0; i < data.size(); i++) {
      unsigned elements = data[i].size();
      totalElems += elements;
      if (elements & 1) {
        dataLoopCycles += 2 + isAbsolute;
      }
      dataLoopCycles += (3 + isAbsolute) * (elements / vectorWidth);
    }
    cycles += (dataLoopCycles + 5) * (histogramCount - 1);
  }
  // post process
  cycles += 3 + (3 * (data.size() - 1));
  cycles += 10 + (histogramCount - 1) * 2;
  return {cycles,
          convertToTypeFlops(histogramFlops(totalElems, isAbsolute), type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(Histogram1D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isAbsolute, const bool splitByLimits) {
  CODELET_FIELD(data);
  CODELET_FIELD(histogram);
  CODELET_FIELD(limits);
  CODELET_SCALAR_VAL(histogramCount, unsigned);
  const auto unpackCostHistogram =
      poputil::internal::getUnpackCost(histogram.getProfilerVectorLayout(0));
  const auto unpackCostLimits =
      poputil::internal::getUnpackCost(limits.getProfilerVectorLayout(0));

  const auto vectorWidth =
      target.getDataPathWidth() / (type == FLOAT ? 32 : 16);
  auto numWorkers = target.getNumWorkerContexts();
  auto flops = histogramFlops(data.size(), isAbsolute);

  if (splitByLimits) {
    return {histogram1DByLimitEstimate(data.size(), histogramCount, isAbsolute,
                                       type == HALF, numWorkers, vectorWidth,
                                       unpackCostHistogram, unpackCostLimits),
            convertToTypeFlops(flops, type)};
  } else {
    return {histogram1DByDataEstimate(data.size(), histogramCount, isAbsolute,
                                      type == HALF, numWorkers, vectorWidth,
                                      unpackCostHistogram, unpackCostLimits),
            convertToTypeFlops(flops, type)};
  }
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ForLoopCounter)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  return {43, 0};
}

static std::uint64_t clampCycles(const Target &target, const Type &type,
                                 unsigned numElems) {
  unsigned cyclesPerVector = 1;
  unsigned overhead = 6;
  unsigned vectorWidth = 1;
  if (type == FLOAT) {
    vectorWidth = target.getDataPathWidth() / 32;
    cyclesPerVector = 2;
  } else if (type == HALF) {
    vectorWidth = target.getDataPathWidth() / 16;
    cyclesPerVector = 2;
  } else if (type == INT) {
    // ld, ld, ld, cmp, movz, cmp, st
    cyclesPerVector = 7;
  }
  return overhead + basicOpLoopCycles(numElems, vectorWidth, cyclesPerVector);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Clamp)(const VertexIntrospector &vertex,
                                const Target &target, const Type &type) {
  uint64_t cycles = 5;
  CODELET_FIELD(in1);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  assert(in3.size() == in1.size());
  unsigned totalElems = 0;
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    assert(in3[i].size() == in1[i].size());
    totalElems += in1[i].size();
    cycles += clampCycles(target, type, in1[i].size());
  }
  return {cycles,
          convertToTypeFlops(static_cast<std::uint64_t>(totalElems) * 2, type)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(ClampInPlace)(const VertexIntrospector &vertex,
                                       const Target &target, const Type &type) {
  uint64_t cycles = 5;
  CODELET_FIELD(in1Out);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == in1Out.size());
  assert(in3.size() == in1Out.size());
  unsigned totalElems = 0;
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    assert(in2[i].size() == in1Out[i].size());
    assert(in3[i].size() == in1Out[i].size());
    totalElems += in1Out[i].size();
    cycles += clampCycles(target, type, in1Out[i].size());
  }
  return {cycles,
          convertToTypeFlops(static_cast<std::uint64_t>(totalElems) * 2, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastClamp)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  // NOTE: Draft version to make UTs pass. Will be update with more accurate
  //       estimates from ASM implementation
  uint64_t cycles = 5;
  CODELET_FIELD(in1);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == 1);
  assert(in3.size() == 1);
  unsigned totalElems = 0;
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    cycles += clampCycles(target, type, in1[i].size());
    totalElems += in1[i].size();
  }
  return {cycles,
          convertToTypeFlops(static_cast<std::uint64_t>(totalElems) * 2, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastClampInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  // NOTE: Draft version to make UTs pass. Will be update with more accurate
  //       estimates from ASM implementation
  uint64_t cycles = 5;
  CODELET_FIELD(in1Out);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == 1);
  assert(in3.size() == 1);
  unsigned totalElems = 0;
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    cycles += clampCycles(target, type, in1Out[i].size());
    totalElems += in1Out[i].size();
  }
  return {cycles,
          convertToTypeFlops(static_cast<std::uint64_t>(totalElems) * 2, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicSlice2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  bool is8bit = target.getTypeSize(type) == 1;
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
      vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
      vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
  const unsigned numRegions =
      vertex.getFieldInfo("numRegions").getInitialValue<unsigned>(target);

  unsigned vectorWidth = 0;
  if (is8bit) {
    vectorWidth = 4;
  } else {
    vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  }
  auto cycles = 23;

  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    if (is8bit) {
      cycles += (50 + 2 * nVectors) * numSubElements + 13;
    } else if (type == HALF)
      cycles += (31 + 2 * nVectors) * numSubElements + 13;
    else
      cycles += (29 + nVectors) * numSubElements + 13;
  }
  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicUpdateSlice2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  bool is8bit = target.getTypeSize(type) == 1;
  const auto is64Bit = type == UNSIGNED_LONGLONG || type == LONGLONG;
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
      vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
      vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
  const unsigned numRegions =
      vertex.getFieldInfo("numRegions").getInitialValue<unsigned>(target);

  unsigned vectorWidth = 0;
  if (is8bit) {
    vectorWidth = 4;
  } else {
    // 64-bit types are treated as 32-bit with twice the region size
    vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  }
  auto cycles = 23;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size() * (1 + is64Bit);
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    if (is8bit) {
      cycles += (50 + 23 * nVectors) * numSubElements + 13;
    } else if (type == HALF)
      cycles += (31 + 2 * nVectors) * numSubElements + 13;
    else
      // additional cycle for 64-bit to scale region size
      cycles += (29 + is64Bit + nVectors) * numSubElements + 13;
  }
  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicSlice1D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  const auto regionSize =
      vertex.getFieldInfo("regionSize").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
      vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
#ifndef NDEBUG
  const unsigned numBaseElements =
      vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
#endif
  const auto baseT = vertex.getFieldInfo("baseT");
  const auto subT = vertex.getFieldInfo("subT");
  assert(subT.size() == numSubElements * regionSize);
  assert(baseT.size() == numBaseElements * regionSize);
  return getDynamicSlice1DEstimate(target, type, regionSize, numSubElements);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicUpdateSlice1D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  return MAKE_PERF_ESTIMATOR_NAME(DynamicSlice1D)(vertex, target, type);
}

static std::uint64_t multiSlicer(const VertexIntrospector &vertex,
                                 const Target &target, const Type &type,
                                 bool isUpdate) {
  const auto regionSize =
      vertex.getFieldInfo("regionSize").getInitialValue<unsigned>(target);
  const auto splitSingleRegion =
      vertex.getFieldInfo("splitSingleRegion").getInitialValue<bool>(target);
  const auto offsets = vertex.getFieldInfo("offsets");
  // Assume worst case scenario where every index is in range for the vertex.
  return getMultiSliceCycleEstimate(MultiSliceTargetParameters{target, type},
                                    regionSize, offsets.size(), offsets.size(),
                                    0, isUpdate, splitSingleRegion);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(MultiSlice)(const VertexIntrospector &vertex,
                                     const Target &target, const Type &type) {
  return multiSlicer(vertex, target, type, false);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(MultiUpdate)(const VertexIntrospector &vertex,
                                      const Target &target, const Type &type) {
  return multiSlicer(vertex, target, type, true);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScaledMultiUpdateOp)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const Type &scaleType, const bool &subWordWritesRequired,
    const Operation &op) {

  // based off the assembly (optimistic for integral types which are still
  // handled by the compiler). Assumes the worst case where all indices are
  // processed by a single worker.
  CODELET_FIELD(offsets);
  CODELET_SCALAR_VAL(regionSize, unsigned short);

  // Assume worst case scenario where every index is in range for one of the
  // workers in the vertex.
  const bool isScaled = true;
  const auto cycles = getMultiUpdateOpCycleEstimate(
      MultiUpdateOpTargetParameters{target, type}, subWordWritesRequired,
      regionSize, offsets.size(), offsets.size(), 0, op, isScaled,
      type == HALF && type != scaleType, false);
  return {cycles, static_cast<std::uint64_t>(regionSize) *
                      (flopsPerBinaryOpElement(BinaryOpType::ADD) +
                       flopsPerBinaryOpElement(BinaryOpType::MULTIPLY))};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(MultiUpdateOp)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool &subWordWritesRequired, const Operation &op) {

  // based off the assembly (optimistic for integral types which are still
  // handled by the compiler). Assumes the worst case where all indices are
  // processed by a single worker.
  CODELET_FIELD(offsets);
  CODELET_SCALAR_VAL(regionSize, unsigned short);

  // Assume worst case scenario where every index is in range for one of the
  // workers in the vertex.
  const bool isScaled = false;
  const auto cycles = getMultiUpdateOpCycleEstimate(
      MultiUpdateOpTargetParameters{target, type}, subWordWritesRequired,
      regionSize, offsets.size(), offsets.size(), 0, op, isScaled, false,
      false);
  return {cycles, static_cast<std::uint64_t>(regionSize) *
                      flopsPerBinaryOpElement(BinaryOpType::ADD)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(SequenceSlice)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(srcOffsetT);
  CODELET_FIELD(dstT);
  CODELET_FIELD(srcT);
  CODELET_SCALAR_VAL(regionSize, unsigned short);
  if (srcOffsetT.size() == 0)
    return 40;

  const auto tN = vertex.getFieldInfo("nElementsT").size();
  std::uint64_t cycles = 60; // call overhead
  // We don't know how much was requested so assume all of one of the buffers is
  // copied. This may be quite pessimistic.
  const unsigned bytesPerAtom = target.getTypeSize(type);
  unsigned nElementsCopied = std::min(dstT.size(), srcT.size());
  unsigned nAtomsCopies = nElementsCopied * regionSize;
  // Assume 32bits/2 cycles for worker c++ codelet.
  // Assume all subsequences are the same size. This may be optimistic.
  auto oneWorkerCycles = (nAtomsCopies * bytesPerAtom / 4 * 2 + tN - 1) / tN;
  // Some workers may be idle.
  unsigned numWorkerContexts = target.getNumWorkerContexts();
  auto numWorkerRuns = (tN + numWorkerContexts - 1) / numWorkerContexts;
  cycles += oneWorkerCycles * numWorkerContexts * numWorkerRuns;
  // Outerloop overhead of 60 cycles for supervisor c++ codelet.
  cycles += tN * 60;
  return cycles;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(CircBufIncrIndex)(const VertexIntrospector &vertex,
                                           const Target &target) {
  return 8;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(CircOffset)(const VertexIntrospector &vertex,
                                     const Target &target) {
  return 10;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(EncodeOneHot)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &indexType, const Type &outputType) {
  CODELET_FIELD(indices);
  if (indexType == UNSIGNED_INT && outputType == HALF) {
    std::uint64_t cycles = basicOpSupervisorOverhead();
    // the encode loop can take the following cycles for each index:
    //  - 22 if index[i] < offset[i],
    //  - 24 if index[i] > out.size(),
    //  - 64 if out[idx + indices[i] - offsets[i]] & 0x3 == 0,
    //  - 58 if out[idx + indices[i] - offsets[i]] & 0x3 == 1,
    // additional 12 cycles for comparing ignore indices
    // as we can't tell which branch the code will take, assume the worst case
    // every iteration.
    cycles += (62 + 12) * indices.size();
    return cycles;
  } else {
    // C++ vertex
    return 100 * indices.size();
  }
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(EncodeOneHotCustomValues)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &indexType, const Type &outputType) {
  CODELET_FIELD(indices);

  // C++ vertex
  std::uint64_t cycles = 100 * indices.size();

  return cycles;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Iota)(const VertexIntrospector &vertex,
                               const Target &target, const Type &outputType) {
  CODELET_FIELD(out);
  CODELET_FIELD(offsets);
  auto vectorWidth = target.getVectorWidth(outputType);

  std::uint64_t cycles = 10;
  for (unsigned region = 0; region != out.size(); ++region) {
    unsigned regionSize = out[region].size();

    auto numVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    // ld start, setzi to set to start, setup loopcount, loopcount-1
    // assume brnzdec
    cycles += 4 + 3 * numVectors;
  }
  return cycles;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(HeapSortVertex)(const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &indexType) {
  std::uint64_t n = vertex.getFieldInfo("out").size();

  // Assuming all the worst cases are hit in the HeapSort codelet
  return 8 * (19 * n * std::floor(std::log2(n)) + 6 * n + 2);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(HeapSortVertexKV)(
    const VertexIntrospector &vertex, const Target &target, const Type &keyType,
    const Type &ValueType) {
  std::uint64_t n = vertex.getFieldInfo("key").size();

  // Assuming all the worst cases are hit in the HeapSort codelet
  return 16 * (19 * n * std::floor(std::log2(n)) + 6 * n + 2);
}

std::uint64_t decrementOrGetParamsCycles(unsigned dataLen, bool isHalf) {
  // Theoretical cycle count based on simple update with -1 loop
  // load index,
  // load inptr, load with index,
  // check for MASKED_LABEL_CODE, branch, subtract,
  // load outptr, store with index.

  // Storing half requires read-modify-write
  return (isHalf ? 12 : 8) * dataLen;
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(UpdateIntervalDEC)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &paramsType) {
  CODELET_SCALAR_VAL(rowCount, unsigned);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(rowCount, paramsType == HALF);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(UpdateIntervalsDEC)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             const Type &paramsType) {
  CODELET_FIELD(params);
  CODELET_VECTOR_VALS(rowCounts, unsigned);
  const auto rowCountsSum =
      std::accumulate(rowCounts.begin(), rowCounts.end(), 0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(params.size() * rowCountsSum,
                                             paramsType == HALF);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(UpdateColumnsDEC)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &paramsType) {
  CODELET_FIELD(params);
  CODELET_VECTOR_VALS(regionWidths, unsigned);
  CODELET_VECTOR_VALS(regionHeights, unsigned);
  const auto regionHeightsSum =
      std::accumulate(regionHeights.begin(), regionHeights.end(), 0);
  const auto regionWidthsSum =
      std::accumulate(regionWidths.begin(), regionWidths.end(), 0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(params.size() * regionWidthsSum *
                                                 regionHeightsSum,
                                             paramsType == HALF);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(SelectFromInterval)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             const Type &paramsType) {
  CODELET_SCALAR_VAL(rowCount, unsigned);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(rowCount, paramsType == HALF);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(SelectFromIntervals)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &paramsType) {
  CODELET_FIELD(params);
  CODELET_VECTOR_VALS(rowCounts, unsigned);
  const auto rowCountsSum =
      std::accumulate(rowCounts.begin(), rowCounts.end(), 0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(params.size() * rowCountsSum,
                                             paramsType == HALF);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(SelectFromRowsInColumns)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &paramsType) {
  CODELET_FIELD(params);
  CODELET_VECTOR_VALS(regionWidths, unsigned);
  CODELET_VECTOR_VALS(regionHeights, unsigned);
  const auto regionHeightsSum =
      std::accumulate(regionHeights.begin(), regionHeights.end(), 0);
  const auto regionWidthsSum =
      std::accumulate(regionWidths.begin(), regionWidths.end(), 0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(params.size() * regionWidthsSum *
                                                 regionHeightsSum,
                                             paramsType == HALF);
}

static std::uint64_t hasNanInnerLoopCycles(const Type &type, unsigned int size,
                                           bool checkBothInfAndNan) {
  std::uint64_t cycles = type == FLOAT ? 10 : 12;
  if (size == 0) {
    return cycles;
  }
  const auto checkOnlyNaNHalf = type == HALF && !checkBothInfAndNan;
  if (type == FLOAT || checkOnlyNaNHalf) {
    const auto numVectors = size / 4;
    const auto additionalCycle =
        (type == FLOAT && checkBothInfAndNan) || checkOnlyNaNHalf;
    cycles += (2 + additionalCycle) * numVectors;
    if (size & 0x2) {
      cycles += 2;
    }
    if (size & 0x1) {
      cycles += 1;
    }
  } else {
    const auto numVectors = size / 8;
    cycles += 2 * numVectors;
    if (size & 0x4) {
      cycles += 2;
    }
    if (size & 0x2) {
      cycles += 2;
    }
    if (size & 0x1) {
      cycles += 2;
    }
  }
  return cycles;
}

unsigned nanCheckCycles(const Type &type, bool checkBothNaNAndInf) {
  return checkBothNaNAndInf + (type == FLOAT ? 7 : 11);
}

static VertexPerfEstimate
hasNaN1DCyles(const Target &target, const Type &inType,
              unsigned sizeIn8BytesPerWorker, unsigned char remWorkerId,
              unsigned char remWorkerExtras, bool hasBothInfOrNaN) {
  unsigned inSize = sizeIn8BytesPerWorker + (remWorkerId > 0);
  inSize = inSize * (8 / target.getTypeSize(inType)) + remWorkerExtras;
  std::uint64_t flops = inSize;
  return {(15 + hasNanInnerLoopCycles(inType, inSize, hasBothInfOrNaN) +
           nanCheckCycles(inType, hasBothInfOrNaN)) *
                  target.getNumWorkerContexts() +
              24,
          flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(HasNaNOrInf1D)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &inType, bool hasNaNOrInf) {
  CODELET_SCALAR_VAL(sizeIn8BytesPerWorker, unsigned);
  CODELET_SCALAR_VAL(remWorkerId, unsigned char);
  CODELET_SCALAR_VAL(remWorkerExtras, unsigned char);
  return hasNaN1DCyles(target, inType, sizeIn8BytesPerWorker, remWorkerId,
                       remWorkerExtras, hasNaNOrInf);
}

static VertexPerfEstimate hasNan2DCycles(const FieldData &in,
                                         const Target &target,
                                         const Type &inType,
                                         bool hasBothInfOrNaN) {
  // initial overhead + exitz
  std::uint64_t cycles = 6;
  if (in.size() == 0) {
    return cycles;
  }

  // post-zero check overhead.
  cycles += 1;
  uint64_t flops = 0;
  for (unsigned i = 0; i < in.size(); ++i) {
    cycles += hasNanInnerLoopCycles(inType, in[i].size(), hasBothInfOrNaN);
    flops += in[i].size();
  }

  // Nan checking
  cycles += nanCheckCycles(inType, hasBothInfOrNaN);
  return {cycles * target.getNumWorkerContexts(), flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(HasNaNOrInf2D)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &inType, bool hasNaNOrInf) {
  CODELET_FIELD(in);
  return hasNan2DCycles(in, target, inType, hasNaNOrInf);
}

static VertexPerfEstimate CompareAndSwapKeyValueCycleEstimate(
    const VertexIntrospector &vertex, const Target &target, const Type &keyType,
    const std::optional<Type> &valueType = std::nullopt,
    bool valuesAreSecondaryKey = false) {
  std::uint64_t cycles = 0;
  std::uint64_t flops = 0;

  CODELET_VECTOR_2D_VALS(worklists, unsigned short);
  CODELET_SCALAR_VAL(distanceToChangeOrder, unsigned);

  assert(keyType == FLOAT || keyType == UNSIGNED_INT || keyType == INT);
  assert(!valueType || *valueType == FLOAT || *valueType == UNSIGNED_INT ||
         *valueType == INT);

  const auto usedWorkers = worklists.size();
  std::uint64_t maxWorkerCycles = 0;
  for (unsigned wid = 0; wid < usedWorkers; ++wid) {
    auto worklistIt = worklists[wid].cbegin();
    const unsigned numEntriesM1 = *worklistIt++;
    const auto numEntries = numEntriesM1 + 1;
    const auto initialOffset = *worklistIt++;
    (void)initialOffset;
    const unsigned lower = *worklistIt++;
    const unsigned upper = *worklistIt++;
    const unsigned packedOrderAndCount = (lower | (upper << 16u));
    const auto initialCount = packedOrderAndCount >> 1u;
    unsigned firstInnerElemCount = *worklistIt++;
    unsigned changeOrderCounter = distanceToChangeOrder - initialCount;

    // The following cycle estimates were obtained from manually analysing the
    // assembly implementation of the vertex.
    std::uint64_t thisWorkerCycles = 0;
    unsigned innerLoopCycles, firstInnerLoopCycles, outerLoopCycles,
        firstOuterLoopCycles, workListItemCycles, overheadCycles;
    if (valueType == std::nullopt) {
      if (keyType == FLOAT) {
        // C++ codelet
        innerLoopCycles = 13;
        firstInnerLoopCycles = innerLoopCycles;
        outerLoopCycles = 13;
        firstOuterLoopCycles = 3;
        workListItemCycles = 17;
        overheadCycles = 40;
      } else if (keyType == UNSIGNED_INT || keyType == INT) {
        // C++ codelet
        innerLoopCycles = 12;
        firstInnerLoopCycles = innerLoopCycles;
        outerLoopCycles = 17;
        firstOuterLoopCycles = 3;
        workListItemCycles = 22;
        overheadCycles = 42;
      }
    } else if (*valueType == FLOAT || *valueType == UNSIGNED_INT ||
               *valueType == INT) {
      if (valuesAreSecondaryKey) {
        // C++ codelets
        if (keyType == FLOAT) {
          innerLoopCycles = 25;
          firstInnerLoopCycles = innerLoopCycles;
          outerLoopCycles = 24;
          firstOuterLoopCycles = 6;
          workListItemCycles = 30;
          overheadCycles = 45;
        } else if (keyType == UNSIGNED_INT || keyType == INT) {
          innerLoopCycles = 27;
          firstInnerLoopCycles = innerLoopCycles;
          outerLoopCycles = 25;
          firstOuterLoopCycles = 7;
          workListItemCycles = 30;
          overheadCycles = 49;
        }
      } else {
        if (keyType == FLOAT) {
          // Assembly codelet
          innerLoopCycles = 11;
          firstInnerLoopCycles = 8;
          outerLoopCycles = 11;
          firstOuterLoopCycles = outerLoopCycles;
          workListItemCycles = 4;
          overheadCycles = 23;
        } else if (keyType == UNSIGNED_INT || keyType == INT) {
          // C++ codelet
          innerLoopCycles = 17;
          firstInnerLoopCycles = innerLoopCycles;
          outerLoopCycles = 22;
          firstOuterLoopCycles = 5;
          workListItemCycles = 30;
          overheadCycles = 43;
        }
      }
    }

    // cycles for constant overhead pre/post numEntries loop
    thisWorkerCycles += overheadCycles;

    // cycles per entry in the worklist
    thisWorkerCycles += numEntries * workListItemCycles;

    bool firstEntry = true;
    for (unsigned entry = 0; entry < numEntries; ++entry) {
      const unsigned distance = *worklistIt++;
      const unsigned numElems = *worklistIt++;
      const auto innerElemCount =
          firstEntry ? firstInnerElemCount : std::min(distance, numElems);
      // Total number of elements
      const auto numInnerLoops = numElems;
      const auto numOuterLoops =
          1 + (gccs::ceildiv(numElems - innerElemCount, distance));
      const auto numChangesOfOrder =
          numElems >= changeOrderCounter
              ? (1 + (numElems - changeOrderCounter) / distanceToChangeOrder)
              : 0;

      // Cycles per element. Note we assume the worst case where every
      // pair of elements must be swapped but this is data dependent in
      // reality.
      thisWorkerCycles +=
          firstInnerLoopCycles + (numInnerLoops - 1) * innerLoopCycles;
      // 1 floating point comparison per element
      flops += numInnerLoops;
      // additional cycles for each outer loop until numElems is exhausted
      thisWorkerCycles +=
          firstOuterLoopCycles + (numOuterLoops - 1) * outerLoopCycles;
      // cycles to change order and reset change order counter
      thisWorkerCycles += numChangesOfOrder * 2;

      firstEntry = false;
    }

    maxWorkerCycles = std::max(maxWorkerCycles, thisWorkerCycles);
  }

  static constexpr std::uint64_t supervisorCycles = 19;
  cycles += supervisorCycles;
  cycles += maxWorkerCycles * target.getNumWorkerContexts();

  return VertexPerfEstimate(cycles, flops);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CompareAndSwapAtDistance)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &keyType) {
  return CompareAndSwapKeyValueCycleEstimate(vertex, target, keyType);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CompareAndSwapAtDistanceKeyVal)(
    const VertexIntrospector &vertex, const Target &target, const Type &keyType,
    const Type &valueType, bool valuesAreSecondaryKey) {
  return CompareAndSwapKeyValueCycleEstimate(vertex, target, keyType, valueType,
                                             valuesAreSecondaryKey);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(NormaliseImage)(const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &inType, const Type &) {
  CODELET_SCALAR_VAL(packedNPixels, unsigned);
  unsigned nPixelsPerWorker = packedNPixels >> 3;
  unsigned remainder = packedNPixels & 0x7;
  unsigned nWorkers = target.getNumWorkerContexts();
  unsigned nPixels = nWorkers * nPixelsPerWorker + remainder;
  auto flops = 3 * 3 * nPixels;
  unsigned cycles = 0;
  unsigned workerCycles;
  if (inType == UNSIGNED_CHAR) {
    workerCycles = 28 + (nPixelsPerWorker + (remainder > 1)) * 8 + 1;
  } else if (inType == HALF) {
    workerCycles = 51 + (nPixelsPerWorker + (remainder > 1)) * 11 + 1;
  } else if (inType == FLOAT) {
    workerCycles = 51 + (nPixelsPerWorker + (remainder > 1)) * 17 + 1;

  } else {
    throw poputil::poplibs_error(
        "NormaliseImage does not support this data type");
  }
  cycles = nWorkers * workerCycles;
  return {cycles, flops};
}

VertexPerfEstimate ScalarMultiply2DEstimator(const std::vector<unsigned> sizes,
                                             bool inplace) {
  const auto elementsPerLoop = 4;

  unsigned cycles = 0;
  unsigned flops = 0;

  auto getLoopCycles = [](unsigned nloops) {
    unsigned cycles = 1;
    cycles += nloops > 0 ? 7 : 0;
    cycles += nloops > 1 ? 5 * (nloops - 1) : 0;
    return cycles;
  };

  auto getRemainderCycles = [](unsigned remainder) {
    unsigned cycles = 1;
    if (!remainder)
      return cycles;
    cycles += 2;
    cycles += remainder != 0 ? 4 : 0;
    cycles += remainder > 1 ? 5 : 0;
    cycles += (remainder == 1 || remainder == 3) ? 6 : 0;
    cycles += 1;
    return cycles;
  };

  auto getInnerLoopCycles = [&getLoopCycles, &getRemainderCycles,
                             elementsPerLoop](unsigned size) {
    unsigned nloops = size / elementsPerLoop;
    unsigned remainder = size % elementsPerLoop;

    unsigned cycles = 0;
    cycles += getLoopCycles(nloops);
    cycles += getRemainderCycles(remainder);
    cycles += 6;
    return cycles;
  };

  cycles += inplace ? 35 : 36;

  for (auto size : sizes) {
    cycles += 11;
    auto cycles_ = getInnerLoopCycles(size);
    cycles += cycles_;
    flops += size * flopsPerBinaryOpElement(BinaryOpType::MULTIPLY);
  }

  // exitz
  cycles += 1;

  return {cycles, convertToTypeFlops(flops, FLOAT)};
}

VertexPerfEstimate ScalarMultiply1DEstimator(const Target &target,
                                             const unsigned size) {
  const auto elementsPerLoop = 4;
  const auto nWorkers = target.getNumWorkerContexts();

  unsigned workerCycles = 0;

  auto getLoopCycles = [](unsigned nloops) {
    unsigned cycles = 1;
    cycles += nloops > 0 ? 7 : 0;
    cycles += nloops > 1 ? 5 * (nloops - 1) : 0;
    return cycles;
  };

  auto getRemainderCycles = [](unsigned remainder) {
    unsigned cycles = 1;
    if (!remainder)
      return cycles;
    cycles += 2;
    cycles += remainder != 0 ? 4 : 0;
    cycles += remainder > 1 ? 5 : 0;
    cycles += (remainder == 1 || remainder == 3) ? 6 : 0;
    cycles += 1;
    return cycles;
  };

  auto getInnerLoopCycles = [&getLoopCycles, &getRemainderCycles,
                             elementsPerLoop,
                             nWorkers](unsigned size, unsigned wid) {
    // Does this worker process the remainder?
    bool flag =
        (((size % (elementsPerLoop * nWorkers)) - 1) / elementsPerLoop) == wid;
    unsigned nloops = (size + elementsPerLoop * (nWorkers - 1 - wid)) /
                      (elementsPerLoop * nWorkers);
    unsigned remainder = (size % elementsPerLoop) * flag;

    unsigned cycles = 0;
    cycles += getLoopCycles(nloops);
    cycles += getRemainderCycles(remainder);
    return cycles;
  };

  // Number of cycles before the main loop. This involves:
  //   - Checking float to half conversion accuracy.
  //   - Loading vertex state.
  //   - Splitting work between 6 workers.
  workerCycles += 50;

  unsigned slowestWorkerCycles = 0;
  for (unsigned wid = 0; wid < 6; wid++) {
    auto workerCycles = getInnerLoopCycles(size, wid);
    slowestWorkerCycles = std::max(slowestWorkerCycles, workerCycles);
  }
  workerCycles += slowestWorkerCycles;

  // Number of cycles after the main loop. This includes exitz.
  workerCycles += 1;

  unsigned flops = size * flopsPerBinaryOpElement(BinaryOpType::MULTIPLY);

  return {workerCycles * nWorkers, convertToTypeFlops(flops, FLOAT)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScalarMultiply1D)(
    const VertexIntrospector &vertex, const Target &target, const Type &in1Type,
    const Type &in2Type) {
  CODELET_FIELD(in1);
  return ScalarMultiply1DEstimator(target, in1.size());
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScalarMultiply1DInplace)(
    const VertexIntrospector &vertex, const Target &target, const Type &in1Type,
    const Type &in2Type) {
  CODELET_FIELD(in1Out);
  return ScalarMultiply1DEstimator(target, in1Out.size());
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScalarMultiply2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &in1Type,
    const Type &in2Type) {
  CODELET_FIELD(in1);

  std::vector<unsigned> sizes;
  for (unsigned i = 0; i < in1.size(); i++) {
    sizes.push_back(in1.getSizeAtIndex(i));
  }

  return ScalarMultiply2DEstimator(sizes, false);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ScalarMultiply2DInplace)(
    const VertexIntrospector &vertex, const Target &target, const Type &in1Type,
    const Type &in2Type) {
  CODELET_FIELD(in1Out);

  std::vector<unsigned> sizes;
  for (unsigned i = 0; i < in1Out.size(); i++) {
    sizes.push_back(in1Out.getSizeAtIndex(i));
  }

  return ScalarMultiply2DEstimator(sizes, true);
}

#define BROADCAST_2TYPE_CYCLE_ESTIM_ENTRIES(vertexName)                        \
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName,                                    \
                        BinaryOpType::VARIANCE_TO_INV_STD_DEV, FLOAT, HALF),   \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName,                                \
                            BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF,       \
                            FLOAT)

// Entries for broadcast of relational operations followed by cast
#define BROADCAST_CAST_CYCLE_ESTIM_ENTRIES(vertexName)                         \
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::EQUAL, UNSIGNED_INT, \
                        FLOAT),                                                \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::EQUAL,           \
                            UNSIGNED_INT, HALF),                               \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::GREATER_THAN,    \
                            UNSIGNED_INT, FLOAT),                              \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::GREATER_THAN,    \
                            UNSIGNED_INT, HALF),                               \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName,                                \
                            BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT,    \
                            FLOAT),                                            \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName,                                \
                            BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT,    \
                            HALF),                                             \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::LESS_THAN,       \
                            UNSIGNED_INT, FLOAT),                              \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::LESS_THAN,       \
                            UNSIGNED_INT, HALF),                               \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::LESS_THAN_EQUAL, \
                            UNSIGNED_INT, FLOAT),                              \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::LESS_THAN_EQUAL, \
                            UNSIGNED_INT, HALF),                               \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::NOT_EQUAL,       \
                            UNSIGNED_INT, FLOAT),                              \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::NOT_EQUAL,       \
                            UNSIGNED_INT, HALF)

// Entries for broadcast outer vertices covering only the 3 basic operations,
// each with an alwaysAligned template parameter
#define BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(vertexName,                 \
                                                   allowMisaligned)            \
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::ADD, FLOAT,          \
                        allowMisaligned),                                      \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::ADD, HALF,       \
                            allowMisaligned),                                  \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::SUBTRACT, FLOAT, \
                            allowMisaligned),                                  \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::SUBTRACT, HALF,  \
                            allowMisaligned),                                  \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::MULTIPLY, FLOAT, \
                            allowMisaligned),                                  \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BinaryOpType::MULTIPLY, HALF,  \
                            allowMisaligned)

// Entries for VectorInner vertices
#define VECTOR_INNER_CYCLE_ESTIM_ENTRIES(name)                                 \
  CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::ADD, FLOAT),               \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::ADD, HALF),            \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::DIVIDE, FLOAT),        \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::DIVIDE, HALF),         \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::SUBTRACT, FLOAT),      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::SUBTRACT, HALF),       \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::MULTIPLY, FLOAT),      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BinaryOpType::MULTIPLY, HALF)

#define SCALED_ADD_CYCLE_ESTIM_ENTRIES(NAME, TYPE1, TYPE2, TYPE3)              \
  CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, TYPE3, true, true),        \
      CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, TYPE3, true, false),   \
      CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, TYPE3, false, true),   \
      CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, TYPE3, false, false)

// A couple of macros to create more compactly the entries for the various
// Cast vertices, for all possible combinations of input and output types
// (float, half, signed/unsinged ints and bool)
#define CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, SRC_TYPE)                   \
  CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, CHAR),                         \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, SIGNED_CHAR),              \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, UNSIGNED_CHAR),            \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, FLOAT),                    \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, HALF),                     \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, INT),                      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, SHORT),                    \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, UNSIGNED_INT),             \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, UNSIGNED_SHORT),           \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, BOOL)

#define CAST_CYCLE_ESTIM_ENTRIES(name)                                         \
  CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, CHAR),                            \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, SIGNED_CHAR),                 \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, UNSIGNED_CHAR),               \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, FLOAT),                       \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, HALF),                        \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, INT),                         \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, SHORT),                       \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, UNSIGNED_INT),                \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, UNSIGNED_SHORT),              \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, BOOL),                        \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SHORT, LONGLONG),                    \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SHORT, UNSIGNED_LONGLONG),           \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_SHORT, LONGLONG),           \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_SHORT, UNSIGNED_LONGLONG),  \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BOOL, LONGLONG),                     \
      CYCLE_ESTIMATOR_ENTRY(popops, name, BOOL, UNSIGNED_LONGLONG),            \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_INT, LONGLONG),             \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_INT, UNSIGNED_LONGLONG),    \
      CYCLE_ESTIMATOR_ENTRY(popops, name, INT, LONGLONG),                      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, INT, UNSIGNED_LONGLONG),             \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SIGNED_CHAR, LONGLONG),              \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SIGNED_CHAR, UNSIGNED_LONGLONG),     \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_CHAR, LONGLONG),            \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_CHAR, UNSIGNED_LONGLONG),   \
      CYCLE_ESTIMATOR_ENTRY(popops, name, CHAR, LONGLONG),                     \
      CYCLE_ESTIMATOR_ENTRY(popops, name, CHAR, UNSIGNED_LONGLONG)

#define CAST_QUARTER_CYCLE_ESTIM_ENTRIES(name)                                 \
  CYCLE_ESTIMATOR_ENTRY(popops, name, HALF, QUARTER),                          \
      CYCLE_ESTIMATOR_ENTRY(popops, name, QUARTER, HALF),                      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, CHAR, QUARTER),                      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, QUARTER, CHAR),                      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_CHAR, QUARTER),             \
      CYCLE_ESTIMATOR_ENTRY(popops, name, QUARTER, UNSIGNED_CHAR),             \
      CYCLE_ESTIMATOR_ENTRY(popops, name, QUARTER, QUARTER)

poputil::internal::PerfEstimatorTable makePerfFunctionTable() {
  poputil::internal::PerfEstimatorTable table = {
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, FLOAT, FLOAT, FLOAT),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, HALF, HALF),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, ScalarMultiply1D, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, ScalarMultiply1DInplace, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, ScalarMultiply2D, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, ScalarMultiply2DInplace, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT, HALF, HALF,
                            true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT, HALF, HALF,
                            false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT, HALF, FLOAT,
                            true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT, HALF, FLOAT,
                            false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT,
                            UNSIGNED_INT, UNSIGNED_INT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT, INT, INT, true,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT,
                            UNSIGNED_INT, UNSIGNED_INT, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT, INT, INT, false,
                            false),

      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAdd2D, FLOAT, FLOAT, FLOAT),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAdd2D, HALF, HALF, HALF),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAdd2D, HALF, HALF, FLOAT),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, FLOAT, HALF, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, FLOAT, HALF, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, FLOAT, FLOAT, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, FLOAT, FLOAT, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, FLOAT, HALF,
                            true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, FLOAT, HALF,
                            false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, FLOAT, FLOAT,
                            true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, FLOAT, FLOAT,
                            false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, HALF, HALF, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, HALF, HALF, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, HALF, FLOAT, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, HALF, FLOAT, false,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT, UNSIGNED_INT,
                            UNSIGNED_INT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT, INT, INT, true, false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT, UNSIGNED_INT,
                            UNSIGNED_INT, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT, INT, INT, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, FLOAT, FLOAT,
                            FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, HALF,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, FLOAT, FLOAT,
                            FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, HALF,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, FLOAT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, UNSIGNED_INT,
                            UNSIGNED_INT, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, INT, INT, INT,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT, HALF,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT, HALF,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, UNSIGNED_INT,
                            UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, INT, INT, false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, HALF, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, HALF, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, HALF, false,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, FLOAT, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, FLOAT, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, FLOAT, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, FLOAT, FLOAT, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, FLOAT, FLOAT, false,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, FLOAT, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, FLOAT, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, FLOAT, FLOAT, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, HALF, HALF, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, HALF, HALF, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, HALF, FLOAT, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, HALF, FLOAT, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, FLOAT, FLOAT, false,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, HALF, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, HALF, HALF, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, HALF, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, HALF, FLOAT, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, FLOAT, FLOAT, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbYSupervisor, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbYSupervisor, HALF, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbYSupervisor, HALF, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbYSupervisor, HALF, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbY2D, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbY2D, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbY2D, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, XMinusaXPlusbY2D, HALF, false, false),

      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner1D),
      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner1DInPlace),
      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner2D),
      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner2DInPlace),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2D,
                            BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DInPlace,
                            BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF),

      BROADCAST_2TYPE_CYCLE_ESTIM_ENTRIES(BroadcastScalar2Types2DData),
      BROADCAST_2TYPE_CYCLE_ESTIM_ENTRIES(BroadcastScalar2Types1D),

      BROADCAST_CAST_CYCLE_ESTIM_ENTRIES(
          BroadcastScalar1DRelationalOpDualOutput),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(BroadcastVectorOuterByColumn1D,
                                                 true),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByColumn1DInPlace, true),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(BroadcastVectorOuterByRow1D,
                                                 true),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByRow1DInPlace, true),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(BroadcastVectorOuterByColumn1D,
                                                 false),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByColumn1DInPlace, false),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(BroadcastVectorOuterByRow1D,
                                                 false),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByRow1DInPlace, false),

      CYCLE_ESTIMATOR_ENTRY(popops, HadamardProd, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HadamardProd, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, Fill, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, BOOL),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, UNSIGNED_LONGLONG),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, LONGLONG),

      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, BOOL),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, UNSIGNED_LONGLONG),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, LONGLONG),

      CAST_CYCLE_ESTIM_ENTRIES(Cast1DSingleWorker),
      CAST_CYCLE_ESTIM_ENTRIES(Cast2D),
      CAST_CYCLE_ESTIM_ENTRIES(Cast1D),
      CAST_QUARTER_CYCLE_ESTIM_ENTRIES(Cast1DSingleWorker),
      CAST_QUARTER_CYCLE_ESTIM_ENTRIES(Cast2D),
      CAST_QUARTER_CYCLE_ESTIM_ENTRIES(Cast1D),

      CYCLE_ESTIMATOR_ENTRY(popops, CheckAccuracyWhenCast, FLOAT, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, BOOL),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, UNSIGNED_LONGLONG),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2D, LONGLONG),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, BOOL),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, UNSIGNED_LONGLONG),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2D, LONGLONG),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, BOOL),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, UNSIGNED_LONGLONG),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1D, LONGLONG),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, BOOL),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, UNSIGNED_LONGLONG),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1D, LONGLONG),

      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, UNSIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, SIGNED_CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, CHAR),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, HALF, HALF, true,
                            Operation::ADD),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, HALF, FLOAT, true,
                            Operation::ADD),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, HALF, HALF, false,
                            Operation::ADD),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, HALF, FLOAT, false,
                            Operation::ADD),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, FLOAT, FLOAT, false,
                            Operation::ADD),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, INT, INT, false,
                            Operation::ADD),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledMultiUpdateOp, UNSIGNED_INT,
                            UNSIGNED_INT, false, Operation::ADD),

      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateOp, HALF, true, Operation::MAX),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateOp, HALF, false, Operation::MAX),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateOp, FLOAT, false,
                            Operation::MAX),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateOp, INT, false, Operation::MAX),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateOp, UNSIGNED_INT, false,
                            Operation::MAX),

      CYCLE_ESTIMATOR_ENTRY(popops, SequenceSlice, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, SequenceSlice, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, SequenceSlice, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, SequenceSlice, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(popops, CircBufIncrIndex),
      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(popops, CircOffset),

      CYCLE_ESTIMATOR_ENTRY(popops, Select, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, Histogram2D, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram2D, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram2D, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram2D, HALF, false),

      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, FLOAT, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, Histogram1D, HALF, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, ForLoopCounter, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, ForLoopCounter, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, Clamp, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Clamp, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Clamp, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, ClampInPlace, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, ClampInPlace, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, ClampInPlace, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastClamp, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastClamp, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastClamp, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastClampInPlace, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastClampInPlace, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastClampInPlace, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, Iota, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Iota, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, UNSIGNED_INT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, UNSIGNED_INT,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, UNSIGNED_INT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, UNSIGNED_INT,
                            INT),

      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, INT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, INT, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, INT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertex, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertex, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertex, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, INT, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, INT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, INT, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, FLOAT, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, FLOAT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, FLOAT, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, HALF, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, HALF, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, HALF, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, UpdateColumnsDEC, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, UpdateIntervalsDEC, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, UpdateIntervalDEC, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, UpdateColumnsDEC, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, UpdateIntervalsDEC, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, UpdateIntervalDEC, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, SelectFromInterval, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectFromIntervals, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectFromRowsInColumns, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectFromInterval, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectFromIntervals, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectFromRowsInColumns, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf2D, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf2D, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf1D, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf1D, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf2D, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf2D, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf1D, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf1D, HALF, true),

      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistance, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistance, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistance, INT),

      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT,
                            FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT,
                            UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT, INT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal,
                            UNSIGNED_INT, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, INT, FLOAT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT,
                            FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT,
                            UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT, INT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal,
                            UNSIGNED_INT, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal,
                            UNSIGNED_INT, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal,
                            UNSIGNED_INT, INT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, INT, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, INT,
                            UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, INT, INT,
                            true),
  };

  for (const auto &entry : unaryOpPerfInfo) {
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp2D, entry.first.first,
                                          entry.first.second));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp1D, entry.first.first,
                                          entry.first.second));
  }
  for (const auto &entry : unaryOpInPlacePerfInfo) {
    table.push_back(CYCLE_ESTIMATOR_ENTRY(
        popops, UnaryOp2DInPlace, entry.first.first, entry.first.second));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(
        popops, UnaryOp1DInPlace, entry.first.first, entry.first.second));
  }

  for (const auto &entry : binaryOpPerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2D, op, type));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1D, op, type));
  }
  for (const auto &entry : binaryOpInPlacePerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2DInPlace, op, type));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DInPlace, op, type));
  }
  for (const auto &entry : broadcastOpPerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2D, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DData, op, type));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar1D, op, type));
  }
  for (const auto &entry : broadcastOpInPlacePerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DInPlace, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DDataInPlace, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar1DInPlace, op, type));
  }

  table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, NormaliseImage, UNSIGNED_CHAR, HALF));
  table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, NormaliseImage, HALF, HALF));
  table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, NormaliseImage, FLOAT, FLOAT));
  return table;
}

} // end namespace popops
