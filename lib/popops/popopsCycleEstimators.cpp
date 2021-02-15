// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popopsCycleEstimators.hpp"
#include "ExprOpUtil.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/FlopEstimation.hpp"
#include "poplibs_support/forceInterleavedEstimates.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/popopsPerformanceEstimation.hpp"
#include "popops/Expr.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

using namespace poplar;
using namespace poplibs_support;

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

// Computes the cycles used by the inner loop in one of the binary op codelets,
// both for supervisor and 2D version, in place or not.
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

// Computes the cycles used by one of the scalar broadcast supervisor codelets
static VertexPerfEstimate broadcastArithmeticSupervisorCycleEstimate(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const OpPerformanceInfo &perfInfo,
    std::uint64_t overheadPerLoop) {
  CODELET_FIELD(data);

  auto numWorkers = target.getNumWorkerContexts();

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

  auto numElems = iceil(data.size(), numWorkers);

  unsigned numLoops = iceil(numElems, elemsPerLoop);
  cycles += numLoops * cyclesPerLoop;
  std::uint64_t flops =
      static_cast<std::uint64_t>(data.size()) * flopsPerBinaryOpElement(op);
  std::uint64_t totalCycles = cycles * numWorkers + basicOpSupervisorOverhead();
  return {totalCycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar1DInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo &perfInfo = broadcastOpInPlacePerfInfo.at({op, type});
  // In the inplace case, if forcing use of interleaved memory, the fast
  // path can always be utilized to reduce the overhead by 1 cycle, making the
  // inner loop one cycle for ADD, SUB and MULTIPLY.
  return broadcastArithmeticSupervisorCycleEstimate(
      vertex, target, op, type, perfInfo,
      hasExternalCodelet(op, type) ? (getForceInterleavedEstimates() ? 0 : 1)
                                   : 4);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo &perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticSupervisorCycleEstimate(
      vertex, target, op, type, perfInfo, hasExternalCodelet(op, type) ? 1 : 4);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2Types1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const Type &outType) {
  // For vectorisation purposes, treat this as if it always processes float,
  // as it casts internally.  An extra cycle to cast to half output
  const OpPerformanceInfo &perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticSupervisorCycleEstimate(
      vertex, target, op, FLOAT, perfInfo, outType == FLOAT ? 0 : 1);
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

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorOuterByColumnInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  // If forcing use of interleaved memory, and in place, the overhead here
  // can be reduced as we can utilise a ldst64pace in the inner loop.
  return BroadcastVectorOuterCycleEstimate(
      vertex, target, op, type, getForceInterleavedEstimates() ? 0 : 1,
      allowMisaligned ? 25 : 7, false);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(
    BroadcastVectorOuterByColumnSupervisor)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            BinaryOpType op, const Type &type,
                                            bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, false);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorOuterByRowInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  // If forcing use of interleaved memory, and in place, the overhead here
  // can be reduced as we can utilise a ldst64pace in the inner loop.
  return BroadcastVectorOuterCycleEstimate(
      vertex, target, op, type, getForceInterleavedEstimates() ? 0 : 1,
      allowMisaligned ? 25 : 7, true);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(
    BroadcastVectorOuterByRowSupervisor)(const VertexIntrospector &vertex,
                                         const Target &target, BinaryOpType op,
                                         const Type &type,
                                         bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, true);
}

static VertexPerfEstimate broadcastArithmeticCycleEstimate(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const OpPerformanceInfo perfInfo,
    std::uint64_t overheadPerLoop) {
  CODELET_FIELD(data);
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

  std::uint64_t totalElems = 0;
  for (unsigned i = 0; i < data.size(); i++) {
    auto numElems = data[i].size();
    totalElems += numElems;
    unsigned numCycles = iceil(numElems, elemsPerLoop);
    cycles += numCycles * cyclesPerLoop;
    cycles += 28;
  }
  std::uint64_t flops = totalElems * flopsPerBinaryOpElement(op);
  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2DDataInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpInPlacePerfInfo.at({op, type});
  // In the inplace case, if forcing use of interleaved memory, the fast
  // path can always be utilized to reduce the overhead by 1 cycle, making the
  // inner loop one cycle for ADD, SUB and MULTIPLY.
  return broadcastArithmeticCycleEstimate(
      vertex, target, op, type, perfInfo,
      hasExternalCodelet(op, type) ? (getForceInterleavedEstimates() ? 0 : 1)
                                   : 4);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2DData)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          hasExternalCodelet(op, type) ? 1 : 4);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2Types2DData)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const Type &outType) {
  const OpPerformanceInfo perfInfo = broadcastOpPerfInfo.at({op, type});
  // For vectorisation purposes, treat this as if it always processes float
  // as casting makes this so. An extra cycle to cast the output to half
  return broadcastArithmeticCycleEstimate(vertex, target, op, FLOAT, perfInfo,
                                          outType == FLOAT ? 0 : 1);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpInPlacePerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          4);
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(BroadcastScalar2D)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            BinaryOpType op, const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          4);
}

enum class ScaledArithmeticOp { ADD, SUBTRACT, AXPLUSBY, AXMINUSBY };

VertexPerfEstimate scaledArithmeticSupervisorCycleEstimate(
    const VertexIntrospector &vertex, const Target &target,
    const Type &dataType, const Type &dataBType, const bool isConstant,
    const bool memConstrained, const ScaledArithmeticOp operation) {
  CODELET_FIELD(A);
  CODELET_FIELD(B);

  if (dataType == INT || dataType == UNSIGNED_INT) {
    std::uint64_t supervisorCycles = 53 // constant overhead
                                     + (26 * (A.size() / 3)); // main loop

    if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant) {
      supervisorCycles += 1;
    }

    if (A.size() % 3 == 0) {
      supervisorCycles += 6; // 6 cycle branch to skip the remainder loop
    } else {
      supervisorCycles += 6                        // --rem
                          + (26 * (A.size() % 3)); // remainder loop
    }
    supervisorCycles += 8; // constant epilogue overhead.
    if (!isConstant) {
      supervisorCycles += 6;
    }
    return supervisorCycles;
  } else {
    assert(isFPType(dataType));
  }

  // calculate count, rem and final
  const auto numWorkers = target.getNumWorkerContexts();
  const unsigned vectorWidth = target.getVectorWidth(dataType);

  const unsigned totalVectors = A.size() / vectorWidth;
  const unsigned remainingElems = A.size() % vectorWidth;

  const unsigned vectorsPerWorker = totalVectors / numWorkers;
  const unsigned remainingVectors = totalVectors % numWorkers;

  const auto aLayout = A.getProfilerVectorLayout(0);
  const auto bLayout = B.getProfilerVectorLayout(0);

  std::uint64_t perTypeSupervisorOverhead = 21;
  // scaled add and subtract for float and half maybe require an extra (bubble)
  // cycle to unpack the pointer.
  if (aLayout == layout::Vector::ScaledPtr64) {
    perTypeSupervisorOverhead += 6;
  }

  std::uint64_t supervisorCycles = perTypeSupervisorOverhead +
                                   basicOpSupervisorOverhead() +
                                   +(remainingElems == 0 ? 7 : 13) + 12;

  if (operation == ScaledArithmeticOp::AXPLUSBY && !isConstant) {
    supervisorCycles +=
        12 + poplibs::getUnpackCost(aLayout) + poplibs::getUnpackCost(bLayout);
  }
  if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant) {
    supervisorCycles += 7;
  }
  if (!isConstant) {
    // setzi + bri, but the branch skips a setzi already counted so just + 6.
    supervisorCycles += 6;
  }

  std::vector<unsigned> workerCycles;
  workerCycles.reserve(numWorkers);
  // Specific mixed precision half, float version
  if (dataType == HALF && dataBType == FLOAT) {
    const auto innerLoopCycles = memConstrained ? 2 : 3;
    for (unsigned wid = 0; wid < numWorkers; ++wid) {
      std::uint64_t cycles = 16; // constant worker prologue cycles
      const auto numVectors = vectorsPerWorker + (wid < remainingVectors);
      if (numVectors != 0) {
        if (numVectors < 3) {
          cycles += 8 // inner loop for < 3 constant overhead (processes 1)
                    + (4 * (numVectors - 1)); // loop cycles
        } else {
          cycles += 16 // inner loop for >= 3 constant overhead (processes 3)
                    + (innerLoopCycles * (numVectors - 3)); // loop cycles
        }
      }
      cycles += 2; // workerID == rem
      if (wid == remainingVectors) {
        cycles += 1; // final == 0?
        if (remainingElems != 0) {
          cycles += 5; // unpack triPtr and check if at least 2 remain
          if (remainingElems >= 2) {
            cycles += 7; // process 2 of the remainder.
            if (remainingElems == 3) {
              cycles += 6; // process final half
            }
          }
        }
      }
      cycles += 1; // exitz
      workerCycles.push_back(cycles);
    }
  }
  // (half,half), (float, half) and (float, float) versions
  else {
    // half/float case handled above
    assert(dataType != HALF || dataBType != FLOAT);
    unsigned innerLoopCycles =
        memConstrained ? 2 : (dataType == dataBType ? 3 : 4);

    if (getForceInterleavedEstimates() && (dataType == dataBType)) {
      // Reduce inner loop cycles by one for (half,half), (float, float) when
      // using interleaved memory.
      innerLoopCycles -= 1;
    }

    for (unsigned wid = 0; wid < numWorkers; ++wid) {
      std::uint64_t cycles = 15; // constant worker prologue cycles
      const auto numVectors = vectorsPerWorker + (remainingVectors < 0);
      if (numVectors != 0) {
        cycles += 6 // inner loop constant overhead
                  + (innerLoopCycles * (numVectors - 1)); // loop cycles
      }
      cycles += 2; // workerID == rem
      if (wid == remainingVectors) {
        cycles += 1; // final == 0?
        if (remainingElems != 0) {
          if (dataType == FLOAT) {
            cycles += 8; // process final float.
          } else {
            cycles += 5; // unpack triPtr and check if at least 2 remain
            if (remainingElems >= 2) {
              cycles += 7; // process 2 of the remainder.
              if (remainingElems == 3) {
                cycles += 6; // process final half
              }
            }
          }
        }
      }
      cycles += 1; // exitz
      workerCycles.push_back(cycles);
    }
  }

  auto maxWorkerCycles =
      *std::max_element(std::begin(workerCycles), std::end(workerCycles));
  std::uint64_t flops =
      A.size() * (flopsPerBinaryOpElement(BinaryOpType::ADD) +
                  flopsPerBinaryOpElement(BinaryOpType::MULTIPLY));
  std::uint64_t cycles = supervisorCycles + maxWorkerCycles * numWorkers;
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
    const Type &BType, const bool memConstrained) {
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
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const bool isConstant, const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, AType,
                                                 isConstant, memConstrained,
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

  unsigned innerLoopCycles = memConstrained ? 2 : 3;
  if (getForceInterleavedEstimates()) {
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
      cycles += poplibs::getUnpackCost(bLayout);
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
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, false,
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
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isConstant, const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, memConstrained,
                                         isConstant,
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
  for (unsigned i = ceilLog2(8); i < ceilLog2(grainSize); ++i) {
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
  for (unsigned i = ceilLog2(4); i < ceilLog2(vectorWidth); ++i) {
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

  std::uint64_t numCycles = 5; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    // loop overhead. A bit more for halves
    if (type == HALF)
      numCycles += 17;
    else
      numCycles += 11;

    auto coreFunc = type == HALF ? vectorInnerAddCoreCycles_half
                                 : vectorInnerAddCoreCycles_float;

    numCycles += coreFunc(vectorWidth, BLen[i], dataBlockCount[i]);
  }

  return numCycles + 1; // exitnz
}

// Cycle count for the common part of all the VectorInnerSupervisor ADD and
// SUBTRACT codelets
std::uint64_t vectorInnerSupervisorAddCycles(unsigned numWorkerContexts,
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

  std::uint64_t numCycles = 8; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    // loop overhead. A bit more for halves
    if (type == HALF)
      numCycles += 17;
    else
      numCycles += 11;

    auto coreFunc = type == HALF ? vectorInnerDivCoreCycles_half
                                 : vectorInnerDivCoreCycles_float;

    numCycles += coreFunc(BLen[i], dataBlockCount[i]);
  }

  return numCycles + 1; // exitnz
}

// Cycle count for the common part of all the VectorInnerSupervisor DIV
// codelets.
std::uint64_t vectorInnerSupervisorDivCycles(unsigned numWorkerContexts,
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
  for (unsigned i = ceilLog2(4); i < ceilLog2(vectorWidth); ++i) {
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
  for (unsigned i = ceilLog2(4); i < ceilLog2(vectorWidth); ++i) {
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

  std::uint64_t numCycles = 5; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    numCycles += type == HALF ? 15 : 11; // loop overhead.

    auto coreFunc = type == HALF ? vectorInnerMulCoreCycles_half
                                 : vectorInnerMulCoreCycles_float;

    numCycles += coreFunc(vectorWidth, BLen[i], dataBlockCount[i], false);
  }

  // Exit
  numCycles += 1;

  return numCycles;
}

// Cycle count for the common part of all the VectorInnerSupervisor MUL
// codelets.
std::uint64_t vectorInnerSupervisorMulCycles(unsigned numWorkerContexts,
                                             unsigned vectorWidth,
                                             uint32_t BLen,
                                             uint16_t dataBlockCountPacked,
                                             const Type &type, bool inPlace) {
  // These numbers may not be exact (e.g. the remainder of
  // dataBlockCountPacked is ignored).

  // Common supervi
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

static std::uint64_t flopsForBinaryOp2D(const std::vector<unsigned> &bLen,
                                        const Type &type, BinaryOpType op) {
  const auto totalElems = std::accumulate(bLen.begin(), bLen.end(), 0);
  return convertToTypeFlops(static_cast<std::uint64_t>(totalElems) *
                                flopsPerBinaryOpElement(op),
                            type);
}

static std::uint64_t flopsForUnaryOp2D(const std::vector<unsigned> &aLen,
                                       const Type &type, UnaryOpType op) {
  const auto totalElems = std::accumulate(aLen.begin(), aLen.end(), 0);
  return convertToTypeFlops(static_cast<std::uint64_t>(totalElems) *
                                flopsPerUnaryOpElement(op),
                            type);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorInnerSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  unsigned numWorkerContexts = target.getNumWorkerContexts();
  unsigned vectorWidth = target.getVectorWidth(type);
  auto flops = flopsForBinaryOp2D({BLen}, type, op);
  // Additional branch in the supervisor, and preamble instructions in the
  // worker part.
  switch (op) {
  case BinaryOpType::ADD: {
    const unsigned addedSuperOverhead = 6;
    const unsigned addedWorkerOverhead = 3;
    return {vectorInnerSupervisorAddCycles(numWorkerContexts, vectorWidth, BLen,
                                           dataBlockCountPacked, type) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::DIVIDE: {
    return {vectorInnerSupervisorDivCycles(numWorkerContexts, BLen,
                                           dataBlockCountPacked, type) +
                1 + 3,
            flops};
  }
  case BinaryOpType::SUBTRACT: {
    const unsigned addedSuperOverhead = 6;
    const unsigned addedWorkerOverhead = 3;
    return {vectorInnerSupervisorAddCycles(numWorkerContexts, vectorWidth, BLen,
                                           dataBlockCountPacked, type) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::MULTIPLY: {
    const unsigned addedSuperOverhead = 0;
    const unsigned addedWorkerOverhead = 2;
    return {vectorInnerSupervisorMulCycles(numWorkerContexts, vectorWidth, BLen,
                                           dataBlockCountPacked, type, false) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(
    BroadcastVectorInnerInPlaceSupervisor)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           BinaryOpType op, const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  const unsigned numWorkerContexts = target.getNumWorkerContexts();
  const unsigned vectorWidth = target.getVectorWidth(type);
  auto flops = flopsForBinaryOp2D({BLen}, type, op);

  switch (op) {
  case BinaryOpType::ADD: {
    const auto addedWorkerOverhead = 2;
    return {vectorInnerSupervisorAddCycles(numWorkerContexts, vectorWidth, BLen,
                                           dataBlockCountPacked, type) +
                addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::DIVIDE: {
    return {vectorInnerSupervisorDivCycles(numWorkerContexts, BLen,
                                           dataBlockCountPacked, type) +
                2,
            flops};
  }
  case BinaryOpType::SUBTRACT: {
    const auto addedSuperOverhead = 6;
    const auto addedWorkerOverhead = 3;
    // Additional branches in the supervisor and worker part.
    return {vectorInnerSupervisorAddCycles(numWorkerContexts, vectorWidth, BLen,
                                           dataBlockCountPacked, type) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  case BinaryOpType::MULTIPLY: {
    const unsigned addedSuperOverhead = 6;
    const unsigned addedWorkerOverhead = 3;
    return {vectorInnerSupervisorMulCycles(numWorkerContexts, vectorWidth, BLen,
                                           dataBlockCountPacked, type, true) +
                addedSuperOverhead + addedWorkerOverhead * numWorkerContexts,
            flops};
  }
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BroadcastVectorInner2D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_VECTOR_VALS(BLen, uint32_t);
  CODELET_VECTOR_VALS(dataBlockCount, uint32_t);

  const unsigned vectorWidth = target.getVectorWidth(type);
  auto flops = flopsForBinaryOp2D(BLen, type, op);

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
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_VECTOR_VALS(BLen, uint32_t);
  CODELET_VECTOR_VALS(dataBlockCount, uint32_t);

  const unsigned vectorWidth = target.getVectorWidth(type);
  auto flops = flopsForBinaryOp2D(BLen, type, op);

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
  auto flops = flopsForBinaryOp2D({totalElems}, type, BinaryOpType::MULTIPLY);
  return {cycles, flops};
}

std::uint64_t _fillCycleEstimate(std::uint64_t size, const Target &target,
                                 const Type &type) {
  const bool isHalf = type == HALF;
  const auto width = target.getDataPathWidth() / (isHalf ? 16 : 32);

  if (isHalf) {
    // Cycle breakdown:
    //
    //  In an eight byte interval there is one 8-byte-aligned address, two
    //  four-byte-aligned addresses and four two-byte-aligned addresses. So if
    //  the aligned addresses are chosen randomly, then on average
    //  two-byte-alignment will occur ~57% of the time, four-byte-alignement
    //  will occur ~29% of the time and eight byte alignment will occur ~14% of
    //  the time. So the return value should slightly bias towards
    //  2-byte-aligment.
    //
    //      size  | 2-byte-aligned | 4-byte-aligned | 8-byte-aligned | return
    //   ---------+----------------+----------------+----------------+---------
    //    2 bytes | 15             | 16             | 16             | 15
    //    4 bytes | 20             | 23             | 22             | 21
    //    8 bytes | 30             | 24             | 22             | 27
    //   16 bytes | 31             | 25             | 23             | 28
    switch (size) {
    case 1:
      return 15;
    case 2:
      return 21;
    default:
      return 26 + size / width;
    }
  }

  // Cycle breakdown:
  //
  // + 16 cycles for pre-loop code, such as loading data and checking alignment.
  // + 6 cycles for all the post-loop checks.
  // + 1 cycle if there are 4 bytes left after the loop.
  // + 1 cycle to on average account for 4 byte alignemnt rather than 8.
  //   There's an additional two cycles if the data is only 4 byte aligned
  //   rather than 8 byte aligned, and as 4 byte alignments are twice as likely
  //   to occur than 8 byte alignments this function could bias towards 4 byte
  //   alignments however in practise the cycle difference is so small, that the
  //   result of the bias is negligble and the returned cycles tend to the
  //   average.
  //
  // This ends up being roughly right in the small cases too:
  //
  //      size  | 4-byte-aligned | 8-byte-aligned | return
  //   ---------+----------------+----------------+----------
  //    4 bytes | 24             | 23             | 24
  //    8 bytes | 25             | 23             | 24
  //   16 bytes | 26             | 24             | 25
  return 16 + size / width + 6 + (size % width == 1) + 1;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Fill)(const VertexIntrospector &vertex,
                               const Target &target, const Type &type) {
  return _fillCycleEstimate(vertex.getFieldInfo("out").size(), target, type);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Fill2d)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &type) {
  // UNSIGNED_INT and INT use the same cycle estimator as FLOAT
  // see: T24721
  const auto out = vertex.getFieldInfo("out");
  std::uint64_t cycles = 5 + (type != HALF);
  for (unsigned i = 0; i < out.size(); ++i)
    cycles += _fillCycleEstimate(out[i].size(), target, type);
  // The 1d fill function includes overhead from loading variables which takes 5
  // cycles for half types and 6 cycles for other types, but the 2d fill
  // function has an additional per-loop overhead of 3 cycles, so subtract two
  // cycles for each call to fill to account for the difference (three cycles
  // for non-halves).
  cycles -= (2 + (type != HALF)) * out.size();
  return cycles;
}

static std::uint64_t castFlops(const Type &fromType, const Type &toType,
                               unsigned numElems) {

  return isFPType(fromType) || isFPType(toType)
             ? static_cast<std::uint64_t>(numElems) * 1
             : 0;
}

// Returns the cycles of one to the 'cast_XXX_XXX_core' functions in assembly,
// or the equivalent section of code in one of the C++ codelets.
static uint64_t castWorkerCycles(const Target &target, const unsigned numElems,
                                 const Type &fromType, const Type &toType) {
  std::uint64_t cycles = 0;
  const bool isCharFloat = (fromType == UNSIGNED_CHAR ||
                            fromType == SIGNED_CHAR || fromType == CHAR) &&
                           (toType == FLOAT || toType == HALF);
  const bool isFloatChar =
      (fromType == FLOAT || fromType == HALF) &&
      (toType == UNSIGNED_CHAR || toType == SIGNED_CHAR || toType == CHAR);

  if (isCharFloat || isFloatChar) {
    // These assembly functions have a common structure, using an atom-sized
    // (2/4 elems) pipelined loop and a "0,1,2 or 3" remainder section.
    auto workCycles = [&](auto atomSize, auto fillDrainCycles,
                          auto cyclesPerLoop, auto rem1, auto rem2, auto rem3) {
      unsigned c = 0;
      unsigned nAtom = numElems / atomSize;
      if (nAtom > 0) {
        c += fillDrainCycles + (nAtom - 1) * cyclesPerLoop;
      }
      unsigned rem = numElems % atomSize;
      if (rem == 0) {
        return c + 2;
      } else if (rem == 1) {
        return c + rem1;
      } else if (rem == 2) {
        return c + rem2;
      } else if (rem == 3) {
        return c + rem3;
      } else {
        throw poputil::poplibs_error("in castWorkerCycles/workCycles, "
                                     "remainder must be 0..3, cannot be " +
                                     std::to_string(rem));
      }
    };
    // setup clamping when casting FROM int8 types
    unsigned clampSetupCycles = (fromType == UNSIGNED_CHAR) ? 3 : 4;
    cycles += 4; // all functions start with a 4 instruction sequence.
    // CastFromInt8.S
    if (fromType == UNSIGNED_CHAR && toType == HALF) {
      cycles += workCycles(4, 14, 9, 12, 14, 22);
    } else if ((fromType == SIGNED_CHAR || fromType == CHAR) &&
               toType == HALF) {
      cycles += workCycles(4, 15, 13, 12, 14, 22);
    } else if (fromType == UNSIGNED_CHAR && toType == FLOAT) {
      cycles += workCycles(4, 14, 10, 9, 15, 21);
    } else if ((fromType == SIGNED_CHAR || fromType == CHAR) &&
               toType == FLOAT) {
      cycles += workCycles(2, 9, 7, 10, 0, 0);
      // CastToInt8.S:
    } else if (fromType == HALF) {
      cycles += clampSetupCycles + workCycles(4, 15, 12, 13, 19, 26);
    } else if (fromType == FLOAT) {
      cycles += clampSetupCycles + workCycles(4, 14, 10, 12, 19, 25);
    }
  } else {
    const auto dataPathWidth = target.getDataPathWidth() / 8;
    const auto fromLoadWidth = dataPathWidth / target.getTypeSize(fromType);
    const auto toStoreWidth = dataPathWidth / target.getTypeSize(toType);

    // We take a guess that the vector width possible for the op will be a
    // function of the available load/store bandwidth and number
    // of read/write ports. e.g. f32v2tof16 is 2 reads of 64-bit and 1 write
    // of 64-bits and f16v2tof32, a 64-bit write is the bottleneck.
    constexpr std::size_t readPorts = 2, writePorts = 1;
    const bool conversionIsAuxPipeline =
        (fromType == FLOAT || fromType == HALF) &&
        (toType == FLOAT || toType == HALF);

    // If not aux pipeline (i.e. not floating point conversion) we give an
    // innaccurate guess anyhow as we will be using C++ code to perform
    // the conversion.
    const auto opVectorWidth =
        conversionIsAuxPipeline
            ? std::min(fromLoadWidth * readPorts, toStoreWidth * writePorts)
            : 1;

    // We then get the number of cycles to calculate each of these vectors.
    // NOTE: We don't use interleaved memory currently hence we don't utilise
    // multiple read ports. We do assume we use separate memory elements for
    // load/store to overlap loads/stores where possible.
    const auto loadCyclesPerVector =
        opVectorWidth /
        std::min(opVectorWidth,
                 fromLoadWidth *
                     (getForceInterleavedEstimates() ? readPorts : 1));
    const auto storeCyclesPerVector =
        opVectorWidth / std::min(opVectorWidth, toStoreWidth);
    const auto cyclesPerVector =
        std::max(loadCyclesPerVector, storeCyclesPerVector) +
        !conversionIsAuxPipeline;

    // Prologue cycles based on Half_Float_core assuming alignment and enough
    // elements to process.
    cycles += 19;
    // Cycles for processing vectors
    cycles += cyclesPerVector * (numElems / opVectorWidth);

    // Rough estimation of cycles for processing remainders. This should be
    // relatively insignificant so rough is fine.
    const auto remainingElems = numElems % opVectorWidth;
    const auto maxRemainderBit = ceilLog2(opVectorWidth);
    for (unsigned i = 0; i < maxRemainderBit; ++i) {
      const auto remainder = (1u << i);
      // Check the remainder. Conservative in that some paths will
      // exit early.
      cycles += 1;
      if (remainingElems & remainder) {
        // 2 cycles, 1 to convert and 1 to store assuming input is
        // already loaded in memory from vector path.
        cycles += 2;
      }
    }
  }
  return cycles;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Cast)(const VertexIntrospector &vertex,
                               const Target &target, const Type &fromType,
                               const Type &toType) {
  CODELET_SCALAR_VAL(numElems, unsigned);

  // Estimate written based on vertices with assembly implementations.
  // Not realistic for others.
  constexpr std::uint64_t getParamsCycles = 3;
  // Get parameters, call core function and exitz
  std::uint64_t cycles = getParamsCycles + 2 +
                         castWorkerCycles(target, numElems, fromType, toType);

  return {cycles, castFlops(fromType, toType, numElems)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CastSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fromType, const Type &toType) {
  CODELET_SCALAR_VAL(partitionParams, unsigned);
  const unsigned workerElems = partitionParams >> 9;
  const unsigned workerCount = (partitionParams >> 6) & 0x7;
  const unsigned workerLast = (partitionParams >> 3) & 0x7;
  const unsigned deltaLast = (partitionParams & 0x7);
  const unsigned totalElems = workerElems * workerCount + deltaLast;

  // Work out workers doing unique workloads from the partitionParams and
  // find the maximum cycles for any of them.
  unsigned max = workerElems;
  unsigned maxM4 = workerElems - 4;
  unsigned numMaxWorkers = workerCount;
  unsigned numMaxM4Workers = target.getNumWorkerContexts() - workerCount;

  // Worker entry from the supervisor, including exitz and call.
  std::uint64_t maxCycles = 0;
  if (workerLast < workerCount) {
    numMaxWorkers--;
    maxCycles = std::max(
        maxCycles, castWorkerCycles(target, max - deltaLast, fromType, toType));
  } else {
    numMaxM4Workers--;
    maxCycles = std::max(maxCycles, castWorkerCycles(target, maxM4 - deltaLast,
                                                     fromType, toType));
  }
  if (numMaxWorkers) {
    maxCycles =
        std::max(maxCycles, castWorkerCycles(target, max, fromType, toType));
  }
  if (numMaxM4Workers) {
    maxCycles =
        std::max(maxCycles, castWorkerCycles(target, maxM4, fromType, toType));
  }
  const std::uint64_t fromSupervisorWorkerCycles = 23;
  maxCycles += fromSupervisorWorkerCycles;

  // setzi, runall, sync, br
  // Assumes runall takes 6 cycles workers are balanced such that
  // sync takes 6 cycles and br takes 6 cycles.
  return {19 + target.getNumWorkerContexts() * maxCycles,
          castFlops(fromType, toType, totalElems)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Cast2d)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &fromType,
                                 const Type &toType) {
  std::uint64_t cycles = 5;
  const auto dst = vertex.getFieldInfo("dst");
  CODELET_FIELD(src);
  assert(src.size() == dst.size());
  unsigned totalElems = 0;
  for (unsigned i = 0; i != dst.size(); ++i) {
    assert(src[i].size() == dst[i].size());
    // Outer-loop cycles including call plus core function per-vector
    cycles += 11 + castWorkerCycles(target, dst[i].size(), fromType, toType);
    totalElems += src[i].size();
  }
  return {cycles, castFlops(fromType, toType, totalElems)};
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

std::uint64_t getBinaryOp1DInPlaceSupervisorEstimate(
    const poplar::Target &target, const Type &type,
    const popops::expr::BinaryOpType op, const unsigned numElems) {
  auto superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 58;
  const auto &info = binaryOpInPlacePerfInfo.at({op, type});

  const auto numWorkers = target.getNumWorkerContexts();
  auto numElemsPerWorker = iceil(numElems, numWorkers);
  workerCycles +=
      binaryOpInnerLoopCycles(target, op, type, info, numElemsPerWorker, true);
  return numWorkers * workerCycles + superviserOverhead;
}

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
  return {cycles, flopsForUnaryOp2D({totalElems}, type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(UnaryOp1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 28;
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
          flopsForUnaryOp2D({static_cast<unsigned>(in.size())}, type, op)};
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
  return {cycles, flopsForUnaryOp2D({totalElems}, type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(UnaryOp1DInPlaceSupervisor)(
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
          flopsForUnaryOp2D({static_cast<unsigned>(inOut.size())}, type, op)};
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
  return {cycles, flopsForBinaryOp2D({totalElems}, type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BinaryOp1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  uint64_t supervisorOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = (type == FLOAT) ? 41 : 32;
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
          flopsForBinaryOp2D({static_cast<unsigned>(in1.size())}, type, op)};
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
  return {cycles, flopsForBinaryOp2D({totalElems}, type, op)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(BinaryOp1DInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  return {
      getBinaryOp1DInPlaceSupervisorEstimate(target, type, op, in1Out.size()),
      flopsForBinaryOp2D({static_cast<unsigned>(in1Out.size())}, type, op)};
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

std::uint64_t getDynamicSlice1dEstimate(const poplar::Target &target,
                                        const Type &type,
                                        const unsigned regionSize,
                                        const unsigned numSubElements) {
  const unsigned numWorkers = target.getNumWorkerContexts();
  const unsigned elementsPerWorker = (regionSize + numWorkers - 1) / numWorkers;
  unsigned vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);

  // Supervisor overhead.
  auto superCycles = basicOpSupervisorOverhead() + 1 + 6 + 1 + 6;

  // This is the more optimistic path - where the inner loop is copying
  // aligned data
  unsigned nCopies = elementsPerWorker / vectorWidth;
  auto workerCycles = 41 + (27 + nCopies) * numSubElements;
  auto cycles = superCycles + workerCycles * numWorkers;

  return cycles;
}

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
      poplibs::getUnpackCost(histogram.getProfilerVectorLayout(0));
  const auto unpackCostLimits =
      poplibs::getUnpackCost(limits.getProfilerVectorLayout(0));

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

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(HistogramSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isAbsolute, const bool splitByLimits) {
  CODELET_FIELD(data);
  CODELET_FIELD(histogram);
  CODELET_FIELD(limits);
  CODELET_SCALAR_VAL(histogramCount, unsigned);
  const auto unpackCostHistogram =
      poplibs::getUnpackCost(histogram.getProfilerVectorLayout(0));
  const auto unpackCostLimits =
      poplibs::getUnpackCost(limits.getProfilerVectorLayout(0));

  const auto vectorWidth =
      target.getDataPathWidth() / (type == FLOAT ? 32 : 16);
  auto numWorkers = target.getNumWorkerContexts();
  auto flops = histogramFlops(data.size(), isAbsolute);

  if (splitByLimits) {
    return {histogramSupervisorByLimitEstimate(
                data.size(), histogramCount, isAbsolute, type == HALF,
                numWorkers, vectorWidth, unpackCostHistogram, unpackCostLimits),
            convertToTypeFlops(flops, type)};
  } else {
    return {histogramSupervisorByDataEstimate(
                data.size(), histogramCount, isAbsolute, type == HALF,
                numWorkers, vectorWidth, unpackCostHistogram, unpackCostLimits),
            convertToTypeFlops(flops, type)};
  }
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

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicSlice2d)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
      vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
      vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  const unsigned numRegions =
      vertex.getFieldInfo("numRegions").getInitialValue<unsigned>(target);
  auto cycles = 23;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    if (type == HALF)
      cycles += (31 + 2 * nVectors) * numSubElements + 13;
    else
      cycles += (29 + nVectors) * numSubElements + 13;
  }
  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicUpdateSlice2d)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
      vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
      vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  const unsigned numRegions =
      vertex.getFieldInfo("numRegions").getInitialValue<unsigned>(target);
  auto cycles = 23;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    if (type == HALF)
      cycles += (31 + 2 * nVectors) * numSubElements + 13;
    else
      cycles += (29 + nVectors) * numSubElements + 13;
  }
  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicSlice1d)(
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
  return getDynamicSlice1dEstimate(target, type, regionSize, numSubElements);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(DynamicUpdateSlice1d)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  return MAKE_PERF_ESTIMATOR_NAME(DynamicSlice1d)(vertex, target, type);
}

static std::uint64_t multiSlicer(const VertexIntrospector &vertex,
                                 const Target &target, const Type &type,
                                 bool /*isUpdate*/) {
  const auto regionSize =
      vertex.getFieldInfo("regionSize").getInitialValue<unsigned>(target);
  const auto offsets = vertex.getFieldInfo("offsets");

  auto numOffsets = offsets.size();
  assert(numOffsets > 0);
  unsigned vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  auto copiesPerOffset = (regionSize + vectorWidth - 1) / vectorWidth;

  std::uint64_t callOverhead = 16;

  // load offset, compare, cond-branch, mpy to get idx, (load, store) per entry,
  // outer loop
  std::uint64_t coreCycles = numOffsets * (19 + copiesPerOffset * 3);

  return callOverhead + coreCycles;
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

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(MultiUpdateAdd)(const VertexIntrospector &vertex,
                                         const Target &target, const Type &type,
                                         const bool &subWordWritesRequired) {

  // based off the assembly (optimistic for integral types which are still
  // handled by the compiler).
  CODELET_FIELD(offsets);
  CODELET_SCALAR_VAL(regionSize, unsigned short);

  std::uint64_t cycles = 3; // load size, zero check and exitz.
  if (offsets.size() == 0) {
    return cycles;
  }

  // pre-outer loop overhead.
  cycles += type == FLOAT ? 14 : 15;

  // outer loop overhead, before and after the inner loop.
  // cycle cost is data dependent on values of offsets, assuming worst case.
  std::uint64_t outerLoopCycles = type == FLOAT ? 11 : 12;

  // inner loop cost.
  // Note gcd is used here for e.g. CPU where the atomic write size is 1.
  const unsigned bytesPerAtom =
      lcm(target.getAtomicStoreGranularity(), target.getTypeSize(type));
  const unsigned elemsPerAtom = bytesPerAtom / target.getTypeSize(type);
  // for the assembly implementation regionSize % vectorWidth == 0 must be
  // zero.
  if (subWordWritesRequired) {
    assert(type == HALF);
    // Not based on anything in particular other than per-element cost in
    // generated code for C++ being high (even higher for half type).
    outerLoopCycles += regionSize * 20;
  } else {
    assert(regionSize != 0 && regionSize % elemsPerAtom == 0);
    outerLoopCycles += (regionSize / elemsPerAtom - 1) * 3;
  }

  cycles += outerLoopCycles * offsets.size();
  return {cycles, static_cast<std::uint64_t>(regionSize) *
                      (flopsPerBinaryOpElement(BinaryOpType::ADD) +
                       flopsPerBinaryOpElement(BinaryOpType::MULTIPLY))};
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

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(HasNaNOrInfSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    bool hasNaNOrInf) {
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
MAKE_PERF_ESTIMATOR_NAME(HasNaNOrInf)(const VertexIntrospector &vertex,
                                      const Target &target, const Type &inType,
                                      bool hasNaNOrInf) {
  CODELET_FIELD(in);
  return hasNan2DCycles(in, target, inType, hasNaNOrInf);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Transpose2d)(const VertexIntrospector &vertex,
                                      const Target &target, const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcRows, unsigned);
  CODELET_SCALAR_VAL(numSrcColumns, unsigned);

  const bool is4ByteType =
      (type == FLOAT || type == UNSIGNED_INT || type == INT);
  // Just to be sure we don't see something unexpected:
  assert(type == FLOAT || type == HALF || type == UNSIGNED_INT ||
         type == UNSIGNED_SHORT || type == INT || type == SHORT);
  const auto matrices = dst.size();
  std::uint64_t cycles;

// TODO T14719: Derive this from IPUArchInfo
#define CSR_W_REPEAT_COUNT__VALUE__MASK 0x0FFF
  auto const hardwareRptCountConstraint = CSR_W_REPEAT_COUNT__VALUE__MASK + 1;

  if (is4ByteType) {
    if (((numSrcRows & 1) == 0) && ((numSrcColumns & 1) == 0) &&
        (numSrcColumns / 2 < hardwareRptCountConstraint) &&
        (numSrcRows * (numSrcColumns - 2) / 2 < 512) && // Largest stride used
        (numSrcRows < 512)) { // Used as a stride over output.
      // Float, fast path estimates
      cycles = 27 + matrices * (11 + (numSrcRows / 2) *
                                         (6 + 3 * (numSrcColumns / 2 - 1)));
    } else {
      // Float, slow path estimates based on numSrcRows being even
      cycles = 13 + matrices * (8 + numSrcColumns * (5 + (numSrcRows * 4) / 2));
    }
  } else {
    if (((numSrcRows & 3) == 0) && ((numSrcColumns & 3) == 0) &&
        (numSrcColumns >= 8) &&
        (numSrcColumns / 4 < hardwareRptCountConstraint) &&
        (1 + 3 * (numSrcColumns / 4) < 512)) { // Largest stride used
      // Half, fast path estimates, with >=8 input columns
      cycles = 39 + matrices * (12 + (numSrcRows / 4) *
                                         (15 + 4 * (numSrcColumns / 4 - 2)));
    } else if (((numSrcRows & 3) == 0) && (numSrcColumns == 4) &&
               (numSrcRows / 4 < hardwareRptCountConstraint) &&
               (1 + 3 * (numSrcRows / 4) < 512)) { // Largest stride used
      // Half, fast path estimates, 4x4 or Nx4 cases
      if (numSrcRows == 4)
        cycles = 34 + 15 * matrices;
      else
        cycles = 30 + matrices * (17 + (20 + 4 * (numSrcRows / 4 - 2)));
    } else {
      // Half, slow path estimates based on numSrcRows being even
      cycles = 15 + matrices * (8 + numSrcColumns * (5 + (numSrcRows * 5) / 2));
    }
  }
  return cycles;
}

// Cycle estimation for the "Transpose" worker (half, fast version)
static std::uint64_t TransposeWorkerCycles(const unsigned short numSrcRowsD4,
                                           const unsigned short numSrcColumnsD4,
                                           const unsigned short numMatrices,
                                           const layout::Vector srcLayout) {
  std::uint64_t cycles;
  if (numSrcRowsD4 == 1 && numSrcColumnsD4 == 1) {
    if (numMatrices == 1)
      cycles = 17 + 12;
    else
      cycles = 17 + 20 + (numMatrices - 2) * 4;
  } else if (numSrcColumnsD4 == 1) {
    cycles = 27 + numMatrices * (15 + (20 + 4 * (numSrcRowsD4 - 2)));
  } else {
    cycles = 29 + numMatrices *
                      (18 + numSrcRowsD4 * (12 + 4 * (numSrcColumnsD4 - 2)));
  }

  // extra might be needed in the prologue to unpack the pointers
  cycles += poplibs::getUnpackCost(srcLayout);

  return cycles;
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Transpose)(const VertexIntrospector &vertex,
                                    const Target &target, const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcRowsD4, unsigned short);
  CODELET_SCALAR_VAL(numSrcColumnsD4, unsigned short);
  CODELET_SCALAR_VAL(numTranspositionsM1, unsigned short);

  const auto srcLayout = src.getProfilerVectorLayout(0);
  assert(srcLayout == dst.getProfilerVectorLayout(0));

  const unsigned matrices = numTranspositionsM1 + 1;

  // only 2-byte types supported
  assert(type == HALF || type == UNSIGNED_SHORT || type == SHORT);

  return TransposeWorkerCycles(numSrcRowsD4, numSrcColumnsD4, matrices,
                               srcLayout);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(TransposeSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcRowsD4, unsigned short);
  CODELET_SCALAR_VAL(numSrcColumnsD4, unsigned short);
  CODELET_SCALAR_VAL(numTranspositions, unsigned short);

  const auto srcLayout = src.getProfilerVectorLayout(0);
  assert(srcLayout == dst.getProfilerVectorLayout(0));

  // only 2-byte types supported
  assert(type == HALF || type == UNSIGNED_SHORT || type == SHORT);

  // This supervisor vertex will start 6 workers: 'workerCount' workers will
  // do 'numTranspositions' matrices, and (6-workerCount) will do
  // one less matrices (numTranspositions-1). We compute the cycles for
  // the slowest ones (transposing 'numTranspositions' matrices).
  // We also add the additional cycles executed, compared to the 'plain'
  // "Transpose" codelet.
  // transpose_half_from_supervisor does 20 or 21 cycles and jumps
  // the first 7 in the worker codelet.
  const std::uint64_t overhead = poplibs::getUnpackCost(srcLayout);
  std::uint64_t maxCycles =
      TransposeWorkerCycles(numSrcRowsD4, numSrcColumnsD4, numTranspositions,
                            srcLayout) +
      overhead - 7;

  // Add 7 for the supervisor code
  return 7 + 6 * maxCycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CompareAndSwapAtDistance)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &keyType) {
  // TODO:
  return 0;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CompareAndSwapAtDistanceKeyVal)(
    const VertexIntrospector &vertex, const Target &target, const Type &keyType,
    const Type &valueType) {
  std::uint64_t cycles = 0;
  std::uint64_t flops = 0;

  CODELET_VECTOR_2D_VALS(worklists, unsigned short);
  CODELET_SCALAR_VAL(distanceToChangeOrder, unsigned);

  assert(keyType == FLOAT && valueType == UNSIGNED_INT);

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

    std::uint64_t thisWorkerCycles = 0;

    // cycles for constant overhead pre/post numEntries loop
    thisWorkerCycles += 23;

    // cycles per entry in the worklist
    thisWorkerCycles += numEntries * 4;

    bool firstEntry = true;
    for (unsigned entry = 0; entry < numEntries; ++entry) {
      const unsigned distance = *worklistIt++;
      const unsigned numElems = *worklistIt++;
      const auto innerElemCount =
          firstEntry ? firstInnerElemCount : std::min(distance, numElems);
      // Total number of elements
      const auto numInnerLoops = numElems;
      const auto numOuterLoops =
          1 + (ceildiv(numElems - innerElemCount, distance));
      const auto numChangesOfOrder =
          numElems >= changeOrderCounter
              ? (1 + (numElems - changeOrderCounter) / distanceToChangeOrder)
              : 0;

      // Cycles per element. Note we assume the worst case where every
      // pair of elements must be swapped but this is data dependent in
      // reality.
      thisWorkerCycles += 8 + (numInnerLoops - 1) * 11;
      // 1 floating point comparison per element
      flops += numInnerLoops;
      // additional cycles for each outer loop until numElems is exhausted
      thisWorkerCycles += numOuterLoops * 11;
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

#define BROADCAST_2TYPE_CYCLE_ESTIM_ENTRIES(vertexName)                        \
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName,                                    \
                        BinaryOpType::VARIANCE_TO_INV_STD_DEV, FLOAT, HALF),   \
      CYCLE_ESTIMATOR_ENTRY(popops, vertexName,                                \
                            BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF,       \
                            FLOAT)

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
  CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, FLOAT),                        \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, HALF),                     \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, INT),                      \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, UNSIGNED_INT),             \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, UNSIGNED_SHORT),           \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SRC_TYPE, BOOL)
#define CAST_CYCLE_ESTIM_ENTRIES(name)                                         \
  CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, FLOAT),                           \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, HALF),                        \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, INT),                         \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, UNSIGNED_INT),                \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, UNSIGNED_SHORT),              \
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, BOOL),                        \
      CYCLE_ESTIMATOR_ENTRY(popops, name, FLOAT, UNSIGNED_CHAR),               \
      CYCLE_ESTIMATOR_ENTRY(popops, name, FLOAT, SIGNED_CHAR),                 \
      CYCLE_ESTIMATOR_ENTRY(popops, name, FLOAT, CHAR),                        \
      CYCLE_ESTIMATOR_ENTRY(popops, name, HALF, UNSIGNED_CHAR),                \
      CYCLE_ESTIMATOR_ENTRY(popops, name, HALF, SIGNED_CHAR),                  \
      CYCLE_ESTIMATOR_ENTRY(popops, name, HALF, CHAR),                         \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_CHAR, FLOAT),               \
      CYCLE_ESTIMATOR_ENTRY(popops, name, UNSIGNED_CHAR, HALF),                \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SIGNED_CHAR, FLOAT),                 \
      CYCLE_ESTIMATOR_ENTRY(popops, name, SIGNED_CHAR, HALF),                  \
      CYCLE_ESTIMATOR_ENTRY(popops, name, CHAR, FLOAT),                        \
      CYCLE_ESTIMATOR_ENTRY(popops, name, CHAR, HALF)

poplibs::PerfEstimatorTable makePerfFunctionTable() {
  poplibs::PerfEstimatorTable table = {
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, FLOAT, FLOAT, FLOAT),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, HALF, HALF),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, FLOAT, HALF),
      SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, HALF, FLOAT),

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
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, FLOAT, FLOAT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, UNSIGNED_INT,
                            UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, INT, INT, false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT,
                            false),

      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, INT, false),

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

      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, HALF, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, FLOAT, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, FLOAT, false, false),

      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbYSupervisor, HALF, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, aXMinusbY2D, HALF, false, false),

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

      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInnerSupervisor),
      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInnerInPlaceSupervisor),
      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner2D),
      VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner2DInPlace),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2D,
                            BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DInPlace,
                            BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF),

      BROADCAST_2TYPE_CYCLE_ESTIM_ENTRIES(BroadcastScalar2Types2DData),
      BROADCAST_2TYPE_CYCLE_ESTIM_ENTRIES(BroadcastScalar2Types1DSupervisor),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByColumnSupervisor, true),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByColumnInPlaceSupervisor, true),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByRowSupervisor, true),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByRowInPlaceSupervisor, true),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByColumnSupervisor, false),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByColumnInPlaceSupervisor, false),

      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByRowSupervisor, false),
      BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(
          BroadcastVectorOuterByRowInPlaceSupervisor, false),

      CYCLE_ESTIMATOR_ENTRY(popops, HadamardProd, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HadamardProd, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, Fill, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Fill2d, INT),

      CAST_CYCLE_ESTIM_ENTRIES(Cast),
      CAST_CYCLE_ESTIM_ENTRIES(Cast2d),
      CAST_CYCLE_ESTIM_ENTRIES(CastSupervisor),

      CYCLE_ESTIMATOR_ENTRY(popops, CheckAccuracyWhenCast, FLOAT, HALF),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1d, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice1d, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1d, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice1d, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, INT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, UNSIGNED_INT, false),

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

      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, FLOAT, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HistogramSupervisor, HALF, false, false),

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

      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInfSupervisor, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInfSupervisor, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInf, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInfSupervisor, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaNOrInfSupervisor, HALF, true),

      CYCLE_ESTIMATOR_ENTRY(popops, Transpose2d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose2d, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose2d, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose2d, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose2d, UNSIGNED_SHORT),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose2d, SHORT),

      CYCLE_ESTIMATOR_ENTRY(popops, Transpose, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose, UNSIGNED_SHORT),
      CYCLE_ESTIMATOR_ENTRY(popops, Transpose, SHORT),

      CYCLE_ESTIMATOR_ENTRY(popops, TransposeSupervisor, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, TransposeSupervisor, UNSIGNED_SHORT),
      CYCLE_ESTIMATOR_ENTRY(popops, TransposeSupervisor, SHORT),

      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistance, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, CompareAndSwapAtDistanceKeyVal, FLOAT,
                            UNSIGNED_INT)};

  for (const auto &entry : unaryOpPerfInfo) {
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp2D, entry.first.first,
                                          entry.first.second));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(
        popops, UnaryOp1DSupervisor, entry.first.first, entry.first.second));
  }
  for (const auto &entry : unaryOpInPlacePerfInfo) {
    table.push_back(CYCLE_ESTIMATOR_ENTRY(
        popops, UnaryOp2DInPlace, entry.first.first, entry.first.second));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp1DInPlaceSupervisor,
                                          entry.first.first,
                                          entry.first.second));
  }

  for (const auto &entry : binaryOpPerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2D, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DSupervisor, op, type));
  }
  for (const auto &entry : binaryOpInPlacePerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2DInPlace, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DInPlaceSupervisor, op, type));
  }
  for (const auto &entry : broadcastOpPerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2D, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DData, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar1DSupervisor, op, type));
  }
  for (const auto &entry : broadcastOpInPlacePerfInfo) {
    BinaryOpType op = entry.first.first;
    Type type = entry.first.second;
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DInPlace, op, type));
    table.push_back(
        CYCLE_ESTIMATOR_ENTRY(popops, BroadcastScalar2DDataInPlace, op, type));
    table.push_back(CYCLE_ESTIMATOR_ENTRY(
        popops, BroadcastScalar1DInPlaceSupervisor, op, type));
  }
  return table;
}

} // end namespace popops
