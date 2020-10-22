// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popopsCycleEstimators.hpp"
#include "ExprOpUtil.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/popopsPerformanceEstimation.hpp"
#include "popops/Expr.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <cmath>
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
                                      const unsigned numElems) {
  unsigned elemsPerLoop;
  unsigned cyclesPerLoop = perfInfo.cyclesPerLoop;
  if (perfInfo.naturalVectorWidth) {
    elemsPerLoop = target.getVectorWidth(type);
    // This is an overhead for each cycle of the inner loop that processes
    // 1 element (or 1 vector). It accounts for load/store and loop decision
    cyclesPerLoop += hasExternalCodelet(op, type) ? 2 : 5;
  } else {
    elemsPerLoop = perfInfo.loopUnrollFactor;
  }
  unsigned numLoops = iceil(numElems, elemsPerLoop);
  return numLoops * cyclesPerLoop;
}

// Computes the cycles used by one of the scalar broadcast supervisor codelets
static std::uint64_t broadcastArithmeticSupervisorCycleEstimate(
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

  return cycles * numWorkers + basicOpSupervisorOverhead();
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar1DInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo &perfInfo = broadcastOpInPlacePerfInfo.at({op, type});
  return broadcastArithmeticSupervisorCycleEstimate(
      vertex, target, op, type, perfInfo, hasExternalCodelet(op, type) ? 1 : 4);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo &perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticSupervisorCycleEstimate(
      vertex, target, op, type, perfInfo, hasExternalCodelet(op, type) ? 1 : 4);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2Types1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const Type &outType) {
  // For vectorisation purposes, treat this as if it always processes float,
  // as it casts internally.  An extra cycle to cast to half output
  const OpPerformanceInfo &perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticSupervisorCycleEstimate(
      vertex, target, op, FLOAT, perfInfo, outType == FLOAT ? 0 : 1);
}

static std::uint64_t BroadcastVectorOuterCycleEstimate(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, std::uint64_t overheadPerInnerLoop,
    std::uint64_t overheadPerOuterLoop, bool byRow) {
  CODELET_SCALAR_VAL(columns, uint16_t);
  CODELET_SCALAR_VAL(rows, uint16_t);

  CODELET_FIELD(data);
  assert(type == HALF || type == FLOAT);
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
  return (15 + numOuterLoops * cycles) * numWorkers + supervisorCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByColumnInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, false);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByColumnSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, false);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByRowInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, true);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByRowSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7, true);
}

static std::uint64_t broadcastArithmeticCycleEstimate(
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

  for (unsigned i = 0; i < data.size(); i++) {
    auto numElems = data[i].size();

    unsigned numCycles = iceil(numElems, elemsPerLoop);
    cycles += numCycles * cyclesPerLoop;
    cycles += 28;
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2DDataInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpInPlacePerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          hasExternalCodelet(op, type) ? 1 : 4);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2DData)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          hasExternalCodelet(op, type) ? 1 : 4);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2Types2DData)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type, const Type &outType) {
  const OpPerformanceInfo perfInfo = broadcastOpPerfInfo.at({op, type});
  // For vectorisation purposes, treat this as if it always processes float
  // as casting makes this so. An extra cycle to cast the output to half
  return broadcastArithmeticCycleEstimate(vertex, target, op, FLOAT, perfInfo,
                                          outType == FLOAT ? 0 : 1);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpInPlacePerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          4);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const OpPerformanceInfo perfInfo = broadcastOpPerfInfo.at({op, type});
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, perfInfo,
                                          4);
}

enum class ScaledArithmeticOp { ADD, SUBTRACT, AXPLUSBY, AXMINUSBY };

std::uint64_t scaledArithmeticSupervisorCycleEstimate(
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
    assert(dataType == HALF || dataType == FLOAT);
  }

  // calculate count, rem and final
  const auto numWorkers = target.getNumWorkerContexts();
  const unsigned atomSize = 8 / target.getTypeSize(dataType);
  const unsigned count = (A.size() / numWorkers / atomSize) * atomSize;
  const unsigned final = A.size() % numWorkers;
  const unsigned rem =
      (A.size() / numWorkers) % numWorkers + iceil(final, atomSize);

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
                                   +(final == 0 ? 7 : 13) + 12;

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

  std::vector<unsigned> workerCycles(numWorkers);
  // Specific mixed precision half, float version
  if (dataType == HALF && dataBType == FLOAT) {
    for (unsigned wid = 0; wid <= numWorkers; ++wid) {
      std::uint64_t cycles = 16; // constant worker prologue cycles
      if (count / atomSize != 0) {
        if (count / atomSize < 3) {
          cycles += 8 // inner loop for < 3 constant overhead (processes 1)
                    + (4 * (count / atomSize - 1)); // loop cycles
        } else {
          cycles += 16 // inner loop for >= 3 constant overhead (processes 3)
                    + (2 * (count / atomSize - 3)); // loop cycles
        }
      }
      cycles += 2; // workerID == rem
      if (wid == rem) {
        cycles += 1; // final == 0?
        if (final != 0) {
          cycles += 5; // unpack triPtr and check if at least 2 remain
          if (final >= 2) {
            cycles += 7; // process 2 of the remainder.
            if (final == 3) {
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
    const unsigned innerLoopCycles =
        memConstrained ? 2
                       : (dataType == dataBType || dataBType == HALF ? 3 : 4);

    for (unsigned wid = 0; wid <= numWorkers; ++wid) {
      std::uint64_t cycles = 15; // constant worker prologue cycles
      if (count / atomSize != 0) {
        cycles += 6 // inner loop constant overhead
                  + (innerLoopCycles * (count / atomSize - 1)); // loop cycles
      }
      cycles += 2; // workerID == rem
      if (wid == rem) {
        cycles += 1; // final == 0?
        if (final != 0) {
          if (dataType == FLOAT) {
            cycles += 8; // process final float.
          } else {
            cycles += 5; // unpack triPtr and check if at least 2 remain
            if (final >= 2) {
              cycles += 7; // process 2 of the remainder.
              if (final == 3) {
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
  return supervisorCycles + maxWorkerCycles * 6;
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
std::uint64_t aXPlusbYMixedSupervisorCycleEstimate(
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
  return supervisorCycles + maxWorkerCycles * 6;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const Type &BType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, BType,
                                                 isConstant, memConstrained,
                                                 ScaledArithmeticOp::ADD);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ScaledSubtractSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const Type &BType, const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, BType,
                                                 false, memConstrained,
                                                 ScaledArithmeticOp::SUBTRACT);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(aXPlusbYSupervisor)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(aXMinusbYSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const bool isConstant, const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, AType,
                                                 isConstant, memConstrained,
                                                 ScaledArithmeticOp::AXMINUSBY);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(XMinusaXPlusbYSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const bool isConstant, const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType, AType,
                                                 isConstant, memConstrained,
                                                 ScaledArithmeticOp::AXPLUSBY);
}

std::uint64_t ScaledArithmetic2DCycleEstimate(
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

  const unsigned innerLoopCycles = memConstrained ? 2 : 3;
  const auto grain = type == HALF ? 4 : 2;
  std::uint64_t cycles = 9; // prologue and epilogue overhead.
  if (!isConstant)
    cycles += 1;
  if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant)
    cycles += 2;
  if (operation == ScaledArithmeticOp::AXPLUSBY && !isConstant)
    cycles += 6;
  if (operation == ScaledArithmeticOp::AXPLUSBY && isConstant)
    cycles += 4;

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
  }

  return cycles;
}

// aX Plus BY vertices where the data is half and the scale coeffs are float
std::uint64_t aXPlusbYMixed2DCycleEstimate(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const bool isConstant,
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
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ScaledAdd2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &AType,
    const Type &BType, const Type &ScaleType, const bool isConstant,
    const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, AType, memConstrained,
                                         isConstant, ScaledArithmeticOp::ADD);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ScaledSubtract2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, false,
                                         memConstrained,
                                         ScaledArithmeticOp::SUBTRACT);
}
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(aXPlusbY2D)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(aXMinusbY2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isConstant, const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, memConstrained,
                                         isConstant,
                                         ScaledArithmeticOp::AXMINUSBY);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(XMinusaXPlusbY2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool isConstant, const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, memConstrained,
                                         isConstant,
                                         ScaledArithmeticOp::AXPLUSBY);
}

// Exact worker cycle count for VectorInnerAdd_core_float
std::uint64_t vectorInnerAddCoreCycles_float(unsigned addendLen,
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
std::uint64_t vectorInnerAddCoreCycles_half_multiple_of_8(unsigned addendLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 2; // add, brneg
  if (blockCount == 1) {
    cycles += 3 + 7 * (addendLen / 8) + 1;
  } else {
    cycles += 4;                                        // after brneg, pre-loop
    cycles += (addendLen / 8) * (8 +                    // pre-rpt
                                 2 * (blockCount - 1) + // rpt body
                                 // post-rpt
                                 7) +
              1; // return
  }
  return cycles;
}
std::uint64_t vectorInnerAddCoreCycles_half_multiple_of_4(unsigned addendLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 5; // pre-loop
  cycles += (addendLen / 4) *
            (7 +                        // pre-rpt
             2 * (blockCount / 2 - 1) + // rpt body
             // post-rpt. The code depends on whether or not blockCount was odd
             1 + (blockCount % 2) + 5);
  return cycles + 1; // return
}

// Exact worker cycle count for VectorInnerMul_core_half
std::uint64_t vectorInnerAddCoreCycles_half(unsigned addendLen,
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
    return cycles +
           vectorInnerAddCoreCycles_half_multiple_of_8(addendLen, blockCount);
  }

  cycles += 2; // cmpult, brnz
  if (blockCount < 2) {
    return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 4 == 0) {
    return cycles +
           vectorInnerAddCoreCycles_half_multiple_of_4(addendLen, blockCount);
  }
  return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
}

// Cycle count for the common part of all the VectorInner2D ADD and
// SUBTRACT codelets (from the .Lworker2d label)
std::uint64_t
vectorInner2DAddCycles(uint32_t n, const std::vector<uint32_t> &BLen,
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

  std::uint64_t numCycles = 5; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    // loop overhead. A bit more for halves
    if (type == HALF)
      numCycles += 17;
    else
      numCycles += 11;

    auto coreFunc = type == HALF ? vectorInnerAddCoreCycles_half
                                 : vectorInnerAddCoreCycles_float;

    numCycles += coreFunc(BLen[i], dataBlockCount[i]);
  }

  return numCycles + 1; // exitnz
}

// Cycle count for the common part of all the VectorInnerSupervisor ADD and
// SUBTRACT codelets
std::uint64_t vectorInnerSupervisorAddCycles(unsigned numWorkerContexts,
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

  // Supervisor overhead: setzi and wait 6 cycles for register to be updated
  // before runall
  std::uint64_t numCycles = 1 + 6;

  // Worker cycles in common part (from the .Lworker label).
  numCycles += numWorkerContexts * (type == HALF ? 27 : 17);

  auto coreFunc = type == HALF ? vectorInnerAddCoreCycles_half
                               : vectorInnerAddCoreCycles_float;

  numCycles += numWorkerContexts * coreFunc(BLen, maxBlocksPerWorker);

  return numCycles + 1; // return; should we count extra cycles for sync?
}

// Exact worker cycle count for VectorInnerMul_core_float
std::uint64_t vectorInnerMulCoreCycles_float(unsigned scaleLen,
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

std::uint64_t vectorInnerMulCoreCycles_half_multiple_of_4(unsigned scaleLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 3;                                // pre-loop
  cycles += (scaleLen / 4) * (4 + 2 * blockCount + 3) + 1; // return
  return cycles;
}

std::uint64_t
vectorInnerMulCoreCycles_half_multiple_of_4_pipeline(unsigned scaleLen,
                                                     unsigned blockCount) {
  std::uint64_t cycles = 3; // pre-loop
  cycles += (scaleLen / 4) * ((blockCount == 1) ? 7 : 6 + blockCount + 3) +
            1; // return
  return cycles;
}

// Exact worker cycle count for VectorInnerMul_core_half
std::uint64_t vectorInnerMulCoreCycles_half(unsigned scaleLen,
                                            unsigned blockCount, bool inPlace) {

  std::uint64_t cycles = 1; // initial check for 0

  cycles += 2; // check for multiple of four
  if (scaleLen % 4 != 0) {
    return cycles + vectorInnerMulCoreCycles_half_scalar(scaleLen, blockCount);
  }

  cycles += 2; // check for in place
  if (inPlace) {
    return cycles +
           vectorInnerMulCoreCycles_half_multiple_of_4(scaleLen, blockCount);
  }

  cycles += 2; // check for > 2044
  if (scaleLen > 2044) {
    return cycles +
           vectorInnerMulCoreCycles_half_multiple_of_4(scaleLen, blockCount);
  }

  cycles += 2; // Check for > 1
  if (blockCount < 2) {
    return cycles +
           vectorInnerMulCoreCycles_half_multiple_of_4(scaleLen, blockCount);
  }

  return cycles + vectorInnerMulCoreCycles_half_scalar(scaleLen, blockCount);
}

// Cycle count for the common part of all the VectorInner2D MUL
// codelets (from the .Lworker2d label)
std::uint64_t
vectorInner2DMulCycles(uint32_t n, const std::vector<uint32_t> &BLen,
                       const std::vector<uint32_t> &dataBlockCount,
                       const Type &type) {

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

    numCycles += coreFunc(BLen[i], dataBlockCount[i], false);
  }

  // Exit
  numCycles += 1;

  return numCycles;
}

// Cycle count for the common part of all the VectorInnerSupervisor MUL
// codelets.
std::uint64_t vectorInnerSupervisorMulCycles(unsigned numWorkerContexts,
                                             uint32_t BLen,
                                             uint16_t dataBlockCountPacked,
                                             const Type &type) {
  // These numbers may not be exact (e.g. the remainder of
  // dataBlockCountPacked is ignored).

  // Supervisor overhead.
  std::uint64_t numCycles = 1 + 6;

  auto approxBlocksPerWorker = dataBlockCountPacked >> 3;

  // Worker cycles (from the .Lworker label)
  numCycles += numWorkerContexts * (type == HALF ? 24 : 17);

  auto coreFunc = type == HALF ? vectorInnerMulCoreCycles_half
                               : vectorInnerMulCoreCycles_float;

  numCycles += numWorkerContexts * coreFunc(BLen, approxBlocksPerWorker, true);

  // Exit
  numCycles += 1;

  return numCycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInnerSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  unsigned numWorkerContexts = target.getNumWorkerContexts();

  // Additional branch in the supervisor, and preamble instructions in the
  // worker part.
  switch (op) {
  case BinaryOpType::ADD:
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                          dataBlockCountPacked, type) +
           1 + 3;
  case BinaryOpType::SUBTRACT:
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                          dataBlockCountPacked, type) +
           1 + 4;
  case BinaryOpType::MULTIPLY:
    return vectorInnerSupervisorMulCycles(numWorkerContexts, BLen,
                                          dataBlockCountPacked, type) +
           2;
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInnerInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  unsigned numWorkerContexts = target.getNumWorkerContexts();

  switch (op) {
  case BinaryOpType::ADD:
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                          dataBlockCountPacked, type) +
           2;
  case BinaryOpType::SUBTRACT:
    // Additional branches in the supervisor and worker part.
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                          dataBlockCountPacked, type) +
           1 + 4;
  case BinaryOpType::MULTIPLY:
    return vectorInnerSupervisorMulCycles(numWorkerContexts, BLen,
                                          dataBlockCountPacked, type) +
           3;
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInner2D)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_VECTOR_VALS(BLen, uint32_t);
  CODELET_VECTOR_VALS(dataBlockCount, uint32_t);

  switch (op) {
  case BinaryOpType::SUBTRACT:
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 4;
  case BinaryOpType::ADD:
    // an additional branch at the start.
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 3;
  case BinaryOpType::MULTIPLY:
    return vectorInner2DMulCycles(n, BLen, dataBlockCount, type) + 2;
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInner2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_VECTOR_VALS(BLen, uint32_t);
  CODELET_VECTOR_VALS(dataBlockCount, uint32_t);

  switch (op) {
  case BinaryOpType::SUBTRACT:
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 4;
  case BinaryOpType::ADD:
    // an additional branch at the start.
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 2;
  case BinaryOpType::MULTIPLY:
    return vectorInner2DMulCycles(n, BLen, dataBlockCount, type) + 3;
  default:
    throw poputil::poplibs_error("BinaryOpType not implemented");
  }
  return 0;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(HadamardProd)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  uint64_t cycles = 5;
  const auto A = vertex.getFieldInfo("A");
  CODELET_FIELD(B);
  assert(A.size() == B.size());
  for (unsigned i = 0; i < A.size(); ++i) {
    assert(A[i].size() == B[i].size());
    unsigned numElem = A[i].size();
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;
    cycles += 5 + (1 + numVectors * 2);
  }
  return cycles;
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(Fill)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type) {
  return _fillCycleEstimate(vertex.getFieldInfo("out").size(), target, type);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Fill2d)(const VertexIntrospector &vertex,
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

// Cycles for a worker assembly Cast codelets (FLOAT->HALF or HALF->FLOAT)
uint64_t castWorkerFloatCycles(const unsigned numElems, const Type &toType) {
  auto extraCylesInPtrConversion = 1U;
  auto extraCyclesOutPtrConversion = 1U + ((toType == HALF) ? 2 : 0);
  auto columns = numElems;
  uint64_t cycles = extraCylesInPtrConversion + extraCyclesOutPtrConversion;
  if (columns < 4) {
    cycles += 11 + (columns * 14) / 3;
  } else {
    cycles += 26 + 2 * (columns / 4) + ((columns & 3) * 14) / 3;
  }
  return cycles;
}

// TODO: T12954 popops::Cast* cycle estimators do not depend on template type
// of the codelet. (a) This may change. (b) It will introduce an annoying
// special case at estimator registration time as we can't automatically
// lookup based on the template name. (c) INSTANTIATE_TEMPLATE_CYCLE_ESTIMATOR
// doesn't handle funcs with more than one template parameter.
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(Cast)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &fromType,
                                              const Type &toType) {
  CODELET_SCALAR_VAL(numElems, unsigned);

  std::uint64_t cycles;

  // Cast float to/from half written in assembly.
  // The equations below are a reasonable approximation for both
  // Estimates for other types not revised
  if ((fromType == FLOAT && toType == HALF) ||
      (fromType == HALF && toType == FLOAT)) {
    cycles = castWorkerFloatCycles(numElems, toType);
  } else {
    // These are not valid for integer and boolean casts
    const auto floatVectorWidth = target.getDataPathWidth() / 32;
    cycles = (numElems + floatVectorWidth - 1) / floatVectorWidth + 5;
  }

  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fromType, const Type &toType) {
  CODELET_SCALAR_VAL(partitionParams, unsigned);
  const unsigned workerElems = partitionParams >> 9;

  // This supervisor vertex will start up to 6 workers:. We compute the cycles
  // for the slowest ones (processing workerElems).
  // + 20 is the additional cycles when started from the supervisor.
  std::uint64_t maxCycles = 20 + castWorkerFloatCycles(workerElems, toType);

  // Add 7 for the supervisor code
  return 7 + target.getNumWorkerContexts() * maxCycles;

  return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Cast2d)(const VertexIntrospector &vertex,
                                  const Target &target, const Type &fromType,
                                  const Type &toType) {
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  std::uint64_t cycles = 5;
  const auto dst = vertex.getFieldInfo("dst");
  CODELET_FIELD(src);
  assert(src.size() == dst.size());
  for (unsigned i = 0; i != dst.size(); ++i) {
    assert(src[i].size() == dst[i].size());
    // Estimate based on 6 cycles of loop overhead per src / dst pointer pair:
    //
    // 1: load src
    // 2: load dst
    // 3: load length
    // 4: load src[0]
    // 5: { load src[1] ; convert src[0] }
    // 6: repeat
    // These are not valid for integer and boolean casts
    cycles += 6 + (dst[i].size() + floatVectorWidth - 1) / floatVectorWidth;
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CheckAccuracyWhenCast)(
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
  uint64_t workerCycles = 13;
  const auto &info = binaryOpInPlacePerfInfo.at({op, type});

  const auto numWorkers = target.getNumWorkerContexts();
  auto numElemsPerWorker = iceil(numElems, numWorkers);
  workerCycles +=
      binaryOpInnerLoopCycles(target, op, type, info, numElemsPerWorker);

  return numWorkers * workerCycles + superviserOverhead;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp2D)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t cycles = 20;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  assert(in.size() == out.size());
  const auto &info = unaryOpPerfInfo.at({op, type});
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    cycles += unaryOpInnerLoopCycles(target, type, info, in[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 20;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  const auto &info = unaryOpPerfInfo.at({op, type});
  assert(in.size() == out.size());
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (in.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // Unary op is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 9;
  return cycles + superviserOverhead;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp2DInPlace)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t cycles = 20;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  for (unsigned i = 0; i < inOut.size(); ++i) {
    cycles += unaryOpInnerLoopCycles(target, type, info, inOut[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp1DInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    popops::expr::UnaryOpType op, const Type &type) {
  uint64_t superviserOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 20;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (inOut.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // UnaryOpInPlace is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 9;
  return cycles + superviserOverhead;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp2D)(const VertexIntrospector &vertex,
                                      const Target &target, BinaryOpType op,
                                      const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());

  const auto &info = binaryOpPerfInfo.at({op, type});

  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    cycles += binaryOpInnerLoopCycles(target, op, type, info, in1[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp1DSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  uint64_t supervisorOverhead = basicOpSupervisorOverhead();
  uint64_t workerCycles = 22;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  const auto &info = binaryOpPerfInfo.at({op, type});

  const auto numWorkers = target.getNumWorkerContexts();
  unsigned numElems = iceil(in1.size(), numWorkers);
  workerCycles += binaryOpInnerLoopCycles(target, op, type, info, numElems);
  return numWorkers * workerCycles + supervisorOverhead;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  uint64_t cycles = 20;
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  const auto &info = binaryOpInPlacePerfInfo.at({op, type});

  for (unsigned i = 0; i < in1Out.size(); ++i) {
    assert(in1Out[i].size() == in2[i].size());
    cycles += binaryOpInnerLoopCycles(target, op, type, info, in1Out[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp1DInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, BinaryOpType op,
    const Type &type) {
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  return getBinaryOp1DInPlaceSupervisorEstimate(target, type, op,
                                                in1Out.size());
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Select)(const VertexIntrospector &vertex,
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
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastSelect)(
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
  return cycles;
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastSelectorSelect)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SelectInPlace)(
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
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastSelectorSelectInPlace)(
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
  return BroadcastSelectorSelectCycles(type, target.getTypeSize(type),
                                       rowSizes);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Histogram2D)(const VertexIntrospector &vertex,
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
  if (type == HALF) {
    cycles += 5;                  // Pre-loop overhead
    uint64_t dataLoopCycles = 16; // Data-loop overhead
    for (unsigned i = 0; i < data.size(); i++) {
      unsigned elements = data[i].size();
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
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(HistogramSupervisor)(
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

  if (splitByLimits) {
    return histogramSupervisorByLimitEstimate(
        data.size(), histogramCount, isAbsolute, type == HALF, numWorkers,
        vectorWidth, unpackCostHistogram, unpackCostLimits);
  } else {
    return histogramSupervisorByDataEstimate(
        data.size(), histogramCount, isAbsolute, type == HALF, numWorkers,
        vectorWidth, unpackCostHistogram, unpackCostLimits);
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(Clamp)(const VertexIntrospector &vertex,
                                               const Target &target,
                                               const Type &type) {
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
    cycles += clampCycles(target, type, in1[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ClampInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  uint64_t cycles = 5;
  CODELET_FIELD(in1Out);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == in1Out.size());
  assert(in3.size() == in1Out.size());
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    assert(in2[i].size() == in1Out[i].size());
    assert(in3[i].size() == in1Out[i].size());
    cycles += clampCycles(target, type, in1Out[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastClamp)(
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
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    cycles += clampCycles(target, type, in1[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BroadcastClampInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  // NOTE: Draft version to make UTs pass. Will be update with more accurate
  //       estimates from ASM implementation
  uint64_t cycles = 5;
  CODELET_FIELD(in1Out);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == 1);
  assert(in3.size() == 1);
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    cycles += clampCycles(target, type, in1Out[i].size());
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice2d)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice2d)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice1d)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice1d)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  return MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice1d)(vertex, target, type);
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MultiSlice)(const VertexIntrospector &vertex,
                                      const Target &target, const Type &type) {
  return multiSlicer(vertex, target, type, false);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MultiUpdate)(const VertexIntrospector &vertex,
                                       const Target &target, const Type &type) {
  return multiSlicer(vertex, target, type, true);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(MultiUpdateAdd)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
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
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CircBufIncrIndex)(const VertexIntrospector &vertex,
                                            const Target &target) {
  return 8;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CircOffset)(const VertexIntrospector &vertex,
                                      const Target &target) {
  return 10;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(EncodeOneHot)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(EncodeOneHotCustomValues)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &indexType, const Type &outputType) {
  CODELET_FIELD(indices);

  // C++ vertex
  std::uint64_t cycles = 100 * indices.size();

  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(Iota)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &outputType) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(HeapSortVertex)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &indexType) {
  std::uint64_t n = vertex.getFieldInfo("out").size();

  // Assuming all the worst cases are hit in the HeapSort codelet
  return 8 * (19 * n * std::floor(std::log2(n)) + 6 * n + 2);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(HeapSortVertexKV)(
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
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UpdateIntervalDEC)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             const Type &paramsType) {
  CODELET_SCALAR_VAL(rowCount, unsigned);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(rowCount, paramsType == HALF);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UpdateIntervalsDEC)(const VertexIntrospector &vertex,
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
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UpdateColumnsDEC)(const VertexIntrospector &vertex,
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
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SelectFromInterval)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &paramsType) {
  CODELET_SCALAR_VAL(rowCount, unsigned);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(rowCount, paramsType == HALF);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SelectFromIntervals)(const VertexIntrospector &vertex,
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
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(SelectFromRowsInColumns)(
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

// cycles derived from inspecting the compiler output. the cycle cost is data
// dependent and therefore this estimate assumes the worst case (ie. no NaN's)
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(HasNaN)(const VertexIntrospector &vertex,
                                  const Target &target, const Type &inType) {
  CODELET_FIELD(in);

  // initial overhead + exitz
  std::uint64_t cycles = 4;
  if (in.size() == 0) {
    return cycles;
  }

  // post-zero check overhead.
  cycles += 2;

  for (unsigned i = 0; i < in.size(); ++i) {
    // outer loop overhead pre-zero size check.
    cycles += 3;
    if (in[i].size() == 0) {
      continue;
    }

    // inner loop cost.
    cycles += (inType == FLOAT ? 9 : 10) * in[i].size();

    // outer loop post-overhead.
    cycles += 3;
  }

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Transpose2d)(const VertexIntrospector &vertex,
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
      cycles = 37 + matrices * (12 + (numSrcRows / 4) *
                                         (15 + 4 * (numSrcColumns / 4 - 2)));
    } else if (((numSrcRows & 3) == 0) && (numSrcColumns == 4) &&
               (numSrcRows / 4 < hardwareRptCountConstraint) &&
               (1 + 3 * (numSrcRows / 4) < 512)) { // Largest stride used
      // Half, fast path estimates, 4x4 or Nx4 cases
      if (numSrcRows == 4)
        cycles = 32 + 15 * matrices;
      else
        cycles = 28 + matrices * (17 + (20 + 4 * (numSrcRows / 4 - 2)));
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Transpose)(const VertexIntrospector &vertex,
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(TransposeSupervisor)(
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
      CAST_CYCLE_ESTIM_ENTRIES_BY_SRC_TYPE(name, BOOL)

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
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

      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(popops, CircBufIncrIndex),
      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(popops, CircOffset),

      CYCLE_ESTIMATOR_ENTRY(popops, Select, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, Select, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelect, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelect, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, INT),
      CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, BOOL),

      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, BroadcastSelectorSelectInPlace, INT),
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
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, FLOAT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, FLOAT, HALF),
      CYCLE_ESTIMATOR_ENTRY(popops, HeapSortVertexKV, HALF, INT),
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

      CYCLE_ESTIMATOR_ENTRY(popops, HasNaN, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popops, HasNaN, HALF),

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
  };

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
