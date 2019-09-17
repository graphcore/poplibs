#include "popopsCycleEstimators.hpp"
#include "popops/Expr.hpp"
#include "poputil/exceptions.hpp"
#include "ExprOpUtil.hpp"
#include <map>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

using namespace poplar;

namespace popops {

namespace {

// integer ceil
int iceil(int x, int y) {
  return x / y + (x % y > 0);
}

/* Cycle cost computation for basic operations */
uint64_t basicOpLoopCycles(unsigned numElems,
                           unsigned vectorSize,
                           unsigned cyclesPerVector) {
  return cyclesPerVector * (numElems + vectorSize-1) / vectorSize;
}

} // unnamed namespace

using BroadcastOpType = popops::expr::BroadcastOpType;

static bool hasExternalCodelet(expr::BroadcastOpType op, Type type) {
  return  (type == FLOAT ||
           type == HALF) &&
          (op == expr::BroadcastOpType::ADD ||
           op == expr::BroadcastOpType::SUBTRACT ||
           op == expr::BroadcastOpType::MULTIPLY);
  }

static bool hasExternalCodelet(expr::BinaryOpType op, Type type) {
  return (type == FLOAT ||
          type == HALF) &&
         (op == expr::BinaryOpType::ADD ||
          op == expr::BinaryOpType::SUBTRACT ||
          op == expr::BinaryOpType::MULTIPLY);
  }

struct OpPerformanceInfo {
  unsigned cyclesPerVector;
  bool vectorize;
  OpPerformanceInfo() = default;
  OpPerformanceInfo(unsigned cyclesPerVector,
                    bool vectorize) :
    cyclesPerVector(cyclesPerVector),
    vectorize(vectorize) {}
  OpPerformanceInfo(unsigned cyclesPerVector) :
    OpPerformanceInfo(cyclesPerVector, false) {}
};

static const std::map<std::pair<BroadcastOpType, poplar::Type>,
                                                            OpPerformanceInfo>
  broadcastOpPerfInfo = {
    { {BroadcastOpType::ADD, FLOAT}, {1, true} },
    { {BroadcastOpType::ADD, HALF}, {1, true} },
    { {BroadcastOpType::INV_STD_DEV_TO_VARIANCE, FLOAT}, {2, true} },
    { {BroadcastOpType::INV_STD_DEV_TO_VARIANCE, HALF}, {8, true} },
    { {BroadcastOpType::MULTIPLY, FLOAT}, {1, true} },
    { {BroadcastOpType::MULTIPLY, HALF}, {1, true} },
    { {BroadcastOpType::SUBTRACT, FLOAT}, {1, true} },
    { {BroadcastOpType::SUBTRACT, HALF}, {1, true} },
    { {BroadcastOpType::VARIANCE_TO_INV_STD_DEV, FLOAT}, {1, true} },
    { {BroadcastOpType::VARIANCE_TO_INV_STD_DEV, HALF}, {7, true} },

};

static std::uint64_t
broadcastArithmeticSupervisorCycleEstimate (const VertexIntrospector &vertex,
                                            const Target &target,
                                            BroadcastOpType op,
                                            const Type &type,
                                            std::uint64_t overheadPerLoop) {
  CODELET_FIELD(data);
  assert(type == HALF || type == FLOAT);
  auto vectorWidth = target.getVectorWidth(type);
  auto numWorkers = target.getNumWorkerContexts();
  auto perfInfo = broadcastOpPerfInfo.at({op, type});

  std::uint64_t cycles = 20;
  std::uint64_t supervisorCycles = 19;
  const auto cyclesPerLoop = perfInfo.cyclesPerVector + overheadPerLoop;
  auto numElems = (data.size() + numWorkers - 1) / numWorkers;
  if(perfInfo.vectorize)
    cycles += cyclesPerLoop * (numElems + vectorWidth - 1) / vectorWidth;
  else
    cycles += cyclesPerLoop * numElems;

  return cycles * numWorkers + supervisorCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar1DInPlaceSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type) {
  return broadcastArithmeticSupervisorCycleEstimate(vertex, target, op, type,
         hasExternalCodelet(op, type) ? 1 : 4);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar1DSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type) {
  return broadcastArithmeticSupervisorCycleEstimate(vertex, target, op, type,
         hasExternalCodelet(op, type) ? 1 : 4);
}

static std::uint64_t
BroadcastVectorOuterCycleEstimate (const VertexIntrospector &vertex,
                                   const Target &target,
                                   BroadcastOpType op,
                                   const Type &type,
                                   std::uint64_t overheadPerInnerLoop,
                                   std::uint64_t overheadPerOuterLoop,
                                   bool byRow) {
  CODELET_SCALAR_VAL(columns, uint16_t);
  CODELET_SCALAR_VAL(rows, uint16_t);

  CODELET_FIELD(data);
  assert(type == HALF || type == FLOAT);
  auto vectorWidth = target.getVectorWidth(type);
  auto numWorkers = target.getNumWorkerContexts();
  auto perfInfo = broadcastOpPerfInfo.at({op, type});

  std::uint64_t cycles = overheadPerOuterLoop;

  std::uint64_t supervisorCycles = 19;
  const auto cyclesPerLoop = perfInfo.cyclesPerVector + overheadPerInnerLoop;
  auto numElems = byRow ? columns :
                          (columns + numWorkers - 1) / numWorkers;
  if(perfInfo.vectorize) {
    cycles += cyclesPerLoop * (numElems + vectorWidth - 1) / vectorWidth;
  } else {
    cycles += cyclesPerLoop * numElems;
  }
  auto numOuterLoops = byRow ? (rows + numWorkers - 1 ) / numWorkers :
                               rows;
  return (15 + numOuterLoops * cycles) * numWorkers + supervisorCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByColumnInPlaceSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type,
                                    bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7,
                                           false);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByColumnSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type,
                                    bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7,
                                           false);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByRowInPlaceSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type,
                                    bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7,
                                           true);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorOuterByRowSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type,
                                    bool allowMisaligned) {
  // Improved loop overheads, as these are written in assembly
  return BroadcastVectorOuterCycleEstimate(vertex, target, op, type, 1,
                                           allowMisaligned ? 25 : 7,
                                           true);
}


static std::uint64_t
broadcastArithmeticCycleEstimate (const VertexIntrospector &vertex,
                                  const Target &target,
                                  BroadcastOpType op,
                                  const Type &type,
                                  std::uint64_t overheadPerLoop) {
  CODELET_FIELD(data);
  assert(type ==HALF || type == FLOAT);
  auto vectorWidth = target.getVectorWidth(type);
  auto perfInfo = broadcastOpPerfInfo.at({op, type});
  const auto cyclesPerLoop = perfInfo.cyclesPerVector + overheadPerLoop;

  std::uint64_t cycles = 20;

  for(unsigned i = 0; i < data.size(); i++){
    auto numElems = data[i].size();
    if(perfInfo.vectorize)
      cycles += (cyclesPerLoop) *
                (numElems + vectorWidth - 1) / vectorWidth;
    else
      cycles += (cyclesPerLoop) * numElems;
    cycles += 28;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2DDataInPlace)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     BroadcastOpType op,
                                     const Type &type) {
  return broadcastArithmeticCycleEstimate(vertex, target, op, type,
                                          hasExternalCodelet(op, type) ? 1 : 4);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2DData)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     BroadcastOpType op,
                                     const Type &type) {
  return broadcastArithmeticCycleEstimate(vertex, target, op, type,
                                          hasExternalCodelet(op, type) ? 1 : 4);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2DInPlace)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     BroadcastOpType op,
                                     const Type &type) {
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, 4);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastScalar2D)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     BroadcastOpType op,
                                     const Type &type) {
  return broadcastArithmeticCycleEstimate(vertex, target, op, type, 4);
}

enum class ScaledArithmeticOp {
  ADD,
  SUBTRACT,
  AXPLUSBY
};

std::uint64_t
scaledArithmeticSupervisorCycleEstimate(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &dataType,
                                     const Type &dataBType,
                                     const bool isConstant,
                                     const bool memConstrained,
                                     const ScaledArithmeticOp operation) {
  CODELET_FIELD(A);

  if (dataType == INT || dataType == UNSIGNED_INT) {
    std::uint64_t supervisorCycles = 53 // constant overhead
      + (26 * (A.size()/3)); // main loop

    if(operation == ScaledArithmeticOp::SUBTRACT && !isConstant) {
      supervisorCycles += 1;
    }

    if (A.size() % 3 == 0) {
      supervisorCycles += 6; // 6 cycle branch to skip the remainder loop
    } else {
      supervisorCycles += 6 // --rem
        + (26 * (A.size()%3)); // remainder loop
    }
    supervisorCycles += 8; // constant epilogue overhead.
    if(!isConstant) {
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
  const unsigned rem = (A.size() / numWorkers) % numWorkers
    + iceil(final, atomSize);

  std::uint64_t supervisorCycles = 12 // per-type supervisor overhead
    + 47 // common supervisor overhead
    + (final == 0 ? 7 : 13)
    + 12;

  if(operation == ScaledArithmeticOp::AXPLUSBY && !isConstant) {
      supervisorCycles += 14;
  }
  if(operation == ScaledArithmeticOp::SUBTRACT && !isConstant) {
      supervisorCycles += 7;
  }
  if(!isConstant) {
    supervisorCycles += 1;
  }

  std::vector<unsigned> workerCycles(numWorkers);
  // Specific mixed precision half, float version
  if (dataType == HALF && dataBType == FLOAT) {
    for (unsigned wid = 0; wid <= numWorkers; ++wid) {
      std::uint64_t cycles = 16; // constant worker prologue cycles
      if (count/atomSize != 0) {
        if (count/atomSize < 3) {
          cycles += 8 // inner loop for < 3 constant overhead (processes 1)
          + (4 * (count/atomSize-1)); // loop cycles
        }
        else {
          cycles += 16 // inner loop for >= 3 constant overhead (processes 3)
          + (2 * (count/atomSize-3)); // loop cycles
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
  // half,half and float, float versions
  else {
    const unsigned innerLoopCycles = memConstrained ? 2 :
                                     (dataType == dataBType ? 3 : 4);

    for (unsigned wid = 0; wid <= numWorkers; ++wid) {
      std::uint64_t cycles = 15; // constant worker prologue cycles
      if (count/atomSize != 0) {
        cycles += 6 // inner loop constant overhead
          + (innerLoopCycles * (count/atomSize-1)); // loop cycles
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
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddSupervisor)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &AType,
                                     const Type &BType,
                                     const bool isConstant,
                                     const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType,
                                                 BType, isConstant,
                                                 memConstrained,
                                                 ScaledArithmeticOp::ADD);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledSubtractSupervisor)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &AType,
                                     const Type &BType,
                                     const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType,
                                                 BType, false, memConstrained,
                                                 ScaledArithmeticOp::SUBTRACT);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(aXPlusbYSupervisor)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &AType,
                                     const bool isConstant,
                                     const bool memConstrained) {
  return scaledArithmeticSupervisorCycleEstimate(vertex, target, AType,
                                                 AType, isConstant,
                                                 memConstrained,
                                                 ScaledArithmeticOp::AXPLUSBY);
}

std::uint64_t
ScaledArithmetic2DCycleEstimate(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type,
                                       const bool isConstant,
                                       const bool memConstrained,
                                       const ScaledArithmeticOp operation) {
  CODELET_FIELD(A);

  if (type == INT || type == UNSIGNED_INT) {
    std::uint64_t cycles = 8; // prologue and epilogue overhead.
    for (unsigned i = 0; i < A.size(); ++i) {
      cycles += 7 // outer loop constant overhead
        + (A[i].size() * 5); // inner loop
    }
    if( !isConstant)
      cycles += 1;
    if(operation == ScaledArithmeticOp::SUBTRACT && !isConstant)
      cycles += 1;
    return cycles;
  } else {
    assert(type == HALF || type == FLOAT);
  }

  const unsigned innerLoopCycles = memConstrained ? 2 : 3;
  const auto grain = type == HALF ? 4 : 2;
  std::uint64_t cycles = 9;// prologue and epilogue overhead.
  if( !isConstant)
    cycles += 1;
  if(operation == ScaledArithmeticOp::SUBTRACT && !isConstant)
    cycles += 2;
  if(operation == ScaledArithmeticOp::AXPLUSBY && !isConstant)
    cycles += 6;
  if(operation == ScaledArithmeticOp::AXPLUSBY && isConstant)
    cycles += 4;

  for (unsigned i = 0; i < A.size(); ++i) {
    cycles += 11 // outer loop constant overhead
      + (A[i].size()/grain != 0 ? 5 : 0) // inner loop overhead
      + (A[i].size()/grain * innerLoopCycles); // inner loop

    if (type == FLOAT) {
      cycles += (A[i].size()%grain != 0 ? 7 : 0); // last element.
    } else {
      auto rem = A[i].size() % grain;
      cycles += (rem > 0 ? 4 : 0) // remainder overhead
        + (rem >= 2 ? 6 : 0) // process 2 more at end.
        + (rem % 2 == 1 ? 7 : 0); // process last element.
    }
  }

  return cycles;
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAdd2D)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type,
                                       const bool isConstant,
                                       const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, memConstrained,
                                         isConstant, ScaledArithmeticOp::ADD);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledSubtract2D)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type,
                                       const bool memConstrained) {
  return ScaledArithmetic2DCycleEstimate(vertex, target, type, false,
                                         memConstrained,
                                         ScaledArithmeticOp::SUBTRACT);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(aXPlusbY2D)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type,
                                       const bool isConstant,
                                       const bool memConstrained) {
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
      cycles += 3; // start of loop
      cycles += 2 * blockCount; // rpt loop
      cycles += 5; // end of loop
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
  std::uint64_t cycles = 2;  // add, brneg
  if (blockCount==1) {
    cycles += 3 + 7*(addendLen/8) + 1;
  } else {
    cycles += 4; // after brneg, pre-loop
    cycles += (addendLen/8) * (
      8 + // pre-rpt
      2 * (blockCount - 1) + // rpt body
      // post-rpt
      7
    ) + 1; // return
  }
  return cycles;
}
std::uint64_t vectorInnerAddCoreCycles_half_multiple_of_4(unsigned addendLen,
                                                          unsigned blockCount) {
  std::uint64_t cycles = 5; // pre-loop
  cycles += (addendLen/4) * (
    7 + // pre-rpt
    2 * (blockCount/2 - 1) + // rpt body
    // post-rpt. The code depends on whether or not blockCount was odd
    1 + (blockCount%2) + 5
  );
  return cycles + 1;  //return
}

// Exact worker cycle count for VectorInnerMul_core_half
std::uint64_t vectorInnerAddCoreCycles_half(unsigned addendLen,
                                            unsigned blockCount) {
  std::uint64_t cycles = 1;   // brz
  if (blockCount==0)
    return cycles;

  cycles += 2; // cmpult > 2048, brz
  if (addendLen > 2048) {
    return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 8 == 0) {
    return cycles + vectorInnerAddCoreCycles_half_multiple_of_8(addendLen,
                                                                blockCount);
  }

  cycles += 2; // cmpult, brnz
  if (blockCount < 2) {
    return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 4 == 0) {
    return cycles + vectorInnerAddCoreCycles_half_multiple_of_4(addendLen,
                                                                blockCount);
  }
  return cycles + vectorInnerAddCoreCycles_half_scalar(addendLen, blockCount);
}

// Cycle count for the common part of all the VectorInner2D ADD and
// SCALED_ADD codelets (from the .Lworker2d label)
std::uint64_t vectorInner2DAddCycles(
  uint32_t n,
  const std::vector<uint32_t> &BLen,
  const std::vector<uint32_t> &dataBlockCount,
  const Type &type) {

  if (BLen.size() != n || dataBlockCount.size() != n) {
    throw poputil::poplibs_error("n (" + std::to_string(n) + ") does not "
                                 "match BLen or dataBlockCount "
                                 "length (" + std::to_string(BLen.size())
                                 + " & " + std::to_string(dataBlockCount.size())
                                 + " respectively) in Broadcast ADD vertex");
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

  return numCycles + 1;  // exitnz
}

// Cycle count for the common part of all the VectorInnerSupervisor ADD and
// SCALED_ADD codelets
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

  auto maxBlocksPerWorker = quotient + (remainder!=0);


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
    cycles += 3; // start of loop
    cycles += 2 * blockCount; // rpt loop
    cycles += 5; // end of loop
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
  std::uint64_t cycles = 3; // pre-loop
  cycles += (scaleLen/4) * (
    4 + 2*blockCount + 3
  ) + 1; //return
  return cycles;
}

std::uint64_t vectorInnerMulCoreCycles_half_multiple_of_4_pipeline(
                                                      unsigned scaleLen,
                                                      unsigned blockCount) {
  std::uint64_t cycles = 3; // pre-loop
  cycles += (scaleLen/4) * (
    (blockCount==1) ? 7 : 6 + blockCount + 3
  ) + 1; //return
  return cycles;
}

// Exact worker cycle count for VectorInnerMul_core_half
std::uint64_t vectorInnerMulCoreCycles_half(unsigned scaleLen,
                                            unsigned blockCount,
                                            bool inPlace) {

  std::uint64_t cycles = 1;  // initial check for 0


  cycles += 2; // check for multiple of four
  if (scaleLen % 4 != 0) {
    return cycles + vectorInnerMulCoreCycles_half_scalar(scaleLen,
                                                         blockCount);
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
std::uint64_t vectorInner2DMulCycles(
  uint32_t n,
  const std::vector<uint32_t> &BLen,
  const std::vector<uint32_t> &dataBlockCount,
  const Type &type) {

  if (BLen.size() != n || dataBlockCount.size() != n) {
    throw poputil::poplibs_error("n (" + std::to_string(n) + ") does not "
                                 "match BLen or dataBlockCount "
                                 "length (" + std::to_string(BLen.size())
                                 + " & " + std::to_string(dataBlockCount.size())
                                 + " respectively) in Broadcast MUL vertex");
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInnerSupervisor)(
                                            const VertexIntrospector &vertex,
                                            const Target &target,
                                            BroadcastOpType op,
                                            const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  unsigned numWorkerContexts = target.getNumWorkerContexts();

  // Additional branch in the supervisor, and preamble instructions in the
  // worker part.
  switch (op) {
  case BroadcastOpType::ADD:
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) + 1 + 3;
  case BroadcastOpType::SCALED_ADD:
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) + 1 + 4;
  case BroadcastOpType::MULTIPLY:
    return vectorInnerSupervisorMulCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) + 2;
  default:
    throw poputil::poplibs_error("BroadcastOpType not implemented");
  }
  return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInnerInPlaceSupervisor)(
                                            const VertexIntrospector &vertex,
                                            const Target &target,
                                            BroadcastOpType op,
                                            const Type &type) {
  CODELET_FIELD(B);
  CODELET_SCALAR_VAL(dataBlockCountPacked, uint16_t);

  uint32_t BLen = B.size();
  unsigned numWorkerContexts = target.getNumWorkerContexts();

  switch (op) {
  case BroadcastOpType::ADD:
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) + 2;
  case BroadcastOpType::SCALED_ADD:
    // Additional branches in the supervisor and worker part.
    return vectorInnerSupervisorAddCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) + 1 + 4;
  case BroadcastOpType::MULTIPLY:
    return vectorInnerSupervisorMulCycles(numWorkerContexts, BLen,
                                            dataBlockCountPacked, type) + 3;
  default:
    throw poputil::poplibs_error("BroadcastOpType not implemented");
  }
  return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInner2D)(
                                            const VertexIntrospector &vertex,
                                            const Target &target,
                                            BroadcastOpType op,
                                            const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_VECTOR_VALS(BLen, uint32_t);
  CODELET_VECTOR_VALS(dataBlockCount, uint32_t);


  switch (op) {
  case BroadcastOpType::SCALED_ADD:
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 4;
  case BroadcastOpType::ADD:
    // an additional branch at the start.
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 3;
  case BroadcastOpType::MULTIPLY:
    return vectorInner2DMulCycles(n, BLen, dataBlockCount, type) + 2;
  default:
    throw poputil::poplibs_error("BroadcastOpType not implemented");
  }
  return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastVectorInner2DInPlace)(
                                            const VertexIntrospector &vertex,
                                            const Target &target,
                                            BroadcastOpType op,
                                            const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_VECTOR_VALS(BLen, uint32_t);
  CODELET_VECTOR_VALS(dataBlockCount, uint32_t);

  switch (op) {
  case BroadcastOpType::SCALED_ADD:
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 4;
  case BroadcastOpType::ADD:
    // an additional branch at the start.
    return vectorInner2DAddCycles(n, BLen, dataBlockCount, type) + 2;
  case BroadcastOpType::MULTIPLY:
    return vectorInner2DMulCycles(n, BLen, dataBlockCount, type) + 3;
  default:
    throw poputil::poplibs_error("BroadcastOpType not implemented");
  }
 return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(HadamardProd)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Zero)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  const auto out = vertex.getFieldInfo("out");
  bool isHalf = type == HALF;
  auto width = target.getDataPathWidth() /  (isHalf ? 16 : 32);

  return 20 + out.size()/width;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Zero2d)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  const auto out = vertex.getFieldInfo("out");
  bool isHalf = type == HALF;
  auto width = target.getDataPathWidth() /  (isHalf ? 16 : 32);

  std::uint64_t cycles = 0;
  for (unsigned i=0; i < out.size(); ++i) {
    cycles += 20 + out[i].size()/width;
  }
  return cycles;
}

// TODO: popops::Cast* cycle estimators do not depend on template type
// of the codelet. (a) This may change. (b) It will introduce an annoying
// special case at estimator registration time as we can't automatically
// lookup based on the template name. (c) INSTANTIATE_TEMPLATE_CYCLE_ESTIMATOR
// doesn't handle funcs with more than one template parameter.
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Cast)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &fromType,
                                const Type &toType) {
  CODELET_SCALAR_VAL(numElems, unsigned);

  std::uint64_t cycles;

  // Cast float to/from half written in assembly.
  // The equations below are a reasonable approximation for both
  // Estimates for other types not revised
  if( (fromType == FLOAT && toType == HALF) ||
      (fromType == HALF && toType == FLOAT) ) {
    auto extraCylesInPtrConversion = 1U;
    auto extraCyclesOutPtrConversion = 1U + ((toType == HALF) ? 2 : 0);
    auto columns = numElems;
    cycles = extraCylesInPtrConversion + extraCyclesOutPtrConversion;
    if (columns < 4) {
      cycles += 11 + (columns * 14 )/3;
    }
    else {
      cycles += 26 + 2 * (columns/4) + ((columns & 3)*14)/3;
    }
  }
  else {
    // These are not valid for integer and boolean casts
    const auto floatVectorWidth = target.getDataPathWidth() / 32;
    cycles = (numElems + floatVectorWidth - 1) / floatVectorWidth + 5;
  }

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Cast2d)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &fromType,
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

// Operations have been benchmarked in a variety of ways, some notes:
//
// Simple operations which are implemented directly with an instruction are
// of course very quick.  Those with a float or half type will produce a
// bundled pair of instructions, hence are faster than int types.  In these
// cases the cycle time can be found by viewing the assembly output.
//
// logarithm, sqrt, divide have float instructions (not int),
// but they are not single cycle.
//
// Others such as sin, cos, logarithm_one_plus, power, atan2
// are not directly implemented with an instruction.
// They run a more complex compiled library function.  In these
// cases the simulator was used to make an estimate of the execution time.
//
// Operations which produce a bool output use the _st8 function to store the
// result, this adds to the cycle count considerably.

using UnaryOpType = popops::expr::UnaryOpType;

static const std::map<std::pair<UnaryOpType, poplar::Type>, OpPerformanceInfo>
unaryOpPerfInfo = {
  { {UnaryOpType::ABSOLUTE, FLOAT}, {1, false} },
  { {UnaryOpType::ABSOLUTE, HALF}, {1, false} },
  { {UnaryOpType::ABSOLUTE, INT}, {2, false} },
  // NOT on AUX side, ldst64pace
  { {UnaryOpType::BITWISE_NOT, INT}, {1, true} },
  { {UnaryOpType::BITWISE_NOT, UNSIGNED_INT}, {1, true} },
  // use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::CEIL, FLOAT}, {2, true} },
  { {UnaryOpType::CEIL, HALF}, {2, true} },
  { {UnaryOpType::COS, FLOAT}, {3600, false} },
  { {UnaryOpType::COS, HALF}, {3600, false} },
  { {UnaryOpType::INVERSE, HALF}, {15, true} },
  { {UnaryOpType::INVERSE, FLOAT}, {5, true} },
  { {UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false} },
  { {UnaryOpType::COUNT_LEADING_ZEROS, UNSIGNED_INT}, {1, false} },
  { {UnaryOpType::EXPONENT, FLOAT}, {2, true} },
  // Use f16v2exp
  { {UnaryOpType::EXPONENT, HALF}, {2, true} },
  { {UnaryOpType::EXPONENT_MINUS_ONE, FLOAT}, {4, false} },
  { {UnaryOpType::EXPONENT_MINUS_ONE, HALF}, {5, true} },

  // Use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::FLOOR, FLOAT}, {2, true} },
  { {UnaryOpType::FLOOR, HALF}, {2, true} },
  // 1 for v==v
  // 1 for v!=INFINITY
  // 1 for anding the two together
  // 1 for converting a match from 0xffff to 0x0001
  // 1 to convert the 32/16bit individual results to 8bits each
  { {UnaryOpType::IS_FINITE, FLOAT}, {5, true} },
  { {UnaryOpType::IS_FINITE, HALF}, {5, true} },
  // 1 for v!=INFINITY
  // 1 for converting a match from 0xffff to 0x0001
  // 1 to convert the 32/16bit individual results to 8bits each
  { {UnaryOpType::IS_INF, FLOAT}, {3, true} },
  { {UnaryOpType::IS_INF, HALF}, {5, true} },
  // 1 for v==v
  // 1 for converting a match from 0xffff to 0x0001
  // 1 to convert the 32/16bit individual results to 8bits each
  { {UnaryOpType::IS_NAN, FLOAT}, {3, true} },
  { {UnaryOpType::IS_NAN, HALF}, {3, true} },
  { {UnaryOpType::LOGARITHM, FLOAT}, {60, true} },
  { {UnaryOpType::LOGARITHM, HALF}, {15, true} },
  { {UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {180, true} },
  { {UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {180, true} },
  { {UnaryOpType::LOGICAL_NOT, BOOL}, {17, false} },
  { {UnaryOpType::NEGATE, FLOAT}, {1, true} },
  { {UnaryOpType::NEGATE, HALF}, {1, true} },
  { {UnaryOpType::NEGATE, INT}, {2, false} },
  { {UnaryOpType::POPCOUNT, INT}, {1, false} },
  { {UnaryOpType::POPCOUNT, UNSIGNED_INT}, {1, false} },
  { {UnaryOpType::ROUND, FLOAT}, {2, true} },
  { {UnaryOpType::ROUND, HALF}, {2, true} },
  { {UnaryOpType::SIGNUM, FLOAT}, {5, true} },
  { {UnaryOpType::SIGNUM, HALF}, {5, true} },
  { {UnaryOpType::SIGNUM, INT}, {5} },
  { {UnaryOpType::SIN, FLOAT}, {3600, false} },
  { {UnaryOpType::SIN, HALF}, {3600, false} },
  { {UnaryOpType::SQRT, FLOAT}, {23, false} },
  { {UnaryOpType::SQRT, HALF}, {23, false} },
  { {UnaryOpType::SQRT, INT}, {110, false} },
  { {UnaryOpType::SQUARE, FLOAT}, {1, true} },
  { {UnaryOpType::SQUARE, HALF}, {1, true} },
  { {UnaryOpType::SQUARE, INT}, {1, true} },
  { {UnaryOpType::SQUARE, UNSIGNED_INT}, {1, true} },
  { {UnaryOpType::TANH, FLOAT}, {1, true} },
  { {UnaryOpType::TANH, HALF}, {2, true} },   // only vectorised v2, not v4
  { {UnaryOpType::SIGMOID, FLOAT}, {1, false} },
  { {UnaryOpType::SIGMOID, HALF}, {2, true} },
  { {UnaryOpType::RSQRT, FLOAT}, {1, false} },
  { {UnaryOpType::RSQRT, HALF}, {3, true} },
};

static const std::map<std::pair<UnaryOpType, poplar::Type>, OpPerformanceInfo>
unaryOpInPlacePerfInfo = {
  { {UnaryOpType::ABSOLUTE, FLOAT}, {1, true} },
  { {UnaryOpType::ABSOLUTE, HALF}, {1, true} },
  { {UnaryOpType::ABSOLUTE, INT}, {2} },
  // NOT on AUX side, ldst64pace
  { {UnaryOpType::BITWISE_NOT, INT}, {1, true} },
  { {UnaryOpType::BITWISE_NOT, UNSIGNED_INT}, {1, true} },
  // use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::CEIL, FLOAT}, {2, true} },
  { {UnaryOpType::CEIL, HALF}, {2, true} },
  { {UnaryOpType::COS, FLOAT}, {3600, false} },
  { {UnaryOpType::COS, HALF}, {3600, false} },
  { {UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false} },
  { {UnaryOpType::COUNT_LEADING_ZEROS, UNSIGNED_INT}, {1, false} },
  { {UnaryOpType::INVERSE, HALF}, {15, true} },
  { {UnaryOpType::INVERSE, FLOAT}, {5, true} },
  { {UnaryOpType::EXPONENT, FLOAT}, {2, true} },
  // Use f16v2exp
  { {UnaryOpType::EXPONENT, HALF}, {2, true} },
  { {UnaryOpType::EXPONENT_MINUS_ONE, FLOAT}, {4, false} },
  { {UnaryOpType::EXPONENT_MINUS_ONE, HALF}, {5, true} },

  // Use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::FLOOR, FLOAT}, {2, true} },
  { {UnaryOpType::FLOOR, HALF}, {2, true} },
  { {UnaryOpType::LOGARITHM, FLOAT}, {60, true} },
  { {UnaryOpType::LOGARITHM, HALF}, {15, true} },
  { {UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {180, true} },
  { {UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {180, true} },
  { {UnaryOpType::LOGICAL_NOT, BOOL}, {17, true} },
  { {UnaryOpType::NEGATE, FLOAT}, {1, true} },
  { {UnaryOpType::NEGATE, HALF}, {1, true} },
  { {UnaryOpType::NEGATE, INT}, {2, false} },
  { {UnaryOpType::POPCOUNT, INT}, {1, false} },
  { {UnaryOpType::POPCOUNT, UNSIGNED_INT}, {1, false} },
  { {UnaryOpType::ROUND, FLOAT}, {2, true} },
  { {UnaryOpType::ROUND, HALF}, {2, true} },
  { {UnaryOpType::SIGNUM, FLOAT}, {5, true} },
  { {UnaryOpType::SIGNUM, HALF}, {5, true} },
  { {UnaryOpType::SIGNUM, INT}, {5} },
  { {UnaryOpType::SIN, FLOAT}, {3600, false} },
  { {UnaryOpType::SIN, HALF}, {3600, false} },
  { {UnaryOpType::SQRT, FLOAT}, {23, false} },
  { {UnaryOpType::SQRT, HALF}, {23, false} },
  { {UnaryOpType::SQRT, INT}, {110, false} },
  { {UnaryOpType::SQUARE, FLOAT}, {1, true} },
  { {UnaryOpType::SQUARE, HALF}, {1, true} },
  { {UnaryOpType::SQUARE, INT}, {1, true} },
  { {UnaryOpType::SQUARE, UNSIGNED_INT}, {1, true} },
  { {UnaryOpType::TANH, FLOAT}, {1, false} },
  { {UnaryOpType::TANH, HALF}, {2, true} },
  { {UnaryOpType::SIGMOID, FLOAT}, {1, false} },
  { {UnaryOpType::SIGMOID, HALF}, {2, true} },
  { {UnaryOpType::RSQRT, FLOAT}, {1, false} },
  { {UnaryOpType::RSQRT, HALF}, {3, true} },
};


using BinaryOpType = popops::expr::BinaryOpType;

static const std::map<std::pair<BinaryOpType, poplar::Type>, OpPerformanceInfo>
binaryOpPerfInfo = {
  { {BinaryOpType::ADD, FLOAT}, {1, true} },
  { {BinaryOpType::ADD, HALF}, {1, true} },
  { {BinaryOpType::ADD, INT}, {2, false} },
  { {BinaryOpType::ADD, UNSIGNED_INT}, {2, false} },
  { {BinaryOpType::ATAN2, FLOAT}, {120, false} },
  { {BinaryOpType::ATAN2, HALF}, {120, false} },

  { {BinaryOpType::BITWISE_AND, INT}, {3, false} },
  { {BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {3, false} },
  { {BinaryOpType::BITWISE_OR, INT}, {3, false} },
  { {BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {3, false} },
  { {BinaryOpType::BITWISE_XOR, INT}, {3, false} },
  { {BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {3, false} },
  { {BinaryOpType::BITWISE_XNOR, INT}, {3, false} },
  { {BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {3, false} },

  { {BinaryOpType::DIVIDE, FLOAT}, {10, false} },
  { {BinaryOpType::DIVIDE, HALF}, {10, false} },
  // ld into aux, ld into aux, div, st
  { {BinaryOpType::DIVIDE, INT}, {40, false} },
  { {BinaryOpType::DIVIDE, UNSIGNED_INT}, {40, false} },
  { {BinaryOpType::LOGICAL_AND, BOOL}, {20, false} },
  { {BinaryOpType::LOGICAL_OR, BOOL}, {20, false} },
  { {BinaryOpType::MAXIMUM, FLOAT}, {1, true} },
  { {BinaryOpType::MAXIMUM, HALF}, {1, true} },
  { {BinaryOpType::MAXIMUM, INT}, {2} },
  { {BinaryOpType::MAXIMUM, UNSIGNED_INT}, {2} },
  { {BinaryOpType::MINIMUM, FLOAT}, {1, true} },
  { {BinaryOpType::MINIMUM, HALF}, {1, true} },
  { {BinaryOpType::MINIMUM, INT}, {2} },
  { {BinaryOpType::MINIMUM, UNSIGNED_INT}, {2} },
  { {BinaryOpType::MULTIPLY, FLOAT}, {1, true} },
  { {BinaryOpType::MULTIPLY, HALF}, {1, true} },
  { {BinaryOpType::MULTIPLY, INT}, {2, false} },
  { {BinaryOpType::MULTIPLY, UNSIGNED_INT}, {2, false} },

  // Accuracy concerns using ln
  // pow(a,b) = exp(b * log(a))
  // Doesn't handle negative values yet

  // Power instruction not used
  { {BinaryOpType::POWER, FLOAT}, {200, false} },
  { {BinaryOpType::POWER, HALF}, {200, false} },

  { {BinaryOpType::REMAINDER, FLOAT}, {10, false} },
  { {BinaryOpType::REMAINDER, HALF}, {10, false} },
  { {BinaryOpType::REMAINDER, INT}, {40, false} },
  { {BinaryOpType::REMAINDER, UNSIGNED_INT}, {40, false} },
  { {BinaryOpType::SHIFT_LEFT, INT}, {3} },
  { {BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT, INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {4} },
  { {BinaryOpType::SUBTRACT, FLOAT}, {1, true} },
  { {BinaryOpType::SUBTRACT, HALF}, {1, true} },
  { {BinaryOpType::SUBTRACT, INT}, {2, false} },
  { {BinaryOpType::SUBTRACT, UNSIGNED_INT}, {2, false} },
};


static const std::map<std::pair<BinaryOpType, poplar::Type>, OpPerformanceInfo>
binaryOpInPlacePerfInfo = {
  { {BinaryOpType::ADD, FLOAT}, {1, true} },
  { {BinaryOpType::ADD, HALF}, {1, true} },
  { {BinaryOpType::ADD, INT}, {2, false} },
  { {BinaryOpType::ADD, UNSIGNED_INT}, {2, false} },
  { {BinaryOpType::ATAN2, FLOAT}, {120, false} },
  { {BinaryOpType::ATAN2, HALF}, {120, false} },

  { {BinaryOpType::BITWISE_AND, INT}, {3, false} },
  { {BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {3, false} },
  { {BinaryOpType::BITWISE_OR, INT}, {3, false} },
  { {BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {3, false} },
  { {BinaryOpType::BITWISE_XOR, INT}, {3, false} },
  { {BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {3, false} },
  { {BinaryOpType::BITWISE_XNOR, INT}, {3, false} },
  { {BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {3, false} },


  { {BinaryOpType::DIVIDE, FLOAT}, {10, false} },
  { {BinaryOpType::DIVIDE, HALF}, {10, false} },
  // ld into aux, ld into aux, div, st
  { {BinaryOpType::DIVIDE, INT}, {40, false} },
  { {BinaryOpType::DIVIDE, UNSIGNED_INT}, {40, false} },
  { {BinaryOpType::LOGICAL_AND, BOOL}, {20, false} },
  { {BinaryOpType::LOGICAL_OR, BOOL}, {20, false} },
  { {BinaryOpType::MAXIMUM, FLOAT}, {1, true} },
  { {BinaryOpType::MAXIMUM, HALF}, {1, true} },
  { {BinaryOpType::MAXIMUM, INT}, {2} },
  { {BinaryOpType::MAXIMUM, UNSIGNED_INT}, {2} },
  { {BinaryOpType::MINIMUM, FLOAT}, {1, true} },
  { {BinaryOpType::MINIMUM, HALF}, {1, true} },
  { {BinaryOpType::MINIMUM, INT}, {2} },
  { {BinaryOpType::MINIMUM, UNSIGNED_INT}, {2} },
  { {BinaryOpType::MULTIPLY, FLOAT}, {1, true} },
  { {BinaryOpType::MULTIPLY, HALF}, {1, true} },
  { {BinaryOpType::MULTIPLY, INT}, {2, false} },
  { {BinaryOpType::MULTIPLY, UNSIGNED_INT}, {2, false} },

  // Accuracy concerns using ln
  // pow(a,b) = exp(b * log(a))
  // Doesn't handle negative values yet

  // Power instruction not used
  { {BinaryOpType::POWER, FLOAT}, {200, false} },
  { {BinaryOpType::POWER, HALF}, {200, false} },

  { {BinaryOpType::REMAINDER, FLOAT}, {10, false} },
  { {BinaryOpType::REMAINDER, HALF}, {10, false} },
  { {BinaryOpType::REMAINDER, INT}, {40, false} },
  { {BinaryOpType::REMAINDER, UNSIGNED_INT}, {40, false} },
  { {BinaryOpType::SHIFT_LEFT, INT}, {3} },
  { {BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT, INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {4} },
  { {BinaryOpType::SUBTRACT, FLOAT}, {1, true} },
  { {BinaryOpType::SUBTRACT, HALF}, {1, true} },
  { {BinaryOpType::SUBTRACT, INT}, {2, false} },
  { {BinaryOpType::SUBTRACT, UNSIGNED_INT}, {2, false} },
};


static const std::map<std::pair<BinaryOpType, poplar::Type>, unsigned>
    comparisonOpPerfInfo = {
  // Dominated by separate _st8 byte function calls
  // even if the actual arithmetic operation is vectorised
  { {BinaryOpType::EQUAL, FLOAT}, 17 },
  { {BinaryOpType::EQUAL, HALF}, 17 },
  { {BinaryOpType::EQUAL, INT}, 17 },
  { {BinaryOpType::EQUAL, UNSIGNED_INT}, 17 },
  { {BinaryOpType::EQUAL, BOOL}, 17 },
  // same as B < A
  // E = A and B, result = A andc E
  { {BinaryOpType::GREATER_THAN, FLOAT}, 17 },
  { {BinaryOpType::GREATER_THAN, HALF}, 17 },
  { {BinaryOpType::GREATER_THAN, INT}, 17 },
  { {BinaryOpType::GREATER_THAN, UNSIGNED_INT}, 17 },
  { {BinaryOpType::GREATER_THAN, BOOL}, 17 },
  { {BinaryOpType::GREATER_THAN_EQUAL, FLOAT}, 17 },
  { {BinaryOpType::GREATER_THAN_EQUAL, HALF}, 17 },
  { {BinaryOpType::GREATER_THAN_EQUAL, INT}, 17 },
  { {BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT}, 17 },
  { {BinaryOpType::GREATER_THAN_EQUAL, BOOL}, 17 },
  { {BinaryOpType::LESS_THAN, FLOAT}, 17 },
  { {BinaryOpType::LESS_THAN, HALF}, 17},
  { {BinaryOpType::LESS_THAN, INT}, 17 },
  { {BinaryOpType::LESS_THAN, UNSIGNED_INT}, 17 },
  { {BinaryOpType::LESS_THAN, BOOL}, 17 },
  { {BinaryOpType::LESS_THAN_EQUAL, FLOAT}, 17 },
  { {BinaryOpType::LESS_THAN_EQUAL, HALF}, 17 },
  { {BinaryOpType::LESS_THAN_EQUAL, INT}, 17 },
  { {BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_INT}, 17 },
  { {BinaryOpType::LESS_THAN_EQUAL, BOOL}, 17 },
  { {BinaryOpType::NOT_EQUAL, FLOAT}, 17 },
  { {BinaryOpType::NOT_EQUAL, HALF}, 17 },
  { {BinaryOpType::NOT_EQUAL, INT}, 17 },
  { {BinaryOpType::NOT_EQUAL, UNSIGNED_INT}, 17 },
  { {BinaryOpType::NOT_EQUAL, BOOL}, 17},
};

static const std::map<std::pair<BinaryOpType, poplar::Type>, unsigned>
    comparisonOpInplacePerfInfo = {
  // E = A and B, F = A or B, G = F andc E, result = 1 andc G
  { {BinaryOpType::EQUAL, BOOL}, 17 },
  // same as B < A
  // E = A and B, result = A andc E
  { {BinaryOpType::GREATER_THAN, BOOL}, 17 },
  { {BinaryOpType::GREATER_THAN_EQUAL, BOOL}, 17},
  { {BinaryOpType::LESS_THAN, BOOL}, 17 },
  { {BinaryOpType::LESS_THAN_EQUAL, BOOL}, 17 },
  { {BinaryOpType::NOT_EQUAL, BOOL}, 17 },
};

static std::uint64_t
unaryOpInnerLoopCycles(const Target &target, const Type &type,
              const OpPerformanceInfo &perfInfo, unsigned numElems) {
  unsigned vectorWidth = 1;
  if (perfInfo.vectorize) {
    vectorWidth = target.getVectorWidth(type);
  }
  // Estimate loop cycles, including a constant loop overhead added to the
  // cycles per vector.  This accounts for load/store and loop decision.
  return basicOpLoopCycles(numElems, vectorWidth, perfInfo.cyclesPerVector + 4);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp2D)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     popops::expr::UnaryOpType op,
                                     const Type &type) {
  uint64_t cycles = 20;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  assert(in.size() == out.size());
  const auto &info = unaryOpPerfInfo.at({op, type});
  for (unsigned i = 0; i < in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    cycles += unaryOpInnerLoopCycles(target, type, info, in[i].size());
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp1DSupervisor)(
                                   const VertexIntrospector &vertex,
                                   const Target &target,
                                   popops::expr::UnaryOpType op,
                                   const Type &type) {
  uint64_t workerCycles = 20;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  const auto &info = unaryOpPerfInfo.at({op, type});
  assert (in.size() == out.size());
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (in.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // Unary op is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 9;
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp2DInPlace)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            popops::expr::UnaryOpType op,
                                            const Type &type) {
  uint64_t cycles = 20;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  for (unsigned i = 0; i < inOut.size(); ++i) {
    cycles += unaryOpInnerLoopCycles(target, type, info, inOut[i].size());
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp1DInPlaceSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    popops::expr::UnaryOpType op,
                                    const Type &type) {
  uint64_t workerCycles = 20;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (inOut.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // UnaryOpInPlace is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 9;
  return cycles;
}

static std::uint64_t
binaryOpInnerLoopCycles(const Target &target, const Type &type,
                        bool isComparison, unsigned numBoolOpCycles,
                        const OpPerformanceInfo &perfInfo, unsigned numElems,
                        const std::uint64_t overheadPerLoop) {
  std::uint64_t cycles = 0;

  unsigned vectorWidth = 1;
  if (perfInfo.vectorize) {
    vectorWidth = target.getVectorWidth(type);
  }
  // Estimate loop cycles, including a constant loop overhead added to the
  // cycles per vector.  This accounts for load/store and loop decision.
  cycles += basicOpLoopCycles(numElems, vectorWidth,
                              perfInfo.cyclesPerVector + overheadPerLoop);

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp2D)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      BinaryOpType op,
                                      const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  auto c = comparisonOpPerfInfo.find({op, type});
  const bool isComparison = c != comparisonOpPerfInfo.end();
  const auto &info =
      isComparison ? OpPerformanceInfo() :
                     binaryOpPerfInfo.at({op, type});
  unsigned numBoolOpCycles = isComparison ? c->second : 0;

  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    cycles += binaryOpInnerLoopCycles(target, type, isComparison,
                                      numBoolOpCycles, info, in1[i].size(),
                                      hasExternalCodelet(op, type) ? 2 : 5);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp1DSupervisor)(
                                             const VertexIntrospector &vertex,
                                             const Target &target,
                                             BinaryOpType op,
                                             const Type &type) {
  uint64_t workerCycles = 22;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  auto c = comparisonOpPerfInfo.find({op, type});
  const bool isComparison = c != comparisonOpPerfInfo.end();
  const auto &info =
      isComparison ? OpPerformanceInfo() :
                     binaryOpPerfInfo.at({op, type});
  unsigned numBoolOpCycles = isComparison ? c->second : 0;
  const auto numWorkers = target.getNumWorkerContexts();
  unsigned numElems = (in1.size() + numWorkers - 1) / numWorkers;
  workerCycles += binaryOpInnerLoopCycles(target, type, isComparison,
                                          numBoolOpCycles, info, numElems,
                                          hasExternalCodelet(op, type) ? 2 : 5);
  return numWorkers * workerCycles + 9;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp2DInPlace)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             BinaryOpType op,
                                             const Type &type) {
  uint64_t cycles = 20;
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  auto c = comparisonOpPerfInfo.find({op, type});
  const bool isComparison = c != comparisonOpPerfInfo.end();
  const auto &info =
      isComparison ? OpPerformanceInfo() :
                     binaryOpInPlacePerfInfo.at({op, type});
  unsigned numBoolOpCycles = isComparison ? c->second : 0;

  for (unsigned i = 0; i < in1Out.size(); ++i) {
    assert(in1Out[i].size() == in2[i].size());
    cycles += binaryOpInnerLoopCycles(target, type, isComparison,
                                      numBoolOpCycles, info, in1Out[i].size(),
                                      hasExternalCodelet(op, type) ? 2 : 5);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp1DInPlaceSupervisor)(
                                           const VertexIntrospector &vertex,
                                           const Target &target,
                                           BinaryOpType op,
                                           const Type &type) {
  uint64_t workerCycles = 13;
  const auto in1Out = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  assert(in1Out.size() == in2.size());
  auto c = comparisonOpPerfInfo.find({op, type});
  const bool isComparison = c != comparisonOpPerfInfo.end();
  const auto &info =
      isComparison ? OpPerformanceInfo() :
                     binaryOpInPlacePerfInfo.at({op, type});
  unsigned numBoolOpCycles = isComparison ? c->second : 0;
  const auto numWorkers = target.getNumWorkerContexts();
  unsigned numElems = (in1Out.size() + numWorkers - 1) / numWorkers;
  workerCycles += binaryOpInnerLoopCycles(target, type, isComparison,
                                          numBoolOpCycles, info, numElems,
                                          hasExternalCodelet(op, type) ? 2 : 5);
  return numWorkers * workerCycles + 9;
}


static std::uint64_t
selectCycles(const Target &target, const Type &type, unsigned numElems) {
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
    cycles += selectCycles(target, type, in1[i].size());
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastSelect)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
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
      case 4:  // INT, FLOAT
        cycles += 5 + 4*n + 3;
        break;
      case 2:  // HALF
        if (n&1) {
          cycles += 23 + n*4;
        } else {
          cycles += 30 + n*4; // Worst case: pointer misaligned
        }
        break;
      case 1: // BOOL
        cycles += 40 + (n/4)*17 + 26;  // Worst case
        break;
      default:
        throw poputil::poplibs_error("Cycle estimator for BroadcastSelect: "
                                     "invalid type:"+type.toString());
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
                                            std::vector<unsigned> rowSizes)
{
  uint64_t cycles = 11+1;
  for (unsigned n : rowSizes) {
    unsigned bytes = n*typeLen;
    // When using ld64/st64 it takes 1 cycles for 8 bytes: 1 cycle/4 bytes
    cycles += 12 + 23 + (bytes/4) + (bytes%4)*5;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastSelectorSelect)(
                                  const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
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
  return BroadcastSelectorSelectCycles(type,
                                       target.getTypeSize(type), rowSizes);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SelectInPlace)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastSelectorSelectInPlace)(
                                  const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  CODELET_FIELD(in1Out);
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == in1Out.size());
  assert(in3.size() == 1);
  std::vector<unsigned> rowSizes;
  for (unsigned i = 0; i < in1Out.size(); ++i) {
    rowSizes.push_back(in1Out[i].size());
  }
  return BroadcastSelectorSelectCycles(type,
                                       target.getTypeSize(type), rowSizes);
}

static std::uint64_t
clampCycles(const Target &target,
            const Type &type,
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Clamp)(const VertexIntrospector &vertex,
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ClampInPlace)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastClamp)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastClampInPlace)(
                                         const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &type) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice2d)(const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &type) {
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
    if(type == HALF)
        cycles += (31 + 2 * nVectors) * numSubElements + 13;
    else
        cycles += (29 + nVectors) * numSubElements + 13;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice2d)(
                                              const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type) {
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
    if(type == HALF)
        cycles += (31 + 2 * nVectors) * numSubElements + 13;
    else
        cycles += (29 + nVectors) * numSubElements + 13;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicSliceSupervisor)(
                                           const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &type) {
  const auto regionSize =
    vertex.getFieldInfo("regionSize").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
#ifndef NDEBUG
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
#endif
  const unsigned numWorkers = target.getNumWorkerContexts();
  const auto baseT = vertex.getFieldInfo("baseT");
  const auto subT = vertex.getFieldInfo("subT");

  assert(subT.size() == numSubElements * regionSize);
  assert(baseT.size() == numBaseElements * regionSize);
  const unsigned elementsPerWorker = (regionSize + numWorkers -1 )/numWorkers;
  unsigned vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  // Supervisor overhead.
  auto superCycles = 1 + 6 + 1 + 6;
  // This is the more optimistic path - where the inner loop is copying
  // aligned data
  unsigned nCopies = elementsPerWorker / vectorWidth;
  auto workerCycles = 41 + (27 + nCopies) * numSubElements;
  auto cycles = superCycles + workerCycles * numWorkers;

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSliceSupervisor)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
 return MAKE_CYCLE_ESTIMATOR_NAME(DynamicSliceSupervisor)(vertex, target, type);
}

static std::uint64_t
multiSlicer(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type,
    bool /*isUpdate*/) {
  const auto regionSize =
    vertex.getFieldInfo("regionSize").getInitialValue<unsigned>(target);
  const auto offsets =
    vertex.getFieldInfo("offsets");

  auto numOffsets = offsets.size();
  assert(numOffsets > 0);
  unsigned vectorWidth = target.getDataPathWidth() / ((type == HALF) ? 16 : 32);
  // estimates for C vertex
  // assume no auto-vectorise
  vectorWidth = 1;
  unsigned subwordCost = type == HALF ||
                         type == SHORT ||
                         type == UNSIGNED_SHORT ? 12 : 0;
  auto copiesPerOffset = (regionSize + vectorWidth - 1) / vectorWidth;

  std::uint64_t callOverhead = 5 + 15;
  // load offset, compare, cond-branch, mpy to get idx, (load, store) per entry,
  // outer loop
  std::uint8_t coreCycles = numOffsets *
                            (5 + 2 * copiesPerOffset + subwordCost);
  return callOverhead + coreCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MultiSlice)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
  return multiSlicer(vertex, target, type, false);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MultiUpdate)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
  return multiSlicer(vertex, target, type, false);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MultiUpdateAdd)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
  return multiSlicer(vertex, target, type, false);
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(EncodeOneHot)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &indexType,
                                        const Type &outputType) {
  CODELET_FIELD(indices);
  CODELET_SCALAR_VAL(outLength, unsigned);

  std::uint64_t cycles = 100; // constant supervisor overhead

  // internally the EncodeOneHot vertex uses the MemsetZeroSupervisor vertex,
  // unfortunately that cycle estimate isn't available from inside poplibs so
  // this is a very rough estimate derived from the formula
  // in MemsetSupervisorTemplate.S
  const auto numWorkers = target.getNumWorkerContexts();
  const auto wordsPerworker =
    (outLength * target.getTypeSize(outputType)) / 8 / numWorkers;
  cycles += 18 + wordsPerworker;

  // the encode loop can take the following cycles for each index:
  //  - 22 if index[i] < offset[i],
  //  - 24 if index[i] > out.size(),
  //  - 64 if out[idx + indices[i] - offsets[i]] & 0x3 == 0,
  //  - 58 if out[idx + indices[i] - offsets[i]] & 0x3 == 1,
  // additional 12 cycles for comparing ignore indices
  // as we can't tell which branch the code will take, assume the worst case
  // every iteration.
  cycles += (64 + 12) * indices.size();

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Iota)(const VertexIntrospector &vertex,
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
MAKE_CYCLE_ESTIMATOR_NAME(EncodeOneHotCustomValues)(
                                        const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &indexType,
                                        const Type &outputType) {
  CODELET_FIELD(indices);
  CODELET_SCALAR_VAL(outLength, unsigned);

  std::uint64_t cycles = 100; // constant supervisor overhead

  cycles += 12; // For the additional loads.

  // internally the EncodeOneHot vertex uses the MemsetZeroSupervisor vertex,
  // unfortunately that cycle estimate isn't available from inside poplibs so
  // this is a very rough estimate derived from the formula
  // in MemsetSupervisorTemplate.S
  const auto numWorkers = target.getNumWorkerContexts();
  const auto wordsPerworker =
    (outLength * target.getTypeSize(outputType)) / 8 / numWorkers;
  cycles += 18 + wordsPerworker;

  // the encode loop can take the following cycles for each index:
  //  - 22 if index[i] < offset[i],
  //  - 24 if index[i] > out.size(),
  //  - 64 if out[idx + indices[i] - offsets[i]] & 0x3 == 0,
  //  - 58 if out[idx + indices[i] - offsets[i]] & 0x3 == 1,
  // as we can't tell which branch the code will take, assume the worst case
  // every iteration.
  cycles += 64 * indices.size();

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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(HeapSortVertexKV)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &keyType,
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
  return cycles + decrementOrGetParamsCycles(rowCount, paramsType ==HALF);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UpdateIntervalsDEC)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &paramsType) {
  CODELET_FIELD(params);
  CODELET_VECTOR_VALS(rowCounts, unsigned);
  const auto rowCountsSum = std::accumulate(rowCounts.begin(), rowCounts.end(),
                                            0);
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
  const auto regionHeightsSum = std::accumulate(regionHeights.begin(),
                                                regionHeights.end(), 0);
  const auto regionWidthsSum = std::accumulate(regionWidths.begin(),
                                               regionWidths.end(), 0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(
                  params.size() * regionWidthsSum * regionHeightsSum,
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
  const auto rowCountsSum = std::accumulate(rowCounts.begin(), rowCounts.end(),
                                            0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(params.size() * rowCountsSum,
                                            paramsType == HALF);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SelectFromRowsInColumns)(
                          const VertexIntrospector &vertex,
                          const Target &target,
                          const Type &paramsType) {
  CODELET_FIELD(params);
  CODELET_VECTOR_VALS(regionWidths, unsigned);
  CODELET_VECTOR_VALS(regionHeights, unsigned);
  const auto regionHeightsSum = std::accumulate(regionHeights.begin(),
                                                regionHeights.end(), 0);
  const auto regionWidthsSum = std::accumulate(regionWidths.begin(),
                                               regionWidths.end(), 0);
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // General load/process vertex state
  cycles += 20;
  return cycles + decrementOrGetParamsCycles(
                  params.size() * regionWidthsSum * regionHeightsSum,
                  paramsType == HALF);
}

// cycles derived from inspecting the compiler output. the cycle cost is data
// dependent and therefore this estimate assumes the worst case (ie. no NaN's)
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(HasNaN)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &inType) {
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
}

// Entries for broadcast vertices covering only the 3 basic operations
#define BROADCAST_BASIC_CYCLE_ESTIM_ENTRIES(vertexName) \
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::ADD, FLOAT),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::ADD, HALF),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::SUBTRACT, FLOAT),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::SUBTRACT, HALF),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::MULTIPLY, FLOAT),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::MULTIPLY, HALF)

// Entries for broadcast vertices covering also additional operations
#define BROADCAST_CYCLE_ESTIM_ENTRIES(vertexName) \
  BROADCAST_BASIC_CYCLE_ESTIM_ENTRIES(vertexName),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName,\
                            BroadcastOpType::VARIANCE_TO_INV_STD_DEV, FLOAT),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName,\
                            BroadcastOpType::VARIANCE_TO_INV_STD_DEV, HALF),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName,\
                            BroadcastOpType::INV_STD_DEV_TO_VARIANCE, FLOAT),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName,\
                            BroadcastOpType::INV_STD_DEV_TO_VARIANCE, HALF)

// Entries for broadcast outer vertices covering only the 3 basic operations,
// each with an alwaysAligned template parameter
#define BROADCAST_VECTOR_OUTER_CYCLE_ESTIM_ENTRIES(vertexName, allowMisaligned)\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::ADD, FLOAT,\
                       allowMisaligned),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::ADD, HALF,\
                       allowMisaligned),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::SUBTRACT, FLOAT,\
                       allowMisaligned),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::SUBTRACT, HALF,\
                       allowMisaligned),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::MULTIPLY, FLOAT,\
                       allowMisaligned),\
  CYCLE_ESTIMATOR_ENTRY(popops, vertexName, BroadcastOpType::MULTIPLY, HALF,\
                       allowMisaligned)

// Entries for VectorInnner vertices
#define VECTOR_INNER_CYCLE_ESTIM_ENTRIES(name) \
    CYCLE_ESTIMATOR_ENTRY(popops, name, BroadcastOpType::ADD, FLOAT),\
    CYCLE_ESTIMATOR_ENTRY(popops, name, BroadcastOpType::ADD, HALF),\
    CYCLE_ESTIMATOR_ENTRY(popops, name, BroadcastOpType::SCALED_ADD, FLOAT),\
    CYCLE_ESTIMATOR_ENTRY(popops, name, BroadcastOpType::SCALED_ADD, HALF),\
    CYCLE_ESTIMATOR_ENTRY(popops, name, BroadcastOpType::MULTIPLY, FLOAT),\
    CYCLE_ESTIMATOR_ENTRY(popops, name, BroadcastOpType::MULTIPLY, HALF)

#define SCALED_ADD_CYCLE_ESTIM_ENTRIES(NAME, TYPE1, TYPE2) \
  CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, true, true),\
  CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, true, false),\
  CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, false, true),\
  CYCLE_ESTIMATOR_ENTRY(popops, NAME, TYPE1, TYPE2, false, false)


poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
    SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, FLOAT, FLOAT),
    SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, HALF),
    SCALED_ADD_CYCLE_ESTIM_ENTRIES(ScaledAddSupervisor, HALF, FLOAT),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT,
                          UNSIGNED_INT, true, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT, INT, true, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT,
                          UNSIGNED_INT, false, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT, INT, false, false),


    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, true, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, true, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, false, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, false, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, true, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, true, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, false, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, false, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT, true, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT, true, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT, false, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT, false, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, FLOAT, FLOAT,false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, UNSIGNED_INT,
                          UNSIGNED_INT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, INT, INT, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, UNSIGNED_INT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, INT, false),

    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, true, true),
    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, false, true),
    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, true, false),
    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbYSupervisor, HALF, false, false),

    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, true, true),
    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, true, false),
    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, false, true),
    CYCLE_ESTIMATOR_ENTRY(popops, aXPlusbY2D, HALF, false, false),

    VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInnerSupervisor),
    VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInnerInPlaceSupervisor),
    VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner2D),
    VECTOR_INNER_CYCLE_ESTIM_ENTRIES(BroadcastVectorInner2DInPlace),

    BROADCAST_CYCLE_ESTIM_ENTRIES(BroadcastScalar2DData),
    BROADCAST_CYCLE_ESTIM_ENTRIES(BroadcastScalar2DDataInPlace),

    BROADCAST_CYCLE_ESTIM_ENTRIES(BroadcastScalar2D),
    BROADCAST_CYCLE_ESTIM_ENTRIES(BroadcastScalar2DInPlace),

    BROADCAST_CYCLE_ESTIM_ENTRIES(BroadcastScalar1DSupervisor),
    BROADCAST_CYCLE_ESTIM_ENTRIES(BroadcastScalar1DInPlaceSupervisor),

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

    CYCLE_ESTIMATOR_ENTRY(popops, Zero, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Zero2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero2d, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, UNSIGNED_INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, UNSIGNED_INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, UNSIGNED_INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, UNSIGNED_INT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, UNSIGNED_INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, UNSIGNED_INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, UNSIGNED_INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, UNSIGNED_INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, UNSIGNED_INT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, UNSIGNED_INT, BOOL),

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

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSliceSupervisor, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSliceSupervisor, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSliceSupervisor, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSliceSupervisor, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSliceSupervisor, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSliceSupervisor, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSliceSupervisor, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSliceSupervisor, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSliceSupervisor, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSliceSupervisor, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiSlice, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdate, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, MultiUpdateAdd, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, CircBufIncrIndex),
    CYCLE_ESTIMATOR_ENTRY(popops, CircOffset),

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
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHotCustomValues, INT, UNSIGNED_INT),
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
  };

  for (const auto &entry : unaryOpPerfInfo) {
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp2D, entry.first.first,
                            entry.first.second)
    );
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp1DSupervisor, entry.first.first,
                            entry.first.second)
    );

  }
  for (const auto &entry : unaryOpInPlacePerfInfo) {
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp2DInPlace, entry.first.first,
                            entry.first.second)
    );
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, UnaryOp1DInPlaceSupervisor,
                            entry.first.first, entry.first.second)
    );
  }

  for (const auto &entry : binaryOpPerfInfo) {
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2D, entry.first.first,
                            entry.first.second)
    );
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DSupervisor, entry.first.first,
                            entry.first.second)
    );
  }
  for (const auto &entry : binaryOpInPlacePerfInfo) {
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2DInPlace, entry.first.first,
                            entry.first.second)
    );
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DInPlaceSupervisor,
                            entry.first.first, entry.first.second)
    );
  }

  for (const auto &entry : comparisonOpPerfInfo) {
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2D, entry.first.first,
                            entry.first.second)
    );
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DSupervisor, entry.first.first,
                            entry.first.second)
    );
  }
  for (const auto &entry : comparisonOpInplacePerfInfo) {
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp2DInPlace, entry.first.first,
                            entry.first.second)
    );
    table.push_back(
      CYCLE_ESTIMATOR_ENTRY(popops, BinaryOp1DInPlaceSupervisor,
                            entry.first.first, entry.first.second)
    );
  }
  return table;
};

} // end namespace popops
