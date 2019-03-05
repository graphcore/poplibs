#include "popopsCycleEstimators.hpp"
#include "popops/Expr.hpp"
#include "ExprOpUtil.hpp"
#include <map>
#include <cassert>
#include <cmath>

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
    { {BroadcastOpType::ADD, FLOAT}, {5, true} },
    { {BroadcastOpType::ADD, HALF}, {5, true} },
    { {BroadcastOpType::INV_STD_DEV_TO_VARIANCE, FLOAT}, {7, true} },
    { {BroadcastOpType::INV_STD_DEV_TO_VARIANCE, HALF}, {13, true} },
    { {BroadcastOpType::MULTIPLY, FLOAT}, {5, true} },
    { {BroadcastOpType::MULTIPLY, HALF}, {5, true} },
    { {BroadcastOpType::SUBTRACT, FLOAT}, {5, true} },
    { {BroadcastOpType::SUBTRACT, HALF}, {5, true} },
    { {BroadcastOpType::VARIANCE_TO_INV_STD_DEV, FLOAT}, {6, true} },
    { {BroadcastOpType::VARIANCE_TO_INV_STD_DEV, HALF}, {12, true} },

};

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastOp1DInPlaceSupervisor)(
                                    const VertexIntrospector &vertex,
                                    const Target &target,
                                    BroadcastOpType op,
                                    const Type &type) {
  CODELET_FIELD(data);
  assert(type ==HALF || type == FLOAT);
  auto vectorWidth = target.getVectorWidth(type);
  auto numWorkers = target.getNumWorkerContexts();
  auto perfInfo = broadcastOpPerfInfo.at({op, type});

  std::uint64_t cycles = 20;
  std::uint64_t supervisorCycles = 19;
  auto numElems = (data.size() + numWorkers - 1) / numWorkers;
  if(perfInfo.vectorize)
    cycles += perfInfo.cyclesPerVector * (numElems + vectorWidth - 1)
                                                              / vectorWidth;
  else
    cycles += perfInfo.cyclesPerVector * numElems;

  return cycles * numWorkers + supervisorCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BroadcastOp2DInPlace)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     BroadcastOpType op,
                                     const Type &type) {
  CODELET_FIELD(data);
  assert(type ==HALF || type == FLOAT);
  auto vectorWidth = target.getVectorWidth(type);
  auto perfInfo = broadcastOpPerfInfo.at({op, type});

  std::uint64_t cycles = 20;

  for(unsigned i = 0; i < data.size(); i++){
    auto numElems = data[i].size();
    if(perfInfo.vectorize)
      cycles += (perfInfo.cyclesPerVector - 1) * (numElems + vectorWidth - 1)
                                                                / vectorWidth;
    else
      cycles += (perfInfo.cyclesPerVector - 1) * numElems;
    cycles += 28;
  }
  return cycles;
}

std::uint64_t
ScaledArithmeticSupervisorCycleEstimate(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &dataType,
                                     const Type &deltaType,
                                     const bool isConstant,
                                     const bool isSubtract) {
  CODELET_FIELD(data);

  if (dataType == INT || dataType == UNSIGNED_INT) {
    std::uint64_t supervisorCycles = 53 // constant overhead
      + (26 * (data.size()/3)); // main loop

    if(isSubtract && !isConstant) {
      supervisorCycles += 1;
    }

    if (data.size() % 3 == 0) {
      supervisorCycles += 6; // 6 cycle branch to skip the remainder loop
    } else {
      supervisorCycles += 6 // --rem
        + (26 * (data.size()%3)); // remainder loop
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
  const unsigned count = (data.size() / numWorkers / atomSize) * atomSize;
  const unsigned final = data.size() % numWorkers;
  const unsigned rem = (data.size() / numWorkers) % numWorkers
    + iceil(final, atomSize);

  std::uint64_t supervisorCycles = 12 // per-type supervisor overhead
    + 47 // common supervisor overhead
    + (final == 0 ? 7 : 13)
    + 12;

  if(isSubtract && !isConstant) {
      supervisorCycles += 7;
  }
  if(!isConstant) {
    supervisorCycles += 1;
  }

  std::vector<unsigned> workerCycles(numWorkers);
  // Specific mixed precision half, float version
  if (dataType == HALF && deltaType == FLOAT) {
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
    for (unsigned wid = 0; wid <= numWorkers; ++wid) {
      std::uint64_t cycles = 15; // constant worker prologue cycles
      if (count/atomSize != 0) {
        cycles += 6 // inner loop constant overhead
          + (2 * (count/atomSize-1)); // loop cycles
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
                                     const Type &dataType,
                                     const Type &deltaType,
                                     const bool isConstant) {
  return ScaledArithmeticSupervisorCycleEstimate(vertex, target, dataType,
                                            deltaType, isConstant, false);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledSubtractSupervisor)(
                                     const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &dataType,
                                     const Type &deltaType) {
  return ScaledArithmeticSupervisorCycleEstimate(vertex, target, dataType,
                                            deltaType, false, true);
}

std::uint64_t
ScaledAddSub2DCycleEstimate(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type,
                                       const bool isConstant,
                                       const bool isSubtract) {
  CODELET_FIELD(data);

  if (type == INT || type == UNSIGNED_INT) {
    std::uint64_t cycles = 8; // prologue and epilogue overhead.
    for (unsigned i = 0; i < data.size(); ++i) {
      cycles += 7 // outer loop constant overhead
        + (data[i].size() * 5); // inner loop
    }
    if( !isConstant)
      cycles += 1;
    if(isSubtract && !isConstant)
      cycles += 1;
    return cycles;
  } else {
    assert(type == HALF || type == FLOAT);
  }

  const auto grain = type == HALF ? 4 : 2;
  std::uint64_t cycles = 9;// prologue and epilogue overhead.
  if( !isConstant)
    cycles += 1;
  if(isSubtract && !isConstant)
    cycles += 2;

  for (unsigned i = 0; i < data.size(); ++i) {
    cycles += 11 // outer loop constant overhead
      + (data[i].size()/grain != 0 ? 5 : 0) // inner loop overhead
      + (data[i].size()/grain * 2); // inner loop

    if (type == FLOAT) {
      cycles += (data[i].size()%grain != 0 ? 7 : 0); // last element.
    } else {
      auto rem = data[i].size() % grain;
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
                                       const bool isConstant) {
  return ScaledAddSub2DCycleEstimate(vertex, target, type, isConstant, false);
}
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledSubtract2D)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type) {
  return ScaledAddSub2DCycleEstimate(vertex, target, type, false, true);
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
  for (unsigned i=0; i<out.size(); ++i) {
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
  const auto dst = vertex.getFieldInfo("dst");
  std::uint64_t cycles;

  // Cast float to/from half written in assembly.
  // The equations below are a reasonable approximation for both
  // Estimates for other types not revised
  if( (fromType == FLOAT && toType == HALF) ||
      (fromType == HALF && toType == FLOAT) ) {
    auto columns=dst.size();
    if (columns < 4) {
      cycles = 11 + (columns * 14 )/3;
    }
    else {
      cycles = 26 + 2 * (columns/4) + ((columns & 3)*14)/3;
    }
  }
  else {
    // These are not valid for integer and boolean casts
    const auto floatVectorWidth = target.getDataPathWidth() / 32;
    cycles = (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
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

  // Use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::FLOOR, FLOAT}, {2, true} },
  { {UnaryOpType::FLOOR, HALF}, {2, true} },
  { {UnaryOpType::LOGARITHM, FLOAT}, {60, true} },
  { {UnaryOpType::LOGARITHM, HALF}, {15, true} },
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
                        const OpPerformanceInfo &perfInfo, unsigned numElems) {
  std::uint64_t cycles = 0;

  unsigned vectorWidth = 1;
  if (perfInfo.vectorize) {
    vectorWidth = target.getVectorWidth(type);
  }
  // Estimate loop cycles, including a constant loop overhead added to the
  // cycles per vector.  This accounts for load/store and loop decision.
  cycles += basicOpLoopCycles(numElems, vectorWidth,
                              5 + perfInfo.cyclesPerVector);

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
                                      numBoolOpCycles, info, in1[i].size());
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
                                          numBoolOpCycles, info, numElems);
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
                                      numBoolOpCycles, info, in1Out[i].size());
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
                                          numBoolOpCycles, info, numElems);
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
  const auto in1 = vertex.getFieldInfo("in1");
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
    cycles += selectCycles(target, type, in1.size());
  }
  return cycles;
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
  const auto in1 = vertex.getFieldInfo("in1");
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
  const auto in1 = vertex.getFieldInfo("in1Out");
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  assert(in2.size() == in1.size());
  assert(in3.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in2[i].size() == in1[i].size());
    assert(in3[i].size() == in1[i].size());
    cycles += clampCycles(target, type, in1[i].size());
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

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  const unsigned numRegions =
          vertex.getFieldInfo("numRegions").getInitialValue<unsigned>(target);
  auto cycles = 12;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    if(type == HALF)
        cycles += (22 + 2 * nVectors) * numSubElements + 16;
    else
        cycles += (20 + nVectors) * numSubElements + 16;
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

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  const unsigned numRegions =
          vertex.getFieldInfo("numRegions").getInitialValue<unsigned>(target);
  auto cycles = 12;
 for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    if(type == HALF)
        cycles += (22 + 2 * nVectors) * numSubElements + 16;
    else
        cycles += (20 + nVectors) * numSubElements + 16;
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
  auto cycles = 42;
  // This is the more optimistic path - where the inner loop is copying
  // aligned data
  unsigned nCopies = elementsPerWorker / 2;
  cycles += (27 + nCopies) * numSubElements;
  cycles *= numWorkers;

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSliceSupervisor)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
 return MAKE_CYCLE_ESTIMATOR_NAME(DynamicSliceSupervisor)(vertex, target, type);
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

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT,
                                                       UNSIGNED_INT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT, INT, true),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT,
                                                       UNSIGNED_INT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT, INT, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF, FLOAT, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT, true),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT, true),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT, false),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT, false),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, UNSIGNED_INT,
                                                       UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtractSupervisor, HALF, FLOAT),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledSubtract2D, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                                      BroadcastOpType::ADD, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                                      BroadcastOpType::ADD, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                                      BroadcastOpType::SUBTRACT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                                      BroadcastOpType::SUBTRACT, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                                       BroadcastOpType::MULTIPLY, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                                      BroadcastOpType::MULTIPLY, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                              BroadcastOpType::VARIANCE_TO_INV_STD_DEV, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                              BroadcastOpType::VARIANCE_TO_INV_STD_DEV, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                              BroadcastOpType::INV_STD_DEV_TO_VARIANCE, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp2DInPlace,
                              BroadcastOpType::INV_STD_DEV_TO_VARIANCE, HALF),


    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                                       BroadcastOpType::ADD, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                                       BroadcastOpType::ADD, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                                       BroadcastOpType::SUBTRACT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                                       BroadcastOpType::SUBTRACT, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                                       BroadcastOpType::MULTIPLY, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                                       BroadcastOpType::MULTIPLY, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                              BroadcastOpType::VARIANCE_TO_INV_STD_DEV, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                              BroadcastOpType::VARIANCE_TO_INV_STD_DEV, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                              BroadcastOpType::INV_STD_DEV_TO_VARIANCE, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, BroadcastOp1DInPlaceSupervisor,
                              BroadcastOpType::INV_STD_DEV_TO_VARIANCE, HALF),


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

    CYCLE_ESTIMATOR_ENTRY(popops, CircBufIncrIndex),
    CYCLE_ESTIMATOR_ENTRY(popops, CircOffset),

    CYCLE_ESTIMATOR_ENTRY(popops, Select, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Select, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Select, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Select, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, SelectInPlace, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Clamp, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Clamp, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Clamp, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ClampInPlace, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ClampInPlace, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ClampInPlace, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, UNSIGNED_INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, EncodeOneHot, INT, INT),

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
