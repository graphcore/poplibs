#include "popopsCycleEstimators.hpp"
#include "popops/Expr.hpp"
#include "ExprOpUtil.hpp"
#include <map>

using namespace poplar;

namespace popops {

/* Cycle cost computation for basic operations */
static uint64_t basicOpLoopCycles(unsigned overhead,
                                  unsigned numElems,
                                  unsigned vectorSize,
                                  unsigned cyclesPerVector) {
  return overhead + (numElems + vectorSize - 1) / vectorSize  * cyclesPerVector;
}

/* Cycles for comparison operations which result in bool as output.
 * For boolean inputs the number of cycles depend on the type of operation
 * as some ops have to be synthesized from the available instruction set
 */
static uint64_t comparisonOpsCycles(unsigned dataPathWidth,
                                    unsigned numElems,
                                    unsigned boolInputComputeCycles,
                                    Type type) {
  if (type == FLOAT) {
    unsigned vectorWidth = dataPathWidth / 32;
    if (sizeof(bool) == 4) {
      // for dataPathWidth = 64:
      // ld64/cmp, ldst64/and on aux
      return basicOpLoopCycles(5, numElems, vectorWidth, 2);
    } else if (sizeof(bool) == 2) {
      // for dataPathWidth = 64:
      // ld64/cmp, ld64/and, st32/sort16
      return basicOpLoopCycles(5, numElems, vectorWidth, 3);
    } else if (sizeof(bool) == 1) {
      // for dataPathWidth = 64:
      // (ld64/cmp, ld64/and, sort16, atom) * 2 on aux
      //   shuf8, shl16, or, st32 on main
      return basicOpLoopCycles(5, numElems, 4 / vectorWidth,
                               (4 / vectorWidth) * 4 + 5);
    }
  } else if (type == HALF) {
    unsigned vectorWidth = dataPathWidth / 32;
    if (sizeof(bool) == 4) {
      // for dataPathWidth = 64:
      // ld64/cmp, ld64/and
      // sort16, sort16/st64
      return basicOpLoopCycles(5, numElems, vectorWidth, 2 + 2 * vectorWidth);
    } else if (sizeof(bool) == 2) {
      // ldst64/cmp, ld64/amp
      return basicOpLoopCycles(5, numElems, vectorWidth, 2);
    } else if (sizeof(bool) == 1) {
      // for dataPathWidth = 64:
      // (ld64/cmp, ld64/and, sort16, atom) * 2 on aux
      //   shuf8, shl16, or, st32 on main
      return basicOpLoopCycles(5, numElems, 4 / vectorWidth,
                               (4 / vectorWidth) * 4 + 2);
    }
  } else if (type == INT) {
    if (sizeof(bool) == 4) {
      return basicOpLoopCycles(5, numElems, 1, 4);
    } else if (sizeof(bool) == 2) {
      // (ld32, ld32, cmp) * 2, sort16, sort16, st32
      return basicOpLoopCycles(5, numElems, 2, 9);
    } else if (sizeof(bool) == 1) {
      // (ld32, ld32, cmp) * 4, sort16, sort16, sort8, st32
      return basicOpLoopCycles(5, numElems, 4, 16);
    }
  } else if (type == BOOL) {
    unsigned vectorWidth = dataPathWidth / sizeof(bool);
    // ld64/ xor(and), ld64st64
    return basicOpLoopCycles(5, numElems, vectorWidth, boolInputComputeCycles);
  }
  assert(0 && "Bool size not supported");
  return 0;
}

static std::uint64_t
scaledAddCycles(std::vector<unsigned> regionSizes,
                 const Target &target,
                 const Type &type,
                 bool is2D) {
  uint64_t cycles = 5;
  if (!is2D)
    assert(regionSizes.size() == 1);

  for (const auto numElem : regionSizes) {
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
    }
    else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
    }
    else {// integer types are not vectorisable
      cyclesPerVector = 4; //ld/mpy/add/st
      vectorWidth = 1;
    }
    // Inner loop uses the axpy instruction.
    cycles += 5 + cyclesPerVector *
             (1 + (numElem + vectorWidth - 1) / vectorWidth);
  }
  if (!is2D)
    cycles -= 3;
  return cycles;
}

std::uint64_t
encodeOneHotCycles(std::vector<unsigned> regionSizes,
                   const Target &target,
                   const Type &indicesType,
                   const Type &encodedType,
                   bool is2D) {
  std::uint64_t cycles = 5; // Vertex overhead

  if (is2D) {
    cycles += 4 + // Load start/end pointer, sub, shr to get length.
              1;  // Sub 1 for brnzdec loop
  } else {
    cycles += 1;  // Load index pointer.
  }

  const auto vectorWidth = target.getVectorWidth(encodedType);
  for (const auto &regionSize : regionSizes) {
    cycles += 3 + // Load start/end pointer, calculate length.
              1;  // Shift length to get number of elements.

    // Write 0's to every element
    const auto numVectors = regionSize / vectorWidth;
    const auto remainder = regionSize % vectorWidth;
    cycles += numVectors;
    if (encodedType == HALF) {
      cycles += 2; // Check for 32-bit remainder
      if (remainder & 0x2) {
        cycles += 1; // Write 32-bits
      }
      cycles += 2; // Check for 16-bit remainder
      if (remainder & 0x1) {
        cycles += 3; // RMW 16-bits
      }
    } else if (encodedType == FLOAT) {
      cycles += 2; // Check for 32-bit remainder
      if (remainder & 0x1) {
        cycles += 1; // Write 32-bits
      }
    }

    // Write a single 1 to element at correct index
    cycles += 1 + // Load index
              2;  // Check in-bounds
    if (encodedType == HALF) {
      cycles += 2 + // Check alignment
                3;  // RMW
    } else if (encodedType == FLOAT) {
      cycles += 1; // Write 32-bits
    }

    if (is2D) {
      cycles += 1; // brnzdec
    }
  }

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddSupervisor)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  CODELET_FIELD(deltas);
  const auto numWorkers = target.getNumWorkerContexts();
  std::vector<unsigned> regionSizes;
  const auto data = vertex.getFieldInfo("data");
  assert(data.size() == deltas.size());
  regionSizes.push_back((data.size() + numWorkers - 1) / numWorkers);
  // 6 additional cycles overhead to divide work in worker and 9 cycles overhead
  // in supervisor
  return 9 +
      (scaledAddCycles(regionSizes, target, type, false) + 6) * numWorkers;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAdd2D)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type) {
  CODELET_FIELD(deltas);
  std::vector<unsigned> regionSizes;
  const auto data = vertex.getFieldInfo("data");
  assert(data.size() == deltas.size());
  for (unsigned i = 0; i < data.size(); ++i) {
    unsigned numElem = data[i].size();
    assert(data[i].size() == deltas[i].size());
    regionSizes.push_back(numElem);
  }
  return 5 + scaledAddCycles(regionSizes, target, type, true);
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
  // TODO: make this more accurate
  const auto out = vertex.getFieldInfo("out");
  bool isFloat = type == FLOAT;
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  std::uint64_t cycles = 2 // run
                         + 5; // vertex overhead
  for (unsigned i=0; i<out.size(); ++i) {
    auto zeroCycles = (out[i].size() + vectorWidth - 1) / vectorWidth;
    auto const loopOverhead = 3;
    cycles += loopOverhead + zeroCycles;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Zero2d)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  const auto dst = vertex.getFieldInfo("out");
  // These are not valid for integer and boolean casts
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
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

  // Cast float to half written in assembly.  Estimates for other types not
  // revised
  if(fromType == FLOAT && toType == HALF)  {
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

using UnaryOpType = popops::expr::UnaryOpType;

static const std::map<std::pair<UnaryOpType, poplar::Type>, OpPerformanceInfo>
unaryOpPerfInfo = {
  { {UnaryOpType::ABSOLUTE, FLOAT}, {2, true} },
  { {UnaryOpType::ABSOLUTE, HALF}, {2, true} },
  { {UnaryOpType::ABSOLUTE, INT}, {3} },
  // NOT on AUX side, ldst64pace
  { {UnaryOpType::BITWISE_NOT, INT}, {1, true} },
  // use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::CEIL, FLOAT}, {1, true} },
  { {UnaryOpType::CEIL, HALF}, {1, true} },
  { {UnaryOpType::COS, FLOAT}, {150, false} },
  { {UnaryOpType::COS, HALF}, {100, false} },
  { {UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false} },
  { {UnaryOpType::EXPONENT, FLOAT}, {3, false} },
  // Use f16v2exp
  { {UnaryOpType::EXPONENT, HALF}, {4, true} },
  { {UnaryOpType::EXPONENT_MINUS_ONE, FLOAT}, {4, false} },
  { {UnaryOpType::EXPONENT_MINUS_ONE, HALF}, {5, true} },

  // Use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::FLOOR, FLOAT}, {1, true} },
  { {UnaryOpType::FLOOR, HALF}, {1, true} },
  // 1 for v==v
  // 1 for v!=INFINITY
  // 1 for anding the two together
  // 1 for converting a match from 0xffff to 0x0001
  // 1 to convert the 32/16bit individual results to 8bits each
  { {UnaryOpType::IS_FINITE, FLOAT}, {5, true} },
  { {UnaryOpType::IS_FINITE, HALF}, {5, true} },
  { {UnaryOpType::LOGARITHM, FLOAT}, {2, true} },
  { {UnaryOpType::LOGARITHM, HALF}, {4, true} },
  { {UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {3, true} },
  { {UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {5, true} },
  { {UnaryOpType::LOGICAL_NOT, BOOL}, {1, true} },
  { {UnaryOpType::NEGATE, FLOAT}, {1, true} },
  { {UnaryOpType::NEGATE, HALF}, {1, true} },
  { {UnaryOpType::NEGATE, INT}, {3} },
  { {UnaryOpType::POPCOUNT, INT}, {1, false} },
  { {UnaryOpType::ROUND, FLOAT}, {2, true} },
  { {UnaryOpType::ROUND, HALF}, {2, true} },
  { {UnaryOpType::SIGNUM, FLOAT}, {1, true} },
  { {UnaryOpType::SIGNUM, HALF}, {1, true} },
  { {UnaryOpType::SIGNUM, INT}, {5} },
  { {UnaryOpType::SIN, FLOAT}, {150, false} },
  { {UnaryOpType::SIN, HALF}, {100, false} },
  { {UnaryOpType::SQRT, FLOAT}, {5, false} },
  { {UnaryOpType::SQRT, HALF}, {7, false} },
  { {UnaryOpType::SQRT, INT}, {10, false} },
  { {UnaryOpType::SQUARE, FLOAT}, {1, false} },
  { {UnaryOpType::SQUARE, HALF}, {1, false} },
  { {UnaryOpType::TANH, FLOAT}, {7, false} },
  { {UnaryOpType::TANH, HALF}, {4, true} },
};

static const std::map<std::pair<UnaryOpType, poplar::Type>, OpPerformanceInfo>
unaryOpInPlacePerfInfo = {
  { {UnaryOpType::ABSOLUTE, FLOAT}, {2, true} },
  { {UnaryOpType::ABSOLUTE, HALF}, {2, true} },
  { {UnaryOpType::ABSOLUTE, INT}, {3} },
  // NOT on AUX side, ldst64pace
  { {UnaryOpType::BITWISE_NOT, INT}, {1, true} },
  // use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::CEIL, FLOAT}, {1, true} },
  { {UnaryOpType::CEIL, HALF}, {1, true} },
  { {UnaryOpType::COS, FLOAT}, {150, false} },
  { {UnaryOpType::COS, HALF}, {100, false} },
  { {UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false} },
  { {UnaryOpType::EXPONENT, FLOAT}, {3, false} },
  // Use f16v2exp
  { {UnaryOpType::EXPONENT, HALF}, {4, true} },

  // Use mul with 1.0 and use correct rounding mode
  { {UnaryOpType::FLOOR, FLOAT}, {1, true} },
  { {UnaryOpType::FLOOR, HALF}, {1, true} },
  { {UnaryOpType::LOGARITHM, FLOAT}, {2, true} },
  { {UnaryOpType::LOGARITHM, HALF}, {4, true} },
  { {UnaryOpType::LOGICAL_NOT, BOOL}, {1, true} },
  { {UnaryOpType::NEGATE, FLOAT}, {1, true} },
  { {UnaryOpType::NEGATE, HALF}, {1, true} },
  { {UnaryOpType::NEGATE, INT}, {3} },
  { {UnaryOpType::POPCOUNT, INT}, {1, false} },
  { {UnaryOpType::ROUND, FLOAT}, {2, true} },
  { {UnaryOpType::ROUND, HALF}, {2, true} },
  { {UnaryOpType::SIGNUM, FLOAT}, {1, true} },
  { {UnaryOpType::SIGNUM, HALF}, {1, true} },
  { {UnaryOpType::SIGNUM, INT}, {5} },
  { {UnaryOpType::SIN, FLOAT}, {150, false} },
  { {UnaryOpType::SIN, HALF}, {100, false} },
  { {UnaryOpType::SQRT, FLOAT}, {5, false} },
  { {UnaryOpType::SQRT, HALF}, {7, false} },
  { {UnaryOpType::SQRT, INT}, {10, false} },
  { {UnaryOpType::SQUARE, FLOAT}, {1, false} },
  { {UnaryOpType::SQUARE, HALF}, {1, false} },
  { {UnaryOpType::TANH, FLOAT}, {7, false} },
  { {UnaryOpType::TANH, HALF}, {4, true} },
};


using BinaryOpType = popops::expr::BinaryOpType;

static const std::map<std::pair<BinaryOpType, poplar::Type>, OpPerformanceInfo>
binaryOpPerfInfo = {
  { {BinaryOpType::ADD, FLOAT}, {2, true} },
  { {BinaryOpType::ADD, HALF}, {2, true} },
  { {BinaryOpType::ADD, INT}, {4} },
  { {BinaryOpType::ADD, UNSIGNED_INT}, {4} },
  { {BinaryOpType::ATAN2, FLOAT}, {25, false} },
  { {BinaryOpType::ATAN2, HALF}, {25 + 3, false} },
  // AND in parallel with ld2xstpace
  { {BinaryOpType::BITWISE_AND, INT}, {1, true} },
  // OR on AUX side, ld2xstpace
  { {BinaryOpType::BITWISE_OR, INT}, {1, true} },
  { {BinaryOpType::DIVIDE, FLOAT}, {1, false} },
  // Convert to f32 using v2 and divide and convert back to f16
  { {BinaryOpType::DIVIDE, HALF}, {8, true} },
  // ld into aux, ld into aux, div, st
  { {BinaryOpType::DIVIDE, INT}, {4, false} },
  { {BinaryOpType::LOGICAL_AND, BOOL}, {2, false} },
  { {BinaryOpType::LOGICAL_OR, BOOL}, {2, false} },
  { {BinaryOpType::MAXIMUM, FLOAT}, {2, true} },
  { {BinaryOpType::MAXIMUM, HALF}, {2, true} },
  { {BinaryOpType::MAXIMUM, INT}, {4} },
  { {BinaryOpType::MINIMUM, FLOAT}, {2, true} },
  { {BinaryOpType::MINIMUM, HALF}, {2, true} },
  { {BinaryOpType::MINIMUM, INT}, {4} },
  { {BinaryOpType::MULTIPLY, FLOAT}, {2, true} },
  { {BinaryOpType::MULTIPLY, HALF}, {2, true} },
  { {BinaryOpType::MULTIPLY, INT}, {4} },
  // This cycles are wrong
  // Accuracy concerns using ln
  // pow(a,b) = exp(b * log(a))
  // Doesn't handle negative values yet
  { {BinaryOpType::POWER, FLOAT}, {100, true} },
  // used f16v4 variant: Accuracy concerns using half precision log
  { {BinaryOpType::POWER, HALF}, {100, true} },
  { {BinaryOpType::REMAINDER, FLOAT}, {4, true} },
  { {BinaryOpType::REMAINDER, HALF}, {4, true} },
  { {BinaryOpType::REMAINDER, INT}, {1} },
  { {BinaryOpType::SHIFT_LEFT, INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT, INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {4} },
  { {BinaryOpType::SUBTRACT, FLOAT}, {2, true} },
  { {BinaryOpType::SUBTRACT, HALF}, {2, true} },
  { {BinaryOpType::SUBTRACT, INT}, {4} },
  { {BinaryOpType::SUBTRACT, UNSIGNED_INT}, {4} },
};


static const std::map<std::pair<BinaryOpType, poplar::Type>, OpPerformanceInfo>
binaryOpInPlacePerfInfo = {
  { {BinaryOpType::ADD, FLOAT}, {2, true} },
  { {BinaryOpType::ADD, HALF}, {2, true} },
  { {BinaryOpType::ADD, INT}, {4} },
  { {BinaryOpType::ADD, UNSIGNED_INT}, {4} },
  { {BinaryOpType::ATAN2, FLOAT}, {25, false} },
  { {BinaryOpType::ATAN2, HALF}, {25 + 3, false} },
  // AND in parallel with ld2xstpace
  { {BinaryOpType::BITWISE_AND, INT}, {1, true} },
  // OR on AUX side, ld2xstpace
  { {BinaryOpType::BITWISE_OR, INT}, {1, true} },
  { {BinaryOpType::DIVIDE, FLOAT}, {1, false} },
  // Convert to f32 using v2 and divide and convert back to f16
  { {BinaryOpType::DIVIDE, HALF}, {8, true} },
  // ld into aux, ld into aux, div, st
  { {BinaryOpType::DIVIDE, INT}, {4, false} },
  { {BinaryOpType::LOGICAL_AND, BOOL}, {2, false} },
  { {BinaryOpType::LOGICAL_OR, BOOL}, {2, false} },
  { {BinaryOpType::MAXIMUM, FLOAT}, {2, true} },
  { {BinaryOpType::MAXIMUM, HALF}, {2, true} },
  { {BinaryOpType::MAXIMUM, INT}, {4} },
  { {BinaryOpType::MINIMUM, FLOAT}, {2, true} },
  { {BinaryOpType::MINIMUM, HALF}, {2, true} },
  { {BinaryOpType::MINIMUM, INT}, {4} },
  { {BinaryOpType::MULTIPLY, FLOAT}, {2, true} },
  { {BinaryOpType::MULTIPLY, HALF}, {2, true} },
  { {BinaryOpType::MULTIPLY, INT}, {4} },
  // This cycles are wrong
  // Accuracy concerns using ln
  // pow(a,b) = exp(b * log(a))
  // Doesn't handle negative values yet
  { {BinaryOpType::POWER, FLOAT}, {100, true} },
  // used f16v4 variant: Accuracy concerns using half precision log
  { {BinaryOpType::POWER, HALF}, {100, true} },
  { {BinaryOpType::REMAINDER, FLOAT}, {4, true} },
  { {BinaryOpType::REMAINDER, HALF}, {4, true} },
  { {BinaryOpType::REMAINDER, INT}, {1} },
  { {BinaryOpType::SHIFT_LEFT, INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT, INT}, {3} },
  { {BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {4} },
  { {BinaryOpType::SUBTRACT, FLOAT}, {2, true} },
  { {BinaryOpType::SUBTRACT, HALF}, {2, true} },
  { {BinaryOpType::SUBTRACT, INT}, {4} },
  { {BinaryOpType::SUBTRACT, UNSIGNED_INT}, {4} },
};


static const std::map<std::pair<BinaryOpType, poplar::Type>, unsigned>
    comparisonOpPerfInfo = {
  // E = A and B, F = A or B, G = F andc E, result = 1 andc G
  { {BinaryOpType::EQUAL, FLOAT}, 0 },
  { {BinaryOpType::EQUAL, HALF}, 0 },
  { {BinaryOpType::EQUAL, INT}, 0 },
  { {BinaryOpType::EQUAL, BOOL}, 4 },
  // same as B < A
  // E = A and B, result = A andc E
  { {BinaryOpType::GREATER_THAN, FLOAT}, 0 },
  { {BinaryOpType::GREATER_THAN, HALF}, 0 },
  { {BinaryOpType::GREATER_THAN, INT}, 0 },
  { {BinaryOpType::GREATER_THAN, BOOL}, 2 },
  { {BinaryOpType::GREATER_THAN_EQUAL, FLOAT}, 0 },
  { {BinaryOpType::GREATER_THAN_EQUAL, HALF}, 0 },
  { {BinaryOpType::GREATER_THAN_EQUAL, INT}, 0 },
  { {BinaryOpType::GREATER_THAN_EQUAL, BOOL}, 2 },
  { {BinaryOpType::LESS_THAN, FLOAT}, 0 },
  { {BinaryOpType::LESS_THAN, HALF}, 0 },
  { {BinaryOpType::LESS_THAN, INT}, 0 },
  { {BinaryOpType::LESS_THAN, BOOL}, 2 },
  { {BinaryOpType::LESS_THAN_EQUAL, FLOAT}, 0 },
  { {BinaryOpType::LESS_THAN_EQUAL, HALF}, 0 },
  { {BinaryOpType::LESS_THAN_EQUAL, INT}, 0 },
  { {BinaryOpType::LESS_THAN_EQUAL, BOOL}, 2 },
  { {BinaryOpType::NOT_EQUAL, FLOAT}, 0 },
  { {BinaryOpType::NOT_EQUAL, HALF}, 0 },
  { {BinaryOpType::NOT_EQUAL, INT}, 0 },
  { {BinaryOpType::NOT_EQUAL, BOOL}, 3 },
};

static const std::map<std::pair<BinaryOpType, poplar::Type>, unsigned>
    comparisonOpInplacePerfInfo = {
  // E = A and B, F = A or B, G = F andc E, result = 1 andc G
  { {BinaryOpType::EQUAL, BOOL}, 4 },
  // same as B < A
  // E = A and B, result = A andc E
  { {BinaryOpType::GREATER_THAN, BOOL}, 2 },
  { {BinaryOpType::GREATER_THAN_EQUAL, BOOL}, 2 },
  { {BinaryOpType::LESS_THAN, BOOL}, 2 },
  { {BinaryOpType::LESS_THAN_EQUAL, BOOL}, 2 },
  { {BinaryOpType::NOT_EQUAL, BOOL}, 3 },
};

static std::uint64_t
unaryOpInnerLoopCycles(const Target &target, const Type &type,
              const OpPerformanceInfo &perfInfo, unsigned numElems) {
  unsigned vectorWidth = 1;
  if (perfInfo.vectorize) {
    vectorWidth = target.getDataPathWidth() / target.getTypeSize(type);
  }
  const auto loopOverhead = 6;
  return basicOpLoopCycles(loopOverhead, numElems, vectorWidth,
                           perfInfo.cyclesPerVector);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp2D)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     popops::expr::UnaryOpType op,
                                     const Type &type) {
  uint64_t cycles = 5;
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
  uint64_t workerCycles = 10;
  const auto in = vertex.getFieldInfo("in");
  const auto out = vertex.getFieldInfo("out");
  const auto &info = unaryOpPerfInfo.at({op, type});
  assert (in.size() == out.size());
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (in.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // Unary op is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 18;
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(UnaryOp2DInPlace)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            popops::expr::UnaryOpType op,
                                            const Type &type) {
  uint64_t cycles = 5;
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
  uint64_t workerCycles = 10;
  const auto inOut = vertex.getFieldInfo("inOut");
  const auto &info = unaryOpInPlacePerfInfo.at({op, type});
  const auto numWorkers = target.getNumWorkerContexts();
  auto numElems = (inOut.size() + numWorkers - 1) / numWorkers;
  workerCycles += unaryOpInnerLoopCycles(target, type, info, numElems);
  // UnaryOpInPlace is a supervisor vertex
  uint64_t cycles = workerCycles * numWorkers + 18;
  return cycles;
}

static std::uint64_t
binaryOpInnerLoopCycles(const Target &target, const Type &type,
                        bool isComparison, unsigned numBoolOpCycles,
                        const OpPerformanceInfo &perfInfo, unsigned numElems) {
  std::uint64_t cycles = 0;
  if (isComparison) {
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                    numElems,
                                    numBoolOpCycles,
                                    type);
  } else {
    unsigned vectorWidth = 1;
    if (perfInfo.vectorize) {
      vectorWidth = target.getDataPathWidth() / target.getTypeSize(type);
    }
    const auto loopOverhead = 6;
    cycles += basicOpLoopCycles(loopOverhead, numElems, vectorWidth,
                                perfInfo.cyclesPerVector);
  }
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
  uint64_t workerCycles = 13;
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

  return numWorkers * workerCycles + 18;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BinaryOp2DInPlace)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             BinaryOpType op,
                                             const Type &type) {
  uint64_t cycles = 5;
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
  return numWorkers * workerCycles + 18;
}


static std::uint64_t
selectCyces(const Target &target, const Type &type, unsigned numElems) {
  unsigned cyclesPerVector = 5;
  unsigned overhead = 6;
  unsigned vectorWidth = 1;
  // ld in1, ld in2, ld in3, movz, st
  // it may be possible to load on the Aux side but then would
  // depend on bool size. If Aux side is used masks must be created after
  // expanding bools to match the input datum size
  return basicOpLoopCycles(overhead, numElems, vectorWidth, cyclesPerVector);
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
    cycles += selectCyces(target, type, in1.size());
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
    cycles += selectCyces(target, type, in1.size());
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
  return basicOpLoopCycles(overhead, numElems, vectorWidth, cyclesPerVector);
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
MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice)(const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &type) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  auto numRegions = baseT.size() / numBaseElements;
  auto cycles = 5;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    cycles += (4 + nVectors) * numSubElements + 4;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  auto numRegions = baseT.size() / numBaseElements;
  auto cycles = 5;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    cycles += (4 + nVectors) * numSubElements + 4;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice2d)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &type) {
  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
  const unsigned elementsPerWorker =
    vertex.getFieldInfo("elementsPerWorker").getInitialValue<unsigned>(target);
  const unsigned numWorkers = target.getNumWorkerContexts();

  auto cycles = 5;
  unsigned nVectors = (elementsPerWorker + vectorWidth - 1) / vectorWidth;
  cycles += (4 + nVectors) * numSubElements + 4;
  cycles *= numWorkers;
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice2d)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
  return MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice2d)(vertex, target, type);
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
  CODELET_FIELD(out);
  std::vector<unsigned> regionSizes;
  regionSizes.push_back(out.size());
  return encodeOneHotCycles(regionSizes, target, indexType, outputType, false);
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAddSupervisor, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd2D, INT),

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
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, BOOL),

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
