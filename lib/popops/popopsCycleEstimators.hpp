// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#ifndef __popopsCycleEstimators_hpp__
#define __popopsCycleEstimators_hpp__

#include "poplar/Type.hpp"
#include "popops/Expr.hpp"
#include <poplibs_support/cyclesTables.hpp>

#include <map>
#include <utility>

using namespace poplar;

using UnaryOpType = popops::expr::UnaryOpType;
using BinaryOpType = popops::expr::BinaryOpType;

namespace popops {

poplibs::CycleEstimatorTable makeCyclesFunctionTable();

namespace {

// Information about processing time (in device cycles) for one elementwise
// operations. This refers only to the timing of the inner loop, not any loop
// setup operations  etc.
// The fields have three patterns of usage:
//
//   naturalVectorWidth == false; loopUnrollFactor == 0
//     This is used only for unary operators.
//     Each loop processes 1 element and takes 'cyclesPerLoop' + an additional
//     (overhead) number of cycles.
//
//   naturalVectorWidth == true; loopUnrollFactor == 0
//     This means that each loop processes the 'natural' vector of the target
//     (for instance 64 bit = 4 halves or 2 floats). Processing 1 vector takes
//     'cyclesPerLoop' + an additional number of overhead cycles that is added
//     later when computing the estimate.
//
//   naturalVectorWidth == false; loopUnrollFactor >= 1
//     This means that each loop processes 'loopUnrollFactor' elements in
//     'cyclesPerLoop'. No overhead cycles are added later.
//
struct OpPerformanceInfo {
  // How many cycles are used to go once through the inner loop, which will
  // process either one "natural" vector of elements or 'loopUnrollFactor'
  // elements.
  unsigned cyclesPerLoop;
  // If true, the operation is performed using the 'natural' vectorization
  // of the target (for instance 4 halves =  64 bit) and an additional number
  // of overhead cycles is added to 'cyclesPerLoop'.
  bool naturalVectorWidth;
  // If 0, or if "naturalVectorWidth" = true, this is disregarded. Otherwise,
  // this indicates the number of elements processed in one loop, and the
  // 'cyclesPerLoop' will NOT have the overhead value added.
  unsigned loopUnrollFactor = 0;
  OpPerformanceInfo() = delete;
  OpPerformanceInfo(unsigned cyclesPerVector, bool naturalVectorWidth)
      : cyclesPerLoop(cyclesPerVector), naturalVectorWidth(naturalVectorWidth) {
  }
  OpPerformanceInfo(unsigned cyclesPerVector, bool naturalVectorWidth,
                    unsigned unroll)
      : cyclesPerLoop(cyclesPerVector), naturalVectorWidth(naturalVectorWidth),
        loopUnrollFactor(unroll) {}
  OpPerformanceInfo(unsigned cyclesPerVector)
      : OpPerformanceInfo(cyclesPerVector, false) {}
};

// Definitions of tables (std::map) that contain OpPerformanceInfo for each
// unary, binary and scalar broadcast operation, both in place and not.

// Operations have been benchmarked in a variety of ways, some notes:
//
// Simple operations which are implemented directly with an instruction are
// of course very quick. Those with a float or half type will produce a
// bundled pair of instructions, hence are faster than int types. In these
// cases the cycle time can be found by viewing the assembly output.
//
// logarithm, sqrt, divide have float instructions (not int), but they are not
// single cycle.
//
// Cycles for many library operations can be *hugely* data dependent.
// This includes unary floating point operations like ASIN, COS, SIN,
// LOGARITHM_ONE_PLUS; binary floating point operations ATAN2, POWER, REMAINDER;
// and also int/unsigned DIVIDE and REMAINDER.
// In these cases the simulator was used to make an estimate of the execution
// time.
// For SIN and COS we use input range -PI, PI when simulating to get a better
// approximation of the cycle estimate.
// For the binary operators, cycles values are from a random distribution of
// input values obtained by running BinaryOpTest

// Some of the operations which produce a bool output use the _st8 function to
// store the result, this adds to the cycle count considerably.

using UnaryOpPerfTable =
    std::map<std::pair<UnaryOpType, poplar::Type>, OpPerformanceInfo>;
using BinaryOpPerfTable =
    std::map<std::pair<BinaryOpType, poplar::Type>, OpPerformanceInfo>;

inline const UnaryOpPerfTable unaryOpPerfInfo = {
    {{UnaryOpType::ABSOLUTE, FLOAT}, {1, false}},
    {{UnaryOpType::ABSOLUTE, HALF}, {1, false}},
    {{UnaryOpType::ABSOLUTE, INT}, {2, false}},
    {{UnaryOpType::ASIN, HALF}, {102, false}},
    {{UnaryOpType::ASIN, FLOAT}, {102, false}},
    // NOT on AUX side, ldst64pace
    {{UnaryOpType::BITWISE_NOT, INT}, {1, true}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_INT}, {1, true}},
    // use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::CEIL, FLOAT}, {2, true}},
    {{UnaryOpType::CEIL, HALF}, {2, true}},
    {{UnaryOpType::COS, FLOAT}, {2300, false}},
    {{UnaryOpType::COS, HALF}, {2300, false}},
    {{UnaryOpType::INVERSE, HALF}, {15, true}},
    {{UnaryOpType::INVERSE, FLOAT}, {5, true}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::EXPONENT, FLOAT}, {2, true}},
    // Use f16v2exp
    {{UnaryOpType::EXPONENT, HALF}, {2, true}},
    {{UnaryOpType::EXPONENT_MINUS_ONE, FLOAT}, {4, false}},
    {{UnaryOpType::EXPONENT_MINUS_ONE, HALF}, {5, true}},

    // Use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::FLOOR, FLOAT}, {2, true}},
    {{UnaryOpType::FLOOR, HALF}, {2, true}},
    // 1 for v==v
    // 1 for v!=INFINITY
    // 1 for anding the two together
    // 1 for converting a match from 0xffff to 0x0001
    // 1 to convert the 32/16bit individual results to 8bits each
    {{UnaryOpType::IS_FINITE, FLOAT}, {5, true}},
    {{UnaryOpType::IS_FINITE, HALF}, {5, true}},
    // 1 for v!=INFINITY
    // 1 for converting a match from 0xffff to 0x0001
    // 1 to convert the 32/16bit individual results to 8bits each
    {{UnaryOpType::IS_INF, FLOAT}, {3, true}},
    {{UnaryOpType::IS_INF, HALF}, {5, true}},
    // 1 for v==v
    // 1 for converting a match from 0xffff to 0x0001
    // 1 to convert the 32/16bit individual results to 8bits each
    {{UnaryOpType::IS_NAN, FLOAT}, {3, true}},
    {{UnaryOpType::IS_NAN, HALF}, {3, true}},
    {{UnaryOpType::LOGARITHM, FLOAT}, {60, true}},
    {{UnaryOpType::LOGARITHM, HALF}, {15, true}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {180, true}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {180, true}},
    {{UnaryOpType::LOGICAL_NOT, BOOL}, {17, false}},
    {{UnaryOpType::NEGATE, FLOAT}, {1, true}},
    {{UnaryOpType::NEGATE, HALF}, {1, true}},
    {{UnaryOpType::NEGATE, INT}, {2, false}},
    {{UnaryOpType::POPCOUNT, INT}, {1, false}},
    {{UnaryOpType::POPCOUNT, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::ROUND, FLOAT}, {2, true}},
    {{UnaryOpType::ROUND, HALF}, {2, true}},
    {{UnaryOpType::SIGNUM, FLOAT}, {5, true}},
    {{UnaryOpType::SIGNUM, HALF}, {5, true}},
    {{UnaryOpType::SIGNUM, INT}, {5}},
    {{UnaryOpType::SIN, FLOAT}, {2300, false}},
    {{UnaryOpType::SIN, HALF}, {2300, false}},
    {{UnaryOpType::SQRT, FLOAT}, {23, false}},
    {{UnaryOpType::SQRT, HALF}, {23, false}},
    {{UnaryOpType::SQRT, INT}, {110, false}},
    {{UnaryOpType::SQUARE, FLOAT}, {1, true}},
    {{UnaryOpType::SQUARE, HALF}, {1, true}},
    {{UnaryOpType::SQUARE, INT}, {1, true}},
    {{UnaryOpType::SQUARE, UNSIGNED_INT}, {1, true}},
    {{UnaryOpType::TAN, FLOAT}, {3900, true}},
    {{UnaryOpType::TAN, HALF}, {3900, true}},
    {{UnaryOpType::TANH, FLOAT}, {1, true}},
    {{UnaryOpType::TANH, HALF}, {2, true}}, // only vectorised v2, not v4
    {{UnaryOpType::SIGMOID, FLOAT}, {1, false}},
    {{UnaryOpType::SIGMOID, HALF}, {2, true}},
    {{UnaryOpType::RSQRT, FLOAT}, {1, false}},
    {{UnaryOpType::RSQRT, HALF}, {3, true}},
};

inline const UnaryOpPerfTable unaryOpInPlacePerfInfo = {
    {{UnaryOpType::ABSOLUTE, FLOAT}, {1, true}},
    {{UnaryOpType::ABSOLUTE, HALF}, {1, true}},
    {{UnaryOpType::ABSOLUTE, INT}, {2}},
    // NOT on AUX side, ldst64pace
    {{UnaryOpType::BITWISE_NOT, INT}, {1, true}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_INT}, {1, true}},
    // use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::CEIL, FLOAT}, {2, true}},
    {{UnaryOpType::CEIL, HALF}, {2, true}},
    {{UnaryOpType::COS, FLOAT}, {2300, false}},
    {{UnaryOpType::COS, HALF}, {2300, false}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::INVERSE, HALF}, {15, true}},
    {{UnaryOpType::INVERSE, FLOAT}, {5, true}},
    {{UnaryOpType::EXPONENT, FLOAT}, {2, true}},
    // Use f16v2exp
    {{UnaryOpType::EXPONENT, HALF}, {2, true}},
    {{UnaryOpType::EXPONENT_MINUS_ONE, FLOAT}, {4, false}},
    {{UnaryOpType::EXPONENT_MINUS_ONE, HALF}, {5, true}},

    // Use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::FLOOR, FLOAT}, {2, true}},
    {{UnaryOpType::FLOOR, HALF}, {2, true}},
    {{UnaryOpType::LOGARITHM, FLOAT}, {60, true}},
    {{UnaryOpType::LOGARITHM, HALF}, {15, true}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {180, true}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {180, true}},
    {{UnaryOpType::LOGICAL_NOT, BOOL}, {17, true}},
    {{UnaryOpType::NEGATE, FLOAT}, {1, true}},
    {{UnaryOpType::NEGATE, HALF}, {1, true}},
    {{UnaryOpType::NEGATE, INT}, {2, false}},
    {{UnaryOpType::POPCOUNT, INT}, {1, false}},
    {{UnaryOpType::POPCOUNT, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::ROUND, FLOAT}, {2, true}},
    {{UnaryOpType::ROUND, HALF}, {2, true}},
    {{UnaryOpType::SIGNUM, FLOAT}, {5, true}},
    {{UnaryOpType::SIGNUM, HALF}, {5, true}},
    {{UnaryOpType::SIGNUM, INT}, {5}},
    {{UnaryOpType::SIN, FLOAT}, {2300, false}},
    {{UnaryOpType::SIN, HALF}, {2300, false}},
    {{UnaryOpType::SQRT, FLOAT}, {23, false}},
    {{UnaryOpType::SQRT, HALF}, {23, false}},
    {{UnaryOpType::SQRT, INT}, {110, false}},
    {{UnaryOpType::SQUARE, FLOAT}, {1, true}},
    {{UnaryOpType::SQUARE, HALF}, {1, true}},
    {{UnaryOpType::SQUARE, INT}, {1, true}},
    {{UnaryOpType::SQUARE, UNSIGNED_INT}, {1, true}},
    {{UnaryOpType::TAN, FLOAT}, {3900, false}},
    {{UnaryOpType::TAN, HALF}, {3900, true}},
    {{UnaryOpType::TANH, FLOAT}, {1, false}},
    {{UnaryOpType::TANH, HALF}, {2, true}},
    {{UnaryOpType::SIGMOID, FLOAT}, {1, false}},
    {{UnaryOpType::SIGMOID, HALF}, {2, true}},
    {{UnaryOpType::RSQRT, FLOAT}, {1, false}},
    {{UnaryOpType::RSQRT, HALF}, {3, true}},
};

inline const BinaryOpPerfTable binaryOpPerfInfo = {
    // Note that the ADD, SUBTRACT and MULTIPLY operators can run code that
    // is *faster* than what is defined here (the 'fastPath'), if the two
    // operands are laid out in memory appropriately to allow a double
    // vector load.
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {7, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {7, false, 1}},

    {{BinaryOpType::DIVIDE, FLOAT}, {10, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {21, false, 4}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {265, false, 1}},
    {{BinaryOpType::LOGICAL_AND, BOOL}, {49, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {23, false, 4}},
    {{BinaryOpType::MAXIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MINIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {7, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {7, false, 1}},

    // Accuracy concerns using ln
    // pow(a,b) = exp(b * log(a))
    // Doesn't handle negative values yet

    // Power instruction not used
    {{BinaryOpType::POWER, FLOAT}, {312, false, 1}},
    {{BinaryOpType::POWER, HALF}, {325, false, 1}},

    {{BinaryOpType::REMAINDER, FLOAT}, {133, false, 2}},
    {{BinaryOpType::REMAINDER, HALF}, {300, false, 4}},
    {{BinaryOpType::REMAINDER, INT}, {292, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_INT}, {282, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {7, false, 1}},

    {{BinaryOpType::EQUAL, FLOAT}, {14, false, 4}},
    {{BinaryOpType::EQUAL, HALF}, {9, false, 4}},
    {{BinaryOpType::EQUAL, INT}, {29, false, 2}},
    {{BinaryOpType::EQUAL, UNSIGNED_INT}, {29, false, 2}},
    {{BinaryOpType::EQUAL, BOOL}, {49, false, 4}},
    // same as B < A
    // E = A and B, result = A andc E
    {{BinaryOpType::GREATER_THAN, FLOAT}, {14, false, 4}},
    {{BinaryOpType::GREATER_THAN, HALF}, {9, false, 4}},
    {{BinaryOpType::GREATER_THAN, INT}, {30, false, 2}},
    {{BinaryOpType::GREATER_THAN, UNSIGNED_INT}, {29, false, 2}},
    {{BinaryOpType::GREATER_THAN, BOOL}, {49, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, FLOAT}, {14, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, HALF}, {9, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, INT}, {31, false, 2}},
    {{BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT}, {31, false, 2}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {17, false, 4}},
    {{BinaryOpType::LESS_THAN, FLOAT}, {14, false, 4}},
    {{BinaryOpType::LESS_THAN, HALF}, {9, false, 4}},
    {{BinaryOpType::LESS_THAN, INT}, {30, false, 2}},
    {{BinaryOpType::LESS_THAN, UNSIGNED_INT}, {30, false, 2}},
    {{BinaryOpType::LESS_THAN, BOOL}, {49, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, FLOAT}, {14, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, HALF}, {9, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, INT}, {31, false, 2}},
    {{BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_INT}, {31, false, 2}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {17, false, 4}},
    {{BinaryOpType::NOT_EQUAL, FLOAT}, {20, false, 4}},
    {{BinaryOpType::NOT_EQUAL, HALF}, {11, false, 4}},
    {{BinaryOpType::NOT_EQUAL, INT}, {29, false, 2}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_INT}, {29, false, 2}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {49, false, 4}},
};

inline const BinaryOpPerfTable binaryOpInPlacePerfInfo = {
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {7, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {7, false, 1}},

    {{BinaryOpType::DIVIDE, FLOAT}, {10, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {21, false, 4}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {264, false, 1}},
    {{BinaryOpType::LOGICAL_AND, BOOL}, {49, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {11, false, 4}},
    {{BinaryOpType::MAXIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MINIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {7, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {7, false, 1}},

    // Accuracy concerns using ln
    // pow(a,b) = exp(b * log(a))
    // Doesn't handle negative values yet

    // Power instruction not used
    {{BinaryOpType::POWER, FLOAT}, {312, false, 1}},
    {{BinaryOpType::POWER, HALF}, {325, false, 1}},

    {{BinaryOpType::REMAINDER, FLOAT}, {133, false, 2}},
    {{BinaryOpType::REMAINDER, HALF}, {300, false, 4}},
    {{BinaryOpType::REMAINDER, INT}, {292, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_INT}, {282, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {7, false, 1}},

    // E = A and B, F = A or B, G = F andc E, result = 1 andc G
    {{BinaryOpType::EQUAL, BOOL}, {49, false, 4}},
    // same as B < A
    // E = A and B, result = A andc E
    {{BinaryOpType::GREATER_THAN, BOOL}, {49, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {13, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {49, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {13, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {49, false, 4}},
};

inline const BinaryOpPerfTable broadcastOpPerfInfo = {
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {6, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {6, false, 1}},

    {{BinaryOpType::DIVIDE, FLOAT}, {8, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {21, false, 4}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {260, false, 1}},
    {{BinaryOpType::LOGICAL_AND, BOOL}, {12, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {12, false, 4}},
    {{BinaryOpType::MAXIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::MINIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {6, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {6, false, 1}},

    {{BinaryOpType::POWER, FLOAT}, {312, false, 1}},
    {{BinaryOpType::POWER, HALF}, {325, false, 1}},

    {{BinaryOpType::REMAINDER, FLOAT}, {93, false, 2}},
    {{BinaryOpType::REMAINDER, HALF}, {225, false, 4}},
    {{BinaryOpType::REMAINDER, INT}, {280, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_INT}, {267, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, FLOAT}, {11, false, 2}},
    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF}, {10, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, FLOAT}, {12, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, HALF}, {24, false, 4}},

    {{BinaryOpType::EQUAL, FLOAT}, {12, false, 4}},
    {{BinaryOpType::EQUAL, HALF}, {8, false, 4}},
    {{BinaryOpType::EQUAL, INT}, {16, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::EQUAL, BOOL}, {25, false, 4}},

    {{BinaryOpType::GREATER_THAN, FLOAT}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN, HALF}, {8, false, 4}},
    {{BinaryOpType::GREATER_THAN, INT}, {16, false, 4}},
    {{BinaryOpType::GREATER_THAN, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::GREATER_THAN, BOOL}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, FLOAT}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, HALF}, {8, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, INT}, {20, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT}, {20, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN, FLOAT}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN, HALF}, {8, false, 4}},
    {{BinaryOpType::LESS_THAN, INT}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, FLOAT}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, HALF}, {8, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, INT}, {20, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_INT}, {20, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, FLOAT}, {18, false, 4}},
    {{BinaryOpType::NOT_EQUAL, HALF}, {10, false, 4}},
    {{BinaryOpType::NOT_EQUAL, INT}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {25, false, 4}},
};

inline const BinaryOpPerfTable broadcastOpInPlacePerfInfo = {
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {6, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {6, false, 1}},

    {{BinaryOpType::DIVIDE, FLOAT}, {8, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {21, false, 4}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {260, false, 1}},
    {{BinaryOpType::LOGICAL_AND, BOOL}, {12, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {12, false, 4}},
    {{BinaryOpType::MAXIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::MINIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {6, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {6, false, 1}},

    {{BinaryOpType::POWER, FLOAT}, {312, false, 1}},
    {{BinaryOpType::POWER, HALF}, {325, false, 1}},

    {{BinaryOpType::REMAINDER, FLOAT}, {93, false, 2}},
    {{BinaryOpType::REMAINDER, HALF}, {225, false, 4}},
    {{BinaryOpType::REMAINDER, INT}, {280, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_INT}, {267, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, FLOAT}, {7, true}},
    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF}, {10, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, FLOAT}, {12, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, HALF}, {24, false, 4}},

    {{BinaryOpType::EQUAL, BOOL}, {25, false, 4}},
    {{BinaryOpType::GREATER_THAN, BOOL}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {25, false, 4}},
};

} // unnamed namespace

} // namespace popops
#endif
