// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#ifndef __popopsCycleEstimators_hpp__
#define __popopsCycleEstimators_hpp__

#include "poplar/Type.hpp"
#include "popops/Expr.hpp"
#include <poputil/cyclesTables.hpp>

#include <map>
#include <utility>

using namespace poplar;

using UnaryOpType = popops::expr::UnaryOpType;
using BinaryOpType = popops::expr::BinaryOpType;

namespace popops {

poputil::internal::PerfEstimatorTable makePerfFunctionTable();

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
// Floating point logarithm, sqrt, divide, sigm, tanh, exp are single
// instructions, but they are NOT single cycle and NOT even constant cycle
// number (the number of cycles depend on the data, especially for 32 bit
// floats).
//
// Cycles for many library operations can be *hugely* data dependent.
// This includes unary floating point operations like ASIN, COS, SIN,
// LOGARITHM_ONE_PLUS; binary floating point operations ATAN2, POWER, REMAINDER;
// and also int/unsigned DIVIDE and REMAINDER.
//
// In all the cases of data dependency, the simulator was used to make an
// estimate of then average execution time, using a random distribution of
// input values.
// For SIN and COS we use input range -PI, PI when simulating to get a better
// approximation of the cycle estimate.
// For the binary operators, cycles values are from a random distribution of
// input values obtained by running BinaryOpTest

// Some of the BinaryOp operations which produce a bool output use the _st8
// function to store the result, this adds to the cycle count considerably.

using UnaryOpPerfTable =
    std::map<std::pair<UnaryOpType, poplar::Type>, OpPerformanceInfo>;
using BinaryOpPerfTable =
    std::map<std::pair<BinaryOpType, poplar::Type>, OpPerformanceInfo>;

inline const UnaryOpPerfTable unaryOpPerfInfo = {
    {{UnaryOpType::ABSOLUTE, FLOAT}, {1, false}},
    {{UnaryOpType::ABSOLUTE, HALF}, {1, false}},
    {{UnaryOpType::ABSOLUTE, INT}, {2, false}},
    {{UnaryOpType::ABSOLUTE, LONGLONG}, {10, false}},

    {{UnaryOpType::ASIN, HALF}, {102, false}},
    {{UnaryOpType::ASIN, FLOAT}, {102, false}},
    // NOT on AUX side, ldst64pace
    {{UnaryOpType::BITWISE_NOT, INT}, {1, true}},
    {{UnaryOpType::BITWISE_NOT, LONGLONG}, {3, true}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_INT}, {1, true}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_LONGLONG}, {3, true}},
    {{UnaryOpType::BITWISE_NOT, SHORT}, {2, false, 4}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_SHORT}, {2, false, 4}},
    // 3 for abs
    // 12 (2 of f32ln) float and 4 (2 of f16v2ln) half for log
    // 3 for load constant and multiply
    // 6 float (2 of f32exp) and 4 (2 of f16v2exp) half for exp
    // 3 for copysign
    {{UnaryOpType::CBRT, FLOAT}, {27, true}},
    {{UnaryOpType::CBRT, HALF}, {17, true}},
    // use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::CEIL, FLOAT}, {2, true}},
    {{UnaryOpType::CEIL, HALF}, {2, true}},
    {{UnaryOpType::COS, FLOAT}, {2300, false}},
    {{UnaryOpType::COS, HALF}, {26, false, 1}},
    {{UnaryOpType::INVERSE, HALF}, {15, false, 4}},
    {{UnaryOpType::INVERSE, FLOAT}, {6, false, 2}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, UNSIGNED_INT}, {1, false}},
    // 1 - abs, 2 - max, 3 - sign
    // 4 to compute eta
    // 10 macs for polynomial
    // derivative of Phi function - 3
    // mul with sign and phi - 2
    // 5 overhead to load constants
    {{UnaryOpType::ERF, FLOAT}, {18, false, 1}},
    // Float + cast in and out
    {{UnaryOpType::ERF, HALF}, {16, false, 1}},

    {{UnaryOpType::EXPONENT, FLOAT}, {6, false, 2}},
    // Use f16v2exp
    {{UnaryOpType::EXPONENT, HALF}, {4, false, 4}},
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
    {{UnaryOpType::LOGARITHM, FLOAT}, {12, false, 2}},
    {{UnaryOpType::LOGARITHM, HALF}, {4, false, 4}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {180, true}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {180, true}},
    {{UnaryOpType::LOGICAL_NOT, BOOL}, {17, false}},
    {{UnaryOpType::NEGATE, FLOAT}, {2, false, 2}},
    {{UnaryOpType::NEGATE, HALF}, {2, false, 4}},
    {{UnaryOpType::NEGATE, INT}, {2, false}},
    {{UnaryOpType::NEGATE, LONGLONG}, {5, false}},

    {{UnaryOpType::POPCOUNT, INT}, {1, false}},
    {{UnaryOpType::POPCOUNT, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::ROUND, FLOAT}, {2, true}},
    {{UnaryOpType::ROUND, HALF}, {2, true}},
    {{UnaryOpType::SIGNUM, FLOAT}, {5, true}},
    {{UnaryOpType::SIGNUM, HALF}, {5, true}},
    {{UnaryOpType::SIGNUM, INT}, {5}},
    {{UnaryOpType::SIN, FLOAT}, {2300, false}},
    {{UnaryOpType::SIN, HALF}, {26, false, 1}},
    {{UnaryOpType::SQRT, FLOAT}, {10, false, 2}},
    {{UnaryOpType::SQRT, HALF}, {23, false, 4}},
    {{UnaryOpType::SQRT, INT}, {110, false}},
    {{UnaryOpType::SQUARE, FLOAT}, {2, false, 2}},
    {{UnaryOpType::SQUARE, HALF}, {2, false, 4}},
    {{UnaryOpType::SQUARE, INT}, {1, true}},
    {{UnaryOpType::SQUARE, UNSIGNED_INT}, {1, true}},
    {{UnaryOpType::TAN, FLOAT}, {5100, false, 1}},
    {{UnaryOpType::TAN, HALF}, {5700, false, 1}},
    {{UnaryOpType::TANH, FLOAT}, {12, false, 2}},
    {{UnaryOpType::TANH, HALF}, {4, false, 4}},
    {{UnaryOpType::SIGMOID, FLOAT}, {12, false, 2}},
    {{UnaryOpType::SIGMOID, HALF}, {6, false, 4}},
    {{UnaryOpType::RSQRT, FLOAT}, {8, false, 2}},
    {{UnaryOpType::RSQRT, HALF}, {19, false, 4}},
    {{UnaryOpType::RELU, FLOAT}, {3, false, 2}},
    {{UnaryOpType::RELU, HALF}, {3, false, 4}},
};

inline const UnaryOpPerfTable unaryOpInPlacePerfInfo = {
    {{UnaryOpType::ABSOLUTE, FLOAT}, {1, true}},
    {{UnaryOpType::ABSOLUTE, HALF}, {1, true}},
    {{UnaryOpType::ABSOLUTE, INT}, {2}},
    {{UnaryOpType::ABSOLUTE, LONGLONG}, {10}},
    {{UnaryOpType::ASIN, HALF}, {102, false}},
    {{UnaryOpType::ASIN, FLOAT}, {102, false}},
    // NOT on AUX side, ldst64pace
    {{UnaryOpType::BITWISE_NOT, INT}, {1, true}},
    {{UnaryOpType::BITWISE_NOT, LONGLONG}, {3, true}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_INT}, {1, true}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_LONGLONG}, {3, true}},
    {{UnaryOpType::BITWISE_NOT, SHORT}, {2, false, 4}},
    {{UnaryOpType::BITWISE_NOT, UNSIGNED_SHORT}, {2, false, 4}},
    // 3 for abs
    // 12 (2 of f32ln) float and 4 (2 of f16v2ln) half for log
    // 3 for load constant and multiply
    // 6 float (2 of f32exp) and 4 (2 of f16v2exp) half for exp
    // 3 for copysign
    {{UnaryOpType::CBRT, FLOAT}, {27, true}},
    {{UnaryOpType::CBRT, HALF}, {17, true}},
    // use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::CEIL, FLOAT}, {2, true}},
    {{UnaryOpType::CEIL, HALF}, {2, true}},
    {{UnaryOpType::COS, FLOAT}, {2300, false}},
    {{UnaryOpType::COS, HALF}, {26, false, 1}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, INT}, {1, false}},
    {{UnaryOpType::COUNT_LEADING_ZEROS, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::INVERSE, HALF}, {15, false, 4}},
    {{UnaryOpType::INVERSE, FLOAT}, {6, false, 2}},
    // 1 - abs, 2 - max, 3 - sign
    // 4 to compute eta
    // 10 macs for polynomial
    // derivative of Phi function - 3
    // mul with sign and phi - 2
    // 5 overhead to load constants
    {{UnaryOpType::ERF, FLOAT}, {18, false, 1}},
    // Float + cast in and out
    {{UnaryOpType::ERF, HALF}, {16, false, 1}},
    {{UnaryOpType::EXPONENT, FLOAT}, {6, false, 2}},
    // Use f16v2exp
    {{UnaryOpType::EXPONENT, HALF}, {4, false, 4}},
    {{UnaryOpType::EXPONENT_MINUS_ONE, FLOAT}, {4, false}},
    {{UnaryOpType::EXPONENT_MINUS_ONE, HALF}, {5, true}},

    // Use mul with 1.0 and use correct rounding mode
    {{UnaryOpType::FLOOR, FLOAT}, {2, true}},
    {{UnaryOpType::FLOOR, HALF}, {2, true}},
    {{UnaryOpType::LOGARITHM, FLOAT}, {14, false, 2}},
    {{UnaryOpType::LOGARITHM, HALF}, {6, false, 4}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, FLOAT}, {180, true}},
    {{UnaryOpType::LOGARITHM_ONE_PLUS, HALF}, {180, true}},
    {{UnaryOpType::LOGICAL_NOT, BOOL}, {17, true}},
    {{UnaryOpType::NEGATE, FLOAT}, {3, false, 2}},
    {{UnaryOpType::NEGATE, HALF}, {3, false, 4}},
    {{UnaryOpType::NEGATE, INT}, {2, false}},
    {{UnaryOpType::NEGATE, LONGLONG}, {5, false}},
    {{UnaryOpType::POPCOUNT, INT}, {1, false}},
    {{UnaryOpType::POPCOUNT, UNSIGNED_INT}, {1, false}},
    {{UnaryOpType::ROUND, FLOAT}, {2, true}},
    {{UnaryOpType::ROUND, HALF}, {2, true}},
    {{UnaryOpType::SIGNUM, FLOAT}, {5, true}},
    {{UnaryOpType::SIGNUM, HALF}, {5, true}},
    {{UnaryOpType::SIGNUM, INT}, {5}},
    {{UnaryOpType::SIN, FLOAT}, {2300, false}},
    {{UnaryOpType::SIN, HALF}, {26, false, 1}},
    {{UnaryOpType::SQRT, FLOAT}, {10, false, 2}},
    {{UnaryOpType::SQRT, HALF}, {23, false, 4}},
    {{UnaryOpType::SQRT, INT}, {110, false}},
    {{UnaryOpType::SQUARE, FLOAT}, {2, false, 2}},
    {{UnaryOpType::SQUARE, HALF}, {2, false, 4}},
    {{UnaryOpType::SQUARE, INT}, {1, true}},
    {{UnaryOpType::SQUARE, UNSIGNED_INT}, {1, true}},
    {{UnaryOpType::TAN, FLOAT}, {5100, false, 1}},
    {{UnaryOpType::TAN, HALF}, {5700, false, 1}},
    {{UnaryOpType::TANH, FLOAT}, {10, false, 2}},
    {{UnaryOpType::TANH, HALF}, {2, false, 4}},
    {{UnaryOpType::SIGMOID, FLOAT}, {10, false, 2}},
    {{UnaryOpType::SIGMOID, HALF}, {4, false, 4}},
    {{UnaryOpType::RSQRT, FLOAT}, {8, false, 2}},
    {{UnaryOpType::RSQRT, HALF}, {19, false, 4}},
    {{UnaryOpType::RELU, FLOAT}, {2, false, 2}},
    {{UnaryOpType::RELU, HALF}, {2, false, 4}},
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
    {{BinaryOpType::ADD, UNSIGNED_LONGLONG}, {11, false, 1}},
    {{BinaryOpType::ADD, LONGLONG}, {11, false, 1}},

    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_AND, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_AND, SHORT}, {3, false, 4}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_SHORT}, {3, false, 4}},

    {{BinaryOpType::BITWISE_OR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_OR, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_OR, SHORT}, {3, false, 4}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_SHORT}, {3, false, 4}},

    {{BinaryOpType::BITWISE_XOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XOR, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XOR, SHORT}, {5, false, 2}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_SHORT}, {5, false, 2}},

    {{BinaryOpType::BITWISE_XNOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, SHORT}, {6, false, 2}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_SHORT}, {6, false, 2}},

    {{BinaryOpType::DIVIDE, FLOAT}, {10, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {21, false, 4}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {265, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_LONGLONG}, {115, false, 1}},
    {{BinaryOpType::DIVIDE, LONGLONG}, {130, false, 1}},

    {{BinaryOpType::LOGICAL_AND, BOOL}, {5, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {5, false, 4}},
    {{BinaryOpType::MAXIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_LONGLONG}, {13, false, 1}},
    {{BinaryOpType::MAXIMUM, LONGLONG}, {15, false, 1}},

    {{BinaryOpType::MINIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_LONGLONG}, {13, false, 1}},
    {{BinaryOpType::MINIMUM, LONGLONG}, {15, false, 1}},

    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {7, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_LONGLONG}, {30, false, 1}},
    {{BinaryOpType::MULTIPLY, LONGLONG}, {30, false, 1}},

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
    {{BinaryOpType::REMAINDER, UNSIGNED_LONGLONG}, {149, false, 1}},
    {{BinaryOpType::REMAINDER, LONGLONG}, {166, false, 1}},

    {{BinaryOpType::SHIFT_LEFT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_LONGLONG}, {18, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, LONGLONG}, {18, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_LONGLONG}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, LONGLONG}, {6, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, LONGLONG}, {19, false, 1}},

    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_LONGLONG}, {11, false, 1}},
    {{BinaryOpType::SUBTRACT, LONGLONG}, {11, false, 1}},

    {{BinaryOpType::EQUAL, FLOAT}, {14, false, 4}},
    {{BinaryOpType::EQUAL, HALF}, {9, false, 4}},
    {{BinaryOpType::EQUAL, INT}, {22, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_INT}, {22, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_LONGLONG}, {48, false, 4}},
    {{BinaryOpType::EQUAL, LONGLONG}, {48, false, 4}},
    {{BinaryOpType::EQUAL, SHORT}, {22, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_SHORT}, {22, false, 4}},
    {{BinaryOpType::EQUAL, BOOL}, {6, false, 4}},
    // same as B < A
    // E = A and B, result = A andc E
    {{BinaryOpType::GREATER_THAN, FLOAT}, {14, false, 4}},
    {{BinaryOpType::GREATER_THAN, HALF}, {9, false, 4}},
    {{BinaryOpType::GREATER_THAN, INT}, {22, false, 4}},
    {{BinaryOpType::GREATER_THAN, UNSIGNED_INT}, {22, false, 4}},
    {{BinaryOpType::GREATER_THAN, UNSIGNED_LONGLONG}, {56, false, 4}},
    {{BinaryOpType::GREATER_THAN, LONGLONG}, {56, false, 4}},

    {{BinaryOpType::GREATER_THAN, BOOL}, {31, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, FLOAT}, {14, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, HALF}, {9, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, INT}, {26, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT}, {26, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_LONGLONG}, {62, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, LONGLONG}, {62, false, 4}},

    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {28, false, 4}},
    {{BinaryOpType::LESS_THAN, FLOAT}, {14, false, 4}},
    {{BinaryOpType::LESS_THAN, HALF}, {9, false, 4}},
    {{BinaryOpType::LESS_THAN, INT}, {22, false, 4}},
    {{BinaryOpType::LESS_THAN, UNSIGNED_INT}, {22, false, 4}},
    {{BinaryOpType::LESS_THAN, UNSIGNED_LONGLONG}, {56, false, 4}},
    {{BinaryOpType::LESS_THAN, LONGLONG}, {56, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {28, false, 4}},

    {{BinaryOpType::LESS_THAN_EQUAL, FLOAT}, {14, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, HALF}, {9, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, INT}, {26, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_INT}, {26, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_LONGLONG}, {62, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, LONGLONG}, {62, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {28, false, 4}},

    {{BinaryOpType::NOT_EQUAL, FLOAT}, {20, false, 4}},
    {{BinaryOpType::NOT_EQUAL, HALF}, {11, false, 4}},
    {{BinaryOpType::NOT_EQUAL, INT}, {22, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_INT}, {22, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_LONGLONG}, {48, false, 4}},
    {{BinaryOpType::NOT_EQUAL, LONGLONG}, {48, false, 4}},
    {{BinaryOpType::NOT_EQUAL, SHORT}, {22, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_SHORT}, {22, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {5, false, 4}},
};

inline const BinaryOpPerfTable binaryOpInPlacePerfInfo = {
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {7, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_LONGLONG}, {11, false, 1}},
    {{BinaryOpType::ADD, LONGLONG}, {11, false, 1}},
    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_AND, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_AND, SHORT}, {3, false, 4}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_SHORT}, {3, false, 4}},

    {{BinaryOpType::BITWISE_OR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_OR, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_OR, SHORT}, {3, false, 4}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_SHORT}, {3, false, 4}},

    {{BinaryOpType::BITWISE_XOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XOR, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XOR, SHORT}, {5, false, 2}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_SHORT}, {5, false, 2}},

    {{BinaryOpType::BITWISE_XNOR, INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, LONGLONG}, {9, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, SHORT}, {6, false, 2}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_SHORT}, {6, false, 2}},

    {{BinaryOpType::DIVIDE, FLOAT}, {10, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {21, false, 4}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {264, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_LONGLONG}, {113, false, 1}},
    {{BinaryOpType::DIVIDE, LONGLONG}, {130, false, 1}},

    {{BinaryOpType::LOGICAL_AND, BOOL}, {5, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {5, false, 4}},

    {{BinaryOpType::MAXIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_LONGLONG}, {13, false, 1}},
    {{BinaryOpType::MAXIMUM, LONGLONG}, {15, false, 1}},

    {{BinaryOpType::MINIMUM, FLOAT}, {4, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {4, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {7, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_LONGLONG}, {13, false, 1}},
    {{BinaryOpType::MINIMUM, LONGLONG}, {15, false, 1}},

    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {7, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_LONGLONG}, {28, false, 1}},
    {{BinaryOpType::MULTIPLY, LONGLONG}, {28, false, 1}},

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
    {{BinaryOpType::REMAINDER, UNSIGNED_LONGLONG}, {146, false, 1}},
    {{BinaryOpType::REMAINDER, LONGLONG}, {165, false, 1}},

    {{BinaryOpType::SHIFT_LEFT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_LONGLONG}, {16, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, LONGLONG}, {16, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_LONGLONG}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, LONGLONG}, {6, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {7, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, LONGLONG}, {17, false, 1}},

    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_LONGLONG}, {11, false, 1}},
    {{BinaryOpType::SUBTRACT, LONGLONG}, {11, false, 1}},

    {{BinaryOpType::EQUAL, BOOL}, {6, false, 4}},
    {{BinaryOpType::GREATER_THAN, BOOL}, {31, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {28, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {28, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {28, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {5, false, 4}},
};

inline const BinaryOpPerfTable broadcastOpPerfInfo = {
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {6, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::ADD, LONGLONG}, {9, false, 1}},

    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, SHORT}, {2, false, 4}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_SHORT}, {2, false, 4}},

    {{BinaryOpType::BITWISE_OR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, SHORT}, {2, false, 4}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_SHORT}, {2, false, 4}},

    {{BinaryOpType::BITWISE_XOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, SHORT}, {4, false, 2}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_SHORT}, {4, false, 2}},

    {{BinaryOpType::BITWISE_XNOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, SHORT}, {4, false, 2}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_SHORT}, {4, false, 2}},

    {{BinaryOpType::DIVIDE, FLOAT}, {6, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {8, false, 2}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {260, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_LONGLONG}, {148, false, 1}},
    {{BinaryOpType::DIVIDE, LONGLONG}, {103, false, 1}},

    {{BinaryOpType::LOGICAL_AND, BOOL}, {4, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {4, false, 4}},

    {{BinaryOpType::MAXIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_LONGLONG}, {11, false, 1}},
    {{BinaryOpType::MAXIMUM, LONGLONG}, {14, false, 1}},

    {{BinaryOpType::MINIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_LONGLONG}, {13, false, 1}},
    {{BinaryOpType::MINIMUM, LONGLONG}, {14, false, 1}},

    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {6, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_LONGLONG}, {26, false, 1}},
    {{BinaryOpType::MULTIPLY, LONGLONG}, {26, false, 1}},

    {{BinaryOpType::POWER, FLOAT}, {312, false, 1}},
    {{BinaryOpType::POWER, HALF}, {325, false, 1}},

    {{BinaryOpType::REMAINDER, FLOAT}, {93, false, 2}},
    {{BinaryOpType::REMAINDER, HALF}, {225, false, 4}},
    {{BinaryOpType::REMAINDER, INT}, {280, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_INT}, {267, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_LONGLONG}, {178, false, 1}},
    {{BinaryOpType::REMAINDER, LONGLONG}, {137, false, 1}},

    {{BinaryOpType::SHIFT_LEFT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, LONGLONG}, {16, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_LONGLONG}, {16, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_LONGLONG}, {5, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, LONGLONG}, {5, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, LONGLONG}, {17, false, 1}},

    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::SUBTRACT, LONGLONG}, {9, false, 1}},

    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, FLOAT}, {11, false, 2}},
    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF}, {10, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, FLOAT}, {12, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, HALF}, {24, false, 4}},

    {{BinaryOpType::EQUAL, FLOAT}, {12, false, 4}},
    {{BinaryOpType::EQUAL, HALF}, {8, false, 4}},
    {{BinaryOpType::EQUAL, INT}, {16, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_LONGLONG}, {32, false, 4}},
    {{BinaryOpType::EQUAL, LONGLONG}, {32, false, 4}},
    {{BinaryOpType::EQUAL, SHORT}, {16, false, 4}},
    {{BinaryOpType::EQUAL, UNSIGNED_SHORT}, {16, false, 4}},
    {{BinaryOpType::EQUAL, BOOL}, {14, false, 4}},

    {{BinaryOpType::GREATER_THAN, FLOAT}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN, HALF}, {8, false, 4}},
    {{BinaryOpType::GREATER_THAN, INT}, {16, false, 4}},
    {{BinaryOpType::GREATER_THAN, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::GREATER_THAN, UNSIGNED_LONGLONG}, {38, false, 4}},
    {{BinaryOpType::GREATER_THAN, LONGLONG}, {38, false, 4}},
    {{BinaryOpType::GREATER_THAN, BOOL}, {12, false, 4}},

    {{BinaryOpType::GREATER_THAN_EQUAL, FLOAT}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, HALF}, {8, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, INT}, {20, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_INT}, {20, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, UNSIGNED_LONGLONG}, {50, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, LONGLONG}, {50, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {12, false, 4}},

    {{BinaryOpType::LESS_THAN, FLOAT}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN, HALF}, {8, false, 4}},
    {{BinaryOpType::LESS_THAN, INT}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN, UNSIGNED_LONGLONG}, {38, false, 4}},
    {{BinaryOpType::LESS_THAN, LONGLONG}, {38, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {16, false, 4}},

    {{BinaryOpType::LESS_THAN_EQUAL, FLOAT}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, HALF}, {8, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, INT}, {20, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_INT}, {20, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, UNSIGNED_LONGLONG}, {50, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, LONGLONG}, {50, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {16, false, 4}},

    {{BinaryOpType::NOT_EQUAL, FLOAT}, {18, false, 4}},
    {{BinaryOpType::NOT_EQUAL, HALF}, {10, false, 4}},
    {{BinaryOpType::NOT_EQUAL, INT}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_INT}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_LONGLONG}, {32, false, 4}},
    {{BinaryOpType::NOT_EQUAL, LONGLONG}, {32, false, 4}},
    {{BinaryOpType::NOT_EQUAL, SHORT}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, UNSIGNED_SHORT}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {4, false, 4}},
};

inline const BinaryOpPerfTable broadcastOpInPlacePerfInfo = {
    {{BinaryOpType::ADD, FLOAT}, {1, true}},
    {{BinaryOpType::ADD, HALF}, {1, true}},
    {{BinaryOpType::ADD, INT}, {6, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::ADD, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::ADD, LONGLONG}, {9, false, 1}},

    {{BinaryOpType::ATAN2, FLOAT}, {272, false, 2}},
    {{BinaryOpType::ATAN2, HALF}, {578, false, 4}},

    {{BinaryOpType::BITWISE_AND, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_AND, SHORT}, {2, false, 4}},
    {{BinaryOpType::BITWISE_AND, UNSIGNED_SHORT}, {2, false, 4}},

    {{BinaryOpType::BITWISE_OR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_OR, SHORT}, {2, false, 4}},
    {{BinaryOpType::BITWISE_OR, UNSIGNED_SHORT}, {2, false, 4}},

    {{BinaryOpType::BITWISE_XOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XOR, SHORT}, {4, false, 2}},
    {{BinaryOpType::BITWISE_XOR, UNSIGNED_SHORT}, {4, false, 2}},

    {{BinaryOpType::BITWISE_XNOR, INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, LONGLONG}, {7, false, 1}},
    {{BinaryOpType::BITWISE_XNOR, SHORT}, {4, false, 2}},
    {{BinaryOpType::BITWISE_XNOR, UNSIGNED_SHORT}, {4, false, 2}},

    {{BinaryOpType::DIVIDE, FLOAT}, {6, false, 2}},
    {{BinaryOpType::DIVIDE, HALF}, {8, false, 2}},
    {{BinaryOpType::DIVIDE, INT}, {277, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_INT}, {260, false, 1}},
    {{BinaryOpType::DIVIDE, UNSIGNED_LONGLONG}, {148, false, 1}},
    {{BinaryOpType::DIVIDE, LONGLONG}, {103, false, 1}},

    {{BinaryOpType::LOGICAL_AND, BOOL}, {4, false, 4}},
    {{BinaryOpType::LOGICAL_OR, BOOL}, {4, false, 4}},

    {{BinaryOpType::MAXIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MAXIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MAXIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_INT}, {7, false, 1}},
    {{BinaryOpType::MAXIMUM, UNSIGNED_LONGLONG}, {11, false, 1}},
    {{BinaryOpType::MAXIMUM, LONGLONG}, {14, false, 1}},

    {{BinaryOpType::MINIMUM, FLOAT}, {3, false, 2}},
    {{BinaryOpType::MINIMUM, HALF}, {3, false, 4}},
    {{BinaryOpType::MINIMUM, INT}, {6, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_INT}, {8, false, 1}},
    {{BinaryOpType::MINIMUM, UNSIGNED_LONGLONG}, {13, false, 1}},
    {{BinaryOpType::MINIMUM, LONGLONG}, {14, false, 1}},

    {{BinaryOpType::MULTIPLY, FLOAT}, {1, true}},
    {{BinaryOpType::MULTIPLY, HALF}, {1, true}},
    {{BinaryOpType::MULTIPLY, INT}, {6, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::MULTIPLY, UNSIGNED_LONGLONG}, {26, false, 1}},
    {{BinaryOpType::MULTIPLY, LONGLONG}, {26, false, 1}},

    {{BinaryOpType::POWER, FLOAT}, {312, false, 1}},
    {{BinaryOpType::POWER, HALF}, {325, false, 1}},

    {{BinaryOpType::REMAINDER, FLOAT}, {93, false, 2}},
    {{BinaryOpType::REMAINDER, HALF}, {225, false, 4}},
    {{BinaryOpType::REMAINDER, INT}, {280, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_INT}, {267, false, 1}},
    {{BinaryOpType::REMAINDER, UNSIGNED_LONGLONG}, {179, false, 1}},
    {{BinaryOpType::REMAINDER, LONGLONG}, {135, false, 1}},

    {{BinaryOpType::SHIFT_LEFT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, UNSIGNED_LONGLONG}, {16, false, 1}},
    {{BinaryOpType::SHIFT_LEFT, LONGLONG}, {16, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, UNSIGNED_LONGLONG}, {5, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT, LONGLONG}, {5, false, 1}},

    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, INT}, {6, false, 1}},
    {{BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, LONGLONG}, {17, false, 1}},

    {{BinaryOpType::SUBTRACT, FLOAT}, {1, true}},
    {{BinaryOpType::SUBTRACT, HALF}, {1, true}},
    {{BinaryOpType::SUBTRACT, INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_INT}, {6, false, 1}},
    {{BinaryOpType::SUBTRACT, UNSIGNED_LONGLONG}, {9, false, 1}},
    {{BinaryOpType::SUBTRACT, LONGLONG}, {9, false, 1}},

    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, FLOAT}, {7, true}},
    {{BinaryOpType::INV_STD_DEV_TO_VARIANCE, HALF}, {10, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, FLOAT}, {12, false, 2}},
    {{BinaryOpType::VARIANCE_TO_INV_STD_DEV, HALF}, {24, false, 4}},

    {{BinaryOpType::EQUAL, BOOL}, {5, false, 4}},
    {{BinaryOpType::GREATER_THAN, BOOL}, {12, false, 4}},
    {{BinaryOpType::GREATER_THAN_EQUAL, BOOL}, {12, false, 4}},
    {{BinaryOpType::LESS_THAN, BOOL}, {16, false, 4}},
    {{BinaryOpType::LESS_THAN_EQUAL, BOOL}, {16, false, 4}},
    {{BinaryOpType::NOT_EQUAL, BOOL}, {4, false, 4}},
};

} // unnamed namespace

} // namespace popops
#endif
