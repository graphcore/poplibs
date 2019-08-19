#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

inline uint64_t getNonLinearityOpCycles(popnn::NonLinearityType nlType,
                                        bool isFloat) {
  // Based off the worst-case cycles from arch_man for float/half
  // transcendental ops.
  uint64_t opCycles;
  switch (nlType) {
  case popnn::NonLinearityType::RELU:
    opCycles = 1;
    break;
  case popnn::NonLinearityType::SIGMOID:
    opCycles = (isFloat ? 5 : 2);
    break;
  case popnn::NonLinearityType::TANH:
    opCycles = (isFloat ? 5 : 1);
    break;
  case popnn::NonLinearityType::GELU:
    //TODO: These are just placeholders. Change these when the nonlinearity
    //      is coded in assembly.
    opCycles = isFloat ? 10:5;
    break;
  default:
    throw poputil::poplibs_error("Unhandled non-linearity type");
    break;
  }
  return opCycles;
}

inline uint64_t getLossTransformCycles(const bool isFloat,
                                       const bool isSoftmax,
                                       const std::size_t size) {
  uint64_t cycles =
        5 // vertex overhead;
      + (isSoftmax ? 6 : 5) // loads of pointers
      + 5 // get base and pointer shifts
      + (isFloat ? 0 : 1) // shift size for halves
      + 2 // 2 load aheads
      + 1 // repeat instruction
      + (isSoftmax ? 9 : 4) * (isFloat ? size : size / 2) // loop
      + (isFloat ? 0 : (2 + (size & 0x1 ? (isSoftmax ? 11 : 6) : 0))) // RMW
      + 1; // exit instruction
  return cycles;
}

#endif // _performance_estimation_h_
