#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>

inline uint64_t getNonLinearityCycles(std::vector<unsigned> regionSizes,
                                      popnn::NonLinearityType nonLinearityType,
                                      bool isFloat,
                                      unsigned dataPathWidth)
{
  uint64_t cycles = 5; // vertex overhead
  for (const auto numItems : regionSizes) {
    const auto floatVectorWidth = dataPathWidth / 32;
    const auto halfVectorWidth =  dataPathWidth / 16;
    cycles += 10; // Loop overhead
    switch (nonLinearityType) {
    case popnn::NonLinearityType::NON_LINEARITY_RELU:
      {
        const unsigned numBlocks = isFloat ?
                  (numItems + floatVectorWidth - 1) / floatVectorWidth :
                  (numItems+ halfVectorWidth - 1) / halfVectorWidth;
        cycles += (numBlocks / 2) * 3 + (numBlocks & 1);
      }
      break;
    case popnn::NonLinearityType::NON_LINEARITY_SIGMOID:
      // scalar operation for floats, vector operation for halves
      // transcendtal operations are ~10cyles for float, ~1cycles for half
      if (isFloat) {
        cycles += numItems * 10;
      } else {
        cycles += (numItems + halfVectorWidth - 1) / halfVectorWidth;
      }
      break;
    case popnn::NonLinearityType::NON_LINEARITY_TANH:
      if (isFloat) {
        cycles += numItems * 10;
      } else {
        cycles += (numItems + halfVectorWidth - 1) / halfVectorWidth;
      }
      break;
    default:
      throw std::runtime_error("Invalid nonlinearity type");
    }
  }
  return cycles;
}

inline uint64_t getBwdNonlinearityDerivativeCycles(
                  unsigned numDeltas,
                  popnn::NonLinearityType nonLinearityType,
                  bool isFloat,
                  unsigned dataPathWidth) {

  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  unsigned numVectors = (numDeltas + vectorWidth - 1) / vectorWidth;

  switch (nonLinearityType) {
  case popnn::NonLinearityType::NON_LINEARITY_SIGMOID:
    return 5 + numVectors * 3;
  case popnn::NonLinearityType::NON_LINEARITY_RELU:
    {
      const unsigned vertexOverhead = 2    // run instruction
                                      + 7; // remaining vertex overhead
      return vertexOverhead + numVectors * 3;
    }
  case popnn::NonLinearityType::NON_LINEARITY_TANH:
    return 5 + numVectors * 3;
  }
  throw std::runtime_error("Invalid nonlinearity type");
}

#endif // _performance_estimation_h_
