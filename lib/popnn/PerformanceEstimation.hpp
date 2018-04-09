#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

inline uint64_t getNonLinearityCycles(std::vector<unsigned> regionSizes,
                                      popnn::NonLinearityType nonLinearityType,
                                      bool isFloat,
                                      bool is2D,
                                      unsigned dataPathWidth)
{
  uint64_t cycles = 5; // vertex overhead
  if (!is2D)
    assert(regionSizes.size() == 1);

  for (const auto numItems : regionSizes) {
    const auto floatVectorWidth = dataPathWidth / 32;
    const auto halfVectorWidth =  dataPathWidth / 16;
    const auto transHalfVectorWidth = 2;
    cycles += 10;
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
      // transcendtal operations are ~7cycles for float, ~2cycles for half
      if (isFloat) {
        cycles += numItems * 7;
      } else {
        cycles += 2 * (numItems + transHalfVectorWidth - 1)
                      / transHalfVectorWidth;
      }
      break;
    case popnn::NonLinearityType::NON_LINEARITY_TANH:
      if (isFloat) {
        cycles += numItems * 7;
      } else {
        cycles += 2 * (numItems + transHalfVectorWidth - 1)
                      / transHalfVectorWidth;
      }
      break;
    case popnn::NonLinearityType::NON_LINEARITY_SOFTMAX:
      throw std::runtime_error("Nonlinearity not implemented as a "
                               "single vertex");
    default:
      throw std::runtime_error("Invalid nonlinearity type");
    }
  }
  if (!is2D)
    cycles -= 2;
  return cycles;
}

inline uint64_t getBwdNonlinearityDerivativeCycles(
                  std::vector<unsigned> regionSizes,
                  popnn::NonLinearityType nonLinearityType,
                  bool isFloat,
                  bool is2D,
                  unsigned dataPathWidth) {
  uint64_t cycles = 5;
  if (!is2D)
    assert(regionSizes.size() == 1);
  for (const auto numItems : regionSizes) {
    const unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    const unsigned numVectors = (numItems + vectorWidth - 1) / vectorWidth;

    switch (nonLinearityType) {
    case popnn::NonLinearityType::NON_LINEARITY_SIGMOID:
      cycles += 5 + numVectors * 3;
      break;
    case popnn::NonLinearityType::NON_LINEARITY_RELU:
      {
        const unsigned vertexOverhead = 2    // run instruction
                                      + 7; // remaining vertex overhead
        cycles += vertexOverhead + numVectors * 3;
      }
      break;
    case popnn::NonLinearityType::NON_LINEARITY_TANH:
      cycles += 5 + numVectors * 3;
        break;
    case popnn::NonLinearityType::NON_LINEARITY_SOFTMAX:
      throw std::runtime_error("Nonlinearity not implemented");
    default:
      throw std::runtime_error("Invalid nonlinearity type");
    }
  }
  if (!is2D)
    cycles -= 4;
  return cycles;
}

#endif // _performance_estimation_h_
