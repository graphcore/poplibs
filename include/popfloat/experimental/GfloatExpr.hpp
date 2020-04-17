// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef _popfloat_gfloat_expr_hpp_
#define _popfloat_gfloat_expr_hpp_

namespace popfloat {
namespace experimental {

enum class RoundType {
  INV, // Invalid rounding mode
  RZ,  // Round-to-zero
  RN,  // Round-to-nearest ties to even
  RA,  // Round-to-nearest ties away
  RU,  // Round towards positive infinity
  RD,  // Round towards negative infinity
  SR,  // Stochastic rounding with maximum number of random bits
  SX   // Stochastic rounding "eXtended" to use of fewer random bits
};

enum class FormatType {
  INVALID_FORMAT,     // Invalid format
  IEEE_FP16,          // IEEE FP16
  QUANTISED_FP32,     // Generic float stored as IEEE FP32
  QUANTISED_FP16,     // Generic float stored as IEEE FP16
  MIN_NORM_ALIGN_GF8, // FP8 with less than 5 exponent bits
  ONE_FIVE_TWO_GF8,   // 1/5/2 with Infs/Nans enabled
  MAX_NORM_ALIGN_GF8, // 1/5/2 with Infs/Nans disabled
  BFLOAT16,           // 1/8/7 format "Google's Bfloat"
  NO_DENORM_GF16,     // A custom FP16 with denorms disabled
  ENABLE_DENORM_GF16  // A custom FP16 with denorms enabled
};

enum class SRDensityType {
  INVALID,               // Invalid SR Noise density
  UNIFORM,               // Uniform SR Noise density
  NORMAL,                // Normal SR Noise density
  TRUNCATED_NORMAL,      // Truncated Normal SR Noise density
  BERNOULLI,             // Bernoulli SR
  LOGISTIC,              // Logistic SR Noise density
  TRUNCATED_LOGISTIC,    // Truncated Logistic SR Noise density
  LAPLACE,               // Laplace SR Noise density
  TRUNCATED_LAPLACE,     // Truncated Laplace SR Noise density
  LOGIT_NORMAL,          // Logit-normal SR Noise density
  TRUNCATED_LOGIT_NORMAL // Truncated-logit-normal SR Noise density
};

enum class SpecType {
  FP32,  // Select poplar::FLOAT to store the cast output
  BF16,  // Select poplar::BFLOAT to store the cast output
  FP16,  // Select poplar::HALF to store the cast output
  INT8,  // Select poplar::CHAR to store the cast output
  INT16, // Select poplar::SHORT to store the cast output
  INT32, // Select poplar::INT to store the cast output
  AUTO   // Select the smallest storage type to represent the cast output
};

} // end namespace experimental
} // end namespace popfloat
#endif // _popfloat_experimental_gfloat_expr_hpp_
