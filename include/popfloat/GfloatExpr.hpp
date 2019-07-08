// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#ifndef _popfloat_gfloat_expr_hpp_
#define _popfloat_gfloat_expr_hpp_

namespace popfloat {

namespace gfexpr {
enum class GfloatCastOpType {
  INVALID_OP,           // Invalid op
  CAST_TO_QUANTISED_GF32, // Cast an FP32 input to a custom Gfloat format that
                          // can be represented as FP32.
  CAST_TO_QUANTISED_GF16, // Cast an FP16/FP32 input to a custom Gfloat format
                          // that can be represented as FP316.
  CAST_FLOAT_TO_SHORT, // Pack a custom FP16 format (a quantised FP32 input)
                       // to its INT16 representation
  CAST_HALF_TO_CHAR, // Pack a custom FP8 format (a quantised FP16 input)
                     // to  its INT8 representation
  CAST_SHORT_TO_FLOAT, // Unpack a custom FP16 format to FP32 from it INT16
                       // representation
  CAST_CHAR_TO_HALF     // Unpack a custom FP8 format to FP16 from it INT8
                        // representation
};

enum class GfloatRoundType {
  INV,  // Invalid rounding mode
  RZ,   // Round-to-zero
  RN,   // Round-to-nearest ties to even
  RA,   // Round-to-nearest ties away
  RU,   // Round towards positive infinity
  RD,   // Round towards negative infinity
  SR,   // Stochastic rounding with maximum number of random bits
  SX    // Stochastic rounding "eXtended" to use of fewer random bits
};

enum class GfloatFormatType {
  INVALID_FORMAT,      // Invalid format
  QUANTISED_FP32,      // Generic float stored as IEEE FP32
  QUANTISED_FP16,      // Generic float stored as IEEE FP16
  MIN_NORM_ALIGN_GF8,  // FP8 with less than 5 exponent bits
  ONE_FIVE_TWO_GF8,    // 1/5/2 with Infs/Nans enabled
  MAX_NORM_ALIGN_GF8,  // 1/5/2 with Infs/Nans disabled
  BFLOAT16,            // 1/8/7 format "Google's Bfloat"
  NO_DENORM_GF16,      // A custom FP16 with denorms disabled
  ENABLE_DENORM_GF16   // A custom FP16 with denorms enabled
};

enum class GfloatSRDensityType {
  INVALID,               // Invalid SR Noise density
  UNIFORM,               // Uniform SR Noise density
  NORMAL,                // Normal SR Noise density
  TRUNCATED_NORMAL,      // Truncated Normal SR Noise density
  BERNOULLI,             // Bernoulli SR
  LOGISTIC,              // Logistic SR Noise density
  LAPLACE,               // Laplace SR Noise density
  LOGIT_NORMAL,          // Logit-normal SR Noise density
  TRUNCATED_LOGIT_NORMAL // Truncted Logit-normal SR Noise density
};

}
} // end namespace popfloat::gfexpr

#endif // _popfloat_gfloat_expr_hpp_
