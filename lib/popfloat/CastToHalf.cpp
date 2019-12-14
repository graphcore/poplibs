// Copyright (c) Graphcore Ltd, All rights reserved.
#include "codelets/GfloatConst.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <experimental/popfloat/CastToHalf.hpp>
#include <experimental/popfloat/GfloatExpr.hpp>
#include <experimental/popfloat/GfloatExprUtil.hpp>

#include <cassert>
#include <unordered_set>

#include <array>
#include <cmath>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace experimental {
namespace popfloat {

static uint32_t roundMantissaRN(uint32_t m_single, uint32_t masklen) {
  bool msfBitVal = (m_single >> (masklen - 1)) & 1;
  bool lsbs = (m_single & ((1 << (masklen - 1)) - 1)) != 0;
  bool lsBitVal = (m_single >> masklen) & 1;
  uint32_t newMant = (m_single >> masklen);
  if (msfBitVal && (lsBitVal || lsbs)) {
    newMant += 1;
  }
  return newMant;
}

uint16_t singleToHalf(float value, bool enNanoo) {

  uint16_t result;
  uint32_t ivalue;
  int exp;

  std::memcpy(&ivalue, &value, sizeof(ivalue));

  result = POPFLOAT_FP32_SIGN(ivalue) << POPFLOAT_FP16_SIGN_SHIFT;
  exp = POPFLOAT_FP32_EXPONENT(ivalue) - POPFLOAT_FP32_EXPONENT_BIAS;

  int maskLen =
      POPFLOAT_NUM_FP32_MANTISSA_BITS - POPFLOAT_NUM_FP16_MANTISSA_BITS;

  if (exp < -25) {
    /* Very small values map to +-0
       - nothing more to do.
     */
  } else if (exp < -24) {
    /* Half the smallest denorm could round up to the smallest denorm.
     */
    uint32_t mant = POPFLOAT_FP32_MANTISSA(ivalue) |
                    (1 << (POPFLOAT_NUM_FP32_MANTISSA_BITS + 1));
    mant = roundMantissaRN(mant, (POPFLOAT_NUM_FP32_MANTISSA_BITS + 1));
    result |= mant;
  } else if (exp < -14) {
    /* Small numbers map to denorms - will lose precision
     */

    /* Shift the exponent into the mantissa
     */
    int shift = -exp - (POPFLOAT_FP16_EXPONENT_BIAS);

    /* Combine with the original mantissa shifted into place
     */
    uint16_t mant = 1 << ((POPFLOAT_NUM_FP16_MANTISSA_BITS - 1) - shift);
    mant += roundMantissaRN(POPFLOAT_FP32_MANTISSA(ivalue),
                            ((maskLen) + shift + 1));
    result |= mant;

  } else if (exp <= 15) {
    /* Normal numbers - will lose precision
     */

    uint32_t mant = POPFLOAT_FP32_MANTISSA(ivalue) |
                    (1 << (POPFLOAT_NUM_FP32_MANTISSA_BITS));
    mant = roundMantissaRN(mant, maskLen);
    if (mant >> (POPFLOAT_NUM_FP16_MANTISSA_BITS + 1)) {
      mant >>= 1;
      ++exp;
    }
    if (exp > 15) {
      if (enNanoo) {
        result |= 0x7BFF;
      } else {
        result |= 65504;
      }
    } else {
      result |= mant & POPFLOAT_FP16_MANTISSA_MASK;
      result |= ((exp + POPFLOAT_FP16_EXPONENT_BIAS)
                 << POPFLOAT_NUM_FP16_MANTISSA_BITS);
    }
  } else if (exp < 128) {
    /* Large numbers map to infinity if NANOO is OFF, otherwise clip to F16 Max
     */
    if (enNanoo) {
      result |= 0x7BFF;
    } else {
      result |= 65504;
    }

  } else if (std::isnan(value)) {
    /* NaNs map to NaNs
     */
    uint16_t mant = roundMantissaRN(POPFLOAT_FP32_MANTISSA(ivalue), maskLen);

    if (POPFLOAT_FP32_IS_QNAN(ivalue)) {
      mant |= (1 << POPFLOAT_FP16_Q_SHIFT);
    } else {
      mant &= ~(1 << POPFLOAT_FP16_Q_SHIFT);

      if (mant == 0) {
        /* Ensure NaNs stay as NaNs (non-zero mantissa)
         */
        mant |= 1;
      }
    }

    result |= POPFLOAT_FP16_INFINITY;
    result |= mant;

  } else {
    /* Infinity maps to infinity
     */
    result |= POPFLOAT_FP16_INFINITY;
  }

  return result;
}

float halfToSingle(uint16_t ihalf) {

  bool neg = POPFLOAT_FP16_IS_NEG(ihalf);
  uint32_t iresult = (neg ? (1U << POPFLOAT_FP32_SIGN_SHIFT) : 0);
  float result;

  uint32_t maskLen =
      POPFLOAT_NUM_FP32_MANTISSA_BITS - POPFLOAT_NUM_FP16_MANTISSA_BITS;

  uint32_t biasCorrection =
      POPFLOAT_FP32_EXPONENT_BIAS - POPFLOAT_FP16_EXPONENT_BIAS;
  if (POPFLOAT_FP16_IS_ZERO(ihalf)) {
    /* +- Zero
       - nothing more to do
     */

  } else if (POPFLOAT_FP16_IS_SUBNORM(ihalf)) {
    /* Subnormal values - represented as normalised values in single precision
     * format
     */
    uint32_t mant = POPFLOAT_FP16_MANTISSA(ihalf) << maskLen;
    int exp = 0;
    while ((mant & (1 << POPFLOAT_NUM_FP32_MANTISSA_BITS)) == 0) {
      exp -= 1;
      mant <<= 1;
    }

    mant &= ~(1 << POPFLOAT_NUM_FP32_MANTISSA_BITS);
    exp += (biasCorrection + 1);

    iresult = POPFLOAT_FP16_SIGN(ihalf) << POPFLOAT_FP32_SIGN_SHIFT;
    iresult |= mant;
    iresult |= (exp << POPFLOAT_FP32_EXPONENT_SHIFT);

  } else if (POPFLOAT_FP16_IS_INFINITY(ihalf)) {
    /* +- Infinity
     */
    iresult |= POPFLOAT_FP32_INFINITY;

  } else if (POPFLOAT_FP16_IS_QNAN(ihalf)) {
    /* +- qNaN
     */
    iresult = POPFLOAT_FP32_INFINITY;
    iresult |= (1 << POPFLOAT_FP32_Q_SHIFT);

  } else if (POPFLOAT_FP16_IS_SNAN(ihalf)) {
    /* +- sNaN
     */
    iresult = POPFLOAT_FP32_INFINITY;

    /* Mantissa must be non-zero but top mantissa bit must be zero
     */
    iresult |= 1;

  } else {
    /* Normalised value
     */
    iresult = POPFLOAT_FP16_SIGN(ihalf) << POPFLOAT_FP32_SIGN_SHIFT;
    iresult |= (POPFLOAT_FP16_MANTISSA(ihalf) << maskLen);
    iresult |= (POPFLOAT_FP16_EXPONENT(ihalf) + (biasCorrection))
               << POPFLOAT_FP32_EXPONENT_SHIFT;
  }

  std::memcpy(&result, &iresult, sizeof(result));

  return result;
}

} // end namespace popfloat
} // end namespace experimental
