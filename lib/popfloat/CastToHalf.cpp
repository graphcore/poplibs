#include "popfloat/CastToGfloat.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <popfloat/GfloatExpr.hpp>
#include <popfloat/GfloatExprUtil.hpp>
#include <popfloat/CastToHalf.hpp>
#include "codelets/GfloatConst.hpp"

#include <unordered_set>
#include <cassert>
#include <cmath>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popfloat::gfexpr;

namespace popfloat {

static
uint32_t roundMantissaRN(uint32_t m_single,
                         uint32_t masklen) {
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

  result = SINGLE_SIGN(ivalue) << HALF_SIGN_SHIFT;
  exp = SINGLE_EXP(ivalue) - SINGLE_EXP_BIAS;

  int maskLen = SINGLE_MAN_SIZE - HALF_MAN_SIZE;

  if (exp < -25) {
    /* Very small values map to +-0
       - nothing more to do.
     */
  } else if (exp < -24) {
    /* Half the smallest denorm could round up to the smallest denorm.
     */
    uint32_t mant = SINGLE_MANT(ivalue)
                    | (1 << (SINGLE_MAN_SIZE + 1));
    mant = roundMantissaRN(mant,
                           (SINGLE_MAN_SIZE + 1));
    result |= (mant << HALF_MANT_SHIFT);
  } else if (exp < -14) {
    /* Small numbers map to denorms - will lose precision
     */

    /* Shift the exponent into the mantissa
     */
    int shift = -exp - (HALF_EXP_BIAS);

    /* Combine with the original mantissa shifted into place
     */
    uint16_t mant = 1 << ((HALF_MAN_SIZE - 1) - shift);
    mant += roundMantissaRN(SINGLE_MANT(ivalue),
                            ((maskLen)+
                             shift + 1));
    result |= (mant << HALF_MANT_SHIFT);

  } else if (exp <= 15) {
    /* Normal numbers - will lose precision
     */

    uint32_t mant = SINGLE_MANT(ivalue)
                    | (1 << (SINGLE_MAN_SIZE));
    mant = roundMantissaRN(mant, maskLen);
    if (mant >> (HALF_MAN_SIZE + 1)) {
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
      result |= (mant & HALF_MANT_MASK) << HALF_MANT_SHIFT;
      result |= ((exp + HALF_EXP_BIAS) <<
                 HALF_MAN_SIZE);
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
    uint16_t mant = roundMantissaRN(SINGLE_MANT(ivalue), maskLen);

    if (SINGLE_IS_QNAN(ivalue)) {
      mant |= (1 << HALF_Q_SHIFT);
    } else {
      mant &= ~(1 << HALF_Q_SHIFT);

      if (mant == 0) {
        /* Ensure NaNs stay as NaNs (non-zero mantissa)
         */
        mant |= 1;
      }

    }

    result |= HALF_INFINITY;
    result |= mant << HALF_MANT_SHIFT;

  } else {
    /* Infinity maps to infinity
     */
    result |= HALF_INFINITY;

  }

  return result;
}

float halfToSingle(uint16_t ihalf) {

  bool neg = HALF_IS_NEG(ihalf);
  uint32_t iresult = (neg ? (1U << SINGLE_SIGN_SHIFT) : 0);
  float result;

  uint32_t maskLen = SINGLE_MAN_SIZE - HALF_MAN_SIZE;

  uint32_t biasCorrection = SINGLE_EXP_BIAS - HALF_EXP_BIAS;
  if (HALF_IS_ZERO(ihalf)) {
    /* +- Zero
       - nothing more to do
     */

  } else if (HALF_IS_SUBNORM(ihalf)) {
    /* Subnormal values - represented as normalised values in single precision
     * format
     */
    uint32_t mant = HALF_MANT(ihalf) << maskLen;
    int exp = 0;
    while ((mant & (1 << SINGLE_MAN_SIZE)) == 0) {
      exp -= 1;
      mant <<= 1;
    }

    mant &= ~(1 << SINGLE_MAN_SIZE);
    exp += (biasCorrection + 1);

    iresult = HALF_SIGN(ihalf) << SINGLE_SIGN_SHIFT;
    iresult |= (mant << SINGLE_MANT_SHIFT);
    iresult |= (exp << SINGLE_EXP_SHIFT);

  } else if (HALF_IS_INFINITY(ihalf)) {
    /* +- Infinity
     */
    iresult |= SINGLE_INFINITY;

  } else if (HALF_IS_QNAN(ihalf)) {
    /* +- qNaN
     */
    iresult = SINGLE_INFINITY;
    iresult |= (1 << SINGLE_Q_SHIFT);

  } else if (HALF_IS_SNAN(ihalf)) {
    /* +- sNaN
     */
    iresult = SINGLE_INFINITY;

    /* Mantissa must be non-zero but top mantissa bit must be zero
     */
    iresult |= 1;

  } else {
    /* Normalised value
     */
    iresult = HALF_SIGN(ihalf) << SINGLE_SIGN_SHIFT;
    iresult |= (HALF_MANT(ihalf) << maskLen)
               << SINGLE_MANT_SHIFT;
    iresult |= (HALF_EXP(ihalf) +  (biasCorrection))
               << SINGLE_EXP_SHIFT;
  }

  std::memcpy(&result, &iresult, sizeof(result));

  return result;
}

}
