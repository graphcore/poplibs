#ifndef popfloat_codelets_vec_h
#define popfloat_codelets_vec_h

#include <array>
#include <cmath>
#include <poplar/IeeeHalf.hpp>
#include <popfloat/GfloatExpr.hpp>
#include "GfloatConst.hpp"

using namespace poplar;
using namespace popfloat::gfexpr;

namespace popfloat {
inline uint64_t gfloat16_nan_or_inf(uint64_t     inValue,
                                    uint64_t     halfExpMask,
                                    uint64_t     enNanooInf) {
  uint64_t isNanOrInf;
  isNanOrInf = (~inValue) & halfExpMask;
  popfloat::compareF16v4Eq(isNanOrInf,
                           0,
                           &isNanOrInf);
  isNanOrInf = enNanooInf & isNanOrInf;

  return isNanOrInf;
}
inline uint64_t gfloat16_dnrm_mask(uint64_t     inValue,
                                   uint64_t     exp,
                                   uint64_t     outBitsMask,
                                   uint64_t     halfExpMask,
                                   uint16_t     hlfPwr10,
                                   uint16_t     hlf2Pm10mMan) {
  uint64_t  isDnrm;
  popfloat::compareF16v4Eq(exp,
                           0,
                           &isDnrm);

  uint64_t manMask = outBitsMask;

  manMask = manMask & (~isDnrm);
  uint64_t dnrmMask = isDnrm & inValue;

  dnrmMask = popfloat::mulF16v4(dnrmMask,
                                hlfPwr10);
  dnrmMask = dnrmMask & halfExpMask;

  dnrmMask = popfloat::mulF16v4(dnrmMask,
                                hlf2Pm10mMan);
  uint16_t minDnrm = 1;
  dnrmMask = popfloat::subF16v4(dnrmMask,
                                minDnrm);

  dnrmMask = isDnrm  & ~dnrmMask;
  manMask  = manMask | dnrmMask;

  return manMask;
}

inline void gfloat16_correction_rn(uint64_t    &outCorr,
                                   uint64_t     inValue,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  uint64_t manLSB;

  outCorr  = ~manMask;

  uint16_t minDnrm = 1;
  manLSB = popfloat::addF16v4(outCorr,
                              minDnrm);

  manLSB = manMask & manLSB;

  uint16_t hlf2Pm1Bits;
  half     hlf2Pm1 = 0.5;
  popfloat::vecAsUInt<half, uint16_t, 1>(&hlf2Pm1, &hlf2Pm1Bits);
  outCorr = popfloat::mulF16v4(manLSB,
                               hlf2Pm1Bits);

  manLSB &= inValue;

  uint64_t truncBits;
  truncBits = inValue & ~manMask;

  uint64_t isTieVec;
  popfloat::compareF16v4Eq(truncBits,
                           outCorr,
                           &isTieVec);

  outCorr = (manLSB & isTieVec) | (outCorr & ~isTieVec);
  outCorr = outCorr | exp;
  outCorr = popfloat::subF16v4(outCorr,
                               exp);
}

inline void gfloat16_correction_ra(uint64_t    &outCorr,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  outCorr = ~manMask;

  uint16_t minDnrm = 1;
  outCorr = popfloat::addF16v4(outCorr,
                               minDnrm);
  outCorr = manMask & outCorr;

  uint16_t hlf2Pm1Bits;
  half     hlf2Pm1 = 0.5;
  popfloat::vecAsUInt<half, uint16_t, 1>(&hlf2Pm1, &hlf2Pm1Bits);
  outCorr = popfloat::mulF16v4(outCorr,
                               hlf2Pm1Bits);

  outCorr  = exp | outCorr;
  outCorr = popfloat::subF16v4(outCorr,
                               exp);
}

inline void gfloat16_correction_rd(uint64_t    &outCorr,
                                   uint64_t     inValue,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  uint64_t isNegVec = 0;
  popfloat::compareF16v4Le(inValue,
                           isNegVec,
                           &isNegVec);
  outCorr = ~(manMask) & isNegVec;
  outCorr = outCorr | exp;
  outCorr = popfloat::subF16v4(outCorr,
                               exp);
}

inline void gfloat16_correction_ru(uint64_t    &outCorr,
                                   uint64_t     inValue,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  uint64_t isPosVec = 0;
  popfloat::compareF16v4Gt(inValue,
                           isPosVec,
                           &isPosVec);
  outCorr = (~manMask) & isPosVec;
  outCorr = outCorr | exp;
  outCorr = popfloat::subF16v4(outCorr,
                               exp);
}

inline void gfloat16_correction_sr(uint64_t    &outCorr,
                                   uint64_t     manMask,
                                   uint64_t     exp,
                                   uint64_t     randBits) {
  outCorr = (~manMask) | exp;
  outCorr = popfloat::subF16v4(outCorr,
                               exp);
}

inline uint64_t gfloat16_correction(uint64_t                 inValue,
                                    uint64_t                 manMask,
                                    uint64_t                 exp,
                                    GfloatRoundType  RMODE) {
  uint64_t outCorr = 0;
  if (RMODE == GfloatRoundType::RN) {
    gfloat16_correction_rn(outCorr,
                           inValue,
                           manMask,
                           exp);
  } else if (RMODE == GfloatRoundType::RU) {
    gfloat16_correction_ru(outCorr,
                           inValue,
                           manMask,
                           exp);
  } else if (RMODE == GfloatRoundType::RD) {
    gfloat16_correction_rd(outCorr,
                           inValue,
                           manMask,
                           exp);
  } else if (RMODE == GfloatRoundType::RA) {
    gfloat16_correction_ra(outCorr,
                           manMask,
                           exp);
  } else if (RMODE >= GfloatRoundType::SR) {
    uint64_t randBits;
    gfloat16_correction_sr(outCorr,
                           manMask,
                           exp,
                           randBits);
  }
  return outCorr;
}

inline void gfloat32_correction_rn(float2      &outCorr,
                                   uint64_t     expMask,
                                   uint64_t     inValue,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  float2 expVec, lsbVec, truncVec;
  uint64_t corrVec, manLSB, inLSB, truncBits, isTie;

  popfloat::uintAsVec<float2, uint64_t, 1>(&expVec, exp);

  corrVec = exp | ~manMask;

  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  outCorr = popfloat::subF32v2(outCorr,
                               expVec);
  popfloat::vecAsUInt<float2, uint64_t, 1>(&outCorr, &corrVec);
  corrVec  = expMask & corrVec;
  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  outCorr = popfloat::mulF32v2(outCorr,
                               2.0);
  lsbVec  = popfloat::addF32v2(expVec,
                               outCorr);

  popfloat::vecAsUInt<float2, uint64_t, 1>(&lsbVec, &manLSB);
  manLSB = manLSB & ~expMask;

  truncBits = inValue & ~manMask;
  truncBits = truncBits | exp;
  popfloat::uintAsVec<float2, uint64_t, 1>(&truncVec, truncBits);

  truncVec = popfloat::subF32v2(truncVec,
                                expVec);
  popfloat::vecAsUInt<float2, uint64_t, 1>(&truncVec, &truncBits);
  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  popfloat::compareF32v2Eq(truncVec,
                           outCorr,
                           &isTie);
  corrVec  = (manLSB & isTie) | (corrVec & ~isTie);
  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
}

inline void gfloat32_correction_ra(float2      &outCorr,
                                   uint64_t     expMask,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  float2 expVec;
  uint64_t corrVec;

  corrVec  = exp | ~manMask;
  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  popfloat::uintAsVec<float2, uint64_t, 1>(&expVec, exp);
  outCorr -= expVec;
  popfloat::vecAsUInt<float2, uint64_t, 1>(&outCorr, &corrVec);
  corrVec  = expMask & corrVec;
  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
}

inline void gfloat32_correction_rd(float2      &outCorr,
                                   uint64_t     inValue,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  uint64_t isPosVec;
  uint64_t corrVec;
  float2 expVec;

  float2 zeroVec, inVec;
  popfloat::uintAsVec<float2, uint64_t, 1>(&zeroVec, 0);
  popfloat::uintAsVec<float2, uint64_t, 1>(&inVec, inValue);
  popfloat::compareF32v2Le(zeroVec,
                           inVec,
                           &isPosVec);
  corrVec  = ~(manMask | isPosVec);
  corrVec  = corrVec | exp;

  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  popfloat::uintAsVec<float2, uint64_t, 1>(&expVec, exp);

  outCorr -= expVec;
}

inline void gfloat32_correction_ru(float2      &outCorr,
                                   uint64_t     inValue,
                                   uint64_t     manMask,
                                   uint64_t     exp) {
  uint64_t isPosVec;
  uint64_t corrVec;
  float2 expVec;

  float2 zeroVec, inVec;
  popfloat::uintAsVec<float2, uint64_t, 1>(&zeroVec, 0);
  popfloat::uintAsVec<float2, uint64_t, 1>(&inVec, inValue);
  popfloat::compareF32v2Le(zeroVec,
                           inVec,
                           &isPosVec);
  corrVec  = (~manMask) & isPosVec;
  corrVec  = corrVec | exp;

  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  popfloat::uintAsVec<float2, uint64_t, 1>(&expVec, exp);

  outCorr -= expVec;
}

inline void gfloat32_correction_dr(float2                  &outCorr,
                                   uint64_t                 expMask,
                                   uint64_t                 inValue,
                                   uint64_t                 manMask,
                                   uint64_t                 exp,
                                   GfloatRoundType  RMODE) {
  if (RMODE == GfloatRoundType::RN) {
    gfloat32_correction_rn(outCorr,
                           expMask,
                           inValue,
                           manMask,
                           exp);
  } else if (RMODE == GfloatRoundType::RU) {
    gfloat32_correction_ru(outCorr,
                           inValue,
                           manMask,
                           exp);
  } else if (RMODE == GfloatRoundType::RD) {
    gfloat32_correction_rd(outCorr,
                           inValue,
                           manMask,
                           exp);
  } else if (RMODE == GfloatRoundType::RA) {
    gfloat32_correction_ra(outCorr,
                           expMask,
                           manMask,
                           exp);
  }
}
inline void gfloat32_correction_sr(float2      &outCorr,
                                   uint64_t     manMask,
                                   uint64_t     exp,
                                   uint64_t     randBits,
                                   float        minValue) {
  uint64_t corrVec;
  float2 expVec;

  corrVec = (~manMask & randBits);
  corrVec = corrVec | exp;
  popfloat::uintAsVec<float2, uint64_t, 1>(&outCorr, corrVec);
  popfloat::uintAsVec<float2, uint64_t, 1>(&expVec, exp);
  outCorr -= expVec;
}

template<typename T>
int msb(T     bitPattern) {
  //do a (31-clz(bitPattern))
  int msbIdx = 0;
  T tmpMsk = bitPattern;
  while (tmpMsk >>= 1) {
    ++msbIdx;
  }
  return msbIdx;
}

template<typename T>
void  gfloat_mantissa_rz(T        &mant,
                         int       masklen) {
  mant >>= masklen;
  mant <<= masklen;
}

template<typename T>
void  gfloat_mantissa_rn(T        &mant,
                         int       masklen) {
  if (masklen > 0) {
    T lsBitVal  = (mant >> masklen) & 1;
    T msfBitVal = (mant >> (masklen - 1)) & 1;

    mant  += ((msfBitVal != 0) && (lsBitVal != 0)) ? (1 << masklen) : 0;
    gfloat_mantissa_rz<T>(mant, masklen);
  }
}

template<typename T>
void gfloat_mantissa_ra(T         &mant,
                        int        masklen) {
  mant  += ((1 << masklen) - 1);
  gfloat_mantissa_rz<T>(mant, masklen);
}

template<typename T>
void gfloat_mantissa_rd(T         &mant,
                        int        masklen,
                        bool       fpSigned) {
  mant  += (fpSigned) ? ((1 << masklen) - 1) : 0;
  gfloat_mantissa_rz<T>(mant, masklen);
}

template<typename T>
void gfloat_mantissa_ru(T         &mant,
                        int        masklen,
                        bool       fpSigned) {
  mant  += (fpSigned) ? 0 : ((1 << masklen) - 1);
  gfloat_mantissa_rz<T>(mant, masklen);
}

template<typename T1>
void gfloat_mantissa_sr(T1      &fpMant,
                        T1       srBits,
                        int      masklen) {
  fpMant = fpMant + srBits;
  fpMant = fpMant >> masklen;
  fpMant = fpMant << masklen;
}

}
#endif
