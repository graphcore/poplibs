// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef POPFLOAT_UTILS_H
#define POPFLOAT_UTILS_H

#include "GfloatConst.hpp"
#include "ipudef.h"
#include <array>
#include <cmath>
#include <poplar/IeeeHalf.hpp>

namespace popfloat {
namespace experimental {

inline float floatFromHalfBits(const uint16_t bitPattern) {
  return float(poplar::IeeeHalf::fromBits(bitPattern));
}

template <typename T1, typename T2, unsigned int VEC_LEN>
void vecAsUInt(T1 *vec, T2 *bitPattern) {
  static_assert(VEC_LEN * sizeof(T1) == sizeof(T2), "vecAsUInt: size mismatch");
  std::memcpy(bitPattern, vec, sizeof(T2));
}

template <typename T1, typename T2, unsigned int VEC_LEN>
void uintAsVec(T1 *vec, T2 bitPattern) {
  static_assert(VEC_LEN * sizeof(T1) == sizeof(T2), "uintAsVec: size mismatch");
  std::memcpy(vec, &bitPattern, sizeof(bitPattern));
}

template <typename T1, typename T2, unsigned VEC_LEN0, unsigned VEC_LEN1>
void vecToVec(T1 *vec0, T2 *vec1) {
  static_assert(VEC_LEN0 * sizeof(T1) == VEC_LEN1 * sizeof(T2),
                "vecToVec: size mismatch");
  std::memcpy(vec0, vec1, sizeof(T2));
}

template <typename T> void constV4copy(T *v4, T val) {
  for (unsigned i = 0; i != 4; ++i) {
    v4[i] = val;
  }
}

template <typename T> void constV2copy(T *v2, T val) {
  for (unsigned i = 0; i != 2; ++i) {
    v2[i] = val;
  }
}

inline uint64_t addF16v4(uint64_t in0, uint16_t in1) {
  short4 outBits;
  uintAsVec<short4, uint64_t, 1>(&outBits, in0);
  auto fpIn1 = floatFromHalfBits(in1);
  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float fpOut = floatFromHalfBits(outBits[idx]);
    fpOut += fpIn1;
    poplar::IeeeHalf hlfOut(fpOut);
    outBits[idx] = hlfOut.bit16();
  }
  uint64_t outVal;
  vecAsUInt<short4, uint64_t, 1>(&outBits, &outVal);
  return outVal;
}

inline uint64_t addF16v4(uint64_t in0, uint64_t in1) {
  short4 outBits, inBits0, inBits1;
  uintAsVec<short4, uint64_t, 1>(&outBits, 0);
  uintAsVec<short4, uint64_t, 1>(&inBits0, in0);
  uintAsVec<short4, uint64_t, 1>(&inBits1, in1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float fpIn = floatFromHalfBits(inBits0[idx]);
    float fpOut = floatFromHalfBits(inBits1[idx]);
    uint32_t fpInBits, fpOutBits;
    std::memcpy(&fpInBits, &fpIn, sizeof(fpInBits));
    std::memcpy(&fpOutBits, &fpOut, sizeof(fpOutBits));
    fpOut += fpIn;
    std::memcpy(&fpOutBits, &fpOut, sizeof(fpOutBits));
    poplar::IeeeHalf hlfOut(fpOut);
    outBits[idx] = hlfOut.bit16();
  }
  uint64_t outVal;
  vecAsUInt<short4, uint64_t, 1>(&outBits, &outVal);
  return outVal;
}

inline float2 addF32v2(float2 in0, float in1) {
  float2 Out;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  Out = in0 + in1;
#else
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = in0[idx] + in1;
  }
#endif
  return Out;
}

inline float2 addF32v2(float2 in0, float2 in1) {
  float2 Out;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  Out = in0 + in1;
#else
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = (float)in0[idx] + (float)in1[idx];
  }
#endif
  return Out;
}

inline uint64_t subF16v4(uint64_t in0, uint16_t in1) {
  short4 outBits;
  uintAsVec<short4, uint64_t, 1>(&outBits, in0);
  float fpIn1 = floatFromHalfBits(in1);
  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float fpOut = floatFromHalfBits(outBits[idx]);
    fpOut -= fpIn1;
    poplar::IeeeHalf hlfOut(fpOut);
    outBits[idx] = hlfOut.bit16();
  }
  uint64_t outVal;
  vecAsUInt<short4, uint64_t, 1>(&outBits, &outVal);
  return outVal;
}

inline uint64_t subF16v4(uint64_t in0, uint64_t in1) {
  short4 outBits, inBits0, inBits1;
  uintAsVec<short4, uint64_t, 1>(&outBits, 0);
  uintAsVec<short4, uint64_t, 1>(&inBits0, in0);
  uintAsVec<short4, uint64_t, 1>(&inBits1, in1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float fpIn = floatFromHalfBits(inBits0[idx]);
    float fpOut = floatFromHalfBits(inBits1[idx]);
    fpOut -= fpIn;
    poplar::IeeeHalf hlfOut(fpOut);
    outBits[idx] = hlfOut.bit16();
  }
  uint64_t outVal;
  vecAsUInt<short4, uint64_t, 1>(&outBits, &outVal);
  return outVal;
}

inline float2 subF32v2(float2 in0, float in1) {
  float2 Out;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  Out = in0 - in1;
#else
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = (float)in0[idx] - (float)in1;
  }
#endif
  return Out;
}

inline float2 subF32v2(float2 in0, float2 in1) {
  float2 Out;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  Out = in0 - in1;
#else
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = (float)in0[idx] - (float)in1[idx];
  }
#endif
  return Out;
}

inline uint64_t mulF16v4(uint64_t in0, uint16_t in1) {
  short4 outBits;
  uintAsVec<short4, uint64_t, 1>(&outBits, in0);
  auto fpIn1 = floatFromHalfBits(in1);
  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float fpOut = floatFromHalfBits(outBits[idx]);
    fpOut *= fpIn1;
    poplar::IeeeHalf hlfOut(fpOut);
    outBits[idx] = hlfOut.bit16();
  }
  uint64_t outVal;
  vecAsUInt<short4, uint64_t, 1>(&outBits, &outVal);
  return outVal;
}

inline uint64_t mulF16v4(uint64_t in0, uint64_t in1) {
  short4 outBits, inBits0, inBits1;
  uintAsVec<short4, uint64_t, 1>(&outBits, 0);
  uintAsVec<short4, uint64_t, 1>(&inBits0, in0);
  uintAsVec<short4, uint64_t, 1>(&inBits1, in1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float fpIn = floatFromHalfBits(inBits0[idx]);
    float fpOut = floatFromHalfBits(inBits1[idx]);
    fpOut *= fpIn;
    poplar::IeeeHalf hlfOut(fpOut);
    outBits[idx] = hlfOut.bit16();
  }
  uint64_t outVal;
  vecAsUInt<short4, uint64_t, 1>(&outBits, &outVal);
  return outVal;
}

inline float2 mulF32v2(float2 in0, float in1) {
  float2 Out;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  Out = in0 * in1;
#else
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = (float)in0[idx] * (float)in1;
  }
#endif
  return Out;
}

inline float2 mulF32v2(float2 in0, float2 in1) {
  float2 Out;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  Out = in0 * in1;
#else
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = (float)in0[idx] * (float)in1[idx];
  }
#endif
  return Out;
}

inline void compareF16v4Eq(uint64_t in0, uint64_t in1, uint64_t *isEqVec) {
  short4 eqMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  eqMask = (in0 == in1);
#else
  short4 vIn0, vIn1;
  vecAsUInt<uint64_t, short4, 1>(&in0, &vIn0);
  vecAsUInt<uint64_t, short4, 1>(&in1, &vIn1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    eqMask[idx] = 0;

    if (vIn0[idx] == vIn1[idx]) {
      eqMask[idx] = ~eqMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&eqMask, isEqVec);
}

inline void compareF32v2Eq(float2 in0, float2 in1, uint64_t *isEqVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 eqMask;
  eqMask = (in0 == in1);
  vecAsUInt<int2, uint64_t, 1>(&eqMask, isEqVec);
#else
  int32_t eqMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    eqMask[idx] = 0;
    float fpIn0 = (float)in0[idx];
    float fpIn1 = (float)in1[idx];
    if (fpIn0 == fpIn1) {
      eqMask[idx] = ~eqMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(eqMask, isEqVec);
#endif
}

inline void compareF32v2Eq(float2 in0, float in1, uint64_t *isEqVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 eqMask;
  eqMask = (in0 == in1);
  vecAsUInt<int2, uint64_t, 1>(&eqMask, isEqVec);
#else
  int32_t eqMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    eqMask[idx] = 0;
    if (in0[idx] == in1) {
      eqMask[idx] = ~eqMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(eqMask, isEqVec);
#endif
}

inline void compareF16v4Le(uint64_t in0, uint64_t in1, uint64_t *isLeVec) {
  short4 leMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  leMask = (in0 == in1);
#else
  short4 vIn0, vIn1;
  uintAsVec<short4, uint64_t, 1>(&vIn0, in0);
  uintAsVec<short4, uint64_t, 1>(&vIn1, in1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    leMask[idx] = 0;
    float fpIn0 = floatFromHalfBits(vIn0[idx]);
    float fpIn1 = floatFromHalfBits(vIn1[idx]);

    if ((float)fpIn0 <= (float)fpIn1) {
      leMask[idx] = ~leMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&leMask, isLeVec);
}

inline void compareF16v4Le(uint64_t in0, float in1, uint64_t *isLeVec) {
  short4 leMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  leMask = (in0 == in1);
#else
  short4 vIn0;
  uintAsVec<short4, uint64_t, 1>(&vIn0, in0);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    leMask[idx] = 0;
    float fpIn0 = floatFromHalfBits(vIn0[idx]);
    if ((float)fpIn0 <= in1) {
      leMask[idx] = ~leMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&leMask, isLeVec);
}

inline void compareF32v2Le(float2 in0, float2 in1, uint64_t *isLeVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 leMask;
  leMask = (in0 <= in1);
  vecAsUInt<int2, uint64_t, 1>(&leMask, isLeVec);
#else
  int32_t leMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    leMask[idx] = 0;
    float fpIn0 = in0[idx];
    float fpIn1 = in1[idx];
    if (fpIn0 <= fpIn1) {
      leMask[idx] = ~leMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(leMask, isLeVec);
#endif
}

inline void compareF32v2Le(float2 in0, float in1, uint64_t *isLeVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 leMask;
  leMask = (in0 <= in1);
  vecAsUInt<int2, uint64_t, 1>(&leMask, isLeVec);
#else
  int32_t leMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    leMask[idx] = 0;
    float fpIn0 = in0[idx];
    float fpIn1 = in1;
    if (fpIn0 <= fpIn1) {
      leMask[idx] = ~leMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(leMask, isLeVec);
#endif
}

inline void compareF16v4Lt(uint64_t in0, uint64_t in1, uint64_t *isLtVec) {
  short4 ltMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  ltMask = (in0 < in1);
#else
  short4 vIn0, vIn1;
  uintAsVec<short4, uint64_t, 1>(&vIn0, in0);
  uintAsVec<short4, uint64_t, 1>(&vIn1, in1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    ltMask[idx] = 0;

    float fpIn0 = floatFromHalfBits(vIn0[idx]);
    float fpIn1 = floatFromHalfBits(vIn1[idx]);

    if ((float)fpIn0 < (float)fpIn1) {
      ltMask[idx] = ~ltMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&ltMask, isLtVec);
}

inline void compareF16v4Lt(uint64_t in0, float in1, uint64_t *isLtVec) {
  short4 ltMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  ltMask = (in0 < in1);
#else
  short4 vIn0;
  uintAsVec<short4, uint64_t, 1>(&vIn0, in0);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    ltMask[idx] = 0;

    float fpIn0 = floatFromHalfBits(vIn0[idx]);

    if ((float)fpIn0 < in1) {
      ltMask[idx] = ~ltMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&ltMask, isLtVec);
}

inline void compareF32v2Lt(float2 in0, float2 in1, uint64_t *isLtVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 ltMask;
  ltMask = (in0 < in1);
  vecAsUInt<int2, uint64_t, 1>(&ltMask, isLtVec);
#else
  int32_t ltMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    ltMask[idx] = 0;
    float fpIn0 = in0[idx];
    float fpIn1 = in1[idx];
    if (fpIn0 < fpIn1) {
      ltMask[idx] = ~ltMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(ltMask, isLtVec);
#endif
}

inline void compareF32v2Lt(float2 in0, float in1, uint64_t *isLtVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 ltMask;
  ltMask = (in0 < in1);
  vecAsUInt<int2, uint64_t, 1>(&ltMask, isLtVec);
#else
  int32_t ltMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    ltMask[idx] = 0;
    float fpIn0 = in0[idx];
    float fpIn1 = in1;
    if (fpIn0 < fpIn1) {
      ltMask[idx] = ~ltMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(ltMask, isLtVec);
#endif
}

inline void compareF16v4Gt(uint64_t in0, uint64_t in1, uint64_t *isGtVec) {
  short4 gtMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  gtMask = (in0 > in1);
#else
  short4 vIn0, vIn1;
  uintAsVec<short4, uint64_t, 1>(&vIn0, in0);
  uintAsVec<short4, uint64_t, 1>(&vIn1, in1);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    gtMask[idx] = 0;

    float fpIn0 = floatFromHalfBits(vIn0[idx]);
    float fpIn1 = floatFromHalfBits(vIn1[idx]);

    if ((float)fpIn0 > (float)fpIn1) {
      gtMask[idx] = ~gtMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&gtMask, isGtVec);
}

inline void compareF16v4Gt(uint64_t in0, float in1, uint64_t *isGtVec) {
  short4 gtMask;
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  gtMask = (in0 < in1);
#else
  short4 vIn0;
  uintAsVec<short4, uint64_t, 1>(&vIn0, in0);

  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    gtMask[idx] = 0;

    float fpIn0 = floatFromHalfBits(vIn0[idx]);

    if ((float)fpIn0 > in1) {
      gtMask[idx] = ~gtMask[idx];
    }
  }
#endif
  vecAsUInt<short4, uint64_t, 1>(&gtMask, isGtVec);
}

inline void compareF32v2Gt(float2 in0, float2 in1, uint64_t *isGtVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 gtMask;
  gtMask = (in0 > in1);
  vecAsUInt<int2, uint64_t, 1>(&gtMask, isGtVec);
#else
  int32_t gtMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    gtMask[idx] = 0;
    float fpIn0 = in0[idx];
    float fpIn1 = in1[idx];
    if (fpIn0 > fpIn1) {
      gtMask[idx] = ~gtMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(gtMask, isGtVec);
#endif
}

inline void compareF32v2Gt(float2 in0, float in1, uint64_t *isGtVec) {
#ifdef POPFLOAT_ENABLE_IPU_VECTORISED_OP
  int2 gtMask;
  gtMask = (in0 > in1);
  vecAsUInt<int2, uint64_t, 1>(&gtMask, isGtVec);
#else
  int32_t gtMask[POPFLOAT_GF32_VEC_SIZE];
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    gtMask[idx] = 0;
    float fpIn0 = in0[idx];
    float fpIn1 = in1;
    if (fpIn0 > fpIn1) {
      gtMask[idx] = ~gtMask[idx];
    }
  }
  vecAsUInt<int32_t, uint64_t, 2>(gtMask, isGtVec);
#endif
}

inline uint64_t genQnanOverflowF16(uint64_t in0, float in1, uint64_t qnanV4) {
  uint64_t Out;
  uint64_t isGtVec, inVec, outVec;
  compareF16v4Gt(in0, in1, &isGtVec);
  inVec = in0 & (~isGtVec);
  outVec = qnanV4 & isGtVec;
  outVec = outVec | inVec;

  return outVec;
}

inline float2 genQnanOverflowF32(float2 in0, float in1, uint64_t qnanV2) {
  float2 Out;
  uint64_t isGtVec, inVec, outVec;
  compareF32v2Gt(in0, in1, &isGtVec);
  vecAsUInt<float2, uint64_t, 1>(&in0, &inVec);
  inVec = inVec & (~isGtVec);
  outVec = qnanV2 & isGtVec;
  outVec = outVec | inVec;
  uintAsVec<float2, uint64_t, 1>(&Out, outVec);
  return (Out);
}

inline uint64_t minF16v4(uint64_t in0, float in1) {
  uint64_t Out;
  short4 inBits, outBits;
  uintAsVec<short4, uint64_t, 1>(&inBits, in0);
  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float tmp = floatFromHalfBits(inBits[idx]);
    tmp = fmin((float)tmp, in1);
    poplar::IeeeHalf tmpHalf(tmp);
    outBits[idx] = tmpHalf.bit16();
  }
  vecAsUInt<short4, uint64_t, 1>(&outBits, &Out);
  return Out;
}

inline float2 minF32v2(float2 in0, float in1) {
  float2 Out;
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = fmin((float)in0[idx], in1);
  }
  return Out;
}

inline uint64_t maxF16v4(uint64_t in0, float in1) {
  uint64_t Out;
  short4 inBits, outBits;
  uintAsVec<short4, uint64_t, 1>(&inBits, in0);
  for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
    float tmp = floatFromHalfBits(inBits[idx]);
    tmp = fmax((float)tmp, in1);
    poplar::IeeeHalf tmpHalf(tmp);
    outBits[idx] = tmpHalf.bit16();
  }
  vecAsUInt<short4, uint64_t, 1>(&outBits, &Out);
  return Out;
}

inline float2 maxF32v2(float2 in0, float in1) {
  float2 Out;
  for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
    Out[idx] = fmax((float)in0[idx], in1);
  }
  return Out;
}

inline uint64_t clipF16v4(uint64_t in0, float maxOut) {
  uint64_t outVec = minF16v4(in0, maxOut);

  return outVec;
}

inline float2 clipF32v2(float2 in0, float maxOut) {
  return minF32v2(in0, maxOut);
}

} // end namespace experimental
} // end namespace popfloat
#endif
