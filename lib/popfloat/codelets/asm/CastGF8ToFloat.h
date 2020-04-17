// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef CAST_GF8_TO_FLOAT_H
#define CAST_GF8_TO_FLOAT_H

#define mInValue0 m0
#define mCastParams m0

#define mInValueV2 m1
#define mInValue1 m1

#define mManSh0 m2
#define mQuotient m2

#define mManSh1 m3

#define mCount m4
#define mTMemBase m4

#define mBiasCorr m5

#define mExpManMask m6
#define mWorkerIdx m6

#define mBaseOut m7

#define mBaseIn m8

#define mGF8Param m9

#define mRemainder m10

#define fpHalf a1

#define expMaskGF8                                                             \
  a0:                                                                          \
  1
#define fpClamp                                                                \
  a0:                                                                          \
  1
#define gf8DenormV2                                                            \
  a0:                                                                          \
  1

#define bf16ValueV2 a2

#define sgnMask a3

#define nanMaskV2                                                              \
  a2:                                                                          \
  3
#define sgnMaskV2                                                              \
  a2:                                                                          \
  3
#define sgnV2                                                                  \
  a2:                                                                          \
  3

#define inValue0 a4

#define inValue1 a5
#define gf8ValueV2 a5

#define gf32ValueV2                                                            \
  a4:                                                                          \
  5

#define gf32AlignScale a6
#define fpBiasCorr a6
#define outValueV2_0 a6

#define fpMinNorm                                                              \
  a6:                                                                          \
  7
#define outValueV2                                                             \
  a6:                                                                          \
  7
#define isZeroV2                                                               \
  a6:                                                                          \
  7
#endif
