// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef CAST_GF8_TO_HALF_H
#define CAST_GF8_TO_HALF_H

#define mBaseIn m0

#define mBaseOut m1

#define mF8SignMask m2

#define mManShr m3
#define mQuotient m3

#define mSignValueV4 m4
#define mSignV2_1 m4

#define mSignV2_0 m5

#define mCount m6
#define mTMemBase m6

#define mInValueV2_1 m7
#define mInValueV4 m7

#define mInValueV2_0 m8

#define mGF8Param m9

#define mWorkerIdx m10

#define mRemainder m11

#define outF16V4_0 a0

#define outF16V4_1 a1

#define scale                                                                  \
  a0:                                                                          \
  1
#define outF16V4                                                               \
  a0:                                                                          \
  1
#define signManV4                                                              \
  a0:                                                                          \
  1

#define scaleHalf a2

#define halfExpMaskV4                                                          \
  a2:                                                                          \
  3
#define isMaxExpV4                                                             \
  a2:                                                                          \
  3

#define fpHalf a4
#define maxExp a4

#define inputClampF16 a5

#define maxExpV4                                                               \
  a4:                                                                          \
  5

#define inValueV4                                                              \
  a6:                                                                          \
  7

#endif
