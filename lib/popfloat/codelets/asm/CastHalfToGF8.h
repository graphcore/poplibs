// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef CAST_HALF_TO_GF8_H
#define CAST_HALF_TO_GF8_H

#define mBaseIn m0

#define mBaseOut m1

#define mGF16ManDiff m2

#define mCount m3
#define mTMemBase m4

#define mOutSign m4
#define mQuotient m4

#define mOut m5
#define mExpManV1 m5

#define mExpManV0 m6
#define mManExp m6

#define mGf8Param m7

#define mCastParams m8

#define mWorkerIdx m9

#define mRemainder m10

#define outFp8V2_0 a0

#define outFp8V2_1 a1

#define outExpV4                                                               \
  a0:                                                                          \
  1
#define manExpV4                                                               \
  a0:                                                                          \
  1

#define signV2_0 a2
#define halfExpMask a2
#define maxExp a2
#define scaleHalf a2

#define signV2_1 a3

#define halfExpMaskV4                                                          \
  a2:                                                                          \
  3
#define signV4                                                                 \
  a2:                                                                          \
  3

#define fp8Sgn0 a4

#define fp8Sgn1 a5

#define isMaxExpV4                                                             \
  a4:                                                                          \
  5
#define fp8SgnV4                                                               \
  a4:                                                                          \
  5

#define shufSgnHi a6

#define shufSgnLo a7

#define outValueV4                                                             \
  a6:                                                                          \
  7

#endif
