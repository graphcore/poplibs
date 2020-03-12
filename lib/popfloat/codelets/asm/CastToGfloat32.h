// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef CAST_TO_GFLOAT32_H
#define CAST_TO_GFLOAT32_H

#define mCastToGF32 m0

#define enHalfMin m1
#define mRoundMode m1

#define mRemainder m2
#define srManMask m2
#define setMode m2

#define mBaseIn m3

#define mGf32Param m4

#define enDenorm m5
#define mRoundOp m5

#define mCount m6

#define mBaseOut m7

#define enNanoo m8

#define mRowCount m9

#define mInRow m10

#define mOutRow m11

#define expV2                                                                  \
  a0:                                                                          \
  1
#define inValueV2                                                              \
  a0:                                                                          \
  1

#define isHalfMin0 a2
#define outV2_0 a2

#define isHalfMin1 a3
#define fpMinNorm a3
#define outV2_1 a3

#define bit23MaskV2                                                            \
  a2:                                                                          \
  3
#define dnrmManMaskV2                                                          \
  a2:                                                                          \
  3
#define isDenormV2                                                             \
  a2:                                                                          \
  3
#define isHalfMinV2                                                            \
  a2:                                                                          \
  3
#define isNanOrInf                                                             \
  a2:                                                                          \
  3
#define isPositiveV2                                                           \
  a2:                                                                          \
  3
#define manLsbMaskV2                                                           \
  a2:                                                                          \
  3
#define sgnV2                                                                  \
  a2:                                                                          \
  3
#define tmpV2                                                                  \
  a2:                                                                          \
  3
#define outNanMaskV2                                                           \
  a2:                                                                          \
  3
#define srExpV2                                                                \
  a2:                                                                          \
  3
#define outV2                                                                  \
  a2:                                                                          \
  3

#define minValueGF32 a4
#define isHalfMin a4

#define fpClampPos a5

#define fpClamp                                                                \
  a4:                                                                          \
  5
#define fpExpMaskV2                                                            \
  a4:                                                                          \
  5
#define isTieV2                                                                \
  a4:                                                                          \
  5
#define fpHalfMinGF32                                                          \
  a4:                                                                          \
  5
#define randomBitsV2                                                           \
  a4:                                                                          \
  5
#define tmpMaskV2                                                              \
  a4:                                                                          \
  5
#define tmpOutV2                                                               \
  a4:                                                                          \
  5
#define srMaskV2                                                               \
  a4:                                                                          \
  5
#define halfMinMaskV2                                                          \
  a4:                                                                          \
  5

#define out0 a6

#define out1 a7

#define nanInfValue                                                            \
  a6:                                                                          \
  7
#define nonZeroV4                                                              \
  a6:                                                                          \
  7
#define outBitMaskV2                                                           \
  a6:                                                                          \
  7
#define roundCorrV2                                                            \
  a6:                                                                          \
  7
#define sgnExpMaskV2                                                           \
  a6:                                                                          \
  7
#define qNanV2                                                                 \
  a6:                                                                          \
  7
#define gf32Max                                                                \
  a6:                                                                          \
  7

#endif
