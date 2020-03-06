// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#ifndef CAST_TO_GFLOAT16_SR_H
#define CAST_TO_GFLOAT16_SR_H

#define mBaseIn m0

#define mBaseOut m1

#define mCorrParams m2

#define mRemainder m3
#define maskOut_0 m3
#define setMode m3

#define mCount m4
#define srMaskBase m4
#define mRoundOp m4

#define nIterations m3
#define enNanoo m5

#define maskOut_1 m6
#define mRoundMode m6

#define mGf16Param m7

#define mCastToGF16 m8

#define mRowCount m9

#define mInRow m10

#define mOutRow m11

#define scaleFloat a0
#define scaleClamp a0
#define maskOut0 a0
#define probBrnoulli a0
#define oneMinCorrV4_0 a0

#define maskOut1 a1
#define scalePair a1
#define oneMinCorrV4_1 a1

#define scale                                                                  \
  a0:                                                                          \
  1
#define maskOut                                                                \
  a0:                                                                          \
  1
#define outBitMaskV4                                                           \
  a0:                                                                          \
  1
#define zeroOutMaskV4                                                          \
  a0:                                                                          \
  1
#define outNanMaskV4                                                           \
  a0:                                                                          \
  1
#define oneMinCorrV4                                                           \
  a0:                                                                          \
  1
#define srMaskV4                                                               \
  a0:                                                                          \
  1

#define scalePmManm10 a2
#define scaleP10 a2
#define inputClampF16 a2
#define halfMinDnrm a2
#define scaleCorr a2
#define clampCorr a2

#define clampOut                                                               \
  a2:                                                                          \
  3
#define trncNorm                                                               \
  a2:                                                                          \
  3
#define clampInput                                                             \
  a2:                                                                          \
  3
#define expV4                                                                  \
  a2:                                                                          \
  3
#define halfExpMaskV4                                                          \
  a2:                                                                          \
  3
#define inputClampF32                                                          \
  a2:                                                                          \
  3
#define manLsbMaskV4                                                           \
  a2:                                                                          \
  3
#define signV4                                                                 \
  a2:                                                                          \
  3
#define halfMinDnrmV4                                                          \
  a2:                                                                          \
  3
#define outValueV2_0                                                           \
  a2:                                                                          \
  3

#define scaledMin a4
#define scaledClamp a4
#define roundCorrV4_0 a4

#define roundCorrV4_1 a5

#define inValueV2_0                                                            \
  a4:                                                                          \
  5
#define inValueV4                                                              \
  a4:                                                                          \
  5
#define isDenormV4                                                             \
  a4:                                                                          \
  5
#define isMaxExpV4                                                             \
  a4:                                                                          \
  5
#define isPositiveV4                                                           \
  a4:                                                                          \
  5
#define isNegativeV4                                                           \
  a4:                                                                          \
  5
#define nanInfValue                                                            \
  a4:                                                                          \
  5
#define randomBitsV4                                                           \
  a4:                                                                          \
  5
#define nonZeroV4                                                              \
  a4:                                                                          \
  5
#define roundCorrV4                                                            \
  a4:                                                                          \
  5
#define qNanV4                                                                 \
  a4:                                                                          \
  5

#define scalePm1 a6
#define outValueV4_0 a6

#define outValueV4_1 a7

#define inValueV2_1                                                            \
  a6:                                                                          \
  7
#define outValueV2_1                                                           \
  a6:                                                                          \
  7
#define outValueV4                                                             \
  a6:                                                                          \
  7

#define inValueF32V4                                                           \
  a4:                                                                          \
  7

#endif
