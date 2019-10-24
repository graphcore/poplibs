#ifndef CAST_TO_GFLOAT32_SR_H
#define CAST_TO_GFLOAT32_SR_H

#define nIterations m0
#define srMaskBase m0
#define isBernoulli m0
#define enNanoo m0
#define mRoundMode m0

#define mRemainder m1
#define maskOut_0 m1
#define setMode m1

#define mBaseIn m2

#define mCastToGF32 m3

#define mGf32Param m4

#define enDenorm m5
#define mRoundOp m5
#define maskOut_1 m5

#define mCount m6

#define mBaseOut m7

#define mCorrParams m8

#define mRowCount m9

#define mInRow m10

#define mOutRow m11

#define maskOut0 a0

#define maskOut1 a1
#define constHalf a1

#define sgnMaskV2                                                              \
  a0:                                                                          \
  1
#define maskOut                                                                \
  a0:                                                                          \
  1
#define expV2                                                                  \
  a0:                                                                          \
  1
#define inValueV2                                                              \
  a0:                                                                          \
  1

#define isHalfMin0 a2
#define outV2_0 a2
#define biasCorr a2
#define constOne a2
#define oneMinCorrV2_0 a2
#define probBrnoulli a2

#define isHalfMin1 a3
#define fpMinNorm a3
#define outV2_1 a3
#define scaleCorr a3
#define oneMinCorrV2_1 a3

#define oneMinCorrV2                                                           \
  a2:                                                                          \
  3
#define corrDenorm                                                             \
  a2:                                                                          \
  3
#define clampCorr                                                              \
  a2:                                                                          \
  3
#define clampOut                                                               \
  a2:                                                                          \
  3
#define trncNorm                                                               \
  a2:                                                                          \
  3
#define dnrmManMaskV2                                                          \
  a2:                                                                          \
  3
#define isDenormV2                                                             \
  a2:                                                                          \
  3
#define isNanOrInf                                                             \
  a2:                                                                          \
  3
#define manLsbMaskV2                                                           \
  a2:                                                                          \
  3
#define sgnV2                                                                  \
  a2:                                                                          \
  3
#define srMaskV2                                                               \
  a2:                                                                          \
  3
#define outNanMaskV2                                                           \
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
#define tmpOutV2                                                               \
  a4:                                                                          \
  5
#define fpExpMaskV2                                                            \
  a4:                                                                          \
  5
#define fpHalfMinGF32                                                          \
  a4:                                                                          \
  5
#define fpNanMaskV2                                                            \
  a4:                                                                          \
  5

#define out0 a6
#define roundCorrV2_0 a6

#define out1 a7
#define roundCorrV2_1 a7

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
