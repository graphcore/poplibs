#ifndef CAST_TO_GFLOAT32_PARAM_H
#define CAST_TO_GFLOAT32_PARAM_H

#define mGF32Man m0
#define mGf16ExpMask m0

#define mNumBits m1
#define mGf32ExpAlign m1

#define enDenorm m2
#define mEnInf m2
#define mManMask m2
#define manExpShift m2
#define expMaskShl m2
#define mManSh0 m2

#define mMaxExp m3
#define mSaveAsFP16 m3

#define mFloatParams m4

#define mConstOne m5

#define mFp32Man m6
#define mMinGF32 m6
#define mZeroExp m6

#define mGF32Exp m7

#define mMinNormGF32 m8
#define mBiasCorr m8
#define mManSh1 m8

#define mHalfMinGF32 m9

#define mFloatStruct m10

#define mGf32Param m11

#define enNanooGF32 a1

#define bit23Mask a2
#define maxPosValue a2
#define enGf32InfMask a2

#define expBitsGF32 a3
#define signMask a3

#define bit23MaskV2                                                            \
  a2:                                                                          \
  3
#define maxExpV2                                                               \
  a2:                                                                          \
  3
#define sgnV2                                                                  \
  a2:                                                                          \
  3
#define qNanV2                                                                 \
  a2:                                                                          \
  3

#define biasCorrection a4
#define expMask a4
#define fpClampPos a4
#define minValueGF32 a4
#define enGF32Inf a4
#define minNormGF32 a4

#define fpHalfMinGF32 a5
#define halfMinGF32 a5
#define gf16ValueV2 a5

#define fpExpMaskV2                                                            \
  a4:                                                                          \
  5
#define minPairGF32                                                            \
  a4:                                                                          \
  5
#define fpNanMaskV2                                                            \
  a4:                                                                          \
  5

#define enNanooMask a7

#define sgnExpMaskV2                                                           \
  a6:                                                                          \
  7

#endif
