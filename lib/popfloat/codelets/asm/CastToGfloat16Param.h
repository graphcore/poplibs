#ifndef CAST_TO_GFLOAT16_PARAM_H
#define CAST_TO_GFLOAT16_PARAM_H

#define inNegMaxAbsExp m0
#define mZeroExp m0

#define mConstOne m1

#define mInScale m2
#define mfp8AlignShr m2

#define mAlignMinNorm m3
#define mExpBitsGF16 m3
#define mMinExp m3
#define mInFP32Param m3
#define m2PwrmManm10 m3

#define mManMaskFP16 m4

#define mMaxNormExp m5
#define mManExp m5

#define mManMaskFP32 m6
#define mAlignExp m6
#define srManMask m6

#define mExpBiasGF16 m7
#define mTruncMan m7
#define mMinNormExp m7

#define mEnInfGF16 m8
#define mEnableDnrm m8
#define clampManMask m8

#define mGF16Man m9
#define inMaxAbsExp m9
#define mInScaleRecip m9

#define mFloatStruct m10

#define mGf16Param m11

#define maxExp a0
#define sgnMask a0
#define fp16RecipScale a0

#define maxExpV4                                                               \
  a0:                                                                          \
  1

#define signMask a2
#define scalePmManm10 a2
#define scaleP10 a2
#define scalePm1 a2
#define expMask a2
#define signV2_0 a2
#define enNanoo a2

#define halfMinDnrm a3
#define fp16ScaleIn a3

#define f16constPair                                                           \
  a2:                                                                          \
  3
#define halfExpMaskV4                                                          \
  a2:                                                                          \
  3
#define signV4                                                                 \
  a2:                                                                          \
  3
#define halfMinDnrmV4                                                          \
  a2:                                                                          \
  3

#define enInfMask a4
#define fp16MaxNeg a4

#define fp16MaxPos a5

#define enNanooMaskV4                                                          \
  a4:                                                                          \
  5
#define qNanFP16V4                                                             \
  a4:                                                                          \
  5
#define fp16MaxClamp                                                           \
  a4:                                                                          \
  5

#define enNanooMask a6
#define twoPwrM10 a6
#define fp16ClampIn a6

#define enNanooGF16 a7

#define fp32ClampIn                                                            \
  a6:                                                                          \
  7
#define enNanooV4                                                              \
  a6:                                                                          \
  7

#endif
