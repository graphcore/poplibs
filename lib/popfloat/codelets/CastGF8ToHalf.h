#ifndef CAST_GF8_TO_HALF_H
#define CAST_GF8_TO_HALF_H

#define mInRow m0

#define mOutRow m1
#define mFP8Load m1

#define mF8SignMask m2

#define mManShr m3
#define mFinalCount m3

#define mSignValueV4 m4
#define mSignV2_1 m4
#define mBaseIn m4

#define mBaseOut m5
#define mSignV2_0 m5
#define mRemainder m5

#define mCount m6

#define mInValueV2_1 m7
#define mInValueV4 m7

#define mInValueV2_0 m8
#define mFP8Input m8

#define mGF8Param m9

#define mRowCount m10

#define mCastToHalf m11

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
