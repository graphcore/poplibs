#ifndef CAST_GF8_TO_HALF_PARAM_H
#define CAST_GF8_TO_HALF_PARAM_H

#define mConstOne         m1

#define mExpBitsGF16      m2

#define mManShr           m3

#define mTruncMan         m4

#define mEnInfGF16        m5

#define mGF16Man          m6

#define mFloatStruct      m7

#define mGF8Param         m9

#define expMask           a0
#define sgnMask           a0

#define halfExpMaskV4     a0:1

#define maxExp            a2

#define maxExpV4          a2:3

#define fpHalf            a4

#endif
