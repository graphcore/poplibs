#ifndef CAST_GF16_TO_FLOAT_PARAM_H
#define CAST_GF16_TO_FLOAT_PARAM_H

#define mGf16Input        m0
#define mGf16ExpMask      m0

#define mGF32Man          m1
#define setLoad           m1

#define mBit23            m2
#define minNorm           m2

#define enDenorm          m3
#define expMaskShl        m3
#define mManSh0           m3

#define mGF16Param        m4

#define mConstOne         m5

#define mGF32Exp          m7

#define mBiasCorr         m8
#define mManSh1           m8

#define isBfloat          m9
#define mGf32ExpAlign     m9

#define mEnInf            m10
#define maxNorm           m10

#define mFloatStruct      m11

#define expMask           a4

#define fpExpMaskV2       a4:5
#endif
