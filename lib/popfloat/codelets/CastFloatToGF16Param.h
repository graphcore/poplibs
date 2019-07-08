#ifndef CAST_FLOAT_TO_GFLOAT16_PARAM_H
#define CAST_FLOAT_TO_GFLOAT16_PARAM_H

#define mGF32StrOffset    m0

#define mNumBits          m1

#define manExpShift       m2

#define mFloatParams      m3

#define mFloatStruct      m4

#define mGF32Exp          m7

#define mMinNormGF32      m8

#define mGf16OutClass     m9

#define mGf32ExpAlign     m10

#define mGf32Param        m11

#define manExpMask        a0

#define manExpMaskV2      a0:1

#define expMask           a4

#define fpExpMaskV2       a4:5

#endif
