#ifndef CAST_FLOAT_TO_GF16_H
#define CAST_FLOAT_TO_GF16_H

#define mOutShr           m0
#define outManExp         m0

#define mOutV2            m1
#define mOutValue0        m1

#define mOutValue1        m2

#define mRemainder        m3

#define mCount            m4

#define mOutSgn           m5

#define mBaseIn           m6

#define mGf16Param        m7

#define mBaseOut          m8

#define mRowCount         m9

#define mInRow            m10

#define mOutRow           m11

#define outValue0         a0

#define outValue1         a1
#define outValue1         a1
#define signMask          a1

#define outValueV2        a0:1

#define sign0             a2
#define sgnF16V2          a2

#define sign1             a3
#define fpMinNorm         a3

#define isDenormV2        a2:3
#define sgnV2             a2:3
#define sgnMaskV2         a2:3

#define biasCorrection    a4

#define inValueV2         a4:5
#define fpExpMaskV2       a4:5

#define manExpMaskV2      a6:7
#define outV2             a6:7

#endif
