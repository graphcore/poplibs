#ifdef __IPU__

#define castToGfloat32Param        __runCodelet_popfloat__CastToGfloat32Param

#include "GfloatConst.hpp"
#include "CastToGfloat32Param.h"

.globl castToGfloat32Param

.type castToGfloat32Param      , @function

.section .text.castToGfloat32Param
.align 8

castToGfloat32Param:
  {
    ld32         $mGf32Param    , $mvertex_base               , $mzero            , POPFLOAT_VBASE_CALC_GFLOAT_PARAM_PTR_OFFSET;
    or           $expMask       , $azero                      , POPFLOAT_FP32_EXPONENT_MASK
  }
  {
    ld32         $mFloatStruct  , $mvertex_base               , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET;
    f32v2add     $fpExpMaskV2   , $expMask:B                  , $azeros
  }
  ld32         $mGF32Man      , $mvertex_base               , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET
  ldz8         $mGF32Man      , $mFloatStruct               , $mzero            , POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET
  ldz8         $mGF32Exp      , $mFloatStruct               , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET;
  {
    setzi        $mConstOne     , 1;
    setzi        $signMask      , POPFLOAT_FP16_SIGN_MASK
  }
  {
    sub          $mManMask      , POPFLOAT_NUM_FP32_MANTISSA_BITS      , $mGF32Man;
    roll16       $signMask      , $azero                      , $signMask
  }
  {
    st64         $fpExpMaskV2   , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2);
    f32v2mul     $sgnV2         , $signMask:B                 , $azeros
  }
  {
    st64         $sgnV2         , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    or64         $sgnExpMaskV2  , $sgnV2                      , $fpExpMaskV2
  }
  {
    st64         $sgnExpMaskV2  , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_EXP_MASK_OFFSET/2);
    f32v2sub     $qNanV2        , $fpExpMaskV2                , $fpExpMaskV2
  }
  st64         $qNanV2        , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_QNAN_MASK_OFFSET/2);
  lds8         $mMinNormGF32  , $mFloatStruct               , $mzero            , POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET;
  ldz8         $mFloatParams  , $mFloatStruct               , $mzero            , POPFLOAT_GF_STRUCT_PARAMS_OFFSET
  shl          $mManMask      , $mConstOne                  , $mManMask
  sub          $mManMask      , $mManMask                   , $mConstOne;
  xnor         $mManMask      , $mManMask                   , $mzero
  st32         $mManMask      , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_NORM_MANT_MASK_OFFSET);
  st32         $mManMask      , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_NORM_MANT_MASK_OFFSET+4)/4;
  {
    cmpeq        $mZeroExp      , $mGF32Exp                   , 0
    or           $bit23Mask     , $azero                      , (1 << POPFLOAT_NUM_FP32_MANTISSA_BITS)
  }
  {
    shl          $mManMask      , $mManMask                   , $mZeroExp;
    f32v2add     $bit23MaskV2   , $bit23Mask:B                , $azeros
  }
  st64         $bit23MaskV2   , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_BIT23_MASK_OFFSET/2);
  st32         $mManMask      , $mworker_base               , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET);
  st32         $mManMask      , $mworker_base               , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET)+1;
  or           $mFp32Man      , $mzero                      , (1 << POPFLOAT_NUM_FP32_MANTISSA_BITS)
  sub          $mFp32Man      , $mFp32Man                   , $mConstOne
  and          $mFp32Man      , $mFp32Man                   , $mManMask
  and          $mEnInf        , $mFloatParams               , POPFLOAT_GF_STRUCT_ENINF_MASK
  shr          $mEnInf        , $mEnInf                     , POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET
  shl          $mMaxExp       , $mConstOne                  , $mGF32Exp
  sub          $mMaxExp       , $mMaxExp                    , $mConstOne
  sub          $mMaxExp       , $mMaxExp                    , $mEnInf
  sub          $mMaxExp       , $mMaxExp                    , $mMinNormGF32
  add          $mMaxExp       , $mMaxExp                    , POPFLOAT_FP32_EXPONENT_BIAS
  shl          $mMaxExp       , $mMaxExp                    , POPFLOAT_NUM_FP32_MANTISSA_BITS
  or           $mMaxExp       , $mMaxExp                    , $mFp32Man
  st32         $mMaxExp       , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET)+1
  or           $mMaxExp       , $mMaxExp                    , POPFLOAT_FP32_SIGN_MASK
  st32         $mMaxExp       , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET)
  and          $enDenorm      , $mFloatParams               , POPFLOAT_GF_STRUCT_ENDENORM_MASK
  mul          $mMinGF32      , $mGF32Man                   , $enDenorm
  sub          $mMinNormGF32  , (1+POPFLOAT_FP32_EXPONENT_BIAS)  , $mMinNormGF32;    // 127+(1-bias)
  sub          $mMinGF32      , $mMinNormGF32               , $mMinGF32
  max          $mMinGF32      , $mMinGF32                   , 0
  sub          $mHalfMinGF32  , $mMinGF32                   , $mConstOne
  max          $mHalfMinGF32  , $mHalfMinGF32               , 0
  shl          $mMinGF32      , $mMinGF32                   , POPFLOAT_NUM_FP32_MANTISSA_BITS;
  shl          $mHalfMinGF32  , $mHalfMinGF32               , POPFLOAT_NUM_FP32_MANTISSA_BITS
  st32         $mMinGF32      , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET)
  st32         $mHalfMinGF32  , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET)
  st32         $mHalfMinGF32  , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET)+1
  cmpne        $mMinGF32      , $mMinGF32                   , $mzero
  and          $enDenorm      , $enDenorm                   , $mMinGF32         // if mMinGF32 is zero, don't bother with denorm mask
  st32         $enDenorm      , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET)
  shl          $mMinNormGF32  , $mMinNormGF32               , POPFLOAT_NUM_FP32_MANTISSA_BITS
  st32         $mMinNormGF32  , $mGf32Param                 , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET)
.LcastToGfloat32Param_end:
  exitz        $mzero
.size castToGfloat32Param        , .-castToGfloat32Param

#endif
