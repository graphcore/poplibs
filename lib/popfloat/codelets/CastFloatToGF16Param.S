#ifdef __IPU__

#define castFloatToGf16Param        __runCodelet_popfloat__CastFloatToGf16Param

#include "GfloatConst.hpp"
#include "CastFloatToGF16Param.h"

.globl castFloatToGf16Param

.type castFloatToGf16Param      , @function

.section .text.castFloatToGf16Param
.align 8

castFloatToGf16Param:
  {
    ld32         $mGf32Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CALC_GFLOAT_PARAM_PTR_OFFSET;
    or           $expMask       , $azero                , POPFLOAT_FP32_EXPONENT_MASK
  }
  {
    ld32         $mFloatStruct  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET;
    f32v2add     $fpExpMaskV2   , $expMask:B            , $azeros
  }
  {
    st64         $fpExpMaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_FP32_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f32cmpeq     $expMask       , $azero                , $azero
  }
  ldz8         $mFloatParams  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_PARAMS_OFFSET;
  ldz8         $mGF32Exp      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET
  add          $manExpShift   , $mGF32Exp             , (POPFLOAT_NUM_FP32_MANTISSA_BITS - 15)
  setzi        $mNumBits      , 0x7FFF
  shl          $mNumBits      , $mNumBits             , $manExpShift
  st32         $mNumBits      , $mGf32Param           , $mzero            , POPFLOAT_FP32_TO_GF16_PARAM_MAN_EXP_MASK_OFFSET;
  st32         $mNumBits      , $mGf32Param           , $mzero            , POPFLOAT_FP32_TO_GF16_PARAM_MAN_EXP_MASK_OFFSET+1;
  lds8         $mMinNormGF32  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET
  add          $mGf32ExpAlign , $mMinNormGF32         , 128
  sub          $mMinNormGF32  , 128                   , $mMinNormGF32
  shl          $mMinNormGF32  , $mMinNormGF32         , 23
  st32         $mMinNormGF32  , $mGf32Param           , $mzero            , (POPFLOAT_FP32_TO_GF16_PARAM_MIN_NORM_OFFSET)
  shl          $mGf32ExpAlign , $mGf32ExpAlign        , 23
  st32         $mGf32ExpAlign , $mGf32Param           , $mzero            , (POPFLOAT_FP32_TO_GF16_PARAM_EXP_ALIGN_OFFSET)
  add          $mNumBits      , $mGF32Exp             , 8
  st32         $mNumBits      , $mGf32Param           , $mzero            , (POPFLOAT_FP32_TO_GF16_PARAM_FP16_SHR_ALIGN_OFFSET)
  exitz        $mzero
.size castFloatToGf16Param    , .-castFloatToGf16Param

#endif
