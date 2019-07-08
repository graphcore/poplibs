#ifdef __IPU__

#define castHalfToGf8Param             __runCodelet_popfloat__CastHalfToGf8Param

#include "GfloatConst.hpp"
#include "CastHalfToGF8Param.h"

.globl castHalfToGf8Param

.type castHalfToGf8Param            , @function

.section .text.castHalfToGf8Param
.align 8

castHalfToGf8Param:
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CALC_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mFloatStruct  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET;
  ldz8         $mExpBitsGF16  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET
  add          $mfp8AlignShr  , $mExpBitsGF16         , (8-POPFLOAT_NUM_FP16_EXPONENT_BITS)
  st32         $mfp8AlignShr  , $mGf16Param           , $mzero            , (POPFLOAT_FP16_TO_GF8_PARAM_SHR_ALIGN_OFFSET)
  ldz8         $mGF16Man      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET
  add          $mGF16Man      , $mGF16Man             , (POPFLOAT_NUM_FP16_MANTISSA_BITS - 7)
  setzi        $mManExp       , 0x7F
  shl          $mManExp       , $mManExp              , $mGF16Man
  //setzi        $mGF16Man      , POPFLOAT_FP16_SIGN_MASK
  //or           $mManExp       , $mManExp              , $mGF16Man
  sort4x16lo   $mManExp       , $mManExp              , $mManExp
  st32         $mManExp       , $mGf16Param           , $mzero            , (POPFLOAT_FP16_TO_GF8_PARAM_MAN_EXP_MASK_OFFSET)
  st32         $mManExp       , $mGf16Param           , $mzero            , (POPFLOAT_FP16_TO_GF8_PARAM_MAN_EXP_MASK_OFFSET+1)
  exitz        $mzero
.size castHalfToGf8Param             , .-castHalfToGf8Param

#endif
