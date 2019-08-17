#ifdef __IPU__

// popfloat::CastToGfloat16<half>
// popfloat::CastToGloat16<float>

#define castGf8ToHalfParam           __runCodelet_popfloat__CastGf8ToHalfParam

#include "GfloatConst.hpp"
#include "CastGF8ToHalfParam.h"

.globl castGf8ToHalfParam

.type castGf8ToHalfParam         , @function

.section .text.castGf8ToHalfParam
.align 8

castGf8ToHalfParam:
  setzi        $mConstOne     , 1
  ld32         $mGF8Param     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CALC_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mFloatStruct  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET;
  ldz8         $mGF16Man      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET
  ldz8         $mExpBitsGF16  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET;
  {
    ldz8         $mEnInfGF16    , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_PARAMS_OFFSET
    setzi        $expMask       , POPFLOAT_FP16_EXPONENT_MASK
  }
  {
    and          $mEnInfGF16    , $mEnInfGF16           , POPFLOAT_GF_STRUCT_ENINF_MASK;
    sort4x16lo   $expMask       , $expMask              , $expMask
  }
  {
    shr          $mEnInfGF16    , $mEnInfGF16           , POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET;
    sort4x32lo   $halfExpMaskV4 , $halfExpMaskV4        , $halfExpMaskV4
  }
  st64         $halfExpMaskV4 , $mGF8Param            , $mzero            , (POPFLOAT_GF8_TO_FP16_PARAM_EXPONENT_MASK_OFFSET/2)
  sub          $mTruncMan     , POPFLOAT_NUM_FP16_MANTISSA_BITS, $mGF16Man
  sub          $mManShr       , POPFLOAT_NUM_FP16_EXPONENT_BITS, $mExpBitsGF16
  st32         $mManShr       , $mGF8Param            , $mzero            , (POPFLOAT_GF8_TO_FP16_PARAM_SHR_ALIGN_OFFSET);
  shl          $mTruncMan     , $mConstOne            , $mTruncMan
  sub          $mTruncMan     , $mTruncMan            , $mConstOne
  xnor         $mTruncMan     , $mTruncMan            , $mzero
  and          $mTruncMan     , $mTruncMan            , POPFLOAT_FP16_MANTISSA_MASK
  shl          $mExpBitsGF16  , $mConstOne            , $mExpBitsGF16
  sub          $mExpBitsGF16  , $mExpBitsGF16         , $mConstOne
  sub          $mExpBitsGF16  , $mExpBitsGF16         , $mEnInfGF16
  shl          $mExpBitsGF16  , $mExpBitsGF16         , POPFLOAT_NUM_FP16_MANTISSA_BITS
  or           $mExpBitsGF16  , $mExpBitsGF16         , $mTruncMan
  sort4x16lo   $mExpBitsGF16  , $mExpBitsGF16         , $mExpBitsGF16
  add          $mExpBitsGF16  , $mExpBitsGF16         , POPFLOAT_FP16_SIGN_MASK
  {
    st32         $mExpBitsGF16  , $mGF8Param            , $mzero            , (POPFLOAT_GF8_TO_FP16_PARAM_CLAMP_INPUT_OFFSET)
    setzi        $sgnMask       , POPFLOAT_FP8_V2_SIGN_MASK
  }
  sort4x16lo   $sgnMask       , $sgnMask              , $sgnMask
  {
    st32         $sgnMask       , $mGF8Param            , $mzero            , (POPFLOAT_GF8_TO_FP16_PARAM_SIGN_MASK_OFFSET)
    setzi        $maxExp        , 0x7800
  }
  f16v4add     $maxExpV4      , $maxExp:BL            , $azeros
  {
    st64         $maxExpV4      , $mGF8Param            , $mzero            , (POPFLOAT_GF8_TO_FP16_PARAM_MAX_EXPONENT_OFFSET/2);
    setzi        $fpHalf        , 0x3800
  }
  exitz        $mzero

.size castGf8ToHalfParam       , .-castGf8ToHalfParam

#endif
