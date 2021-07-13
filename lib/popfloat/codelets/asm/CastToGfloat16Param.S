// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#define castToGfloat16Param             __runCodelet_popfloat__experimental__CastToGfloat16Param

#include "CastToGfloat16Param.h"
#include "GfloatConst.hpp"
#include "poplar/StackSizeDefs.hpp"

.globl castToGfloat16Param

.type castToGfloat16Param           , @function

DEF_STACK_USAGE 0 castToGfloat16Param
.section .text.castToGfloat16Param
.align 8

castToGfloat16Param:
  {
    ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CALC_GFLOAT_PARAM_PTR_OFFSET;
    setzi        $expMask       , POPFLOAT_FP16_EXPONENT_MASK
  }
  {
    ld32         $mFloatStruct  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET;
    sort4x16lo   $expMask       , $expMask              , $expMask
  }
  sort4x32lo   $halfExpMaskV4 , $halfExpMaskV4        , $halfExpMaskV4
  setzi        $qnanLower     , 0x9c
  {
    st64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f16v4add     $qNanFP16V4    , $qnanLower:BL         , $azeros
  }
  or64           $qNanFP16V4    , $qNanFP16V4           , $halfExpMaskV4
  {
    st64         $qNanFP16V4    , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_QNAN_OUTPUT_OFFSET/2);
    setzi        $signMask      , 0xBC00
  }
  {
    setzi        $mConstOne     , 1;
    f16v4mul     $signV4        , $signMask:BL          , $azeros
  }
  {
    st64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    setzi        $scalePm1      , 0x3800
  }
  st32         $scalePm1      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M1_OFFSET);
  lds8         $mExpBiasGF16  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET
  ldz8         $mGF16Man      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET
  ldz8         $mExpBitsGF16  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET
  add          $mfp8AlignShr  , $mExpBitsGF16         , (POPFLOAT_NUM_FP16_MANTISSA_BITS - 7)
  setzi        $mManExp       , 0x7F
  shl          $mManExp       , $mManExp              , $mfp8AlignShr
  sort4x16lo   $mManExp       , $mManExp              , $mManExp
  st32         $mManExp       , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_BITS_MASK_OFFSET)
  st32         $mManExp       , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_BITS_MASK_OFFSET+1)
  add          $mfp8AlignShr  , $mExpBitsGF16         , (8-POPFLOAT_NUM_FP16_EXPONENT_BITS)
  st32         $mfp8AlignShr  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_SHR_ALIGN_OFFSET)
  sub          $mfp8AlignShr  , POPFLOAT_NUM_FP16_EXPONENT_BITS, $mExpBitsGF16
  st32         $mfp8AlignShr  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_UNPACK_SHR_ALIGN_OFFSET);
  ldz8         $mFloatStruct  , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_PARAMS_OFFSET
  shl          $mMaxNormExp   , $mConstOne            , $mExpBitsGF16
  sub          $mMaxNormExp   , $mMaxNormExp          , $mConstOne
  and          $mEnInfGF16    , $mFloatStruct         , POPFLOAT_GF_STRUCT_ENINF_MASK
  shr          $mEnInfGF16    , $mEnInfGF16           , POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET
  cmpeq        $mZeroExp      , $mExpBitsGF16         , 0
  movz         $mEnInfGF16    , $mZeroExp             , $mzero
  sub          $mMaxNormExp   , $mMaxNormExp          , $mEnInfGF16
  add          $mManMaskFP16  , $mMaxNormExp          , -POPFLOAT_FP16_EXPONENT_BIAS
  movz         $mMaxNormExp   , $mZeroExp             , $mConstOne
  sub          $mMaxNormExp   , $mMaxNormExp          , $mExpBiasGF16
  cmpult       $mAlignMinNorm , $mExpBitsGF16         , POPFLOAT_NUM_FP16_EXPONENT_BITS
  or           $mAlignMinNorm , $mAlignMinNorm        , $mEnInfGF16
  sub          $mInScale      , $mExpBiasGF16         , $mManMaskFP16
  add          $mAlignExp     , $mExpBiasGF16         , -POPFLOAT_FP16_EXPONENT_BIAS
  movz         $mInScale      , $mAlignMinNorm        , $mAlignExp
  sub          $mExpBiasGF16  , $mzero                , $mExpBiasGF16
  movz         $mMaxNormExp   , $mZeroExp             , $mExpBiasGF16
  and          $mEnableDnrm   , $mFloatStruct         , POPFLOAT_GF_STRUCT_ENDENORM_MASK
  setzi        $mMinNormExp   , 0x200
  shl          $mMinNormExp   , $mMinNormExp          , $mAlignMinNorm
  cmpeq        $mAlignMinNorm , $mAlignMinNorm        , 0
  sub          $mManMaskFP16  , $mManMaskFP16         , $mAlignMinNorm
  mul          $mMinExp       , $mEnableDnrm          , $mGF16Man
  shr          $mMinExp       , $mMinNormExp          , $mMinExp
  sort4x16lo   $mMinExp       , $mMinExp              , $mMinExp
  st32         $mMinExp       , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET)
  shr          $mMinNormExp   , $mMinNormExp          , $mGF16Man
  st32         $mMinNormExp   , $mworker_base         , $mzero            , 0
  ld32         $halfMinDnrm   , $mworker_base         , $mzero            , 0
  {
    sub          $m2PwrmManm10  , POPFLOAT_FP16_MAX_EXP , $mGF16Man;
    f16v4add     $halfMinDnrmV4 , $halfMinDnrm:BL       , $azeros
  }
  st64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2)
  shl          $m2PwrmManm10  , $m2PwrmManm10         , POPFLOAT_NUM_FP16_MANTISSA_BITS
  st32         $m2PwrmManm10  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M_MAN_10_OFFSET)
  sub          $mTruncMan     , POPFLOAT_NUM_FP16_MANTISSA_BITS, $mGF16Man
  shl          $mTruncMan     , $mConstOne            , $mTruncMan
  sub          $mTruncMan     , $mTruncMan            , $mConstOne
  xnor         $mTruncMan     , $mTruncMan            , $mzero
  add          $clampManMask  , $mManMaskFP16         , POPFLOAT_FP16_EXPONENT_BIAS
  shl          $clampManMask  , $clampManMask         , POPFLOAT_NUM_FP16_MANTISSA_BITS
  and          $mManMaskFP16  , $mTruncMan            , POPFLOAT_FP16_MANTISSA_MASK
  or           $clampManMask  , $mManMaskFP16         , $clampManMask
  sort4x16lo   $clampManMask  , $clampManMask         , $clampManMask
  add          $clampManMask  , $clampManMask         , POPFLOAT_FP16_SIGN_MASK
  st32         $clampManMask  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET)
  sort4x16lo   $mTruncMan     , $mTruncMan            , $mTruncMan
  {
    st32         $mTruncMan     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_NORM_MAN_MASK_OFFSET)
    setzi        $sgnMask       , POPFLOAT_FP8_V2_SIGN_MASK
  }
  {
    st32         $mTruncMan     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_NORM_MAN_MASK_OFFSET)+1
    sort4x16lo   $sgnMask       , $sgnMask              , $sgnMask
  }
  {
    ld32         $scalePmManm10 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M_MAN_10_OFFSET)
    setzi        $twoPwrM10     , 0x1400
  }
  {
    st32         $sgnMask       , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GP16_PARAM_GF8_SIGN_MASK_OFFSET)
    f16v2mul     $scalePmManm10 , $twoPwrM10            , $scalePmManm10
  }
  {
    st32         $scalePmManm10 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M_MAN_10_OFFSET)
    setzi        $scaleP10      , 0x6400
  }
  sort4x16lo   $scaleP10      , $scaleP10             , $twoPwrM10
  {
    st32         $scaleP10      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET);
    setzi        $scalePm1      , 0x3800
  }
  sub          $mInScaleRecip , POPFLOAT_FP32_EXPONENT_BIAS, $mInScale
  max          $mInScaleRecip , $mInScaleRecip        , $mzero
  min          $mInScaleRecip , $mInScaleRecip        , 0xFE
  add          $mInScale      , $mInScale             , POPFLOAT_FP32_EXPONENT_BIAS
  max          $mInScale      , $mInScale             , $mzero
  min          $mInScale      , $mInScale             , 0xFE
  shl          $mInScale      , $mInScale             , POPFLOAT_NUM_FP32_MANTISSA_BITS
  shl          $mInScaleRecip , $mInScaleRecip        , POPFLOAT_NUM_FP32_MANTISSA_BITS
  st32         $mInScale      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET)
  st32         $mInScaleRecip , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET)
  add          $inMaxAbsExp   , $mMaxNormExp          , POPFLOAT_FP32_EXPONENT_BIAS
  max          $inMaxAbsExp   , $inMaxAbsExp          , $mzero
  min          $inMaxAbsExp   , $inMaxAbsExp          , 0xFE
  shl          $inMaxAbsExp   , $inMaxAbsExp          , POPFLOAT_NUM_FP32_MANTISSA_BITS
  shl          $mManMaskFP16  , $mManMaskFP16         , $mZeroExp
  and          $mManMaskFP32  , $mManMaskFP16         , POPFLOAT_FP16_MANTISSA_MASK
  {
    shl          $mManMaskFP32  , $mManMaskFP16         , (POPFLOAT_NUM_FP32_MANTISSA_BITS-POPFLOAT_NUM_FP16_MANTISSA_BITS)
    setzi        $maxExp        , 0x7800
  }
  {
    or           $inMaxAbsExp   , $inMaxAbsExp          , $mManMaskFP32
    f16v4add     $maxExpV4      , $maxExp:BL            , $azeros
  }
  st64         $maxExpV4      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MAX_EXPONENT_OFFSET/2);
  or           $inNegMaxAbsExp, $inMaxAbsExp          , POPFLOAT_FP32_SIGN_MASK
  {
    st32         $inNegMaxAbsExp, $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET)
    setzi        $fp16MaxPos    , 0x7BFF
  }
  {
    st32         $inMaxAbsExp   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET)+1
    f16tof32     $fp16MaxPos    , $fp16MaxPos
  }
  {
    ld64         $fp32ClampIn   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET/2)
    or           $fp16MaxNeg    , $fp16MaxPos           , POPFLOAT_FP32_SIGN_MASK
  }
  {
    ld32         $fp16ScaleIn   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET)
    f32v2clamp   $fp32ClampIn   , $fp32ClampIn          , $fp16MaxClamp
  }
  {
    ld32         $fp16RecipScale, $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET)
    f32tof16     $fp16ScaleIn   , $fp16ScaleIn
  }
  {
    st32         $fp16ScaleIn   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET+1)
    f32v2tof16   $fp16ClampIn   , $fp32ClampIn
  }
  {
    st32         $fp16ClampIn   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET)
    f32tof16     $fp16RecipScale, $fp16RecipScale
  }
  st32         $fp16RecipScale, $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET+1)

.LcastToGfloat16Param_end:
  exitz        $mzero
.size castToGfloat16Param            , .-castToGfloat16Param

#endif
