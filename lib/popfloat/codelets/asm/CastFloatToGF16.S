// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#ifdef __IPU__

// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastFloatToGF16.h"

.macro CAST_FLOAT_TO_GF16 FORMAT
.section .text.castFloatToGf16_\FORMAT\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastFloatToGf16___popfloat__experimental__FormatType__\FORMAT\()
  .type __runCodelet_popfloat__experimental__CastFloatToGf16___popfloat__experimental__FormatType__\FORMAT\(), @function
  __runCodelet_popfloat__experimental__CastFloatToGf16___popfloat__experimental__FormatType__\FORMAT\():

  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET+1
  add          $mRowCount     , $mRowCount            , -1
1:
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  {
    ld32         $mCount        , $mzero                , $mBaseIn          , 0;
    or           $signMask      , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2mul     $sgnMaskV2     , $signMask:B           , $azeros
  }
  add          $mCount        , $mCount               , 1
  shr          $mCount        , $mCount               , 1
  andc64       $outValueV2    , $inValueV2            , $sgnMaskV2
  and64        $sgnV2         , $inValueV2            , $sgnMaskV2;
.ifc \FORMAT, BFLOAT16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_BFLOAT
.align 8
  {
    rpt          $mCount        , ((3f - 2f)/8) - 1;
    or64         $outValueV2    , $inValueV2            , $azeros
  }
2:
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=        , 1;
    sort4x16hi   $outValue0     , $outValue0            , $outValue1
  }
  {
    st32step     $outValue0     , $mzero                , $mOutRow+=      , 1;
    or64         $outValueV2    , $inValueV2            , $azeros
  }
3:
  {
    ld32         $mCount        , $mzero                , $mBaseIn          , 0;
    sort4x16hi   $outValue0     , $outValue0            , $outValue1
  }
  ldb16        $outValue1     , $mzero                , $mOutRow        , 1
  {
    and          $mCount        , $mCount               , 1
    roll16       $outValue0     , $outValue0            , $outValue1
  }
  brz          $mCount        , 5f
  st32step     $outValue0     , $mzero                , $mOutRow+=      , 1
#else
.error "GF16_BFLOAT not enabled"
#endif
.else
  add          $mCount        , $mCount               , -1
.ifc \FORMAT, NO___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_NO_DENORM
  ld64         $manExpMaskV2  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_BITS_MASK_OFFSET/2)
4:
  {
    ld32         $biasCorrection, $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET);
    sort4x16hi   $sgnF16V2      , $sign0                , $sign1
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2mul     $outValueV2    , $biasCorrection:B     , $outValueV2
  }
  and64        $outValueV2    , $outValueV2           , $manExpMaskV2
  atom         $mOutValue0    , $outValue0
  atom         $mOutValue1    , $outValue1;
  ld32         $mOutShr       , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_SHR_ALIGN_OFFSET);
  shr          $mOutValue0    , $mOutValue0           , $mOutShr;
  shr          $mOutValue1    , $mOutValue1           , $mOutShr;
  {
    sort4x16lo   $outManExp     , $mOutValue0           , $mOutValue1;
    or           $signMask      , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
  {
    atom         $mOutSgn       , $sgnF16V2;
    f32v2mul     $sgnMaskV2     , $signMask:B           , $azeros
  }
  {
    or           $mOutV2        , $mOutSgn              , $outManExp;
    andc64       $outValueV2    , $inValueV2            , $sgnMaskV2
  }
  brz          $mCount        , 4f
  {
    st32step     $mOutV2        , $mzero                , $mOutRow+=        , 1;
    and64        $sgnV2         , $inValueV2            , $sgnMaskV2
  }
  brnzdec      $mCount        , 4b
#else
.error "GF16_N_DENORM not enabled"
#endif
.else
.ifc \FORMAT, ENABLE___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_EN_DENORM
4:
  {
    ld32         $fpMinNorm     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET);
    sort4x16hi   $sgnF16V2      , $sign0                , $sign1
  }
  {
    atom         $mOutSgn       , $sgnF16V2;
    f32v2add     $outV2         , $fpMinNorm:B          , $outValueV2
  }
  {
    ld64         $fpExpMaskV2   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2);
    f32v2cmpgt   $isDenormV2    , $fpMinNorm:B          , $outValueV2
  }
  {
    ld32         $mOutShr       , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_SHR_ALIGN_OFFSET);
    andc64       $outV2         , $outV2                , $fpExpMaskV2;
  }
  and64        $outV2         , $outV2                , $isDenormV2
  {
    ld32         $biasCorrection, $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET);
    andc64       $outValueV2    , $outValueV2           , $isDenormV2
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2mul     $outValueV2    , $biasCorrection:B     , $outValueV2
  }
  {
    ld64         $manExpMaskV2  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_BITS_MASK_OFFSET/2);
    or64         $outValueV2    , $outValueV2           , $outV2
  }
  and64        $outValueV2    , $outValueV2           , $manExpMaskV2
  atom         $mOutValue0    , $outValue0;
  atom         $mOutValue1    , $outValue1;
  shr          $mOutValue0    , $mOutValue0           , $mOutShr;
  {
    shr          $mOutValue1    , $mOutValue1           , $mOutShr;
    or           $signMask      , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
  {
    sort4x16lo   $mOutValue0    , $mOutValue0           , $mOutValue1;
    f32v2mul     $sgnMaskV2     , $signMask:B           , $azeros
  }
  {
    or           $mOutV2        , $mOutSgn              , $mOutValue0;
    andc64       $outValueV2    , $inValueV2            , $sgnMaskV2
  }
  brz          $mCount        , 4f
  {
    st32step     $mOutV2        , $mzero                , $mOutRow+=        , 1;
    and64        $sgnV2         , $inValueV2            , $sgnMaskV2
  }
  brnzdec      $mCount        , 4b
#else
.error "GF16_EN_DENORM not enabled"
#endif
.else
.error "GF16 fromat not supported"
.endif // ENABLE___DENORM___GF16
.endif // NO___DENORM___GF16
4:
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  and          $mCount        , $mCount               , 1
  brz          $mCount        , 4f
  ld32         $mOutSgn       , $mzero                , $mOutRow          , 1
  roll16       $mOutV2        , $mOutSgn              , $mOutV2
  roll16       $mOutV2        , $mOutV2               , $mOutV2
4:
  st32step     $mOutV2        , $mzero                , $mOutRow+=        , 1
.endif // BFLOAT16
5:
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  brnzdec      $mRowCount     , 1b
  exitz        $mzero
.size castFloatToGf16_\FORMAT\(),\
  .-__runCodelet_popfloat__experimental__CastFloatToGf16___popfloat__experimental__FormatType__\FORMAT\()
.endm

CAST_FLOAT_TO_GF16 BFLOAT16
CAST_FLOAT_TO_GF16 NO___DENORM___GF16
CAST_FLOAT_TO_GF16 ENABLE___DENORM___GF16

#endif
