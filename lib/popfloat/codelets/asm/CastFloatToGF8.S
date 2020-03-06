// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#ifdef __IPU__

// popfloatCastFloatToGf8

#include "GfloatConst.hpp"
#include "CastFloatToGF8.h"

.section .text.castFloatToGf8
.align 4
  .globl __runCodelet_popfloat__experimental__CastFloatToGf8
  .type __runCodelet_popfloat__experimental__CastFloatToGf8, @function
  __runCodelet_popfloat__experimental__CastFloatToGf8:

  ld32         $mGf8Param     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
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
  add          $mCount        , $mCount               , 3
  shr          $mCount        , $mCount               , 2
  andc64       $outValueV2    , $inValueV2            , $sgnMaskV2
  and64        $sgnV2         , $inValueV2            , $sgnMaskV2;
  add          $mCount        , $mCount               , -1
4:
.align 8
  setzi        $mStackOffset  , 0
  rpt          2              , ((2f-1f)/8) - 1
1:
  {
    ld32         $fpMinNorm     , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET);
    sort4x16hi   $sgnF16V2      , $sign0                , $sign1
  }
  {
    atom         $mOutSgn       , $sgnF16V2;
    f32v2add     $outV2         , $fpMinNorm:B          , $outValueV2
  }
  {
    ld64         $fpExpMaskV2   , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2);
    f32v2cmpgt   $isDenormV2    , $fpMinNorm:B          , $outValueV2
  }
  {
    ld32         $mOutShr       , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_SHR_ALIGN_OFFSET);
    andc64       $outV2         , $outV2                , $fpExpMaskV2;
  }
  {
    nop;
    and64        $outV2         , $outV2                , $isDenormV2
  }
  {
    ld32         $biasCorrection, $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET);
    andc64       $outValueV2    , $outValueV2           , $isDenormV2
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2mul     $outValueV2    , $biasCorrection:B     , $outValueV2
  }
  {
    ld64         $manExpMaskV2  , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_BITS_MASK_OFFSET/2);
    or64         $outValueV2    , $outValueV2           , $outV2
  }
  {
    nop;
    and64        $outValueV2    , $outValueV2           , $manExpMaskV2
  }
  {
    atom         $mOutValue0    , $outValue0;
    fnop
  }
  {
    atom         $mOutValue1    , $outValue1;
    fnop
  }
  {
    shr          $mOutValue0    , $mOutValue0           , $mOutShr;
    fnop
  }
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
  {
    st32step     $mOutV2        , $mworker_base         , $mStackOffset+=   , 1;
    and64        $sgnV2         , $inValueV2            , $sgnMaskV2
  }
2:
  ld32         $mOutValue0    , $mworker_base         , $mzero            , 0;
  ld32         $mOutValue1    , $mworker_base         , $mzero            , 1;
  sort8x8hi    $mOutV2        , $mOutValue0           , $mOutValue1
  brz          $mCount        , 2f
  st32step     $mOutV2        , $mzero                , $mOutRow+=        , 1;
  brnzdec      $mCount        , 4b
2:
  and          $mCount        , $mCount               , 3
  brnz         $mCount        , 2f
  st32step     $mOutV2        , $mzero                , $mOutRow+=        , 1
  bri          4f
2:
  cmpult       $mRemainder    , $mCount               , 3
  ld32         $mOutValue1    , $mzero                , $mOutRow          , 0
  brnz         $mRemainder    , 2f
  roll8l       $mOutV2        , $mOutValue1           , $mOutV2
  roll8r       $mOutV2        , $mOutV2               , $mOutV2
  bri          3f
2:
  cmpult       $mRemainder    , $mCount               , 2
  brnz         $mRemainder    , 2f
  roll16       $mOutV2        , $mOutValue1           , $mOutV2
  roll16       $mOutV2        , $mOutV2               , $mOutV2
  bri          3f
2:
  roll8r       $mOutV2        , $mOutValue1           , $mOutV2
  roll8l       $mOutV2        , $mOutV2               , $mOutV2
3:
  st32step     $mOutV2        , $mzero                , $mOutRow+=        , 1
4:
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  brnzdec      $mRowCount     , 1b
  exitz        $mzero

.size castFloatToGf8, .-__runCodelet_popfloat__experimental__CastFloatToGf8

#endif
