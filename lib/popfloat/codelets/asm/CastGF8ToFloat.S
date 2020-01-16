// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__

// popfloatCastGf8ToFloat

#include "GfloatConst.hpp"
#include "CastGF8ToFloat.h"

.section .text.castGf8ToFloat
.align 4
  .globl __runCodelet_popfloat__experimental__CastGf8ToFloat
  .type __runCodelet_popfloat__experimental__CastGf8ToFloat, @function
  __runCodelet_popfloat__experimental__CastGf8ToFloat:

  ld32         $mGF8Param     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
1:
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  st32         $mBaseIn       , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_IN_BASE_PTR_OFFSET);
  st32         $mBaseOut      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BASE_PTR_OFFSET);
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  add          $mCount        , $mCount               , 1
  shr          $mCount        , $mCount               , 1
  add          $mCount        , $mCount               , -1
  ld32         $mManSh0       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET)
  setzi        $mExpManMask   , 0x7FFF
  shl          $mExpManMask   , $mExpManMask          , $mManSh0
  ld32         $mBiasCorr     , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_EXP_ALIGN_OFFSET)
  ld32         $mManSh0       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET)
  ld32         $mManSh1       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT1_OFFSET)
  ldz16step    $mInValueV2    , $mzero                , $mInRow+=         , 1;
  shuf8x8lo    $mInValueV2    , $mzero                , $mInValueV2
  st32         $mInValueV2    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF8_SHUF8_OFFSET)
  ld32         $gf8ValueV2    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF8_SHUF8_OFFSET)
  shl          $mInValue0     , $mInValueV2           , $mManSh0;
  shr          $mInValue1     , $mInValueV2           , $mManSh1
  and          $mInValue0     , $mInValue0            , $mExpManMask
  and          $mInValue1     , $mInValue1            , $mExpManMask
  {
    ld64         $expMaskGF8    , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET/2);
    roll16       $inValue0      , $azero                , $gf8ValueV2
  }
  {
    add          $mInValue0     , $mInValue0            , $mBiasCorr;
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
4:
  {
    add          $mInValue1     , $mInValue1            , $mBiasCorr;
    f32v2mul     $sgnMaskV2     , $sgnMask:B            , $azeros
  }
  {
    st32         $mInValue0     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET);
    sort4x16hi   $inValue1      , $azero                , $gf8ValueV2
  }
  {
    st32         $mInValue1     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET)+1;
    and64        $sgnV2         , $gf32ValueV2          , $sgnMaskV2
  }
  {
    ld64         $gf32ValueV2   , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET);
    and64        $gf8DenormV2   , $gf32ValueV2          , $expMaskGF8
  }
  {
    ld64         $fpMinNorm     , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET/2);
    f32v2cmpeq   $gf8DenormV2   , $azeros               , $gf8DenormV2
  }
  {
    ldz16step    $mInValueV2    , $mzero                , $mInRow+=         , 1;
    f32v2sub     $fpMinNorm     , $gf32ValueV2          , $fpMinNorm
  }
  shuf8x8lo    $mInValueV2    , $mzero                , $mInValueV2
  st32         $mInValueV2    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF8_SHUF8_OFFSET)
  {
    shl          $mInValue0     , $mInValueV2           , $mManSh0;
    and64        $fpMinNorm     , $fpMinNorm            , $gf8DenormV2
  }
  {
    shr          $mInValue1     , $mInValueV2           , $mManSh1;
    andc64       $gf32ValueV2   , $gf32ValueV2          , $gf8DenormV2
  }
  {
    and          $mInValue0     , $mInValue0            , $mExpManMask;
    or           $fpHalf        , $azero                , 0x3F000000
  }
  {
    and          $mInValue1     , $mInValue1            , $mExpManMask;
    f32v2mul     $gf32ValueV2   , $fpHalf:B             , $gf32ValueV2
  }
  {
    add          $mInValue0     , $mInValue0            , $mBiasCorr;
    or64         $outValueV2    , $gf32ValueV2          , $outValueV2
  }
  {
    ld64         $fpClamp       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
    or64         $gf32ValueV2   , $outValueV2           , $sgnV2
  }
  f32v2clamp   $outValueV2    , $gf32ValueV2          , $fpClamp
  {
    ld64         $expMaskGF8    , $mGF8Param             , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET/2);
    f32v2cmpeq   $nanMaskV2     , $outValueV2           , $gf32ValueV2  
  }
  {
    ld32         $gf8ValueV2    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF8_SHUF8_OFFSET)
    andc64       $nanMaskV2     , $expMaskGF8           , $nanMaskV2
  }
  {
    brz          $mCount        , 5f;
    or64         $outValueV2    , $nanMaskV2            , $outValueV2
  }
  {
    st64step     $outValueV2    , $mzero                , $mOutRow+=        , 1;
    roll16       $inValue0      , $azero                , $gf8ValueV2
  }
  {
    brnzdec      $mCount        , 4b
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
5:
  ld32         $mBaseIn       , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_IN_BASE_PTR_OFFSET);
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  and          $mCount        , $mCount               , 1
  brnz         $mCount        , 3f
  st64step     $outValueV2    , $mzero                , $mOutRow+=        , 1;
  bri          2f
3:
  st32step     $outValueV2_0  , $mzero                , $mOutRow+=        , 1
2:
  ld32         $mBaseIn       , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_IN_BASE_PTR_OFFSET);
  ld32         $mBaseOut      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BASE_PTR_OFFSET);
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  brnzdec      $mRowCount   , 1b
  exitz        $mzero

.size castGf8ToFloat, .-__runCodelet_popfloat__experimental__CastGf8ToFloat

#endif
