// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// popfloatCastGf8ToFloat

#include "GfloatConst.hpp"
#include "CastGF8ToFloat.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastGf8ToFloatSupervisor
.section .text.castGf8ToFloatSupervisor
.align 4
  .globl __runCodelet_popfloat__experimental__CastGf8ToFloatSupervisor
  .type __runCodelet_popfloat__experimental__CastGf8ToFloatSupervisor, @function
  __runCodelet_popfloat__experimental__CastGf8ToFloatSupervisor:
.supervisor
castGf8ToFloatSupervisor:
  POPFLOAT_SUPERVISOR_CAST_OP castGf8ToFloat

.worker
castGf8ToFloat:
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mGF8Param, $mvertex_base, POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseIn, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mGF8Param
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseOut
  setzi        $mTMemBase     , (TMEM_REGION0_BASE_ADDR / 4)
  POPFLOAT_CONVERT_SCALED_PTR32_TO_PTR $mBaseIn $mTMemBase
  ldz16        $mCount        , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ELEMENTS_PER_WORKER_OFFSET
  POPFLOAT_GET_WORKER_INDEX $mWorkerIdx
  ldz8         $mQuotient     , $mvertex_base         , $mzero            , 2 *   POPFLOAT_VBASE_CAST_LAST_WORKER_PARAM_OFFSET
  cmpult       $mRemainder    , $mWorkerIdx           , $mQuotient
  add          $mCount        , $mCount               , $mRemainder
  ldz8         $mRemainder    , $mvertex_base         , $mzero            , 2 *   POPFLOAT_VBASE_CAST_LAST_WORKER_PARAM_OFFSET + 1
  cmpeq        $mQuotient     , $mQuotient            , $mWorkerIdx
  mul          $mRemainder    , $mRemainder           , $mQuotient
  brz          $mQuotient     , 1f
  cmpult       $mQuotient     , $mzero                , $mRemainder
  add          $mCount        , $mCount               , $mQuotient
1:
  brz          $mCount        , 2f
  add          $mCount        , $mCount               , -1
  ldz16step    $mzero         , $mzero                , $mBaseIn+=        , $mWorkerIdx
  ld64step     $azeros        , $mzero                , $mBaseOut+=       , $mWorkerIdx
  ld32         $mManSh0       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET)
  setzi        $mExpManMask   , 0x7FFF
  shl          $mExpManMask   , $mExpManMask          , $mManSh0
  ld32         $mBiasCorr     , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_EXP_ALIGN_OFFSET)
  ld32         $mManSh0       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET)
  ld32         $mManSh1       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT1_OFFSET)
  ldz16step    $mInValueV2    , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
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
  bri          5f
4:
  {
    st64step     $outValueV2    , $mzero                , $mBaseOut+=       , CTXT_WORKERS
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
5:
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
    ldz16step    $mInValueV2    , $mzero                , $mBaseIn+=        , CTXT_WORKERS
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
  or64         $outValueV2    , $nanMaskV2            , $outValueV2
  {
    brnzdec      $mCount        , 4b
    roll16       $inValue0      , $azero                , $gf8ValueV2
  }
5:
  brnz         $mRemainder    , 3f
  st64step     $outValueV2    , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
  bri          2f
3:
  st32step     $outValueV2_0  , $mzero                , $mBaseOut+=       , CTXT_WORKERS
2:
  exitz        $mzero

.size castGf8ToFloatSupervisor, .-__runCodelet_popfloat__experimental__CastGf8ToFloatSupervisor

#endif
