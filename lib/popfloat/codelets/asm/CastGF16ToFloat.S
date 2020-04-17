// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastGF16ToFloat.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

.macro CAST_GF16_TO_FLOAT FORMAT
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastGf16ToFloatSupervisor___popfloat__experimental__FormatType__\FORMAT\()
.section .text.castGf16ToFloatSupervisor_\FORMAT\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastGf16ToFloatSupervisor___popfloat__experimental__FormatType__\FORMAT\()
  .type __runCodelet_popfloat__experimental__CastGf16ToFloatSupervisor___popfloat__experimental__FormatType__\FORMAT\(), @function
  __runCodelet_popfloat__experimental__CastGf16ToFloatSupervisor___popfloat__experimental__FormatType__\FORMAT\():
.supervisor
castGf16ToFloatSupervisor_\FORMAT\():
  POPFLOAT_SUPERVISOR_CAST_OP castGf16ToFloat_\FORMAT\()

.worker
castGf16ToFloat_\FORMAT\():
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mGF16Param, $mvertex_base, POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseIn, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mGF16Param
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
  brz          $mCount        , 6f
  ld32step     $azero         , $mzero                , $mBaseIn+=        , $mWorkerIdx
  ld64step     $azeros        , $mzero                , $mBaseOut+=       , $mWorkerIdx
1:
.ifc \FORMAT, BFLOAT16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_BFLOAT
  ld32step     $bf16ValueV2   , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
.align 8
  {
    rpt          $mCount        , ((3f - 2f)/8) - 1;
    roll16       $inValue0      , $azero                , $bf16ValueV2
  }
2:
  {
    ld32step     $bf16ValueV2   , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    sort4x16hi   $inValue1      , $azero                , $bf16ValueV2
  }
  {
    st64step     $gf32ValueV2   , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
    roll16       $inValue0      , $azero                , $bf16ValueV2
  }
3:
  brz          $mRemainder    , 6f
  st32step     $inValue0      , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
#else
.error "GF16_BFLOAT not enabled"
#endif
.else
  add          $mCount        , $mCount               , -1
  ld32         $mManSh0       , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET)
  setzi        $mExpManMask   , 0x7FFF
  shl          $mExpManMask   , $mExpManMask          , $mManSh0
  ld32         $mBiasCorr     , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_EXP_ALIGN_OFFSET)
  ld32         $mManSh0       , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET)
  ld32         $mManSh1       , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT1_OFFSET)
  ld32         $mInValueV2    , $mzero                , $mBaseIn          , 0
  ld32step     $gf16ValueV2   , $mzero                , $mBaseIn+=        , CTXT_WORKERS
  shl          $mInValue0     , $mInValueV2           , $mManSh0;
  shr          $mInValue1     , $mInValueV2           , $mManSh1
  and          $mInValue0     , $mInValue0            , $mExpManMask
  and          $mInValue1     , $mInValue1            , $mExpManMask
  {
    ld64         $expMaskGF16   , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET/2);
    roll16       $inValue0      , $azero                , $gf16ValueV2
  }
  {
    add          $mInValue0     , $mInValue0            , $mBiasCorr;
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
.ifc \FORMAT, NO___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_NO_DENORM
  ld64         $fpClamp       , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
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
    sort4x16hi   $inValue1      , $azero                , $gf16ValueV2
  }
  {
    st32         $mInValue1     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET)+1;
    and64        $sgnV2         , $gf32ValueV2          , $sgnMaskV2
  }
  {
    ld64         $gf32ValueV2   , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET);
    f32v2cmpeq   $isZeroV2      , $azeros               , $gf32ValueV2
  }
  {
    ld32         $mInValueV2    , $mzero                , $mBaseIn          , 0;
    andc64       $gf32ValueV2   , $gf32ValueV2          , $isZeroV2
  }
  {
    shl          $mInValue0     , $mInValueV2           , $mManSh0;
    or64         $gf32ValueV2   , $gf32ValueV2          , $sgnV2
  }
  {
    shr          $mInValue1     , $mInValue1            , $mManSh1;
    f32v2clamp   $outValueV2    , $gf32ValueV2          , $fpClamp
  }
  {
    ld64         $gf32ValueV2   , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET/2);
    f32v2cmpeq   $nanMaskV2     , $outValueV2           , $gf32ValueV2
  }
  {
    and          $mInValue0     , $mInValue0            , $mExpManMask;
    andc64       $nanMaskV2     , $gf32ValueV2          , $nanMaskV2
  }
  {
    and          $mInValue1     , $mInValue1            , $mExpManMask;
    or64         $outValueV2    , $outValueV2           , $nanMaskV2
  }
  {
    ld32step     $gf16ValueV2   , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2add     $outValueV2    , $outValueV2           , $azeros
  }
  {
    add          $mInValue0     , $mInValue0            , $mBiasCorr;
    roll16       $inValue0      , $azero                , $gf16ValueV2
  }
  brnzdec      $mCount        , 4b
#else
.error "GF16_NO_DENORM not enabled"
#endif
.else
.ifc \FORMAT, ENABLE___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_EN_DENORM
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
    sort4x16hi   $inValue1      , $azero                , $gf16ValueV2
  }
  {
    st32         $mInValue1     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET)+1;
    and64        $sgnV2         , $gf32ValueV2          , $sgnMaskV2
  }
  {
    ld64         $gf32ValueV2   , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_UNPACK_INPUT_OFFSET);
    and64        $gf16DenormV2  , $gf32ValueV2          , $expMaskGF16
  }
  {
    ld64         $fpMinNorm     , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET/2);
    f32v2cmpeq   $gf16DenormV2  , $azeros               , $gf16DenormV2
  }
  {
    ld32         $mInValueV2    , $mzero                , $mBaseIn          , 0;
    f32v2sub     $fpMinNorm     , $gf32ValueV2          , $fpMinNorm
  }
  {
    shl          $mInValue0     , $mInValueV2           , $mManSh0;
    and64        $fpMinNorm     , $fpMinNorm            , $gf16DenormV2
  }
  {
    shr          $mInValue1     , $mInValueV2           , $mManSh1;
    andc64       $gf32ValueV2   , $gf32ValueV2          , $gf16DenormV2
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
    ld64         $fpClamp       , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
    or64         $gf32ValueV2   , $outValueV2           , $sgnV2
  }
  f32v2clamp   $outValueV2    , $gf32ValueV2          , $fpClamp
  {
    ld64         $expMaskGF16   , $mGF16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET/2);
    f32v2cmpeq   $nanMaskV2     , $outValueV2           , $gf32ValueV2
  }
  {
    ld32step     $gf16ValueV2   , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    andc64       $nanMaskV2     , $expMaskGF16          , $nanMaskV2
  }
  or64         $outValueV2    , $nanMaskV2            , $outValueV2
  {
    brnzdec      $mCount        , 4b
    roll16       $inValue0      , $azero                , $gf16ValueV2
  }
#else
.error "GF16_EN_DENORM not enabled"
#endif
.else
.error "GF16 format not supported"
.endif // GF16_NO_DENORM
.endif // GF16_EN_DENORM
5:
  brnz         $mRemainder    , 3f
  st64step     $outValueV2    , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
  bri          6f
3:
  st32step     $outValueV2_0  , $mzero                , $mBaseOut+=       , 1;
.endif // GF16_BFLOAT
6:
  exitz        $mzero

.size castGf16ToFloatSupervisor_\FORMAT\(),\
  .-__runCodelet_popfloat__experimental__CastGf16ToFloatSupervisor___popfloat__experimental__FormatType__\FORMAT\()
.endm

CAST_GF16_TO_FLOAT BFLOAT16
CAST_GF16_TO_FLOAT NO___DENORM___GF16
CAST_GF16_TO_FLOAT ENABLE___DENORM___GF16
#endif
