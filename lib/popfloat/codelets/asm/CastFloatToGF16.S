// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastFloatToGF16.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

.macro CAST_FLOAT_TO_GF16 FORMAT
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastFloatToGf16Supervisor___popfloat__experimental__FormatType__\FORMAT\()
.section .text.castFloatToGf16Supervisor_\FORMAT\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastFloatToGf16Supervisor___popfloat__experimental__FormatType__\FORMAT\()
  .type __runCodelet_popfloat__experimental__CastFloatToGf16Supervisor___popfloat__experimental__FormatType__\FORMAT\(), @function
  __runCodelet_popfloat__experimental__CastFloatToGf16Supervisor___popfloat__experimental__FormatType__\FORMAT\():

.supervisor
castFloatToGf16Supervisor_\FORMAT\():
  POPFLOAT_SUPERVISOR_CAST_OP castFloatToGf16_\FORMAT\()

.worker
castFloatToGf16_\FORMAT\():
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mGf16Param, $mvertex_base, POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseIn, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mGf16Param
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseIn
  setzi        $mTMemBase     , (TMEM_REGION0_BASE_ADDR / 4)
  POPFLOAT_CONVERT_SCALED_PTR32_TO_PTR $mBaseOut $mTMemBase
  {
    ldz16        $mCount        , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ELEMENTS_PER_WORKER_OFFSET;
    or           $signMask      , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
  POPFLOAT_GET_WORKER_INDEX $mWorkerIdx
  ldz16        $mCount        , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ELEMENTS_PER_WORKER_OFFSET
  cmpult       $mRemainder    , $mWorkerIdx           , $mQuotient
  add          $mCount        , $mCount               , $mRemainder
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
  ld64step     $azeros        , $mzero                , $mBaseIn+=        , $mWorkerIdx
  ld32step     $azero         , $mzero                , $mBaseOut+=       , $mWorkerIdx
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2mul     $sgnMaskV2     , $signMask:B           , $azeros
  }
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
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=       , CTXT_WORKERS;
    sort4x16hi   $outValue0     , $outValue0            , $outValue1
  }
  {
    st32step     $outValue0     , $mzero                , $mBaseOut+=     , CTXT_WORKERS;
    or64         $outValueV2    , $inValueV2            , $azeros
  }
3:
  {
    brz          $mRemainder    , 6f
    sort4x16hi   $outValue0     , $outValue0            , $outValue1
  }
  ldb16        $outValue1     , $mzero                , $mBaseOut         , 1
  roll16       $outValue0     , $outValue0            , $outValue1
  st32step     $outValue0     , $mzero                , $mBaseOut+=        , CTXT_WORKERS
#else
.error "GF16_BFLOAT not enabled"
#endif
.else
  add          $mCount        , $mCount               , -1
.ifc \FORMAT, NO___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_NO_DENORM
  ld64         $manExpMaskV2  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_BITS_MASK_OFFSET/2)
  bri          5f
4:
  st32step     $mOutV2        , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
5:
  {
    ld32         $biasCorrection, $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET);
    sort4x16hi   $sgnF16V2      , $sign0                , $sign1
  }
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
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
  {
    brnzdec      $mCount        , 4b
    and64        $sgnV2         , $inValueV2            , $sgnMaskV2
  }
#else
.error "GF16_EN_DENORM not enabled"
#endif
.else
.ifc \FORMAT, ENABLE___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_EN_DENORM
  bri          5f
4:
  st32step     $mOutV2        , $mzero                , $mBaseOut+=       , CTXT_WORKERS
5:
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
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
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
  {
    brnzdec      $mCount        , 4b
    and64        $sgnV2         , $inValueV2            , $sgnMaskV2
  }
#else
.error "GF16_EN_DENORM not enabled"
#endif
.else
.error "GF16 fromat not supported"
.endif // ENABLE___DENORM___GF16
.endif // NO___DENORM___GF16
4:
  brz          $mRemainder    , 4f
  ld32         $mOutSgn       , $mzero                , $mBaseOut         , 1
  roll16       $mOutV2        , $mOutSgn              , $mOutV2
  roll16       $mOutV2        , $mOutV2               , $mOutV2
4:
  st32step     $mOutV2        , $mzero                , $mBaseOut+=       , 1
.endif // BFLOAT16
6:
  exitz        $mzero
.size castFloatToGf16Supervisor_\FORMAT\(),\
  .-__runCodelet_popfloat__experimental__CastFloatToGf16Supervisor___popfloat__experimental__FormatType__\FORMAT\()
.endm

CAST_FLOAT_TO_GF16 BFLOAT16
CAST_FLOAT_TO_GF16 NO___DENORM___GF16
CAST_FLOAT_TO_GF16 ENABLE___DENORM___GF16

#endif
