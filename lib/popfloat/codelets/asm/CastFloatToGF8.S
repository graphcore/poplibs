// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// popfloatCastFloatToGf8

#include "GfloatConst.hpp"
#include "CastFloatToGF8.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastFloatToGf8Supervisor
.section .text.castFloatToGf8Supervisor
.align 4
  .globl __runCodelet_popfloat__experimental__CastFloatToGf8Supervisor
  .type __runCodelet_popfloat__experimental__CastFloatToGf8Supervisor, @function
  __runCodelet_popfloat__experimental__CastFloatToGf8Supervisor:
.supervisor
castFloatToGf8Supervisor:
  POPFLOAT_SUPERVISOR_CAST_OP castFloatToGf8

.worker
castFloatToGf8:
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mGf8Param, $mvertex_base, POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseIn, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mGf8Param
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseIn
  setzi        $mTMemBase     , (TMEM_REGION0_BASE_ADDR / 4)
  POPFLOAT_CONVERT_SCALED_PTR32_TO_PTR $mBaseOut $mTMemBase
  {
    ldz16        $mCount        , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ELEMENTS_PER_WORKER_OFFSET;
    or           $signMask      , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
  POPFLOAT_GET_WORKER_INDEX $mWorkerIdx
  ldz8         $mQuotient     , $mvertex_base         , $mzero            , 2 *   POPFLOAT_VBASE_CAST_LAST_WORKER_PARAM_OFFSET
  cmpult       $mRemainder    , $mWorkerIdx           , $mQuotient
  add          $mCount        , $mCount               , $mRemainder
  ldz8         $mRemainder    , $mvertex_base         , $mzero            , 2 *   POPFLOAT_VBASE_CAST_LAST_WORKER_PARAM_OFFSET + 1
  {
    cmpeq        $mQuotient     , $mQuotient            , $mWorkerIdx
    setzi        $increment0    , (2*CTXT_WORKERS - 1)
  }
  {
    mul          $mRemainder    , $mRemainder           , $mQuotient;
    setzi        $increment1    , 1
  }
  brz          $mQuotient     , 1f
  cmpult       $mQuotient     , $mzero                , $mRemainder
  add          $mCount        , $mCount               , $mQuotient
1:
  brz          $mCount        , 4f
  ld32step     $azero         , $mzero                , $mBaseOut+=       , $mWorkerIdx
  ld64step     $azeros        , $mzero                , $mBaseIn+=        , $mWorkerIdx
  ld64step     $azeros        , $mzero                , $mBaseIn+=        , $mWorkerIdx
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , 1;
    f32v2mul     $sgnMaskV2     , $signMask:B           , $azeros
  }
  {
    st32         $mRemainder    , $mworker_base         , $mzero            , 4;
    andc64       $outValueV2    , $inValueV2            , $sgnMaskV2
  }
  {
    st64         $increment     , $mworker_base         , $mzero            , 1;
    and64        $sgnV2         , $inValueV2            , $sgnMaskV2;
  }
  add          $mCount        , $mCount               , -1
  bri          1f
3:
  st32step     $mOutV2        , $mzero                , $mBaseOut+=       , CTXT_WORKERS
1:
  setzi        $mIncrementPtr , 2
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
    ld32         $mIncrement    , $mworker_base         , $mzero            , $mIncrementPtr
    and64        $outV2         , $outV2                , $isDenormV2
  }
  {
    ld32         $biasCorrection, $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET);
    andc64       $outValueV2    , $outValueV2           , $isDenormV2
  }
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , $mIncrement;
    f32v2mul     $outValueV2    , $biasCorrection:B     , $outValueV2
  }
  {
    ld64         $manExpMaskV2  , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_PACK_BITS_MASK_OFFSET/2);
    or64         $outValueV2    , $outValueV2           , $outV2
  }
  {
    add          $mIncrementPtr , $mIncrementPtr        , 1;
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
  brnzdec      $mCount        , 3b
2:
  ld32         $mRemainder    , $mworker_base         , $mzero            , 4;
  brnz         $mRemainder    , 2f
  st32step     $mOutV2        , $mzero                , $mBaseOut+=       , CTXT_WORKERS
  exitz        $mzero
2:
  cmpult       $mCount        , $mRemainder           , 3
  brnz         $mCount    , 2f
  ld32         $mOutValue1    , $mzero                , $mBaseOut         , 0
  brnz         $mCount        , 2f
  roll8l       $mOutV2        , $mOutValue1           , $mOutV2
  roll8r       $mOutV2        , $mOutV2               , $mOutV2
  bri          3f
2:
  cmpult       $mCount        , $mRemainder           , 2
  brnz         $mCount        , 2f
  roll16       $mOutV2        , $mOutValue1           , $mOutV2
  roll16       $mOutV2        , $mOutV2               , $mOutV2
  bri          3f
2:
  roll8r       $mOutV2        , $mOutValue1           , $mOutV2
  roll8l       $mOutV2        , $mOutV2               , $mOutV2
3:
  st32step     $mOutV2        , $mzero                , $mBaseOut+=       , CTXT_WORKERS
4:
  exitz        $mzero

.size castFloatToGf8Supervisor, .-__runCodelet_popfloat__experimental__CastFloatToGf8Supervisor

#endif
