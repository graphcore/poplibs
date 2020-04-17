// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// popfloat::CastToGfloat16<half>
// popfloat::CastToGloat16<float>
#include "GfloatConst.hpp"
#include "CastGF8ToHalf.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

.macro CAST_GF8_TO_HALF FORMAT
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastGf8ToHalfSupervisor___popfloat__experimental__FormatType__\FORMAT\()
.section .text.castGf8ToHalfSupervisor_\FORMAT\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastGf8ToHalfSupervisor___popfloat__experimental__FormatType__\FORMAT\()
  .type __runCodelet_popfloat__experimental__CastGf8ToHalfSupervisor___popfloat__experimental__FormatType__\FORMAT\(), @function
  __runCodelet_popfloat__experimental__CastGf8ToHalfSupervisor___popfloat__experimental__FormatType__\FORMAT\():
.supervisor
castGf8ToHalfSupervisor_\FORMAT\():
  POPFLOAT_SUPERVISOR_CAST_OP castGf8ToHalf_\FORMAT\()

.worker
castGf8ToHalf_\FORMAT\():
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
  brz          $mCount        , 3f
  ld32step     $azero         , $mzero                , $mBaseIn+=        , $mWorkerIdx
  ld64step     $azeros        , $mzero                , $mBaseOut+=       , $mWorkerIdx
  add          $mCount        , $mCount               , -1
  ld32         $scaleHalf     , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET+1)
.ifc \FORMAT, ONE___FIVE___TWO___GF8
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_1_5_2
  bri          1f
2:
  st64step     $outF16V4      , $mzero                , $mBaseOut+=       , CTXT_WORKERS
1:
  ld32step     $mInValueV4    , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
  shuf8x8lo    $mInValueV2_0  , $mzero                , $mInValueV4;
  shuf8x8hi    $mInValueV2_1  , $mzero                , $mInValueV4;
  st32         $mInValueV2_0  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_SCALED_IN_OFFSET);
  st32         $mInValueV2_1  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_SCALED_IN_OFFSET+1);
  ld64         $outF16V4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_SCALED_IN_OFFSET/2);
  {
    brnzdec      $mCount        , 2b
    f16v4mul     $outF16V4      , $scaleHalf:BL         , $outF16V4         // Scale values
  }
#else
.error "GF8_ONE_FIVE_TWO no enabled"
#endif
.else
.ifc \FORMAT, MIN___NORM___ALIGN___GF8
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_MIN_NORM_ALIGN
  ld32         $inputClampF16 , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET)
  ld32         $mManShr       , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_UNPACK_SHR_ALIGN_OFFSET)
  ld32         $mF8SignMask   , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GP16_PARAM_GF8_SIGN_MASK_OFFSET);
  ld32step     $mInValueV4    , $mzero                , $mBaseIn+=        , CTXT_WORKERS
  and          $mSignValueV4  , $mInValueV4           , $mF8SignMask;
  xor          $mInValueV4    , $mInValueV4           , $mSignValueV4;
  shuf8x8lo    $mSignV2_0     , $mzero                , $mSignValueV4;
  bri          1f
2:
  st64step     $outF16V4      , $mzero                , $mBaseOut+=       , CTXT_WORKERS
1:
  shuf8x8hi    $mSignV2_1     , $mzero                , $mSignValueV4;
  shuf8x8lo    $mInValueV2_0  , $mzero                , $mInValueV4;
  shuf8x8hi    $mInValueV2_1  , $mzero                , $mInValueV4;
  shr          $mInValueV2_0  , $mInValueV2_0         , $mManShr;
  shr          $mInValueV2_1  , $mInValueV2_1         , $mManShr;
  or           $mInValueV2_0  , $mInValueV2_0         , $mSignV2_0;
  or           $mInValueV2_1  , $mInValueV2_1         , $mSignV2_1;
  st32         $mInValueV2_0  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET)
  st32         $mInValueV2_1  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET)+1;
  ld64         $inValueV4     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET/2);
  {
    ld32step     $mInValueV4    , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f16v4clamp   $outF16V4      , $inValueV4            , $inputClampF16
  }
  {
    and          $mSignValueV4  , $mInValueV4           , $mF8SignMask;
    f16v4cmpeq   $outF16V4      , $inValueV4            , $outF16V4
  }
  ld64         $halfExpMaskV4 , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
  {
    xor          $mInValueV4    , $mInValueV4           , $mSignValueV4;
    andc64       $outF16V4      , $halfExpMaskV4        , $outF16V4
  }
  {
    shuf8x8lo    $mSignV2_0     , $mzero                , $mSignValueV4;
    or64         $outF16V4      , $inValueV4            , $outF16V4
  }
  {
    ld32         $scaleHalf     , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET+1)
    f16v4add     $outF16V4      , $outF16V4             , $azeros
  }
  {
    brnzdec      $mCount        , 2b;
    f16v4mul     $outF16V4      , $scaleHalf:BL         , $outF16V4         // Scale values
  }
#else
.error "G8_MIN_NORM_ALIGN not enabled"
#endif
.else
.ifc \FORMAT, MAX___NORM___ALIGN___GF8
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_MAX_NORM_ALIGN
  ld32step     $mInValueV4    , $mzero                , $mBaseIn+=        , CTXT_WORKERS
  shuf8x8lo    $mInValueV2_0  , $mzero                , $mInValueV4
  shuf8x8hi    $mInValueV2_1  , $mzero                , $mInValueV4
  st32         $mInValueV2_0  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET);
  st32         $mInValueV2_1  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET)+1;
  {
    ld64         $inValueV4     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET/2);
    setzi        $fpHalf        , 0x3800
  }
  ld64         $halfExpMaskV4 , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2)
  bri          1f
2:
  st64step     $outF16V4      , $mzero                , $mBaseOut+=       , CTXT_WORKERS
1:
  {
    ld32step     $mInValueV4    , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    andc64       $signManV4     , $inValueV4            , $halfExpMaskV4
  }
  {
    shuf8x8lo    $mInValueV2_0  , $mzero                , $mInValueV4;
    andc64       $isMaxExpV4    , $halfExpMaskV4        , $inValueV4
  }
  {
    shuf8x8hi    $mInValueV2_1  , $mzero                , $mInValueV4;
    f16v4cmpeq   $isMaxExpV4    , $azeros               , $isMaxExpV4
  }
  {
    ld64         $maxExpV4      , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MAX_EXPONENT_OFFSET/2)
    f16v4mul     $inValueV4     , $fpHalf:BL            , $inValueV4
  }
  {
    st32         $mInValueV2_0  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET)
    or64         $signManV4     , $signManV4            , $maxExpV4
  }
  {
    st32         $mInValueV2_1  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET)+1
    andc64       $inValueV4     , $inValueV4            , $isMaxExpV4
  }
  {
    ld32         $scaleHalf     , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET+1)
    and64        $signManV4     , $signManV4            , $isMaxExpV4
  }
  {
    ld64         $inValueV4     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_INPUT_OFFSET/2)
    or64         $outF16V4      , $inValueV4            , $signManV4
  }
  {
    ld64         $halfExpMaskV4 , $mGF8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2)
    f16v4mul     $outF16V4      , $scaleHalf:BL         , $outF16V4         // Scale values
  }
  {
    brnzdec      $mCount        , 2b
    setzi        $fpHalf        , 0x3800
  }
#else
.error "G8_MAX_NORM_ALIGN not enabled"
#endif
//.else
//.error "GF8 format not supported"
.endif // MAX___NORM___ALIGN___GF8
.endif // MIN___NORM___ALIGN___GF8
.endif // ONE___FIVE___TWO___GF8
2:
  brnz         $mRemainder    , 2f
  st64step     $outF16V4      , $mzero                , $mBaseOut+=       , CTXT_WORKERS
  exitz        $mzero
2:
  cmpult       $mCount        , $mRemainder           , 3;
  brnz         $mRemainder    , 2f
  st32step     $outF16V4_0    , $mzero                , $mBaseOut+=       , CTXT_WORKERS
  {
    add          $mRemainder    , $mRemainder           , -2;
    or           $outF16V4_0    , $outF16V4_1           , $azero
  }
2:
  cmpult       $mCount        , $mRemainder           , 2
  brnz         $mCount        , 2f
  ldb16        $outF16V4_1    , $mzero                , $mBaseOut         , 1
  sort4x16lo   $outF16V4_0    , $outF16V4_0           , $outF16V4_1
2:
  st32step     $outF16V4_0    , $mzero                , $mBaseOut+=       , 1
3:
  exitz        $mzero

.size castGf8ToHalfSupervisor_\FORMAT\(),\
  .-__runCodelet_popfloat__experimental__CastGf8ToHalfSupervisor___popfloat__experimental__FormatType__\FORMAT\()
.endm

CAST_GF8_TO_HALF ONE___FIVE___TWO___GF8
CAST_GF8_TO_HALF MIN___NORM___ALIGN___GF8
CAST_GF8_TO_HALF MAX___NORM___ALIGN___GF8

#endif
