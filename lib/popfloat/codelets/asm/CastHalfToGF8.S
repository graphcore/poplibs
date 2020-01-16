// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
// popfloat::CastHalfToGF8
#include "GfloatConst.hpp"
#include "CastHalfToGF8.h"


.macro CAST_HALF_TO_GF8 FORMAT
.section .text.castHalfToGf8_\FORMAT\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastHalfToGf8___popfloat__experimental__FormatType__\FORMAT\()
  .type __runCodelet_popfloat__experimental__CastHalfToGf8___popfloat__experimental__FormatType__\FORMAT\(), @function
  __runCodelet_popfloat__experimental__CastHalfToGf8___popfloat__experimental__FormatType__\FORMAT\():

  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
.ifnc \FORMAT, MAX___NORM___ALIGN___GF8
  ld32         $mGf8Param     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
.else
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_MAX_NORM_ALIGN
  {
    ld32         $mGf8Param     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
    setzi        $halfExpMask   , POPFLOAT_FP16_EXPONENT_MASK
  }
  sort4x16lo   $halfExpMask   , $halfExpMask          , $halfExpMask
  sort4x32lo   $halfExpMaskV4 , $halfExpMaskV4        , $halfExpMaskV4
  st64         $halfExpMaskV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_EXP_MASK_OFFSET/2)
#endif
.endif
1:
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  add          $mCount        , $mCount               , 3
  shr          $mCount        , $mCount               , 2
  ld32         $scaleHalf     , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET+1)
  ld64step     $outValueV4    , $mzero                , $mInRow+=         , 1
  {
    add          $mCount        , $mCount               , -1
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $outValueV4       // Scale values
  }
.ifc \FORMAT, ONE___FIVE___TWO___GF8
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_1_5_2
2:
  st64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_SCALED_IN_OFFSET/2)
  ld32         $mExpManV0     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_SCALED_IN_OFFSET)
  ld32         $mExpManV1     , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_SCALED_IN_OFFSET+1)
  sort8x8hi    $mOut          , $mExpManV0            , $mExpManV1;
  brz          $mCount        , 2f
  ld64step     $outValueV4    , $mzero                , $mInRow+=         , 1
  {
    st32step     $mOut          , $mzero                , $mOutRow+=        , 1
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $outValueV4       // Scale values
  }
  brnzdec      $mCount        , 2b
#else
.error "GF8_ONE_FIVE_TWO not enabled"
#endif
.else
.ifc \FORMAT, MAX___NORM___ALIGN___GF8
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_MAX_NORM_ALIGN
  setzi        $maxExp        , 0x7800
  {
    st32         $maxExp        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_MAX_EXP_OFFSET);
    f16v4absadd  $outExpV4      , $outValueV4           , $azeros
  }
  {
    ld64         $halfExpMaskV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_EXP_MASK_OFFSET/2);
    f16v4cmple   $isMaxExpV4    , $maxExp:BL            , $outExpV4
  }
  or64         $outExpV4      , $outValueV4           , $halfExpMaskV4
  {
    ld32         $mGF16ManDiff  , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_SHR_ALIGN_OFFSET);
    and64        $outExpV4      , $outExpV4             , $isMaxExpV4
  }
  f16v4add     $outValueV4    , $outValueV4           , $outValueV4
2:
  {
    ld32         $scaleHalf     , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET+1)
    andc64       $outValueV4    , $outValueV4           , $isMaxExpV4
  }
  {
    ld64step     $outValueV4    , $mzero                , $mInRow+=         , 1;
    or64         $outExpV4      , $outValueV4           , $outExpV4
  }
  {
    ld32         $maxExp        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_MAX_EXP_OFFSET);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $outValueV4       // Scale values
  }
  {
    atom         $mExpManV0     , $outFp8V2_0;
    f16v4absadd  $isMaxExpV4    , $outValueV4           , $azeros
  }
  {
    ld64         $halfExpMaskV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_GF8_EXP_MASK_OFFSET/2);
    f16v4cmple   $isMaxExpV4    , $maxExp:BL            , $isMaxExpV4
  }
  {
    atom         $mExpManV1     , $outFp8V2_1;
    or64         $outExpV4      , $outValueV4           , $halfExpMaskV4
  }
  {
    sort8x8hi    $mOut          , $mExpManV0            , $mExpManV1;
    and64        $outExpV4      , $outExpV4             , $isMaxExpV4
  }
  brz          $mCount        , 2f
  {
    st32step     $mOut          , $mzero                , $mOutRow+=       , 1;
    f16v4add     $outValueV4    , $outValueV4           , $outValueV4
  }
  brnzdec      $mCount        , 2b
#else
.error "GF8_MAX_NORM_ALIGN not enabled"
#endif
.else
.ifc \FORMAT, MIN___NORM___ALIGN___GF8
#ifdef POPFLOAT_ENABLE_GF16_CLASS_FP8_MIN_NORM_ALIGN
  or           $signV2_0      , $azero                , POPFLOAT_FP32_SIGN_MASK
  or           $fp8Sgn0       , $signV2_0             , 0x80
  {
    ld32         $mGF16ManDiff  , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_SHR_ALIGN_OFFSET);
    f16v4mul     $signV4        , $signV2_0:BU          , $azeros
  }
  {
    ld64         $manExpV4      , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_BITS_MASK_OFFSET/2);
    or           $fp8Sgn1       , $fp8Sgn0              , $azero
  }
  and64        $outExpV4      , $outValueV4           , $manExpV4
2:
  {
    atom         $mExpManV1     , $outFp8V2_1;
    f16v4cmpgt   $signV4        , $azeros               , $outValueV4
  }
  {
    atom         $mExpManV0     , $outFp8V2_0;
    and64        $signV4        , $signV4               , $fp8SgnV4
  }
  {
    shr          $mExpManV0     , $mExpManV0            , $mGF16ManDiff;
    sort4x16lo   $shufSgnLo     , $signV2_0             , $signV2_1
  }
  {
    shr          $mExpManV1     , $mExpManV1            , $mGF16ManDiff;
    sort4x16hi   $shufSgnHi     , $signV2_0             , $signV2_1
  }
  {
    sort8x8lo    $mManExp       , $mExpManV0            , $mExpManV1;
    or           $shufSgnLo     , $shufSgnHi            , $shufSgnLo
  }
  atom         $mOutSign      , $shufSgnLo
  {
    or           $mOut          , $mManExp              , $mOutSign;
    setzi        $signV2_0      , POPFLOAT_FP16_SIGN_MASK
  }
  {
    ld64step     $outValueV4    , $mzero                , $mInRow+=         , 1;
    f16v4mul     $signV4        , $signV2_0:BL          , $azeros
  }
  ld32         $scaleHalf     , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET+1)
  {
    ld64         $manExpV4      , $mGf8Param            , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_PACK_BITS_MASK_OFFSET/2);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $outValueV4       // Scale values
  }
  {
    brz          $mCount        , 2f;
    and64        $outExpV4    , $outValueV4           , $manExpV4
  }
  st32step     $mOut          , $mzero                , $mOutRow+=        , 1;
  brnzdec      $mCount      , 2b
#else
.error "GF8_MIN_NORM_ALIGN not enabled"
#endif
.else
.endif // MIN___NORM___ALIGN___GF8
.endif // MAX___NORM___ALIGN___GF8
.endif // ONE___FIVE___TWO___GF8
2:
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  and          $mCount        , $mCount               , 3
  brnz         $mCount        , 2f
  st32step     $mOut          , $mzero                , $mOutRow+=        , 1
  bri          4f
2:
  cmpult       $mRemainder    , $mCount               , 3
  ld32         $mExpManV0     , $mzero                , $mOutRow          , 0
  brnz         $mRemainder    , 2f
  roll8l       $mOut          , $mExpManV0            , $mOut
  roll8r       $mOut          , $mOut                 , $mOut
  bri          3f
2:
  cmpult       $mRemainder    , $mCount               , 2
  brnz         $mRemainder    , 2f
  roll16       $mOut          , $mExpManV0            , $mOut
  roll16       $mOut          , $mOut                 , $mOut
  bri          3f
2:
  roll8r       $mOut          , $mExpManV0            , $mOut
  roll8l       $mOut          , $mOut                 , $mOut
3:
  st32step     $mOut          , $mzero                , $mOutRow+=        , 1
4:
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  brnzdec      $mRowCount     , 1b
  exitz        $mzero

.size castHalfToGf8_\FORMAT\(),\
  .-__runCodelet_popfloat__experimental__CastHalfToGf8___popfloat__experimental__FormatType__\FORMAT\()
.endm

CAST_HALF_TO_GF8 ONE___FIVE___TWO___GF8
CAST_HALF_TO_GF8 MIN___NORM___ALIGN___GF8
CAST_HALF_TO_GF8 MAX___NORM___ALIGN___GF8

#endif
