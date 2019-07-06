#ifdef __IPU__

// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastGF16ToFloat.h"

.macro CAST_GF16_TO_FLOAT FORMAT
.section .text.castGf16ToFloat_\FORMAT\()
.align 4
  .globl __runCodelet_popfloat__CastGf16ToFloat___popfloat__gfexpr__GfloatFormatType__\FORMAT\()
  .type __runCodelet_popfloat__CastGf16ToFloat___popfloat__gfexpr__GfloatFormatType__\FORMAT\(), @function
  __runCodelet_popfloat__CastGf16ToFloat___popfloat__gfexpr__GfloatFormatType__\FORMAT\():

  ld32         $mGF16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
1:
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  st32         $mBaseIn       , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_BASE_PTR_OFFSET);
  st32         $mBaseOut      , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_OUT_BASE_PTR_OFFSET);
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
.ifc \FORMAT, BFLOAT16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_BFLOAT
  shr          $mCount        , $mCount               , 1
  ld32step     $bf16ValueV2   , $mzero                , $mInRow+=         , 1;
.align 8
  {
    rpt          $mCount        , ((3f - 2f)/8) - 1;
    roll16       $inValue0      , $azero                , $bf16ValueV2
  }
2:
  {
    ld32step     $bf16ValueV2   , $mzero                , $mInRow+=         , 1;
    sort4x16hi   $inValue1      , $azero                , $bf16ValueV2
  }
  {
    st64step     $gf32ValueV2   , $mzero                , $mOutRow+=        , 1;
    roll16       $inValue0      , $azero                , $bf16ValueV2
  }
3:
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  and          $mCount        , $mCount               , 1
  brz          $mCount        , 2f
  st32step     $inValue0      , $mzero                , $mOutRow+=        , 1;
#else
.error "GF16_BFLOAT not enabled"
#endif
.else
  add          $mCount        , $mCount               , 1
  shr          $mCount        , $mCount               , 1
  add          $mCount        , $mCount               , -1
  ld32         $mManSh0       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_SHIFT0_OFFSET)
  setzi        $mExpManMask   , 0x7FFF
  shl          $mExpManMask   , $mExpManMask          , $mManSh0
  ld32         $mBiasCorr     , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_ALIGN_OFFSET)
  ld32         $mManSh0       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_SHIFT0_OFFSET)
  ld32         $mManSh1       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_SHIFT1_OFFSET)
  ld32         $mInValueV2    , $mzero                , $mInRow           , 0
  ld32step     $gf16ValueV2   , $mzero                , $mInRow+=         , 1
  shl          $mInValue0     , $mInValueV2           , $mManSh0;
  shr          $mInValue1     , $mInValueV2           , $mManSh1
  and          $mInValue0     , $mInValue0            , $mExpManMask
  and          $mInValue1     , $mInValue1            , $mExpManMask
  {
    ld64         $expMaskGF16   , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_MASK_OFFSET/2);
    roll16       $inValue0      , $azero                , $gf16ValueV2
  }
  {
    add          $mInValue0     , $mInValue0            , $mBiasCorr;
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
.ifc \FORMAT, NO___DENORM___GF16
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_NO_DENORM
  ld64         $fpClamp       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_CLAMP_OFFSET/2);
4:
  {
    add          $mInValue1     , $mInValue1            , $mBiasCorr;
    f32v2mul     $sgnMaskV2     , $sgnMask:B            , $azeros
  }
  {
    st32         $mInValue0     , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_VALUE_OFFSET);
    sort4x16hi   $inValue1      , $azero                , $gf16ValueV2
  }
  {
    st32         $mInValue1     , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_VALUE_OFFSET)+1;
    and64        $sgnV2         , $gf32ValueV2          , $sgnMaskV2
  }
  {
    ld64         $gf32ValueV2   , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_VALUE_OFFSET);
    f32v2cmpeq   $isZeroV2      , $azeros               , $gf32ValueV2
  }
  {
    ld32         $mInValueV2    , $mzero                , $mInRow           , 0;
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
    ld64         $gf32ValueV2   , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_MASK_OFFSET/2);
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
    ld32step     $gf16ValueV2   , $mzero                , $mInRow+=         , 1;
    f32v2add     $outValueV2    , $outValueV2           , $azeros
  }
  brz          $mCount        , 5f;
  {
    st64step     $outValueV2    , $mzero                , $mOutRow+=        , 1;
    roll16       $inValue0      , $azero                , $gf16ValueV2
  }
  {
    add          $mInValue0     , $mInValue0            , $mBiasCorr;
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
  brnzdec      $mCount        , 4b
#else
.error "GF16_NO_DENORM not enabled"
#endif
.else
.ifc \FORMAT, ENABLE___DENORM___GF16   
#ifdef POPFLOAT_ENABLE_GF32_CLASS_GF16_EN_DENORM
4:
  {
    add          $mInValue1     , $mInValue1            , $mBiasCorr;
    f32v2mul     $sgnMaskV2     , $sgnMask:B            , $azeros
  }
  {
    st32         $mInValue0     , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_VALUE_OFFSET);
    sort4x16hi   $inValue1      , $azero                , $gf16ValueV2
  }
  {
    st32         $mInValue1     , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_VALUE_OFFSET)+1;
    and64        $sgnV2         , $gf32ValueV2          , $sgnMaskV2
  }
  {
    ld64         $gf32ValueV2   , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_VALUE_OFFSET);
    and64        $gf16DenormV2  , $gf32ValueV2          , $expMaskGF16
  }
  {
    ld64         $fpMinNorm     , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_MIN_NORM_OFFSET/2);
    f32v2cmpeq   $gf16DenormV2  , $azeros               , $gf16DenormV2
  }
  {
    ld32         $mInValueV2    , $mzero                , $mInRow           , 0;
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
    ld64         $fpClamp       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_CLAMP_OFFSET/2);
    or64         $gf32ValueV2   , $outValueV2           , $sgnV2
  }
  f32v2clamp   $outValueV2    , $gf32ValueV2          , $fpClamp
  {
    ld64         $expMaskGF16   , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_MASK_OFFSET/2);
    f32v2cmpeq   $nanMaskV2     , $outValueV2           , $gf32ValueV2  
  }
  {
    ld32step     $gf16ValueV2   , $mzero                , $mInRow+=         , 1;
    andc64       $nanMaskV2     , $expMaskGF16          , $nanMaskV2
  }
  {
    brz          $mCount        , 5f;
    or64         $outValueV2    , $nanMaskV2            , $outValueV2
  }
  {
    st64step     $outValueV2    , $mzero                , $mOutRow+=        , 1;
    roll16       $inValue0      , $azero                , $gf16ValueV2
  }
  {
    brnzdec      $mCount        , 4b
    or           $sgnMask       , $azero                , POPFLOAT_FP32_SIGN_MASK
  }
#else
.error "GF16_EN_DENORM not enabled"
#endif
.else
.error "GF16 format not supported"
.endif // GF16_NO_DENORM
.endif // GF16_EN_DENORM
5:
  ld32         $mBaseIn       , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_BASE_PTR_OFFSET);
  ld32         $mCount        , $mzero                , $mBaseIn          , 0
  and          $mCount        , $mCount               , 1
  brnz         $mCount        , 3f
  st64step     $outValueV2    , $mzero                , $mOutRow+=        , 1;
  bri          2f
3:
  st32step     $outValueV2_0  , $mzero                , $mOutRow+=        , 1;
.endif // GF16_BFLOAT
2:
  ld32         $mBaseIn       , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_IN_BASE_PTR_OFFSET);
  ld32         $mBaseOut      , $mworker_base         , $mzero            , (POPFLOAT_GF16_TO_FP32_STACK_OUT_BASE_PTR_OFFSET);
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  brnzdec      $mRowCount   , 1b
  exitz        $mzero

.size castGf16ToFloat_\FORMAT\(),\
  .-__runCodelet_popfloat__CastGf16ToFloat___popfloat__gfexpr__GfloatFormatType__\FORMAT\()
.endm

CAST_GF16_TO_FLOAT BFLOAT16
CAST_GF16_TO_FLOAT NO___DENORM___GF16
CAST_GF16_TO_FLOAT ENABLE___DENORM___GF16
#endif
