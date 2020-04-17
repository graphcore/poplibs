// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastToGfloat32Sr.h"
#include "arch/gc_tile_defines.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

.macro CAST_TO_GFLOAT32_SR SAVEFP32 NANOO DENSITY INPLACE
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mGf32Param, $mvertex_base, POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseIn, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_GET_WORKER_INDEX $mWorkerIdx
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mGf32Param
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseIn
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseOut
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mCastParams
.if \INPLACE == 1
  ldz16        $mCount        , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ELEMENTS_PER_WORKER_OFFSET
  ldz8         $mQuotient     , $mvertex_base         , $mzero            , 2 * POPFLOAT_VBASE_CAST_INPLACE_LAST_WORKER_PARAM_OFFSET
.else
  ldz16        $mCount        , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ELEMENTS_PER_WORKER_OFFSET
  ldz8         $mQuotient     , $mvertex_base         , $mzero            , 2 *   POPFLOAT_VBASE_CAST_LAST_WORKER_PARAM_OFFSET
.endif
  cmpult       $mRemainder    , $mWorkerIdx           , $mQuotient
  add          $mCount        , $mCount               , $mRemainder
.if \INPLACE == 1
  ldz8         $mRemainder    , $mvertex_base         , $mzero            , 2 * POPFLOAT_VBASE_CAST_INPLACE_LAST_WORKER_PARAM_OFFSET + 1
.else
  ldz8         $mRemainder    , $mvertex_base         , $mzero            , 2 *   POPFLOAT_VBASE_CAST_LAST_WORKER_PARAM_OFFSET + 1
.endif
  cmpeq        $mQuotient     , $mQuotient            , $mWorkerIdx
  mul          $mRemainder    , $mRemainder           , $mQuotient
  brz          $mQuotient     , 1f
  cmpult       $mQuotient     , $mzero                , $mRemainder
  add          $mCount        , $mCount               , $mQuotient
1:
  brz          $mCount        , 7f
  add          $mCount        , $mCount               , -1
  ld64step     $azeros        , $mzero                , $mBaseIn+=        , $mWorkerIdx
.if \SAVEFP32 == 1
  ld64step     $azeros        , $mzero                , $mBaseOut+=       , $mWorkerIdx
.else
  ld32step     $azero         , $mzero                , $mBaseOut+=       , $mWorkerIdx
.endif
  ld64         $inValueV2     , $mzero                , $mBaseIn          , 0
  ld32         $enDenorm      , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET)
  bri          1f
2:
.if \SAVEFP32 == 1
  st64step     $outV2         , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.else
  f32v2tof16   $out0          , $outV2
  st32step     $out0          , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.endif
1:
  ld64         $fpExpMaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2)
  {
    ld32         $fpMinNorm     , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET);
    and64        $expV2         , $inValueV2            , $fpExpMaskV2      // Extract exponents
  }
  {
    ld64         $outBitMaskV2  , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_NORM_MANT_MASK_OFFSET/2);
    f32v2cmpgt   $isDenormV2    , $fpMinNorm:B          , $expV2            // Create a mask for denorms
  }
  brz          $enDenorm      , 3f
  {
    ld64         $fpHalfMinGF32 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2);
    andc64       $outBitMaskV2  , $outBitMaskV2         , $isDenormV2       // Mantissa mask for norms
  }
  {
    st64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    and64        $dnrmManMaskV2 , $expV2                , $isDenormV2       // Copy exponents to denorm lanes
  }
  {
    ld64         $sgnExpMaskV2  , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_EXP_MASK_OFFSET/2);
    f32v2sub     $dnrmManMaskV2 , $dnrmManMaskV2        , $fpHalfMinGF32    // Denorm mantissa
  }
  {
    ld64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    or64         $dnrmManMaskV2 , $dnrmManMaskV2        , $sgnExpMaskV2     // Set FP32 sign and exponent bits
  }
  {
    ld64         $fpExpMaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2);
    or64         $outBitMaskV2  , $outBitMaskV2         , $dnrmManMaskV2    // Combine norm/denorm masks
  }
3:
  {
    st64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    not64        $roundCorrV2   , $outBitMaskV2
  }
  or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
.ifc \DENSITY, BERNOULLI
  ld32         $probBrnoulli  , $mCastParams          , $mzero            , POPFLOAT_CAST_PARAMS_FP32_DENSITY_PARAM_OFFSET;
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2rmask   $roundCorrV2   , $roundCorrV2          , $probBrnoulli
  }
  ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS
.else
  f32v2add     $manLsbMaskV2  , $roundCorrV2          , $roundCorrV2      // Mantissa LSB power
  and64        $manLsbMaskV2  , $manLsbMaskV2          , $fpExpMaskV2      // Extract exponent of result (half mantissa LSB)
  st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
.ifc \DENSITY, LOGIT___NORMAL
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET/2);
    f32v2grand   $roundCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  f32sigm      $roundCorrV2_0 , $roundCorrV2_0
  {
    ld64         $clampCorr     , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET/2)
    f32sigm      $roundCorrV2_1 , $roundCorrV2_1
  }
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
.else
.ifc \DENSITY, LAPLACE
  urand64      $roundCorrV2
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_IN_OFFSET/2);
    f32v2sufromui $roundCorrV2  , $roundCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $sgnMaskV2    , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
  f32v2cmplt   $corrDenorm    , $azeros               , $roundCorrV2       // Mask positive values
  and64        $sgnMaskV2     , $corrDenorm           , $sgnMaskV2         // Create mask to flip sign
  f32v2absadd  $roundCorrV2   , $roundCorrV2          , $roundCorrV2
  or           $constOne      , $azero                , (POPFLOAT_FP32_EXPONENT_BIAS << POPFLOAT_NUM_FP32_MANTISSA_BITS)  // ONE
  f32v2add     $corrDenorm    , $constOne:B           , $azeros
  f32v2sub     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  f32ln        $roundCorrV2_0 , $roundCorrV2_0
  f32ln        $roundCorrV2_1 , $roundCorrV2_1
  or64         $corrDenorm    , $corrDenorm           , $sgnMaskV2         // Flip sign
  {
    ld64         $clampCorr     , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET/2)
    f32v2mul     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  }
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.else
.ifc \DENSITY, UNIFORM
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET/2);
    urand64      $roundCorrV2
  }
  f32v2sufromui $roundCorrV2  , $roundCorrV2
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.else
.ifc \DENSITY, NORMAL
  {
    ld64         $clampCorr     , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET/2)
    f32v2grand   $roundCorrV2
  }
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.else
.ifc \DENSITY, LOGISTIC
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  {
    ld64          $corrDenorm   , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_IN_OFFSET/2);
    f32v2sufromui $roundCorrV2  , $roundCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  or           $constHalf     , $azero                , ((POPFLOAT_FP32_EXPONENT_BIAS) << POPFLOAT_NUM_FP32_MANTISSA_BITS)  // 1
  f32v2sub     $oneMinCorrV2  , $constHalf:B          , $roundCorrV2        // One minus ~U[0,1]
  f32ln        $roundCorrV2_0 , $roundCorrV2_0
  f32ln        $roundCorrV2_1 , $roundCorrV2_1
  f32ln        $oneMinCorrV2_0, $oneMinCorrV2_0
  f32ln        $oneMinCorrV2_1, $oneMinCorrV2_1
  {
    ld64         $clampCorr     , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET/2)
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $oneMinCorrV2
  }
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.else # TRUNCATE_NORMAL/TRUNCATE_LOGIT_NORMAL
  {
    ld32         $nIterations   , $mCastParams          , $mzero            , POPFLOAT_CAST_PARAMS_FP32_DENSITY_PARAM_OFFSET
    f32v2grand   $roundCorrV2
  }
  {
    st64         $azeros        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_TRUNCATED_NORM_OFFSET/2)
    and64        $maskOut       , $maskOut              , $azeros
  }
.LtruncatedNormal_loop_start_\SAVEFP32\()_\INPLACE\()_\NANOO\()_\DENSITY\():
  {
    ld64         $clampCorr     , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET/2)
    andc64       $roundCorrV2   , $roundCorrV2          , $maskOut
  }
  f32v2clamp   $clampOut      , $roundCorrV2          , $clampCorr
  f32v2cmpeq   $clampOut      , $clampOut             , $roundCorrV2
  and64        $roundCorrV2   , $roundCorrV2          , $clampOut
  {
    ld64         $trncNorm      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_TRUNCATED_NORM_OFFSET/2);
    or64         $maskOut       , $maskOut              , $clampOut
  }
  atom         $maskOut_0     , $maskOut0
  {
    atom         $maskOut_1     , $maskOut1;
    or64         $trncNorm      , $trncNorm             , $roundCorrV2
  }
  and          $maskOut_0     , $maskOut_0            , $maskOut_1
  xnor         $maskOut_0     , $maskOut_0            , $mzero;
  {
    st64         $trncNorm      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_TRUNCATED_NORM_OFFSET/2);
    f32v2grand   $roundCorrV2
  }
  brz         $maskOut_0        , .LtruncatedNormal_loop_end_\SAVEFP32\()_\INPLACE\()_\NANOO\()_\DENSITY\()
  brnzdec     $nIterations      , .LtruncatedNormal_loop_start_\SAVEFP32\()_\INPLACE\()_\NANOO\()_\DENSITY\()
  brnz        $maskOut_0        , .LtruncatedNormal_loop_start_\SAVEFP32\()_\INPLACE\()_\NANOO\()_\DENSITY\()
.LtruncatedNormal_loop_end_\SAVEFP32\()_\INPLACE\()_\NANOO\()_\DENSITY\():
  {
    ld64         $corrDenorm    , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET/2);
    or64         $roundCorrV2   , $trncNorm             , $azeros
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.ifc \DENSITY, TRUNCATED___LOGIT___NORMAL
  f32sigm      $roundCorrV2_0 , $roundCorrV2_0
  f32sigm      $roundCorrV2_1 , $roundCorrV2_1
.endif // TRUNCATED_LOGIT_NORMAT/TRUNCATED_NORMAL
.endif // .ifc \DENSITY, LOGISTIC
.endif // .ifc \DENSITY, NORMAL
.endif // .ifc \DENSITY, UNIFORM
.endif // .ifc \DENSITY, LAPLACE
.endif // .ifc \DENSITY LOGIT___NORMAL
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS
    f32v2mul     $roundCorrV2   , $roundCorrV2          , $manLsbMaskV2
  }
  {
    ld64         $srMaskV2      , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_SR_MASK_OFFSET/2)
    f32v2add     $roundCorrV2   , $roundCorrV2          , $manLsbMaskV2
  }
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    and64        $roundCorrV2   , $roundCorrV2          , $srMaskV2
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $manLsbMaskV2
  }
.endif # .ifc \DENSITY, BERNOULLI
  and64        $sgnV2         , $inValueV2            , $sgnV2
  {
    ld64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2)
    f32v2absadd  $inValueV2     , $inValueV2            , $roundCorrV2      // Add correction
  }
  {
    ld32         $minValueGF32  , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET);
    and64        $inValueV2     , $inValueV2            , $outBitMaskV2     // Apply mask
  }
  {
    ld64         $fpClamp       , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
    f32v2cmple   $nonZeroV4     , $minValueGF32:B       , $inValueV2        // Mask for values greater-than or equal minDenorm
  }
  and64        $inValueV2     , $inValueV2            , $nonZeroV4        // Set Values less than minDenorm to 0
.ifc \NANOO, true
  {
    st64         $sgnV2         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF16_SIGN_OFFSET/2);
    f32v2cmplt   $outNanMaskV2  , $fpClampPos:B         , $inValueV2
  }
  {
    ld64         $qNanV2        , $mGf32Param           , $mzero          , (POPFLOAT_CAST_TO_GF32_PARAM_QNAN_MASK_OFFSET/2);
    andc64       $inValueV2     , $inValueV2            , $outNanMaskV2
  }
  {
    ld64         $fpClamp       , $mGf32Param           , $mzero          , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
    and64        $outNanMaskV2  , $qNanV2               , $outNanMaskV2
  }
  {
    ld64         $sgnV2         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF16_SIGN_OFFSET/2);
    or64         $inValueV2     , $outNanMaskV2         , $inValueV2
  }
.endif
  {
    ld64         $inValueV2     , $mzero                , $mBaseIn          , 0;
    f32v2clamp   $tmpOutV2      , $inValueV2            , $fpClamp          // Clamp values to max float (Nans will propagate)
  }
  or64         $outV2         , $tmpOutV2             , $sgnV2
  f32v2cmpeq   $tmpOutV2      , $outV2                , $azeros           // Mask for +/-0.0
  {
    brnzdec      $mCount        , 2b
    andc64       $outV2         , $outV2                , $tmpOutV2         // Convert all -0.0 into +0.0
  }
  brnz         $mRemainder    , 1f
.if \SAVEFP32 == 1
  st64step     $outV2         , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.else
  f32v2tof16   $out0          , $outV2
  st32step     $out0          , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.endif
  bri          7f
1:
.if \SAVEFP32 == 0
  {
    ldb16        $outV2_1       , $mzero                , $mBaseOut         , 1
    f32tof16     $outV2_0       , $outV2_0
  }
  roll16       $outV2_0       , $outV2_0              , $outV2_1
.endif
  st32         $outV2_0       , $mzero                , $mBaseOut         , 0
7:
  exitz        $mzero
.endm

.macro CAST_TO_GFLOAT32_SR_OP TYPE1, TYPE2, NANOO, DENSITY
DEF_STACK_USAGE  0 __runCodelet_popfloat__experimental__CastToGfloat32SrSupervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\()
.section .text.castToGfloat32SrSupervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\DENSITY\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat32SrSupervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\()
  .type __runCodelet_popfloat__experimental__CastToGfloat32SrSupervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat32SrSupervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\():
.supervisor
castToGfloat32SrSupervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\DENSITY\():
  POPFLOAT_SUPERVISOR_CAST_OP castToGfloat32Sr_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\DENSITY\()

.worker
castToGfloat32Sr_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\DENSITY\():
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mCastParams, $mvertex_base, POPFLOAT_VBASE_CAST_ROUNDING_PARAM_OFFSET
.ifc \TYPE1, \TYPE2
  CAST_TO_GFLOAT32_SR 1 \NANOO \DENSITY 0
.else
  CAST_TO_GFLOAT32_SR 0 \NANOO \DENSITY 0
.endif
.size castToGfloat32SrSupervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\DENSITY\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat32SrSupervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\()

.endm

CAST_TO_GFLOAT32_SR_OP float, float, true , UNIFORM
CAST_TO_GFLOAT32_SR_OP float, half , true , UNIFORM
CAST_TO_GFLOAT32_SR_OP float, float, false, UNIFORM
CAST_TO_GFLOAT32_SR_OP float, half , false, UNIFORM

CAST_TO_GFLOAT32_SR_OP float, float, true , NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, NORMAL

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___NORMAL

CAST_TO_GFLOAT32_SR_OP float, float, true , BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, half , true , BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, float, false, BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, half , false, BERNOULLI

CAST_TO_GFLOAT32_SR_OP float, float, true , LAPLACE
CAST_TO_GFLOAT32_SR_OP float, short, true , LAPLACE
CAST_TO_GFLOAT32_SR_OP float, float, false, LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , false, LAPLACE

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___LAPLACE

CAST_TO_GFLOAT32_SR_OP float, float, true , LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , true , LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, float, false, LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , false, LOGISTIC

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___LOGISTIC

CAST_TO_GFLOAT32_SR_OP float, float, true , LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, LOGIT___NORMAL

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___LOGIT___NORMAL

.macro CAST_TO_GFLOAT32_SR_INPLACE_OP NANOO DENSITY
DEF_STACK_USAGE  0 __runCodelet_popfloat__experimental__CastToGfloat32SrInPlaceSupervisor___\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\()
.section .text.castToGfloat32SrInPlaceSupervisor_\NANOO\()_\DENSITY\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat32SrInPlaceSupervisor___\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\()
  .type __runCodelet_popfloat__experimental__CastToGfloat32SrInPlaceSupervisor___\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat32SrInPlaceSupervisor___\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\():
.supervisor
castToGfloat32SrInPlaceSupervisor_\NANOO\()_\DENSITY\():
  POPFLOAT_SUPERVISOR_CAST_OP castToGfloat32SrInPlace_\NANOO\()_\DENSITY\()

.worker
castToGfloat32SrInPlace_\NANOO\()_\DENSITY\():
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mCastParams, $mvertex_base, POPFLOAT_VBASE_CAST_INPLACE_ROUNDING_PARAM_OFFSET
  CAST_TO_GFLOAT32_SR 1 \NANOO \DENSITY 1
.size castToGfloat32SrInPlaceSupervisor_\NANOO\()_\DENSITY\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat32SrInPlaceSupervisor___\NANOO\()_popfloat__experimental__SRDensityType__\DENSITY\()
.endm

CAST_TO_GFLOAT32_SR_INPLACE_OP true , UNIFORM
CAST_TO_GFLOAT32_SR_INPLACE_OP false, UNIFORM

CAST_TO_GFLOAT32_SR_INPLACE_OP true , NORMAL
CAST_TO_GFLOAT32_SR_INPLACE_OP false, NORMAL

CAST_TO_GFLOAT32_SR_INPLACE_OP true , TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_INPLACE_OP false, TRUNCATED___NORMAL

CAST_TO_GFLOAT32_SR_INPLACE_OP true , BERNOULLI
CAST_TO_GFLOAT32_SR_INPLACE_OP false, BERNOULLI

CAST_TO_GFLOAT32_SR_INPLACE_OP true , LAPLACE
CAST_TO_GFLOAT32_SR_INPLACE_OP false, LAPLACE

CAST_TO_GFLOAT32_SR_INPLACE_OP true , TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_INPLACE_OP false, TRUNCATED___LAPLACE

CAST_TO_GFLOAT32_SR_INPLACE_OP true , LOGISTIC
CAST_TO_GFLOAT32_SR_INPLACE_OP false, LOGISTIC

CAST_TO_GFLOAT32_SR_INPLACE_OP true , TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_INPLACE_OP false, TRUNCATED___LOGISTIC

CAST_TO_GFLOAT32_SR_INPLACE_OP true , LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_INPLACE_OP false, LOGIT___NORMAL

CAST_TO_GFLOAT32_SR_INPLACE_OP true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_INPLACE_OP false, TRUNCATED___LOGIT___NORMAL

#endif
