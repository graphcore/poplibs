// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastToGfloat32Sr.h"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT32_SR SAVEFP32 INPLACE
  ld32         $mGf32Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET+1
  add          $mRowCount     , $mRowCount            , -1
  setzi        $mCastToGF32   , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_non_bernoulli
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_NORMAL
  movz         $mCastToGF32   , $setMode              , $mRoundOp
  brz          $mRoundMode    , 1f
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_logit_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_LOGIT_NORMAL
  movz         $mCastToGF32   , $setMode              , $mRoundOp
  brnz         $setMode       , 1f
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_truncated_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_TRUNCATED_LOGIT_NORMAL
  movz         $mCastToGF32   , $setMode              , $mRoundOp
  brnz         $setMode       , 1f
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_logistic
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_LOGISTIC
  movz         $mCastToGF32   , $setMode              , $mRoundOp
  brnz         $setMode       , 1f
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_uniform
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_UNIFORM
  movz         $mCastToGF32   , $setMode              , $mRoundOp
  brnz         $setMode       , 1f
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_laplace
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_LAPLACE
  movz         $mCastToGF32   , $setMode              , $mRoundOp
  brnz         $setMode       , 1f
  setzi        $mRoundOp      , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_truncated_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_TRUNCATED_NORMAL
  movz         $mCastToGF32   , $setMode              , $mRoundOp
1:
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld64         $inValueV2     , $mzero                , $mInRow           , 0
  ld32         $enDenorm      , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET)
2:
  ld64         $fpExpMaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2);
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
  {
.if \INPLACE == 1
    ld32         $isBernoulli   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ROUND_MODE_OFFSET;
.else
    ld32         $isBernoulli   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ROUND_MODE_OFFSET;
.endif
    or64         $roundCorrV2   , $roundCorrV2          , $expV2
  }
  {
    cmpeq        $isBernoulli   , $isBernoulli          , POPFLOAT_SR_DENSITY_BERNOULLI
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2
  }
  f32v2add     $manLsbMaskV2  , $roundCorrV2          , $roundCorrV2       // Mantissa LSB power
  {
    brz          $isBernoulli , .LcalcCorrection_\SAVEFP32\()_\INPLACE\()_non_bernoulli
    and64        $manLsbMaskV2, $manLsbMaskV2          , $fpExpMaskV2      // Extract exponent of result (half mantissa LSB)
  }
.if \INPLACE == 1
  ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET;
.else
  ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET;
.endif
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2rmask   $roundCorrV2   , $roundCorrV2          , $probBrnoulli 
  }
  ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1
  bri          .LaddCorrection_\SAVEFP32\()_\INPLACE\()
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_non_bernoulli:
  st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
  br           $mCastToGF32
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_logit_normal:
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_OUT_PARAMS_OFFSET/2);
    f32v2grand   $roundCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  f32sigm      $roundCorrV2_0 , $roundCorrV2_0
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_CLAMP_OUT_PARAMS_OFFSET/2)
    f32sigm      $roundCorrV2_1 , $roundCorrV2_1
  }
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
  bri          .LmaskBits_\SAVEFP32\()_\INPLACE\()
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_truncated_normal:
  {
.if \INPLACE == 1
    ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET
.else
    ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET
.endif
    f32v2grand   $roundCorrV2
  }
  {
    st64         $azeros        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_TRUNCATED_NORM_OFFSET/2)
    and64        $maskOut       , $maskOut              , $azeros
  }
.LtruncatedNormal_loop_start_\SAVEFP32\()_\INPLACE\():
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_CLAMP_OUT_PARAMS_OFFSET/2)
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
  brz         $maskOut_0        , .LtruncatedNormal_loop_end_\SAVEFP32\()_\INPLACE\()
  brnzdec     $nIterations      , .LtruncatedNormal_loop_start_\SAVEFP32\()_\INPLACE\()
  brnz        $maskOut_0        , .LtruncatedNormal_loop_start_\SAVEFP32\()_\INPLACE\()
.LtruncatedNormal_loop_end_\SAVEFP32\()_\INPLACE\():
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_OUT_PARAMS_OFFSET/2);
    or64         $roundCorrV2   , $trncNorm             , $azeros
  }
  {
.if \INPLACE == 1
    ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ROUND_MODE_OFFSET
.else
    ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ROUND_MODE_OFFSET
.endif
    f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  }
  {
    cmpeq        $mRoundMode    , $mRoundMode           , POPFLOAT_SR_DENSITY_TRUNCATED_NORMAL
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
  ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
  brnz         $mRoundMode    , .LmaskBits_\SAVEFP32\()_\INPLACE\()
  f32sigm      $roundCorrV2_0 , $roundCorrV2_0
  {
    bri          .LmaskBits_\SAVEFP32\()_\INPLACE\()
    f32sigm      $roundCorrV2_1 , $roundCorrV2_1
  }
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_laplace:
  urand64      $roundCorrV2
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_IN_PARAMS_OFFSET/2);
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
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_CLAMP_OUT_PARAMS_OFFSET/2)
    f32v2mul     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  }
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_OUT_PARAMS_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
  bri          .Lscale_samples_\SAVEFP32\()_\INPLACE\()
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_uniform:
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_OUT_PARAMS_OFFSET/2);
    urand64      $roundCorrV2
  }
  {
    bri          .Lscale_samples_\SAVEFP32\()_\INPLACE\()
    f32v2sufromui $roundCorrV2  , $roundCorrV2
  }
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_normal:
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_CLAMP_OUT_PARAMS_OFFSET/2)
    f32v2grand   $roundCorrV2
  }
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_OUT_PARAMS_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
  bri          .Lscale_samples_\SAVEFP32\()_\INPLACE\()
.LcalcCorrection_\SAVEFP32\()_\INPLACE\()_logistic:
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  {
    ld64          $corrDenorm   , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_IN_PARAMS_OFFSET/2);
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
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_CLAMP_OUT_PARAMS_OFFSET/2)
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $oneMinCorrV2
  }
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32SR_SCALE_OUT_PARAMS_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
.Lscale_samples_\SAVEFP32\()_\INPLACE\():
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.LmaskBits_\SAVEFP32\()_\INPLACE\():
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1
    f32v2mul     $roundCorrV2   , $roundCorrV2          , $manLsbMaskV2
  }
  {
    ld64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
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
.LaddCorrection_\SAVEFP32\()_\INPLACE\():
  {
    ld32         $enDenorm      , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET)
    and64        $sgnV2         , $inValueV2            , $sgnV2
  }
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
.if \INPLACE == 1
  ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_NANOO_OFFSET
.else
  ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_NANOO_OFFSET
.endif
  {
    brz          $enNanoo       , 3f
    and64        $inValueV2     , $inValueV2            , $nonZeroV4        // Set Values less than minDenorm to 0
  }
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
3:
  {
    ld64         $inValueV2     , $mzero                , $mInRow           , 0;
    f32v2clamp   $tmpOutV2      , $inValueV2            , $fpClamp          // Clamp values to max float (Nans will not propagate
  }
  {
    cmpeq        $mRemainder    , $mCount               , 1
    or64         $outV2         , $tmpOutV2             , $sgnV2
  }
  brnz         $mRemainder    , 6f
.if \SAVEFP32 == 1
  st64step     $outV2         , $mzero                , $mOutRow+=        , 1;
.else
  f32v2tof16   $out0          , $outV2
  st32step     $out0          , $mzero                , $mOutRow+=        , 1;
.endif
  add          $mCount        , $mCount               , -2
  brnz         $mCount        , 2b
  bri          7f
6:
.if \SAVEFP32 == 0
  {
    ldb16        $outV2_1       , $mzero                , $mOutRow          , 1
    f32tof16     $outV2_0       , $outV2_0
  }
  roll16       $outV2_0       , $outV2_0              , $outV2_1
.endif
  st32         $outV2_0       , $mzero                , $mOutRow          , 0
7:
  brnzdec      $mRowCount     , 1b;
.endm

.macro CAST_TO_GFLOAT32_SR_OP TYPE1, TYPE2
.section .text.castToGfloat32Sr_\TYPE1\()_to_\TYPE2\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\()
  .type __runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\():
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
  ld64         $srMaskV2      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
  ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ROUND_MODE_OFFSET
.ifc \TYPE1, \TYPE2
  CAST_TO_GFLOAT32_SR 1 0
.else
  CAST_TO_GFLOAT32_SR 0 0
.endif

  exitz        $mzero

.size castToGfloat32Sr_\TYPE1\()_to_\TYPE2\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\()

.endm

CAST_TO_GFLOAT32_SR_OP float, float
CAST_TO_GFLOAT32_SR_OP float, half

.macro CAST_TO_GFLOAT32_SR_INPLACE_OP

.section .text.castToGfloat32SrInPlace
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat32SrInPlace
  .type __runCodelet_experimental__popfloat__CastToGfloat32SrInPlace, @function
  __runCodelet_experimental__popfloat__CastToGfloat32SrInPlace:
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
  ld64         $srMaskV2      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
  ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ROUND_MODE_OFFSET
  CAST_TO_GFLOAT32_SR 1 1

  exitz        $mzero

.size castToGfloat32SrInPlace,\
  .-__runCodelet_experimental__popfloat__CastToGfloat32SrInPlace
.endm

CAST_TO_GFLOAT32_SR_INPLACE_OP

#endif
