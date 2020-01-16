// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
#include "CastToGfloat16Sr.h"
#include "GfloatConst.hpp"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT16_SR TYPE1 TYPE2 INPLACE
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
.if \INPLACE == 1
  ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_NANOO_OFFSET
.else
  ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_NANOO_OFFSET
.endif
.if \INPLACE == 1
    ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ROUND_MODE_OFFSET
.else
    ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ROUND_MODE_OFFSET
.endif
  setzi        $mCastToGF16   , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_bernoulli
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_NORMAL
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brz          $mRoundMode    , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_logit_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_LOGIT_NORMAL
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_truncated_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_TRUNCATED_LOGIT_NORMAL
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_logistic
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_LOGISTIC
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_uniform
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_UNIFORM
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_laplace
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_LAPLACE
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_truncated_normal
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_SR_DENSITY_TRUNCATED_NORMAL
  movz         $mCastToGF16   , $setMode              , $mRoundOp
.Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\():
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2)
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1
  brz          $mCount        , .Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcast_to_gfloat16Sr_inner_start_\TYPE1\()_\TYPE2\()_\INPLACE\():
.ifnc \TYPE1, float
  brnz         $enNanoo       , .LscaleInput_\TYPE1\()_\TYPE2\()_\INPLACE\()
  ld32         $inputClampF16 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET);
  f16v4clamp   $inValueV4     , $inValueV4            , $inputClampF16    // Clip values before scaling (CLAMP)
.LscaleInput_\TYPE1\()_\TYPE2\()_\INPLACE\():
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f16v4mul     $outValueV4    , $scalePair:BL         , $inValueV4        // Scale values
  }
.else
  ld64step     $inValueV2_1   , $mzero                , $mInRow+=         , 1
  brnz         $enNanoo       , .LscaleInput_\TYPE1\()_\TYPE2\()_\INPLACE\()
  ld64         $inputClampF32 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET/2);
  f32v2clamp   $inValueV2_0   , $inValueV2_0          , $inputClampF32    // Clip values before scaling (CLAMP)
  f32v2clamp   $inValueV2_1   , $inValueV2_1          , $inputClampF32    // Clip values before scaling (CLAMP)
.LscaleInput_\TYPE1\()_\TYPE2\()_\INPLACE\():
  f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0      // Scale values
  f32v2mul     $inValueV2_1   , $scaleFloat:B         , $inValueV2_1      // Scale values and generate Nan if value is outside the range
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f32v4tof16   $outValueV4    , $inValueF32V4                             // Copy f32v4 vector to f16.
  }
.endif
  and64        $expV4         , $outValueV4           , $halfExpMaskV4    // Extract exponents
  {
    st64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    f16v4cmpeq   $isDenormV4    , $azeros               , $expV4            // Check for ties
  }
  {
    ld64         $outBitMaskV4  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_NORM_MAN_MASK_OFFSET/2);
    and64        $isDenormV4    , $isDenormV4           , $outBitMaskV4
  }
  {
    br           $mCastToGF16
    or64         $outBitMaskV4  , $isDenormV4           , $outBitMaskV4
  }
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_bernoulli:
  st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
  {
.ifc \INPLACE, true
    ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET
.else
    ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET
.endif
    not64        $roundCorrV4   , $outBitMaskV4
  }  
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    f16v4rmask   $roundCorrV4   , $roundCorrV4          , $probBrnoulli
  }
  or64         $roundCorrV4   , $roundCorrV4          , $expV4
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4sub     $roundCorrV4   , $roundCorrV4          , $expV4
  }
  bri          .LaddCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_logit_normal:
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v2grand   $roundCorrV4_0
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_IN_PARAMS_OFFSET
    f16v2grand   $roundCorrV4_1
  }
  f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
  f16v2sigm    $roundCorrV4_0 , $roundCorrV4_0
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_CLAMP_OUT_PARAMS_OFFSET
    f16v2sigm    $roundCorrV4_1 , $roundCorrV4_1
  }
  {
    bri          .LmaskCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_uniform:
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    urand64      $roundCorrV4
  }
  {
    ld32          $scaleCorr    , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_OUT_PARAMS_OFFSET
    f16v4sufromui $roundCorrV4  , $roundCorrV4
  }
  bri          .LscaleCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_normal:
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v2grand   $roundCorrV4_0
  }
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_CLAMP_OUT_PARAMS_OFFSET
    f16v2grand   $roundCorrV4_1
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_OUT_PARAMS_OFFSET
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
  bri          .LscaleCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_truncated_normal:
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v2grand   $roundCorrV4_0
  }
  {
    st64         $azeros        , $mworker_base          , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_TRUNCATED_NORM_OFFSET/2)
    f16v2grand   $roundCorrV4_1
  }
  {
.ifc \INPLACE, true
    ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET
.else
    ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET
.endif
    and64        $maskOut       , $maskOut              , $azeros
  }
.LtruncatedNormal_loop_start_\TYPE1\()_\TYPE2\()_\INPLACE\():
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_CLAMP_OUT_PARAMS_OFFSET;
    andc64       $roundCorrV4   , $roundCorrV4          , $maskOut
  }
  f16v4clamp   $clampOut      , $roundCorrV4          , $clampCorr
  f16v4cmpeq   $clampOut      , $clampOut             , $roundCorrV4
  and64        $roundCorrV4   , $roundCorrV4          , $clampOut
  {
    ld64         $trncNorm      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_TRUNCATED_NORM_OFFSET/2);
    or64         $maskOut       , $maskOut              , $clampOut
  }
  atom         $maskOut_0     , $maskOut0
  {
    atom         $maskOut_1     , $maskOut1;
    or64         $trncNorm      , $trncNorm             , $roundCorrV4
  }
  and          $maskOut_0       , $maskOut_0            , $maskOut_1
  {
    xnor         $maskOut_0     , $maskOut_0            , $mzero;
    f16v2grand   $roundCorrV4_0
  }
  {
    st64         $trncNorm      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_TRUNCATED_NORM_OFFSET/2);
    f16v2grand   $roundCorrV4_1
  }
  brz          $maskOut_0     , .LtruncatedNormal_loop_end_\TYPE1\()_\TYPE2\()_\INPLACE\()
  brnzdec      $nIterations   , .LtruncatedNormal_loop_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  brnz         $maskOut_0     , .LtruncatedNormal_loop_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
.LtruncatedNormal_loop_end_\TYPE1\()_\TYPE2\()_\INPLACE\():
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_OUT_PARAMS_OFFSET;
    or64         $roundCorrV4   , $trncNorm             , $azeros
  }
  {
.if \INPLACE == 1
    ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ROUND_MODE_OFFSET
.else
    ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ROUND_MODE_OFFSET
.endif
    f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  }
  {
    cmpeq        $mRoundMode    , $mRoundMode           , POPFLOAT_SR_DENSITY_TRUNCATED_NORMAL
    f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
  }
  brnz         $mRoundMode    , .LmaskCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
  f16v2sigm    $roundCorrV4_0 , $roundCorrV4_0
  {
    bri          .LmaskCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
    f16v2sigm    $roundCorrV4_1 , $roundCorrV4_1
  }
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_laplace:
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    urand64      $roundCorrV4
  }
  {
    ld32          $scaleCorr    , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_IN_PARAMS_OFFSET
    f16v4sufromui $roundCorrV4  , $roundCorrV4
  }
  f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4cmplt   $scale         , $azeros               , $roundCorrV4      // Positive values
  }
  and64        $signV4        , $scale                , $signV4           // Flip sign
  f16v4absadd  $roundCorrV4   , $roundCorrV4          , $roundCorrV4
  setzi        $scalePair     , (POPFLOAT_FP16_EXPONENT_BIAS << POPFLOAT_NUM_FP16_MANTISSA_BITS)
  f16v4sub     $roundCorrV4   , $scalePair:BL           , $roundCorrV4
  f16v2ln      $roundCorrV4_0 , $roundCorrV4_0
  f16v2ln      $roundCorrV4_1 , $roundCorrV4_1
  f16v4add     $scale         , $scalePair:BL         , $azeros
  or64         $scale         , $scale                , $signV4           // Flip sign
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_CLAMP_OUT_PARAMS_OFFSET
    f16v4mul     $roundCorrV4   , $scale                , $roundCorrV4
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_OUT_PARAMS_OFFSET
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
  bri          .LscaleCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()
.LcalcCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\()_logistic:
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    urand64      $roundCorrV4
  }
  {
    ld32          $scaleCorr    , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_OUT_PARAMS_OFFSET
    f16v4sufromui $roundCorrV4  , $roundCorrV4
  }
  f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
  setzi        $scaleCorr     , ((POPFLOAT_FP16_EXPONENT_BIAS) << POPFLOAT_NUM_FP16_MANTISSA_BITS) // 1
  f16v4sub     $oneMinCorrV4  , $scaleCorr:BL         , $roundCorrV4
  f16v2ln      $roundCorrV4_0 , $roundCorrV4_0
  f16v2ln      $roundCorrV4_1 , $roundCorrV4_1
  f16v2ln      $oneMinCorrV4_0, $oneMinCorrV4_0
  f16v2ln      $oneMinCorrV4_1, $oneMinCorrV4_1
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_CLAMP_OUT_PARAMS_OFFSET
    f16v4sub     $roundCorrV4   , $roundCorrV4          , $oneMinCorrV4
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16SR_SCALE_OUT_PARAMS_OFFSET
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
.LscaleCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\():
  f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
.LmaskCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\():
  setzi        $halfMinDnrm   , ((POPFLOAT_FP16_EXPONENT_BIAS) << POPFLOAT_NUM_FP16_MANTISSA_BITS) // 1
  {
    ld64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
    f16v4add     $roundCorrV4   , $halfMinDnrm:BL       , $roundCorrV4
  }
  and64        $roundCorrV4   , $roundCorrV4          , $srMaskV4
  f16v2sub     $halfMinDnrm   , $azero                , $halfMinDnrm
  f16v4add     $roundCorrV4   , $halfMinDnrm:BL       , $roundCorrV4       // Add 1 lsb to inverted bits to set mantissa LSB
  {
    ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    setzi        $halfMinDnrm   , 1
  }
  not64        $outBitMaskV4  , $outBitMaskV4
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    f16v4add     $outBitMaskV4  , $halfMinDnrm:BL       , $outBitMaskV4      // Add 1 lsb to inverted bits to set mantissa LSB
  }
  or64         $outBitMaskV4  , $outBitMaskV4         , $expV4
  f16v4sub     $outBitMaskV4  , $outBitMaskV4         , $expV4
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4mul     $roundCorrV4   , $roundCorrV4          , $outBitMaskV4
  }
.LaddCorrection_\TYPE1\()_\TYPE2\()_\INPLACE\():
  {
.if \INPLACE == 1
    ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_NANOO_OFFSET
.else
    ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_NANOO_OFFSET
.endif
    and64        $signV4        , $outValueV4           , $signV4           // Extract signs
  }
  {
    ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v4absadd  $outValueV4    , $outValueV4           , $roundCorrV4
  }
  {
    ld32         $scaledMin     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET);
    and64        $outValueV4    , $outValueV4           , $outBitMaskV4     // Truncate matissa
  }
  {
    ld32         $scaledClamp   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET);
    f16v4cmple   $zeroOutMaskV4 , $scaledMin:BU         , $outValueV4
  }
  {
    brz          $enNanoo       , .LclampOutput_\TYPE1\()_\TYPE2\()_\INPLACE\()
    and64        $outValueV4    , $outValueV4           , $zeroOutMaskV4
  }
  f16v4cmplt   $outNanMaskV4  , $scaledClamp:BU       , $outValueV4
  {
    ld64         $qNanV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_QNAN_OUTPUT_OFFSET/2);
    andc64       $outValueV4    , $outValueV4           , $outNanMaskV4
  }
  and64        $outNanMaskV4  , $qNanV4               , $outNanMaskV4
  {
    ld32         $scaledClamp   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET);
    or64         $outValueV4    , $outNanMaskV4         , $outValueV4
  }
.LclampOutput_\TYPE1\()_\TYPE2\()_\INPLACE\():
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET/2)
    f16v4clamp   $outValueV4    , $outValueV4           , $scaledClamp
  }
  {
    ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1;
    or64         $outValueV4    , $outValueV4           , $signV4
  }
.ifc \TYPE2, half
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
    f16v4mul     $outValueV4    , $scalePair:BL         , $outValueV4       // Scale values
  }
.else
  f16v2tof32   $outValueV2_0  , $outValueV4_0
  f16v2tof32   $outValueV2_1  , $outValueV4_1
  f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0
  f32v2mul     $inValueV2_1   , $scaleFloat:B         , $inValueV2_1
.endif
  cmpult       $mRemainder    , $mCount               , 4
  brnz         $mRemainder    , .Lcast_to_gfloat16Sr_inner_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
  add          $mCount        , $mCount               , -4
.ifc \TYPE2, half  
  st64step     $outValueV4    , $mzero                , $mOutRow+=        , 1
.else
  st64step     $inValueV2_0   , $mzero                , $mOutRow+=        , 1
  st64step     $inValueV2_1   , $mzero                , $mOutRow+=        , 1
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
.endif  
  brnz         $mCount        , .Lcast_to_gfloat16Sr_inner_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  bri          .Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcast_to_gfloat16Sr_inner_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\():
  cmpult       $mRemainder    , $mCount               , 3
  brnz         $mRemainder    , .Lcast_to_gfloat16Sr_inner_last2_\TYPE1\()_\TYPE2\()_\INPLACE\()
.ifc \TYPE2, half
  {
    st32step     $outValueV4_0  , $mzero                , $mOutRow+=        , 1
    or           $outValueV4_0  , $outValueV4_1         , $azero
  }
.else
  {
    st64step     $inValueV2_0   , $mzero                , $mOutRow+=        , 1
    or64         $inValueV2_0   , $inValueV2_1          , $azeros
  }
.endif
  add          $mCount        , $mCount               , -2

.Lcast_to_gfloat16Sr_inner_last2_\TYPE1\()_\TYPE2\()_\INPLACE\():
  cmpult       $mRemainder    , $mCount               , 2
  brnz         $mRemainder    , .Lcast_to_gfloat16Sr_inner_last1_\TYPE1\()_\TYPE2\()_\INPLACE\()
.ifc \TYPE2, half
  bri          .Lcast_to_gfloat16Sr_inner_store_\TYPE1\()_\TYPE2\()_\INPLACE\()
.else
  st64step     $outValueV4    , $mzero                , $mOutRow+=        , 1
  bri          .Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
.endif
.Lcast_to_gfloat16Sr_inner_last1_\TYPE1\()_\TYPE2\()_\INPLACE\():
.ifc \TYPE2, half
  ldb16        $outValueV4_1  , $mzero                , $mOutRow          , 1
  sort4x16lo   $outValueV4_0  , $outValueV4_0         , $outValueV4_1
.else
.endif
.Lcast_to_gfloat16Sr_inner_store_\TYPE1\()_\TYPE2\()_\INPLACE\():
  st32step     $outValueV4_0  , $mzero                , $mOutRow+=        , 1

.Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\():
  brnzdec      $mRowCount     , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  exitz        $mzero
.endm

.macro CAST_TO_GFLOAT16_SR_OP TYPE1, TYPE2
.section .text.castToGfloat16Sr_\TYPE1\()_to_\TYPE2\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat16Sr___\TYPE1\()_\TYPE2\()
  .type __runCodelet_popfloat__experimental__CastToGfloat16Sr___\TYPE1\()_\TYPE2\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat16Sr___\TYPE1\()_\TYPE2\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
  ld64         $srMaskV4      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
  CAST_TO_GFLOAT16_SR \TYPE1, \TYPE2, 0

.size castToGfloat16Sr_\TYPE1\()_to_\TYPE2\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat16Sr___\TYPE1\()_\TYPE2\()
.endm

CAST_TO_GFLOAT16_SR_OP float, float
CAST_TO_GFLOAT16_SR_OP float, half 
CAST_TO_GFLOAT16_SR_OP half , half

.macro CAST_TO_GFLOAT16_SR_INPLACE_OP TYPE
.section .text.castToGfloat16SrInPlace_\TYPE\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat16SrInPlace___\TYPE\()
  .type __runCodelet_popfloat__experimental__CastToGfloat16SrInPlace___\TYPE\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat16SrInPlace___\TYPE\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
  ld64         $srMaskV4      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
  CAST_TO_GFLOAT16_SR \TYPE, \TYPE, 1

.size castToGfloat16SrInPlace_\TYPE\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat16SrInPlace___\TYPE\()
.endm

CAST_TO_GFLOAT16_SR_INPLACE_OP float
CAST_TO_GFLOAT16_SR_INPLACE_OP half 
#endif
