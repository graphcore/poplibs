#ifdef __IPU__
// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastToGfloat32Sr.h"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT32_SR_TRUNCATED_NORMAL INPLACE
.ifc \INPLACE, true
  ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET
.else
  ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET
.endif
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2grand   $roundCorrV2
  }
  {
    st64         $azeros        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_TRUNCATED_NORM_OFFSET/2)
    and64        $maskOut       , $maskOut              , $azeros
  }
1:
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
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
  brz         $maskOut_0        , 1f
  brnzdec     $nIterations      , 1b
  brnz        $maskOut_0        , 1b
1:
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
    or64         $roundCorrV2   , $trncNorm             , $azeros
  }
.endm

.macro CAST_TO_GFLOAT32_SR SAVEFP32 NANOO DENSITY INPLACE
  ld32         $mGf32Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET+1
  add          $mRowCount     , $mRowCount            , -1
  ld32         $enDenorm      , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET)
1:
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld64         $inValueV2     , $mzero                , $mInRow           , 0
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
.ifc \DENSITY, BERNOULLI
  {
    st64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    not64        $roundCorrV2   , $outBitMaskV2
  }
  {
.ifc \INPLACE, true
    ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET;
.else
    ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET;
.endif
    or64         $roundCorrV2   , $roundCorrV2          , $expV2
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2rmask   $roundCorrV2   , $roundCorrV2          , $probBrnoulli 
  }
.else
  {
    st64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2)
    not64        $roundCorrV2   , $outBitMaskV2
  }
  or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  and64        $roundCorrV2   , $roundCorrV2          , $fpExpMaskV2      // Extract exponent of result (half mantissa LSB)
  f32v2add     $manLsbMaskV2  , $roundCorrV2          , $roundCorrV2      // Mantissa LSB power
.ifc \DENSITY, LOGIT___NORMAL
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2grand   $roundCorrV2
  }
  ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  f32sigm      $roundCorrV2_0 , $roundCorrV2_0
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
    f32sigm      $roundCorrV2_1 , $roundCorrV2_1
  }
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
.else
.ifc \DENSITY, TRUNCATED___LOGIT___NORMAL
  CAST_TO_GFLOAT32_SR_TRUNCATED_NORMAL INPLACE
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  f32sigm      $roundCorrV2_0 , $roundCorrV2_0
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32sigm      $roundCorrV2_1 , $roundCorrV2_1
  }
.else
.ifc \DENSITY, TRUNCATED___LAPLACE
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
    f32v2sufromui $roundCorrV2  , $roundCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $sgnMaskV2    , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
  and64        $sgnMaskV2     , $corrDenorm           , $sgnMaskV2        // Flip sign
  f32v2absadd  $roundCorrV2   , $roundCorrV2          , $roundCorrV2
  or           $constOne      , $azero                , (POPFLOAT_FP32_EXPONENT_BIAS << POPFLOAT_NUM_FP32_MANTISSA_BITS)  // ONE
  f32v2add     $corrDenorm    , $constOne:B           , $azeros
  f32v2sub     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  f32ln        $roundCorrV2_0 , $roundCorrV2_0
  f32ln        $roundCorrV2_1 , $roundCorrV2_1
  or64         $corrDenorm    , $corrDenorm           , $sgnMaskV2         // Flip sign
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
    f32v2mul     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.else
.ifc \DENSITY, TRUNCATED___LOGISTIC
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  {
    ld64          $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
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
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $oneMinCorrV2
  }
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.else
.ifc \DENSITY, UNIFORM
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  {
    ld64          $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
    f32v2sufromui $roundCorrV2  , $roundCorrV2
  }
.else
.ifc \DENSITY, NORMAL
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2grand   $roundCorrV2
  }
  ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
.else
.ifc \DENSITY, TRUNCATED___NORMAL
  CAST_TO_GFLOAT32_SR_TRUNCATED_NORMAL \INPLACE
.else
.ifc \DENSITY, LAPLACE
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  {
    ld64          $sgnMaskV2    , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
    f32v2sufromui $roundCorrV2  , $roundCorrV2
  }
  f32v2cmpgt   $corrDenorm    , $azeros               , $roundCorrV2      // Positive values
  and64        $sgnMaskV2     , $corrDenorm           , $sgnMaskV2        // Flip sign
  f32v2absadd  $roundCorrV2   , $roundCorrV2          , $roundCorrV2
  or           $constOne      , $azero                , (POPFLOAT_FP32_EXPONENT_BIAS << POPFLOAT_NUM_FP32_MANTISSA_BITS)  // ONE
  f32v2add     $corrDenorm    , $constOne:B           , $azeros
  f32v2sub     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  f32ln        $roundCorrV2_0 , $roundCorrV2_0
  f32ln        $roundCorrV2_1 , $roundCorrV2_1
  or64         $corrDenorm    , $corrDenorm           , $sgnMaskV2         // Flip sign
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
    f32v2mul     $roundCorrV2   , $corrDenorm           , $roundCorrV2
  }
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }
.else
.ifc \DENSITY, LOGISTIC
  {
    st64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    urand64      $roundCorrV2
  }
  f32v2sufromui $roundCorrV2  , $roundCorrV2
  or           $constHalf     , $azero                , ((POPFLOAT_FP32_EXPONENT_BIAS-1) << POPFLOAT_NUM_FP32_MANTISSA_BITS)  // 1/2
  f32v2sub     $oneMinCorrV2  , $constHalf:B          , $roundCorrV2        // One minus correction
  f32v2add     $roundCorrV2   , $constHalf:B          , $roundCorrV2
  f32ln        $roundCorrV2_0 , $roundCorrV2_0
  f32ln        $roundCorrV2_1 , $roundCorrV2_1
  f32ln        $oneMinCorrV2_0, $oneMinCorrV2_0
  f32ln        $oneMinCorrV2_1, $oneMinCorrV2_1
  {
    ld64         $clampCorr     , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_CLAMP_PARAMS_OFFSET/2)
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $oneMinCorrV2
  }
  {
    ld64         $corrDenorm    , $mCorrParams          , $mzero            , (POPFLOAT_CAST_TO_GF32_SR_CORR_SCALE_PARAMS_OFFSET/2);
    f32v2clamp   $roundCorrV2   , $roundCorrV2          , $clampCorr
  }

.endif // .ifc \DENSITY, LOGISTIC
.endif // .ifc \DENSITY, LAPLACE
.endif  // .ifc \DENSITY, TRUNCATED___NORMAL
.endif  // .ifc \DENSITY, NORMAL
.endif  // .ifc \DENSITY, UNIFORM
  f32v2mul     $roundCorrV2   , $scaleCorr:B          , $roundCorrV2
  {
    ld64         $manLsbMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_MAN_LSB_OFFSET/2)
    f32v2add     $roundCorrV2   , $biasCorr:B           , $roundCorrV2
  }
.endif  // .ifc \DENSITY, TRUNCATED___LOGISTIC
.endif  // .ifc \DENSITY, TRUNCATED___LAPLACE
.endif // .ifc \DENSITY, TRUNCATED___LOGIT___NORMAL
.endif // .ifc \DENSITY, LOGIT___NORMAL
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
.endif  // .ifc \DENSITY, BERNOULLI
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
  f32v2cmple   $nonZeroV4     , $minValueGF32:B       , $inValueV2        // Mask for values greater-than or equal minDenorm
  {
    ld64         $fpClamp       , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
    and64        $inValueV2     , $inValueV2            , $nonZeroV4        // Set Values less than minDenorm to 0
  }
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

.macro CAST_TO_GFLOAT32_SR_OP TYPE1, TYPE2, NANOO, DENSITY
.section .text.castToGfloat32Sr_\TYPE1\()_to_\TYPE2\()_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\()
  .type __runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\():
  {
    ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
    or           $constOne      , $azero                , (126 << 23)
  }
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
  ld64         $srMaskV2      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
.ifc \TYPE1, \TYPE2
  CAST_TO_GFLOAT32_SR 1 \NANOO \DENSITY false
.else
  CAST_TO_GFLOAT32_SR 0 \NANOO \DENSITY false
.endif

  exitz        $mzero

.size castToGfloat32Sr_\TYPE1\()_to_\TYPE2\()_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat32Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\()

.endm

CAST_TO_GFLOAT32_SR_OP float, float, true , UNIFORM
CAST_TO_GFLOAT32_SR_OP float, short, true , UNIFORM
CAST_TO_GFLOAT32_SR_OP float, half , true , UNIFORM
CAST_TO_GFLOAT32_SR_OP float, float, false, UNIFORM
CAST_TO_GFLOAT32_SR_OP float, short, false, UNIFORM
CAST_TO_GFLOAT32_SR_OP float, half , false, UNIFORM

CAST_TO_GFLOAT32_SR_OP float, float, true , NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, true , NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, false, NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, NORMAL

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, true , TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, false, TRUNCATED___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___NORMAL

CAST_TO_GFLOAT32_SR_OP float, float, true , BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, short, true , BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, half , true , BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, float, false, BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, short, false, BERNOULLI
CAST_TO_GFLOAT32_SR_OP float, half , false, BERNOULLI

CAST_TO_GFLOAT32_SR_OP float, float, true , LAPLACE
CAST_TO_GFLOAT32_SR_OP float, short, true , LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , true , LAPLACE
CAST_TO_GFLOAT32_SR_OP float, float, false, LAPLACE
CAST_TO_GFLOAT32_SR_OP float, short, false, LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , false, LAPLACE

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, short, true , TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, short, false, TRUNCATED___LAPLACE
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___LAPLACE

CAST_TO_GFLOAT32_SR_OP float, float, true , LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, short, true , LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , true , LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, float, false, LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, short, false, LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , false, LOGISTIC

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, short, true , TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, short, false, TRUNCATED___LOGISTIC
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___LOGISTIC

CAST_TO_GFLOAT32_SR_OP float, float, true , LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, true , LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, false, LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, LOGIT___NORMAL

CAST_TO_GFLOAT32_SR_OP float, float, true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, float, false, TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, short, false, TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT32_SR_OP float, half , false, TRUNCATED___LOGIT___NORMAL

.macro CAST_TO_GFLOAT32_SR_INPLACE_OP NANOO DENSITY

.section .text.castToGfloat32SrInPlace_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat32SrInPlace___\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\()
  .type __runCodelet_experimental__popfloat__CastToGfloat32SrInPlace___\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat32SrInPlace___\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\():
  {
    ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
    or           $constOne      , $azero                , (126 << 23)
  }
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
  ld64         $srMaskV2      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
  CAST_TO_GFLOAT32_SR 1 \NANOO \DENSITY true

  exitz        $mzero

.size castToGfloat32SrInPlace_\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat32SrInPlace___\NANOO\()_experimental__popfloat__SRDensityType__\DENSITY\()
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
