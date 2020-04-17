// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastToGfloat32.h"
#include "arch/gc_tile_defines.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

.macro CAST_TO_GFLOAT32 NANOO RMODE SAVEFP32 INPLACE
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
  bri          2f
1:
.if \SAVEFP32 == 1
  st64step     $outV2         , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.else
  f32v2tof16   $out0          , $outV2
  st32step     $out0          , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.endif
2:
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
.ifc \RMODE, RZ
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS
    not64        $roundCorrV2   , $outBitMaskV2
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    and64        $roundCorrV2   , $roundCorrV2          , $azeros
  }
.else
  {
    st64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    not64        $roundCorrV2   , $outBitMaskV2
  }
.ifc \RMODE, RA
  {
    ld64         $halfMinMaskV2 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  f32v2cmpgt   $halfMinMaskV2 , $expV2                , $halfMinMaskV2
  {
    ld64         $fpExpMaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2)
    and64        $expV2         , $expV2                , $halfMinMaskV2
  }
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    and64        $roundCorrV2   , $roundCorrV2          , $fpExpMaskV2      // Correction is half the mantissa LSB
  }
.else
.ifc \RMODE, RN
  {
    ld64         $halfMinMaskV2 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  f32v2cmpge   $halfMinMaskV2 , $expV2                , $halfMinMaskV2
  and64        $halfMinMaskV2 , $expV2                , $halfMinMaskV2
  {
    ld64         $fpExpMaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET/2)
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $halfMinMaskV2    // Subtract 2^Exp from correction
  }
  and64        $roundCorrV2   , $roundCorrV2          , $fpExpMaskV2      // Extract exponent of result (half mantissa LSB)
  {
    st64         $roundCorrV2   , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_CORRECTION_OFFSET/2);
    f32v2add     $manLsbMaskV2  , $roundCorrV2          , $roundCorrV2      // Mantissa LSB power
  }
  {
    ld64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    f32v2add     $manLsbMaskV2  , $expV2                , $manLsbMaskV2     // Set LSB to 1
  }
  {
    ld64         $tmpMaskV2     , $mzero                , $mBaseIn          , 0;
    andc64       $manLsbMaskV2  , $manLsbMaskV2         , $fpExpMaskV2      // Extract mantissa
  }
  and64        $manLsbMaskV2  , $manLsbMaskV2         , $tmpMaskV2        // Extract mantissa LSB
  or64         $manLsbMaskV2  , $manLsbMaskV2         , $expV2            // Set exponent bits
  f32v2sub     $manLsbMaskV2  , $manLsbMaskV2         , $expV2            // Subtract 2^Exp from correction
  andc64       $tmpMaskV2     , $tmpMaskV2            , $outBitMaskV2     // Extract truncated bits
  or64         $tmpMaskV2     , $expV2                , $tmpMaskV2        // Set exponent bits
  {
    ld64         $roundCorrV2   , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_CORRECTION_OFFSET/2);
    f32v2sub     $tmpMaskV2     , $tmpMaskV2            , $expV2            // Subtract 2^Exp from correction
  }
  f32v2cmpeq   $isTieV2       , $roundCorrV2          , $tmpMaskV2
  and64        $manLsbMaskV2  , $manLsbMaskV2         , $isTieV2
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    andc64       $roundCorrV2   , $roundCorrV2          , $isTieV2
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    or64         $roundCorrV2   , $roundCorrV2          , $manLsbMaskV2
  }
.else
.ifc \RMODE, RU
  {
    ld64         $halfMinMaskV2 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  f32v2cmpgt   $halfMinMaskV2 , $expV2                , $halfMinMaskV2
  and64        $expV2         , $expV2                , $halfMinMaskV2
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  f32v2cmple   $isPositiveV2  , $azeros               , $inValueV2        // Mask for positive values
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    and64        $roundCorrV2   , $roundCorrV2          , $isPositiveV2      // Zero out correction for negative values
  }
.else
.ifc \RMODE, RD
  {
    ld64         $halfMinMaskV2 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  f32v2cmpgt   $halfMinMaskV2 , $expV2                , $halfMinMaskV2
  and64        $expV2         , $expV2                , $halfMinMaskV2
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  f32v2cmple   $isPositiveV2  , $azeros               , $inValueV2        // Mask for positive values
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    andc64       $roundCorrV2    , $roundCorrV2         , $isPositiveV2     // Zero out correction for positive values
  }
.else
.ifc \RMODE, SX
  or64         $srExpV2       , $expV2                , $roundCorrV2
  {
    ld64         $srMaskV2      , $mzero                , $mCastParams      , (POPFLOAT_CAST_PARAMS_SR_MASK_OFFSET/2)
    f32v2sub     $srExpV2       , $srExpV2              , $expV2
  }
  and64        $srExpV2       , $srExpV2              , $srMaskV2
  f32v2add     $srExpV2       , $expV2                , $srExpV2
  and64        $roundCorrV2   , $roundCorrV2          , $srExpV2
  ld64         $fpHalfMinGF32 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
  {
    st64         $azeros        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2);
    f32v2cmpeq   $isHalfMinV2   , $fpHalfMinGF32        , $expV2
  }
  {
    ld64         $bit23MaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_BIT23_MASK_OFFSET/2);
    or           $isHalfMin     , $isHalfMin0           , $isHalfMin1
  }
  {
    atom         $enHalfMin     , $isHalfMin;
    urand64      $randomBitsV2
  }
  {
    brz          $enHalfMin     , 4f;
    and64        $roundCorrV2   , $randomBitsV2         , $roundCorrV2      // Apply truncate mask to random bits
  }
  and64        $halfMinMaskV2 , $bit23MaskV2          , $randomBitsV2
  {
    ld64         $fpHalfMinGF32 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
    f32v2cmpeq   $bit23MaskV2   , $bit23MaskV2          , $halfMinMaskV2    // For sub-denorms, keep those with bit-23 set to 1
  }
  f32v2cmpeq   $halfMinMaskV2 , $fpHalfMinGF32        , $expV2            // Mask for floats >= fpMinDenorm
  and64        $halfMinMaskV2 , $bit23MaskV2          , $halfMinMaskV2    // Enable if bit23 is set and enHalfMinV2 (exp<minDenorm)
  and64        $halfMinMaskV2 , $halfMinMaskV2        , $expV2            // Set half min correction to exponent
  st64         $halfMinMaskV2 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2)
4:
  {
    ld64         $halfMinMaskV2 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2);
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    f32v2add     $roundCorrV2   , $roundCorrV2          , $halfMinMaskV2    // Add exponent correction for sub-denorms
  }
.else
.ifc \RMODE, SR
  ld64         $fpHalfMinGF32 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
  {
    st64         $azeros        , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2);
    f32v2cmpeq   $isHalfMinV2   , $fpHalfMinGF32        , $expV2
  }
  {
    ld64         $bit23MaskV2   , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_BIT23_MASK_OFFSET/2);
    or           $isHalfMin     , $isHalfMin0           , $isHalfMin1
  }
  {
    atom         $enHalfMin     , $isHalfMin;
    urand64      $randomBitsV2
  }
  {
    brz          $enHalfMin     , 4f;
    and64        $roundCorrV2   , $randomBitsV2         , $roundCorrV2      // Apply truncate mask to random bits
  }
  and64        $halfMinMaskV2 , $bit23MaskV2          , $randomBitsV2
  {
    ld64         $fpHalfMinGF32 , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET/2)
    f32v2cmpeq   $bit23MaskV2   , $bit23MaskV2          , $halfMinMaskV2    // For sub-denorms, keep those with bit-23 set to 1
  }
  f32v2cmpeq   $halfMinMaskV2 , $fpHalfMinGF32        , $expV2            // Mask for floats >= fpMinDenorm
  and64        $halfMinMaskV2 , $bit23MaskV2          , $halfMinMaskV2    // Enable if bit23 is set and enHalfMinV2 (exp<minDenorm)
  and64        $halfMinMaskV2 , $halfMinMaskV2        , $expV2            // Set half min correction to exponent
  st64         $halfMinMaskV2 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2)
4:
  {
    ld64         $halfMinMaskV2 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2);
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  {
    ld64step     $inValueV2     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    f32v2add     $roundCorrV2   , $roundCorrV2          , $halfMinMaskV2    // Add exponent correction for sub-denorms
  }
.endif // .ifc \RMODE, SR
.endif // .ifc \RMODE, SX
.endif // .ifc \RMODE, RD
.endif // .ifc \RMODE, RU
.endif // .ifc \RMODE, RN
.endif // .ifc \RMODE, RA
.endif // .ifc \RMODE, RZ
  and64        $sgnV2         , $inValueV2            , $sgnV2
  {
    ld64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
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
    brnzdec      $mCount        , 1b
    andc64       $outV2         , $outV2                , $tmpOutV2         // Convert all -0.0 into +0.0
  }
  brnz         $mRemainder    , 6f
.if \SAVEFP32 == 1
  st64step     $outV2         , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.else
  f32v2tof16   $out0          , $outV2
  st32step     $out0          , $mzero                , $mBaseOut+=       , CTXT_WORKERS;
.endif
  exitz        $mzero
6:
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

.macro CAST_TO_GFLOAT32_OP TYPE1 TYPE2 NANOO RMODE
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastToGfloat32Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.section .text.castToGfloat32Supervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat32Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
  .type __runCodelet_popfloat__experimental__CastToGfloat32Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat32Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\():
.supervisor
castToGfloat32Supervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\():
  POPFLOAT_SUPERVISOR_CAST_OP castToGfloat32_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\()

.worker
castToGfloat32_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\():
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mCastParams, $mvertex_base, POPFLOAT_VBASE_CAST_ROUNDING_PARAM_OFFSET
.ifc \TYPE1, \TYPE2
  CAST_TO_GFLOAT32 \NANOO \RMODE 1 0
.else
  CAST_TO_GFLOAT32 \NANOO \RMODE 0 0
.endif

.size castToGfloat32Supervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat32Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.endm

CAST_TO_GFLOAT32_OP float, float, true, RZ
CAST_TO_GFLOAT32_OP float, float, true, RA
CAST_TO_GFLOAT32_OP float, float, true, RN
CAST_TO_GFLOAT32_OP float, float, true, RU
CAST_TO_GFLOAT32_OP float, float, true, RD
CAST_TO_GFLOAT32_OP float, float, true, SR
CAST_TO_GFLOAT32_OP float, float, true, SX

CAST_TO_GFLOAT32_OP float, half, true, RZ
CAST_TO_GFLOAT32_OP float, half, true, RA
CAST_TO_GFLOAT32_OP float, half, true, RN
CAST_TO_GFLOAT32_OP float, half, true, RU
CAST_TO_GFLOAT32_OP float, half, true, RD
CAST_TO_GFLOAT32_OP float, half, true, SR
CAST_TO_GFLOAT32_OP float, half, true, SX

CAST_TO_GFLOAT32_OP float, float, false, RZ
CAST_TO_GFLOAT32_OP float, float, false, RA
CAST_TO_GFLOAT32_OP float, float, false, RN
CAST_TO_GFLOAT32_OP float, float, false, RU
CAST_TO_GFLOAT32_OP float, float, false, RD
CAST_TO_GFLOAT32_OP float, float, false, SR
CAST_TO_GFLOAT32_OP float, float, false, SX

CAST_TO_GFLOAT32_OP float, half, false, RZ
CAST_TO_GFLOAT32_OP float, half, false, RA
CAST_TO_GFLOAT32_OP float, half, false, RN
CAST_TO_GFLOAT32_OP float, half, false, RU
CAST_TO_GFLOAT32_OP float, half, false, RD
CAST_TO_GFLOAT32_OP float, half, false, SR
CAST_TO_GFLOAT32_OP float, half, false, SX

.macro CAST_TO_GFLOAT32_INPLACE_OP NANOO RMODE
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastToGfloat32InPlaceSupervisor___\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.section .text.castToGfloat32InPlaceSupervisor_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat32InPlaceSupervisor___\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
  .type __runCodelet_popfloat__experimental__CastToGfloat32InPlacecastToGfloat32InPlaceSupervisor___\NANOO\()_popfloat__experimental__RoundType__\RMODE\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat32InPlaceSupervisor___\NANOO\()_popfloat__experimental__RoundType__\RMODE\():
.supervisor
castToGfloat32InPlaceSupervisor_\NANOO\()_\RMODE\():
  POPFLOAT_SUPERVISOR_CAST_OP castToGfloat32InPlace_\NANOO\()_\RMODE\()

.worker
castToGfloat32InPlace_\NANOO\()_\RMODE\():
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mCastParams, $mvertex_base, POPFLOAT_VBASE_CAST_INPLACE_ROUNDING_PARAM_OFFSET
  CAST_TO_GFLOAT32 \NANOO \RMODE 1 1

.size castToGfloat32InPlaceSupervisor_\NANOO\()_\RMODE\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat32InPlaceSupervisor___\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.endm

CAST_TO_GFLOAT32_INPLACE_OP true, RZ
CAST_TO_GFLOAT32_INPLACE_OP true, RA
CAST_TO_GFLOAT32_INPLACE_OP true, RN
CAST_TO_GFLOAT32_INPLACE_OP true, RU
CAST_TO_GFLOAT32_INPLACE_OP true, RD
CAST_TO_GFLOAT32_INPLACE_OP true, SR
CAST_TO_GFLOAT32_INPLACE_OP true, SX

CAST_TO_GFLOAT32_INPLACE_OP false, RZ
CAST_TO_GFLOAT32_INPLACE_OP false, RA
CAST_TO_GFLOAT32_INPLACE_OP false, RN
CAST_TO_GFLOAT32_INPLACE_OP false, RU
CAST_TO_GFLOAT32_INPLACE_OP false, RD
CAST_TO_GFLOAT32_INPLACE_OP false, SR
CAST_TO_GFLOAT32_INPLACE_OP false, SX

#endif
