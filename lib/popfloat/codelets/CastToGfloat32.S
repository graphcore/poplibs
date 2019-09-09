#ifdef __IPU__
// popfloat::CastToGloat32

#include "GfloatConst.hpp"
#include "CastToGfloat32.h"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT32 RMODE SAVEFP32 NANOO
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
  ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1
  ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2)
  and64        $sgnV2         , $inValueV2            , $sgnV2
  f32v2absadd  $inValueV2     , $inValueV2            , $azeros           // Take abs
.else
  {
    st64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2)
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
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
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
    ld64         $tmpMaskV2     , $mzero                , $mInRow           , 0;
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
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
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
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  f32v2cmple   $isPositiveV2  , $azeros               , $inValueV2        // Mask for positive values
  {
    ld64         $sgnV2         , $mGf32Param           , $mzero             , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
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
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  f32v2cmple   $isPositiveV2  , $azeros               , $inValueV2        // Mask for positive values
  {
    ld64         $sgnV2          , $mGf32Param          , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    andc64       $roundCorrV2    , $roundCorrV2         , $isPositiveV2     // Zero out correction for positive values
  }
.else
.ifc \RMODE, SX
  or64         $srExpV2       , $expV2                , $roundCorrV2
  {
    ld64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
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
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  {
    ld64         $halfMinMaskV2 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2);
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
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
    ld64         $sgnV2         , $mGf32Param           , $mzero            , (POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET/2);
    or64         $roundCorrV2   , $expV2                , $roundCorrV2      // Add exponent field
  }
  {
    ld64         $halfMinMaskV2 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_HALFMIN_MASK_OFFSET/2);
    f32v2sub     $roundCorrV2   , $roundCorrV2          , $expV2            // Subtract 2^Exp from correction
  }
  {
    ld64step     $inValueV2     , $mzero                , $mInRow+=         , 1;
    f32v2add     $roundCorrV2   , $roundCorrV2          , $halfMinMaskV2    // Add exponent correction for sub-denorms
  }
.endif // SR
.endif // SX
.endif // RD
.endif // RU
.endif // RN
.endif // RA
  and64        $sgnV2         , $inValueV2            , $sgnV2
  {
    ld64         $outBitMaskV2  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_OUT_BITMASK_OFFSET/2);
    f32v2absadd  $inValueV2     , $inValueV2            , $roundCorrV2      // Add correction
  }
.endif // RZ
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
    ld64         $qNanV2        , $mGf32Param             , $mzero          , (POPFLOAT_CAST_TO_GF32_PARAM_QNAN_MASK_OFFSET/2);
    andc64       $inValueV2     , $inValueV2              , $outNanMaskV2
  }
  {
    ld64         $fpClamp       , $mGf32Param             , $mzero          , (POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET/2);
    and64        $outNanMaskV2  , $qNanV2                 , $outNanMaskV2
  }
  {
    ld64         $sgnV2         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_GF16_SIGN_OFFSET/2);
    or64         $inValueV2     , $outNanMaskV2           , $inValueV2
  }
.endif
  {
    ld64         $inValueV2     , $mzero                , $mInRow           , 0;
    f32v2clamp   $tmpOutV2      , $inValueV2            , $fpClamp          // Clamp values to max float (Nans will propagate)
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

.macro CAST_TO_GFLOAT32_OP TYPE1, TYPE2, NANOO, RMODE
.section .text.castToGfloat32_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat32___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
  .type __runCodelet_experimental__popfloat__CastToGfloat32___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat32___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\():
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
.ifc \RMODE, SX
  ld32         $srManMask     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
  ld64         $srMaskV2      , $mzero                , $srManMask        , 0
  st64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
.endif
.ifc \TYPE1, \TYPE2
  CAST_TO_GFLOAT32 \RMODE 1 \NANOO
.else
  CAST_TO_GFLOAT32 \RMODE 0 \NANOO
.endif

  exitz        $mzero

.size castToGfloat32_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat32___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\()

.endm

CAST_TO_GFLOAT32_OP float, float, true  , RZ
CAST_TO_GFLOAT32_OP float, half , true  , RZ
CAST_TO_GFLOAT32_OP float, float, false , RZ
CAST_TO_GFLOAT32_OP float, half , false , RZ
                                                
CAST_TO_GFLOAT32_OP float, float, true  , RN
CAST_TO_GFLOAT32_OP float, half , true  , RN
CAST_TO_GFLOAT32_OP float, float, false , RN
CAST_TO_GFLOAT32_OP float, half , false , RN
                                                
CAST_TO_GFLOAT32_OP float, float, true  , RA
CAST_TO_GFLOAT32_OP float, half , true  , RA
CAST_TO_GFLOAT32_OP float, float, false , RA
CAST_TO_GFLOAT32_OP float, half , false , RA

CAST_TO_GFLOAT32_OP float, float, true  , RU
CAST_TO_GFLOAT32_OP float, half , true  , RU
CAST_TO_GFLOAT32_OP float, float, false , RU
CAST_TO_GFLOAT32_OP float, half , false , RU

CAST_TO_GFLOAT32_OP float, float, true  , RD
CAST_TO_GFLOAT32_OP float, half , true  , RD
CAST_TO_GFLOAT32_OP float, float, false , RD
CAST_TO_GFLOAT32_OP float, half , false , RD

CAST_TO_GFLOAT32_OP float, float, true  , SR
CAST_TO_GFLOAT32_OP float, half , true  , SR
CAST_TO_GFLOAT32_OP float, float, false , SR
CAST_TO_GFLOAT32_OP float, half , false , SR

CAST_TO_GFLOAT32_OP float, float, true  , SX
CAST_TO_GFLOAT32_OP float, half , true  , SX
CAST_TO_GFLOAT32_OP float, float, false , SX
CAST_TO_GFLOAT32_OP float, half , false , SX

.macro CAST_TO_GFLOAT32_INPLACE_OP NANOO RMODE

.section .text.castToGfloat32InPlace_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat32InPlace___\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
  .type __runCodelet_experimental__popfloat__CastToGfloat32InPlace___\NANOO\()_experimental__popfloat__RoundType__\RMODE\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat32InPlace___\NANOO\()_experimental__popfloat__RoundType__\RMODE\():

  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
.ifc \RMODE, SX
  ld32         $srManMask     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
  ld64         $srMaskV2      , $mzero                , $srManMask        , 0
  st64         $srMaskV2      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF32_STACK_SR_MASK_OFFSET/2)
.endif
  CAST_TO_GFLOAT32 \RMODE 1 \NANOO

  exitz        $mzero

.size castToGfloat32InPlace_\NANOO\()_\RMODE\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat32InPlace___\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
.endm

CAST_TO_GFLOAT32_INPLACE_OP true, RZ
CAST_TO_GFLOAT32_INPLACE_OP true, RN
CAST_TO_GFLOAT32_INPLACE_OP true, RA
CAST_TO_GFLOAT32_INPLACE_OP true, RU
CAST_TO_GFLOAT32_INPLACE_OP true, RD
CAST_TO_GFLOAT32_INPLACE_OP true, SR
CAST_TO_GFLOAT32_INPLACE_OP true, SX

CAST_TO_GFLOAT32_INPLACE_OP false, RZ
CAST_TO_GFLOAT32_INPLACE_OP false, RN
CAST_TO_GFLOAT32_INPLACE_OP false, RA
CAST_TO_GFLOAT32_INPLACE_OP false, RU
CAST_TO_GFLOAT32_INPLACE_OP false, RD
CAST_TO_GFLOAT32_INPLACE_OP false, SR
CAST_TO_GFLOAT32_INPLACE_OP false, SX
#endif
