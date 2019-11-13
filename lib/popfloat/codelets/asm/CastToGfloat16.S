#ifdef __IPU__
#include "CastToGfloat16.h"
#include "GfloatConst.hpp"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT16 TYPE1 TYPE2 INPLACE
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
.if \INPLACE == 1
  ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_NANOO_OFFSET
.else
  ld32         $enNanoo       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_NANOO_OFFSET
.endif
  setzi        $mCastToGF16   , .Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()
.if \INPLACE == 1
  ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_ROUND_MODE_OFFSET
.else
  ld32         $mRoundMode    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_ROUND_MODE_OFFSET
.endif
  brz          $mRoundMode    , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
.if \INPLACE == 1
  ld32         $srManMask     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
.else
  ld32         $srManMask     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
.endif
  ld64         $srMaskV4      , $mzero                , $srManMask        , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
  setzi        $mCastToGF16   , .Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_sx
  setzi        $mRoundOp      , .Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_sr
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_ROUND_STOCHASTIC
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_ra
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_ROUND_TO_NEAREST_AWAY
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  setzi        $mRoundOp      , .Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_rn
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_ROUND_TO_NEAREST_EVEN
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  {
    ld64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);  
    setzi        $scaleHalf     , 0x3800
  }
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
  setzi        $mRoundOp      , .Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_ru
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_ROUND_TO_POS_INF
  movz         $mCastToGF16   , $setMode              , $mRoundOp
  brnz         $setMode       , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  {
    ld64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);  
    setzi        $scaleHalf     , 0x3800
  }
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
  setzi        $mRoundOp      , .Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_rd
  cmpeq        $setMode       , $mRoundMode           , POPFLOAT_ROUND_TO_NEG_INF
  movz         $mCastToGF16   , $setMode              , $mRoundOp
.Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\():
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2)
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1
  brz          $mCount        , .Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcast_to_gfloat16_inner_start_\TYPE1\()_\TYPE2\()_\INPLACE\():
.ifnc \TYPE1, float
  ld32         $inputClampF16 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET);
  brnz         $enNanoo       , 1f
  f16v4clamp   $inValueV4     , $inValueV4            , $inputClampF16    // Clip values before scaling (CLAMP)
1:
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $inValueV4        // Scale values
  }
.else
  ld64step     $inValueV2_1   , $mzero                , $mInRow+=         , 1;
  brz          $enNanoo       , 1f
  ld64         $inputClampF32 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET/2);
  f32v2clamp   $inValueV2_0   , $inValueV2_0          , $inputClampF32    // Clip values before scaling (CLAMP)
  f32v2clamp   $inValueV2_1   , $inValueV2_1          , $inputClampF32    // Clip values before scaling (CLAMP)
1:
  f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0      // Scale values
  f32v2mul     $inValueV2_1   , $scaleFloat:B         , $inValueV2_1      // Scale values and generate Nan if value is outside the range
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f32v4tof16   $outValueV4    , $inValueF32V4                             // Copy f32v4 vector to f16.
  }
.endif
.Lcalculate_truncation_mask_\TYPE1\()_\TYPE2\()_\INPLACE\():
  {
    ld64         $outBitMaskV4  , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);
    and64        $expV4         , $outValueV4           , $halfExpMaskV4    // Extract exponents
  }
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
.Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_ra:
  {
    st64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUTPUT_OFFSET/2);
    setzi        $halfMinDnrm   , 1
  }
  not64        $roundCorrV4   , $outBitMaskV4
  f16v4add     $roundCorrV4   , $halfMinDnrm:BL       , $roundCorrV4      // Add 1 lsb to inverted bits to set mantissa LSB
  {
    ld32         $scalePm1      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M1_OFFSET);
    and64        $roundCorrV4   , $roundCorrV4          , $outBitMaskV4
  }
  {
    ld64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUTPUT_OFFSET/2);
    f16v4mul     $roundCorrV4   , $scalePm1:BL          , $roundCorrV4
  }
  bri          .Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_rn:
  {
    st64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    setzi        $halfMinDnrm   , 1
  }
  {
    st64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUTPUT_OFFSET/2);
    not64        $roundCorrV4   , $outBitMaskV4
  }
  {
    ld32         $scalePm1      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M1_OFFSET);
    f16v4add     $manLsbMaskV4  , $halfMinDnrm:BL       , $roundCorrV4      // Add 1 lsb to inverted bits to set mantissa LSB
  }
  {
    ld64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUTPUT_OFFSET/2);
    f16v4mul     $roundCorrV4   , $scalePm1:BL          , $manLsbMaskV4
  }
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2);
    andc64       $truncBitsV4   , $outValueV4           , $outBitMaskV4     // Extract to-be-truncated bits
  }
  and64        $manLsbMaskV4  , $manLsbMaskV4         , $outValueV4       // Extract LSB
  f16v4cmpeq   $isTie         , $roundCorrV4          , $truncBitsV4      // Check for ties
  and64        $manLsbMaskV4  , $manLsbMaskV4         , $isTie            // Set correction for Ties
  {
    ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2);
    andc64       $roundCorrV4   , $roundCorrV4          , $isTie            // Correction for other truncated bit batterns
  }
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    or64         $roundCorrV4   , $roundCorrV4          , $manLsbMaskV4     // Create RN mask
  }
  bri          .Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_ru:
  {
    ld64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
    f16v4absadd  $isPositiveV4  , $azeros               , $outValueV4
  }
  f16v4cmplt   $isPositiveV4  , $isPositiveV4         , $halfMinDnrmV4    // Abs is less than half min
  andc64       $roundCorrV4   , $outValueV4           , $isPositiveV4     // Zero-out abs is less than half min
  f16v4cmplt   $isPositiveV4  , $azeros               , $roundCorrV4
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    andc64       $roundCorrV4   , $isPositiveV4         , $outBitMaskV4     // Mask correction bits
  }
  bri          .Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_rd:
  {
    ld64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
    f16v4absadd  $isPositiveV4  , $azeros               , $outValueV4
  }
  f16v4cmplt   $isPositiveV4  , $isPositiveV4         , $halfMinDnrmV4    // Abs is less than half min
  andc64       $roundCorrV4   , $outValueV4           , $isPositiveV4     // Zero-out abs is less than half min
  f16v4cmplt   $isNegativeV4  , $roundCorrV4          , $azeros
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    andc64       $roundCorrV4   , $isNegativeV4         , $outBitMaskV4     // Mask correction bits
  }
  bri          .Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_sx:
  {
    ld32         $scaleP10      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET);
    not64        $roundCorrV4   , $outBitMaskV4                             // Truncated bits
  }
  {
    ld64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
    f16v4mul     $roundCorrV4   , $scaleP10:BL          , $roundCorrV4      // Treat truncated bits as a denorm, then convert to a norm FP16 value
  }
  {
    ld32         $scaleP10      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET);
    and64        $roundCorrV4   , $roundCorrV4          , $srMaskV4
  }
  f16v4mul     $manLsbMaskV4  , $scaleP10:BU          , $roundCorrV4      // Scale down to de-normalise round correction
  urand64      $randomBitsV4                                              // Generate PRNG bits
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2)
    and64        $randomBitsV4  , $randomBitsV4         , $manLsbMaskV4
  }
  bri          .Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()
.Lcalc_correction_\TYPE1\()_\TYPE2\()_\INPLACE\()_sr:
  urand64      $randomBitsV4                                              // Generate random bit pattern
  andc64       $roundCorrV4   , $randomBitsV4         , $outBitMaskV4     // Apply SR bit mask
.Ladd_correction_\TYPE1\()_\TYPE2\()_\INPLACE\():
  or64         $roundCorrV4   , $expV4                , $roundCorrV4      // Add exponent to truncated bits
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4sub     $roundCorrV4   , $roundCorrV4          , $expV4            // Subtract exponent from correct
  }
  and64        $signV4        , $outValueV4           , $signV4           // Extract signs
  {
    ld32         $scaledMin     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET);
    f16v4absadd  $outValueV4    , $outValueV4           , $roundCorrV4      // Add correction
  }
  and64        $outValueV4    , $outValueV4           , $outBitMaskV4     // Truncate matissa
  {
    ld32         $scaledClamp   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET);
    f16v4cmple   $zeroOutMaskV4 , $scaledMin:BU         , $outValueV4
  }
  {
    brz            $enNanoo       , .LclampOutput_\TYPE1\()_\TYPE2\()_\INPLACE\()
    and64          $outValueV4    , $outValueV4           , $zeroOutMaskV4
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
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET/2);    
    f16v4clamp   $outValueV4    , $outValueV4           , $scaledClamp
  }
  {
    ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1;
    or64         $outValueV4    , $outValueV4           , $signV4
  }
  cmpult       $mRemainder    , $mCount               , 4
.ifc \TYPE2, half
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $outValueV4       // Scale values
  }
  brnz         $mRemainder    , .Lcast_to_gfloat16_inner_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
  add          $mCount        , $mCount               , -4
  st64step     $outValueV4    , $mzero                , $mOutRow+=        , 1
.else
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET/2);    
    f16v2tof32   $outValueV2_0  , $outValueV4_0
  }
  {
    brnz         $mRemainder    , .Lcast_to_gfloat16_inner_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()
    f16v2tof32   $outValueV2_1  , $outValueV4_1
  }
  {
    add          $mCount        , $mCount               , -4
    f32v2mul     $outValueV2_0  , $scaleFloat:B         , $outValueV2_0     // Scale values
  }
  {
    st64step     $outValueV2_0  , $mzero                , $mOutRow+=        , 1
    f32v2mul     $outValueV2_1  , $scaleFloat:B         , $outValueV2_1     // Scale values
  }
  st64step     $outValueV2_1  , $mzero                , $mOutRow+=        , 1
.endif
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
  brnz         $mCount        , .Lcast_to_gfloat16_inner_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  bri          .Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()

.Lcast_to_gfloat16_inner_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\():
.ifc \TYPE2, half
  cmpult       $mRemainder    , $mCount               , 3
.else
  {
    cmpult       $mRemainder    , $mCount               , 3
    f32v2mul     $outValueV2_0  , $scaleFloat:B         , $outValueV2_0     // Scale values
  }
.endif
  brnz         $mRemainder    , .Lcast_to_gfloat16_inner_last2_\TYPE1\()_\TYPE2\()_\INPLACE\()
.ifc \TYPE2, half
  {
    st32step     $outValueV4_0  , $mzero                , $mOutRow+=        , 1
    or           $outValueV4_0  , $outValueV4_1         , $azero
  }
.else
  {
    st64step     $outValueV2_0  , $mzero                , $mOutRow+=        , 1
    f32v2mul     $outValueV2_0  , $scaleFloat:B         , $outValueV2_1      // Scale values
  }
.endif
  add          $mCount        , $mCount               , -2

.Lcast_to_gfloat16_inner_last2_\TYPE1\()_\TYPE2\()_\INPLACE\():
  cmpult       $mRemainder    , $mCount               , 2
  brnz         $mRemainder    , .Lcast_to_gfloat16_inner_last1_\TYPE1\()_\TYPE2\()_\INPLACE\()
  bri          .Lcast_to_gfloat16_inner_store_\TYPE1\()_\TYPE2\()_\INPLACE\()

.Lcast_to_gfloat16_inner_last1_\TYPE1\()_\TYPE2\()_\INPLACE\():
.ifc \TYPE2, half
  ldb16        $outValueV4_1  , $mzero                , $mOutRow          , 1
  sort4x16lo   $outValueV4_0  , $outValueV4_0         , $outValueV4_1
.endif

.Lcast_to_gfloat16_inner_store_\TYPE1\()_\TYPE2\()_\INPLACE\():
  st32step     $outValueV4_0  , $mzero                , $mOutRow+=        , 1

.Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\():
  brnzdec      $mRowCount     , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\INPLACE\()
  exitz        $mzero
.endm

.macro CAST_TO_GFLOAT16_OP TYPE1 TYPE2
.section .text.castToGfloat16_\TYPE1\()_to_\TYPE2\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\()
  .type __runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  CAST_TO_GFLOAT16 \TYPE1, \TYPE2, 0

.size castToGfloat16_\TYPE1\()_to_\TYPE2\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\()
.endm

.macro CAST_TO_GFLOAT16_INPLACE_OP TYPE
.section .text.castToGfloat16InPlace_\TYPE\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\()
  .type __runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  CAST_TO_GFLOAT16 \TYPE, \TYPE, 1

.size castToGfloat16InPlace_\TYPE\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\()
.endm

CAST_TO_GFLOAT16_OP float, float
CAST_TO_GFLOAT16_OP float, half
CAST_TO_GFLOAT16_OP half , half

CAST_TO_GFLOAT16_INPLACE_OP float
CAST_TO_GFLOAT16_INPLACE_OP half

#endif
