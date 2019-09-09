#ifdef __IPU__
#include "CastToGfloat16.h"
#include "GfloatConst.hpp"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT16 RMODE TYPE1 TYPE2 NANOO INPLACE
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
.Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2)
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1
  brz          $mCount        , .Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
.Lcast_to_gfloat16_inner_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
.ifnc \TYPE1, float
.ifc \NANOO, false
  ld32         $inputClampF16 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET);
  f16v4clamp   $inValueV4     , $inValueV4            , $inputClampF16    // Clip values before scaling (CLAMP)
.endif
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $inValueV4        // Scale values
  }
.else
.ifc \NANOO, false
  ld64         $inputClampF32 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET/2);
  {
    ld64step     $inValueV2_1   , $mzero                , $mInRow+=         , 1;
    f32v2clamp   $inValueV2_0   , $inValueV2_0          , $inputClampF32    // Clip values before scaling (CLAMP)
  }
  f32v2clamp   $inValueV2_1   , $inValueV2_1          , $inputClampF32    // Clip values before scaling (CLAMP)
  f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0      // Scale values
.else
  {
    ld64step     $inValueV2_1   , $mzero                , $mInRow+=         , 1;
    f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0      // Scale values and generate Nan if value is outside the range
  }
.endif
  f32v2mul     $inValueV2_1   , $scaleFloat:B         , $inValueV2_1      // Scale values and generate Nan if value is outside the range
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f32v4tof16   $outValueV4    , $inValueF32V4                             // Copy f32v4 vector to f16.
  }
.endif
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
  or64         $outBitMaskV4    , $isDenormV4           , $outBitMaskV4
.ifnc \RMODE, RZ
.ifc \RMODE, RA
  {
    st64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    setzi        $halfMinDnrm   , 1
  }
  {
    st64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUTPUT_OFFSET/2);
    not64        $roundCorrV4   , $outBitMaskV4
  }
  {
    ld64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    f16v4add     $roundCorrV4   , $halfMinDnrm:BL       , $roundCorrV4      // Add 1 lsb to inverted bits to set mantissa LSB
  }
  {
    ld32         $scalePm1      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M1_OFFSET);
    and64        $roundCorrV4   , $roundCorrV4          , $outBitMaskV4
  }
  {
    ld64         $outValueV4    , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUTPUT_OFFSET/2);
    f16v4mul     $roundCorrV4   , $scalePm1:BL          , $roundCorrV4
  }
  or64         $roundCorrV4   , $expV4                , $roundCorrV4      // Add exponent to truncated bits
.else
.ifc \RMODE, RN
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
  or64         $roundCorrV4   , $expV4                , $roundCorrV4        // Add exponent to truncated bits
.else
.ifc \RMODE, RU
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
  or64         $roundCorrV4   , $expV4                , $roundCorrV4      // Add exponent to truncated bits
.else
.ifc \RMODE, RD
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
  or64         $roundCorrV4   , $expV4                , $roundCorrV4      // Add exponent to truncated bits
.else
.ifc \RMODE, SX
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
  or64         $roundCorrV4   , $expV4                , $roundCorrV4      // Add exponent to truncated bits
.else
.ifc \RMODE, SR
  urand64      $randomBitsV4                                              // Generate random bit pattern
  andc64       $roundCorrV4   , $randomBitsV4         , $outBitMaskV4     // Apply SR bit mask
  or64         $roundCorrV4   , $expV4                , $roundCorrV4      // Add exponent to truncated bits
.else
.error "Rounding mode not supported"
.endif // SR
.endif // SX
.endif // RD
.endif // RU
.endif // RN
.endif // RA
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4sub     $roundCorrV4   , $roundCorrV4          , $expV4            // Subtract exponent from correct
  }
  and64        $signV4        , $outValueV4           , $signV4           // Extract signs
  {
    ld32         $scaledMin     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET);
    f16v4absadd  $outValueV4    , $outValueV4           , $roundCorrV4      // Add correction
  }
.else
  ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
  {
    ld32         $scaledMin     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET);
    and64        $signV4        , $outValueV4           , $signV4           // Extract signs
  }
  f16v4absadd  $outValueV4    , $outValueV4           , $azeros
.endif // RZ
  and64        $outValueV4    , $outValueV4           , $outBitMaskV4     // Truncate matissa
  {
    ld32         $scaledClamp   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET);
    f16v4cmple   $zeroOutMaskV4 , $scaledMin:BU         , $outValueV4
  }
  and64          $outValueV4    , $outValueV4           , $zeroOutMaskV4
.ifc \NANOO, true
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
.endif
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
  brnz         $mRemainder    , .Lcast_to_gfloat16_inner_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
  add          $mCount        , $mCount               , -4
  st64step     $outValueV4    , $mzero                , $mOutRow+=        , 1
.else
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET/2);    
    f16v2tof32   $outValueV2_0  , $outValueV4_0
  }
  {
    brnz         $mRemainder    , .Lcast_to_gfloat16_inner_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
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
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
.endif
  brnz         $mCount        , .Lcast_to_gfloat16_inner_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
  bri          .Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()

.Lcast_to_gfloat16_inner_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
.ifc \TYPE2, half
  cmpult       $mRemainder    , $mCount               , 3
.else
  {
    cmpult       $mRemainder    , $mCount               , 3
    f32v2mul     $outValueV2_0  , $scaleFloat:B         , $outValueV2_0     // Scale values
  }
.endif
  brnz         $mRemainder    , .Lcast_to_gfloat16_inner_last2_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
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

.Lcast_to_gfloat16_inner_last2_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
  cmpult       $mRemainder    , $mCount               , 2
  brnz         $mRemainder    , .Lcast_to_gfloat16_inner_last1_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
  bri          .Lcast_to_gfloat16_inner_store_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()

.Lcast_to_gfloat16_inner_last1_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
.ifc \TYPE2, half
  ldb16        $outValueV4_1  , $mzero                , $mOutRow          , 1
  sort4x16lo   $outValueV4_0  , $outValueV4_0         , $outValueV4_1
.endif

.Lcast_to_gfloat16_inner_store_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
  st32step     $outValueV4_0  , $mzero                , $mOutRow+=        , 1

.Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\():
  brnzdec      $mRowCount     , .Lcast_to_gfloat16_outer_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\RMODE\()_\INPLACE\()
  exitz        $mzero
.endm

.macro CAST_TO_GFLOAT16_OP TYPE1, TYPE2, RMODE, NANOO
.section .text.castToGfloat16_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
  .type __runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET


.ifc \RMODE, SX
  ld32         $srManMask     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
  ld64         $srMaskV4      , $mzero                , $srManMask        , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
.else
.ifc \RMODE, RU
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  {
    ld32         $scaledMin     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET)
    setzi        $scaleHalf     , 0x3800
  }
  f16v4add     $halfMinDnrmV4 , $scaledMin:BL         , $azeros
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
.else
.ifc \RMODE, RD
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  {
    ld32         $scaledMin     , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET)
    setzi        $scaleHalf     , 0x3800
  }
  f16v4add     $halfMinDnrmV4 , $scaledMin:BL         , $azeros
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
.endif
.endif
.endif
  CAST_TO_GFLOAT16 \RMODE, \TYPE1, \TYPE2, \NANOO, false

.size castToGfloat16_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat16___\TYPE1\()_\TYPE2\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
.endm

.macro CAST_TO_GFLOAT16_INPLACE_OP TYPE RMODE NANOO
.section .text.castToGfloat16InPlace_\TYPE\()_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
  .type __runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\(), @function
  __runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET

.ifc \RMODE, SX
  ld32         $srManMask     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
  ld64         $srMaskV4      , $mzero                , $srManMask        , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
.else
.ifc \RMODE, RU
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  {
    ld64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);  
    setzi        $scaleHalf     , 0x3800
  }
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
.else
.ifc \RMODE, RD
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  {
    ld64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);  
    setzi        $scaleHalf     , 0x3800
  }
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
.endif
.endif
.endif
  CAST_TO_GFLOAT16 \RMODE, \TYPE, \TYPE, \NANOO, true

.size castToGfloat16InPlace_\TYPE\()_\NANOO\()_\RMODE\(),\
  .-__runCodelet_experimental__popfloat__CastToGfloat16InPlace___\TYPE\()_\NANOO\()_experimental__popfloat__RoundType__\RMODE\()
.endm

CAST_TO_GFLOAT16_OP float, float, RZ, true 
CAST_TO_GFLOAT16_OP float, float, RZ, false
CAST_TO_GFLOAT16_OP float, float, RN, true 
CAST_TO_GFLOAT16_OP float, float, RN, false
CAST_TO_GFLOAT16_OP float, float, RA, true 
CAST_TO_GFLOAT16_OP float, float, RA, false
CAST_TO_GFLOAT16_OP float, float, RU, true 
CAST_TO_GFLOAT16_OP float, float, RU, false
CAST_TO_GFLOAT16_OP float, float, RD, true 
CAST_TO_GFLOAT16_OP float, float, RD, false
CAST_TO_GFLOAT16_OP float, float, SR, true 
CAST_TO_GFLOAT16_OP float, float, SR, false
CAST_TO_GFLOAT16_OP float, float, SX, true 
CAST_TO_GFLOAT16_OP float, float, SX, false

CAST_TO_GFLOAT16_OP float, half, RZ, true 
CAST_TO_GFLOAT16_OP float, half, RZ, false
CAST_TO_GFLOAT16_OP float, half, RN, true 
CAST_TO_GFLOAT16_OP float, half, RN, false
CAST_TO_GFLOAT16_OP float, half, RA, true 
CAST_TO_GFLOAT16_OP float, half, RA, false
CAST_TO_GFLOAT16_OP float, half, RU, true 
CAST_TO_GFLOAT16_OP float, half, RU, false
CAST_TO_GFLOAT16_OP float, half, RD, true 
CAST_TO_GFLOAT16_OP float, half, RD, false
CAST_TO_GFLOAT16_OP float, half, SR, true 
CAST_TO_GFLOAT16_OP float, half, SR, false
CAST_TO_GFLOAT16_OP float, half, SX, true 
CAST_TO_GFLOAT16_OP float, half, SX, false

CAST_TO_GFLOAT16_OP half , half, RZ, true 
CAST_TO_GFLOAT16_OP half , half, RZ, false
CAST_TO_GFLOAT16_OP half , half, RN, true 
CAST_TO_GFLOAT16_OP half , half, RN, false
CAST_TO_GFLOAT16_OP half , half, RA, true 
CAST_TO_GFLOAT16_OP half , half, RA, false
CAST_TO_GFLOAT16_OP half , half, RU, true 
CAST_TO_GFLOAT16_OP half , half, RU, false
CAST_TO_GFLOAT16_OP half , half, RD, true 
CAST_TO_GFLOAT16_OP half , half, RD, false
CAST_TO_GFLOAT16_OP half , half, SR, true 
CAST_TO_GFLOAT16_OP half , half, SR, false
CAST_TO_GFLOAT16_OP half , half, SX, true 
CAST_TO_GFLOAT16_OP half , half, SX, false

CAST_TO_GFLOAT16_INPLACE_OP float, RZ, true 
CAST_TO_GFLOAT16_INPLACE_OP float, RZ, false
CAST_TO_GFLOAT16_INPLACE_OP float, RN, true 
CAST_TO_GFLOAT16_INPLACE_OP float, RN, false
CAST_TO_GFLOAT16_INPLACE_OP float, RA, true 
CAST_TO_GFLOAT16_INPLACE_OP float, RA, false
CAST_TO_GFLOAT16_INPLACE_OP float, RU, true 
CAST_TO_GFLOAT16_INPLACE_OP float, RU, false
CAST_TO_GFLOAT16_INPLACE_OP float, RD, true 
CAST_TO_GFLOAT16_INPLACE_OP float, RD, false
CAST_TO_GFLOAT16_INPLACE_OP float, SR, true 
CAST_TO_GFLOAT16_INPLACE_OP float, SR, false
CAST_TO_GFLOAT16_INPLACE_OP float, SX, true 
CAST_TO_GFLOAT16_INPLACE_OP float, SX, false

CAST_TO_GFLOAT16_INPLACE_OP half, RZ, true 
CAST_TO_GFLOAT16_INPLACE_OP half, RZ, false
CAST_TO_GFLOAT16_INPLACE_OP half, RN, true 
CAST_TO_GFLOAT16_INPLACE_OP half, RN, false
CAST_TO_GFLOAT16_INPLACE_OP half, RA, true 
CAST_TO_GFLOAT16_INPLACE_OP half, RA, false
CAST_TO_GFLOAT16_INPLACE_OP half, RU, true 
CAST_TO_GFLOAT16_INPLACE_OP half, RU, false
CAST_TO_GFLOAT16_INPLACE_OP half, RD, true 
CAST_TO_GFLOAT16_INPLACE_OP half, RD, false
CAST_TO_GFLOAT16_INPLACE_OP half, SR, true 
CAST_TO_GFLOAT16_INPLACE_OP half, SR, false
CAST_TO_GFLOAT16_INPLACE_OP half, SX, true 
CAST_TO_GFLOAT16_INPLACE_OP half, SX, false

#endif
