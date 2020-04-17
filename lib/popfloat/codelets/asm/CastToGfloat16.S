// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include "CastToGfloat16.h"
#include "GfloatConst.hpp"
#include "arch/gc_tile_defines.h"
#include "poplar/StackSizeDefs.hpp"
#include "popfloatCommon.inc"

.macro CAST_TO_GFLOAT16 TYPE1 TYPE2 NANOO RMODE INPLACE
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mGf16Param, $mvertex_base, POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseIn, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_GET_WORKER_INDEX $mWorkerIdx
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mGf16Param
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mCastParams
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseIn
  POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR $mBaseOut
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
  brz          $mCount        , .Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\()
  add          $mCount        , $mCount               , -1
  ld64step     $azeros        , $mzero                , $mBaseIn+=        , $mWorkerIdx
  ld64step     $azeros        , $mzero                , $mBaseOut+=       , $mWorkerIdx
.ifnc \TYPE1, half
  ld64step     $azeros        , $mzero                , $mBaseIn+=        , $mWorkerIdx
.endif
.ifnc \TYPE2, half
  ld64step     $azeros        , $mzero                , $mBaseOut+=       , $mWorkerIdx
.endif
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2)
.ifc \TYPE1, half
  ld64step     $inValueV4     , $mzero                , $mBaseIn+=        , CTXT_WORKERS
.else
  ld64step     $inValueV4     , $mzero                , $mBaseIn+=        , 1
.endif
.ifc \RMODE, RU
  {
    ld64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);
    setzi        $scaleHalf     , 0x3800
  }
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
.endif
.ifc \RMODE, RD
  {
    ld64         $halfMinDnrmV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET/2);
    setzi        $scaleHalf     , 0x3800
  }
  f16v4mul     $halfMinDnrmV4 , $scaleHalf:BL         , $halfMinDnrmV4
  st64         $halfMinDnrmV4 , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_HALF_MIN_OFFSET/2)
.endif
  bri          1f
.Lcast_to_gfloat16_inner_start_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\():
.ifc \TYPE2, half
  st64step     $outValueV4    , $mzero                , $mBaseOut+=       , CTXT_WORKERS
.else
  st64step     $outValueV2_0  , $mzero                , $mBaseOut+=       , 1
  st64step     $outValueV2_1  , $mzero                , $mBaseOut+=       , (2*CTXT_WORKERS-1)
.endif
1:
.ifc \TYPE1, half
.ifnc \NANOO, true
  ld32         $inputClampF16 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET);
  f16v4clamp   $inValueV4     , $inValueV4            , $inputClampF16    // Clip values before scaling (CLAMP)
.endif
1:
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $inValueV4        // Scale values
  }
.else
  ld64step     $inValueV2_1   , $mzero                , $mBaseIn+=        , (2*CTXT_WORKERS-1);
.ifnc \NANOO, true
  ld64         $inputClampF32 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET/2);
  f32v2clamp   $inValueV2_0   , $inValueV2_0          , $inputClampF32    // Clip values before scaling (CLAMP)
  f32v2clamp   $inValueV2_1   , $inValueV2_1          , $inputClampF32    // Clip values before scaling (CLAMP)
.endif
1:
  f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0      // Scale values
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
  or64         $outBitMaskV4  , $isDenormV4           , $outBitMaskV4
.ifnc \RMODE, RZ
.ifc \RMODE, RA
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
.else
.ifc \RMODE, SX
  {
    ld32         $scaleP10      , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET);
    not64        $roundCorrV4   , $outBitMaskV4                             // Truncated bits
  }
  {
    ld64         $srMaskV4      , $mCastParams          , $mzero            , (POPFLOAT_CAST_PARAMS_SR_MASK_OFFSET/2)
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
.else
.ifc \RMODE, SR
  urand64      $randomBitsV4                                              // Generate random bit pattern
  andc64       $roundCorrV4   , $randomBitsV4         , $outBitMaskV4     // Apply SR bit mask
.endif // .ifc \RMODE, SR
.endif // .ifc \RMODE, SX
.endif // .ifc \RMODE, RD
.endif // .ifc \RMODE, RU
.endif // .ifc \RMODE, RN
.endif // .ifc \RMODE, RA
.endif // .ifc \RMODE, RZ
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
  and64        $outValueV4    , $outValueV4           , $zeroOutMaskV4
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
.ifc \TYPE1, half
    ld64step     $inValueV4     , $mzero                , $mBaseIn+=        , CTXT_WORKERS;
.else
    ld64step     $inValueV4     , $mzero                , $mBaseIn+=        , 1;
.endif
    or64         $outValueV4    , $outValueV4           , $signV4
  }
  f16v4cmpeq   $signV4        , $outValueV4           , $azeros           // Mask for +/-0.0
  andc64       $outValueV4    , $outValueV4           , $signV4           // Convert all -0.0 into +0.0
.ifc \TYPE2, half
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
    f16v4mul     $outValueV4    , $scaleHalf:BL         , $outValueV4       // Scale values
  }
.else
  f16v2tof32   $outValueV2_0  , $outValueV4_0
  f16v2tof32   $outValueV2_1  , $outValueV4_1
  f32v2mul     $outValueV2_0  , $scaleFloat:B         , $outValueV2_0     // Scale values
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
    f32v2mul     $outValueV2_1  , $scaleFloat:B         , $outValueV2_1     // Scale values
  }
.endif
  brnzdec      $mCount        , .Lcast_to_gfloat16_inner_start_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\()
  brnz         $mRemainder    , 1f
.ifc \TYPE2, half
  st64step     $outValueV4    , $mzero                , $mBaseOut+=       , CTXT_WORKERS
.else
  st64step     $outValueV2_0  , $mzero                , $mBaseOut+=       , 1
  st64step     $outValueV2_1  , $mzero                , $mBaseOut+=       , 0
.endif
  exitz        $mzero
1:
  cmpult       $mCount        , $mRemainder           , 3
  brnz         $mCount        , .Lcast_to_gfloat16_inner_last2_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\()
.ifc \TYPE2, half
  {
    st32step     $outValueV4_0  , $mzero                , $mBaseOut+=       , 1
    or           $outValueV4_0  , $outValueV4_1         , $azero
  }
.else
  {
    st64step     $outValueV2_0  , $mzero                , $mBaseOut+=       , 1
    or64         $outValueV2_0  , $outValueV2_1         , $azeros
  }
.endif
  add          $mRemainder    , $mRemainder           , -2
.Lcast_to_gfloat16_inner_last2_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\():
  cmpult       $mCount        , $mRemainder           , 2
  brnz         $mCount        , .Lcast_to_gfloat16_inner_last1_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\()
.ifc \TYPE2, half
  st32step     $outValueV4_0  , $mzero                , $mBaseOut+=       , CTXT_WORKERS
.else
  st64step     $outValueV2_0  , $mzero                , $mBaseOut+=       , CTXT_WORKERS
.endif
  exitz        $mzero
.Lcast_to_gfloat16_inner_last1_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\():
.ifc \TYPE2, half
  ldb16        $outValueV4_1  , $mzero                , $mBaseOut         , 1
  sort4x16lo   $outValueV4_0  , $outValueV4_0         , $outValueV4_1
.endif
  st32step     $outValueV4_0  , $mzero                , $mBaseOut+=       , CTXT_WORKERS
.Lcast_to_gfloat16_outer_epilog_\TYPE1\()_\TYPE2\()_\INPLACE\()_\NANOO\()_\RMODE\():
  exitz        $mzero
.endm

.macro CAST_TO_GFLOAT16_OP TYPE1 TYPE2 NANOO RMODE
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastToGfloat16Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.section .text.castToGfloat16Supervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat16Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
  .type __runCodelet_popfloat__experimental__CastToGfloat16Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat16Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\():

.supervisor
castToGfloat16Supervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\():
  POPFLOAT_SUPERVISOR_CAST_OP castToGfloat16_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\()

.worker
castToGfloat16_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\():
.align 8
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mCastParams, $mvertex_base, POPFLOAT_VBASE_CAST_ROUNDING_PARAM_OFFSET
  CAST_TO_GFLOAT16 \TYPE1, \TYPE2, \NANOO, \RMODE, 0

.size castToGfloat16Supervisor_\TYPE1\()_to_\TYPE2\()_\NANOO\()_\RMODE\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat16Supervisor___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.endm

.macro CAST_TO_GFLOAT16_INPLACE_OP TYPE NANOO RMODE
DEF_STACK_USAGE 0 __runCodelet_popfloat__experimental__CastToGfloat16InPlaceSupervisor___\TYPE\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.section .text.castToGfloat16InPlaceSupervisor_\TYPE\()_\NANOO\()_\RMODE\()
.align 4
  .globl __runCodelet_popfloat__experimental__CastToGfloat16InPlaceSupervisor___\TYPE\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
  .type __runCodelet_popfloat__experimental__CastToGfloat16InPlaceSupervisor___\TYPE\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\(), @function
  __runCodelet_popfloat__experimental__CastToGfloat16InPlaceSupervisor___\TYPE\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\():

.supervisor
castToGfloat16InPlaceSupervisor_\TYPE\()_\NANOO\()_\RMODE\():
  POPFLOAT_SUPERVISOR_CAST_OP castToGfloat16InPlace_\TYPE\()_\NANOO\()_\RMODE\()
.worker
castToGfloat16InPlace_\TYPE\()_\NANOO\()_\RMODE\():
.align 8
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mBaseOut, $mvertex_base, POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  POPFLOAT_MAYBE_LOAD_SCALED_PTR $mCastParams, $mvertex_base, POPFLOAT_VBASE_CAST_INPLACE_ROUNDING_PARAM_OFFSET
  CAST_TO_GFLOAT16 \TYPE, \TYPE, \NANOO, \RMODE, 1

.size castToGfloat16InPlaceSupervisor_\TYPE\()_\NANOO\()_\RMODE\(),\
  .-__runCodelet_popfloat__experimental__CastToGfloat16InPlaceSupervisor___\TYPE\()_\NANOO\()_popfloat__experimental__RoundType__\RMODE\()
.endm

CAST_TO_GFLOAT16_OP float, float, true, RZ
CAST_TO_GFLOAT16_OP float, float, true, RA
CAST_TO_GFLOAT16_OP float, float, true, RN
CAST_TO_GFLOAT16_OP float, float, true, RU
CAST_TO_GFLOAT16_OP float, float, true, RD
CAST_TO_GFLOAT16_OP float, float, true, SR
CAST_TO_GFLOAT16_OP float, float, true, SX

CAST_TO_GFLOAT16_OP float, half, true, RZ
CAST_TO_GFLOAT16_OP float, half, true, RA
CAST_TO_GFLOAT16_OP float, half, true, RN
CAST_TO_GFLOAT16_OP float, half, true, RU
CAST_TO_GFLOAT16_OP float, half, true, RD
CAST_TO_GFLOAT16_OP float, half, true, SR
CAST_TO_GFLOAT16_OP float, half, true, SX

CAST_TO_GFLOAT16_OP half , half, true, RZ
CAST_TO_GFLOAT16_OP half , half, true, RA
CAST_TO_GFLOAT16_OP half , half, true, RN
CAST_TO_GFLOAT16_OP half , half, true, RU
CAST_TO_GFLOAT16_OP half , half, true, RD
CAST_TO_GFLOAT16_OP half , half, true, SR
CAST_TO_GFLOAT16_OP half , half, true, SX

CAST_TO_GFLOAT16_OP float, float, false, RZ
CAST_TO_GFLOAT16_OP float, float, false, RA
CAST_TO_GFLOAT16_OP float, float, false, RN
CAST_TO_GFLOAT16_OP float, float, false, RU
CAST_TO_GFLOAT16_OP float, float, false, RD
CAST_TO_GFLOAT16_OP float, float, false, SR
CAST_TO_GFLOAT16_OP float, float, false, SX

CAST_TO_GFLOAT16_OP float, half, false, RZ
CAST_TO_GFLOAT16_OP float, half, false, RA
CAST_TO_GFLOAT16_OP float, half, false, RN
CAST_TO_GFLOAT16_OP float, half, false, RU
CAST_TO_GFLOAT16_OP float, half, false, RD
CAST_TO_GFLOAT16_OP float, half, false, SR
CAST_TO_GFLOAT16_OP float, half, false, SX

CAST_TO_GFLOAT16_OP half , half, false, RZ
CAST_TO_GFLOAT16_OP half , half, false, RA
CAST_TO_GFLOAT16_OP half , half, false, RN
CAST_TO_GFLOAT16_OP half , half, false, RU
CAST_TO_GFLOAT16_OP half , half, false, RD
CAST_TO_GFLOAT16_OP half , half, false, SR
CAST_TO_GFLOAT16_OP half , half, false, SX

CAST_TO_GFLOAT16_INPLACE_OP float, true, RZ
CAST_TO_GFLOAT16_INPLACE_OP float, true, RA
CAST_TO_GFLOAT16_INPLACE_OP float, true, RN
CAST_TO_GFLOAT16_INPLACE_OP float, true, RU
CAST_TO_GFLOAT16_INPLACE_OP float, true, RD
CAST_TO_GFLOAT16_INPLACE_OP float, true, SR
CAST_TO_GFLOAT16_INPLACE_OP float, true, SX

CAST_TO_GFLOAT16_INPLACE_OP half, true, RZ
CAST_TO_GFLOAT16_INPLACE_OP half, true, RA
CAST_TO_GFLOAT16_INPLACE_OP half, true, RN
CAST_TO_GFLOAT16_INPLACE_OP half, true, RU
CAST_TO_GFLOAT16_INPLACE_OP half, true, RD
CAST_TO_GFLOAT16_INPLACE_OP half, true, SR
CAST_TO_GFLOAT16_INPLACE_OP half, true, SX

CAST_TO_GFLOAT16_INPLACE_OP float, false, RZ
CAST_TO_GFLOAT16_INPLACE_OP float, false, RA
CAST_TO_GFLOAT16_INPLACE_OP float, false, RN
CAST_TO_GFLOAT16_INPLACE_OP float, false, RU
CAST_TO_GFLOAT16_INPLACE_OP float, false, RD
CAST_TO_GFLOAT16_INPLACE_OP float, false, SR
CAST_TO_GFLOAT16_INPLACE_OP float, false, SX

CAST_TO_GFLOAT16_INPLACE_OP half, false, RZ
CAST_TO_GFLOAT16_INPLACE_OP half, false, RA
CAST_TO_GFLOAT16_INPLACE_OP half, false, RN
CAST_TO_GFLOAT16_INPLACE_OP half, false, RU
CAST_TO_GFLOAT16_INPLACE_OP half, false, RD
CAST_TO_GFLOAT16_INPLACE_OP half, false, SR
CAST_TO_GFLOAT16_INPLACE_OP half, false, SX

#endif
