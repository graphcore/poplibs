#ifdef __IPU__
#include "CastToGfloat16Sr.h"
#include "GfloatConst.hpp"
#include "arch/gc_tile_defines.h"

.macro CAST_TO_GFLOAT16_SR_TRUNCATED_NORMAL INPLACE
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
    ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET
.else
    ld32         $nIterations   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET
.endif
    and64        $maskOut       , $maskOut              , $azeros
  }
1:
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_CLAMP_PARAMS_OFFSET;
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
  brz         $maskOut_0        , 1f
  brnzdec     $nIterations      , 1b
1:
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_SCALE_PARAMS_OFFSET;
    or64         $roundCorrV4   , $trncNorm             , $azeros
  }
.endm

.macro CAST_TO_GFLOAT16_SR TYPE1 TYPE2 NANOO INPLACE DENSITY
  ld32         $mGf16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mBaseIn       , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mRowCount     , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_SIZE_PTR_OFFSET
  add          $mRowCount     , $mRowCount            , -1
  ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2)
.Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
  ld32step     $mInRow        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mCount        , $mzero                , $mBaseIn+=        , 1
  ld32step     $mOutRow       , $mzero                , $mBaseOut+=       , 2
  ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1
  brz          $mCount        , .Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()
.Lcast_to_gfloat16Sr_inner_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
.ifnc \TYPE1, float
.ifc \NANOO, false
  ld32         $inputClampF16 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET);
  f16v4clamp   $inValueV4     , $inValueV4            , $inputClampF16    // Clip values before scaling (CLAMP)
.endif
  {
    ld64         $halfExpMaskV4 , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET/2);
    f16v4mul     $outValueV4    , $scalePair:BL         , $inValueV4        // Scale values
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
    st64         $expV4         , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_EXPONENT_OFFSET/2);
    or64         $outBitMaskV4  , $isDenormV4           , $outBitMaskV4
  }
.ifc \DENSITY, BERNOULLI
  {
.ifc \INPLACE, true
    ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_DIST_PARAM_PTR_OFFSET
.else
    ld32         $probBrnoulli  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_DIST_PARAM_PTR_OFFSET
.endif
    not64        $manLsbMaskV4  , $outBitMaskV4
  }  
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4rmask   $roundCorrV4   , $manLsbMaskV4         , $probBrnoulli
  }
.else
.ifc \DENSITY, LOGIT___NORMAL
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v2grand   $roundCorrV4_0
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_SCALE_PARAMS_OFFSET
    f16v2grand   $roundCorrV4_1
  }
  f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
  f16v2sigm    $roundCorrV4_0 , $roundCorrV4_0
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_CLAMP_PARAMS_OFFSET
    f16v2sigm    $roundCorrV4_1 , $roundCorrV4_1
  }
  f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
.else
.ifc \DENSITY, TRUNCATED___LOGIT___NORMAL
  CAST_TO_GFLOAT16_SR_TRUNCATED_NORMAL INPLACE
  f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
  f16v2sigm    $roundCorrV4_0 , $roundCorrV4_0
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_CLAMP_PARAMS_OFFSET
    f16v2sigm    $roundCorrV4_1 , $roundCorrV4_1
  }
  {
    ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
.else
.ifc \DENSITY, UNIFORM
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    urand64      $roundCorrV4
  }
  {
    ld32          $scaleCorr    , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_SCALE_PARAMS_OFFSET
    f16v4sufromui $roundCorrV4  , $roundCorrV4
  }
.else
.ifc \DENSITY, NORMAL
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v2grand   $roundCorrV4_0
  }
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_CLAMP_PARAMS_OFFSET
    f16v2grand   $roundCorrV4_1
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_SCALE_PARAMS_OFFSET
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
.else
.ifc \DENSITY, TRUNCATED___NORMAL
  CAST_TO_GFLOAT16_SR_TRUNCATED_NORMAL INPLACE
  ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
.else
.ifc \DENSITY, LAPLACE
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    urand64      $roundCorrV4
  }
  {
    ld64          $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4sufromui $roundCorrV4  , $roundCorrV4
  }
  f16v4cmpgt    $scale          , $azeros               , $roundCorrV4      // Positive values
  and64         $signV4         , $scale                , $signV4           // Flip sign
  f16v4absadd  $roundCorrV4   , $roundCorrV4          , $roundCorrV4
  setzi        $scalePair     , (POPFLOAT_FP16_EXPONENT_BIAS << POPFLOAT_NUM_FP16_MANTISSA_BITS)
  f16v4sub     $roundCorrV4   , $scalePair:BL           , $roundCorrV4
  f16v2ln      $roundCorrV4_0 , $roundCorrV4_0
  f16v2ln      $roundCorrV4_1 , $roundCorrV4_1
  f16v4add     $scale         , $scalePair:BL         , $azeros
  or64         $scale         , $scale                , $signV4           // Flip sign
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_CLAMP_PARAMS_OFFSET
    f16v4mul     $roundCorrV4   , $scale                , $roundCorrV4
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_SCALE_PARAMS_OFFSET
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
.else
.ifc \DENSITY, LOGISTIC
  {
    st64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    urand64      $roundCorrV4
  }
  f16v4sufromui $roundCorrV4  , $roundCorrV4
  setzi        $scalePair     , ((POPFLOAT_FP16_EXPONENT_BIAS-1) << POPFLOAT_NUM_FP16_MANTISSA_BITS) // 1/2
  f16v4sub     $oneMinCorrV4  , $scalePair:BL         , $roundCorrV4        // y=f16v4sufromui(x): y~U[-1/2,1/2]=x~U[0,1]-1/2 => 1-x=1/2-y
  f16v4add     $roundCorrV4   , $scalePair:BL         , $roundCorrV4        // y=f16v4sufromui(x): y~U[-1/2,1/2]=x~U[0,1]-1/2 => x=1/2+y
  f16v2ln      $roundCorrV4_0 , $roundCorrV4_0
  f16v2ln      $roundCorrV4_1 , $roundCorrV4_1
  f16v2ln      $oneMinCorrV4_0, $oneMinCorrV4_0
  f16v2ln      $oneMinCorrV4_1, $oneMinCorrV4_1
  {
    ld32         $clampCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_CLAMP_PARAMS_OFFSET
    f16v4sub     $roundCorrV4   , $roundCorrV4          , $oneMinCorrV4
  }
  {
    ld32         $scaleCorr     , $mCorrParams          , $mzero            , POPFLOAT_CAST_TO_GF16_SR_CORR_SCALE_PARAMS_OFFSET
    f16v4clamp   $roundCorrV4   , $roundCorrV4          , $clampCorr
  }
.endif // .ifc \DENSITY, LOGISTIC
.endif // .ifc \DENSITY, LAPLACE
.endif // .ifc \DENSITY, TRUNCATED___NORMAL
.endif // .ifc \DENSITY, NORMAL
.endif // .ifc \DENSITY, UNIFORM
  {
    ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v4mul     $roundCorrV4   , $scaleCorr:BU         , $roundCorrV4
  }
  f16v4add     $roundCorrV4   , $scaleCorr:BL         , $roundCorrV4
.endif // .ifc \DENSITY, TRUNCATED___LOGIT___NORMAL
.endif // .ifc \DENSITY, LOGIT___NORMAL
  setzi        $halfMinDnrm   , 1
  not64        $outBitMaskV4  , $outBitMaskV4
  f16v4add     $manLsbMaskV4  , $halfMinDnrm:BL       , $outBitMaskV4      // Add 1 lsb to inverted bits to set mantissa LSB
  {
    ld64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
    f16v4mul     $roundCorrV4   , $roundCorrV4          , $manLsbMaskV4
  }
  f16v4add     $roundCorrV4   , $roundCorrV4          , $manLsbMaskV4
  and64        $roundCorrV4   , $roundCorrV4          , $srMaskV4
  {
    ld64         $signV4        , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET/2);
    f16v4sub     $roundCorrV4   , $roundCorrV4          , $manLsbMaskV4
  }
.endif // .ifc \DENSITY, BERNOULLI
  and64        $signV4        , $outValueV4           , $signV4           // Extract signs
  {
    ld64         $outBitMaskV4  , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_OUT_BITMASK_OFFSET/2)
    f16v4absadd  $outValueV4    , $outValueV4           , $roundCorrV4
  }
  {
    ld32         $scaledClamp   , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET);
    and64        $outValueV4    , $outValueV4           , $outBitMaskV4     // Truncate matissa
  }
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
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET/2);
    f16v4clamp   $outValueV4    , $outValueV4           , $scaledClamp
  }
  {
    ld64step     $inValueV4     , $mzero                , $mInRow+=         , 1;
    or64         $outValueV4    , $outValueV4           , $signV4
  }
  cmpult       $mRemainder    , $mCount               , 4
#if 0
.ifc \TYPE1, float
  {
    ld64         $scale         , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET/2);
    f16v2tof32   $inValueV2_1   , $inValueV4_1
  }
  f16v2tof32   $inValueV2_0   , $inValueV4_0
  f32v2mul     $inValueV2_0   , $scaleFloat:B         , $inValueV2_0
  f32v2mul     $inValueV2_1   , $scaleFloat:B         , $inValueV2_1
.else
  ld64        $scale          , $mGf16Param           , $mzero            , (POPFLOAT_CAST_TO_GF16_PARAM_SCALE_IN_RECIP_OFFSET/2)
  f16v4mul    $outValueV4     , $scalePair:BL         , $outValueV4       // Scale values
.endif
#endif
  brnz         $mRemainder    , .Lcast_to_gfloat16Sr_inner_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()
  add          $mCount        , $mCount               , -4
  st64step     $outValueV4    , $mzero                , $mOutRow+=        , 1
  brnz         $mCount        , .Lcast_to_gfloat16Sr_inner_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()
  bri          .Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()

.Lcast_to_gfloat16Sr_inner_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
  cmpult       $mRemainder    , $mCount               , 3
  brnz         $mRemainder    , .Lcast_to_gfloat16Sr_inner_last2_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()
  {
    st32step     $outValueV2_0  , $mzero                , $mOutRow+=        , 1
    or           $outValueV2_0  , $outValueV2_1         , $azero
  }
  add          $mCount        , $mCount               , -2

.Lcast_to_gfloat16Sr_inner_last2_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
  cmpult       $mRemainder    , $mCount               , 2
  brnz         $mRemainder    , .Lcast_to_gfloat16Sr_inner_last1_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()
  bri          .Lcast_to_gfloat16Sr_inner_store_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()

.Lcast_to_gfloat16Sr_inner_last1_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
  ldb16        $outValueV2_1  , $mzero                , $mOutRow          , 1
  sort4x16lo   $outValueV2_0  , $outValueV2_0         , $outValueV2_1

.Lcast_to_gfloat16Sr_inner_store_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
  st32step     $outValueV2_0  , $mzero                , $mOutRow+=        , 1

.Lcast_to_gfloat16Sr_outer_epilog_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\():
  brnzdec      $mRowCount     , .Lcast_to_gfloat16Sr_outer_start_\TYPE1\()_\TYPE2\()_\NANOO\()_\INPLACE\()_\DENSITY\()
  exitz        $mzero
.endm

.macro CAST_TO_GFLOAT16_SR_OP TYPE1, TYPE2, NANOO, DENSITY
.section .text.castToGfloat16Sr_\TYPE1\()_to_\TYPE2\()_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\()
.align 4
  .globl __runCodelet_popfloat__CastToGfloat16Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\()
  .type __runCodelet_popfloat__CastToGfloat16Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\(), @function
  __runCodelet_popfloat__CastToGfloat16Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_OUTPUT_BASE_PTR_OFFSET
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_SR_MASK_OFFSET
  ld64         $srMaskV4      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
  CAST_TO_GFLOAT16_SR \TYPE1, \TYPE2, \NANOO, false, \DENSITY

.size castToGfloat16Sr_\TYPE1\()_to_\TYPE2\()_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\(),\
  .-__runCodelet_popfloat__CastToGfloat16Sr___\TYPE1\()_\TYPE2\()_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\()
.endm

CAST_TO_GFLOAT16_SR_OP float, half , true , UNIFORM
CAST_TO_GFLOAT16_SR_OP float, half , false, UNIFORM
CAST_TO_GFLOAT16_SR_OP half , half , true , UNIFORM
CAST_TO_GFLOAT16_SR_OP half , half , false, UNIFORM

CAST_TO_GFLOAT16_SR_OP float, half , true , NORMAL
CAST_TO_GFLOAT16_SR_OP float, half , false, NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , true , NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , false, NORMAL

CAST_TO_GFLOAT16_SR_OP float, half , true , TRUNCATED___NORMAL
CAST_TO_GFLOAT16_SR_OP float, half , false, TRUNCATED___NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , true , TRUNCATED___NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , false, TRUNCATED___NORMAL

CAST_TO_GFLOAT16_SR_OP float, half , true , BERNOULLI
CAST_TO_GFLOAT16_SR_OP float, half , false, BERNOULLI
CAST_TO_GFLOAT16_SR_OP half , half , true , BERNOULLI
CAST_TO_GFLOAT16_SR_OP half , half , false, BERNOULLI

CAST_TO_GFLOAT16_SR_OP float, half , true , LAPLACE
CAST_TO_GFLOAT16_SR_OP float, half , false, LAPLACE
CAST_TO_GFLOAT16_SR_OP half , half , true , LAPLACE
CAST_TO_GFLOAT16_SR_OP half , half , false, LAPLACE

CAST_TO_GFLOAT16_SR_OP float, half , true , LOGISTIC
CAST_TO_GFLOAT16_SR_OP float, half , false, LOGISTIC
CAST_TO_GFLOAT16_SR_OP half , half , true , LOGISTIC
CAST_TO_GFLOAT16_SR_OP half , half , false, LOGISTIC

CAST_TO_GFLOAT16_SR_OP float, half , true , LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_OP float, half , false, LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , true , LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , false, LOGIT___NORMAL

CAST_TO_GFLOAT16_SR_OP float, half , true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_OP float, half , false, TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_OP half , half , false, TRUNCATED___LOGIT___NORMAL

.macro CAST_TO_GFLOAT16_SR_INPLACE_OP NANOO, DENSITY
.section .text.castToGfloat16SrInPlace_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\()
.align 4
  .globl __runCodelet_popfloat__CastToGfloat16SrInPlace___\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\()
  .type __runCodelet_popfloat__CastToGfloat16SrInPlace___\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\(), @function
  __runCodelet_popfloat__CastToGfloat16SrInPlace___\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\():

.align 8
  ld32         $mBaseOut      , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPUT_BASE_PTR_OFFSET
  ld32         $mCorrParams   , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_CORR_PARAMS_PTR_OFFSET
  ld32         $srMaskBase    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CAST_INPLACE_SR_MASK_OFFSET
  ld64         $srMaskV4      , $mzero                , $srMaskBase       , 0
  st64         $srMaskV4      , $mworker_base         , $mzero            , (POPFLOAT_CAST_TO_GF16_STACK_SR_MASK_OFFSET/2)
  CAST_TO_GFLOAT16_SR half, half, \NANOO, true, \DENSITY

.size castToGfloat16SrInPlace_\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\(),\
  .-__runCodelet_popfloat__CastToGfloat16SrInPlace___\NANOO\()_popfloat__gfexpr__GfloatSRDensityType__\DENSITY\()
.endm

CAST_TO_GFLOAT16_SR_INPLACE_OP true , UNIFORM
CAST_TO_GFLOAT16_SR_INPLACE_OP false, UNIFORM

CAST_TO_GFLOAT16_SR_INPLACE_OP true , NORMAL
CAST_TO_GFLOAT16_SR_INPLACE_OP false, NORMAL

CAST_TO_GFLOAT16_SR_INPLACE_OP true , TRUNCATED___NORMAL
CAST_TO_GFLOAT16_SR_INPLACE_OP false, TRUNCATED___NORMAL

CAST_TO_GFLOAT16_SR_INPLACE_OP true , BERNOULLI
CAST_TO_GFLOAT16_SR_INPLACE_OP false, BERNOULLI

CAST_TO_GFLOAT16_SR_INPLACE_OP true , LAPLACE
CAST_TO_GFLOAT16_SR_INPLACE_OP false, LAPLACE

CAST_TO_GFLOAT16_SR_INPLACE_OP true , LOGISTIC
CAST_TO_GFLOAT16_SR_INPLACE_OP false, LOGISTIC

CAST_TO_GFLOAT16_SR_INPLACE_OP true , LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_INPLACE_OP false, LOGIT___NORMAL

CAST_TO_GFLOAT16_SR_INPLACE_OP true , TRUNCATED___LOGIT___NORMAL
CAST_TO_GFLOAT16_SR_INPLACE_OP false, TRUNCATED___LOGIT___NORMAL

#endif
