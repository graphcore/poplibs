#ifdef __IPU__

// popfloat::CastToGloat32

#define CastGF16ToFloatParam        __runCodelet_popfloat__CastGf16ToFloatParam

#include "GfloatConst.hpp"
#include "CastGF16ToFloatParam.h"

.globl CastGF16ToFloatParam
.type CastGF16ToFloatParam      , @function

.section .text.CastGF16ToFloatParam
.align 8

CastGF16ToFloatParam:
  ld32         $mGF16Param    , $mvertex_base         , $mzero            , POPFLOAT_VBASE_CALC_GFLOAT_PARAM_PTR_OFFSET
  ld32         $mFloatStruct  , $mvertex_base         , $mzero            , POPFLOAT_VBASE_GFLOAT_STRUCT_PTR_OFFSET;
  ldz8         $mGF32Man      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET
  ldz8         $mGF32Exp      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET
  setzi        $mConstOne     , 1
  ldz8         $enDenorm      , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_PARAMS_OFFSET
  and          $enDenorm      , $enDenorm             , POPFLOAT_GF_STRUCT_ENDENORM_MASK
  lds8         $mBiasCorr     , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET
  sub          $mGf32ExpAlign , 127                   , $mBiasCorr
  add          $mGf32ExpAlign , $mGf32ExpAlign        , $enDenorm
  shl          $mGf32ExpAlign , $mGf32ExpAlign        , POPFLOAT_NUM_FP32_MANTISSA_BITS
  st32         $mGf32ExpAlign , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_ALIGN_OFFSET)
  sub          $expMaskShl    , 31                    , $mGF32Exp
  shl          $mGf16ExpMask  , $mConstOne            , $mGF32Exp
  add          $mGf16ExpMask  , $mGf16ExpMask         , -1
  shl          $mGf16ExpMask  , $mGf16ExpMask         , $expMaskShl
  st32         $mGf16ExpMask  , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_MASK_OFFSET)
  st32         $mGf16ExpMask  , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_EXP_MASK_OFFSET+1)
  add          $mManSh0       , $mGF32Exp             , POPFLOAT_NUM_FP32_EXPONENT_BITS
  st32         $mManSh0       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_SHIFT0_OFFSET)
  sub          $mManSh1       , 8                     , $mGF32Exp
  st32         $mManSh1       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_SHIFT1_OFFSET)
  lds8         $mBiasCorr     , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET
  shl          $mGF32Exp      , $mConstOne            , $mGF32Exp
  sub          $mGF32Exp      , $mGF32Exp             , $mConstOne
  shl          $mBit23        , $mConstOne            , POPFLOAT_NUM_FP32_MANTISSA_BITS
  sub          $mGF32Man      , POPFLOAT_NUM_FP32_MANTISSA_BITS, $mGF32Man
  shl          $mGF32Man      , $mConstOne            , $mGF32Man
  sub          $mGF32Man      , $mBit23               , $mGF32Man
  ldz8         $maxNorm       , $mFloatStruct         , $mzero            , POPFLOAT_GF_STRUCT_PARAMS_OFFSET
  and          $maxNorm       , $maxNorm              , POPFLOAT_GF_STRUCT_ENINF_MASK
  shr          $maxNorm       , $maxNorm              , POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET
  sub          $maxNorm       , $mGF32Exp             , $maxNorm
  sub          $mBiasCorr     , POPFLOAT_FP32_EXPONENT_BIAS, $mBiasCorr
  add          $maxNorm       , $maxNorm              , $mBiasCorr
  add          $minNorm       , $mBiasCorr            , 1
  shl          $minNorm       , $minNorm              , POPFLOAT_NUM_FP32_MANTISSA_BITS
  st32         $minNorm       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_MIN_NORM_OFFSET)
  st32         $minNorm       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_MIN_NORM_OFFSET)+1
  shl          $maxNorm       , $maxNorm              , POPFLOAT_NUM_FP32_MANTISSA_BITS
  or           $maxNorm       , $maxNorm              , $mGF32Man
  st32         $maxNorm       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_CLAMP_OFFSET)+1
  or           $maxNorm       , $maxNorm              , POPFLOAT_FP32_SIGN_MASK
  st32         $maxNorm       , $mGF16Param           , $mzero            , (POPFLOAT_GF16_TO_FP32_PARAM_CLAMP_OFFSET)
  exitz        $mzero

.size CastGF16ToFloatParam    , .-CastGF16ToFloatParam

#endif
