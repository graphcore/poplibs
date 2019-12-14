// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__

#define PackedGfloatParams        __runCodelet_experimental__popfloat__PackedGfloatParams

#include "PackedGfloatParams.h"
#include "GfloatConst.hpp"

.globl PackedGfloatParams
.type PackedGfloatParams      , @function

.section .text.PackedGfloatParams
.align 8
PackedGfloatParams:
  ld32         $mGfPackedParam , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_STRUCT
  ld32         $mGfManSize     , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_MAN_BITS
  ld32         $mGfExpSize     , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_EXP_BITS
  shl          $mGfPackedStruct, $mGfManSize           , (8*POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET)
  ld32         $mGfExpSize     , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_EXP_BITS
  shl          $mGfExpSize     , $mGfExpSize           , (8*POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET)
  add          $mGfPackedStruct, $mGfExpSize           , $mGfPackedStruct
  ld32         $mExpBias       , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_EXP_BIAS
  and          $mExpBias       , $mExpBias             , (0xFF)
  shl          $mExpBias       , $mExpBias             , (8*POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET)
  add          $mGfPackedStruct, $mGfPackedStruct      , $mExpBias
  ld32         $mEnDnrm        , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_EN_DENORM
  shl          $mEnDnrm        , $mEnDnrm              , POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET
  ld32         $mEnInf         , $mvertex_base         , $mzero            , POPFLOAT_VERTEX_CALC_PACKED_GF_PARAM_EN_INF
  shl          $mEnInf         , $mEnInf               , POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET
  add          $mGfOpts        , $mEnDnrm              , $mEnInf
  shl          $mGfOpts        , $mGfOpts              , (8*POPFLOAT_GF_STRUCT_PARAMS_OFFSET)
  add          $mGfPackedStruct, $mGfPackedStruct      , $mGfOpts
  st32         $mGfPackedStruct, $mzero                , $mGfPackedParam   , 0
  exitz        $mzero
.size PackedGfloatParams        , .-PackedGfloatParams
#endif
