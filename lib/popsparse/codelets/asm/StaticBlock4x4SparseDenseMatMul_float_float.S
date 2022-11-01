// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 4x4
//

#ifdef __IPU__
#include "StaticBlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseMatMulBlock___float_float_4_4
#define WORKER_NAME_POSTFIX ff4x4
#define BLOCK_LENGTH 4

// =============================================================================

// worker registers
#define w_qBaseLoadRetained                m3
#define w_sBase                            m2
#define w_qBase                            m9

#define w_tripack                          m10:11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11
#define w_qOffset                          m10

#define w_xin                              a0:1
#define w_xin_l                            a0
#define w_xin_h                            a1

#define w_pin0_1                           a0:3
#define w_pin0                             a0:1
#define w_pin1                             a2:3
#define w_pin                              a4:5
#define w_pout                             a6:7
#define fp_clr_reg                         a7


WORKER_FN_HEADER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
LOAD_VERTEX_STATE

add                   $w_num_orig, $w_num_orig, -1
mul                   $w_off, $w_off, BLOCK_LENGTH

ld32step              $azero, $mzero, $w_sBaseRetained+=, $w_off
ld32step              $azero, $mzero, $w_qBaseRetained+=, $w_off

WKR_FUNCTION_LABEL_XRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// offset by X and Y positions for the block processed
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
// w_xOffsetInQ is in multiples of 8 bytes
mul                   $w_xOffsetInQ, $w_xOffsetInQ, 8 * BLOCK_LENGTH
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
{
  add                   $w_qBaseLoadRetained, $w_qBase, 16
  fnop
}
// Fall through to code that uses partials as zeros
{
  ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  ld64step              $w_xin, $w_sBaseRetained, $w_yOffsetInS+=, 1
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  brneg                 $w_num_orig, LEndWorker
  f32sisoamp            $azeros, $w_xin_l, $azeros, TAMP_F32_E2_P0
}
{
  ld64step              $w_xin, $w_sBaseRetained, $w_yOffsetInS+=, 1
  f32sisoamp            $azeros, $w_xin_h, $azeros, TAMP_F32_E2_P1
}
{
  mov                   $w_qOffset, $mzero
  f32sisoamp            $azeros, $w_xin_l, $azeros, TAMP_F32_E2_P2
}  
{
  ld64step              $w_xin, $w_sBaseRetained, $w_yOffsetInS+=, 1
  f32sisoamp            $azeros, $w_xin_h, $azeros, TAMP_F32_E2_P3
}
// result available here
{
  rpt $w_num_orig, 3
  f32sisoamp            $w_pout, $w_xin_l, $azeros, TAMP_F32_E2_P0
}
  {
    ld64step              $w_xin, $w_sBaseRetained, $w_yOffsetInS+=, 1
    f32sisoamp            $azeros, $w_xin_h, $azeros, TAMP_F32_E2_P1
  }  
  {
    st64step              $w_pout, $w_qBase, $w_qOffset+=, 1
    f32sisoamp            $w_pout, $w_xin_l, $azeros, TAMP_F32_E2_P2
  } 
  {
    ld64step              $w_xin, $w_sBaseRetained, $w_yOffsetInS+=, 1
    f32sisoamp            $azeros, $w_xin_h, $azeros, TAMP_F32_E2_P3
  }  
  {
    st64step              $w_pout, $w_qBase, $w_qOffset+=, 1
    f32sisoamp            $w_pout, $w_xin_l, $azeros, TAMP_F32_E2_P0
  }  
{
  st64step              $w_pout, $w_qBase, $w_qOffset+=, 1
  f32sisoamp            $w_pout, $azero, $azeros, TAMP_F32_E1_P2
}
st64step              $w_pout, $w_qBase, $w_qOffset+=, 1
exitz                 $mzero

//==============================================================================

WKR_FUNCTION_LABEL_YRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

ld128                 $w_pin0_1, $w_qBase, 0
{
  ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  tapack                $w_tripack, $w_sBase, $w_qBaseLoadRetained, $w_qBase
  f32sisoamp            $w_pout, $azero, $w_pin0, TAMP_F32_E2_P0
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $azero, $w_pin1, TAMP_F32_E2_P2
}
{
  // We don't check for the worker doing no work. The offset in Z is guaranteed
  // to be 0 if there's no work done
  brneg                 $w_num_orig, LEndWorker
  f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E2_P0
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_h, $azeros, TAMP_F32_E2_P1
}
f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E2_P2
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_h, $azeros, TAMP_F32_E2_P3
}
// result available here
{
  rpt $w_num_orig, 3
  f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E2_P0
}
  {
    ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $azeros, $w_xin_h, $azeros, TAMP_F32_E2_P1
  }  
  {
    st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
    f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E2_P2
  } 
  {
    ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $azeros, $w_xin_h, $azeros, TAMP_F32_E2_P3
  }  
  {
    st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
    f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E2_P0
  }  
{
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout, $azero, $azeros, TAMP_F32_E1_P2
}
st64pace              $w_pout, $w_tripack+=, $mzero, 0b00

LEndWorker:
exitz                 $mzero

WORKER_FN_FOOTER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX


// =============================================================================
// Supervisor codelet

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME float float WORKER_NAME_POSTFIX BLOCK_LENGTH

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
