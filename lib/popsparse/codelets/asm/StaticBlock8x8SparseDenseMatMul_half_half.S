// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 8x8
//

#ifdef __IPU__
#include "StaticBlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseMatMulBlock___half_half_8_8
#define WORKER_NAME_POSTFIX hh8x8
#define BLOCK_LENGTH 8

// =============================================================================

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////

// =============================================================================

// worker registers
#define w_qBaseRead                        m3
#define w_sBase                            m2
#define w_qBase                            m9

#define w_tripack                          m10:11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11

#define w_xin                              a0:1
#define w_pin1_2                           a0:3
#define w_pin1                             a0:1
#define w_pin2                             a2:3
#define w_xin_pin2                         a0:3
#define w_pout1                            a4:5
#define w_pout                             a6:7
#define fp_clr_reg                         a6

WORKER_FN_HEADER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
LOAD_VERTEX_STATE
add                   $w_num_orig, $w_num_orig, -1
mul                   $w_off, $w_off, BLOCK_LENGTH
// dummy loads for halves
ldz16step             $mzero, $mzero, $w_sBaseRetained+=, $w_off
ldz16step             $mzero, $mzero, $w_qBaseRetained+=, $w_off

WKR_FUNCTION_LABEL_XRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// offset by X and Y positions for the block processed
// Note:
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
mul                   $w_xOffsetInQ, $w_xOffsetInQ, 8 * BLOCK_LENGTH
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
add                   $w_qBaseRead, $w_qBase, 16

// Fall through to code that uses zero partials
brneg                 $w_num_orig, LEndWorker
{
  ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{ 
  add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
  uput                  $FP_CLR, $fp_clr_reg 
}
ld64step              $w_xin, $mzero, $w_sBase+=, 1
{
  ld64step              $w_xin, $mzero, $w_sBase+=, 1
  f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E2_P0
}
{
  ld64step              $w_xin, $mzero, $w_sBase+=, 1
  f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E2_P1
}
{
  ld64step              $w_xin, $mzero, $w_sBase+=, 1
  f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E2_P0
}
tapack                $w_tripack, $w_sBase, $w_qBaseRead, $w_qBase
{
  rpt $w_num_orig, 1
  fnop
}
  {
    ldst64pace            $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0000
    f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E2_P1
  }  
  {
    ldst64pace            $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0000
    f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E2_P0
  }  
{
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f16v4hihov4amp        $w_pout, $azeros, $azeros, TAMP_F16V4_E2_P1
}
st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
exitz                 $mzero

WKR_FUNCTION_LABEL_YRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
brneg                 $w_num_orig, LEndWorker
// The partials are over-read by 6*4 = 24 bytes and the input by 20

ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
{ 
  add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  ld128                 $w_pin1_2, $mzero, $w_qBase, 0        
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  tapack                $w_tripack, $w_sBase, $w_qBaseRead, $w_qBase
  f16v4hihov4amp        $w_pout, $azeros, $w_pin1, TAMP_F16V4_E2_P0
}
{
  ld2x64pace            $w_xin, $w_pin2,  $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $azeros, $w_pin2, TAMP_F16V4_E2_P1
}
{
  ld2x64pace            $w_xin, $w_pin2,  $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $w_xin, $w_pin2, TAMP_F16V4_E2_P0
}
{
  ld2x64pace            $w_xin, $w_pin2,  $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $w_xin, $w_pin2, TAMP_F16V4_E2_P1
}
{
  ld2x64pace            $w_xin, $w_pin2,  $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $w_xin, $w_pin2, TAMP_F16V4_E2_P0
}
{
  ld2x64pace            $w_xin, $w_pin2,  $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout1, $w_xin, $w_pin2, TAMP_F16V4_E2_P1
}
{
  rpt $w_num_orig, 1
  fnop
}
  {
    ld2xst64pace          $w_xin_pin2, $w_pout, $w_tripack+=, $mzero, 0b000000
    f16v4hihov4amp        $w_pout, $w_xin, $w_pin2, TAMP_F16V4_E2_P0
  }  
  {
    ld2xst64pace          $w_xin_pin2, $w_pout1, $w_tripack+=, $mzero, 0b000000
    f16v4hihov4amp        $w_pout1, $w_xin, $w_pin2, TAMP_F16V4_E2_P1
  }  
st64pace            $w_pout, $w_tripack+=, $mzero, 0b00
st64pace            $w_pout1, $w_tripack+=, $mzero, 0b00
LEndWorker:
exitz                 $mzero

WORKER_FN_FOOTER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// =============================================================================
// Supervisor codelet

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half half WORKER_NAME_POSTFIX BLOCK_LENGTH

// =============================================================================


// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
