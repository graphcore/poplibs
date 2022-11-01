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

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseMatMulBlock___half_half_4_4
#define WORKER_NAME_POSTFIX hh4x4
#define BLOCK_LENGTH 4

// =============================================================================

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////

// =============================================================================

// worker registers
#define w_numDiv2                          m3
#define w_rem                              m6
#define w_sBase                            m2
#define w_qBase                            m9
#define w_num_012                          m5

#define w_tripack                          m10:11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11

#define w_xin                              a0:1
#define w_pin                              a2:3
#define w_xin_pin                          a0:3
#define w_pout1                            a4:5
#define w_pout                             a6:7
#define fp_clr_reg                         a1

#define w_xin_sp                           a4:5
#define w_pin0_sp                          a0:1
#define w_pin1_sp                          a2:3

WORKER_FN_HEADER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
LOAD_VERTEX_STATE

shr                   $w_numDiv2, $w_num_orig, 1
and                   $w_rem, $w_num_orig, 0x1
add                   $w_num_012, $w_num_orig, -3

// Multiply w_of by 4 * size_of s or q
ld64step              $azeros, $mzero, $w_sBaseRetained+=, $w_off
ld64step              $azeros, $mzero, $w_qBaseRetained+=, $w_off

WKR_FUNCTION_LABEL_XRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// Fall through to code implicitly zeros partials
{ 
  ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{ 
  mul                   $w_xOffsetInQ, $w_xOffsetInQ, 8 * BLOCK_LENGTH
  uput                  $FP_CLR, $fp_clr_reg 
}
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
brneg                 $w_num_012, LNumEq0And1And2_zp
add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
{
  ld64step              $w_xin, $mzero,  $w_sBase+=, 1
  f16v4hihov4amp        $w_pout, $azeros, $azeros, TAMP_F16V4_E1_P0
}
{
  ld64step              $w_xin, $mzero,  $w_sBase+=, 1
  f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E1_P0
}
// result available here 
{
  ld64step              $w_xin, $mzero, $w_sBase+=, 1
  f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E1_P0
}
tapack                $w_tripack, $w_sBase, $w_qBase, $w_qBase
rpt $w_numDiv2, 1
  {
    ldst64pace            $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0000
    f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E1_P0
  }  
  {
    ldst64pace            $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0000
    f16v4hihov4amp        $w_pout, $w_xin, $azeros, TAMP_F16V4_E1_P0
  }  
{
  brz                   $w_rem, LEndWorker
  // dummy read to ensure all accumumulators are zero
  f16v4hihov4amp        $w_pout1, $azeros, $azeros, TAMP_F16V4_E1_P0
}
st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
exitz                 $mzero

LNumEq0And1And2_zp:
brz                   $w_num_orig, LEndWorker

LNumEq1_zp:
ld64                  $w_xin_sp, $w_yOffsetInS, $w_sBaseRetained, 0
{
  // over-read for w_num = 1
  ld64                  $w_xin_sp, $w_yOffsetInS, $w_sBaseRetained, 1
  f16v4hihov4amp        $w_pin0_sp, $w_xin_sp, $azeros, TAMP_F16V4_E1_P0
}
{
  brz                   $w_rem, LNumEq2_sp
  f16v4hihov4amp        $w_pin0_sp, $w_xin_sp, $azeros, TAMP_F16V4_E1_P0
}
{
  st64                  $w_pin0_sp, $w_qBase, 0
  // Ensure zeros in accumulator
  f16v4hihov4amp        $azeros, $azeros, $azeros, TAMP_F16V4_E1_P0
}
exitz                 $mzero

LNumEq2_sp:
{
  st64                  $w_pin0_sp, $w_qBase, 0
  f16v4hihov4amp        $w_pin1_sp, $azeros, $azeros, TAMP_F16V4_E1_P0
}
st64                  $w_pin1_sp, $w_qBase, 1
exitz                 $mzero

WKR_FUNCTION_LABEL_YRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
brneg                 $w_num_012, LNumEq0And1And2
add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
tapack                $w_tripack, $w_sBase, $w_qBase, $w_qBase

// Allow over-reads as the maximum over-reads for numZ = 1 is 24 bytes
ld2x64pace            $azeros, $w_pin, $w_tripack+=, $mzero, 0b0011
{
  ld2x64pace            $w_xin, $w_pin,  $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $azeros, $w_pin, TAMP_F16V4_E1_P0
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $w_xin, $w_pin, TAMP_F16V4_E1_P0
}
// result available here 
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E1_P0
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f16v4hihov4amp        $w_pout, $w_xin, $w_pin, TAMP_F16V4_E1_P0
}
{ 
  rpt $w_numDiv2, 1
  fnop
}
  {
    ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000000
    f16v4hihov4amp        $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E1_P0
  }  
  {
    ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b000000
    f16v4hihov4amp        $w_pout, $w_xin, $w_pin, TAMP_F16V4_E1_P0
  }  

{
  brz                   $w_rem, LEndWorker
  // ensure zeros in accumulators
  f16v4hihov4amp        $azeros, $azeros, $azeros, TAMP_F16V4_E1_P0
}
st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00
LEndWorker:
exitz                 $mzero

LNumEq0And1And2:
brz                   $w_num_orig, LEndWorker

LNumEq1:
{
  ld64                  $w_xin_sp, $w_yOffsetInS, $w_sBaseRetained, 0
  f16v4hihov4amp        $azeros, $azeros, $w_pin0_sp, TAMP_F16V4_E1_P0
}
{
  // over-read for w_num = 1
  ld64                  $w_xin_sp, $w_yOffsetInS, $w_sBaseRetained, 1
  f16v4hihov4amp        $azeros, $w_xin_sp, $w_pin1_sp, TAMP_F16V4_E1_P0
}
{
  brz                   $w_rem, LNumEq2
  f16v4hihov4amp        $w_pin0_sp, $w_xin_sp, $azeros, TAMP_F16V4_E1_P0
}
{
  st64                  $w_pin0_sp, $w_qBase, 0
  // feed in zeros in accumulator
  f16v4hihov4amp        $azeros, $azeros, $azeros, TAMP_F16V4_E1_P0
}
exitz                 $mzero

LNumEq2:
{
  st64                  $w_pin0_sp, $w_qBase, 0
  f16v4hihov4amp        $w_pin1_sp, $azeros, $azeros, TAMP_F16V4_E1_P0
}
st64                  $w_pin1_sp, $w_qBase, 1
exitz                 $mzero

WORKER_FN_FOOTER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// =============================================================================
// Supervisor codelet

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half half WORKER_NAME_POSTFIX BLOCK_LENGTH

// =============================================================================


// Zero output/partials
//
// Performance: 12 + num_samples / 2
#define ZERO_WORKER_CODELET zeroStaticDenseOutput64

DEF_STACK_USAGE 0 ZERO_WORKER_CODELET
.section .text.ZERO_WORKER_CODELET, FUNCTION_IS_WORKER
.type ZERO_WORKER_CODELET, @function
.globl ZERO_WORKER_CODELET
.worker
.align 8

#define wkr_id_zv                       m0
#define zero_info_zv                    m1
#define zero_info_div_12_zv             m2
#define outchan_ptr_zv                  m3
 
ZERO_WORKER_CODELET:
get           $wkr_id_zv, $WSR
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
ld32          $zero_info_zv, $mvertex_base, VBASE_ZERO_NUM64/4
brz           $zero_info_zv, Loop_end_zero_64

// For n with 0 <= n <= 65533 this does a division by 6 with the remainder
// split amongst workers.
add           $zero_info_div_12_zv, $zero_info_zv, 6
sub           $zero_info_div_12_zv, $zero_info_div_12_zv, $wkr_id_zv
mul           $zero_info_div_12_zv, $zero_info_div_12_zv, 21845
shr           $zero_info_div_12_zv, $zero_info_div_12_zv, 17

ld32          $outchan_ptr_zv, $mvertex_base, VBASE_ZERO_PTR/4
ld64step      $azeros, $mzero, $outchan_ptr_zv+=, $wkr_id_zv
{
  rpt           $zero_info_div_12_zv, (Loop_end_zero_64 - Loop_start_zero_64)/8 - 1
  fnop
}
Loop_start_zero_64:
  {
    st64step      $azeros, $mzero, $outchan_ptr_zv+=, 6
    fnop
  }
Loop_end_zero_64:
exitz         $mzero

.size ZERO_WORKER_CODELET, . - ZERO_WORKER_CODELET


// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
