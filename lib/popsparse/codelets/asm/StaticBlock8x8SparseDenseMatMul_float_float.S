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

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseMatMulBlock___float_float_8_8
#define WORKER_NAME_POSTFIX ff8x8
#define BLOCK_LENGTH 8

// =============================================================================

// worker registers
#define w_qBaseLoadRetained                m3
#define w_sBase                            m2
#define w_qBase                            m9
#define w_qRewind                          m4

#define w_tripack                          m10:11
#define w_tripack_l                        m10
#define w_tripack_h                        m11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11

#define w_xin                              a0:1
#define w_xin_l                            a0
#define w_xin_h                            a1
#define w_xin2                             a2:3
#define w_xin2_l                           a2
#define w_xin2_h                           a3



#define w_pin0_1                           a0:3  
#define w_pin0                             a0:1
#define w_pin1                             a2:3
#define w_pin                              a4:5
#define w_pout                             a6:7
#define w_pout_final2                      a4:5
#define fp_clr_reg                         a7
#define w_pout_final1                      a2:3
#define w_xin_pin                          a0:3
#define w_xin_pin_init                     a4:7
#define w_pin_init                         a6:7


WORKER_FN_HEADER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
LOAD_VERTEX_STATE
add                   $w_num_orig, $w_num_orig, -1
mul                   $w_off, $w_off, BLOCK_LENGTH

ld32step              $azero, $mzero, $w_sBaseRetained+=, $w_off
ld32step              $azero, $mzero, $w_qBaseRetained+=, $w_off

WKR_FUNCTION_LABEL_XRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// offset by X and Y positions for the block processed
// Note:
// set partials in w_pin0 and w_pin1 to zero as partials are zeroed
// anyway.
{
  ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
  mov                   $w_pin0, $azeros
}
{
  mul                   $w_xOffsetInQ, $w_xOffsetInQ, 8 * BLOCK_LENGTH
  mov                   $w_pin1, $azeros
}
{
  add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}
{
  add                   $w_qBaseLoadRetained, $w_qBase, 32
  uput                  $FP_CLR, $fp_clr_reg 
}

WKR_FUNCTION_LABEL_YRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
{
  add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
  f32sisoamp            $w_pout, $azero, $w_pin0, TAMP_F32_E4_P0
}


{  
  ld128                 $w_pin0_1, $w_qBase, 1
  f32sisoamp            $w_pout, $azero, $w_pin1, TAMP_F32_E4_P2
}
{
  tapack                $w_tripack, $w_sBase, $w_qBaseLoadRetained, $w_qBase
  f32sisoamp            $w_pout, $azero, $w_pin0, TAMP_F32_E4_P4
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $azero, $w_pin1, TAMP_F32_E4_P6
}
{
  // When no work is allocated to the worker, the offset is always set to the
  // start of the partials
  brneg                 $w_num_orig, LEndWorker
  f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P0
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_h, $azeros, TAMP_F32_E4_P1
}
{
  // For each sample, the q read pointer is incremented by 8 * 4 bytes
  mul                   $w_qRewind, $w_num_orig, 8 * 4
  f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P2
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_h, $azeros, TAMP_F32_E4_P3
}
{
  // Compensate for num_orig decrement by 1 and 5 over-reads
  add                   $w_qRewind, $w_qRewind, 5 * 8 + 4 * 8
  f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P4
}
{
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_h, $azeros, TAMP_F32_E4_P5
}
{
  ld2x64pace            $w_xin2, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P6
}
{
  rpt                   $w_num_orig, (Loop_end_Amp - Loop_start_Amp)/8-1
  f32sisoamp            $w_pout, $w_xin_h, $azeros, TAMP_F32_E4_P7
}
Loop_start_Amp:
  {
    ld2x64pace            $azeros, $w_pin, $w_tripack+=, $mzero, 0b0011
    f32sisoamp            $w_pout, $w_xin2_l, $w_pin, TAMP_F32_E4_P0
  }
  {
    ldst64pace            $w_xin2, $w_pout, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $w_pout, $w_xin2_h, $azeros, TAMP_F32_E4_P1
  }
  {
    ld2x64pace            $azeros, $w_pin, $w_tripack+=, $mzero, 0b0011
    f32sisoamp            $w_pout, $w_xin2_l, $w_pin, TAMP_F32_E4_P2
  }
  {
    ldst64pace            $w_xin2, $w_pout, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $w_pout, $w_xin2_h, $azeros, TAMP_F32_E4_P3
  }
  {
    ld2x64pace            $azeros, $w_pin, $w_tripack+=, $mzero, 0b0011
    f32sisoamp            $w_pout, $w_xin2_l, $w_pin, TAMP_F32_E4_P4
  }
  {
    ldst64pace            $w_xin2, $w_pout, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $w_pout, $w_xin2_h, $azeros, TAMP_F32_E4_P5
  }
  {
    ld2x64pace            $azeros, $w_pin, $w_tripack+=, $mzero, 0b0011
    f32sisoamp            $w_pout, $w_xin2_l, $w_pin, TAMP_F32_E4_P6
  }
  {
    ldst64pace            $w_xin2, $w_pout, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $w_pout, $w_xin2_h, $azeros, TAMP_F32_E4_P7
  }
Loop_end_Amp:
{
  // rewind partials read pointer to point to beginning
  sub                   $w_tripack_h, $w_tripack_h, $w_qRewind 
  f32sisoamp            $w_pout, $azero, $azeros, TAMP_F32_E4_P0
}
{
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout_final1, $azero, $azeros, TAMP_F32_E4_P2
}
{
  // load first partial
  ld2xst64pace          $w_xin_pin_init, $w_pout_final1, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout_final1, $azero, $azeros, TAMP_F32_E4_P4
}
{
  // load second partial
  ld2xst64pace          $w_xin_pin, $w_pout_final1, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout_final2, $azero, $azeros, TAMP_F32_E4_P6
}
{
  st64pace              $w_pout_final2, $w_tripack+=, $mzero, 0b00
  mov                   $w_pin0, $w_pin_init
}
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
