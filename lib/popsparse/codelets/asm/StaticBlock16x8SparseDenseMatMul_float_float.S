// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 16x8. This worker is called twice to
// compute a full 16x16.
//

#ifdef __IPU__
#include "StaticBlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseMatMulBlock___float_float_16_16
#define WORKER_NAME_POSTFIX ff16x8
#define BLOCK_LENGTH 16

// =============================================================================

// worker registers
#define w_qBaseLoadRetained                m3
#define w_sBase                            m2
#define w_qBase                            m9

#define w_tripack                          m10:11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11
#define w_num                              m4
#define w_stride                           m7

#define w_pin0_1                           a0:3  // retained over the x loop
#define w_pin0                             a0:1
#define w_pin1                             a2:3
#define w_xin_pin                          a0:3
#define w_xin                              a0:1
#define w_xin_l                            a0
#define w_xin_h                            a1
#define w_pin                              a2:3
#define w_pout                             a4:5
#define w_pout1                            a6:7
#define fp_clr_reg                         a7


WORKER_FN_HEADER WORKER_NAME_PREFIX WORKER_NAME_POSTFIX
LOAD_VERTEX_STATE
add                   $w_num_orig, $w_num_orig, -1
mul                   $w_off, $w_off, BLOCK_LENGTH
// Stride: 0x1 | 0x5
setzi                 $w_stride, 0x405   

ld32step              $azero, $mzero, $w_sBaseRetained+=, $w_off
ld32step              $azero, $mzero, $w_qBaseRetained+=, $w_off
brnz                  $w_num_orig, AfterStrideSetup
setzi                 $w_stride, 0
AfterStrideSetup:
add                   $w_num, $w_num_orig, -1

WKR_FUNCTION_LABEL_XRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

// offset by X and Y positions for the block processed
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
mul                   $w_xOffsetInQ, $w_xOffsetInQ, 8 * BLOCK_LENGTH
{
  add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  add                   $w_qBaseLoadRetained, $w_qBase, 64
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}
WKR_FUNCTION_LABEL_YRETAINED  WORKER_NAME_PREFIX WORKER_NAME_POSTFIX

ld128                 $w_pin0_1, $w_qBase, 0
{
  ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
  f32sisov2amp          $w_pout, $azero, $w_pin0, TAMP_F32_E4_P0
}
{  
  ld128                 $w_pin0_1, $w_qBase, 1
  f32sisov2amp          $w_pout, $azero, $w_pin1, TAMP_F32_E4_P1
}
{  
  brneg                 $w_num_orig, LEndWorker
  f32sisov2amp          $w_pout, $azero, $w_pin0, TAMP_F32_E4_P2
}
{  
  ld128                 $w_pin0_1, $w_qBase, 2
  f32sisov2amp          $w_pout, $azero, $w_pin1, TAMP_F32_E4_P3
}
{  
  add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS
  f32sisov2amp          $w_pout, $azero, $w_pin0, TAMP_F32_E4_P4
}
{  
  ld128                 $w_pin0_1, $w_qBase, 3
  f32sisov2amp          $w_pout, $azero, $w_pin1, TAMP_F32_E4_P5
}
{  
  tapack                $w_tripack, $w_sBase, $w_qBaseLoadRetained, $w_qBase
  f32sisov2amp          $w_pout, $azero, $w_pin0, TAMP_F32_E4_P6
}
{  
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_stride, 0b1000
  f32sisov2amp          $w_pout, $azero, $w_pin1, TAMP_F32_E4_P7
}
{  
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_stride, 0b1011
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P0
}
{  
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_stride, 0b1000
  f32sisov2amp          $w_pout, $w_xin_h, $w_pin, TAMP_F32_E4_P1
}
{  
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_stride, 0b1011
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P2
}
{  
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_stride, 0b1000
  f32sisov2amp          $w_pout, $w_xin_h, $w_pin, TAMP_F32_E4_P3
}
{  
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_stride, 0b1011
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P4
}
{  
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_stride, 0b1001
  f32sisov2amp          $w_pout, $w_xin_h, $w_pin, TAMP_F32_E4_P5
}
{  
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_stride, 0b1011
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P6
}
{  
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_stride, 0b1010
  f32sisov2amp          $w_pout, $w_xin_h, $w_pin, TAMP_F32_E4_P7
}
{  
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_stride, 0b1011
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P0
}
{  
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_stride, 0b1011
  f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P1
}
brz                   $w_num_orig, NumElemsEq1
{  
  ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b0000
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P2
}
{  
  ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000001
  f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P3
}
{  
  ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b000000
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P4
}

rpt $w_num,           (Loop_end_Amp - Loop_start_Amp)/8-1
Loop_start_Amp:
  {  
    ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000001
    f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P5
  }
  {  
    ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $w_stride, 0b000001
    f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P6
  }
  {  
    ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000001
    f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P7
  }
  {  
    ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b000000
    f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P0
  }
  {  
    ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000001
    f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P1
  }  
  {  
    ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b000000
    f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P2
  }
  {  
    ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000001
    f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P3
  }
  {  
    ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b000000
    f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P4
  }
Loop_end_Amp:
{  
  ld2xst64pace          $w_xin_pin, $w_pout1, $w_tripack+=, $mzero, 0b000001
  f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P5
}
{  
  ld2xst64pace          $w_xin_pin, $w_pout, $w_tripack+=, $mzero, 0b001111
  f32sisov2amp          $w_pout, $w_xin_l, $w_pin, TAMP_F32_E4_P6
}
{  
  st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout1, $w_xin_h, $w_pin, TAMP_F32_E4_P7
}
{  
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout, $azero, $azeros, TAMP_F32_E4_P0
}
{
  st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00 
  f32sisov2amp          $w_pout1, $azero, $azeros, TAMP_F32_E4_P1  
}
NumElemsEq1:
{  
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout, $azero, $azeros, TAMP_F32_E4_P2
}
{  
  st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout1, $azero, $azeros, TAMP_F32_E4_P3
}
{  
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout, $azero, $azeros, TAMP_F32_E4_P4
}
{  
  st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout1, $azero, $azeros, TAMP_F32_E4_P5
}
{  
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout, $azero, $azeros, TAMP_F32_E4_P6
}
{  
  st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00
  f32sisov2amp          $w_pout1, $azero, $azeros, TAMP_F32_E4_P7
}
st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
st64pace              $w_pout1, $w_tripack+=, $mzero, 0b00
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
