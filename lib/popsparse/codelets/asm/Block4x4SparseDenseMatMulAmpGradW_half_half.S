// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 4x4. Uses AMP to divide the
// 4x16 output matrix computation among workers. 
//
// Each worker processes a set of columns and the split is 
// [1, 1, 1, 1, 0, 0]. The imbalance between the max and minimum
// is deliberate to allow supervisor processing if possible.

#if defined(__IPU__)
#include "BlockSparseMatMulAmpGradW.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockAmpGradW___half_half_4_4

// =============================================================================

#define w_id                   m1
#define w_lt4                  m2
#define w_zOff                 m5
#define w_qWorkerOff           m6
#define w_rWorkerOff           m7
#define w_rGrad                m8
#define w_qGrad                m9
#define w_qGradRetained        m10

#define w_fp_clr_reg          a0
#define w_xin                 a0:1
#define w_pin_low             a2
#define w_pin_high            a3
#define w_pin                 a2:3
#define w_xinpin              a0:3
#define w_pout_low            a4
#define w_pout_high           a5
#define w_pout                a4:5
#define w_pin_retained        a6:7


#define SIZEOF_INPUT_TYPE     2
#define SIZEOF_PARTIALS_TYPE  2
#define Z_PER_PASS            16
#define BLOCK_SIZE            4
#define WORKER_STACK_SIZE     0

DEF_STACK_USAGE WORKER_STACK_SIZE blockSparseDenseMultiplyGradWAmp_hh4x4
.section ".text.blockSparseDenseMultiplyGradWAmp_hh4x4", FUNCTION_IS_WORKER
.type blockSparseDenseMultiplyGradWAmp_hh4x4, @function
.align 4
.worker
blockSparseDenseMultiplyGradWAmp_hh4x4:
get               $w_id, $WSR
and               $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
cmpult            $w_lt4, $w_id, 4
mul               $w_qWorkerOff, $w_id, Z_PER_PASS * SIZEOF_INPUT_TYPE
mul               $w_rWorkerOff, $w_id, SIZEOF_PARTIALS_TYPE * BLOCK_SIZE

blockSparseDenseMultiplyGradWAmp_hh4x4_retained:
ld32              $w_rGrad, $mvertex_base, W_AMP_RGRAD_BASE_BLOCK/4

blockSparseDenseMultiplyGradWAmp_hh4x4_retained_zOff:
{  
  brz               $w_lt4, LExit
  setzi             $w_fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{ 
  ld64              $w_pin, $w_rWorkerOff, $w_rGrad, 0
  uput              $FP_CLR, $w_fp_clr_reg 
}
// Begin feeding partials
{
  ld32              $w_zOff, $mvertex_base, W_AMP_OFFSET_Z/4
  f16v4hihoamp      $w_pout_low, $azeros, $w_pin_low, TAMP_F16V4_E4_P0
}
{
  add               $w_zOff,  $w_qWorkerOff, $w_zOff
  f16v4hihoamp      $w_pout_high, $azeros, $w_pin_high, TAMP_F16V4_E4_P1 
}
{
  ld32              $w_qGradRetained, $mvertex_base, W_AMP_QGRAD_BASE/4
  f16v4hihoamp      $w_pout_low, $azeros, $azero, TAMP_F16V4_E4_P2 
}
{ 
  ld64              $w_xin, $w_zOff, $w_qGradRetained, 0
  f16v4hihoamp      $w_pout_high, $azeros, $azero, TAMP_F16V4_E4_P3 
}
// Begin feeding inputs
{ 
  ld64              $w_xin, $w_zOff, $w_qGradRetained, 1
  f16v4hihoamp      $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P0 
}
{ 
  ld64              $w_xin, $w_zOff, $w_qGradRetained, 2
  f16v4hihoamp      $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P1 
}
{ 
  ld64              $w_xin, $w_zOff, $w_qGradRetained, 3
  f16v4hihoamp      $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P2 
}
f16v4hihoamp      $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P3 

// Outputs are now available
f16v4hihoamp      $w_pout_low, $azeros, $azero, TAMP_F16V4_E4_P0 
f16v4hihoamp      $w_pout_high, $azeros, $azero, TAMP_F16V4_E4_P1 
st64              $w_pout, $w_rWorkerOff, $w_rGrad, 0

LExit:
exitz             $mzero

.size blockSparseDenseMultiplyGradWAmp_hh4x4, . - blockSparseDenseMultiplyGradWAmp_hh4x4

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL_GRADW_AMP_SUP CODELET_NAME half half 4 blockSparseDenseMultiplyGradWAmp_hh4x4

#endif // defined(__IPU__)
