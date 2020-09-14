// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 4x4. Uses AMP to divide the
// 4x8 output matrix computation among workers. 
//
// Each worker processes a set of columns and the split is 
// [1, 1, 1, 1, 0, 0]. The imbalance between the max and minimum
// is deliberate to allow supervisor processing if possible.


#ifdef __IPU__
#include "BlockSparseMatMulAmpGradW.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockAmpGradW___float_float_4_4
// =============================================================================

// =============================================================================

#define w_numZ                 m0
#define w_id                   m1
#define w_lt4                  m2
#define w_zOff                 m5
#define w_qWorkerOff           m6
#define w_rWorkerOff           m7
#define w_rGrad                m8
#define w_qGrad                m9

#define w_fp_clr_reg           a1
#define w_xin_0                a0
#define w_xin_1                a1
#define w_xin                  a0:1
#define w_pin                  a2:3
#define w_null2                a4:5
#define w_pout                 a6:7
#define w_null1                a14
#define w_null                 azeros

#define SIZEOF_INPUT_TYPE      4
#define SIZEOF_PARTIALS_TYPE   4
#define Z_PER_PASS             8
#define BLOCK_SIZE             4
#define WORKER_STACK_SIZE      0

DEF_STACK_USAGE WORKER_STACK_SIZE blockSparseDenseMultiplyGradWAmp_ff4x4
.section ".text.blockSparseDenseMultiplyGradWAmp_ff4x4", FUNCTION_IS_WORKER
.type blockSparseDenseMultiplyGradWAmp_ff4x4, @function
.align 4
.worker
blockSparseDenseMultiplyGradWAmp_ff4x4:
ld32              $w_numZ, $mvertex_base, W_AMP_NUM_Z/4
get               $w_id, $WSR
and               $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
cmpult            $w_lt4, $w_id, 4
mul               $w_qWorkerOff, $w_id, $w_numZ
mul               $w_rWorkerOff, $w_id, SIZEOF_PARTIALS_TYPE * BLOCK_SIZE

blockSparseDenseMultiplyGradWAmp_ff4x4_retained:
ld32              $w_rGrad, $mvertex_base, W_AMP_RGRAD_BASE_BLOCK/4

blockSparseDenseMultiplyGradWAmp_ff4x4_retained_zOff:
brz               $w_lt4, LExit
ld32              $w_zOff, $mvertex_base, W_AMP_OFFSET_Z/4
{
  ld32              $w_qGrad, $mvertex_base, W_AMP_QGRAD_BASE/4
  setzi             $w_fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  ld64              $w_pin, $w_rWorkerOff, $w_rGrad, 0
  uput              $FP_CLR, $w_fp_clr_reg 
}
{
  ld64              $w_pin, $w_rWorkerOff, $w_rGrad, 1
  f32sisoamp        $w_pout, $w_null1, $w_pin, TAMP_F32_E4_P0
}
{
  mul               $w_zOff, $w_zOff, Z_PER_PASS * SIZEOF_INPUT_TYPE
  f32sisoamp        $w_pout, $w_null1, $w_pin, TAMP_F32_E4_P2
}
{
  ld32step          $mzero, $mzero, $w_qGrad+=, $w_qWorkerOff
  f32sisoamp        $w_pout, $w_null1, $azeros, TAMP_F32_E4_P4
}
{
  ld64step          $w_xin, $w_zOff, $w_qGrad+=, 1
  f32sisoamp        $w_pout, $w_null1, $azeros, TAMP_F32_E4_P6
}
// Start providing inputs
f32sisoamp        $w_pout, $w_xin_0, $azeros, TAMP_F32_E4_P0
{
  ld64step          $w_xin, $w_zOff, $w_qGrad+=, 1
  f32sisoamp        $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P1
}
f32sisoamp        $w_pout, $w_xin_0, $azeros, TAMP_F32_E4_P2
{
  ld64step          $w_xin, $w_zOff, $w_qGrad+=, 1
  f32sisoamp        $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P3
}
f32sisoamp        $w_pout, $w_xin_0, $azeros, TAMP_F32_E4_P4
{
  ld64            $w_xin, $w_zOff, $w_qGrad, 0
  f32sisoamp        $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P5
}
f32sisoamp        $w_pout, $w_xin_0, $azeros, TAMP_F32_E4_P6
f32sisoamp        $w_null, $w_xin_1, $azeros, TAMP_F32_E4_P7
// Start recording output
f32sisoamp        $w_pout, $azero, $azeros, TAMP_F32_E4_P0
{
  st64              $w_pout, $w_rWorkerOff, $w_rGrad, 0
  f32sisoamp        $w_pout, $azero, $azeros, TAMP_F32_E4_P2
}
st64              $w_pout, $w_rWorkerOff, $w_rGrad, 1
LExit:
exitz             $mzero

.size blockSparseDenseMultiplyGradWAmp_ff4x4, . - blockSparseDenseMultiplyGradWAmp_ff4x4

// =============================================================================

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL_GRADW_AMP_SUP CODELET_NAME float float 4 blockSparseDenseMultiplyGradWAmp_ff4x4

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
