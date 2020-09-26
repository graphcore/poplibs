// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 16x16. Uses AMP to divide the
// 16x16 output matrix computation among workers. Each call
// processes an [8x16] * [16x16] matrix. Two calls are needed
// to process the whole [16x16] matrix.
//
// Each worker processes a set of columns and the split is 
// [3, 3, 3, 3, 3, 1]. The imbalance between the max and minimum
// is deliberate to allow supervisor processing if possible.

#if defined(__IPU__)
#if __IPU_ARCH_VERSION__ < 2
#include "BlockSparseMatMulAmpGradW.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockAmpGradW___half_half_16_16

// =============================================================================

#define w_id                   m1
#define w_lt5                  m2
#define w_zOff                 m3
#define w_strides8             m4
#define w_qWorkerOff           m6
#define w_rWorkerOff           m7
#define w_qGrad                m9
#define w_tripack              m8:9
#define w_qGradRetained        m10
#define w_rGrad                m11

#define w_fp_clr_reg          a0
#define w_xin                 a0:1
#define w_pin_low             a2
#define w_pin_high            a3
#define w_pin                 a2:3
#define w_xinpin              a0:3
#define w_pout_low            a4
#define w_pout_high           a5
#define w_pout                a4:5

#define SIZEOF_INPUT_TYPE     2
#define SIZEOF_PARTIALS_TYPE  2
#define Z_PER_PASS            16
#define BLOCK_SIZE            16
#define WORKER_STACK_SIZE     0

DEF_STACK_USAGE WORKER_STACK_SIZE blockSparseDenseMultiplyGradWAmp_hh16x16
.section ".text.blockSparseDenseMultiplyGradWAmp_hh16x16", FUNCTION_IS_WORKER
.type blockSparseDenseMultiplyGradWAmp_hh16x16, @function
.global blockSparseDenseMultiplyGradWAmp_hh16x16
.global blockSparseDenseMultiplyGradWAmp_hh16x16_retained
.align 4
.worker
blockSparseDenseMultiplyGradWAmp_hh16x16:
get               $w_id, $WSR
and               $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
cmpult            $w_lt5, $w_id, 5
mul               $w_qWorkerOff, $w_id, 3 * SIZEOF_INPUT_TYPE * Z_PER_PASS
mul               $w_rWorkerOff, $w_id, SIZEOF_PARTIALS_TYPE * BLOCK_SIZE * 3

blockSparseDenseMultiplyGradWAmp_hh16x16_retained:
ld32              $w_qGradRetained, $mvertex_base, W_AMP_QGRAD_BASE/4
add               $w_qGradRetained, $w_qGradRetained, $w_qWorkerOff

blockSparseDenseMultiplyGradWAmp_hh16x16_retained_zOff:

// move pointers to correct position
ld32              $w_rGrad, $mvertex_base, W_AMP_RGRAD_BASE_BLOCK/4
add               $w_rGrad, $w_rGrad, $w_rWorkerOff

ld32              $w_zOff, $mvertex_base, W_AMP_OFFSET_Z/4
add               $w_qGrad, $w_qGradRetained, $w_zOff

{ 
  tapack            $w_tripack, $w_qGrad, $w_rGrad, $w_rGrad
  setzi             $w_fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0001
  uput              $FP_CLR, $w_fp_clr_reg 
}
// Begin feeding partials
{
  mul               $w_strides8, $w_lt5, 3
  f16v4hihoamp      $w_pout_low, $azeros, $w_pin_low, TAMP_F16V4_E4_P0
}
{ 
  ld2x64pace        $azeros, $w_pin, $w_tripack+=, $w_strides8, 0b0110
  f16v4hihoamp      $w_pout_high, $azeros, $w_pin_high, TAMP_F16V4_E4_P1 
}

f16v4hihoamp      $w_pout_low, $azeros, $w_pin_low, TAMP_F16V4_E4_P2 
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4hihoamp      $w_pout_high, $azeros, $w_pin_high, TAMP_F16V4_E4_P3 
}
// Begin feeding inputs
{ 
  ld2x64pace        $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
  f16v4hihoamp      $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P0 
}
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_strides8, 0b0100
  f16v4hihoamp      $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P1 
}
{ 
  ld2x64pace        $w_xin, $azeros, $w_tripack+=, $mzero, 0b1100
  f16v4hihoamp      $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P2 
}
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P3 
}
// Outputs are now available
{ 
  ld2x64pace        $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
  f16v4hihoamp      $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P0 
}
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_strides8, 0b0100
  f16v4hihoamp      $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P1 
}
brz               $w_lt5, LNumElemsEq1
{ 
  ldst64pace        $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0000
  f16v4hihoamp      $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P2 
}
{ 
  ld2x64pace        $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
  f16v4hihoamp      $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P3 
}
// Stop loading/feeding partials
{ 
  ldst64pace        $w_xin, $w_pout, $w_tripack+=, $w_strides8, 0b0100
  f16v4hihoamp      $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P0 
}
{ 
  ld2x64pace        $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
  f16v4hihoamp      $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P1 
}
{ 
  ldst64pace        $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0001
  f16v4hihoamp      $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P2 
}
// Stop loading inputs
f16v4hihoamp      $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P3
{ 
  st64pace          $w_pout, $w_tripack+=, $w_strides8, 0b01
  f16v4hihoamp      $w_pout_low, $azeros, $azero, TAMP_F16V4_E4_P0 
}

f16v4hihoamp      $w_pout_high, $azeros, $azero, TAMP_F16V4_E4_P1

LNumElemsEq1:
{ 
  st64pace          $w_pout, $w_tripack+=, $mzero, 0b00
  f16v4hihoamp      $w_pout_low, $azeros, $azero, TAMP_F16V4_E4_P2 
}

f16v4hihoamp      $w_pout_high, $azeros, $azero, TAMP_F16V4_E4_P3
st64pace          $w_pout, $w_tripack+=, $mzero, 0b00
exitz             $mzero

.size blockSparseDenseMultiplyGradWAmp_hh16x16, . - blockSparseDenseMultiplyGradWAmp_hh16x16

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL_GRADW_AMP_SUP CODELET_NAME half half 16 blockSparseDenseMultiplyGradWAmp_hh16x16

#endif // __IPU_ARCH_VERSION__ <2
#endif // defined(__IPU__)
