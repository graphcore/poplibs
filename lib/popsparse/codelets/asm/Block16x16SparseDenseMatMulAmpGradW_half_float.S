// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 16x16

#if defined(__IPU__)
#include "BlockSparseMatMulAmpGradW.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockAmpGradW___half_float_16_16

// =============================================================================

#define w_numZ                 m0
#define w_id                   m1
#define w_lt5                  m2
#define w_strides              m3
#define w_workerOff            m4
#define w_zOff                 m1
#define w_qWorkerOff           m6
#define w_rWorkerOff           m7
#define w_qGrad                m9
#define w_tripack              m8:9
#define w_rGrad                m10
#define w_qGradRetained        m11
#define w_strides8             m5


#define w_fp_clr_reg a0
#define w_xin a0:1
#define w_pin a2:3
#define w_xinpin a0:3
#define w_pout1 a4:5
#define w_pout2 a6:7

#define SIZEOF_INPUT_TYPE     2
#define SIZEOF_PARTIALS_TYPE  4
#define Z_PER_PASS            16
#define BLOCK_SIZE            16
#define WORKER_STACK_SIZE     0

DEF_STACK_USAGE WORKER_STACK_SIZE blockSparseDenseMultiplyGradWAmp_hf16x16
.section ".text.blockSparseDenseMultiplyGradWAmp_hf16x16", FUNCTION_IS_WORKER
.type blockSparseDenseMultiplyGradWAmp_hf16x16, @function
.global blockSparseDenseMultiplyGradWAmp_hf16x16
.global blockSparseDenseMultiplyGradWAmp_hf16x16_retained
.align 4
.worker
blockSparseDenseMultiplyGradWAmp_hf16x16:
ld32              $w_numZ, $mvertex_base, W_AMP_NUM_Z/4
get               $w_id, $WSR
and               $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
cmpult            $w_lt5, $w_id, 5
mul               $w_strides8, $w_lt5, 5
// Strides [0 | $w_numZ/4 - 3 | 0/5]
ld32              $w_strides, $mvertex_base, W_AMP_STRIDES/4
or                $w_strides, $w_strides, $w_strides8
mul               $w_strides, $w_strides, $w_lt5
mul               $w_workerOff, $w_id, 3 * SIZEOF_INPUT_TYPE
mul               $w_qWorkerOff, $w_workerOff, $w_numZ
mul               $w_rWorkerOff, $w_workerOff, SIZEOF_PARTIALS_TYPE * BLOCK_SIZE / SIZEOF_INPUT_TYPE

blockSparseDenseMultiplyGradWAmp_hf16x16_retained:

ld32              $w_qGradRetained, $mvertex_base, W_AMP_QGRAD_BASE/4
add               $w_qGradRetained, $w_qGradRetained, $w_qWorkerOff

// move pointers to correct position
blockSparseDenseMultiplyGradWAmp_hf16x16_retained_zOff:
ld32              $w_rGrad, $mvertex_base, W_AMP_RGRAD_BASE_BLOCK/4
add               $w_rGrad, $w_rGrad, $w_rWorkerOff

ld32              $w_zOff, $mvertex_base, W_AMP_OFFSET_Z/4
mul               $w_zOff, $w_zOff, Z_PER_PASS * SIZEOF_INPUT_TYPE
add               $w_qGrad, $w_qGradRetained, $w_zOff

{ 
  tapack            $w_tripack, $w_qGrad, $w_rGrad, $w_rGrad
  setzi             $w_fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{ 
  ld2x64pace        $azeros, $w_pin, $w_tripack+=, $mzero, 0b0001
  uput              $FP_CLR, $w_fp_clr_reg 
}
// Begin feeding partials
{ 
  ld2x64pace        $azeros, $w_pin, $w_tripack+=, $mzero, 0b0001
  f16v4sisoamp      $w_pout1, $azeros, $w_pin, TAMP_F16V4_E4_P0 
}

{ 
  ld2x64pace        $azeros, $w_pin, $w_tripack+=, $mzero, 0b0001
  f16v4sisoamp      $w_pout1, $azeros, $w_pin, TAMP_F16V4_E4_P1 
}

{ 
  ld2x64pace        $azeros, $w_pin, $w_tripack+=, $w_strides, 0b0111
  f16v4sisoamp      $w_pout1, $azeros, $w_pin, TAMP_F16V4_E4_P2 
}
  
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4sisoamp      $w_pout1, $azeros, $w_pin, TAMP_F16V4_E4_P3 
}

// Begin feeding inputs
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4sisoamp      $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E4_P0 
}

{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4sisoamp      $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E4_P1 
}

{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_strides, 0b0110
  f16v4sisoamp      $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E4_P2 
}

{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4sisoamp      $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E4_P3 
}

// Outputs are now available
{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4sisoamp      $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E4_P0 
}

{ 
  ld2x64pace        $w_xin, $w_pin, $w_tripack+=, $w_lt5, 0b0100
  f16v4sisoamp      $w_pout2, $w_xin, $w_pin, TAMP_F16V4_E4_P1 
}

brz               $w_lt5, LNumElemsEq1
{ 
  ld2xst64pace      $w_xinpin, $w_pout1, $w_tripack+=, $w_strides, 0b000110
  f16v4sisoamp      $w_pout1, $w_xin, $w_pin, TAMP_F16V4_E4_P2 
}

// Stop loading/feeding partials
{ 
  ldst64pace        $w_xin, $w_pout2, $w_tripack+=, $mzero, 0b0000
  f16v4sisoamp      $w_pout2, $w_xin, $w_pin, TAMP_F16V4_E4_P3 
}

{ 
  ldst64pace        $w_xin, $w_pout1, $w_tripack+=, $mzero, 0b0000
  f16v4sisoamp      $w_pout1, $w_xin, $azeros, TAMP_F16V4_E4_P0 
}

{ 
  ldst64pace        $w_xin, $w_pout2, $w_tripack+=, $w_strides, 0b0100
  f16v4sisoamp      $w_pout2, $w_xin, $azeros, TAMP_F16V4_E4_P1 
}

{ 
  ldst64pace        $w_xin, $w_pout1, $w_tripack+=, $mzero, 0b0000
  f16v4sisoamp      $w_pout1, $w_xin, $azeros, TAMP_F16V4_E4_P2 
}

// Stop loading inputs
{ 
  st64pace          $w_pout2, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp      $w_pout2, $w_xin, $azeros, TAMP_F16V4_E4_P3 
}

// Stop feeding inputs
{ 
  st64pace          $w_pout1, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp      $w_pout1, $azeros, $azeros, TAMP_F16V4_E4_P0 
}

{ 
  st64pace           $w_pout2, $w_tripack+=, $w_strides, 0b01
  f16v4sisoamp       $w_pout2, $azeros, $azeros, TAMP_F16V4_E4_P1 
}

LNumElemsEq1:
{ 
  st64pace          $w_pout1, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp      $w_pout1, $azeros, $azeros, TAMP_F16V4_E4_P2 
}

{ 
  st64pace          $w_pout2, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp      $w_pout2, $azeros, $azeros, TAMP_F16V4_E4_P3 
}

st64pace          $w_pout1, $w_tripack+=, $mzero, 0b00
st64pace          $w_pout2, $w_tripack+=, $mzero, 0b00
exitz             $mzero
.size blockSparseDenseMultiplyGradWAmp_hf16x16, . - blockSparseDenseMultiplyGradWAmp_hf16x16

// =============================================================================

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL_GRADW_AMP_SUP CODELET_NAME half float 16 blockSparseDenseMultiplyGradWAmp_hf16x16

#endif // defined(__IPU__)
