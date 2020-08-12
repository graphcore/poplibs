// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 16x16

#if defined(__IPU__)
#if __IPU_ARCH_VERSION__ < 2
#include "BlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlock___half_half_16_16

// =============================================================================

.extern zeroDenseOutFloat

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////

// =============================================================================

#define w_id m0
#define w_worklist m1
#define w_offset m2
#define w_num_orig m3 // Retain
#define w_zStrideInQ m4
#define w_zStrideInS m5
#define w_offsetScaledQ m6
#define w_offsetScaledS m7
#define w_zStrides1 m8 // Retain
#define w_zStrides2 m9 // Retain

#define w_xOffsetInQ m10
#define w_yOffsetInS m11

#define w_fp_clr_reg a0
#define w_xin a0:1
#define w_pin_low a2
#define w_pin_high a3
#define w_pin a2:3
#define w_xinpin a0:3
#define w_pout_low a4
#define w_pout_high a5
#define w_pout a4:5

#define WORKER_STACK_SIZE 0

DEF_STACK_USAGE WORKER_STACK_SIZE blockSparseDenseMultiply_hh16x16
.section ".text.blockSparseDenseMultiply_hh16x16", FUNCTION_IS_WORKER
.type blockSparseDenseMultiply_hh16x16, @function
.align 8
.worker
blockSparseDenseMultiply_hh16x16:
  get $w_id, $WSR
  and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

  // Two unsigned shorts per worker in worklist, get byte offset
  shl $w_id, $w_id, 2
  ld32 $w_worklist, $mvertex_base, W_WORKLIST/4
  ldz16 $w_offset, $w_id, $w_worklist, 0
  ldz16 $w_num_orig, $w_id, $w_worklist, 1

  ld32 $w_zStrideInQ, $mvertex_base, W_ZSTRIDEINQ/4
  ld32 $w_zStrideInS, $mvertex_base, W_ZSTRIDEINS/4

  mul $w_offsetScaledQ, $w_zStrideInQ, $w_offset
  mul $w_offsetScaledS, $w_zStrideInS, $w_offset

  // The striding in terms of half elements in the inner loop for inputs is:
  // 4, 4, 4, stride - 12
  // The striding in terms of half elements in the inner loop for partials is:
  // 4, stride - 4
  // and since stride is in multiples of 64-bits we divide this calculation by
  // the number of halfs per 64-bits, 4.
  add $w_zStrideInQ, $w_zStrideInQ, -1
  add $w_zStrideInS, $w_zStrideInS, -3

  // For num > 2:
  // w_zStrides1 = [ 0 | (zStrideinQ - 4) | (zStrideInS - 12) ]
  // w_zStrides2 = [ 0 | (zStrideinQ - 4) | (zStrideInS - 12) ]
  shl $w_zStrides1, $w_zStrideInQ, 10
  or $w_zStrides1, $w_zStrides1, $w_zStrideInS

//====================
#undef w_zStrideInQ
//====================
#define w_num m4
//====================
  add $w_num, $w_num_orig, -2
  brpos $w_num, .LSetPosStrides

  // For num = 1
  // w_zStrides1 = [ 0 |                0 |                 0 ]
  // w_zStrides2 = [ 0 |                0 |                 0 ]
  mov $w_zStrides1, $mzero
  mov $w_zStrides2, $mzero
  bri .LSetStridesComplete

.LSetPosStrides:
  brnz $w_num, .LSetStridesComplete

  // For num = 2
  // w_zStrides1 = [ 0 | (zStrideinQ - 4) | (zStrideInS - 12) ]
  // w_zStrides2 = [ 0 |                4 | (zStrideInS - 12) ]
  setzi $w_zStrides2, (1 << 10)
  or $w_zStrides2, $w_zStrides2, $w_zStrideInS

.LSetStridesComplete:
  add $w_num, $w_num, -1

//====================
#undef w_id 
#undef w_worklist
#undef w_offset
//====================
#define w_qBaseRetained m0 // Retain
#define w_sBaseRetained m1 // Retain
#define w_qOffset m2 // Retain
//====================
  ld32 $w_qBaseRetained, $mvertex_base, W_Q_BASE/4
  ld32 $w_sBaseRetained, $mvertex_base, W_S_BASE/4
  setzi $w_qOffset, 0

  ld64step $azeros, $mzero, $w_qBaseRetained+=, $w_offsetScaledQ
  ld64step $azeros, $mzero, $w_sBaseRetained+=, $w_offsetScaledS

//====================
#undef w_offsetScaledQ
#undef w_offsetScaledS
//====================
#define w_qBase m6
#define w_sBase m7
//====================

blockSparseDenseMultiply_hh16x16_retained:
  brz $w_num_orig, 0f

  ld32 $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
  shl $w_xOffsetInQ, $w_xOffsetInQ, 1
  add $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
  add $w_qBase, $w_qBase, $w_qOffset
  ld32 $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
  shl $w_yOffsetInS, $w_yOffsetInS, 1
  add $w_sBase, $w_sBaseRetained, $w_yOffsetInS

//====================
#undef w_xOffsetInQ
#undef w_yOffsetInS
//====================
#define w_tripack m10:11
//====================

  { tapack $w_tripack, $w_sBase, $w_qBase, $w_qBase
    setzi $w_fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT }

  // s_ptr += 0, q_ld_ptr += 4
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0001
    uput $FP_CLR, $w_fp_clr_reg }

  // Begin feeding partials

  // s_ptr += 0, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0101
    f16v4hihoamp $w_pout_low, $azeros, $w_pin_low, TAMP_F16V4_E4_P0 }

  // s_ptr += 0, q_ld_ptr += (zStrideinQ - 4)
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $w_zStrides1, 0b1011
    f16v4hihoamp $w_pout_high, $azeros, $w_pin_high, TAMP_F16V4_E4_P1 }

  brneg $w_num, .LNumElemsEq1Or2

  // s_ptr += 0, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0101
    f16v4hihoamp $w_pout_low, $azeros, $w_pin_low, TAMP_F16V4_E4_P2 }
  
  // s_ptr += 4, q_ld_ptr += 4
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
    f16v4hihoamp $w_pout_high, $azeros, $w_pin_high, TAMP_F16V4_E4_P3 }

  // Begin feeding inputs

  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P0 }

  // s_ptr += 4, q_ld_ptr += (zStrideinQ - 4)
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $w_zStrides1, 0b1000
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P1 }

  // s_ptr += (zStrideInS - 12), q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $w_zStrides1, 0b1101
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P2 }

  // s_ptr += 4, q_ld_ptr += 4
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P3 }

  // Outputs are now available

  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P0 }

  // s_ptr += 4, q_ld_ptr += (zStrideinQ - 4)
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $w_zStrides1, 0b1000
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P1 }

  // Begin storing
  rpt $w_num, (2f - 1f) / 8 - 1
1:
  // s_ptr += (zStrideInS - 12), q_st_ptr += 4
  { ldst64pace $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0001
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P2 }

  // s_ptr += 4, q_ld_ptr += 4
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P3 }

  // s_ptr += 4, q_st_ptr += (zStrideinQ - 4)
  { ldst64pace $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b1000
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P0 }
  
  // s_ptr += 4, q_ld_ptr += (zStrideinQ - 4)
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $w_zStrides1, 0b1000
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P1 }
2:

  // s_ptr += (zStrideInS - 12), q_st_ptr += 4
  { ldst64pace $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0001
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P2 }

  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P3 }

  // Stop loading/feeding partials

  // s_ptr += 4, q_st_ptr += (zStrideinQ - 4)
  { ldst64pace $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b1000
    f16v4hihoamp $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P0 }

  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
    f16v4hihoamp $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P1 }

.LNumElemsEq2:

  // s_ptr += 0, q_st_ptr += 4
  { ldst64pace $w_xin, $w_pout, $w_tripack+=, $mzero, 0b0001
    f16v4hihoamp $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P2 }

  // Stop loading inputs

  f16v4hihoamp $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P3

  // Stop feeding inputs

  // q_st_ptr += (zStridesInQ - 12)
  { st64pace $w_pout, $w_tripack+=, $w_zStrides1, 0b10
    f16v4hihoamp $w_pout_low, $azeros, $azero, TAMP_F16V4_E4_P0 }

  f16v4hihoamp $w_pout_high, $azeros, $azero, TAMP_F16V4_E4_P1

.LNumElemsEq1:

  // q_st_ptr += 4
  { st64pace $w_pout, $w_tripack+=, $mzero, 0b00
    f16v4hihoamp $w_pout_low, $azeros, $azero, TAMP_F16V4_E4_P2 }

  f16v4hihoamp $w_pout_high, $azeros, $azero, TAMP_F16V4_E4_P3

  // q_st_ptr += 0
  st64pace $w_pout, $w_tripack+=, $mzero, 0b01

  sub $w_qOffset, 16, $w_qOffset

0:
  exitz $mzero

.LNumElemsEq1Or2:

  // For num = 1:
  // s_ptr += 0, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 0, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0101
    f16v4hihoamp $w_pin_low, $azeros, $w_pin_low, TAMP_F16V4_E4_P2 }
  
  // For num = 1:
  // s_ptr += 4, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 4, q_ld_ptr += 4
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $w_zStrides2, 0b1000
    f16v4hihoamp $w_pout_high, $azeros, $w_pin_high, TAMP_F16V4_E4_P3 }

  // Begin feeding inputs

  // For num = 1:
  // s_ptr += 4, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $mzero, 0b0100
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P0 }

  // For num = 1:
  // s_ptr += 4, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0100
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P1 }

  // For num = 1:
  // s_ptr += 0, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += (zStrideInS - 12), q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $w_zStrides2, 0b1101
    f16v4hihoamp $w_pout_low, $w_xin, $w_pin_low, TAMP_F16V4_E4_P2 }

  // For num = 1:
  // s_ptr += 0, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $w_zStrides2, 0b1110
    f16v4hihoamp $w_pout_high, $w_xin, $w_pin_high, TAMP_F16V4_E4_P3 }

  // Outputs are now available

  // For num = 1:
  // s_ptr += 0, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $w_zStrides2, 0b1110
    f16v4hihoamp $w_pout_low, $w_xin, $azero, TAMP_F16V4_E4_P0 }

  // For num = 1:
  // s_ptr += 0, q_ld_ptr += 0
  // For num = 2:
  // s_ptr += 4, q_ld_ptr += 0
  { ld2x64pace $w_xin, $azeros, $w_tripack+=, $w_zStrides2, 0b1110
    f16v4hihoamp $w_pout_high, $w_xin, $azero, TAMP_F16V4_E4_P1 }
    
  // Completely zero strides in w_zStrides2 indicates the num = 1 path
  brz $w_zStrides2, .LNumElemsEq1
  bri .LNumElemsEq2

.size blockSparseDenseMultiply_hh16x16, . - blockSparseDenseMultiply_hh16x16

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half half hh16x16 16

#endif // __IPU_ARCH_VERSION__ < 2
#endif // defined(__IPU__)
