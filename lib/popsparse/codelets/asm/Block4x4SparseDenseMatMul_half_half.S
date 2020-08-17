// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 4x4
//

#ifdef __IPU__
#include "BlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlock___half_half_4_4

// =============================================================================

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////

// =============================================================================

// worker registers
#define w_qBaseRetained                    m0
#define w_sBaseRetained                    m1
#define w_id                               m2
#define w_num                              m3
#define w_off                              m4
#define w_zStrides1                        m5
#define w_zStrideInQ                       m5
#define w_zStrides2                        m6
#define w_zStrideInS                       m6
#define w_workList                         m7
#define w_num_orig                         m8
#define w_offsetScaledQ                    m2
#define w_offsetScaledS                    m9
#define w_sBase                            m2
#define w_qBase                            m9

#define w_tripack                          m10:11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11

#define w_pin                              a2:3
#define w_pin_0                            a2
#define w_pin_1                            a3
#define w_xin                              a0:1
#define w_xin1                             a4:5
#define w_pout                             a6:7
#define w_pout_0                           a6
#define w_pout_1                           a7
#define fp_clr_reg                         a1


#define w_StackSize                        0

DEF_STACK_USAGE w_StackSize blockSparseDenseMultiply_hh4x4
.section ".text.blockSparseDenseMultiply_hf4x4", FUNCTION_IS_WORKER
.type blockSparseDenseMultiply_hh4x4, @function
.global blockSparseDenseMultiply_hh4x4
.global blockSparseDenseMultiply_hh4x4_retained
.align 8
.worker
// worker code

blockSparseDenseMultiply_hh4x4:

get                   $w_id, $WSR
and                   $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

// Two short entries per worker: multiply by 4 to get byte offset
shl                   $w_id, $w_id, 2
// load amount of work to do for the worker and 
ld32                  $w_workList, $mvertex_base, W_WORKLIST/4
ldz16                 $w_off, $w_id, $w_workList, 0
ldz16                 $w_num_orig, $w_id, $w_workList, 1

// We need the Z strides in Q and S for two purposes.
// 1. Offset at the correct batch allocated to this worker
// 2. Stride between consecutive batches as part of number of batches allocated
ld32                  $w_zStrideInQ, $mvertex_base, W_ZSTRIDEINQ/4
ld32                  $w_zStrideInS, $mvertex_base, W_ZSTRIDEINS/4

// To offset Q and S pointers allocated to this workers
mul                   $w_offsetScaledQ, $w_zStrideInQ, $w_off
mul                   $w_offsetScaledS, $w_zStrideInS, $w_off

// strides: (zStrideInQ << 10) | (zStrideInS)
shl                   $w_zStrides1, $w_zStrideInQ, 10
or                    $w_zStrides1, $w_zStrides1, $w_zStrideInS

// Keep a second copy as we need different strides due to the pipeline which
// requires us to not read locations with strides.
mov                   $w_zStrides2, $w_zStrides1
add                   $w_num, $w_num_orig, -2
brpos                 $w_num, LSetPosStrides
// For num = 1:
// w_zStrides1 = [ 0  |  0            | 0]
// w_zStrides2 = [ 0  |  0            | 0]
mov                   $w_zStrides1, $mzero 
mov                   $w_zStrides2, $mzero 
bri                   LSetStridesComplete

// For num > 2:
// w_zStrides1 = [ 0  |  (zStrideInQ) | (zStrideInS)]
// w_zStrides2 = [ 0  |  (zStrideInQ| zStrideInS]
LSetPosStrides:
brnz                  $w_num, LSetStridesComplete
// For num = 2:
// w_zStrides1 = [ 0  |  w_zStrideInQ | w_zStrideInS]
// w_zStrides2 = [ 0  |  0            | 0]
mov                   $w_zStrides2, $mzero 

LSetStridesComplete:

// we actually need a count subtracted by 3
add                   $w_num, $w_num, -1
ld32                  $w_qBaseRetained, $mvertex_base, W_Q_BASE/4
ld32                  $w_sBaseRetained, $mvertex_base, W_S_BASE/4

// w_off is already in multiple of 64, so we can just directly increment
ld64step              $azeros, $mzero, $w_sBaseRetained+=, $w_offsetScaledS
ld64step              $azeros, $mzero, $w_qBaseRetained+=, $w_offsetScaledQ

blockSparseDenseMultiply_hh4x4_retained:
brz                   $w_num_orig, LEndWorker

// offset by X and Y positions for the block processed
// Note:
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
shl                   $w_xOffsetInQ, $w_xOffsetInQ, 1
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
shl                   $w_yOffsetInS, $w_yOffsetInS, 1
add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS

{
  tapack                $w_tripack, $w_sBase, $w_qBase, $w_qBase
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += 0, q_ld_ptr += q_stride
  //
  // For num > 2 : s_ptr += 0, q_ld_ptr += q_stride
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_zStrides1, 0b1011
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += s_stride, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += s_stride, q_ld_ptr += 0
  ld2x64pace            $w_xin, $azeros,  $w_tripack+=, $w_zStrides1, 0b1101
  f16v4hihoamp          $w_pout_0, $azeros, $w_pin_0, TAMP_F16V4_E2_P0
}
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += 0, q_ld_ptr += q_stride
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_zStrides2, 0b1011
  f16v4hihoamp          $w_pout_0, $azeros, $w_pin_1, TAMP_F16V4_E2_P1
}
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += s_stride, q_ld_ptr += 0  
  ld2x64pace            $w_xin, $azeros, $w_tripack+=, $w_zStrides2, 0b1101
  f16v4hihoamp          $w_pout_0, $w_xin, $w_pin_0, TAMP_F16V4_E2_P0
}
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += s_stride, q_ld_ptr += q_stride  
  ld2x64pace            $w_xin1, $w_pin, $w_tripack+=, $w_zStrides2, 0b1001
  f16v4hihoamp          $w_pout_0, $azeros, $w_pin_1, TAMP_F16V4_E2_P1
}

{
  brz                   $w_zStrides1, StoreNumEq1
  f16v4hihoamp          $w_pout_0, $w_xin, $w_pin_0, TAMP_F16V4_E2_P0
}

brneg               $w_num, StoreNumEq2
rpt $w_num, 1
  {
    // For num > 3 : s_ptr += 0, q_ld_ptr += q_stride  
    ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_zStrides2, 0b1011
    f16v4hihoamp          $w_pout_1, $azeros, $w_pin_1, TAMP_F16V4_E2_P1 
  }
  {
    // For num > 3 : s_ptr += s_stride, q_st_ptr += q_stride  
    ldst64pace            $w_xin1, $w_pout, $w_tripack+=, $w_zStrides2, 0b1001
    f16v4hihoamp          $w_pout_0, $w_xin1, $w_pin_0, TAMP_F16V4_E2_P0
  }

f16v4hihoamp          $w_pout_1, $azeros, $w_pin_1, TAMP_F16V4_E2_P1

{
  st64pace              $w_pout, $w_tripack+=, $w_zStrides1, 0b10
  f16v4hihoamp          $w_pout_0, $w_xin1, $azero, TAMP_F16V4_E2_P0
}

StoreNumEq2:
f16v4hihoamp          $w_pout_1, $azeros, $azero, TAMP_F16V4_E2_P1

{
  st64pace              $w_pout, $w_tripack+=, $w_zStrides1, 0b1010
  f16v4hihoamp          $w_pout_0, $azeros, $azero, TAMP_F16V4_E2_P0
}

StoreNumEq1:
f16v4hihoamp          $w_pout_1, $azeros, $azero, TAMP_F16V4_E2_P1
st64pace              $w_pout, $w_tripack+=, $w_zStrides1, 0b1010

LEndWorker:
exitz                 $mzero


.size blockSparseDenseMultiply_hh4x4, . - blockSparseDenseMultiply_hh4x4

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half half hh4x4 4 0

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
