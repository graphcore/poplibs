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

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlock___float_float_4_4

// =============================================================================


// Zero output/partials
//
// Performance: 12 + num_samples / 2

DEF_STACK_USAGE 0 zeroDenseOutput64
.section ".text.zeroDenseOutput64", FUNCTION_IS_WORKER
.type zeroDenseOutput64, @function
.globl zeroDenseOutput64
.worker
.align 8

#define wkr_id_zv                       m0
#define zero_info_zv                    m1
#define zero_info_div_12_zv             m2
// Registers above must be retained between calls
#define outchan_ptr_zv                  m3

zeroDenseOutput64:
get           $wkr_id_zv, $WSR
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
ldz16         $zero_info_zv, $mvertex_base, SUP_VBASE_ZERO_INFO/2

// we could get zero information as this vertex could be called multiple times
// but zero infor field must be zero only in the first call
brz           $zero_info_zv, Loop_end_zero_64

// For n with 0 <= n <= 65533 this does a division by 6 with the remainder
// split amongst workers.
add           $zero_info_div_12_zv, $zero_info_zv, 6
sub           $zero_info_div_12_zv, $zero_info_div_12_zv, $wkr_id_zv
mul           $zero_info_div_12_zv, $zero_info_div_12_zv, 21845
shr           $zero_info_div_12_zv, $zero_info_div_12_zv, 17

ld32          $outchan_ptr_zv, $mvertex_base, SUP_VBASE_Q_BASE/4
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

.size zeroDenseOutput64, . - zeroDenseOutput64


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
#define w_xin                              a0:1
#define w_xin_0                            a0
#define w_xin_1                            a1
#define w_null                             a4:5
#define w_pout                             a6:7
#define fp_clr_reg                         a1


#define w_StackSize                        0

DEF_STACK_USAGE w_StackSize blockSparseDenseMultiply_ff4x4
.section ".text.blockSparseDenseMultiply_ff4x4", FUNCTION_IS_WORKER
.type blockSparseDenseMultiply_ff4x4, @function
.global blockSparseDenseMultiply_ff4x4
.global blockSparseDenseMultiply_ff4x4_retained
.align 8
.worker
// worker code

blockSparseDenseMultiply_ff4x4:

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
// 2. Stride between consecutive batched as part of number of batches allocated
ld32                  $w_zStrideInQ, $mvertex_base, W_ZSTRIDEINQ/4
ld32                  $w_zStrideInS, $mvertex_base, W_ZSTRIDEINS/4

// To offset Q and S pointers allocated to this workers
mul                   $w_offsetScaledQ, $w_zStrideInQ, $w_off
mul                   $w_offsetScaledS, $w_zStrideInS, $w_off

// The stride pattern is +=2, +=(stride - 2) as we do load 2 elements at a time.
// But as we use a 64 bit ld/st we must divide that by 2.
add                   $w_zStrideInQ, $w_zStrideInQ, -1
add                   $w_zStrideInS, $w_zStrideInS, -1

// strides: (zStrideInQ - 2) << 10) | (zStrideInS - 2)
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
// w_zStrides1 = [ 0  |  (zStrideInQ - 2) | (zStrideInS - 2)]
// w_zStrides2 = [ 0  |  (zStrideInQ | zStrideInS]
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

blockSparseDenseMultiply_ff4x4_retained:

// offset by X and Y positions for the block processed
// Note:
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
shl                   $w_xOffsetInQ, $w_xOffsetInQ, 2
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ

blockSparseDenseMultiply_ff4x4_retained_yOff:
brz                   $w_num_orig, LEndWorker

ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
shl                   $w_yOffsetInS, $w_yOffsetInS, 2
add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS


{
  tapack                $w_tripack, $w_sBase, $w_qBase, $w_qBase
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  // s_ptr+=0, q_ld_ptr += 2 
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $mzero, 0b0001
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += 0, q_ld_ptr += q_stride - 2
  //
  // For num > 2 : s_ptr += 0, q_ld_ptr += q_stride - 2
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_zStrides1, 0b1011
  f32sisoamp            $w_pout, $azero, $w_pin, TAMP_F32_E2_P0
}
{
  // For num = 1 : s_ptr += 2, q_ld_ptr += 2
  //
  // For num = 2 : s_ptr += 2, q_ld_ptr += 2
  //
  // For num > 2 : s_ptr += 2, q_ld_ptr += 2
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $azeros, $azero, $w_pin, TAMP_F32_E2_P2
}

// partials are loaded into the two engines used and now we can start 
// feeding inputs to the AMP
{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += 0, q_ld_ptr += q_stride - 2
  //
  // feed input 0 to AMP
  ld2x64pace            $azeros, $w_pin, $w_tripack+=, $w_zStrides2, 0b1011
  f32sisoamp            $w_pout, $w_xin_0, $w_pin, TAMP_F32_E2_P0
}

{
  // For num = 1 : s_ptr += 0, q_ld_ptr += 0
  //
  // For num = 2 : s_ptr += s_stride - 2, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += s_stride - 2, q_ld_ptr += 0
  //
  // feed input 1 to AMP
  ld2x64pace            $w_xin, $azeros, $w_tripack+=, $w_zStrides1, 0b1101
  f32sisoamp            $w_pout, $w_xin_1, $azeros, TAMP_F32_E2_P1
}
{
  // feed input 2 to AMP
  brz                   $w_zStrides1, FinalStores
  f32sisoamp            $w_pout, $w_xin_0, $w_pin, TAMP_F32_E2_P2
}
{
  // For num = 1 : s_ptr += 2, q_ld_ptr += 2
  //
  // For num = 2 : s_ptr += 2, q_ld_ptr += 2
  //
  // For num > 2 : s_ptr += 2, q_ld_ptr += 2
  //  
  // feed input 3 to AMP
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
  f32sisoamp            $w_pout, $w_xin_1, $azeros, TAMP_F32_E2_P3
}
{
  brneg                 $w_num, NumEq2Stores
  // result available
  f32sisoamp            $w_pout, $w_xin_0, $w_pin, TAMP_F32_E2_P0
}
{
  rpt $w_num, 3
  fnop
}
  {
    // For num > 3 : s_ptr += s_stride - 2, q_ld_ptr += q_stride - 2
    //  
    // s_ptr += s_ld_ptr, q_ld_ptr += 0
    ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_zStrides2, 0b1001
    f32sisoamp            $azeros, $w_xin_1, $azeros, TAMP_F32_E2_P1
  }
  {
    // q_st_ptr += 2
    st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
    f32sisoamp            $w_pout, $w_xin_0, $w_pin, TAMP_F32_E2_P2
  }
  {
    // For num > 3 : s_ptr += 2, q_ld_ptr += 2
    ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $mzero, 0b0000
    f32sisoamp            $azeros, $w_xin_1, $azeros, TAMP_F32_E2_P3
  }
  {
    // q_st_ptr += q_stride - 2
    st64pace              $w_pout, $w_tripack+=, $w_zStrides2, 0b10
    f32sisoamp            $w_pout, $w_xin_0, $w_pin, TAMP_F32_E2_P0
  }
{
  // For num > 2 : s_ptr += s_stride - 2, q_ld_ptr += 0
  //  
  // s_ptr += s_ld_ptr, q_ld_ptr += 0
  ld2x64pace            $w_xin, $w_pin, $w_tripack+=, $w_zStrides2, 0b1101
  f32sisoamp            $azeros, $w_xin_1, $azeros, TAMP_F32_E2_P1
}
{
  // q_st_ptr += 2
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout, $w_xin_0, $w_pin, TAMP_F32_E2_P2
}
{
  // For num > 2 : s_ptr += 2, q_ld_ptr += 0
  ld2x64pace            $w_xin, $w_null, $w_tripack+=, $mzero, 0b1100
  f32sisoamp            $azeros, $w_xin_1, $azeros, TAMP_F32_E2_P3
}
{
  // q_st_ptr += q_stride - 2
  st64pace              $w_pout, $w_tripack+=, $w_zStrides2, 0b10
  f32sisoamp            $w_pout, $w_xin_0, $azeros, TAMP_F32_E2_P0
}
NumEq2Stores:
{
  // For num = 2 : s_ptr += 2, q_ld_ptr += 0
  //
  // For num > 2 : s_ptr += 2, q_ld_ptr += 0
  //  
  // s_ptr += s_ld_ptr, q_ld_ptr += 0
  ld2x64pace            $w_xin, $azeros, $w_tripack+=, $w_zStrides2, 0b1100
  f32sisoamp            $w_null, $w_xin_1, $azeros, TAMP_F32_E2_P1
}
{
  // q_st_ptr += 2
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout, $w_xin_0, $azeros, TAMP_F32_E2_P2
}

FinalStores:
// For num = 1, this is a redundant write as the stride for the write is 0
{
  st64pace              $w_pout, $w_tripack+=, $w_zStrides1, 0b10
  f32sisoamp            $azeros, $w_xin_1, $azeros, TAMP_F32_E2_P3
}

// q_st_ptr += q_stride - 2
f32sisoamp            $w_pout, $azero, $azeros, TAMP_F32_E2_P0

{
  // q_st_ptr += 2
  st64pace              $w_pout, $w_tripack+=, $mzero, 0b00
  f32sisoamp            $w_pout, $azero, $azeros, TAMP_F32_E2_P2
}
// q_st_ptr += 0
st64pace              $w_pout, $w_tripack+=, $mzero, 0b11

LEndWorker:
exitz                 $mzero


.size blockSparseDenseMultiply_ff4x4, . - blockSparseDenseMultiply_ff4x4

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME float float ff4x4 4 0

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
