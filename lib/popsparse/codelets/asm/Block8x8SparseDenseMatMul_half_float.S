// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 8x8
//

#ifdef __IPU__
#include "BlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlock___half_float_8_8

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
#define w_zStrides3                        m4
#define cmp_res                            m7

#define w_tripack                          m10:11
#define w_xOffsetInQ                       m10
#define w_yOffsetInS                       m11

#define fp_clr_reg                         a1
#define w_xin_0                            a0
#define w_xin_1                            a1
#define w_xin                              a0:1
#define w_pin                              a2:3
#define w_null2                            a4:5
#define w_pout                             a6:7
#define w_null1                            a14
#define w_null                             azeros

#define w_StackSize                        0

DEF_STACK_USAGE w_StackSize blockSparseDenseMultiply_hf8x8
.section ".text.blockSparseDenseMultiply_hf8x8", FUNCTION_IS_WORKER
.type blockSparseDenseMultiply_hf8x8, @function
.global blockSparseDenseMultiply_hf8x8
.global blockSparseDenseMultiply_hf8x8_retained
.align 8
.worker
// worker code

blockSparseDenseMultiply_hf8x8:

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

// account that we load 4 32-bit elements
add                   $w_zStrideInQ, $w_zStrideInQ, -3
add                   $w_zStrideInS, $w_zStrideInS, -1

// strides: (zStrideInQ - 3) << 10) | (zStrideInS - 1)
shl                   $w_zStrides1, $w_zStrideInQ, 10
or                    $w_zStrides1, $w_zStrides1, $w_zStrideInS
add                   $w_num, $w_num_orig, -3

// Check strides to use in the AMP loop. The strides to use are dependent on
// the number of elements to avoid excess strided reads
brpos                 $w_num, LStridesSet

// This code fragment is called if number of elements are 0, 1, or 2
add                   $cmp_res, $w_num, 1
cmpeq                 $w_zStrides3, $w_num, -1
brnz                  $w_zStrides3, LStridesSet

// w_num = 1
// w_zStrides3 = [0   0   0]
// w_zStrides1 = [0   0   0]
setzi                 $w_zStrides1, 0
LStridesSet:

// we actually need a count subtracted by 3
ld32                  $w_qBaseRetained, $mvertex_base, W_Q_BASE/4
ld32                  $w_sBaseRetained, $mvertex_base, W_S_BASE/4

// w_off is already in multiple of 64, so we can just directly increment
ld64step              $azeros, $mzero, $w_sBaseRetained+=, $w_offsetScaledS
ld64step              $azeros, $mzero, $w_qBaseRetained+=, $w_offsetScaledQ

blockSparseDenseMultiply_hf8x8_retained:

brz                   $w_num_orig, LEndWorker

// offset by X and Y positions for the block processed
// Note:
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
shl                   $w_xOffsetInQ, $w_xOffsetInQ, 2
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
shl                   $w_yOffsetInS, $w_yOffsetInS, 1
add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS

tapack                $w_tripack, $w_sBase, $w_qBase, $w_qBase

ld2x64pace    $azeros, $a2:3, $w_tripack+=, $mzero, 0b0011
{
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $mzero, 0b0011  
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 0
  // partials_ptr1 += 1         -> load into a4:5
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $mzero, 0b0011
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P1
}
brneg         $w_num, NumElemsEq1AndEq2
{
  // input_ptr += 0
  // partials_ptr1 += outstride -> load into a2:3
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $w_zStrides1, 0b1011
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $mzero, 0b0000
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // restore write pointer to correct address after writing original value
  // input_ptr += 0           
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $w_zStrides1, 0b0011
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 0           
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $mzero, 0b0011
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
{
  // input_ptr += instride      -> load into a0:1
  // partials_ptr1 += outstride -> load into a2:3
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $w_zStrides1, 0b1001
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $mzero, 0b0000
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // input_ptr += 0             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $mzero, 0b0011
  f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 0          
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $mzero, 0b0011
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}

{
  rpt $w_num, (Loop_end_Amp-Loop_start_Amp)/8-1
  fnop
}
Loop_start_Amp:
  // The reads in the last pass are effectively dummy to avoid code bloat
  {
    // input_ptr += instride      -> load into a0:1
    // partials_ptr1 += outstride -> load into a2:3
    // partials_ptr2 += 1         -> store from a4:5
    ld2xst64pace  $a0:3, $a4:5, $w_tripack+=, $w_zStrides1, 0b001001
    f16v4sisoamp  $a4:5, $azeros, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // input_ptr += 1             -> load into a0:1
    // partials_ptr1 += 1         -> load into a2:3
    // partials_ptr2 += 1         -> store from a6:7
    ld2xst64pace  $a0:3, $a6:7, $w_tripack+=, $mzero, 0b000000
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // input_ptr += 0             -> load into a0:1
    // partials_ptr1 += 1         -> load into a2:3
    // partials_ptr2 += 1         -> store from a4:5
    ld2xst64pace  $a0:3, $a4:5, $w_tripack+=, $mzero, 0b000011
    f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // input_ptr += 0             -> load into a0:1
    // partials_ptr1 += 1         -> load into a2:3
    // partials_ptr2 += outstride -> store from a4:5
    ld2xst64pace  $a0:3, $a6:7, $w_tripack+=, $w_zStrides1, 0b100011
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }
Loop_end_Amp:

{
  // input_ptr += instride      -> load into a0:1
  // partials_ptr1 += 0         -> load into a2:3
  // partials_ptr2 += 1         -> store from a4:5
  ld2xst64pace  $a0:3, $a4:5, $w_tripack+=, $w_zStrides1, 0b001101
  f16v4sisoamp  $a4:5, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr2 += 1         -> store from a6:7
  ldst64pace    $a0:1, $a6:7, $w_tripack+=, $mzero, 0b0000
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // input_ptr += 0             -> load into a0:1
  // partials_ptr2 += 1         -> store from a4:5
  ldst64pace    $a0:1, $a4:5, $w_tripack+=, $mzero, 0b0011
  f16v4sisoamp  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // partials_ptr2 += outstride -> store from a6:7
  st64pace      $a6:7, $w_tripack+=, $w_zStrides1, 0b10
  f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
LNumElemsEq2:
{
  // partials_ptr2 += 1         -> store from a4:5
  st64pace      $a4:5, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P2
}
{
  // partials_ptr2 += 1         -> store from a6:7
  st64pace      $a6:7, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P3
}
{
  // partials_ptr2 += 1         -> store from a4:5
  st64pace      $a4:5, $w_tripack+=, $mzero, 0b00
  f16v4sisoamp  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P0
}
{
  // partials_ptr2 += outstride -> store from a6:7
  st64pace      $a6:7, $w_tripack+=, $w_zStrides1, 0b10
  f16v4sisoamp  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P1
}

LNumElemsEq1:

// This may need to change if partials for the next loop could be loaded
// with the store of old results
{
  // partials_ptr2 += 1         -> store from a4:5
  st64pace      $a4:5,          $w_tripack+=, $mzero, 0b00
  f16v4sisoamp  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P2
}
{
  // partials_ptr2 += 1 -> store from a6:7
  st64pace      $a6:7,          $w_tripack+=, $mzero, 0b00
  f16v4sisoamp  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P3
}

// partials_ptr2 += 1         -> store from a4:5
st64pace      $a4:5,          $w_tripack+=, $mzero, 0b00

// partials_ptr2 += 1 -> store from a6:7
st64pace      $a6:7,          $w_tripack+=, $mzero, 0b00

LEndWorker:
exitz         $m15

// Handles the case of number of elements <=2
// stride1 at any point contains strides for both input and output. These
// may be modified to avoid overreading partials
NumElemsEq1AndEq2:
{
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $w_zStrides1, 0b1011
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $w_zStrides3, 0b0100
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // restore write pointer to correct address after writing original value
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $w_zStrides3, 0b0111
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $w_zStrides3, 0b0111
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
{
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $w_zStrides1, 0b1101
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $w_zStrides3, 0b1101
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  ld2x64pace    $a0:1, $a2:3, $w_tripack+=, $w_zStrides3, 0b1111
  f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $azeros, $a2:3, $w_tripack+=, $w_zStrides3, 0b1111
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
brz             $w_zStrides1, LNumElemsEq1
bri             LNumElemsEq2


.size blockSparseDenseMultiply_hf8x8, . - blockSparseDenseMultiply_hf8x8

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half float hf8x8 8 0

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
