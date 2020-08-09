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

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlock___float_float_8_8

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

DEF_STACK_USAGE w_StackSize blockSparseDenseMultiply_ff8x8
.section ".text.blockSparseDenseMultiply_ff8x8", FUNCTION_IS_WORKER
.type blockSparseDenseMultiply_ff8x8, @function
.align 8
.worker
// worker code

blockSparseDenseMultiply_ff8x8:

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
add                   $w_zStrideInS, $w_zStrideInS, -3

// strides: (zStrideInQ - 3) << 10) | (zStrideInS - 3)
shl                   $w_zStrides1, $w_zStrideInQ, 10
or                    $w_zStrides1, $w_zStrides1, $w_zStrideInS
add                   $w_num, $w_num_orig, -3

// Check strides to use in the AMP loop. The strides to use are dependent on
// the number of elements to avoid excess strided reads
brpos                 $w_num, SetStridesNumElemsGte3


// Code fragment to set strides for num_elems = {0, 1, 2}
// Jumps back to main program after setting strides
add           $cmp_res, $w_num, 1
brz           $cmp_res, LStrideCheckElemsEq2
// Number of elems = 1
// Stride1 = [0][0][in index]
// Stride2 = Stride3 = [0][0][0]
// Need to zero the [out index] bitfield (bits 10-20) inside $stride1, passed
// by supervisor, to avoid accessing elements outside the 'out/partial' vector.
and           $w_zStrides1, $w_zStrides1, 0x3FF
mov           $w_zStrides2, $mzero
mov           $w_zStrides3, $mzero
bri           AfterStrideSet

LStrideCheckElemsEq2:
// Number of elems = 2
// Stride1 = [0][out index][in index]
// Stride2 = [0][0][in index]
// Stride3 = [0][0][1]
and           $w_zStrides2, $w_zStrides1, 0x3FF
setzi         $w_zStrides3, 1
bri           AfterStrideSet

SetStridesNumElemsGte3:

// case of num_elems >= 3
// Stride1 = Stride2 = [0][out index][in index] are the default
mov           $w_zStrides2, $w_zStrides1
setzi         $w_zStrides3, 1

AfterStrideSet:

// we actually need a count subtracted by 3
ld32                  $w_qBaseRetained, $mvertex_base, W_Q_BASE/4
ld32                  $w_sBaseRetained, $mvertex_base, W_S_BASE/4

// w_off is already in multiple of 64, so we can just directly increment
ld64step              $azeros, $mzero, $w_sBaseRetained+=, $w_offsetScaledS
ld64step              $azeros, $mzero, $w_qBaseRetained+=, $w_offsetScaledQ

blockSparseDenseMultiply_ff8x8_retained:

brz                   $w_num_orig, LEndWorker

// offset by X and Y positions for the block processed
// Note:
ld32                  $w_xOffsetInQ, $mvertex_base, W_XOFFSET/4
shl                   $w_xOffsetInQ, $w_xOffsetInQ, 2
add                   $w_qBase, $w_qBaseRetained, $w_xOffsetInQ
ld32                  $w_yOffsetInS, $mvertex_base, W_YOFFSET/4
shl                   $w_yOffsetInS, $w_yOffsetInS, 2
add                   $w_sBase, $w_sBaseRetained, $w_yOffsetInS


// Get compact representation of physical addresses
tapack        $w_tripack, $w_sBase, $w_qBase, $w_qBase

// Assumption that groups in conv is directly stored as actual value -1
//   &input += 0, &partials += 1
ld2x64pace      $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b0011
{
  // &input += 0, &partials += 1
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b0011
  f32sisoamp    $w_pout, $w_null1, $w_pin, TAMP_F32_E4_P0
}
{
  // &input += 0, &partials += 1
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b0011
  f32sisoamp    $w_pout, $w_null1, $w_pin, TAMP_F32_E4_P2
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : [out index])
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b1011
  f32sisoamp    $w_pout, $w_null1, $w_pin, TAMP_F32_E4_P4
}
{
  // &input += 1, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    $w_xin, $w_pin, $w_tripack+=, $w_zStrides3, 0b0100
  f32sisoamp    $w_pout, $w_null1, $w_pin, TAMP_F32_E4_P6
}

// Start providing inputs ----------------------------------------------------
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides3, 0b0111
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P0
}
{
  // &input += 1, &partials += 0
  ld2x64pace    $w_xin, $w_null2, $w_tripack+=, $w_zStrides1, 0b1100
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P1
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides3, 0b0111
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P2
}
{
  // &input += 1, &partials += 0
  ld2x64pace    $w_xin, $w_null2, $w_tripack+=, $w_zStrides1, 0b1100
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P3
}
{
  // &input += 0, &partials += (num_elems >= 3 ? [out index] : 0)
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides2, 0b1011
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P4
}
{
  // &input += (num_elems == 1 ? 0 : [in index]), &partials += 0
  ld2x64pace    $w_xin, $w_null2, $w_tripack+=, $w_zStrides2, 0b1101
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P5
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides3, 0b0111
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P6
}
{
  // &input += (num_elems == 1 ? 0 : 1), &partials += 0
  ld2x64pace    $w_xin, $w_null2, $w_tripack+=, $w_zStrides3, 0b1101
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P7
}
// Start recording output ----------------------------------------------------
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides3, 0b0111
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P0
}
{
  // &input += (num_elems == 1 ? 0 : 1), &output += 1
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides3, 0b0001
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P1
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides3, 0b0111
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P2
}
{
  // &input += (num_elems == 1 ? 0 : 1), &output += 1
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides3, 0b0001
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P3
}

// exit paths for special cases of num_elems = {1, 2}

// w_zStride2 is a proxy for num_elems == 1
brz           $w_zStrides2, LNumElemsEq1
brneg         $w_num, LNumElemsEq2

{
  rpt $w_num, (Loop_end_Amp - Loop_start_Amp)/8-1
  fnop
}
Loop_start_Amp:
  // The reads in the last pass are effectively dummy to avoid code bloat
  {
    // &input += 0, &partials += [out index]
    ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b1011
    f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P4
  }
  {
    // &input += [in index], &output += 1
    ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0001
    f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P5
  }
  {
    // &input += 0, &partials += 1
    ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b0011
    f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P6
  }
  {
    // &input += 1, &output += [out index]
    ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b1000
    f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P7
  }
  {
    // &input += 0, &partials += 1
    ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b0011
    f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P0
  }
  {
    // &input += 1, &output += 1
    ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0000
    f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P1
  }
  {
    // &input += 0, &partials += 1
    ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b0011
    f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P2
  }
  {
    // &input += 1, &output += 1
    ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0000
    f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P3
  }
Loop_end_Amp:

{
  // &input += 0, &partials += 0
  ld2x64pace    $w_null, $w_pin, $w_tripack+=, $w_zStrides1, 0b1111
  f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P4
}
{
  // &input += [in index], &output += 1
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0001
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P5
}
f32sisoamp    $w_pout, $w_xin_0, $w_pin, TAMP_F32_E4_P6
{
  // &input += 1, &output += [out index]
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b1000
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P7
}
// Stop providing partials ---------------------------------------------------
f32sisoamp    $w_pout, $w_xin_0, $w_null, TAMP_F32_E4_P0
{
  // &input += 1, &output += 1
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0000
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P1
}
f32sisoamp    $w_pout, $w_xin_0, $w_null, TAMP_F32_E4_P2
{
  // &input += 1, &output += 1
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0000
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P3
}

LNumElemsEq2:

f32sisoamp    $w_pout, $w_xin_0, $w_null, TAMP_F32_E4_P4
{
  // &input += 0, &output += 1
  ldst64pace    $w_xin, $w_pout, $w_tripack+=, $w_zStrides1, 0b0011
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P5
}
f32sisoamp    $w_pout, $w_xin_0, $w_null, TAMP_F32_E4_P6
{
  // &output += [out index]
  st64pace      $w_pout, $w_tripack+=, $w_zStrides1, 0b10
  f32sisoamp    $w_null, $w_xin_1, $w_null, TAMP_F32_E4_P7
}
// Stop providing input ------------------------------------------------------
f32sisoamp    $w_pout, $w_null1, $w_null, TAMP_F32_E4_P0
{
  // &output += 1
  st64pace      $w_pout, $w_tripack+=, $w_zStrides1, 0b00
  f32sisoamp    $w_pout, $w_null1, $w_null, TAMP_F32_E4_P2
}

// &output += 1
st64pace      $w_pout, $w_tripack+=, $w_zStrides1, 0b00

LNumElemsEq1:

// This may need to change if partials for the next loop could be loaded
// with the store of old results
f32sisoamp    $w_pout, $w_null1, $w_null, TAMP_F32_E4_P4
{
  // &output += 1
  st64pace      $w_pout,          $w_tripack+=, $w_zStrides1, 0b00
  f32sisoamp    $w_pout, $w_null1, $w_null, TAMP_F32_E4_P6
}
// &output += 0
st64pace      $w_pout,          $w_tripack+=, $w_zStrides1, 0b11

LEndWorker:
exitz                 $mzero


.size blockSparseDenseMultiply_ff8x8, . - blockSparseDenseMultiply_ff8x8

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME float ff8x8 8

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
