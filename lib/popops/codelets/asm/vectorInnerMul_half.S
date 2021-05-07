// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// # Overview
//
// This file contains the assembly for BroadcastVectorInner<MULTIPLY> codelets,
// originally named 'ChannelMul', (created for a specific function).
// Because of that, the input vectors were called:
//   'acts_in' (activations), this is the 'data' vertex state field.
//   'scale' (activations are'scaled' by the elements of this vector), this is
//           the'B' vertex state field.
//   'acts_out' Output of the function, this is the 'out' vertex field.
//
// The task is, given two vectors:
//
// Acts: [0 1 2 3 4 5 6 7 8 9 10 11]
// Scale: [0 1 2]
//
// Repeat scale, and multiply acts by it, then set ActsOut to the result:
//
// ActsOut = [0 1 2  3 4 5  6 7 8  9 10 11] .*
//           [0 1 2][0 1 2][0 1 2][0  1  2]
//
//
// The code here handles both inplace (where ActsOut is the same as Acts)
// and non-inplace, where they are different (in different memory banks, so
// ldst64pace can load/store without conflicts).
//
// The core routine has several code paths that can be followed:
// Fast, Medium Speed, and Slow.
//
// ####### Fast Pipelined Paths
// If the scale_len is a multiple of 4 halves (i.e. 8 bytes), we can use
// pipelined code that can process 4 halves per cycle (ldst64 and f16v4mul in
// one cycle).
// We cannot use this if scale len is too big (limitiation in the length
// of the immediate increment field in ldst64pace or if we have less than two
// 'blocks' (rows) to process.
// There are two variants of this path, to be used depending on in-place/not-in-
// place and the B length being also multiple of 8 (see below)
//
// ####### Medium Speed Path
// If scale_len is a multiple of 4 halves, but we cannot use the fast path, as
// described above, we can still use a 'rpt' loop to process 4 halves in
// two cycles.
//
// ####### Slow Path
// Otherwise, e.g. if it is odd we must take the slow path. This is a little
// inconvienient because of misaligned accesses, e.g. if the scale length it
// is 5, then we have:
//
// [0 1 2 3 4  5 6 7 8 9 10 11 12 13 14] .*
// [0 1 2 3 4][0 1 2 3 4][0  1  2  3  4]
//             ^---- Annoying unaligned access.
//
// If it is a multiple of 2 there is probably another "medium speed path" but
// it is not coded to keep the code size small. Also if the scale_len is
// exactly 1 or 2 we could duplicate it and use one of the fast paths
// but this turns out to be rather complex. If the scale_len is 1 we can use
// a different vertex anyway.
//
//
// --------------------------------------------------------------------------
// Description of the 'fast pipelined' path.
//
// We load the first 4 elements of 'scale' ('B0') and run the inner rpt loop
// with that, then the second 4 elements (B1) and run the inner loop with that,
// etc.
//
// In the following description of the inner loop, for ease of notation we use:
//   A0, A1, B0 are '64 bit registers' (i.e. pairs of 'a' ARF registers).
//   'acts_in' (1st operand) begins at 0x40000.
//   'acts_out' (result) begins at 0x50000.
//   'S' is the stride, i.e. the length of the full 2nd operand (scale len).
//   Each line is a single cycle with up to three operations (LOAD/STORE + MUL).
//
// Stride is is the length of 'scale' ('B'), a multiple of 4 halves (8 bytes).
// In this example is 28 halves = 56 (0x38) bytes
//
// All pointer postincrements are by 'S' (stride).
//
//  MOV B0 <- first 4 elements from 'scale'
//
//  0:  LOAD A0  from [40000]+=S;
//  1:                                                          MUL A1  <- A0*B0
//  2:  LOAD A0  from [40038]+=S;
//  3:  LOAD A0' from [40070]+=S;    STORE A1 to [50000]+=S;    MUL A1' <- A0*B0
//  4:  LOAD A0' from [400a8]+=S;    STORE A1 to [40038]+=S;    MUL A1' <- A0*B0
//  5:  LOAD A0' from [400e0]+=S;    STORE A1 to [40070]+=S;    MUL A1' <- A0*B0
//  6:  LOAD A0' from [40118]+=S;    STORE A1 to [400a8]+=S;    MUL A1' <- A0*B0
//  7:  LOAD A0' from [40150]+=S;    STORE A1 to [400e0]+=S;    MUL A1' <- A0*B0
//  8:  LOAD A0' from [40188]+=S;    STORE A1 to [40118]+=S;    MUL A1' <- A0*B0
//  9:  LOAD A0' from [401c0]+=S;    STORE A1 to [40150]+=S;    MUL A1' <- A0*B0
// 10:                               STORE A1 to [40188]+=S;    MUL A1' <- A0*B0
// 11:                               STORE A1 to [401c0]+=S;
//
// This cannot be used for the in-place case, because the ldst64 instruction
// would be loading and storing at the same even-odd multiple of 8-byte (64-bit)
// in the interleaved memory element, causing an exception.
//
//
// --------------------------------------------------------------------------
// Description of the 'fast pipelined delayed' path.
//
// If we can delay by one cycle the store, we can do a 'full pipeline' for the
// in place as well. The delay can be done by using another pair of ARF
// registers ('A2' below):
//
//  0:  LOAD A0 from [40000]+=S;
//  1:  LOAD A1 from [40038]+=S;                               MUL A0 <- A0*B0
//  2:  LOAD A2 from [40070]+=S;                               MUL A1 <- A0*B0
//  3:  LOAD A0 from [400a8]+=S;    STORE A0 to [40000]+=S;    MUL A2 <- A0*B0
//  4:  LOAD A1 from [400e0]+=S;    STORE A1 to [40038]+=S;    MUL A0 <- A0*B0
//  5:  LOAD A2 from [40118]+=S;    STORE A2 to [40070]+=S;    MUL A1 <- A0*B0
//  6:  LOAD A0 from [40150]+=S;    STORE A0 to [400a8]+=S;    MUL A2 <- A0*B0
//  7:  LOAD A1 from [40188]+=S;    STORE A1 to [400e0]+=S;    MUL A0 <- A0*B0
//  8:  LOAD A2 from [401c0]+=S;    STORE A2 to [40118]+=S;    MUL A1 <- A0*B0
//  9:  LOAD A0 from [401f8]+=S;    STORE A0 to [40150]+=S;    MUL A2 <- A0*B0
// 10:                              STORE A1 to [40188]+=S;    MUL A0 <- A0*B0
// 11:                              STORE A2 to [401c0]+=S;
// 11:                              STORE A0 to [401f8]+=S;
//
// In this case the ldst64 will be operating of different even/odd multiple of
// 8 bytes for load and store pointers.
//
// There are two drawbacks with this:
// 1. This works only if 'B' length is a multiple of 4 but not of 8 (like in the
//    example: B length = 28 halves = 56 (0x38) bytes). If the length was 32
//    halves (0x40 bytes) the ldst64 will be operating on the same multiple of
//    8 boundaries, causing an exception.
// 2. The loop kernel needs to be 3 bundles long, which complicates the coding
//    of prologue and epilogue (more overhead in code size and cycles). Cannot
//    be done with 4 bundles, as we don't have enough ARF registers.
//
// because of the above reasons, the 'fast pipelined delayed' is used only for
// the specific case of 'B' length = exactly 4 (see below).
//
// There is a specialized path for 'B' length = exactly 4, which uses the very
// same code described above for the inner loop (both 'fast pipelined' and
// 'fast pipelined delayed'), but runs it only once (no outer loop), and so
// saves a few cycles in this case.


#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// This macro defines a label as global, function
.macro EXPORT_FN label
.globl \label
.type \label, @function
.endm

// This macro associates to the symbol 'label' a size defined as (Current_loc - label)
.macro FN_SIZE label
.size \label, . - \label
.endm


// Let's create a macro to shorten the name for the entry points, which are
// very long, as required by C++ name mangling.
// TYPE needs to be 'Supervisor', 'InPlaceSupervisor', '2D' or '2DInPlace'.
#define CPP_FUNCNAME(TYPE)  __runCodelet_popops__BroadcastVectorInner ## TYPE ## ___popops__expr__BinaryOpType__MULTIPLY_half

// Another similar macro to name the sections where each function is contained
#define FUNC_SECTION(TYPE)  .section .text.VectorInner_ ## TYPE ## _MULTIPLY_half


# In this file we have 4 externally visible functions
EXPORT_FN   CPP_FUNCNAME(Supervisor)
EXPORT_FN   CPP_FUNCNAME(InPlaceSupervisor)
EXPORT_FN   CPP_FUNCNAME(2D)
EXPORT_FN   CPP_FUNCNAME(2DInPlace)

// Scratch area is used to save saome registers
#define SCRATCH_OFFSET_SCALE_ITERATOR 0
#define SCRATCH_OFFSET_ACTS_IN_ITERATOR 1
#define SCRATCH_OFFSET_ACTS_OUT_ITERATOR 2
#define SCRATCH_OFFSET_QUOTIENT 3
#define SCRATCH_OFFSET_LOOP_COUNT 4


// This is the main function that does the actual work. It takes the following
// register arguments:
//
#define scale m0
#define scale_len m1
#define acts_block_count m4
# acts_in an d acts_out must be consecutive registers (to allow dummy load)
#define acts_in_out m2:3
#define acts_in m2
#define acts_out m3

//
// All input registers are clobbered. $m10 ($lr) is used for
// the return address. It also uses the following scratch registers.
// The lifetime of packed_ldst_addrs does not overlap mscratch0
// so it can share the same registers.
//
#define mscratch0 m6
#define scale_loop_count m5          // Only used in scalar path.
#define outer_stride m7              // Only used in scalar path.
#define acts_loop_count m8           // Only used in scalar path.
#define stride m5                    // Only used in multiple-of-4 path.
// 'packed_ldst_addrs' MUST BE 'in_ptr:out_ptr'
#define packed_ldst_addrs m6:7       // Only used in multiple-of-4 path.
#define in_ptr m6                    // Only used in multiple-of-8 path.
#define out_ptr m7                   // Only used in multiple-of-8 path.

#define quotient m11
#define remainder m8
#define loop_count m9
#define mscratch1 m9

#define tmp0 a0:1
#define tmp1 a2:3
#define tmp2 a6:7
#define dummy a8:9
#define tmp0_lower a0
#define current_scale a4:5
#define current_scale_lower a4
#define ascratch0 a7

// -------------------------------------------------------------------------
// Macro for the inner loop for the case where:
//   Not-in-place ('data' and 'out' are in different memory elements)
//   'B' size is any multiple of 4 ('B' size = 4, 8, 12, 16, 20, 24, 28, ...)
//   4 <= 'B' size <= 2044
// The inner RPT loop is the fastest possible (1 cycles/4 half)
//    INPUTS:  $acts_in, $acts_out :
//             $current_scale      :
//             $acts_block_count   : counter for RPT loop
//             $stride             :
// -------------------------------------------------------------------------
.macro FAST_PIPELINED_CORE
  tapack $packed_ldst_addrs, $acts_in, $mzero, $acts_out

  // Cycle 0:      Load 0
  // dummy store without incrementing store pointer. It is safe to do so
  // because that address will be over-written with a valid value
  ldst64pace $tmp0, $tmp1, $packed_ldst_addrs+=, $stride, 0x0D
  // Cycle 1                 Mul 0
  {
    brneg $acts_block_count, .Lstore_final\@
    f16v4mul $tmp1, $current_scale, $tmp0
  }
  // Cycle 2:      Load 1
  // dummy store again
  ldst64pace $tmp0, $tmp1, $packed_ldst_addrs+=, $stride, 0x0D

  rpt $acts_block_count, (2f - 1f)/8-1
1:
  {
    ldst64pace $tmp0, $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp1, $current_scale, $tmp0
  }
2:
  // end of rpt
  {
    st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp1, $current_scale, $tmp0
  }
.Lstore_final\@:
  st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x05
.endm


// -------------------------------------------------------------------------
// Preparation for using FAST_PIPELINED_DELAYED_CORE
// Get quotient and remainder of acts_block_count/3. The fixed point division
// gives the correct results only when $acts_block_count range is [0..32768]
//
//    INPUT:   $acts_block_count : number of blocks to process
//
//    OUTPUT:  $quotient         : blocks / 3
//             $remainder        : blocks % 3
//             $acts_block_count : quotient -1 (i.e. counter for RPT loop)
//
// -------------------------------------------------------------------------
.macro DIVIDE_BLOCKS_BY_3
  mul $quotient, $acts_block_count, 0x5556;
  shr $quotient, $quotient, 16
  mul $mscratch0, $quotient, 3
  sub $remainder, $acts_block_count, $mscratch0
  sub $acts_block_count, $quotient, 1
.endm

// -------------------------------------------------------------------------
// Macro for the inner loop for the case where:
//   In-place (the in/out 'data' is in an interleaved memory element)
//   'B' size is multiple of 4, but not of 8 ('B' size = 4, 12, 20, 28, 36, ...)
//   4 <= 'B' size <= 2044
//
// The inner RPT loop is the fastest possible (1 cycles/4 half)
//    INPUTS:  $acts_in, $acts_out :
//             $current_scale      :
//             $acts_block_count   : counter for RPT loop
//             $stride             :
//             $quotient           :
//             $remainder          :
// -------------------------------------------------------------------------
.macro FAST_PIPELINED_DELAYED_CORE
  brnz   $quotient, 2f

  // -- we have only 1 or 2 atoms to process (can't go to RPT loop)

  # Move two regs in one instr.: ($acts_in, $acts_out) => ($in_ptr, $out_ptr)
  tapack $packed_ldst_addrs, $acts_in, $acts_out, $mzero
  mov    $loop_count, $remainder // need to preserve remainder
1:
  ld64step $tmp0, $mzero, $in_ptr+=, $stride
  {
    add $loop_count, $loop_count, -1
    f16v4mul $tmp0, $tmp0, $current_scale
  }
  st64step $tmp0,  $mzero, $out_ptr+=, $stride
  brnz $loop_count, 1b
  bri .Lend_fast_in_place\@
2:
  // -- We have at least 3 atoms to process: prepare RPT loop
  mov $in_ptr, $acts_in // need to preserve $acts_in

  ld64step $tmp0, $mzero, $in_ptr+=, $stride
  {
    ld64step $tmp1, $mzero, $in_ptr+=, $stride
    f16v4mul $tmp0, $tmp0, $current_scale
  }
  {
    ld64step $tmp2, $mzero, $in_ptr+=, $stride
    f16v4mul $tmp1, $tmp1, $current_scale
  }

  tapack $packed_ldst_addrs, $in_ptr, $mzero, $acts_out

  // Inner RPT loop processing 3 atoms of 4 half values
  rpt $acts_block_count, (2f - 1f)/8-1
1:
  {
    ldst64pace $tmp0, $tmp0, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp2, $tmp2, $current_scale
  }
  {
    ldst64pace $tmp1, $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp0, $tmp0, $current_scale
  }
  {
    ldst64pace $tmp2, $tmp2, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp1, $tmp1, $current_scale
  }
2:
  // Epilogue of pipelined loop ('Drain'). We have 3 different epilogues
  // to process the possible remainder elements (0, 1 or 2)
  brnz $remainder, 1f
  // remainder = 0
  {
    st64pace $tmp0, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mul $tmp2, $tmp2, $current_scale
  }
  st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x01
  st64pace $tmp2, $packed_ldst_addrs+=, $stride, 0x01
  bri .Lend_fast_in_place\@
1:
  // remainder = 1 or 2
  {
    ldst64pace $tmp0, $tmp0, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp2, $tmp2, $current_scale
  }
  cmpeq $mscratch1, $remainder,1
  brz $mscratch1, 2f
  // remainder = 1
  {
    st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mul $tmp0, $tmp0, $current_scale
  }
  st64pace $tmp2, $packed_ldst_addrs+=, $stride, 0x01
  st64pace $tmp0, $packed_ldst_addrs+=, $stride, 0x01

  bri .Lend_fast_in_place\@
2:
  // remainder = 2
  {
    ldst64pace $tmp1, $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp0, $tmp0, $current_scale
  }
  {
    st64pace $tmp2, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mul $tmp1, $tmp1, $current_scale
  }
  st64pace $tmp0, $packed_ldst_addrs+=, $stride, 0x01
  st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x01

.Lend_fast_in_place\@:
.endm


///////////////////////////////////////////////////////////////////////////////
// Entry code that dispatches to the correct code paths, given B length and
// in-place/not-in-place
///////////////////////////////////////////////////////////////////////////////
.section .text.VectorInnerMul_core_half
.align 8
nop
VectorInnerMul_core_half:
  // If we have no blocks to do, exit.
  brz $acts_block_count, .Lreturn

  // Now we are prepared to do the computation, but we have different
  // code paths depending on whether the scale_len is a multiple of 4 halves
  // or not, & other considerations.

  // There is an optimisation for a length of exactly 4
  cmpeq $mscratch0, $scale_len, 4
  brnz $mscratch0, .Lexactly_four

  // If scale len is not a multiple of 4 halves we use the slow path.
  and $mscratch0, $scale_len, 0x03
  brnz $mscratch0, .Lscale_scalar

  // It's a multiple of 4

  // If inplace we jump to the appropriate path
  cmpeq $mscratch0, $acts_in, $acts_out   // Check if in place (i.e. in==out)
  brnz $mscratch0, .Lmultiple_of_four

  // The current multiple of 4 pipeline uses ldst64pace. The
  // stride of that instruction is a signed 10-bit number of 64-bit words. So
  // the maximum stride is 8 * (2^9-1) = 4088 bytes = 2044 halves.
  //
  // So if scale_len is more than 2044 we must use the medium speed path.
  cmpult $mscratch0, $scale_len, 2045
  brz $mscratch0, .Lmultiple_of_four

  // We need to use the medium speed path if there are too few blocks to
  // fill the multiple-of-four pipeline.
  cmpult $mscratch0, $acts_block_count, 2
  brnz $mscratch0, .Lmultiple_of_four

  // Fall through and do the fast path.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                     Multiple of Four Pipelined Fast                       //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
.Lmultiple_of_four_pipeline:
  // Work out the stride, in units of 64-bits. It's scale_len / 4.
  shr $stride, $scale_len, 2
  // Divide the scale_len by 4 ($stride has already done this) and
  // subtract 1 so we can use brnzdec.
  add $scale_len, $stride, -1
  // Work out how many of the main cycles we need to do.
  // We process 1 block per loop.
  add $acts_block_count, $acts_block_count, -2

  // Loop over the 4-element blocks in the scale.
.Lmultiple_of_four_pipeline_scale_loop:
  // Load the next 4 scales.
  ld64step $current_scale, $mzero, $scale+=, 1

  FAST_PIPELINED_CORE

  // Dummy load to increment by 1 atom (4 bytes) 'acts_in' and 'acts_out'.
  ld2x64pace $azeros, $dummy, $acts_in_out+=, $mzero, 0
  // Loop and process the next 4 values of scale, if there are any.
  brnzdec $scale_len, .Lmultiple_of_four_pipeline_scale_loop
  br $lr

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                    Multiple of Four medium speed                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
.align 8
.Lmultiple_of_four:

  // Work out the stride, in units of 64-bits. It's scale_len / 4.
  shr $stride, $scale_len, 2

  // Divide the scale_len by 4 ($stride has already done this) and
  // subtract 1 so we can use brnzdec.
  add $scale_len, $stride, -1


  // Also subtract 1 so we don't process past the end.
  //add $acts_block_count, $acts_block_count, -1

  // Loop over the 4-element blocks in the scale.
.Lmultiple_of_four_scale_loop:

  // Load the next 4 elements of the scale.
  ld64step $current_scale, $mzero, $scale+=, 1

  # Move two registers in one instruction:
  #    ($acts_in, $acts_out) into ($in_ptr, $out_ptr)
  tapack $packed_ldst_addrs, $acts_in, $acts_out, $mzero

  ld64step $tmp0, $mzero, $in_ptr+=, $stride

  // Loop through acts_in. This must be 8-byte aligned which can be done with
  // `.align 8` but that might insert a `nop` and waste a cycle. Instead
  // we do it manually using bundles if necessary.
  rpt $acts_block_count, (2f - 1f)/8-1
1:
  {
    ld64step $tmp0, $mzero, $in_ptr+=, $stride
    f16v4mul $tmp1, $current_scale, $tmp0
  }
  {
    st64step $tmp1, $mzero, $out_ptr+=, $stride
    fnop
  }
2:


  // Move to the next 4 elements of acts_in/out. Cannot use a dummy ld2x64pace
  // as acts_in and acts_out might be the same address, causing an exception.
  add $acts_in, $acts_in, 8
  add $acts_out, $acts_out, 8

  // Loop and process the next 4 values of scale, if there are any.
  brnzdec $scale_len, .Lmultiple_of_four_scale_loop

  br $lr

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                             Exactly Four                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
.align 8
nop
.Lexactly_four:
  // Load the 4 scales.
  ld64step $current_scale, $mzero, $scale+=, 1

  setzi $stride, 1

  cmpeq $mscratch0, $acts_in, $acts_out   // Check if in place (i.e. in==out)
  brnz $mscratch0, .Lexactly_four_in_place
  // ===== NOT-IN-PLACE

  // Work out how many of the main cycles we need to do.
  // We process 1 block per loop.
  add $acts_block_count, $acts_block_count, -2

  FAST_PIPELINED_CORE
  br $lr

.align 8
.Lexactly_four_in_place:
  // =====  IN-PLACE
  st32 $quotient,   $mworker_base, $mzero, SCRATCH_OFFSET_QUOTIENT
  st32 $loop_count, $mworker_base, $mzero, SCRATCH_OFFSET_LOOP_COUNT
  DIVIDE_BLOCKS_BY_3
  FAST_PIPELINED_DELAYED_CORE
  ld32 $quotient,   $mworker_base, $mzero, SCRATCH_OFFSET_QUOTIENT
  ld32 $loop_count, $mworker_base, $mzero, SCRATCH_OFFSET_LOOP_COUNT
  br $lr

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                              Scalar Code                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
.Lscale_scalar:
  // This code can handle any scale_len, and any acts_block_count (other
  // than 0), for cases where the fast path can't be used.
  //
  // Very very slow but simple code. We don't use rpt and we load and store
  // 1 half per loop. You can do better than this, e.g. by treating a
  // len-3 scale as a len-6 (or even len-12) by repeating it. But....
  //
  // This does 1 half per ~10 cycles, vs 4 per cycle for the optimised code.

  // Calculate the stride for the outer loop. This is subtracted from
  // acts_in and acts_out to get it back to where they started, plus one
  // half further.
  mul $outer_stride, $acts_block_count, $scale_len
  add $outer_stride, $outer_stride, -1
  shl $outer_stride, $outer_stride, 1

  // Subtract one so that brnzdec can be used for the loop.
  add $scale_loop_count, $scale_len, -1

// for i = 0..scale_loop_count
.Lscalar_scale_loop:
  // Get the current scale.
  ldb16step $current_scale_lower, $mzero, $scale+=, 1

  // Decrement the loop counter so we can use brnzdec
  add $acts_loop_count, $acts_block_count, -1

// for j = 0..acts_len
.Lscalar_acts_loop:

  // Load the acts value.
  ldb16step $tmp0_lower, $mzero, $acts_in+=, $scale_len

  // Instruction from __st16, but moved here because we can bundle it.
{ and $mscratch0, $acts_out, 0x02

  // Multiply it by the scale.
  f16v2mul $tmp0_lower, $tmp0_lower, $current_scale_lower }

  /////// __st16($acts_out, $tmp0_lower), but using the ARF /////////
  //                                                               //
  // Moved into bundle above.
  //   and $mscratch0, $acts_out, 0x02
  // Jump if $acts_out is 32-bit aligned.
  brz $mscratch0, .Lscalar_aligned_store
.Lscalar_misaligned_store:
  // Get aligned pointer.
  add $mscratch0, $acts_out, -2
  // Load the lower f16.
  ldb16 $ascratch0, $mscratch0, $mzero, 0
  // Combine the two halves.
  sort4x16lo $ascratch0, $ascratch0, $tmp0_lower
  // Store back.
  st32 $ascratch0, $mscratch0, $mzero, 0
  // Done.
  bri .Lscalar_store_end
.Lscalar_aligned_store:
  // Load the upper f16
  ldb16 $ascratch0, $acts_out, $mzero, 1
  // Combine the two halves.
  sort4x16lo $ascratch0, $tmp0_lower, $ascratch0
  // Store back.
  st32 $ascratch0, $acts_out, $mzero, 0
.Lscalar_store_end:
  //                                                               //
  ///////////////////////////////////////////////////////////////////

  // Move the acts_acts_out_ptr forward using a dummy load.
  ldb16step $azero, $mzero, $acts_out+=, $scale_len
  // Loop to the next block.
  brnzdec $acts_loop_count, .Lscalar_acts_loop

  // Move the acts pointers back to the next element.
  sub $acts_in, $acts_in, $outer_stride
  sub $acts_out, $acts_out, $outer_stride
  // Loop to the next element of the scale.
  brnzdec $scale_loop_count, .Lscalar_scale_loop

.Lreturn:
  br $lr

FN_SIZE VectorInnerMul_core_half


#undef mscratch0
#undef stride
#undef scale_loop_count
#undef outer_stride
#undef acts_loop_count
#undef packed_ldst_addrs
#undef tmp0
#undef tmp1
#undef tmp0_lower
#undef current_scale
#undef current_scale_lower
#undef ascratch0


/////////////////// VectorInner Supervisor Vertices ///////////////////////

// Vertex state layout
#define VERTEX_DATA_SCALE_OFFSET 0                 // In 32-bits
#define VERTEX_DATA_SCALE_SIZE_OFFSET 1            // In 32-bits
#define VERTEX_DATA_ACTS_IN_OFFSET 2               // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 6      // In 16-bits
// Additional value for the non-inplace variant
#define VERTEX_DATA_ACTS_OUT_OFFSET 4              // In 32-bits


// The following supervisor variables are used. The vertex base is
// passed in as $m0.
#define supervisor_vertex_base m0
#define worker_entry m1

DEF_STACK_USAGE 0 CPP_FUNCNAME(InPlaceSupervisor)
FUNC_SECTION(InPlaceSupervisor)
.align 4
.supervisor
CPP_FUNCNAME(InPlaceSupervisor):
  setzi $worker_entry, .LvectorInnerMul_inplace_worker
  bri .Lrun_workers


DEF_STACK_USAGE 0 CPP_FUNCNAME(Supervisor)
FUNC_SECTION(Supervisor)
.align 4
.supervisor
CPP_FUNCNAME(Supervisor):
  // Set the entry point for the workers.
  setzi $worker_entry, .LvectorInnerMul_worker

.Lrun_workers:
  // Start all workers. Some may have no work to do and just exit.
  runall $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br $lr

FN_SIZE CPP_FUNCNAME(Supervisor)

#undef supervisor_vertex_base
#undef worker_entry

// Worker code

#define blocks_per_worker m5
#define worker_id m6
#define block_begin m7
#define remaining_blocks m8
#define mscratch0 m9


FUNC_SECTION(WorkerInPlace)
.align 4
.worker
.LvectorInnerMul_inplace_worker:
  ld32 $acts_in, $mvertex_base, $mzero, VERTEX_DATA_ACTS_IN_OFFSET
  mov $acts_out, $acts_in
  bri .Lworker


FUNC_SECTION(Worker):
.align 4
.worker
.LvectorInnerMul_worker:
  ld32 $acts_in,           $mvertex_base, $mzero, VERTEX_DATA_ACTS_IN_OFFSET
  ld32 $acts_out,          $mvertex_base, $mzero, VERTEX_DATA_ACTS_OUT_OFFSET
  // Fall through.

.Lworker:
  // Load vertex state.
  ld32 $scale,             $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  ld32 $scale_len,         $mvertex_base, $mzero, VERTEX_DATA_SCALE_SIZE_OFFSET
  ldz16 $acts_block_count, $mvertex_base, $mzero, VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Get the worker ID.
  get $worker_id, $WSR
  and $worker_id, $worker_id, CSR_W_WSR__CTXTID_M1__MASK

  // Get blocks per worker and remainder.
  shr $blocks_per_worker, $acts_block_count, 3
  and $remaining_blocks, $acts_block_count, 0x7

  // Work out block begin, accounting for remainders (each worker may
  // get one additional block depending on its ID).
  mul $block_begin, $blocks_per_worker, $worker_id
  min $mscratch0, $worker_id, $remaining_blocks
  add $block_begin, $block_begin, $mscratch0

  // Add an extra block to workers with IDs less than the remainder.
  cmpult $mscratch0, $worker_id, $remaining_blocks
  add $acts_block_count, $blocks_per_worker, $mscratch0

  // Skip redistribution if scale is a multiple of two as sub-word writes are
  // not possible
  and $mscratch0, $scale_len, 0x1
  brz $mscratch0, update_acts_ptrs

  // All workers except the last one must do an even number of blocks
  // to avoid subword write issues.

  // If block_begin is odd, round it down and increment acts_block_count.
  and $mscratch0, $block_begin, 1
  add $acts_block_count, $acts_block_count, $mscratch0
  andc $block_begin, $block_begin, 1
  // If we aren't the last worker with blocks, round $acts_block_count down to
  // an even number. The last worker with blocks is 5 if $blocks_per_worker is
  // not 0, or $remaining_blocks otherwise.
  brz $blocks_per_worker, 1f
  setzi $remaining_blocks, 5
1:
  // $mscratch0 is the id of the last worker with blocks.
  cmpeq $mscratch0, $worker_id, $remaining_blocks
  // Don't alter acts_block_count if we are the last worker.
  brnz $mscratch0, 1f
  // Round acts_block_count down to the next even number.
  andc $acts_block_count, $acts_block_count, 1
1:

  // How many elements to advance $acts_in/out
update_acts_ptrs:
  mul $mscratch0, $block_begin, $scale_len
  // Advance $acts_in and $acts_out by 2*$mscratch0 bytes to the
  // $block_begin'th block using a dummy load.
  ldb16step $azero, $mzero, $acts_in+=, $mscratch0
  ldb16step $azero, $mzero, $acts_out+=, $mscratch0

  call $lr, VectorInnerMul_core_half

  exitnz $mzero

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0




///////////////////// VectorInner2D Worker Vertices ////////////////////////

// Vertex state layout for BroadcastVectorInner2D
#define VERTEX2D_DATA_SCALE_OFFSET 0                // In 32-bits
#define VERTEX2D_DATA_WORKLIST_OFFSET 1             // In 32-bits
#define VERTEX2D_DATA_ACTS_IN_OFFSET 2              // In 32-bits
// Additional value for the non-inplace variant
#define VERTEX2D_DATA_ACTS_OUT_OFFSET 3             // In 32-bits


#define scale_iterator m5              // Overwritten by function call
#define acts_in_iterator m7            // Overwritten by function call
#define acts_out_iterator m8           // Overwritten by function call
#define workList_iterator m9
#define nM1 m11



DEF_STACK_USAGE 0 CPP_FUNCNAME(2DInPlace)
FUNC_SECTION(2DInPlace)
.align 4
CPP_FUNCNAME(2DInPlace):
  ld32 $acts_in_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_IN_OFFSET
  mov $acts_out_iterator, $acts_in_iterator
  bri .Lworker2d

FN_SIZE CPP_FUNCNAME(2DInPlace)


DEF_STACK_USAGE 0 CPP_FUNCNAME(2D)
FUNC_SECTION(2D)
.align 4
CPP_FUNCNAME(2D):
  ld32 $acts_in_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_IN_OFFSET
  ld32 $acts_out_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OUT_OFFSET
  // Fall through

.Lworker2d:
  ld32 $workList_iterator,         $mvertex_base, $mzero, VERTEX2D_DATA_WORKLIST_OFFSET
  ldz16step $nM1, $mzero, $workList_iterator+=, 1
  ld32 $scale_iterator,            $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET

.Louter_loop:
  // Advance all the iterators.
  ld32step  $scale,            $mzero, $scale_iterator+=, 1
  ldz16step $scale_len,        $mzero, $workList_iterator+=, 1
  ld32step  $acts_in,          $mzero, $acts_in_iterator+=, 1
  ld32step  $acts_out,         $mzero, $acts_out_iterator+=, 1
  ldz16step $acts_block_count, $mzero, $workList_iterator+=, 1

  // We need to save & restore these as they are clobbered by the function call.
  st32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  st32 $acts_in_iterator,   $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_IN_ITERATOR
  st32 $acts_out_iterator,  $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_OUT_ITERATOR

  call $lr, VectorInnerMul_core_half

  ld32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  ld32 $acts_in_iterator,   $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_IN_ITERATOR
  ld32 $acts_out_iterator,  $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_OUT_ITERATOR

  brnzdec $nM1, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2D)


#undef scale_iterator
#undef workList_iterator
#undef acts_in_iterator
#undef acts_out_iterator
#undef nM1

#endif // __IPU__
