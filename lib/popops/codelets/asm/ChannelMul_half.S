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
// The core routine has three code paths that can be followed:
// Fast, Medium Speed, and Slow.
//
// ####### Fast Path
// If the scale_len is a multiple of 4 halves (i.e. 8 bytes), we can use
// pipelined code that can process 4 halves per cycle (ldst64 and f16v4mul in
// one cycle).
// This cannot be used for inplace opereation as this will cause address
// error in ldst64pace (load/store at the same address).
// Also, we cannot use this if scale len is too big (limitiation in the length
// of the immediate increment field in ldst64pace or if we have less than two
// 'blocks' (rows) to process
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
//
// If it is a multiple of 2 there is probably another "medium speed path" but
// it is not coded to keep the code size small. Also if the scale_len is
// exactly 1 or 2 we could duplicate it and use one of the fast paths
// but this turns out to be rather complex. If the scale_len is 1 we can use
// a different vertex anyway.
//
//
// Description of the 'Multiple of four pipelined' path
// This describes the inner loop only.
//
// In the following, for ease of notation we use:
//   A0, A1, B0 are '64 bit registers' (i.e. pairs of 'a' ARF registers).
//   'acts_in' (1st operand) data begins at 0x40000.
//   'acts_out' (result) begins at 0x50000.
//   'S' is the stride, i.e. the length of the full 2nd operand (scale len).
//   Each line is a single cycle with up to three operations (LOAD/STORE + MUL).
//
// Stride is multiple of 4 halves (8 bytes), e.g. 64 halves = 0x40 bytes
//
// All pointer postincrements are by 'S' (stride), except first two stores.
//
//      MOV B0 <- 4 elements from 'scale'
//
//  0:  LOAD A0  from [40000]+=S;    STORE A1 to [50000]+=0
//  1:                                                          MUL A1  <- A0*B0
//  2:  LOAD A0' from [40040]+=S;    STORE A1 to [50000]+=0;    MUL A1' <- A0*B0
//  3:  LOAD A0' from [40080]+=S;    STORE A1 to [50000]+=S;    MUL A1' <- A0*B0
//  4:  LOAD A0' from [400c0]+=S;    STORE A1 to [50040]+=S;    MUL A1' <- A0*B0
//  5:  LOAD A0' from [40100]+=S;    STORE A1 to [50080]+=S;    MUL A1' <- A0*B0
//  6:  LOAD A0' from [40140]+=S;    STORE A1 to [500c0]+=S;    MUL A1' <- A0*B0
//  7:  LOAD A0' from [40180]+=S;    STORE A1 to [50100]+=S;    MUL A1' <- A0*B0
//  8:  LOAD A0' from [401c0]+=S;    STORE A1 to [50140]+=S;    MUL A1' <- A0*B0
//  9:                               STORE A1 to [50180]+=S;    MUL A1' <- A0*B0
//  9:                               STORE A1 to [501c0]+=S;



#include "poplibs_support/TileConstants.hpp"
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
#define acts_block_count_was_odd m8  // Only used in multiple-of-4 path.
// 'packed_ldst_addrs' MUST BE 'in_ptr:out_ptr'
#define packed_ldst_addrs m6:7       // Only used in multiple-of-4 path.
#define in_ptr m6                    // Only used in multiple-of-8 path.
#define out_ptr m7                   // Only used in multiple-of-8 path.

#define tmp0 a0:1
#define tmp1 a2:3
#define tmp2 a6:7
#define dummy a8:9
#define tmp0_lower a0
#define current_scale a4:5
#define current_scale_lower a4
#define ascratch0 a7


.section .text.VectorInnerMul_core_half
.align 8
VectorInnerMul_core_half:
  // If we have no blocks to do, exit.
  brz $acts_block_count, .Lreturn

  // Now we are prepared to do the computation, but we have different
  // code paths depending on whether the scale_len is a multiple of 4 halves
  // or not, & other considerations.

  // If scale len is not a multiple of 4 halves we use the slow path.
  and $mscratch0, $scale_len, 0x03
  brnz $mscratch0, .Lscale_scalar

  // It's a multiple of 4

  // If inplace we jump to the medium speed path
  cmpeq $mscratch0, $acts_in, $acts_out   // Check if in place (i.e. in==out)
  brnz $mscratch0, .Lmultiple_of_four

  // The current multiple of 4 pipeline uses ldst64pace. The
  // stride of that instruction is a signed 10-bit number of 64-bit words. So
  // the maximum stride is 8 * (2^9-1) = 4088 bytes = 2044 halves.
  //
  // So if scale_len is more than 2044 we must use the medium speed path.
  cmpult $mscratch0, $scale_len, 2045
  brz $mscratch0, .Lmultiple_of_four

  // We also need to use the medium speed path if there are too few blocks to
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
  // Cycle 0:      Load 0
  tapack $packed_ldst_addrs, $acts_in, $mzero, $acts_out

  // dummy store without incrementing store pointer. It is safe to do so
  // because that address will be over-written with a valid value
  ldst64pace $tmp0, $tmp1, $packed_ldst_addrs+=, $stride, 0x0D
  {
    brneg $acts_block_count, store_final
    f16v4mul $tmp1, $current_scale, $tmp0
  }

  // Cycle 1:      Load 1    Add 0
  {
    ldst64pace $tmp0, $tmp1, $packed_ldst_addrs+=, $stride, 0x0D
    f16v4mul $tmp1, $current_scale, $tmp0
  }

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

store_final:
  st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x05


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
#undef acts_block_count_was_odd
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
#define VERTEX2D_DATA_N_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_SCALE_OFFSET 1                // In 32-bits
#define VERTEX2D_DATA_SCALE_LEN_OFFSET 2            // In 32-bits
#define VERTEX2D_DATA_ACTS_IN_OFFSET 3              // In 32-bits
#define VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET 4     // In 32-bits
// Additional value for the non-inplace variant
#define VERTEX2D_DATA_ACTS_OUT_OFFSET 5             // In 32-bits


#define scale_iterator m5              // Overwritten by function call
#define scale_len_iterator m6          // Overwritten by function call
#define acts_in_iterator m7            // Overwritten by function call
#define acts_out_iterator m8           // Overwritten by function call
#define acts_block_count_iterator m9
#define n m11

#define SCRATCH_OFFSET_SCALE_ITERATOR 0
#define SCRATCH_OFFSET_SCALE_LEN_ITERATOR 1
#define SCRATCH_OFFSET_ACTS_IN_ITERATOR 2
#define SCRATCH_OFFSET_ACTS_OUT_ITERATOR 3


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
  ld32 $n,                         $mvertex_base, $mzero, VERTEX2D_DATA_N_OFFSET
  ld32 $scale_iterator,            $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  ld32 $scale_len_iterator,        $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_LEN_OFFSET
  ld32 $acts_block_count_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Subtract one for brnzdec
  add $n, $n, -1

.Louter_loop:
  // Advance all the iterators.
  ld32step  $scale,            $mzero, $scale_iterator+=, 1
  ldz16step $scale_len,        $mzero, $scale_len_iterator+=, 1
  ld32step  $acts_in,          $mzero, $acts_in_iterator+=, 1
  ld32step  $acts_out,         $mzero, $acts_out_iterator+=, 1
  ldz16step $acts_block_count, $mzero, $acts_block_count_iterator+=, 1

  // We need to save & restore these as they are clobbered by the function call.
  st32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  st32 $scale_len_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_LEN_ITERATOR
  st32 $acts_in_iterator,   $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_IN_ITERATOR
  st32 $acts_out_iterator,  $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_OUT_ITERATOR

  call $lr, VectorInnerMul_core_half

  ld32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  ld32 $scale_len_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_LEN_ITERATOR
  ld32 $acts_in_iterator,   $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_IN_ITERATOR
  ld32 $acts_out_iterator,  $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_OUT_ITERATOR

  brnzdec $n, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2D)


#undef scale_iterator
#undef scale_len_iterator
#undef acts_in_iterator
#undef acts_out_iterator
#undef acts_block_count_iterator
#undef n

#endif // __IPU__
