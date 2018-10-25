#ifdef __IPU__

// # Overview
//
// This file contains the assembly for ChannelMul. The task is, given two
// vectors:
//
// Acts: [0 1 2 3 4 5 6 7 8 9 10 11]
// Scale: [0 1 2]
//
// Repeat scale, and multiply acts by it, then set ActsOut to the result:
//
// ActsOut = [0 1 2  3 4 5  6 7 8  9 10 11] .*
//           [0 1 2][0 1 2][0 1 2][0  1  2]
//
// This is a fair bit simpler than the ChannelMul half code because it
// isn't in-place, so we can use memory constraints to have the entirety of
// the input and output in different memory elements, and we don't have
// to worry about conflicts at all. This means we don't need a separate
// code path for multiple-of-4 and multiple-of-8.
//
// ## Fast Path
//
// If the scale_len is a multiple of 4, we can use pipelined code that can
// process 4 halfs per cycle.
//
// If it is a multiple of 2 there is probably a "medium speed path" but I have
// not coded it to keep the code size small. Also if the scale_len is
// exactly 1 or 2 we could duplicate it and use one of the fast paths
// but this turns out to be rather complex. If the scale_len is 1 we can use
// a different vertex anyway.
//
// Otherwise, e.g. if it is odd we must take the slow path. This is a little
// inconvienient because of misaligned accesses, e.g. if the scale length it
// is 5, then we have:
//
// [0 1 2 3 4][0 1 2 3 4][0  1  2  3  4] +
// [0 1 2 3 4  5 6 7 8 9 10 11 12 13 14]
//             ^---- Annoying unaligned access.
//
// In that case we use st16, which is very slow.
//
// ## scale Length is a Multiple of 4
//
// We can use a 3-stage pipline as we don't have to worry about memory
// conflicts.
//
//  0:         Load 0
//  1:         Load 1      Mul 0
//  2:         Load 2      Mul 1      Store 0
//  3:         Load 3      Mul 2      Store 1
//  4:         Load 4      Mul 3      Store 2
//  5:         Load 5      Mul 4      Store 3
//  6:        (Load 6)     Mul 5      Store 4
//  7:        (Load 7)    (Mul 6)     Store 5
//

#include "tileimplconsts.h"
#include "tilearch.h"

.globl __runCodelet_poplin__ChannelMul___half
.type __runCodelet_poplin__ChannelMul___half, @function
.globl __runCodelet_poplin__ChannelMul2D___half
.type __runCodelet_poplin__ChannelMul2D___half, @function

.section .text.poplin__ChannelMul__half_core
.align 8

// This is the main function that does the actual work. It takes the following
// register arguments:
//
#define scale m0
#define scale_len m1
#define acts_in m2
#define acts_out m3
#define acts_block_count m4
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
#define packed_ldst_addrs m6:7       // Only used in multiple-of-4 path.

#define tmp0 a0:1
#define tmp1 a2:3
#define tmp0_lower a0
#define current_scale a4:5
#define current_scale_lower a4
#define ascratch0 a7

poplin__ChannelMul__half_core:
  // If we have no blocks to do, exit.
  brz $acts_block_count, .Lreturn

  // Now we are prepared to do the computation, but we have different
  // code paths depending on whether the scale_len is a multiple of 4 or not.

  // The current multiple of 4 pipelines uses ldst64pace. The
  // stride of that instruction is a signed 10-bit number of 64-bit words. So
  // the maximum stride is 8 * (2^9-1) = 4088 bytes = 2044 halfs.
  //
  // So if scale_len is more than 2044 we must use the scalar path.
  cmpult $mscratch0, $scale_len, 2045
  brz $mscratch0, .Lscale_scalar

  // Also we need to use the slow path if there are too few blocks to fill the
  // multiple-of-four pipeline. We could use a fast non-pipelined path but
  // that is yet more code.
  cmpult $mscratch0, $acts_block_count, 2
  brnz $mscratch0, .Lscale_scalar

  // Check if the scale len is a multiple of 4.
  and $mscratch0, $scale_len, 0x03
  brz $mscratch0, .Lmultiple_of_four_pipeline

  // If the length is less exactly 2 or 1, we could still do it by duplicating
  // the scale and using either the multiple-of-4 or multiple-of-8 code.
  // But this adds a fair bit of complexity and is never the case for resnet50
  // so I removed that code.

  // Fall through and do it slowly.


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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                           Multiple of Four                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

.Lmultiple_of_four_pipeline:

  // Work out the stride, in units of 64-bits. It's scale_len / 4.
  shr $stride, $scale_len, 2

  // Divide the scale_len by 4 ($stride has already done this) and
  // subtract 1 so we can use brnzdec.
  add $scale_len, $stride, -1

  // Note if $acts_block_count is odd.
  and $acts_block_count_was_odd, $acts_block_count, 0x01
  // Work out how many of the main cycles we need to do.
  // We process 2 blocks per loop.
  shr $acts_block_count, $acts_block_count, 1
  // Also subtract 1 so we don't process past the end.
  // The minimum number of blocks required for this pipeline is 2.
  add $acts_block_count, $acts_block_count, -1

  // Loop over the 4-element blocks in the scale.
.Lmultiple_of_four_pipeline_scale_loop:

  // Load the next 4 scales.
  ld64step $current_scale, $mzero, $scale+=, 1

  // Copy acts_in as the load address for tapack later (to avoid modifying
  // acts_in).
  mov $mscratch0, $acts_in

  // Cycle 0:      Load 0
  ld64step $tmp1, $mzero, $mscratch0+=, $stride

  // Cycle 1:      Load 1    Add 0
  {
    ld64step $tmp0, $mzero, $mscratch0+=, $stride
    f16v4mul $tmp1, $current_scale, $tmp1
  }

  // First address is the load pointer. Second is ignored. Third is the store
  // pointer.
  tapack $packed_ldst_addrs, $mscratch0, $mzero, $acts_out

  // Loop through acts. This must be 8-byte aligned which can be done with
  // `.align 8` but that might insert a `nop` and waste a cycle. Instead
  // we do it manually using bundles if necessary.
  {
    rpt $acts_block_count, (2f - 1f)/8-1
    fnop
  }
1:
  {
    ldst64pace $tmp1, $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp0, $current_scale, $tmp0
  }
  {
    ldst64pace $tmp0, $tmp0, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp1, $current_scale, $tmp1
  }
2:

  // We may need either 2 or 3 more stores depending on the oddness of
  // $acts_block_count (before we divided it by 2).

  brz $acts_block_count_was_odd, 1f
  // odd case
  {
    ldst64pace $tmp0, $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp1, $current_scale, $tmp0
  }
1:
  // even case
  {
    st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mul $tmp0, $current_scale, $tmp0
  }
  st64pace $tmp0, $packed_ldst_addrs+=, $stride, 0x05


2:

  // Move to the next 4 elements of acts.
  ld2x64pace $azeros, $a8:9, $acts_in:3+=, $mzero, 0


  // Loop and process the next 4 values of scale, if there are any.
  brnzdec $scale_len, .Lmultiple_of_four_pipeline_scale_loop

  br $lr

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

.size poplin__ChannelMul__half_core, . - poplin__ChannelMul__half_core

/////////////////// ChannelMul Supervisor Vertices ///////////////////////

// Vertex state layout
#define VERTEX_DATA_SCALE_OFFSET 0                 // In 32-bits
#define VERTEX_DATA_SCALE_SIZE_OFFSET 1            // In 32-bits
#define VERTEX_DATA_ACTS_IN_OFFSET 2               // In 32-bits
#define VERTEX_DATA_ACTS_OUT_OFFSET 3              // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 8      // In 16-bits

.section .text.__runCodelet_poplin__ChannelMul___half
.align 4

// The following supervisor variables are used. The vertex base is
// passed in as $m0.
#define supervisor_vertex_base m0
#define worker_entry m1

__runCodelet_poplin__ChannelMul___half:
  // Set the entry point for the workers.
  setzi $worker_entry, .LChannelMul_worker
  // Start all workers. Some may have no work to do and just exit.
  runall $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br $lr

#undef supervisor_vertex_base
#undef worker_entry

// Worker code

#define blocks_per_worker m5
#define worker_id m6
#define block_begin m7
#define remaining_blocks m8
#define mscratch0 m9

.LChannelMul_worker:

  // Load vertex state.
  ld32 $scale,             $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  ld32 $scale_len,         $mvertex_base, $mzero, VERTEX_DATA_SCALE_SIZE_OFFSET
  ld32 $acts_in,           $mvertex_base, $mzero, VERTEX_DATA_ACTS_IN_OFFSET
  ld32 $acts_out,          $mvertex_base, $mzero, VERTEX_DATA_ACTS_OUT_OFFSET
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

  // How many elements to advance $acts.
  mul $mscratch0, $block_begin, $scale_len
  // Advance $acts_in and $acts_out by 2*$mscratch0 bytes to the
  // $block_begin'th block using a dummy load.
  ldb16step $azero, $mzero, $acts_in+=, $mscratch0
  ldb16step $azero, $mzero, $acts_out+=, $mscratch0

  call $lr, poplin__ChannelMul__half_core

  exitnz $mzero

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0

.size __runCodelet_poplin__ChannelMul___half, . - __runCodelet_poplin__ChannelMul___half




///////////////////// ChannelMul2D Worker Vertices ////////////////////////

// Vertex state layout for ChannelMul2D
#define VERTEX2D_DATA_N_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_SCALE_OFFSET 1                // In 32-bits
#define VERTEX2D_DATA_SCALE_LEN_OFFSET 2            // In 32-bits
#define VERTEX2D_DATA_ACTS_IN_OFFSET 3              // In 32-bits
#define VERTEX2D_DATA_ACTS_OUT_OFFSET 4             // In 32-bits
#define VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET 5     // In 32-bits

.section .text.__runCodelet_poplin__ChannelMul2D___half
.align 4


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

__runCodelet_poplin__ChannelMul2D___half:

  ld32 $n,                         $mvertex_base, $mzero, VERTEX2D_DATA_N_OFFSET
  ld32 $scale_iterator,            $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  ld32 $scale_len_iterator,        $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_LEN_OFFSET
  ld32 $acts_in_iterator,          $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_IN_OFFSET
  ld32 $acts_out_iterator,         $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OUT_OFFSET
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

  call $lr, poplin__ChannelMul__half_core

  ld32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  ld32 $scale_len_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_LEN_ITERATOR
  ld32 $acts_in_iterator,   $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_IN_ITERATOR
  ld32 $acts_out_iterator,  $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_OUT_ITERATOR

  brnzdec $n, .Louter_loop

  exitnz $mzero

#undef scale_iterator
#undef scale_len_iterator
#undef acts_in_iterator
#undef acts_out_iterator
#undef acts_block_count_iterator
#undef n

.size __runCodelet_poplin__ChannelMul2D___half, . - __runCodelet_poplin__ChannelMul2D___half

#endif // __IPU__
