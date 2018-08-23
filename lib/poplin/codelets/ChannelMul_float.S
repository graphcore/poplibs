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
// The f32 case is a lot simpler than f16 because we have no subword accesses
// and there are fewer paths.
//
// ## Fast Paths
//
// The best we can do is process 2 f32's per cycle using a `rpt` of.
//
//   { ldst64pace; f32v2mul }
//
// Currently this fast path is not implemented.

#include "tileimplconsts_tommy.h"
#include "tilearch.h"

.globl __runCodelet_poplin__ChannelMul___float
.type __runCodelet_poplin__ChannelMul___float, @function
.globl __runCodelet_poplin__ChannelMul2D___float
.type __runCodelet_poplin__ChannelMul2D___float, @function

.section .text.poplin__ChannelMul__float_core
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
// $m0, $m2 and $m3 are modified, but the others are not. $m10 ($lr) is used for
// the return address. It also uses the following scratch registers:
//
#define outer_stride m5
#define scale_loop_count m6
#define tmp0 a5
#define tmp1 a6
#define current_scale a7

poplin__ChannelMul__float_core:
  // If we have no blocks to do, return.
  brz $acts_block_count, .Lreturn

  // We use `rpt` which has a limited loop count, but this is taken care of
  // in the poplin host code.

  // In future this could be optimised by using a dedicated multiple-of-2
  // pipeline as in the half code. But this code is not as slow
  // as the half scalar code so it isn't such a priority.

.Lscalar:
  // Calculate the stride for the outer loop. This is subtracted from
  // acts_in and acts_out to get it back to where they started, plus one
  // float further.
  mul $outer_stride, $acts_block_count, $scale_len
  add $outer_stride, $outer_stride, -1
  shl $outer_stride, $outer_stride, 2

  // Subtract one so that brnzdec can be used for the loop.
  add $scale_loop_count, $scale_len, -1

  // Subtract one so we don't read past the end (which might matter if we
  // are at the very end of memory).
  add $acts_block_count, $acts_block_count, -1

// for i = 0..scale_len
.Lscalar_scale_loop:
  // Get the next scale.
  ld32step $current_scale, $mzero, $scale+=, 1

  // Load the first value.
  ld32step $tmp0, $mzero, $acts_in+=, $scale_len

  // Loop through acts. This must be 8-byte aligned which can be done with
  // `.align 8` but that might insert a `nop` and waste a cycle. Instead
  // we do it manually using bundles if necessary.
  {
    rpt $acts_block_count, (2f - 1f)/8-1
    fnop
  }
1:
  {
    ld32step $tmp0, $mzero, $acts_in+=, $scale_len
    f32mul $tmp1, $tmp0, $current_scale
  }
  {
    st32step $tmp1, $mzero, $acts_out+=, $scale_len
    fnop
  }
2:

  // Multiply and store the last value.
  f32mul $tmp1, $tmp0, $current_scale
  st32step $tmp1, $mzero, $acts_out+=, $scale_len

  // Move the acts pointers back to the next element.
  sub $acts_in, $acts_in, $outer_stride
  sub $acts_out, $acts_out, $outer_stride

  // If scale_len != 0 decrement it and loop.
  brnzdec $scale_loop_count, .Lscalar_scale_loop

.Lreturn:
  br $lr

// Undefine scratch registers. Arguments are left defined for the functions
// below.
#undef outer_stride
#undef scale_loop_count
#undef tmp0
#undef tmp1
#undef current_scale

.size poplin__ChannelMul__float_core, . - poplin__ChannelMul__float_core


/////////////////// ChannelMul Supervisor Vertices ///////////////////////

// Vertex state layout for ChannelMul
#define VERTEX_DATA_SCALE_OFFSET 0                 // In 32-bits
#define VERTEX_DATA_SCALE_END_OFFSET 1             // In 32-bits
#define VERTEX_DATA_ACTS_IN_OFFSET 2               // In 32-bits
#define VERTEX_DATA_ACTS_OUT_OFFSET 3              // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 8      // In 16-bits

.section .text.__runCodelet_poplin__ChannelMul___float
.align 4

// The following supervisor variables are used. The vertex base is
// passed in as $m0.
#define supervisor_vertex_base m0
#define worker_entry m1

__runCodelet_poplin__ChannelMul___float:
  // Set the entry point for the workers.
  setzi        $worker_entry, .LChannelMul_worker
  // Start all workers. Some may have no work to do and just exit.
  runall       $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync         TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br           $lr

#undef supervisor_vertex_base
#undef worker_entry

// Worker code.

#define blocks_per_worker m5
#define worker_id m6
#define block_begin m7
#define remaining_blocks m8
#define mscratch0 m9

.LChannelMul_worker:

  // Load vertex state.
  ld32 $scale,             $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  ld32 $mscratch0,         $mvertex_base, $mzero, VERTEX_DATA_SCALE_END_OFFSET
  ld32 $acts_in,           $mvertex_base, $mzero, VERTEX_DATA_ACTS_IN_OFFSET
  ld32 $acts_out,          $mvertex_base, $mzero, VERTEX_DATA_ACTS_OUT_OFFSET
  ldz16 $acts_block_count, $mvertex_base, $mzero, VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Calculate the number of elements in scale.
  // Optimisation: This could be changed from TWO_PTR to ONE_PTR and a
  // separate count - and then it could probably be a uint16_t too. This can
  // also be done for AddToChannel.
  sub $scale_len, $mscratch0, $scale
  shr $scale_len, $scale_len, 2

  // Get the worker ID.
  get $worker_id, $WSR
  and $worker_id, $worker_id, CSR_W_WSR__CTXTID_M1__MASK

  // Get blocks per worker and remainder.
  shr $blocks_per_worker, $acts_block_count, 3
  and $remaining_blocks, $acts_block_count, 0x7

  // Work out block begin, accounting for remainders.
  mul $block_begin, $blocks_per_worker, $worker_id
  min $mscratch0, $worker_id, $remaining_blocks
  add $block_begin, $block_begin, $mscratch0

  // Add remainder to workers with IDs less than the remainder.
  cmpult $mscratch0, $worker_id, $remaining_blocks
  add $acts_block_count, $blocks_per_worker, $mscratch0

  // How many elements to advance $acts.
  mul $mscratch0, $block_begin, $scale_len
  // Advance $acts_in and $acts_out by 4*$mscratch0 bytes to the
  // $block_begin'th block using a dummy load.
  ld32step $azero, $mzero, $acts_in+=, $mscratch0
  ld32step $azero, $mzero, $acts_out+=, $mscratch0

  call $lr, poplin__ChannelMul__float_core

  exitnz $mzero

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0

.size __runCodelet_poplin__ChannelMul___float, . - __runCodelet_poplin__ChannelMul___float

///////////////////// ChannelMul2D Worker Vertices ////////////////////////

// Vertex state layout for ChannelMul2D
#define VERTEX2D_DATA_N_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_SCALE_OFFSET 1                // In 32-bits
#define VERTEX2D_DATA_SCALE_LEN_OFFSET 2            // In 32-bits
#define VERTEX2D_DATA_ACTS_IN_OFFSET 3              // In 32-bits
#define VERTEX2D_DATA_ACTS_OUT_OFFSET 4             // In 32-bits
#define VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET 5     // In 32-bits

.section .text.__runCodelet_poplin__ChannelMul2D___float
.align 4

#define scale_iterator m5       // Overwritten by function call
#define scale_len_iterator m6   // Overwritten by function call
#define acts_in_iterator m7
#define acts_out_iterator m8
#define acts_block_count_iterator m9
#define n m11

#define SCRATCH_OFFSET_SCALE_ITERATOR 0
#define SCRATCH_OFFSET_SCALE_LEN_ITERATOR 1

__runCodelet_poplin__ChannelMul2D___float:

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

  call $lr, poplin__ChannelMul__float_core

  ld32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  ld32 $scale_len_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_LEN_ITERATOR

  brnzdec $n, .Louter_loop

  exitnz $mzero

#undef scale_iterator
#undef scale_len_iterator
#undef acts_in_iterator
#undef acts_out_iterator
#undef acts_block_count_iterator
#undef n

.size __runCodelet_poplin__ChannelMul2D___float, . - __runCodelet_poplin__ChannelMul2D___float

#endif // __IPU__
