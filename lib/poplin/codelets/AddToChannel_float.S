#ifdef __IPU__

// # Overview
//
// This file contains the assembly for AddToChannel (and ScaledAddToChannel
// which is very similar). The task is, given two vectors:
//
// Acts: [0 1 2 3 4 5 6 7 8 9 10 11]
// Addend: [0 1 2]
//
// Repead addend, and add it to acts (after scaling it in the case of
// ScaledAddToChannel).
//
// Acts = [0 1 2  3 4 5  6 7 8  9 10 11] +
//        [0 1 2][0 1 2][0 1 2][0  1  2]
//
// The f32 case is a lot simpler than f16 because we have no subword accesses
// and there are fewer paths.
//
// ## Fast Paths
//
// The best we can do is process 2 f32's per cycle using a `rpt` of.
//
//   { ldst64pace; f32v2axpy }
//
// Currently this fast path is not implemented, instead we use

#include "tileimplconsts_tommy.h"
#include "tilearch.h"

.globl __runCodelet_poplin__AddToChannel___float
.type __runCodelet_poplin__AddToChannel___float, @function
.globl __runCodelet_poplin__AddToChannel2D___float
.type __runCodelet_poplin__AddToChannel2D___float, @function
.globl __runCodelet_poplin__ScaledAddToChannel___float
.type __runCodelet_poplin__ScaledAddToChannel___float, @function
.globl __runCodelet_poplin__ScaledAddToChannel2D___float
.type __runCodelet_poplin__ScaledAddToChannel2D___float, @function

.section .text.AddToChannel_float_core
.align 8

// This is the main function that does the actual work. It takes the following
// register arguments:
//
#define addend m0
#define addend_len m1
#define acts m2
#define acts_block_count m3
#define scale a0
//
// $m0 and $m2 are modified, but the others are not. $m10 ($lr) is used for
// the return address. It also uses the following scratch registers:
//
#define outer_stride m4
#define addend_loop_count m5
#define tmp0 a5
#define tmp1 a6
#define current_addend a7

poplin_AddToChannel__float_core:
  // If we have no blocks to do, return.
  brz $acts_block_count, .Lreturn

  // We use `rpt` which has a limited loop count, but this is taken care of
  // in the popconv host code.

  // In future this could be optimised by using dedicated multiple-of-2 and
  // multiple-of-4 pipelines as in the half code. But this code is not as slow
  // as the half scalar code so it isn't such a priority.

.Laddend_scalar:
  // Calculate the stride for the outer loop. This is subtracted from
  // acts to get it back to where they started, plus one
  // float further.
  mul $outer_stride, $acts_block_count, $addend_len
  add $outer_stride, $outer_stride, -1
  shl $outer_stride, $outer_stride, 2

  // Subtract one so that brnzdec can be used for the loop.
  add $addend_loop_count, $addend_len, -1

  // Subtract one so we don't read past the end (which might matter if we
  // are at the very end of memory).
  add $acts_block_count, $acts_block_count, -1

// for i = 0..addend_len
.Lscalar_addend_loop:
  // Get the next addend.
  ld32step $current_addend, $mzero, $addend+=, 1

  // Load the first value.
{ ld32 $tmp0, $mzero, $acts, 0
  // Multiply the addend by the scale.
  f32mul $current_addend, $current_addend, $scale }

  // Loop through acts. This must be 8-byte aligned which can be done with
  // `.align 8` but that might insert a `nop` and waste a cycle. Instead
  // we do it manually using bundles if necessary.
  rpt $acts_block_count, (2f - 1f)/8-1
1:
  {
    ld32 $tmp0, $mzero, $acts, $addend_len
    f32add $tmp1, $tmp0, $current_addend
  }
  {
    st32step $tmp1, $mzero, $acts+=, $addend_len
    fnop
  }
2:

  // Add and store the last value.
  f32add $tmp1, $tmp0, $current_addend
  st32step $tmp1, $mzero, $acts+=, $addend_len

  // Move the acts pointer back to the next element.
  sub $acts, $acts, $outer_stride

  // If addend_len != 0 decrement it and loop.
  brnzdec $addend_loop_count, .Lscalar_addend_loop

.Lreturn:
  br $lr

// Undefine scratch registers. Arguments are left defined for the functions
// below.
#undef outer_stride
#undef addend_loop_count
#undef tmp0
#undef tmp1
#undef current_addend

.size poplin_AddToChannel__float_core, . - poplin_AddToChannel__float_core


/////////////////// AddToChannel Supervisor Vertices ///////////////////////

// Vertex state layout for AddToChannel
#define VERTEX_DATA_ADDEND_OFFSET 0            // In 32-bits
#define VERTEX_DATA_ADDEND_END_OFFSET 1        // In 32-bits
#define VERTEX_DATA_ACTS_OFFSET 2              // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 6  // In 16-bits
// Additional state for ScaledAddToChannel
#define VERTEX_DATA_SCALE_OFFSET 4             // In 32-bits

.section .text.AddToChannel_float
.align 4

// The following supervisor variables are used. The vertex base is
// passed in as $m0.
#define supervisor_vertex_base m0
#define worker_entry m1

__runCodelet_poplin__ScaledAddToChannel___float:
  // Set the entry point for the workers.
  setzi        $worker_entry, .Lscaled_addtochannel_worker
  // Run the workers.
  bri .Lrun_workers

.size __runCodelet_poplin__ScaledAddToChannel___float, . - __runCodelet_poplin__ScaledAddToChannel___float

__runCodelet_poplin__AddToChannel___float:
  // Set the entry point for the workers.
  setzi        $worker_entry, .Laddtochannel_worker
  // Fall through.

.Lrun_workers:
  // Start all workers. Some may have no work to do and just exit.
  runall       $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync         TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br           $lr

#undef supervisor_vertex_base
#undef worker_entry

// Worker code.

#define blocks_per_worker m4
#define worker_id m5
#define block_begin m6
#define remaining_blocks m7
#define mscratch0 m8

.Lscaled_addtochannel_worker:
  // Load the scale.
  ld32 $scale, $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker

.Laddtochannel_worker:
  // Set the scale to 1.0.
  f32exp $scale, $azero
  // Fall through.

.Lworker:
  // Load vertex state.
  ld32 $addend,            $mvertex_base, $mzero, VERTEX_DATA_ADDEND_OFFSET
  ld32 $mscratch0,         $mvertex_base, $mzero, VERTEX_DATA_ADDEND_END_OFFSET
  ld32 $acts,              $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  ldz16 $acts_block_count, $mvertex_base, $mzero, VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Calculate the number of elements in addend.
  // Optimisation: If there were a ONE_PTR_AND_COUNT layout I could avoid
  // these two instructions.
  sub $addend_len, $mscratch0, $addend
  shr $addend_len, $addend_len, 2

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
  mul $mscratch0, $block_begin, $addend_len
  // Advance $acts by 4*$mscratch0 bytes to the $block_begin'th block using
  // a dummy load.
  ld32step $azero, $mzero, $acts+=, $mscratch0

  call $lr, poplin_AddToChannel__float_core

  exitnz $mzero

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0

.size __runCodelet_poplin__AddToChannel___float, . - __runCodelet_poplin__AddToChannel___float

///////////////////// AddToChannel2D Worker Vertices ////////////////////////

// Vertex state layout for AddToChannel2D
#define VERTEX2D_DATA_N_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_ADDEND_OFFSET 1               // In 32-bits
#define VERTEX2D_DATA_ADDEND_LEN_OFFSET 2           // In 32-bits
#define VERTEX2D_DATA_ACTS_OFFSET 3                 // In 32-bits
#define VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET 4     // In 32-bits
// Additional state for ScaledAddToChannel2D
#define VERTEX2D_DATA_SCALE_OFFSET 5                // In 32-bits

.section .text.AddToChannel2D_float
.align 4

__runCodelet_poplin__ScaledAddToChannel2D___float:
  // Load the scale.
  ld32 $scale, $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker2d

.size __runCodelet_poplin__ScaledAddToChannel2D___float, . - __runCodelet_poplin__ScaledAddToChannel2D___float

__runCodelet_poplin__AddToChannel2D___float:
  // Set the scale to 1.0.
  f32exp $scale, $azero
  // Fall through.

#define addend_iterator m6
#define addend_len_iterator m7
#define acts_iterator m8
#define acts_block_count_iterator m9
#define n m11

.Lworker2d:

  ld32 $n,                         $mvertex_base, $mzero, VERTEX2D_DATA_N_OFFSET
  ld32 $addend_iterator,           $mvertex_base, $mzero, VERTEX2D_DATA_ADDEND_OFFSET
  ld32 $addend_len_iterator,       $mvertex_base, $mzero, VERTEX2D_DATA_ADDEND_LEN_OFFSET
  ld32 $acts_iterator,             $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  ld32 $acts_block_count_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Subtract one for brnzdec
  add $n, $n, -1

.Louter_loop:
  // Advance all the iterators.
  ld32step  $addend,           $mzero, $addend_iterator+=, 1
  ldz16step $addend_len,       $mzero, $addend_len_iterator+=, 1
  ld32step  $acts,             $mzero, $acts_iterator+=, 1
  ldz16step $acts_block_count, $mzero, $acts_block_count_iterator+=, 1

  call $lr, poplin_AddToChannel__float_core

  brnzdec $n, .Louter_loop

  exitnz $mzero

#undef addend_iterator
#undef addend_len_iterator
#undef acts_iterator
#undef acts_block_count_iterator
#undef n

.size __runCodelet_poplin__AddToChannel2D___float, . - __runCodelet_poplin__AddToChannel2D___float

#endif // __IPU__
