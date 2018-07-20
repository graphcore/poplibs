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
// If the addend len is odd then we resort to the slow path, which is a lot
// quicker than for f16s - we can even use `rpt` in that case and process one
// float every 3 cycles.
//
// In fact, by pipelining even the slow path could do one float per two cycles
// which is pretty good, but I haven't done that yet to keep things simple.
//
// The multiple-of-2 and multiple-of-4 fast paths could be implemented
// as they are for the f16 version.

#include "tileimplconsts_tommy.h"
#include "tilearch.h"

// Vertex state layout
#define VERTEX_DATA_ADDEND_OFFSET 0            // In 32-bits
#define VERTEX_DATA_ADDEND_END_OFFSET 1        // In 32-bits
#define VERTEX_DATA_ACTS_OFFSET 2              // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 6  // In 16-bits
// For ScaledAddToChannel
#define VERTEX_DATA_SCALE_OFFSET 4             // In 32-bits

/////////// Supervisor Variables ///////////////

// The vertex base is passed in as the first parameter
#define supervisor_vertex_base m0
#define worker_entry m1

/////////// Worker Variables //////////////////

// Integer variables
#define addend m0
#define addend_len m1
#define acts m2
#define acts_block_count m3

#define acts_ptr m6

#define blocks_per_worker m5
#define worker_id m6
#define block_begin m7
#define remaining_blocks m8

#define outer_loop_count m9

#define mscratch0 m10

// Float variables
#define tmp0 a0
#define scale a3
#define current_addend0 a4

// Mangled function names

// popconv::AddToChannel<half>
#define addtochannel_f32 __runCodelet_popconv__AddToChannel___float
// popconv::ScaledAddToChannel<half>
#define scaled_addtochannel_f32 __runCodelet_popconv__ScaledAddToChannel___float

.globl addtochannel_f32
.globl scaled_addtochannel_f32
.type addtochannel_f32, @function
.type scaled_addtochannel_f32, @function

.section .text.AddToChannel_float
.align 8

addtochannel_f32:
  // Set the entry point for the workers.
  setzi        $worker_entry, .Laddtochannel_worker
  // Run the workers.
  bri .Lrun_workers

scaled_addtochannel_f32:
  // Set the entry point for the workers.
  setzi        $worker_entry, .Lscaled_addtochannel_worker
  // Fall through.

.Lrun_workers:
  // Start all workers. Some may have no work to do and just exit.
  runall       $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync         TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br           $lr

.Lscaled_addtochannel_worker:
  // Load the scale into the lower half of $TAS, and 1.0 into the upper half.
  ld32 $scale, $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker

.Laddtochannel_worker:
  // Load 1.0 into $TAS
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

  // If we have no blocks to do, exit.
  brz $acts_block_count, .Lend

  // How many elements to advance $acts.
  mul $mscratch0, $block_begin, $addend_len
  // Advance $acts by 4*$mscratch0 bytes to the $block_begin'th block using
  // a dummy load.
  ld32step $azero, $mzero, $acts+=, $mscratch0

  // Now we are prepared to do the computation, but we have different
  // code paths depending on whether the addend_len is a multiple of 4 or 2
  // or neither.

  // However, for now we'll just do the slow path which is not as slow as
  // f16's slow path.

  // Otherwise fall through and do it slowly.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                              Scalar Code                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
.Laddend_scalar:
  // Subtract one so that brnzdec can be used for the loop.
  add $outer_loop_count, $addend_len, -1

// for i = 0..addend_len
.Lscalar_addend_loop:
  // Get the next addend.
  ld32step $current_addend0, $mzero, $addend+=, 1

  // Store the acts pointer
{ mov $acts_ptr, $acts
  // Multiply the addend by the scale.
  f32mul $current_addend0, $current_addend0, $scale }

  // Move the acts pointer to the next element for the next loop.
  add $acts, $acts, 4

  // Loop through acts. This must be 8-byte aligned which can be done with
  // `.align 8` but that might insert a `nop` and waste a cycle. Instead
  // we do it manually using bundles if necessary.
  {
    rpt $acts_block_count, (2f - 1f)/8-1
    fnop
  }
1:
  {
    ld32 $tmp0, $mzero, $acts_ptr, 0
    fnop
  }
  {
    nop
    f32add $tmp0, $tmp0, $current_addend0
  }
  {
    st32step $tmp0, $mzero, $acts_ptr+=, $addend_len
    fnop
  }
2:

  // If addend_len != 0 decrement it and loop.
  brnzdec $outer_loop_count, .Lscalar_addend_loop

.Lend:
  exitnz $mzero

// Set the symbol sizes. It doesn't *really* make sense, but I think this is
// the least confusing.
.size addtochannel_f32, . - addtochannel_f32
.size scaled_addtochannel_f32, . - addtochannel_f32

#endif // __IPU__
