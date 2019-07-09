#ifdef __IPU__

// # Overview
//
// This file contains the assembly for BroadcastVectorInner<ADD> and
// BroadcastVectorInner<SCALED_ADD> codelets, originally named 'AddToChannel'
// and 'ScaledAddToChannel' (created for a specific function).
// Because of that, the input vectors were called:
//   'acts' (activations), this is the 'data' vertex state field.
//   'addend' (added to activations), this is the 'B' vertex state field.
//
// The task is, given two vectors:
//
// Acts: [0 1 2 3 4 5 6 7 8 9 10 11]
// Addend: [0 1 2]
//
// Repeat addend, and add it to acts (after scaling it in the case of
// Scaled_Add).
//
// Acts = [0 1 2  3 4 5  6 7 8  9 10 11] +
//        [0 1 2][0 1 2][0 1 2][0  1  2]
//
// ## Fast Paths
//
// If the addend_len is a multiple of 4, we can use pipelined code that can
// process 4 halfs per cycle.
//
// If it is a multiple of 2 there is probably a "medium speed path" but is
// not coded to keep the code size small. Also if the addend_len is
// exactly 1 or 2 we could duplicate the addend and use one of the fast paths
// but this turns out to be rather complex. If the addend_len is 1 we can use
// a different vertex anyway.
//
// Otherwise, e.g. if it is odd we must take the slow path. This is a little
// inconvienient because of misaligned accesses, e.g. if the addend length it
// is 5, then we have:
//
// [0 1 2 3 4][0 1 2 3 4][0  1  2  3  4] +
// [0 1 2 3 4  5 6 7 8 9 10 11 12 13 14]
//             ^---- Annoying unaligned access.
//
// In that case we use st16, which is very slow.
//
// ## Addend Length is a Multiple of 4, but not of 8
//
// We can use a 4-stage pipline to avoid memory conflicts by delaying the
// store one cycle.
//
//  0:         Load 0
//  1:         Load 1      Add 0
//  2:         Load 2      Add 1
//  3:         Load 3      Add 2      Store 0
//  4:         Load 4      Add 3      Store 1
//  5:         Load 5      Add 4      Store 2
//  6:        (Load 6)     Add 5      Store 3
//  7:        (Load 7)    (Add 6)     Store 4
//  8:        (Load 8)    (Add 7)     Store 5
//
// To delay the store we can use f16v4mix, which stores and loads values
// from the temporary accumulator ($AACC[0, 2]). This also gives us the scale
// for free by setting the $TAS register.
//
// So we use a scheme like this (shown for cycles 3 and 4).
//
//  rpt {
//    {
//       acts[0] = tmp1, tmp1 = acts[3]
//       tmp0 = $AACC, $AACC = tmp0 + addend
//    }
//    {
//       acts[1] = tmp0, tmp0 = acts[4]
//       tmp1 = $AACC, $AACC = tmp1 + addend
//    }
//  }
//
// ## Addend Length is a Multiple of 8
//
// When addend_len is a multiple of 4, but not a multiple of 8, we could
// process data like this.
//
// addend[0:3]   += acts[0:3]
// addend[12:15] += acts[0:3]
// addend[24:27] += acts[0:3]
// addend[36:39] += acts[0:3]
// addend[4:7]   += acts[4:7]
// addend[16:19] += acts[4:7]
// addend[28:31] += acts[4:7]
// addend[40:43] += acts[4:7]
// addend[8:11]  += acts[8:11]
// addend[20:23] += acts[8:11]
// addend[32:35] += acts[8:11]
// addend[44:47] += acts[8:11]
//
// However this would result in memory conflicts for multiples of 8. Instead
// we need to instead process it like this:
//
// addend[0:3]   += acts[0:3]
// addend[4:7]   += acts[4:7]
// addend[16:19] += acts[0:3]
// addend[20:23] += acts[4:7]
// addend[32:35] += acts[0:3]
// addend[36:39] += acts[4:7]
// addend[8:11]  += acts[8:11]
// addend[12:15] += acts[12:15]
// addend[24:27] += acts[8:11]
// addend[28:31] += acts[12:15]
// addend[40:43] += acts[8:11]
// addend[44:47] += acts[12:15]
//
// In some ways this is simpler than the multiple-of-four case since each loop
// only processes one block so we don't need to check if acts_block_count is
// odd at the end of the loop, and there is no minimum number of blocks
// required to use the pipeline. Also this processes more data in one loop
// so the overhead tends to be slightly lower.

#include "poplibs_support/TileConstants.hpp"

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
// OPERATION needs to be ADD or SCALED___ADD (with three underscores, because of
// C++ name mangling rules)
#define CPP_FUNCNAME(TYPE, OPERATION) __runCodelet_popops__BroadcastVectorInner ## TYPE ## ___popops__expr__BroadcastOpType__ ## OPERATION ## _half

// Another similar macro to name the sections where each function is contained
#define FUNC_SECTION(TYPE, OPERATION) .section .text.VectorInner_ ## TYPE ## _ ## OPERATION ## _half


# In this file we have 8 externally visible functions
EXPORT_FN   CPP_FUNCNAME(Supervisor,ADD)
EXPORT_FN   CPP_FUNCNAME(Supervisor,SCALED___ADD)
EXPORT_FN   CPP_FUNCNAME(InPlaceSupervisor,ADD)
EXPORT_FN   CPP_FUNCNAME(InPlaceSupervisor,SCALED___ADD)
EXPORT_FN   CPP_FUNCNAME(2D,ADD)
EXPORT_FN   CPP_FUNCNAME(2D,SCALED___ADD)
EXPORT_FN   CPP_FUNCNAME(2DInPlace,ADD)
EXPORT_FN   CPP_FUNCNAME(2DInPlace,SCALED___ADD)


// This is the main function that does the actual work. It takes the following
// register arguments:
//
#define addend m0
#define addend_len m1
#define acts m2
#define acts_block_count m3
#define out m11
//
// Also, the scale must have been loaded into $TAS.
//
// All input registers are clobbered. $m10 ($lr) is used for
// the return address. It also uses the following scratch registers.
// The lifetime of packed_ldst_addrs does not overlap mscratch0
// so it can share the same registers. Similarly for acts_block_count_was_odd
// and acts_rpt_count, which are used in different pipelines.
//
#define mscratch0 m4
#define outer_stride m8
#define packed_ldst_addrs m4:5
#define stride m6
#define addend_loop_count m6
#define acts_block_count_was_odd m7
#define acts_rpt_count m7

#define tmp0 a0:1
#define tmp1 a2:3
#define tmp0_lower a0
#define current_addend0 a4:5
#define current_addend0_lower a4
#define current_addend0_upper a5
#define current_addend1 a6:7
#define scale a6
#define ascratch0 a7

.section .text.VectorInnerAdd_core_half
.align 8
VectorInnerAdd_core_half:
  // If we have no blocks to do, exit.
  brz $acts_block_count, .Lreturn

  // Now we are prepared to do the computation, but we have different
  // code paths depending on whether the addend_len is a multiple of 8,
  // or 4, or otherwise.

  // The current multiple of 4 and 8 pipelines use ldst64pace. The
  // stride of that instruction is a signed 10-bit number of 64-bit words. So
  // the maximum stride is 8 * (2^9-1) = 4088 bytes = 2044 halfs.
  //
  // The stride is equal to the channel size for the multiple-of-4 pipeline,
  // and 8 bytes less than that for the multiple-of-8 pipeline. Therefore the
  // maximum channel size is 2048 halfs (because that is a multiple of 8,
  // so the extra 4 halfs are fine).
  //
  // So if addend_len is more than 2048 we must use the scalar path.
  cmpult $mscratch0, $addend_len, 2049
  brz $mscratch0, .Laddend_scalar

  // Check if the addend len is a multiple of 8. This is the most common case.
  // When I checked resnet50, all the addends are either 8 or 16 elements.
  {
    and $mscratch0, $addend_len, 0x07
    setzi $a0, CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT
  }
  {
    brz $mscratch0, .Lmultiple_of_eight_pipeline
    uput $FP_CLR, $a0
  }

  // Also we need to use the slow path if there are too few blocks to fill the
  // multiple-of-four pipeline. We could use a fast non-pipelined path but
  // that is yet more code.
  cmpult $mscratch0, $acts_block_count, 2
  brnz $mscratch0, .Laddend_scalar

  // Check if the addend len is a multiple of 4. This is just as fast as 8 but
  // we have to use a different method.
  and $mscratch0, $addend_len, 0x03
  brz $mscratch0, .Lmultiple_of_four_pipeline

  // If the length is less exactly 2 or 1, we could still do it by duplicating
  // the addend and using either the multiple-of-4 or multiple-of-8 code.
  // But this adds a fair bit of complexity and is never the case for resnet50
  // so I removed that code.

  // Fall through and do it slowly.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                              Scalar Code                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
.Laddend_scalar:
  // This code can handle any addend_len, and any acts_block_count (other
  // than 0), for cases where the fast path can't be used.
  //
  // Very very slow but simple code. We don't use rpt and we load and store
  // 1 half per loop. You can do better than this, e.g. by treating a
  // len-3 addend as a len-6 (or even len-12) by repeating it. But....
  //
  // This does 1 half per ~10 cycles, vs 4 per cycle for the optimised code.

  // Calculate the stride for the outer loop. This is subtracted from
  // acts and out to get it back to where they started, plus one
  // half further.
  mul $outer_stride, $acts_block_count, $addend_len
  add $outer_stride, $outer_stride, -1
  shl $outer_stride, $outer_stride, 1

  // Get the scale from $TAS
  get $scale, $TAS

  // Subtract one so that brnzdec can be used for the loop.
  add $addend_loop_count, $addend_len, -1

// for i = 0..addend_loop_count
.Lscalar_addend_loop:
  // Get the current addend.
  ldb16step $current_addend0_lower, $mzero, $addend+=, 1

  // Decrement the loop counter so we can use brnzdec
{ add $acts_rpt_count, $acts_block_count, -1
  // Multiply the addend by the scale.
  f16v2mul $current_addend0_lower, $current_addend0_lower, $scale }

// for j = 0..acts_len
.Lscalar_acts_loop:

  // Load the acts value.
  ldb16step $tmp0_lower, $mzero, $acts+=, $addend_len

  // Instruction from __st16, but moved here because we can bundle it.
{ and $mscratch0, $out, 0x02

  // Add the (scaled) addend.
  f16v2add $tmp0_lower, $tmp0_lower, $current_addend0_lower }

  /////// __st16($out, $tmp0_lower), but using the ARF /////////
  //                                                               //
  // Moved into bundle above.
  //   and $mscratch0, $out, 0x02
  // Jump if $out is 32-bit aligned.
  brz $mscratch0, .Lscalar_aligned_store
.Lscalar_misaligned_store:
  // Get aligned pointer.
  add $mscratch0, $out, -2
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
  ldb16 $ascratch0, $out, $mzero, 1
  // Combine the two halves.
  sort4x16lo $ascratch0, $tmp0_lower, $ascratch0
  // Store back.
  st32 $ascratch0, $out, $mzero, 0
.Lscalar_store_end:
  //                                                               //
  ///////////////////////////////////////////////////////////////////

  // Move the out_ptr forward with a dummy load.
  ldb16step $azero, $mzero, $out+=, $addend_len

  // Loop to the next block.
  brnzdec $acts_rpt_count, .Lscalar_acts_loop

  // Move the acts and out pointers back, for the next addend.
  sub $acts, $acts, $outer_stride
  sub $out, $out, $outer_stride
  // Loop to the next element of the addend.
  brnzdec $addend_loop_count, .Lscalar_addend_loop

.Lreturn:
  br $lr

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                           Multiple of Four                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
nop  # to align following 'rpt'

.Lmultiple_of_four_pipeline:

  // Work out the stride, in units of 64-bits. It's addend_len / 4.
  shr $stride, $addend_len, 2

  // Divide the addend_len by 4 ($stride has already done this) and
  // subtract 1 so we can use brnzdec.
  add $addend_len, $stride, -1

  // Note if $acts_block_count is odd.
  and $acts_block_count_was_odd, $acts_block_count, 0x01
  // Work out how many of the main cycles we need to do.
  // We process 2 blocks per loop.
  shr $acts_block_count, $acts_block_count, 1
  // Also subtract 1 so we don't process past the end.
  // The minimum number of blocks required for this pipeline is 2.
  add $acts_block_count, $acts_block_count, -1

  // Loop over the 4-element blocks in the addend.
.Lmultiple_of_four_pipeline_addend_loop:

  // Load the next 4 addends.
  ld64step $current_addend0, $mzero, $addend+=, 1

  // Copy the address of the start of the acts as the store address, and the
  // load address for the pipeline fill stage.
  mov $mscratch0, $acts

  // Cycle 0:      Load 0
  ld64step $tmp0, $mzero, $mscratch0+=, $stride

  // Cycle 1:      Load 1    Add 0
  {
    ld64step $tmp1, $mzero, $mscratch0+=, $stride
    f16v4mix $azeros, $current_addend0, $tmp0
  }

  // Cycle 2:      Load 2    Add 1
  {
    ld64step $tmp0, $mzero, $mscratch0+=, $stride
    f16v4mix $tmp1, $current_addend0, $tmp1
  }

  // First address is the load pointer. Second is ignored. Third is the store
  // pointer.
  tapack $packed_ldst_addrs, $mscratch0, $mzero, $out

  rpt $acts_block_count, (2f - 1f)/8-1
1:
  {
    ldst64pace $tmp1, $tmp1, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mix $tmp0, $current_addend0, $tmp0
  }
  {
    ldst64pace $tmp0, $tmp0, $packed_ldst_addrs+=, $stride, 0x05
    f16v4mix $tmp1, $current_addend0, $tmp1
  }
2:

  brz $acts_block_count_was_odd, 1f
  // If it's odd, do this:
  {
    st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mix $tmp1, $current_addend0, $tmp0
  }

1:
  {
    st64pace $tmp1, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mix $tmp0, $current_addend0, $azeros
  }
  st64pace $tmp0, $packed_ldst_addrs+=, $stride, 0x01

  // Move to the next 4 elements of acts and out (could use a dummy ldst64 here?).
  add $acts, $acts, 8
  add $out, $out, 8

  // Loop and process the next 4 values of addend, if there are any.
  brnzdec $addend_len, .Lmultiple_of_four_pipeline_addend_loop

  br $lr


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                           Multiple of Eight                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

.Lmultiple_of_eight_pipeline:

  // Subtract 2 from $acts_block_count. This means we don't process past
  // the end. The minimum number of blocks required for this pipeline is 1.
  add $acts_block_count, $acts_block_count, -2
  brneg $acts_block_count, _handle_1

  // Work out the stride, in units of 64-bits. It's addend_len / 4.
  shr $stride, $addend_len, 2
  // For this we need the stride minus 4 halfs.
  add $stride, $stride, -1

  // Divide the addend_len by 8 and subtract 1 so we can just brnzdec.
  shr $addend_len, $addend_len, 3
  add $addend_len, $addend_len, -1

  // Loop over the 8-element blocks in the addend.
.Lmultiple_of_eight_pipeline_addend_loop:
  // Load the next 8 elements of the addend.
  ld64step $current_addend0, $mzero, $addend+=, 1
  ld64step $current_addend1, $mzero, $addend+=, 1
  // Pipeline fill.

  // Copy the address of the start of the acts as the store address, and the
  // load address for the pipeline fill stage.
  mov $mscratch0, $acts

  // Cycle 0:      Load 0
  ld64step $tmp0, $mzero, $mscratch0+=, 1

  // Cycle 1:      Load 1    Add 0
  {
    ld64step $tmp1, $mzero, $mscratch0+=, $stride
    f16v4mix $azeros, $current_addend0, $tmp0
  }

  // Cycle 2:      Load 2    Add 1
  {
    ld64step $tmp0, $mzero, $mscratch0+=, 1
    f16v4mix $tmp1, $current_addend1, $tmp1
  }

  // First address is the load pointer. Second is ignored. Third is the store
  // pointer.
  tapack $packed_ldst_addrs, $mscratch0, $mzero, $out

  rpt $acts_block_count, (2f - 1f)/8-1

1:
  {
    // Advance load address by $stride, and store address by 1.
    ldst64pace $tmp1, $tmp1, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mix $tmp0, $current_addend0, $tmp0
  }
  {
    // Advance load address by 1, and store address by $stride.
    ldst64pace $tmp0, $tmp0, $packed_ldst_addrs+=, $stride, 0x04
    f16v4mix $tmp1, $current_addend1, $tmp1
  }
2:
  {
    // Advance load address by $stride, and store address by 1.
    ldst64pace $tmp1, $tmp1, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mix $tmp0, $current_addend0, $tmp0
  }

  {
    // Advance load address by 1, and store address by $stride.
    st64pace $tmp0, $packed_ldst_addrs+=, $stride, 0x01
    f16v4mix $tmp1, $current_addend1, $tmp1
  }

  // Store the last one without processing past the end.
  {
    // Advance store address by 1.
    st64pace $tmp1, $packed_ldst_addrs+=, $mzero, 0
    f16v4mix $tmp0, $current_addend0, $azeros
  }
  st64pace $tmp0, $packed_ldst_addrs+=, $mzero, 0

  // Advance to the next 8 elements of acts and out.
  add $acts, $acts, 16
  add $out, $out, 16

  // Loop and process the next 8 values of addend, if there are any.
  brnzdec $addend_len, .Lmultiple_of_eight_pipeline_addend_loop

  br $lr


// Process a single activtion block with addend length a multiple of 8.
// Performs an extra 64-bit load

_handle_1:

  // Divide the addend_len by 8 and subtract 1 so we can just brnzdec.
  shr $addend_len, $addend_len, 3
  add $addend_len, $addend_len, -1

  ld64step $current_addend0, $mzero, $addend+=, 1

.Loop_acts_eq_1:
    // could use 128-bit loads if $acts were aligned to 16 bytes
    ld64step $tmp0, $mzero, $acts+=, 1
    ld64step $tmp1, $mzero, $acts+=, 1

    {
      ld64step $current_addend1, $mzero, $addend+=, 1
      f16v4mix $azeros, $current_addend0, $tmp0
    }
    {
      ld64step $current_addend0, $mzero, $addend+=, 1
      f16v4mix $tmp0, $current_addend1, $tmp1
    }
    {
      st64step $tmp0, $mzero, $out+=, 1
      f16v4mix $tmp0, $azeros, $azeros
    }
    st64step $tmp0, $mzero, $out+=, 1
    brnzdec $addend_len, .Loop_acts_eq_1
  br $lr

FN_SIZE VectorInnerAdd_core_half


#undef mscratch0
#undef packed_ldst_addrs
#undef stride
#undef acts_block_count_was_odd
#undef tmp0
#undef tmp1
#undef tmp0_lower
#undef current_addend0
#undef current_addend0_lower
#undef current_addend0_upper
#undef current_addend1
#undef scale
#undef ascratch0


/////////////////// VectorInner Supervisor Vertices ///////////////////////

// Vertex state layout for VectorInnerSupervisor
#define VERTEX_DATA_ADDEND_OFFSET 0              // In 32-bits
#define VERTEX_DATA_ADDEND_SIZE_OFFSET 1         // In 32-bits
#define VERTEX_DATA_ACTS_OFFSET 2                // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 6    // In 16-bits
// Additional state for SCALED_ADD version
#define VERTEX_DATA_SCALE_OFFSET 7               // In 16-bits

// Additional state for non-in place, Scaled
#define VERTEX_DATA_OUT_OFFSET  4               // In 32-bits


// The following supervisor variables are used. The vertex base is
// passed in as $m0.
#define supervisor_vertex_base m0
#define worker_entry m1

FUNC_SECTION(Supervisor,SCALED___ADD)
.align 4
.supervisor
CPP_FUNCNAME(Supervisor,SCALED___ADD):
  // Set the entry point for the workers.
  setzi $worker_entry, .LvectorInnerScaledAdd_worker
  // Run the workers.
  bri .Lrun_workers

FN_SIZE CPP_FUNCNAME(Supervisor,SCALED___ADD)


FUNC_SECTION(Supervisor,ADD)
.align 4
.supervisor
CPP_FUNCNAME(Supervisor,ADD):
  // Set the entry point for the workers.
  setzi $worker_entry, .LvectorInnerAdd_worker
  // Run the workers.
  bri .Lrun_workers

FN_SIZE CPP_FUNCNAME(Supervisor,ADD)


FUNC_SECTION(InPlaceSupervisor,SCALED___ADD)
.align 4
.supervisor
CPP_FUNCNAME(InPlaceSupervisor,SCALED___ADD):
  // Set the entry point for the workers.
  setzi $worker_entry, .LvectorInnerScaledAdd_inplace_worker
  // Run the workers.
  bri .Lrun_workers

FN_SIZE CPP_FUNCNAME(InPlaceSupervisor,SCALED___ADD)


FUNC_SECTION(InPlaceSupervisor,ADD)
.align 4
.supervisor
CPP_FUNCNAME(InPlaceSupervisor,ADD):
  // Set the entry point for the workers.
  setzi $worker_entry, .LvectorInnerAdd_inplace_worker
  // Fall through.

.Lrun_workers:
  // Start all workers. Some may have no work to do and just exit.
  runall $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br $lr

FN_SIZE CPP_FUNCNAME(InPlaceSupervisor,ADD)


#undef supervisor_vertex_base
#undef worker_entry

// Worker code

#define blocks_per_worker m4
#define worker_id m5
#define block_begin m6
#define remaining_blocks m7
#define mscratch0 m8
#define scale a0
#define ascratch0 a1


FUNC_SECTION(Worker,SCALED___ADD)
.align 4
.worker
.LvectorInnerScaledAdd_worker:
  ld32 $acts, $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_DATA_OUT_OFFSET
  // Load the scale into the lower half of $scale, and 1.0 into the upper half.
{ ldb16 $scale, $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  setzi $ascratch0, 1.0h }
{ // Jump to the shared worker code.
  bri .Lworker
  sort4x16lo $scale, $scale, $ascratch0 }


FUNC_SECTION(Worker,ADD)
.align 4
.worker
.LvectorInnerAdd_worker:
  ld32 $acts, $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_DATA_OUT_OFFSET
{ // Jump to the shared worker code + set the scale to 1.0.
  bri .Lworker
  // Load (1.0, 1.0) into $scale. This case is special-cased and gives the exact
  // answer and always takes one cycle.
  f16v2exp $scale, $azero}


FUNC_SECTION(WorkerInPlace,SCALED__ADD)
.align 4
.worker
.LvectorInnerScaledAdd_inplace_worker:
  ld32 $acts, $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  mov $out, $acts
  // Load the scale into the lower half of $scale, and 1.0 into the upper half.
{ ldb16 $scale, $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  setzi $ascratch0, 1.0h }
{ // Jump to the shared worker code.
  bri .Lworker
  sort4x16lo $scale, $scale, $ascratch0 }


FUNC_SECTION(WorkerInPlace,ADD)
.align 4
.worker
.LvectorInnerAdd_inplace_worker:
  ld32 $acts, $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
{
  mov $out, $acts
  // Load (1.0, 1.0) into $scale. This case is special-cased and gives the exact
  // answer and always takes one cycle.
  f16v2exp $scale, $azero}
  // Fall through.

.Lworker:

  // Load rest of vertex state.
  ld32 $addend,            $mvertex_base, $mzero, VERTEX_DATA_ADDEND_OFFSET
  ld32 $addend_len,        $mvertex_base, $mzero, VERTEX_DATA_ADDEND_SIZE_OFFSET
{ ldz16 $acts_block_count, $mvertex_base, $mzero, VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Set $TAS to the $scale. This isn't used for the slow path but it is
  // bundled so doesn't cost an extra cycle in that case.
  put $TAS, $scale }

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
  and $mscratch0, $addend_len, 0x1
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

  // How many elements to advance $acts.
update_acts_ptrs:
  mul $mscratch0, $block_begin, $addend_len
  // Advance $acts by 2*$mscratch0 bytes to the $block_begin'th block using
  // a dummy load.
  ldb16step $azero, $mzero, $acts+=, $mscratch0
  ldb16step $azero, $mzero, $out+=, $mscratch0

  call $lr, VectorInnerAdd_core_half

  exitnz $mzero

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0
#undef scale
#undef ascratch0



///////////////////// VectorInner2D Worker Vertices ////////////////////////

// Vertex state layout for BroadcastVectorInner2D
#define VERTEX2D_DATA_N_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_ADDEND_OFFSET 1               // In 32-bits
#define VERTEX2D_DATA_ADDEND_LEN_OFFSET 2           // In 32-bits
#define VERTEX2D_DATA_ACTS_OFFSET 3                 // In 32-bits
#define VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET 4     // In 32-bits
// Additional state for non-inplace & scaled variants
#define VERTEX2D_DATA_OUT_OFFSET 5                  // In 32-bits
#define VERTEX2D_DATA_SCALE_OFFSET 10               // In 16-bits
#define VERTEX2D_DATA_SCALED_OUT_OFFSET 6           // In 32-bits


#define scale a0
#define ascratch0 a1
#define addend_iterator m6
#define addend_len_iterator m7
#define acts_iterator m8
#define acts_block_count_iterator m9
#define n m11
#define out_iterator m4

#define SCRATCH_OFFSET_ADDEND_ITERATOR 0
#define SCRATCH_OFFSET_ADDEND_LEN_ITERATOR 1
#define SCRATCH_OFFSET_N 2
#define SCRATCH_OFFSET_OUT_ITERATOR   3
#define SCRATCH_OFFSET_ACTS_ITERATOR   4


FUNC_SECTION(2D,SCALED___ADD)
.align 4
CPP_FUNCNAME(2D,SCALED___ADD):
  ld32 $acts_iterator,  $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  ld32 $out_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_SCALED_OUT_OFFSET
  // Load the scale into the lower half of $scale, and 1.0 into the upper half.
{ ldb16 $scale, $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  setzi $ascratch0, 1.0h }
{ // Jump to the shared worker code.
  bri .Lworker2d
  sort4x16lo $scale, $scale, $ascratch0 }

FN_SIZE CPP_FUNCNAME(2D,SCALED___ADD)


FUNC_SECTION(2D,ADD)
.align 4
CPP_FUNCNAME(2D,ADD):
  ld32 $acts_iterator,  $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  ld32 $out_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_OUT_OFFSET
{ // Jump to the shared worker code + set the scale to 1.0.
  bri .Lworker2d
  // Load (1.0, 1.0) into $scale. This case is special-cased and gives the exact
  // answer and always takes one cycle.
  f16v2exp $scale, $azero}

FN_SIZE CPP_FUNCNAME(2D,ADD)


FUNC_SECTION(2DInPlace,SCALED___ADD)
.align 4
CPP_FUNCNAME(2DInPlace,SCALED___ADD):
  ld32 $acts_iterator,  $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  mov $out_iterator, $acts_iterator
  // Load the scale into the lower half of $scale, and 1.0 into the upper half.
{ ldb16 $scale, $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  setzi $ascratch0, 1.0h }
{ // Jump to the shared worker code.
  bri .Lworker2d
  sort4x16lo $scale, $scale, $ascratch0 }

FN_SIZE CPP_FUNCNAME(2DInPlace,SCALED___ADD)


FUNC_SECTION(2DInPlace,ADD)
.align 4
CPP_FUNCNAME(2DInPlace,ADD):
  ld32 $acts_iterator,  $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
{
  mov $out_iterator, $acts_iterator
  // Load (1.0, 1.0) into $scale. This case is special-cased and gives the exact
  // answer and always takes one cycle.
  f16v2exp $scale, $azero}
  // Fall through.

// $scale is needed below.
#undef ascratch0



.Lworker2d:

  ld32 $n,                         $mvertex_base, $mzero, VERTEX2D_DATA_N_OFFSET
  ld32 $addend_iterator,           $mvertex_base, $mzero, VERTEX2D_DATA_ADDEND_OFFSET
  ld32 $addend_len_iterator,       $mvertex_base, $mzero, VERTEX2D_DATA_ADDEND_LEN_OFFSET
{ ld32 $acts_block_count_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Set $TAS to the $scale. This isn't used for the slow path but it is
  // bundled so doesn't cost an extra cycle in that case.
  put $TAS, $scale }

  // Subtract one for brnzdec
  add $n, $n, -1

.Louter_loop:
  // We need to save this straight away (as it's an alias of 'out')
  st32  $n, $mworker_base, $mzero, SCRATCH_OFFSET_N

  // Advance all the iterators.
  ld32step  $addend,           $mzero, $addend_iterator+=, 1
  ldz16step $addend_len,       $mzero, $addend_len_iterator+=, 1
  ld32step  $acts,             $mzero, $acts_iterator+=, 1
  ldz16step $acts_block_count, $mzero, $acts_block_count_iterator+=, 1
  ld32step  $out,              $mzero, $out_iterator+=, 1

  // We need to save & restore these as they are clobbered by the function call.
  st32 $acts_iterator,       $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_ITERATOR
  st32 $addend_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_ADDEND_ITERATOR
  st32 $addend_len_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_ADDEND_LEN_ITERATOR
  st32 $out_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_OUT_ITERATOR

  call $lr, VectorInnerAdd_core_half

  ld32 $acts_iterator,       $mworker_base, $mzero, SCRATCH_OFFSET_ACTS_ITERATOR
  ld32 $addend_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_ADDEND_ITERATOR
  ld32 $addend_len_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_ADDEND_LEN_ITERATOR
  ld32 $out_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_OUT_ITERATOR
  ld32  $n, $mworker_base, $mzero, SCRATCH_OFFSET_N

  brnzdec $n, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2DInPlace,ADD)

#undef addend_iterator
#undef addend_len_iterator
#undef acts_iterator
#undef acts_block_count_iterator
#undef n

#endif // __IPU__
