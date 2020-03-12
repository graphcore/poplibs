// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
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
// The f32 case is a lot simpler than f16 because we have no subword accesses
// and there are fewer paths.
//
// ## Fast Paths
//
// The best we could do is process 2 f32's per cycle using a `rpt` of.
//
//   { ldst64pace; f32v2axpy }
//
// Currently this fast path is not implemented, as it would requires 8 byte
// alignment for 'data'/'out' and also them being allocated in different memory
// regions or in interleaved memory (for inplace operations).

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
#define CPP_FUNCNAME(TYPE, OPERATION) __runCodelet_popops__BroadcastVectorInner ## TYPE ## ___popops__expr__BroadcastOpType__ ## OPERATION ## _float

// Another similar macro to name the sections where each function is contained
#define FUNC_SECTION(TYPE, OPERATION) .section .text.VectorInner_ ## TYPE ## _ ## OPERATION ## _float


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
#define scale a0
#define out m11
//
// $m0, $m2 and $m11 are modified, but the others are not. $m10 ($lr) is used for
// the return address. It also uses the following scratch registers:
//
#define outer_stride m4
#define addend_loop_count m5
#define tmp0 a5
#define tmp1 a6
#define current_addend a7

.section .text.VectorInnerAdd_core_float
.align 8
VectorInnerAdd_core_float:
  // If we have no blocks to do, return.
  brz $acts_block_count, .Lreturn

  // We use `rpt` which has a limited loop count, but this is taken care of
  // in the popconv host code.

  // In future this could be optimised by using dedicated multiple-of-2 and
  // multiple-of-4 pipelines as in the half code. But this code is not as slow
  // as the half scalar code so it isn't such a priority.

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
{ ld32step $tmp0, $mzero, $acts+=, $addend_len
  // Multiply the addend by the scale.
  f32mul $current_addend, $current_addend, $scale }

  // Loop through acts. This must be 8-byte aligned which can be done with
  // `.align 8` but that might insert a `nop` and waste a cycle. Instead
  // we do it manually using bundles if necessary.
  rpt $acts_block_count, (2f - 1f)/8-1
1:
  {
    ld32step $tmp0, $mzero, $acts+=, $addend_len
    f32add $tmp1, $tmp0, $current_addend
  }
  {
    st32step $tmp1, $mzero, $out+=, $addend_len
    fnop
  }
2:

  // Add and store the last value.
  f32add $tmp1, $tmp0, $current_addend
  st32step $tmp1, $mzero, $out+=, $addend_len

  // Move the acts and out pointers back to the next element.
  sub $acts, $acts, $outer_stride
  sub $out, $out, $outer_stride

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

FN_SIZE VectorInnerAdd_core_float


/////////////////// VectorInner Supervisor Vertices ///////////////////////

// Vertex state layout for VectorInnerSupervisor
#define VERTEX_DATA_ADDEND_OFFSET 0            // In 32-bits
#define VERTEX_DATA_ADDEND_END_OFFSET 1        // In 32-bits
#define VERTEX_DATA_ACTS_OFFSET 2              // In 32-bits
#define VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET 6  // In 16-bits
// Additional state for SCALED_ADD version
#define VERTEX_DATA_SCALE_OFFSET 4             // In 32-bits

// Additional state for non-in place, Scaled
#define VERTEX_DATA_OUT_OFFSET  4              // In 32-bits
#define VERTEX_DATA_SCALED_OUT_OFFSET  5       // In 32-bits


// The following supervisor variables are used. The vertex base is
// passed in as $m0.
#define supervisor_vertex_base m0
#define worker_entry m1


FUNC_SECTION(Supervisor,SCALED___ADD)
.align 4
CPP_FUNCNAME(Supervisor,SCALED___ADD):
  // Set the entry point for the workers.
  setzi        $worker_entry, .LvectorInnerScaledAdd_worker
  // Run the workers.
  bri .Lrun_workers

FN_SIZE CPP_FUNCNAME(Supervisor,SCALED___ADD)


FUNC_SECTION(Supervisor,ADD)
.align 4
CPP_FUNCNAME(Supervisor,ADD):
  // Set the entry point for the workers.
  setzi        $worker_entry, .LvectorInnerAdd_worker
  // Run the workers.
  bri .Lrun_workers

FN_SIZE CPP_FUNCNAME(Supervisor,ADD)


FUNC_SECTION(InPlaceSupervisor,SCALED___ADD)
.align 4
CPP_FUNCNAME(InPlaceSupervisor,SCALED___ADD):
  // Set the entry point for the workers.
  setzi        $worker_entry, .LvectorInnerScaledAdd_inplace_worker
  // Run the workers.
  bri .Lrun_workers

FN_SIZE CPP_FUNCNAME(InPlaceSupervisor,SCALED___ADD)


FUNC_SECTION(InPlaceSupervisor,ADD)
.align 4
.supervisor
CPP_FUNCNAME(InPlaceSupervisor,ADD):
  // Set the entry point for the workers.
  setzi        $worker_entry, .LvectorInnerAdd_inplace_worker
  // Fall through.

.Lrun_workers:
  // Start all workers. Some may have no work to do and just exit.
  runall       $worker_entry, $supervisor_vertex_base, 0
  // Wait for all the workers to exit.
  sync         TEXCH_SYNCZONE_LOCAL
  // Return to caller.
  br           $lr

FN_SIZE CPP_FUNCNAME(InPlaceSupervisor,ADD)


#undef supervisor_vertex_base
#undef worker_entry

// Worker code.

#define blocks_per_worker m4
#define worker_id m5
#define block_begin m6
#define remaining_blocks m7
#define mscratch0 m8


FUNC_SECTION(Worker,SCALED___ADD)
.align 4
.worker
.LvectorInnerScaledAdd_worker:
  ld32 $acts,  $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  ld32 $out,   $mvertex_base, $mzero, VERTEX_DATA_SCALED_OUT_OFFSET
  // Load the scale.
  ld32 $scale, $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker


FUNC_SECTION(Worker,ADD)
.align 4
.LvectorInnerAdd_worker:
  ld32 $acts,  $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  ld32 $out,   $mvertex_base, $mzero, VERTEX_DATA_OUT_OFFSET
{ // Jump to the shared worker code + set the scale to 1.0.
  bri .Lworker
  f32exp $scale, $azero}


FUNC_SECTION(WorkerInPlace,SCALED__ADD)
.align 4
.LvectorInnerScaledAdd_inplace_worker:
  ld32 $acts,  $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  mov  $out, $acts
  // Load the scale.
  ld32 $scale, $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker


FUNC_SECTION(WorkerInPlace,ADD)
.align 4
.LvectorInnerAdd_inplace_worker:
  ld32 $acts,  $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
{
  mov  $out, $acts
  // Set the scale to 1.0.
  f32exp $scale, $azero}
  // Fall through.

.Lworker:
  // Load vertex state.
  ld32 $addend,            $mvertex_base, $mzero, VERTEX_DATA_ADDEND_OFFSET
  ld32 $addend_len,        $mvertex_base, $mzero, VERTEX_DATA_ADDEND_END_OFFSET
  ldz16 $acts_block_count, $mvertex_base, $mzero, VERTEX_DATA_ACTS_BLOCK_COUNT_OFFSET

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
  // Advance $acts and $out by 4*$mscratch0 bytes to the $block_begin'th block,
  // using a dummy load.
  ld32step $azero, $mzero, $acts+=, $mscratch0
  ld32step $azero, $mzero, $out+=, $mscratch0

  call $lr, VectorInnerAdd_core_float

  exitnz $mzero

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0

///////////////////// VectorInner2D Worker Vertices ////////////////////////

// Vertex state layout for BroadcastVectorInner2D
#define VERTEX2D_DATA_N_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_ADDEND_OFFSET 1               // In 32-bits
#define VERTEX2D_DATA_ADDEND_LEN_OFFSET 2           // In 32-bits
#define VERTEX2D_DATA_ACTS_OFFSET 3                 // In 32-bits
#define VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET 4     // In 32-bits
// Additional state for non-inplace & scaled variants
#define VERTEX2D_DATA_OUT_OFFSET 5                  // In 32-bits
#define VERTEX2D_DATA_SCALE_OFFSET 5                // In 32-bits
#define VERTEX2D_DATA_SCALED_OUT_OFFSET 6           // In 32-bits


#define addend_iterator m6
#define addend_len_iterator m7
#define acts_iterator m8
#define acts_block_count_iterator m9
#define n m11
#define out_iterator m4

#define SCRATCH_OFFSET_N              0
#define SCRATCH_OFFSET_OUT_ITERATOR   1


FUNC_SECTION(2D,SCALED___ADD)
.align 4
CPP_FUNCNAME(2D,SCALED___ADD):
  ld32 $acts_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  ld32 $out_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_SCALED_OUT_OFFSET
  // Load the scale.
  ld32 $scale, $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker2d

FN_SIZE CPP_FUNCNAME(2D,SCALED___ADD)


FUNC_SECTION(2D,ADD)
.align 4
CPP_FUNCNAME(2D,ADD):
  ld32 $acts_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  ld32 $out_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_OUT_OFFSET
{ // Jump to the shared worker code + set the scale to 1.0.
  bri .Lworker2d
  f32exp $scale, $azero}

FN_SIZE CPP_FUNCNAME(2D,ADD)


FUNC_SECTION(2DInPlace,SCALED___ADD)
.align 4
CPP_FUNCNAME(2DInPlace,SCALED___ADD):
  ld32 $acts_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
  mov $out_iterator, $acts_iterator
  // Load the scale.
  ld32 $scale, $mvertex_base, $mzero, VERTEX2D_DATA_SCALE_OFFSET
  // Jump to the shared worker code.
  bri .Lworker2d

FN_SIZE CPP_FUNCNAME(2DInPlace,SCALED___ADD)


FUNC_SECTION(2DInPlace,ADD)
.align 4
CPP_FUNCNAME(2DInPlace,ADD):
  ld32 $acts_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_OFFSET
{
  mov $out_iterator, $acts_iterator
  // Set the scale to 1.0.
  f32exp $scale, $azero}

  // Fall through.
.Lworker2d:

  ld32 $n,                         $mvertex_base, $mzero, VERTEX2D_DATA_N_OFFSET
  ld32 $addend_iterator,           $mvertex_base, $mzero, VERTEX2D_DATA_ADDEND_OFFSET
  ld32 $addend_len_iterator,       $mvertex_base, $mzero, VERTEX2D_DATA_ADDEND_LEN_OFFSET
  ld32 $acts_block_count_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_BLOCK_COUNT_OFFSET

  // Subtract one for brnzdec
  add $n, $n, -1

.Louter_loop:
  // We need to save this straight away (as it's an alias of 'out')
  st32 $n, $mworker_base, $mzero, SCRATCH_OFFSET_N

  // Advance all the iterators.
  ld32step  $addend,           $mzero, $addend_iterator+=, 1
  ldz16step $addend_len,       $mzero, $addend_len_iterator+=, 1
  ld32step  $acts,             $mzero, $acts_iterator+=, 1
  ldz16step $acts_block_count, $mzero, $acts_block_count_iterator+=, 1
  ld32step  $out,              $mzero, $out_iterator+=, 1

  // We need to save & restore these as they are clobbered by the function call.
  st32 $out_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_OUT_ITERATOR

  call $lr, VectorInnerAdd_core_float

  ld32 $out_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_OUT_ITERATOR
  ld32 $n, $mworker_base, $mzero, SCRATCH_OFFSET_N

  brnzdec $n, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2DInPlace,ADD)

#undef addend_iterator
#undef addend_len_iterator
#undef acts_iterator
#undef acts_block_count_iterator
#undef n

#endif // __IPU__
