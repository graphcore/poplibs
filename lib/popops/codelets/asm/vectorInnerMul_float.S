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

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

// Let's create a macro to shorten the name for the entry points, which are
// very long, as required by C++ name mangling.
// TYPE needs to be '1D', '1DInPlace', '2D' or '2DInPlace'.
#define CPP_FUNCNAME(TYPE)  __runCodelet_popops__BroadcastVectorInner ## TYPE ## ___popops__expr__BinaryOpType__MULTIPLY_float

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

FN_SECTION VectorInnerMul_core_float 8
VectorInnerMul_core_float:
  // If we have no blocks to do, return.
  brz $acts_block_count, .Lreturn

  // We use `rpt` which has a limited loop count, but this is taken care of
  // in the popops host code.

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

FN_SIZE VectorInnerMul_core_float


// Undefine scratch registers. Arguments are left defined for the functions
// below.
#undef outer_stride
#undef scale_loop_count
#undef tmp0
#undef tmp1
#undef current_scale



/////////////////// VectorInner MultiVertex ///////////////////////

// Vertex state layout for the MultiVertex
#define VERTEX_DATA_SCALE_OFFSET 0                // In 32-bits
#define VERTEX_DATA_ACTS_OFFSET 1               // In 32-bits
#define VERTEX_DATA_BSLICES_M1_OFFSET 4      // In 16-bits
#define VERTEX_DATA_BSLICE_LEN_OFFSET 5         // In 16-bits
#define VERTEX_DATA_BBROADCAST_FACTOR_OFFSET 6               // In 16-bits
// (Gap)
// Additional state for non-in place, Scaled
#define VERTEX_DATA_OUT_OFFSET  4                // In 32-bits

#define blocks_per_worker m5
#define worker_id m6
#define block_begin m7
#define remaining_blocks m8
#define mscratch0 m9

#include "vectorInnerCommon.h.S"


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(1DInPlace)
  ld32 $acts_in, $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  mov $acts_out, $acts_in
  bri .Lworker
FN_SIZE CPP_FUNCNAME(1DInPlace)


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(1D)
  ld32 $acts_in, $mvertex_base, $mzero, VERTEX_DATA_ACTS_OFFSET
  ld32 $acts_out, $mvertex_base, $mzero, VERTEX_DATA_OUT_OFFSET
  // Fall through.

.Lworker:
  // Load vertex state.
  ld32 $scale,             $mvertex_base, $mzero, VERTEX_DATA_SCALE_OFFSET
  ldz16 $scale_len,         $mvertex_base, $mzero, VERTEX_DATA_BBROADCAST_FACTOR_OFFSET

  // Get the worker ID.
  get $worker_id, $WSR
  and $worker_id, $worker_id, CSR_W_WSR__CTXTID_M1__MASK

  // We need to know the number of loops for each worker, and the offset into the
  // $acts data to start processing.  Do this by finding the amount of work for
  // worker zero and multiplying by $worked_id.
  // Optimised by expanding out and simplifying the DIVIDE_BY_WORKER macro

  add $acts_block_count, $scale_len, CTXT_WORKERS-1
  // (Using worker=0)
  setzi $mscratch0, RECIPROCAL_3_SHL17
  mul   $acts_block_count, $acts_block_count, $mscratch0
  shr   $acts_block_count, $acts_block_count, DIV_6_SHIFT

  mul   $block_begin, $acts_block_count, $worker_id
  // If the begin block is after the end then this worker has no work
  cmpult $mscratch0,  $scale_len, $block_begin
  brnz   $mscratch0, 9f

  // Check if this worker has less than the maximum work
  add    $mscratch0, $block_begin, $acts_block_count
  cmpult $mscratch0, $mscratch0, $scale_len
  brnz   $mscratch0, update_acts_ptrs
  // numBlocks = bReps - numBlocks*wid;
  sub    $acts_block_count, $scale_len, $block_begin

update_acts_ptrs:
  // How many elements to advance $acts.
  ldz16 $scale_len, $mvertex_base, $mzero, VERTEX_DATA_BSLICE_LEN_OFFSET
  mul $mscratch0, $block_begin, $scale_len
  // Advance $acts_in and $acts_out by 4*$mscratch0 bytes to the
  // $block_begin'th block using a dummy load.
  ld32step $azero, $mzero, $acts_in+=, $mscratch0
  ld32step $azero, $mzero, $acts_out+=, $mscratch0

  ldz16 $remaining_blocks, $mvertex_base, $mzero, VERTEX_DATA_BSLICES_M1_OFFSET
  brnz $remaining_blocks, 2f

  call $lr, VectorInnerMul_core_float
9:
  exitnz $mzero

// Multiple outer loops, call in a loop, but use the stack (Scratch) to preserve registers
2:
  VECTOR_INNER_OUTER_LOOP_WRAPPER VectorInnerMul_core_float ld32step $acts_in $acts_out $scale_len $scale $acts_block_count
  exitnz $mzero

FN_SIZE CPP_FUNCNAME(1D)

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
// Additional state for non-inplace variant
#define VERTEX2D_DATA_ACTS_OUT_OFFSET 3             // In 32-bits


#define scale_iterator m5       // Overwritten by function call
#define acts_in_iterator m7
#define acts_out_iterator m8
#define workList_iterator m9
#define nM1 m11

#define SCRATCH_OFFSET_SCALE_ITERATOR 0

FN_WORKER_ENTRY_POINT CPP_FUNCNAME(2DInPlace)
  ld32 $acts_in_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_ACTS_IN_OFFSET
  mov $acts_out_iterator, $acts_in_iterator
  bri .Lworker2d

FN_SIZE CPP_FUNCNAME(2DInPlace)


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(2D)
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

  call $lr, VectorInnerMul_core_float

  ld32 $scale_iterator,     $mworker_base, $mzero, SCRATCH_OFFSET_SCALE_ITERATOR
  brnzdec $nM1, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2D)


#undef scale_iterator
#undef workList_iterator
#undef acts_in_iterator
#undef acts_out_iterator
#undef nM1

#endif // __IPU__
