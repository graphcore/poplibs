// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

// Data arrangement.
// Each of the data vectors: in, out, fwdActIn, fwdActsOut is laid out in this way:

// ------------------------------------------------------------------------
// | Slice 0 Row 0 | Slice 0 Row 1 || Slice 1 Row 0  | Slice 1 Row 1 || Slice 2....
// ------------------------------------------------------------------------
//                Slice 0          ||              Slice 1           || Slice 2
//
// The vertex processes a number of slices, each of which is split into a
// number of rows.  Each slice is the same size and represents a channel group.
// A channel group can have any multiple of 4 (half) or 2 (float) channels in it.
// Each worker is responsible for a part of the workList.  A piece of its workList
// will process Slice0-Row0 and Slice1-Row0 in an identical way.
//
// Due to the possibility of a stride in the pooling operation, an `in` vector
// can be a different size to an `out` vector.
//
// The outmost loop is for each row.
// The next loop in is for each worklist entry
// The next loop is over the slices
// So the inner most loop processes a worklist entry over a given row within the slice.
//
// Observations on efficiency:
// We need the effective row length to be large to get to use the inner loop with a
// large loop count - this is the most efficient the vertex can be.
// Several worklist entries are required to process any output - a 2x2 kernel needs
// 4 worklist entries, so the worklist loop needs to run several times.
// There can be just one row, one slice (Possible with work division between tiles),
// so the slice loop is inside the worklist loop and the slice loop may not even run.
// We could swap the worklist and slice loops, but the slice loop is simple and the
// worklist loop complex.

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
#define COMPACT_VECTOR_TYPES_AVAILABLE 1
#else
#define COMPACT_VECTOR_TYPES_AVAILABLE 0
#endif

#if COMPACT_VECTOR_TYPES_AVAILABLE
// vertex state, all offsets are 16-bit. all of the output and inputs (inc.
// the activations) are aligned to 64-bit and the size is guaranteed to align
// to 32-bits (so, there will be an even amount of halves).
#define VERTEX_OUT_PTR_OFFSET 0
#define VERTEX_IN_PTR_OFFSET 2
#define VERTEX_START_POS_PTR_OFFSET 4
#define VERTEX_OFFSET_BASE_PTR_OFFSET 6
#define VERTEX_WORK_LIST_BASE_OFFSET 8
#define VERTEX_WORK_LIST_DELTA_PTR_OFFSET 12
#define VERTEX_INIT_INFO_OFFSET 14
#define VERTEX_NUM_CHAN_GROUPS_OFFSET 16
#define VERTEX_CHANS_PER_GROUP_OFFSET 18
#define VERTEX_IN_STRIDE_OFFSET 20
#define VERTEX_OUT_STRIDE_OFFSET 24
#define VERTEX_IN_SLICE_SIZE  28
#define VERTEX_OUT_SLICE_SIZE 32
#define VERTEX_SCALE_OFFSET 36 // SumPooling only
#define VERTEX_FWD_ACTS_IN_PTR_OFFSET 36 // MaxPoolingGrad only
#define VERTEX_FWD_ACTS_OUT_PTR_OFFSET 38 // MaxPoolingGrad only

#else

#define VERTEX_OUT_PTR_OFFSET 0
#define VERTEX_IN_PTR_OFFSET 4
#define VERTEX_START_POS_PTR_OFFSET 8
#define VERTEX_OFFSET_BASE_PTR_OFFSET 12
#define VERTEX_WORK_LIST_BASE_OFFSET 16
#define VERTEX_WORK_LIST_DELTA_PTR_OFFSET 20
#define VERTEX_INIT_INFO_OFFSET 24
#define VERTEX_NUM_CHAN_GROUPS_OFFSET 26
#define VERTEX_CHANS_PER_GROUP_OFFSET 28
#define VERTEX_IN_STRIDE_OFFSET 32
#define VERTEX_OUT_STRIDE_OFFSET 36
#define VERTEX_IN_SLICE_SIZE  40
#define VERTEX_OUT_SLICE_SIZE 44
#define VERTEX_SCALE_OFFSET 48 // SumPooling only
#define VERTEX_FWD_ACTS_IN_PTR_OFFSET 48 // MaxPoolingGrad only
#define VERTEX_FWD_ACTS_OUT_PTR_OFFSET 52 // MaxPoolingGrad only
#endif

// stack state, all offsets are 32-bit
#define STACK_NUM_ROWS_OFFSET 0
#define STACK_NUM_WORK_ITEMS_OFFSET 1
#define STACK_START_POS_OFFSET 2
#define STACK_WORK_LIST_BASE_OFFSET 3
#define STACK_PACKED_INOUT_POS_OFFSET 4
#define STACK_OFFSET_BASE_PTR_OFFSET 5
#define STACK_WORKLIST_DELTA_PTR_OFFSET 6
#define STACK_OUT_PTR_OFFSET 7
#define STACK_IN_PTR_OFFSET 8
#define STACK_FWD_ACTS_IN_PTR_OFFSET 9
#define STACK_FWD_ACTS_OUT_PTR_OFFSET 10

// constants
#define LDCONST_MASK ((1<<20)-1)
#define SCALED_PTR32_SHL_BITS 2
#define SCALED_PTR64_SHL_BITS 3

#if COMPACT_VECTOR_TYPES_AVAILABLE
#define DELTAN_BASE_PTR_BITS 20
#define DELTAN_COUNT_BITS 12
#define DELTAN_OFFSET_BITS 18
#define DELTAN_LENGTH_BITS 14
#else
#define DELTAN_BASE_PTR_BITS 24
#define DELTAN_COUNT_BITS 8
#define DELTAN_OFFSET_BITS 20
#define DELTAN_LENGTH_BITS 12
#endif

// Number of implicit zeroes in variously aligned addresses
#define PTR16_SHL_BITS 1
#define PTR32_SHL_BITS 2
#define PTR64_SHL_BITS 3

// integer variables, due to register pressure there is quite a bit of overlap.
// we must be careful that the liveness of aliases registers does not clash.

// Define loop variables - need to be the same as those defined before the inner loop macros
// are used
#define fwdActsOutPtr m0
#define fwdActsInPtr m3
#define inPtr m4
#define outPtr m8
#define outOffset m6
#define inOffset m7
#define inStride m9
#define outStride m10

#define numElementsM1 m11


// NumRows isn't valid throughout but is referenced throughout
#define numRows m8

// =============================================================================
// Inner loop macros
// =============================================================================
.macro MAX_POOLING_BWD_FLOAT
  // pipeline
  ld64 $a2:3, $fwdActsOutPtr, $inOffset, 0

  // note that in for the backwards pass it is correct to index into the
  // fwdActsIn pointer with outOffset and the fwdActsOut pointer with inOffset.
  {
    rpt $numElementsM1, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld64 $a0:1, $fwdActsInPtr, $outOffset, 0
    fnop
  }
  {
    ld64step $a4:5, $inPtr, $inOffset+=, $inStride
    f32v2cmpeq $a0:1, $a0:1, $a2:3
  }
  {
    ld64 $a6:7, $outPtr, $outOffset, 0
    and64 $a4:5, $a0:1, $a4:5
  }
  {
    ld64 $a2:3, $fwdActsOutPtr, $inOffset, 0
    f32v2add $a6:7, $a4:5, $a6:7
  }
  {
    st64step $a6:7, $outPtr, $outOffset+=, $outStride
    fnop
  }
2:
  ld64 $a0:1, $fwdActsInPtr, $outOffset, 0
  {
    ld64step $a4:5, $inPtr, $inOffset+=, $inStride
    f32v2cmpeq $a0:1, $a0:1, $a2:3
  }
  {
    ld64 $a6:7, $outPtr, $outOffset, 0
    and64 $a4:5, $a0:1, $a4:5
  }
  f32v2add $a6:7, $a4:5, $a6:7
  st64step $a6:7, $outPtr, $outOffset+=, $outStride
.endm
// =============================================================================
.macro MAX_POOLING_FWD_FLOAT
  // pipeline first values
  ld64 $a0:1, $outPtr, $outOffset, 0
  {
    ld64step $a2:3, $inPtr, $inOffset+=, $inStride
    fnop
  }
  {
    rpt $numElementsM1, (2f-1f)/8-1
    f32v2max $a6:7, $a0:1, $a2:3
  }
1:
  {
    ld64step $a2:3, $inPtr, $inOffset+=, $inStride
    fnop
  }
  {
    ld64 $a0:1, $outPtr, $outOffset, $outStride
    fnop
  }
  {
    st64step $a6:7, $outPtr, $outOffset+=, $outStride
    f32v2max $a6:7, $a0:1, $a2:3
  }
2:
  st64 $a6:7, $outPtr, $outOffset, 0
.endm
// =============================================================================
.macro MAX_POOLING_BWD_HALF
 // pipeline
  ld64 $a2:3, $fwdActsOutPtr, $inOffset, 0

  // note that in this loop it is correct to index into the fwdActsIn pointer
  // with outOffset and the fwdActsOut pointer with inOffset.
  {
    rpt $numElementsM1, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld64 $a0:1, $fwdActsInPtr, $outOffset, 0
    fnop
  }
  {
    ld64step $a4:5, $inPtr, $inOffset+=, $inStride
    f16v4cmpeq $a0:1, $a0:1, $a2:3
  }
  {
    ld64 $a6:7, $outPtr, $outOffset, 0
    and64 $a4:5, $a0:1, $a4:5
  }
  {
    ld64 $a2:3, $fwdActsOutPtr, $inOffset, 0
    f16v4add $a6:7, $a4:5, $a6:7
  }
  {
    st64step $a6:7, $outPtr, $outOffset+=, $outStride
    fnop
  }
2:
  ld64 $a0:1, $fwdActsInPtr, $outOffset, 0
  {
    ld64step $a4:5, $inPtr, $inOffset+=, $inStride
    f16v4cmpeq $a0:1, $a0:1, $a2:3
  }
  {
    ld64 $a6:7, $outPtr, $outOffset, 0
    and64 $a4:5, $a0:1, $a4:5
  }
  f16v4add $a6:7, $a4:5, $a6:7
  st64step $a6:7, $outPtr, $outOffset+=, $outStride
.endm
// =============================================================================
.macro MAX_POOLING_FWD_HALF
  // pipeline first values
  ld64 $a0:1, $outPtr, $outOffset, 0
  {
    ld64step $a2:3, $inPtr, $inOffset+=, $inStride
    fnop
  }
  {
    rpt $numElementsM1, (2f-1f)/8-1
    f16v4max $a6:7, $a0:1, $a2:3
  }
1:
  {
    ld64step $a2:3, $inPtr, $inOffset+=, $inStride
    fnop
  }
  {
    ld64 $a0:1, $outPtr, $outOffset, $outStride
    fnop
  }
  {
    st64step $a6:7, $outPtr, $outOffset+=, $outStride
    f16v4max $a6:7, $a0:1, $a2:3
  }
2:
  st64 $a6:7, $outPtr, $outOffset, 0
.endm
// =============================================================================
.macro SUM_POOLING_FLOAT
  // pipeline first value.
  ld64step $a2:3, $inPtr, $inOffset+=, $inStride
  {
    rpt $numElementsM1, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld64 $a0:1, $outPtr, $outOffset, 0
    f32v2mul $a2:3, $a2:3, $a6:7
  }
  {
    ld64step $a2:3, $inPtr, $inOffset+=, $inStride
    f32v2add $a0:1, $a0:1, $a2:3
  }
  {
    st64step $a0:1, $outPtr, $outOffset+=, $outStride
    fnop
  }
2:
  {
    ld64 $a0:1, $outPtr, $outOffset, 0
    f32v2mul $a2:3, $a2:3, $a6:7
  }
  f32v2add $a0:1, $a0:1, $a2:3
  st64step $a0:1, $outPtr, $outOffset+=, $outStride
.endm
// =============================================================================
.macro SUM_POOLING_HALF
  // pipeline first value
  ld64step $a2:3, $inPtr, $inOffset+=, $inStride
  {
    rpt $numElementsM1, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld64 $a0:1, $outPtr, $outOffset, 0
    f16v4mul $a2:3, $a2:3, $a6:7
  }
  {
    ld64step $a2:3, $inPtr, $inOffset+=, $inStride
    f16v4add $a0:1, $a0:1, $a2:3
  }
  {
    st64step $a0:1, $outPtr, $outOffset+=, $outStride
    fnop
  }
2:
  {
    ld64 $a0:1, $outPtr, $outOffset, 0
    f16v4mul $a2:3, $a2:3, $a6:7
  }
  f16v4add $a0:1, $a0:1, $a2:3
  st64step $a0:1, $outPtr, $outOffset+=, $outStride
.endm

// ============================
#undef fwdActsOutPtr
#undef fwdActsInPtr
#undef inPtr
#undef outPtr
#undef outOffset
#undef inOffset
#undef inStride
#undef outStride

#undef numElementsM1
// ============================

// =============================================================================
// Macro for supervisor entry then worker function
.macro DEFINE_VERTEX symbol isMaxPool isFloat isBwdPass INNER_LOOP_FUNCTION
// Set initial value for the actual calculation
.if \isMaxPool && !\isBwdPass
  .if \isFloat
    // -inf
    .equ INITIAL_VALUE, 0xff800000
  .else
    // [-65504, -65504]
    .equ INITIAL_VALUE, 0xfbfffbff
  .endif
.else
  .equ INITIAL_VALUE, 0
.endif

.globl \symbol
.type \symbol @function

DEF_STACK_USAGE 0 \symbol
.section .text.\symbol

.align 8
.worker
  nop       // rpt alignment
\symbol:

// ============================
#define numChanGroupsM1 m1

#define chansPerGroup m2
#define initInfo m5
#define worker_x8_i m5

#define outPtr_i m7
#define numElems_i  m9
#define outPtr m8
#define rem_i       m10
#define extra_i     m10
#define workerPtr m11
#if COMPACT_VECTOR_TYPES_AVAILABLE
#define base m4
#else
#define base mzero
#endif
// ============================

  // load vertex state needed for initialisation first.
  ldz16 $initInfo, $mzero, $mvertex_base, VERTEX_INIT_INFO_OFFSET/2
  ldz16 $numChanGroupsM1, $mzero, $mvertex_base, VERTEX_NUM_CHAN_GROUPS_OFFSET/2
  ldz16 $chansPerGroup, $mzero, $mvertex_base, VERTEX_CHANS_PER_GROUP_OFFSET/2
  add   $chansPerGroup, $chansPerGroup, 1

  // unpack the scaled output pointer
#if COMPACT_VECTOR_TYPES_AVAILABLE
  setzi $base, TMEM_REGION0_BASE_ADDR
  ldz16 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/2
  shl $outPtr, $outPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
#endif
  st32 $outPtr, $mzero, $mworker_base, STACK_OUT_PTR_OFFSET

  // most of the state (including the offsets inside the work list) are scaled
  // by chansPerGroup to save memory, expand init info now, we will scale up
  // the two strides when we load them after output initialisation.
  mul $initInfo, $initInfo, $chansPerGroup

  // identity for max pool float is -inf, half is -65504 and all sum pool and
  // max pool grad is 0. (set on macro entry). split the ldconst up to bundle,
  // although not necessary when zero it doesn't cost any cycles as we can bunde

 {
    get $workerPtr, $WSR
    setzi $a0, INITIAL_VALUE & LDCONST_MASK
  }
  {
    and $workerPtr, $workerPtr, CSR_W_WSR__CTXTID_M1__MASK
    or $a0, $a0, INITIAL_VALUE & ~LDCONST_MASK
  }
  // Divide work equally per channel amongst workers
  mul $numElems_i, $initInfo, 21846 // sufficient to cover a range of [0:4095*6]
  shr $numElems_i, $numElems_i, 17
  mul $rem_i, $numElems_i, CTXT_WORKERS
  sub $rem_i, $initInfo, $rem_i
  // workers with lower worker id get one extra from the remainder
  // numElems_i % 6
  cmpult $extra_i, $workerPtr, $rem_i
  add $numElems_i, $numElems_i, $extra_i
  shl $worker_x8_i, $workerPtr, 3

  ld32 $chansPerGroup, $mzero, $mvertex_base, VERTEX_OUT_SLICE_SIZE/4
.if \isFloat
  shl $chansPerGroup, $chansPerGroup, 2   // Floats - 4 bytes per
.else
  shl $chansPerGroup, $chansPerGroup, 1   // Halves - 2 bytes per
.endif

init_loop\@:
  mov $outPtr_i, $outPtr

  {
    rpt $numElems_i, (2f-1f)/8-1
    mov $a1, $a0
  }
1:
  {
    st64step $a0:1, $worker_x8_i, $outPtr_i+=, 6
    fnop
  }
2:
  add $worker_x8_i, $worker_x8_i, $chansPerGroup
  brnzdec $numChanGroupsM1, init_loop\@

// ============================
#undef chansPerGroup
#undef numElems_i
#undef rem_i
#undef extra_i
#undef worker_x8_i
#undef outPtr_i
#undef initInfo
#undef outPtr
#undef numChanGroupsM1

#define startPosPtr m7
#define startPos m11
// ============================

  // next we initialise the numRows and startPos registers, do this before
  // loading more state so we can reuse the startPosPtr register afterwards.
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $startPosPtr, $mzero, $mvertex_base, VERTEX_START_POS_PTR_OFFSET/2
  shl $startPosPtr, $startPosPtr, SCALED_PTR32_SHL_BITS
#else
  ld32 $startPosPtr, $mzero, $mvertex_base, VERTEX_START_POS_PTR_OFFSET/4
#endif

  // get the num rows and starting position in the work list for this worker
  ldz16 $numRows, $base, $startPosPtr, $workerPtr
  brnzdec $workerPtr, 1f

  // for the first worker
  zero $startPos
  bri 2f
1:
  // for every other worker, the worker id register is aliased with the startPos
  // register so it is important that we load numRows first.
  ldz16 $startPos, $base, $startPosPtr, $workerPtr
  sub $numRows, $numRows, $startPos
2:
  // numRows may be zero for some of the workers, in those cases there is nothing
  // to do.
  brz $numRows, epilogue\@

  // save startPos to the stack to ease register pressure.
  st32 $startPos, $mzero, $mworker_base, STACK_START_POS_OFFSET

// ============================
#undef startPosPtr
#undef workerPtr
#undef startPos

#define inPtrTemp m3
// ============================
 // load the rest of the vertex state, in preperation for the main loop.
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $inPtrTemp, $mzero, $mvertex_base, VERTEX_IN_PTR_OFFSET/2
  shl $inPtrTemp, $inPtrTemp, SCALED_PTR64_SHL_BITS
#else
  ld32 $inPtrTemp, $mzero, $mvertex_base, VERTEX_IN_PTR_OFFSET/4
#endif
  st32 $inPtrTemp, $mzero, $mworker_base, STACK_IN_PTR_OFFSET
// ============================
#undef inPtrTemp

#define workListBase m9
// ============================

  // sum pooling state includes a scale too.
.if \isMaxPool
  // unpack the rest of the scaled pointers at the top level of the state.
  ld32 $workListBase, $mvertex_base, $mzero, VERTEX_WORK_LIST_BASE_OFFSET/4
.endif
.if !\isMaxPool && \isFloat
  ld32 $a6, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET/4
  {
    // unpack the rest of the scaled pointers at the top level of the state.
    ld32 $workListBase, $mvertex_base, $mzero, VERTEX_WORK_LIST_BASE_OFFSET/4
    mov $a7, $a6
  }
.endif
.if !\isMaxPool && !\isFloat
  ldb16 $a6, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET/2
  {
    // unpack the rest of the scaled pointers at the top level of the state.
    ld32 $workListBase, $mvertex_base, $mzero, VERTEX_WORK_LIST_BASE_OFFSET/4
    mov $a7, $a6
  }
.endif

// ============================
#define fwdActsOutPtr m0
#define fwdActsInPtr m3
// ============================
.if \isBwdPass
  // Load and unpack the activations pointers for the backwards pass
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $fwdActsOutPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_OUT_PTR_OFFSET/2
  ldz16 $fwdActsInPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_IN_PTR_OFFSET/2

  shl $fwdActsOutPtr, $fwdActsOutPtr, SCALED_PTR64_SHL_BITS
  shl $fwdActsInPtr, $fwdActsInPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $fwdActsOutPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_OUT_PTR_OFFSET/4
  ld32 $fwdActsInPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_IN_PTR_OFFSET/4
#endif

  st32 $fwdActsOutPtr, $mzero, $mworker_base, STACK_FWD_ACTS_OUT_PTR_OFFSET
  st32 $fwdActsInPtr, $mzero, $mworker_base, STACK_FWD_ACTS_IN_PTR_OFFSET
.endif
// ============================
#undef fwdActsInPtr
#undef fwdActsOutPtr
// ============================

  // we only want the work list base, not the size.
  shl $workListBase, $workListBase, DELTAN_COUNT_BITS
  shr $workListBase, $workListBase, DELTAN_COUNT_BITS
  st32 $workListBase, $mzero, $mworker_base, STACK_WORK_LIST_BASE_OFFSET

  // load and expand offsetBasePtr and workListDeltaPtr and store them on the
  // stack for later use.
// ============================
#define workListDeltaPtr m6
#define offsetBasePtr m10
// ============================

#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $offsetBasePtr, $mzero, $mvertex_base, VERTEX_OFFSET_BASE_PTR_OFFSET/2
  ldz16 $workListDeltaPtr, $mzero, $mvertex_base, VERTEX_WORK_LIST_DELTA_PTR_OFFSET/2

  shl $offsetBasePtr, $offsetBasePtr, SCALED_PTR32_SHL_BITS
  shl $workListDeltaPtr, $workListDeltaPtr, SCALED_PTR32_SHL_BITS
  // Adding and applying base address costs time but means we don't need to load it
  // inside a loop
  add  $workListDeltaPtr, $workListDeltaPtr, $base
  add  $offsetBasePtr, $offsetBasePtr, $base

#else
  ld32 $offsetBasePtr, $mzero, $mvertex_base, VERTEX_OFFSET_BASE_PTR_OFFSET/4
    ld32 $workListDeltaPtr, $mzero, $mvertex_base, VERTEX_WORK_LIST_DELTA_PTR_OFFSET/4
  // we only want the work list base, not the size.
  shl $workListDeltaPtr, $workListDeltaPtr, DELTAN_COUNT_BITS
  shr $workListDeltaPtr, $workListDeltaPtr, DELTAN_COUNT_BITS
#endif
  st32 $offsetBasePtr, $mzero, $mworker_base, STACK_OFFSET_BASE_PTR_OFFSET
  st32 $workListDeltaPtr, $mzero, $mworker_base, STACK_WORKLIST_DELTA_PTR_OFFSET

// ============================
#undef base
// ============================

  sub $numRows, $numRows, 1

row_loop\@:
  ld32 $offsetBasePtr, $mzero, $mworker_base, STACK_OFFSET_BASE_PTR_OFFSET
  ld32 $workListDeltaPtr, $mzero, $mworker_base, STACK_WORKLIST_DELTA_PTR_OFFSET

  // our loop counter gets trashed by an inner loop so save it now.
  st32 $numRows, $mzero, $mworker_base, STACK_NUM_ROWS_OFFSET

// ============================
#define startPos m11 // also numElementsM1
#define inPos m7
#define workListOffset m5
// ============================
 // startPos aliases to numElementsM1 so it is safe it use it here.
  ld32 $startPos, $mzero, $mworker_base, STACK_START_POS_OFFSET
  add $startPos, $startPos, $numRows

  ld32 $workListOffset, $mzero, $workListDeltaPtr, $startPos

  // before we lose startPos, use it to load the out and in offset bases.
  // these form the base of our eventual indices into the vectors in the
  // innermost loop. times 4 as we want the index 2*pos as a byte offset.

  // dummy load to setup pointer. Since offsetBasePtr is SCALED_PTR32 it is
  // always guaranteed to be a multiple of 4
  ld32 $inPos, $mzero, $offsetBasePtr, $startPos
// ============================
#undef startPos
#undef offsetBasePtr
#undef workListDeltaPtr

#define numWorkItems m10
// ============================
  // store our current intermediate outPos and inPos onto the stack for
  // access each iteration of the next loop (where they are altered for the
  // current work list).
  st32 $inPos, $mzero, $mworker_base, STACK_PACKED_INOUT_POS_OFFSET

  // unpack the work list size and offset now as we are done with the
  // offsetBasePtr (setting numWorkItems trashes it).
  shr $numWorkItems, $workListOffset, DELTAN_OFFSET_BITS
  shl $workListOffset, $workListOffset, DELTAN_LENGTH_BITS
#if COMPACT_VECTOR_TYPES_AVAILABLE
  shr $workListOffset, $workListOffset, DELTAN_LENGTH_BITS
#else
  shr $workListOffset, $workListOffset, DELTAN_LENGTH_BITS-PTR16_SHL_BITS
#endif

// ============================
#define chansPerGroup m2
#define outPos m6
#define scratch m11
// ============================

work_loop\@:
  // chansPerGroup gets trashed by an inner loop so reload now.
  ldz16 $chansPerGroup, $mzero, $mvertex_base, VERTEX_CHANS_PER_GROUP_OFFSET/2
  add $chansPerGroup, $chansPerGroup, 1

  // numWorkItems and outStride registers alias each other so save numWorkItems.
  st32 $numWorkItems, $mzero, $mworker_base, STACK_NUM_WORK_ITEMS_OFFSET

  ld32 $workListBase, $mzero, $mworker_base, STACK_WORK_LIST_BASE_OFFSET

  // modify the outPos and inPos by the offsets contained in the work list.
  ldz16step $outPos, $workListBase, $workListOffset+=, 1
  ldz16 $scratch, $mzero, $mworker_base, 2*STACK_PACKED_INOUT_POS_OFFSET
  add $outPos, $outPos, $scratch
  mul $outPos, $outPos, $chansPerGroup

  ldz16step $inPos, $workListBase, $workListOffset+=, 1
  ldz16 $scratch, $mzero, $mworker_base, 2*STACK_PACKED_INOUT_POS_OFFSET + 1
  add $inPos, $inPos, $scratch
  mul $inPos, $inPos, $chansPerGroup
// ============================
#undef scratch
#undef chansPerGroup
// Define loop variables - need to be the same as those defined before the inner loop macros
// are defined
#define fwdActsOutPtr m0
#define numChanGroupsM1 m1

#define fwdActsInPtr m3
#define inPtr m4
#define outOffset m6
#define inOffset m7
#define outPtr m8
#define inStride m9
#define outStride m10

#define numElementsM1 m11
// ============================

  // load the numElementsM1, which is used for the innermost loop.
  ldz16step $numElementsM1, $workListBase, $workListOffset+=, 1

// ============================
#undef workListOffset
// ============================

  // finally load the strides.
  ld32 $outStride, $mzero, $mvertex_base, VERTEX_OUT_STRIDE_OFFSET/4
  ld32 $inStride, $mzero, $mvertex_base, VERTEX_IN_STRIDE_OFFSET/4

  // reload the loop condition variable.
  ldz16 $numChanGroupsM1, $mzero, $mvertex_base, VERTEX_NUM_CHAN_GROUPS_OFFSET/2

.if \isBwdPass
  // For the backwards pass load and offset the acts pointers
  ld32 $fwdActsOutPtr, $mzero, $mworker_base, STACK_FWD_ACTS_OUT_PTR_OFFSET
  ld32 $fwdActsInPtr, $mzero, $mworker_base, STACK_FWD_ACTS_IN_PTR_OFFSET
    // note fwdActsInPtr is being offset by outPos and vice versa.
    // this is intentional.
  ld64step $azeros, $mzero, $fwdActsInPtr+=, $outPos
  ld64step $azeros, $mzero, $fwdActsOutPtr+=, $inPos
.endif
  // Offset the in, out pointers by their respective worklist positions,
  // The pointer is always 64 bit aligned, so ld64step is safe regardless of the
  // data type.
  ld32 $outPtr, $mzero, $mworker_base, STACK_OUT_PTR_OFFSET
  ld32 $inPtr, $mzero, $mworker_base, STACK_IN_PTR_OFFSET
.if \isFloat
  ld64step $azeros, $mzero, $outPtr+=, $outPos
  ld64step $azeros, $mzero, $inPtr+=, $inPos
.else
  ld64step $azeros, $mzero, $outPtr+=, $outPos
  ld64step $azeros, $mzero, $inPtr+=, $inPos
.endif
 bri   chan_groups_loop_first_time\@
// ============================
#undef outPos
#undef inPos
// ============================

chan_groups_loop\@:
  // Offset to the next slice (Input/output flattened vector)
  // With one loop pass this wouldn't be needed at all, and as this can frequently be
  // the case putting it here saves a few cycles
  ld32 $outOffset, $mzero, $mvertex_base, VERTEX_OUT_SLICE_SIZE/4
  ld32 $inOffset, $mzero, $mvertex_base, VERTEX_IN_SLICE_SIZE/4
.if \isFloat
 ld32step $mzero, $mzero, $outPtr+=, $outOffset
 ld32step $mzero, $mzero, $inPtr+=, $inOffset
  .if \isBwdPass
    // note fwdActsInPtr is being offset by outOffset and vice versa.
    // this is intentional.
   ld32step $mzero, $mzero, $fwdActsInPtr+=, $outOffset
   ld32step $mzero, $mzero, $fwdActsOutPtr+=, $inOffset
  .endif
.else
 ldz16step $mzero, $mzero, $outPtr+=, $outOffset
 ldz16step $mzero, $mzero, $inPtr+=, $inOffset
  .if \isBwdPass
    // note fwdActsInPtr is being offset by outOffset and vice versa.
    // this is intentional.
   ldz16step $mzero, $mzero, $fwdActsInPtr+=, $outOffset
   ldz16step $mzero, $mzero, $fwdActsOutPtr+=, $inOffset
  .endif
.endif

chan_groups_loop_first_time\@:
// ============================
#define chansPerGroup m2
// ============================

  // reload the loop condition variable.
  ldz16 $chansPerGroup, $mzero, $mvertex_base, VERTEX_CHANS_PER_GROUP_OFFSET/2
chans_per_group_loop\@:
  // turn the loop counter into a byte offset that is incremented in the rpt
  // loop for each of the strides.
  shl $outOffset, $chansPerGroup, 3
  mov $inOffset, $outOffset

  // =====================================
  // Call the selected inner loop function
  // =====================================
  \INNER_LOOP_FUNCTION

  brnzdec $chansPerGroup, chans_per_group_loop\@

  brnzdec $numChanGroupsM1, chan_groups_loop\@

  ld32 $numWorkItems, $mzero, $mworker_base, STACK_NUM_WORK_ITEMS_OFFSET
  add $numWorkItems, $numWorkItems, -3
  brnz $numWorkItems, work_loop\@

  ld32 $numRows, $mzero, $mworker_base, STACK_NUM_ROWS_OFFSET
  brnzdec $numRows, row_loop\@

epilogue\@:
  exitz $mzero

.size \symbol, .-\symbol

.endm
// =============================================================================
// Instantiate main vertex macros, which will reference the inner loop macro that
// is passed in

// 2nd parameter is whether or not this is a max pool,
// 3rd parameter is whether or not the out type is float.
// 4th parameter is if it is a backwards pass, (only the case for max pooling).
// 5th parameter is the name of the inner loop macro
DEFINE_VERTEX __runCodelet_popnn__MaxPooling___float 1 1 0 MAX_POOLING_FWD_FLOAT
DEFINE_VERTEX __runCodelet_popnn__MaxPooling___half 1 0 0 MAX_POOLING_FWD_HALF
DEFINE_VERTEX __runCodelet_popnn__MaxPoolingGrad___float 1 1 1 MAX_POOLING_BWD_FLOAT
DEFINE_VERTEX __runCodelet_popnn__MaxPoolingGrad___half 1 0 1 MAX_POOLING_BWD_HALF
DEFINE_VERTEX __runCodelet_popnn__SumPooling___float 0 1 0 SUM_POOLING_FLOAT
DEFINE_VERTEX __runCodelet_popnn__SumPooling___half 0 0 0 SUM_POOLING_HALF

#endif // __IPU__
