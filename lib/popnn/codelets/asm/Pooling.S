// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
#define COMPACT_VECTOR_TYPES_AVAILABLE 1
#else
#define COMPACT_VECTOR_TYPES_AVAILABLE 0
#endif

#if COMPACT_VECTOR_TYPES_AVAILABLE
// vertex state, all offsets are 16-bit. all of the output and inputs (inc.
// the activations) are aligned to 64-bit and the size is guaranteed to align
// to 32-bits (so, there will be an even amount of halves).
#define VERTEX_OUT_PTR_PTR_OFFSET 0
#define VERTEX_IN_PTR_PTR_OFFSET 2
#define VERTEX_START_POS_PTR_OFFSET 4
#define VERTEX_OFFSET_BASE_PTR_OFFSET 6
#define VERTEX_WORK_LIST_BASE_OFFSET 8
#define VERTEX_WORK_LIST_DELTA_PTR_OFFSET 12
#define VERTEX_INIT_INFO_OFFSET 14
#define VERTEX_NUM_CHAN_GROUPS_OFFSET 16
#define VERTEX_CHANS_PER_GROUP_OFFSET 18
#define VERTEX_IN_STRIDE_OFFSET 20
#define VERTEX_OUT_STRIDE_OFFSET 24
#define VERTEX_SCALE_OFFSET 28 // SumPooling only
#define VERTEX_FWD_ACTS_IN_PTR_PTR_OFFSET 28 // MaxPoolingGrad only
#define VERTEX_FWD_ACTS_OUT_PTR_PTR_OFFSET 30 // MaxPoolingGrad only
#else
#define VERTEX_OUT_PTR_PTR_OFFSET 0
#define VERTEX_IN_PTR_PTR_OFFSET 4
#define VERTEX_START_POS_PTR_OFFSET 8
#define VERTEX_OFFSET_BASE_PTR_OFFSET 12
#define VERTEX_WORK_LIST_BASE_OFFSET 16
#define VERTEX_WORK_LIST_DELTA_PTR_OFFSET 20
#define VERTEX_INIT_INFO_OFFSET 24
#define VERTEX_NUM_CHAN_GROUPS_OFFSET 26
#define VERTEX_CHANS_PER_GROUP_OFFSET 28
#define VERTEX_IN_STRIDE_OFFSET 32
#define VERTEX_OUT_STRIDE_OFFSET 36
#define VERTEX_SCALE_OFFSET 40 // SumPooling only
#define VERTEX_FWD_ACTS_IN_PTR_PTR_OFFSET 40 // MaxPoolingGrad only
#define VERTEX_FWD_ACTS_OUT_PTR_PTR_OFFSET 44 // MaxPoolingGrad only
#endif

// stack state, all offsets are 32-bit
#define STACK_NUM_ROWS_OFFSET 0
#define STACK_NUM_WORK_ITEMS_OFFSET 1
#define STACK_START_POS_OFFSET 2
#define STACK_WORK_LIST_OFFSET_OFFSET 3
#define STACK_PACKED_INOUT_POS_OFFSET 4
#define STACK_IN_POS_OFFSET 5
#define STACK_OUT_POS_OFFSET 6
#define STACK_OUT_PTR_PTR_OFFSET 7
#define STACK_IN_PTR_PTR_OFFSET 8
#define STACK_FWD_ACTS_IN_PTR_PTR_OFFSET 9
#define STACK_FWD_ACTS_OUT_PTR_PTR_OFFSET 10

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
#define outPtrPtr m0
#define fwdActsOutPtrPtr m0 // same as outPtrPtr
#define fwdActsOutPtr m0 // same as outPtrPtr
#define numChanGroupsM1 m1
#define chansPerGroup m2
#define inPtrPtr m3
#define fwdActsInPtrPtr m3 // same as inPtrPtr
#define fwdActsInPtr m3 // same as inPtrPtr
#define inPtr m4
#if COMPACT_VECTOR_TYPES_AVAILABLE
#define base m4 // also inPtr
#else
#define base mzero
#endif
#define workListBase m5
#define initInfo m5 // also workListBase
#define outPos m6
#define workListDeltaPtr m6 // also outPos and outOffset
#define outOffset m6 // also outPos and workListDeltaPtr
#define inPos m7
#define startPosPtr m7 // also inPos and inOffset
#define inOffset m7 // also inPos and startPosPtr
#define numRows m8
#define outPtr m8 // also numRows
#define workListOffset m9
#define inStride m9 // also workListOffset
#define numWorkItems m10
#define offsetBasePtr m10 // also numWorkItems and outStride
#define outStride m10 // also numWorkItems and offsetBasePtr
#define numElementsM1 m11
#define startPos m11 // also numElementsM1
#define workerPtr m11 // also startPos and numElementsM1

// variables used for initialising the output buffers
#define numElems_i  m9
#define rem_i       m10
#define extra_i     m10
#define worker_x8_i m5

.macro DEFINE_VERTEX symbol isMaxPool isFloat isBwdPass

.globl \symbol
.type \symbol @function

.section .text.\symbol
.align 4
.supervisor
\symbol:
  setzi $m1, kernel\@ // 6 cycles
  runall $m1, $m0, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.align 8
.worker
kernel\@:
  // load vertex state needed for initialisation first.
  ldz16 $initInfo, $mzero, $mvertex_base, VERTEX_INIT_INFO_OFFSET/2
  ldz16 $numChanGroupsM1, $mzero, $mvertex_base, VERTEX_NUM_CHAN_GROUPS_OFFSET/2
  ldz16 $chansPerGroup, $mzero, $mvertex_base, VERTEX_CHANS_PER_GROUP_OFFSET/2

  // unpack the scaled output pointer
#if COMPACT_VECTOR_TYPES_AVAILABLE
  setzi $base, TMEM_REGION0_BASE_ADDR
  ldz16 $outPtrPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_PTR_OFFSET/2
  shl $outPtrPtr, $outPtrPtr, SCALED_PTR32_SHL_BITS
#else
  ld32 $outPtrPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_PTR_OFFSET/4
#endif

  // most of the state (including the offsets inside the work list) are scaled
  // by chansPerGroup to save memory, expand init info now, we will scale up
  // the two strides when we load them after output initialisation.
  mul $initInfo, $initInfo, $chansPerGroup

  // identity for max pool float is -inf, half is -65504 and all sum pool and
  // max pool grad is 0. split the ldconst up to bundle, also make sure that
  // each branch uses an even number of instructions to ensure instruction
  // alignment of the rpt below.
.if \isMaxPool
.if \isBwdPass
  {
    get $workerPtr, $WSR
    zero $a0
  }
  {
    and $workerPtr, $workerPtr, CSR_W_WSR__CTXTID_M1__MASK
    fnop
  }
.else
.if \isFloat
  {
    get $workerPtr, $WSR
    or $a0, $azero, 0xff800000 & ~LDCONST_MASK
  }
  {
    and $workerPtr, $workerPtr, CSR_W_WSR__CTXTID_M1__MASK
    fnop
  }
.else
  // [-65504, -65504]
  {
    get $workerPtr, $WSR
    setzi $a0, 0xfbfffbff & LDCONST_MASK
  }
  {
    and $workerPtr, $workerPtr, CSR_W_WSR__CTXTID_M1__MASK
    or $a0, $a0, 0xfbfffbff & ~LDCONST_MASK
  }
.endif
.endif
.else
  {
    get $workerPtr, $WSR
    zero $a0
  }
  {
    and $workerPtr, $workerPtr, CSR_W_WSR__CTXTID_M1__MASK
    fnop
  }
.endif

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

init_loop\@:
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $outPtr, $base, $outPtrPtr, $numChanGroupsM1
  shl $outPtr, $outPtr, SCALED_PTR64_SHL_BITS
#else
  {
    ld32 $outPtr, $base, $outPtrPtr, $numChanGroupsM1
    fnop // rpt alignment
  }
#endif
  {
    rpt $numElems_i, (2f-1f)/8-1
    mov $a1, $a0
  }
1:
  {
    st64step $a0:1, $worker_x8_i, $outPtr+=, 6
    fnop
  }
2:
  brnzdec $numChanGroupsM1, init_loop\@

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

  // load the rest of the vertex state, in preperation for the main loop.
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $inPtrPtr, $mzero, $mvertex_base, VERTEX_IN_PTR_PTR_OFFSET/2
  shl $inPtrPtr, $inPtrPtr, SCALED_PTR32_SHL_BITS
#else
  ld32 $inPtrPtr, $mzero, $mvertex_base, VERTEX_IN_PTR_PTR_OFFSET/4
#endif

  // sum pooling state includes a scale too.
.if !\isMaxPool
.if \isFloat
  ld32 $a6, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET/4
  {
    // unpack the rest of the scaled pointers at the top level of the state.
    ld32 $workListBase, $mvertex_base, $mzero, VERTEX_WORK_LIST_BASE_OFFSET/4
    mov $a7, $a6
  }
.else
  ldb16 $a6, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET/2
  {
    // unpack the rest of the scaled pointers at the top level of the state.
    ld32 $workListBase, $mvertex_base, $mzero, VERTEX_WORK_LIST_BASE_OFFSET/4
    mov $a7, $a6
  }
.endif
.else
  // unpack the rest of the scaled pointers at the top level of the state.
  ld32 $workListBase, $mvertex_base, $mzero, VERTEX_WORK_LIST_BASE_OFFSET/4
.endif

.if \isBwdPass
  // save the two outer pointers on the stack and then load and unpack the
  // activations pointers for the backwards pass (they share the same registers)
  st32 $outPtrPtr, $mzero, $mworker_base, STACK_OUT_PTR_PTR_OFFSET
  st32 $inPtrPtr, $mzero, $mworker_base, STACK_IN_PTR_PTR_OFFSET

#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $fwdActsOutPtrPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_OUT_PTR_PTR_OFFSET/2
  ldz16 $fwdActsInPtrPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_IN_PTR_PTR_OFFSET/2

  shl $fwdActsOutPtrPtr, $fwdActsOutPtrPtr, SCALED_PTR32_SHL_BITS
  shl $fwdActsInPtrPtr, $fwdActsInPtrPtr, SCALED_PTR32_SHL_BITS
#else
  ld32 $fwdActsOutPtrPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_OUT_PTR_PTR_OFFSET/4
  ld32 $fwdActsInPtrPtr, $mzero, $mvertex_base, VERTEX_FWD_ACTS_IN_PTR_PTR_OFFSET/4
#endif

  st32 $fwdActsOutPtrPtr, $mzero, $mworker_base, STACK_FWD_ACTS_OUT_PTR_PTR_OFFSET
  st32 $fwdActsInPtrPtr, $mzero, $mworker_base, STACK_FWD_ACTS_IN_PTR_PTR_OFFSET
.endif

  // we only want the work list base, not the size.
  shl $workListBase, $workListBase, DELTAN_COUNT_BITS
  shr $workListBase, $workListBase, DELTAN_COUNT_BITS

  sub $numRows, $numRows, 1

row_loop\@:
  // load and expand offsetBasePtr and workListDeltaPtr each row iteration.
  // TODO - T15184: Expand pointers once and store them on the stack for later
  // use. Note that this routine already uses up most of the available scratch
  // space.
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $offsetBasePtr, $mzero, $mvertex_base, VERTEX_OFFSET_BASE_PTR_OFFSET/2
  ldz16 $workListDeltaPtr, $mzero, $mvertex_base, VERTEX_WORK_LIST_DELTA_PTR_OFFSET/2

  shl $offsetBasePtr, $offsetBasePtr, SCALED_PTR32_SHL_BITS
  shl $workListDeltaPtr, $workListDeltaPtr, SCALED_PTR32_SHL_BITS

  setzi $base, TMEM_REGION0_BASE_ADDR
#else
  ld32 $offsetBasePtr, $mzero, $mvertex_base, VERTEX_OFFSET_BASE_PTR_OFFSET/4
  {
    ld32 $workListDeltaPtr, $mzero, $mvertex_base, VERTEX_WORK_LIST_DELTA_PTR_OFFSET/4
    fnop // rpt alignment
  }
  // we only want the work list base, not the size.
  shl $workListDeltaPtr, $workListDeltaPtr, DELTAN_COUNT_BITS
  shr $workListDeltaPtr, $workListDeltaPtr, DELTAN_COUNT_BITS
#endif

  // our loop counter gets trashed by an inner loop so save it now.
  st32 $numRows, $mzero, $mworker_base, STACK_NUM_ROWS_OFFSET

  // startPos aliases to numElementsM1 so it is safe it use it here.
  ld32 $startPos, $mzero, $mworker_base, STACK_START_POS_OFFSET
  add $startPos, $startPos, $numRows

  ld32 $workListOffset, $base, $workListDeltaPtr, $startPos

  // before we lose startPos, use it to load the out and in offset bases.
  // these form the base of our eventual indices into the vectors in the
  // innermost loop. times 4 as we want the index 2*pos as a byte offset.

  // dummy load to setup pointer. Since offsetBasePtr is SCALED_PTR32 it is
  // always guaranteed to be a multiple of 4
  ld32 $inPos, $base, $offsetBasePtr, $startPos

  // store are current intermediate outPos and inPos onto the stack for
  // access each iteration of the next loop (where they are altered for the
  // current work list).
  st32 $inPos, $mzero, $mworker_base, STACK_PACKED_INOUT_POS_OFFSET

  // unpack the work list size and offset now as we are done with the
  // offsetBasePtr (setting numWorkItems trashes it).
#if COMPACT_VECTOR_TYPES_AVAILABLE
shr $numWorkItems, $workListOffset, DELTAN_OFFSET_BITS
shl $workListOffset, $workListOffset, DELTAN_LENGTH_BITS
shr $workListOffset, $workListOffset, DELTAN_LENGTH_BITS
#else
  shr $numWorkItems, $workListOffset, DELTAN_OFFSET_BITS
  shl $workListOffset, $workListOffset, DELTAN_LENGTH_BITS
  shr $workListOffset, $workListOffset, DELTAN_LENGTH_BITS-PTR16_SHL_BITS
#endif
  // the workListOffset and inStride registers alias each other so save the
  // offset to the stack.
  st32 $workListOffset, $mzero, $mworker_base, STACK_WORK_LIST_OFFSET_OFFSET

work_loop\@:
  // chansPerGroup gets trashed by an inner loop so reload now.
  ldz16 $chansPerGroup, $mzero, $mvertex_base, VERTEX_CHANS_PER_GROUP_OFFSET/2

  // numWorkItems and outStride registers alias each other so save numWorkItems.
  st32 $numWorkItems, $mzero, $mworker_base, STACK_NUM_WORK_ITEMS_OFFSET

  ld32 $workListOffset, $mzero, $mworker_base, STACK_WORK_LIST_OFFSET_OFFSET

  // modify the outPos and inPos by the offsets contained in the work list.
  // use numElementsM1 as a scratch register.
  ldz16step $outPos, $workListBase, $workListOffset+=, 1
  ldz16 $numElementsM1, $mzero, $mworker_base, 2*STACK_PACKED_INOUT_POS_OFFSET
  add $outPos, $outPos, $numElementsM1
  mul $outPos, $outPos, $chansPerGroup

  ldz16step $inPos, $workListBase, $workListOffset+=, 1
  ldz16 $numElementsM1, $mzero, $mworker_base, 2*STACK_PACKED_INOUT_POS_OFFSET + 1
  add $inPos, $inPos, $numElementsM1
  mul $inPos, $inPos, $chansPerGroup

.if !\isFloat
  shl $inPos, $inPos, 2
  shl $outPos, $outPos, 2
.else
  shl $inPos, $inPos, 1
  shl $outPos, $outPos, 1
.endif

  // save the final positions onto the stack so we can reuse the registers.
  st32 $outPos, $mzero, $mworker_base, STACK_OUT_POS_OFFSET
  st32 $inPos, $mzero, $mworker_base, STACK_IN_POS_OFFSET

  // load the numElementsM1, which is used for the innermost loop.
  ldz16step $numElementsM1, $workListBase, $workListOffset+=, 1

  // we have modified workListOffset so save it again
  st32 $workListOffset, $mzero, $mworker_base, STACK_WORK_LIST_OFFSET_OFFSET

  // finally load the strides.
  ld32 $outStride, $mzero, $mvertex_base, VERTEX_OUT_STRIDE_OFFSET/4
  ld32 $inStride, $mzero, $mvertex_base, VERTEX_IN_STRIDE_OFFSET/4

  // reload the loop condition variable.
  ldz16 $numChanGroupsM1, $mzero, $mvertex_base, VERTEX_NUM_CHAN_GROUPS_OFFSET/2
chan_groups_loop\@:
  ld32 $outPos, $mzero, $mworker_base, STACK_OUT_POS_OFFSET
  ld32 $inPos, $mzero, $mworker_base, STACK_IN_POS_OFFSET

.if \isBwdPass
  // For the backwards pass the outer pointer registers are reused so we must
  // load them off the stack.
  ld32 $outPtrPtr, $mzero, $mworker_base, STACK_OUT_PTR_PTR_OFFSET
  ld32 $inPtrPtr, $mzero, $mworker_base, STACK_IN_PTR_PTR_OFFSET
.endif

  // at this point we should load the correct output and input pointers. use
  // chansPerGroup as a scratch register.
#if COMPACT_VECTOR_TYPES_AVAILABLE
  setzi $chansPerGroup, TMEM_REGION0_BASE_ADDR
  ldz16 $outPtr, $chansPerGroup, $outPtrPtr, $numChanGroupsM1
  shl $outPtr, $outPtr, SCALED_PTR64_SHL_BITS
  ldz16 $inPtr, $chansPerGroup, $inPtrPtr, $numChanGroupsM1
  shl $inPtr, $inPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $outPtr, $base, $outPtrPtr, $numChanGroupsM1
  {
    ld32 $inPtr, $base, $inPtrPtr, $numChanGroupsM1
    fnop
  }
#endif

.if \isBwdPass
  // The inner fwdActs pointers use the same registers as their outer
  // counterparts.
  ld32 $fwdActsOutPtrPtr, $mzero, $mworker_base, STACK_FWD_ACTS_OUT_PTR_PTR_OFFSET
  ld32 $fwdActsInPtrPtr, $mzero, $mworker_base, STACK_FWD_ACTS_IN_PTR_PTR_OFFSET

#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $fwdActsOutPtr, $chansPerGroup, $fwdActsOutPtrPtr, $numChanGroupsM1
  shl $fwdActsOutPtr, $fwdActsOutPtr, SCALED_PTR64_SHL_BITS
  ldz16 $fwdActsInPtr, $chansPerGroup, $fwdActsInPtrPtr, $numChanGroupsM1
  shl $fwdActsInPtr, $fwdActsInPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $fwdActsOutPtr, $base, $fwdActsOutPtrPtr, $numChanGroupsM1
  ld32 $fwdActsInPtr, $base, $fwdActsInPtrPtr, $numChanGroupsM1
#endif
.endif

  // move the two pointers on by the current positions with a dummy load.
.if \isFloat
  ld32step $mzero, $mzero, $outPtr+=, $outPos
  ld32step $mzero, $mzero, $inPtr+=, $inPos

.if \isBwdPass
  // note fwdActsInPtr is being offset by outPos and vice versa.
  // this is intentional.
  ld32step $mzero, $mzero, $fwdActsInPtr+=, $outPos
  ld32step $mzero, $mzero, $fwdActsOutPtr+=, $inPos
.endif
.else
  ldz16step $mzero, $mzero, $outPtr+=, $outPos
  ldz16step $mzero, $mzero, $inPtr+=, $inPos

.if \isBwdPass
  // note fwdActsInPtr is being offset by outPos and vice versa.
  // this is intentional.
  ldz16step $mzero, $mzero, $fwdActsInPtr+=, $outPos
  ldz16step $mzero, $mzero, $fwdActsOutPtr+=, $inPos
.endif
.endif

  // reload the loop condition variable.
  ldz16 $chansPerGroup, $mzero, $mvertex_base, VERTEX_CHANS_PER_GROUP_OFFSET/2
  sub $chansPerGroup, $chansPerGroup, 1
chans_per_group_loop\@:
  // turn the loop counter into a byte offset that is incremented in the rpt
  // loop for each of the strides.
  shl $outOffset, $chansPerGroup, 3
  mov $inOffset, $outOffset

.if \isMaxPool
.if \isFloat
.if \isBwdPass
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
.else // if \isBwdPass
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
.endif // if \isBwdPass
.else // if \isFloat
  // isHalf
.if \isBwdPass
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
.else // if \isBwdPass
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
.endif // if \isBwdPass
.endif // if \isFloat
.else
  // sum pooling
.if \isFloat
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
.else // if \isFloat
  // isHalf
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
.endif // if \isFloat
.endif // if \isMaxPool

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

// 2nd parameter is whether or not this is a max pool, 3rd parameter is whether
// or not the out type is float. 4th parameter is if it is a backwards pass,
// note this is only the case for max pooling.
DEFINE_VERTEX __runCodelet_popnn__MaxPooling___float 1 1 0
DEFINE_VERTEX __runCodelet_popnn__MaxPooling___half 1 0 0
DEFINE_VERTEX __runCodelet_popnn__MaxPoolingGrad___float 1 1 1
DEFINE_VERTEX __runCodelet_popnn__MaxPoolingGrad___half 1 0 1
DEFINE_VERTEX __runCodelet_popnn__SumPooling___float 0 1 0
DEFINE_VERTEX __runCodelet_popnn__SumPooling___half 0 0 0

#endif // __IPU__
