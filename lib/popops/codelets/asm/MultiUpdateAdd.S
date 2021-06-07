// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define SUPERVISOR_VERTEX(type) __runCodelet_popops__MultiUpdateAdd___##type##_false
#define WORKER_FUNCTION(type) MultiUpdateAdd___##type##_false
#define STACK_SIZE 0

// vertex states, all offsets are in bytes
#if defined(VECTOR_AVAIL_SCALED_PTR32)
#define VERTEX_STATE_STATE_OFFSET             0
#define VERTEX_STATE_POINTER_OFFSET           4
#define VERTEX_STATE_SIZE_OFFSET              8
#define VERTEX_STATE_SUB_T_OFFSET             12
#define VERTEX_STATE_BASE_T_OFFSET            16
#define VERTEX_STATE_REGION_SIZE_OFFSET       18
#define VERTEX_STATE_BASE_OFFSET_OFFSET       20
#define VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET 24
#define VERTEX_STATE_ELEMENTS_PER_WORKER      28

#else
#define VERTEX_STATE_STATE_OFFSET             0
#define VERTEX_STATE_POINTER_OFFSET           4
#define VERTEX_STATE_SIZE_OFFSET              8
#define VERTEX_STATE_SUB_T_OFFSET             12
#define VERTEX_STATE_BASE_T_OFFSET            16
#define VERTEX_STATE_REGION_SIZE_OFFSET       20
#define VERTEX_STATE_BASE_OFFSET_OFFSET       24
#define VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET 28
#define VERTEX_STATE_ELEMENTS_PER_WORKER      32
#endif

// constants
#define SCALED_PTR32_SHL_BITS 2
#define SIZEOF_HALF 2
#define SIZEOF_FLOAT 4
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// integer variables
#define offsetPtr m0
#define offsetSize m1
#define baseTPtr m2
#define subTPtr m3
#define baseOffset m4
#define numBaseElements m5
#define regionSize m6
#define regionSizeM1 m7
#define regionBytesOffset m8
#define baseIdx m9
#define subTOffset m10
#define maxElementsPerWorker m7
#define workerId m8
#define wkrStart m9
#define wkrEnd m8

// supervisor register aliases
#define sv_workerFunction m1

// floating point variables
#define scale a7

// scratch variables
#define mscratch m11

// Update worker offset and sizes
// Updated/new outputs :
//   - $baseOffset, $wkrStart, $numBaseElements
// Given the baseOffset and numBaseElements for this supervisor vertex, computes
// the baseOffset and numBaseElements for this worker. This it does by using
// the "maxElementsPerWorker" vertex field. Workers have atmost "maxElementsPerWorker"
// elements in the sliced dimension.
.macro UPDATE_OFFSET_AND_SIZES
  ld32 $numBaseElements, $mzero, $mvertex_base, VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET/4
  ld32 $maxElementsPerWorker, $mzero, $mvertex_base, VERTEX_STATE_ELEMENTS_PER_WORKER/4
  get $workerId, $WSR
  and $workerId, $workerId, CSR_W_WSR__CTXTID_M1__MASK
  mul $wkrStart, $workerId, $maxElementsPerWorker
  min $wkrStart, $wkrStart, $numBaseElements
  add $wkrEnd, $wkrStart, $maxElementsPerWorker
  min $wkrEnd, $wkrEnd, $numBaseElements
  sub $numBaseElements, $wkrEnd, $wkrStart
  ld32 $baseOffset, $mzero, $mvertex_base, VERTEX_STATE_BASE_OFFSET_OFFSET/4
  add  $baseOffset, $baseOffset, $wkrStart
.endm


.type WORKER_FUNCTION(float), @function

DEF_STACK_USAGE 0 WORKER_FUNCTION(float)
.section .text.WORKER_FUNCTION(float)
.align 8
#ifndef VECTOR_AVAIL_SCALED_PTR32
  nop // rpt loop aligment
#endif
WORKER_FUNCTION(float):
  // load vertex state, do a size check on offsets
  ld32 $offsetSize, $mzero, $mvertex_base, VERTEX_STATE_SIZE_OFFSET/4
  brz $offsetSize, .Lfloat_epilogue

  UPDATE_OFFSET_AND_SIZES

  ld32 $offsetPtr, $mzero, $mvertex_base, VERTEX_STATE_POINTER_OFFSET/4
  ld32 $subTPtr, $mzero, $mvertex_base, VERTEX_STATE_SUB_T_OFFSET/4
#if defined(VECTOR_AVAIL_SCALED_PTR32)
  ldz16 $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET/2
#else
  ld32  $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET/4
#endif
  ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET/2

  // load scale
  ld32 $mscratch, $mzero, $mvertex_base, VERTEX_STATE_STATE_OFFSET/4
  ld32 $scale, $mzero, $mscratch, 0

#if defined(VECTOR_AVAIL_SCALED_PTR32)
  // expand the SCALED_PTR32 pointer
  shl $baseTPtr, $baseTPtr, SCALED_PTR32_SHL_BITS
  setzi $mscratch, TMEM_REGION0_BASE_ADDR
  add $baseTPtr, $baseTPtr, $mscratch
#endif

  // minus 1 from the region size because we pipeline it.
  sub $regionSizeM1, $regionSize, 1

  // we offset both baseT and subT by regionSize * sizeof(T) so precalculate
  // that outside of the main loop.
  mul $regionBytesOffset, $regionSize, SIZEOF_FLOAT

  // adjust pointers by offset for this worker
  mul $wkrStart, $wkrStart, $regionBytesOffset
  add $baseTPtr, $baseTPtr, $wkrStart
  
  sub $offsetSize, $offsetSize, 1
.Lfloat_offset_loop:
  ld32 $baseIdx, $offsetPtr, $mzero, $offsetSize

  sub $baseIdx, $baseIdx, $baseOffset
  // check baseIdx is within the range of the values in baseT by doing:
  //  if (baseIdx > numBaseElements) continue;
  // note: this overflow relies on baseIdx and numBaseElements being smaller
  // then 2^31.
  cmpult $mscratch, $baseIdx, $numBaseElements
  brz $mscratch, .Lfloat_offset_loop_epilogue

  // correct baseIdx to the current offset and move it onto the correct region
  mul $baseIdx, $baseIdx, $regionBytesOffset

  // move subT on onto the correct region
  mul $subTOffset, $offsetSize, $regionBytesOffset

  // load from the first two pointers.
  ld32step $a1, $subTPtr, $subTOffset+=, 1
  {
    ld32 $a0, $baseTPtr, $baseIdx, 0
    f32mul $a1, $a1, $scale
  }
  {
    rpt $regionSizeM1, (2f-1f)/8-1
    f32add $a3, $a0, $a1
  }
1:
  {
    ld32step $a1, $subTPtr, $subTOffset+=, 1
    f32add $a2, $a0, $a1
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    f32mul $a1, $a1, $scale
  }
  {
    st32step $a2, $baseTPtr, $baseIdx+=, 1
    f32add $a3, $a0, $a1
  }
2:
  // process the final element
  st32 $a3, $baseTPtr, $baseIdx, 0

.Lfloat_offset_loop_epilogue:
  brnzdec $offsetSize, .Lfloat_offset_loop

.Lfloat_epilogue:
  exitz $mzero
.size WORKER_FUNCTION(float), . - WORKER_FUNCTION(float)

.globl WORKER_FUNCTION(half)
.type WORKER_FUNCTION(half), @function

DEF_STACK_USAGE 0 WORKER_FUNCTION(half)
.section .text.WORKER_FUNCTION(half)
.align 8
#ifdef VECTOR_AVAIL_SCALED_PTR32
  nop // rpt loop aligment
#endif
WORKER_FUNCTION(half):
  // load vertex state, do a size check on offsets
  ld32 $offsetSize, $mzero, $mvertex_base, VERTEX_STATE_SIZE_OFFSET/4
  {
    brz $offsetSize, .Lhalf_epilogue
    setzi $a0, ZAACC_BITMASK
  }

  UPDATE_OFFSET_AND_SIZES

  // clear the accumulators incase there is anything nefarious in there for the
  // first call to f16v4mix.
  {
    ld32 $offsetPtr, $mzero, $mvertex_base, VERTEX_STATE_POINTER_OFFSET/4
    uput $FP_CLR, $a0
  }
  ld32 $subTPtr, $mzero, $mvertex_base, VERTEX_STATE_SUB_T_OFFSET/4
#if defined(VECTOR_AVAIL_SCALED_PTR32)
  ldz16 $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET/2
#else
  ld32  $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET/4
#endif
  ld32 $mscratch, $mzero, $mvertex_base, VERTEX_STATE_STATE_OFFSET/4

  // load scale and place {1, scale} into the $TAS CSR
  {
    ldb16 $scale, $mzero, $mscratch, 0
    f16v2exp $a0, $azero
  }
  {
    ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET/2
    sort4x16lo $scale, $a0, $scale
  }

#if defined(VECTOR_AVAIL_SCALED_PTR32)  
  // expand the SCALED_PTR32 pointer
  setzi $mscratch, TMEM_REGION0_BASE_ADDR
  shl $baseTPtr, $baseTPtr, SCALED_PTR32_SHL_BITS
  add $baseTPtr, $baseTPtr, $mscratch
#endif

  // we process 32-bits at a time so halve the region size. the host code must
  // enforce this. finally minus 1 from the result because we pipeline it.
  // also as we don't have an f16v2mix instruction need to zero the odd
  // registers in each pair that we plan to use.
  {
    shr $regionSizeM1, $regionSize, 1
    uput $TAS, $scale
  }
  {
    sub $regionSizeM1, $regionSizeM1, 1
    zero $a1
  }

  // we offset both baseT and subT by regionSize * sizeof(T) so precalculate
  // that outside of the main loop.
  {
    mul $regionBytesOffset, $regionSize, SIZEOF_HALF
    zero $a3
  }

  // adjust pointers by offset for this worker
  mul $wkrStart, $wkrStart, $regionBytesOffset
  add $baseTPtr, $baseTPtr, $wkrStart

  sub $offsetSize, $offsetSize, 1
.Lhalf_offset_loop:
  ld32 $baseIdx, $offsetPtr, $mzero, $offsetSize

  sub $baseIdx, $baseIdx, $baseOffset
  // check baseIdx is within the range of the values in baseT by doing:
  //  if (baseIdx > numBaseElements) continue;
  // note: this overflow relies on baseIdx and numBaseElements being smaller
  // then 2^31.
  cmpult $mscratch, $baseIdx, $numBaseElements
  brz $mscratch, .Lhalf_offset_loop_epilogue

  // correct baseIdx to the current offset and move it onto the correct region
  mul $baseIdx, $baseIdx, $regionBytesOffset

  // move subT on onto the correct region
  mul $subTOffset, $offsetSize, $regionBytesOffset

  // load from the first two pointers.
  ld32 $a0, $baseTPtr, $baseIdx, 0
  ld32step $a2, $subTPtr, $subTOffset+=, 1

  {
    rpt $regionSizeM1, (2f-1f)/8-1
    f16v4mix $azeros, $a0:1, $a2:3
  }
1:
  {
    ld32step $a2, $subTPtr, $subTOffset+=, 1
    f16v4mix $a4:5, $azeros, $azeros
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    fnop
  }
  {
    st32step $a4, $baseTPtr, $baseIdx+=, 1
    f16v4mix $azeros, $a0:1, $a2:3
  }
2:
  // process the final element
  f16v4mix $a4:5, $azeros, $azeros
  st32 $a4, $baseTPtr, $baseIdx, 0

.Lhalf_offset_loop_epilogue:
  brnzdec $offsetSize, .Lhalf_offset_loop

.Lhalf_epilogue:
  exitz $mzero
.size WORKER_FUNCTION(half), . - WORKER_FUNCTION(half)


.macro SUPERVISOR_ENTRY supervisor_name, worker_name
.globl \supervisor_name\()
.type \supervisor_name\(), @function

DEF_STACK_SIZE_OWN STACK_SIZE \supervisor_name\()
.section .text.\supervisor_name\()
.align 4
.supervisor
\supervisor_name\():
  // pointer to the worker code to run
  setzi $sv_workerFunction, \worker_name\()
  runall $sv_workerFunction, $m0, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.size \supervisor_name\(), .-\supervisor_name\()
.endm

// Instantiate supervisor
SUPERVISOR_ENTRY SUPERVISOR_VERTEX(float) WORKER_FUNCTION(float) 
SUPERVISOR_ENTRY SUPERVISOR_VERTEX(half) WORKER_FUNCTION(half) 

#endif // __IPU__
