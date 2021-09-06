// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "MultiSliceUpdateCommon.h.S"

#define WORKER_FUNCTION(type,op,scaled) MultiUpdateOp_##type##_##op##_##scaled
#define STACK_SIZE 0

// vertex states, all offsets are in bytes
#if defined(VECTOR_AVAIL_SCALED_PTR32)
#define VERTEX_STATE_POINTER_OFFSET           0
#define VERTEX_STATE_SIZE_OFFSET              4
#define VERTEX_STATE_SUB_T_OFFSET             8
#define VERTEX_STATE_BASE_T_OFFSET            12
#define VERTEX_STATE_REGION_SIZE_OFFSET       14
#define VERTEX_STATE_BASE_OFFSET_OFFSET       16
#define VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET 20
#define VERTEX_STATE_ELEMENTS_PER_WORKER      24
#define VERTEX_STATE_SCALE_OFFSET             28
#else
#define VERTEX_STATE_POINTER_OFFSET           0
#define VERTEX_STATE_SIZE_OFFSET              4
#define VERTEX_STATE_SUB_T_OFFSET             8
#define VERTEX_STATE_BASE_T_OFFSET            12
#define VERTEX_STATE_REGION_SIZE_OFFSET       16
#define VERTEX_STATE_BASE_OFFSET_OFFSET       20
#define VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET 24
#define VERTEX_STATE_ELEMENTS_PER_WORKER      28
#define VERTEX_STATE_SCALE_OFFSET             32
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

#define SCALED_CODELET_NAME(type) __runCodelet_popops__ScaledMultiUpdateOp___##type##_false_popops__Operation__\OPTYPE\()
#define UNSCALED_CODELET_NAME(type) __runCodelet_popops__MultiUpdateOp___##type##_false_popops__Operation__\OPTYPE\()

.macro CODELET_FUNCTION_FLOAT OPTYPE, SCALED
.ifc \SCALED, true
.type SCALED_CODELET_NAME(float), @function
DEF_STACK_USAGE 0 SCALED_CODELET_NAME(float)
.section .text.SCALED_CODELET_NAME(float)
.globl SCALED_CODELET_NAME(float)
.else 
.type UNSCALED_CODELET_NAME(float), @function
DEF_STACK_USAGE 0 UNSCALED_CODELET_NAME(float)
.section .text.UNSCALED_CODELET_NAME(float)
.globl UNSCALED_CODELET_NAME(float)
.endif

.align 8
#ifndef VECTOR_AVAIL_SCALED_PTR32
  nop // rpt loop aligment
#endif

.ifc \SCALED, true
SCALED_CODELET_NAME(float):
.else
UNSCALED_CODELET_NAME(float):
.endif
  // load vertex state, do a size check on offsets
  ld32 $offsetSize, $mzero, $mvertex_base, VERTEX_STATE_SIZE_OFFSET/4
  brz $offsetSize, .Lfloat_\OPTYPE\()_\SCALED\()_epilogue

  UPDATE_OFFSET_AND_SIZES $numBaseElements $maxElementsPerWorker $workerId $wkrStart $wkrEnd $baseOffset VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET VERTEX_STATE_ELEMENTS_PER_WORKER VERTEX_STATE_BASE_OFFSET_OFFSET

  ld32 $offsetPtr, $mzero, $mvertex_base, VERTEX_STATE_POINTER_OFFSET/4
  ld32 $subTPtr, $mzero, $mvertex_base, VERTEX_STATE_SUB_T_OFFSET/4
#if defined(VECTOR_AVAIL_SCALED_PTR32)
  ldz16 $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET/2
#else
  ld32  $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET/4
#endif
  ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET/2

.ifc \SCALED, true
  // load scale
  ld32 $mscratch, $mzero, $mvertex_base, VERTEX_STATE_SCALE_OFFSET/4
  ld32 $scale, $mzero, $mscratch, 0
.endif

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
.LFloat_\OPTYPE\()_\SCALED\()_offset_loop:
  ld32 $baseIdx, $offsetPtr, $mzero, $offsetSize

  sub $baseIdx, $baseIdx, $baseOffset
  // check baseIdx is within the range of the values in baseT by doing:
  //  if (baseIdx > numBaseElements) continue;
  // note: this overflow relies on baseIdx and numBaseElements being smaller
  // then 2^31.
  cmpult $mscratch, $baseIdx, $numBaseElements
  brz $mscratch, .LFloat_\OPTYPE\()_\SCALED\()_offset_loop_epilogue

  // correct baseIdx to the current offset and move it onto the correct region
  mul $baseIdx, $baseIdx, $regionBytesOffset

  // move subT on onto the correct region
  mul $subTOffset, $offsetSize, $regionBytesOffset

  // load from the first two pointers.
  ld32step $a1, $subTPtr, $subTOffset+=, 1

.ifc \SCALED, true
.ifc \OPTYPE, ADD
  {
    ld32 $a0, $baseTPtr, $baseIdx, 0
    f32mul $a1, $a1, $scale
  }
  {
    rpt $regionSizeM1, (2f-1f)/8-1
    f32add $a2, $a0, $a1
  }
1:
  {
    ld32step $a1, $subTPtr, $subTOffset+=, 1
    fnop
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    f32mul $a1, $a1, $scale
  }
  {
    st32step $a2, $baseTPtr, $baseIdx+=, 1
    f32add $a2, $a0, $a1
  }
2:
.else // OPTYPE
.error "Scaled variant of multiUpdate only supported for operation type ADD" 
.endif

.else // .ifc \SCALED, true
.ifc \OPTYPE, MAX
    {
      ld32 $a0, $baseTPtr, $baseIdx, 0
      fnop
    }
    {
      rpt $regionSizeM1, (2f-1f)/8-1
      f32max $a2, $a0, $a1
    }
1:
  {
    ld32step $a1, $subTPtr, $subTOffset+=, 1
    fnop
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    fnop
  }
  {
    st32step $a2, $baseTPtr, $baseIdx+=, 1
    f32max $a2, $a0, $a1
  }
2:
.else
.error "Non scaled variant of multiUpdate only supported for operation type MAX"
.endif // .if \OPTYPE, MAX
.endif
  // process the final element
  st32 $a2, $baseTPtr, $baseIdx, 0

.LFloat_\OPTYPE\()_\SCALED\()_offset_loop_epilogue:
  brnzdec $offsetSize, .LFloat_\OPTYPE\()_\SCALED\()_offset_loop

.Lfloat_\OPTYPE\()_\SCALED\()_epilogue:
  exitz $mzero
.ifc \SCALED, true
.size SCALED_CODELET_NAME(float), . -  SCALED_CODELET_NAME(float)
.else
.size  UNSCALED_CODELET_NAME(float), . - UNSCALED_CODELET_NAME(float)
.endif
.endm

.macro CODELET_FUNCTION_HALF OPTYPE, SCALED

.ifc \SCALED, true
.type SCALED_CODELET_NAME(half), @function
DEF_STACK_USAGE 0 SCALED_CODELET_NAME(half)
.section .text.SCALED_CODELET_NAME(half)
.globl SCALED_CODELET_NAME(half)
.else 
.type UNSCALED_CODELET_NAME(half), @function
DEF_STACK_USAGE 0 UNSCALED_CODELET_NAME(half)
.section .text.UNSCALED_CODELET_NAME(half)
.globl UNSCALED_CODELET_NAME(half)
.endif
.align 8
#ifdef VECTOR_AVAIL_SCALED_PTR32
  nop // rpt loop aligment
#endif
.ifc \SCALED, true
SCALED_CODELET_NAME(half):
.else
UNSCALED_CODELET_NAME(half):
.endif
  // load vertex state, do a size check on offsets
  ld32 $offsetSize, $mzero, $mvertex_base, VERTEX_STATE_SIZE_OFFSET/4
  {
    brz $offsetSize, .Lhalf_\OPTYPE\()_\SCALED\()_epilogue
    setzi $a0, ZAACC_BITMASK
  }

  UPDATE_OFFSET_AND_SIZES $numBaseElements $maxElementsPerWorker $workerId $wkrStart $wkrEnd $baseOffset VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET VERTEX_STATE_ELEMENTS_PER_WORKER VERTEX_STATE_BASE_OFFSET_OFFSET

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

.ifc \SCALED, true
   ld32 $mscratch, $mzero, $mvertex_base, VERTEX_STATE_SCALE_OFFSET/4

  // load scale and place {1, scale} into the $TAS CSR
  {
    ldb16 $scale, $mzero, $mscratch, 0
    f16v2exp $a0, $azero
  }
  {
    ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET/2
    sort4x16lo $scale, $a0, $scale
  }
  // we process 32-bits at a time so halve the region size. the host code must
  // enforce this. finally minus 1 from the result because we pipeline it.
  // also as we don't have an f16v2mix instruction need to zero the odd
  // registers in each pair that we plan to use.
  {
    shr $regionSizeM1, $regionSize, 1
    uput $TAS, $scale
  }
.else
  ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET/2
  {
    shr $regionSizeM1, $regionSize, 1
    fnop
  }
.endif

#if defined(VECTOR_AVAIL_SCALED_PTR32)  
  // expand the SCALED_PTR32 pointer
  setzi $mscratch, TMEM_REGION0_BASE_ADDR
  shl $baseTPtr, $baseTPtr, SCALED_PTR32_SHL_BITS
  add $baseTPtr, $baseTPtr, $mscratch
#endif

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
.Lhalf_\OPTYPE\()_\SCALED\()_offset_loop:
  ld32 $baseIdx, $offsetPtr, $mzero, $offsetSize

  sub $baseIdx, $baseIdx, $baseOffset
  // check baseIdx is within the range of the values in baseT by doing:
  //  if (baseIdx > numBaseElements) continue;
  // note: this overflow relies on baseIdx and numBaseElements being smaller
  // then 2^31.
  cmpult $mscratch, $baseIdx, $numBaseElements
  brz $mscratch, .Lhalf_\OPTYPE\()_\SCALED\()_offset_loop_epilogue

  // correct baseIdx to the current offset and move it onto the correct region
  mul $baseIdx, $baseIdx, $regionBytesOffset

  // move subT on onto the correct region
  mul $subTOffset, $offsetSize, $regionBytesOffset

  // load from the first two pointers.
  ld32 $a0, $baseTPtr, $baseIdx, 0
  ld32step $a2, $subTPtr, $subTOffset+=, 1
.ifc \SCALED, true
.ifc \OPTYPE, ADD
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
.else // .ifc \OPTYPE, ADD
.error "Scaled variant of multiUpdateOp is only supported for operand type ADD"
.endif // .ifc \OPTYPE, ADD

.else // .ifc \SCALED, true

.ifc \OPTYPE, MAX

  {
    rpt $regionSizeM1, (2f-1f)/8-1
    f16v2max $a4, $a0, $a2
  }
1:
  {
    ld32step $a2, $subTPtr, $subTOffset+=, 1
    fnop
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    fnop
  }
  {
    st32step $a4, $baseTPtr, $baseIdx+=, 1
    f16v2max $a4, $a0, $a2
  }
2:

.else // .ifc \OPTYPE, MAX
.error "Non-scaled variant of multiUpdateOp is only supported for operand type MAX"
.endif // .ifc \OPTYPE, MAX

.endif // .ifc \SCALED, true

  st32 $a4, $baseTPtr, $baseIdx, 0

.Lhalf_\OPTYPE\()_\SCALED\()_offset_loop_epilogue:
  brnzdec $offsetSize, .Lhalf_\OPTYPE\()_\SCALED\()_offset_loop

.Lhalf_\OPTYPE\()_\SCALED\()_epilogue:
  exitz $mzero
.ifc \SCALED, true
.size SCALED_CODELET_NAME(half), . - SCALED_CODELET_NAME(half)
.else
.size UNSCALED_CODELET_NAME(half), . -  UNSCALED_CODELET_NAME(half)
.endif
.endm

// Instantiate workers
CODELET_FUNCTION_FLOAT ADD true
CODELET_FUNCTION_HALF ADD true
CODELET_FUNCTION_FLOAT MAX false
CODELET_FUNCTION_HALF MAX false


#endif // __IPU__
