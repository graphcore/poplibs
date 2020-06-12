// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty

// vertex state (offsets in bytes)

#if defined(VECTOR_AVAIL_SCALED_PTR64)
  //
  // Vertex state where Scale type is half or Scale pointer occupies 16 bits
  //
  #define VERTEX_DATA_A_OFFSET 0
  #define VERTEX_PACKED_COUNT_OFFSET 2
  #define VERTEX_DATA_B_OFFSET 4
  #define VERTEX_SCALE_OFFSET 6
  #define VERTEX_SCALE_B_CONST_OFFSET 8

  //
  // Vertex state where Scale type is float or Scale pointer occupies 32 bits
  //
  #define VERTEX_DATA_A_OFFSET 0
  #define VERTEX_PACKED_COUNT_OFFSET 2
  #define VERTEX_DATA_B_OFFSET 4
  #define VERTEX_SCALE_B_TENSOR_OFFSET 8
  // scale variable offset (float) option.
  #define VERTEX_SCALE_OFFSET_FLOAT_CONST 8
  #define VERTEX_SCALE_B_OFFSET_FLOAT_CONST 12
#else
  //
  // Vertex state where Scale type is half and occupies 16 bits
  //
  #define VERTEX_DATA_A_OFFSET 0
  #define VERTEX_PACKED_COUNT_OFFSET 4
  #define VERTEX_DATA_B_OFFSET 8
  #define VERTEX_SCALE_OFFSET 12
  #define VERTEX_SCALE_B_CONST_OFFSET 14

  //
  // Vertex state where Scale type is float or Scale pointer occupies 32 bits
  //
  #define VERTEX_DATA_A_OFFSET 0
  #define VERTEX_PACKED_COUNT_OFFSET 4
  #define VERTEX_DATA_B_OFFSET 8
  #define VERTEX_SCALE_OFFSET 12
  #define VERTEX_SCALE_B_TENSOR_OFFSET 16
  // scale variable offset (float) option.
  #define VERTEX_SCALE_OFFSET_FLOAT_CONST 12
  #define VERTEX_SCALE_B_OFFSET_FLOAT_CONST 16
#endif

// create a new vertex state on the supervisor stack that has the input values
// preprocessed for all of the workers to use.
#define SV_STATE_DATA_OFFSET 0
#define SV_STATE_COUNT_OFFSET 4
#define SV_STATE_REMM1_OFFSET 8
#define SV_STATE_FINAL_OFFSET 12
#define SV_STATE_SCALES_OFFSET 16
#define SV_STATE_DATA_B_OFFSET 20
#define SV_STATE_MEM_CONSTRAINTS 24 // Constraints unused here, but space allowed anyhow in shared code
#define SV_STATE_SCALEB_OFFSET   28 // Unused here, space needed in shared code

#define SV_STATE_SIZE 32

// total space required on the stack
#define STACK_SIZE (SV_STATE_SIZE)

// constants
#define SCALED_PTR32_SHL_BITS 2
#define SCALED_PTR64_SHL_BITS 3

// to avoid sub-word writes we must make sure that each worker processes
// a number of elements so that we fall exactly into a 64-bit load. for floats
// this is 8/sizeof(float) = 2 and 8/sizeof(half) = 4
#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// supervisor variables
#define vertexPtr m0
#define countD2 m1
#define final m2

#define remM1 m3

#define mscratch m4
#define mscratch2 m5

#define mworkerFunction m6

#define log2AtomSize m7
#define atomSizeMask m8
#define mscratch3 m6


//******************************************************************************
// Float entry points - macro
//******************************************************************************

.macro SUPERVISOR_ENTRY LABEL FLOAT_SCALE TENSOR WORKER_FUNCTION
.globl VERTEX(\LABEL)
.type VERTEX(\LABEL), @function

DEF_STACK_SIZE_OWN STACK_SIZE VERTEX(\LABEL)
.section .text.VERTEX(\LABEL)
.align 4
.supervisor
VERTEX(\LABEL):
.if \TENSOR
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $mscratch, $mscratch, SCALED_PTR64_SHL_BITS
#else
  ld32  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/4
#endif
.else
.if \FLOAT_SCALE
  ld32  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET_FLOAT_CONST/4
.else
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
.endif
.endif
  // keeping this before the branch means it doesn't cause a stall later
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(\WORKER_FUNCTION\()).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
.if \TENSOR
  // load factor using its pointer - here to avoid pipeline hit
.if \FLOAT_SCALE
  ld32  $mscratch, $mzero, $mscratch, 0
.else
  ldz16 $mscratch, $mzero, $mscratch, 0
.endif
.endif
  // Branch, divide work, store vertex state for the worker and then call the workers
  // workers - entering the code below
  bri   VERTEX(supervisor)
.size VERTEX(\LABEL), .-VERTEX(\LABEL)
.endm

// Create the entry points

SUPERVISOR_ENTRY float_half_half_false_false 0 1 half_scale
SUPERVISOR_ENTRY float_half_half_true_false 0 0 half_scale

SUPERVISOR_ENTRY float_half_float_false_false 1 1 float_scale
SUPERVISOR_ENTRY float_half_float_true_false 1 0 float_scale


// clear supervisor variables
#undef vertexPtr
#undef dataPtr
#undef countD2
#undef final
#undef remM1
#undef mscratch
#undef mscratch2

//******************************************************************************
// worker variables

// integer variables
#define dataPtr m1
#define remM1 m2
#define final m3
#define countD2 m4
#define dataBPtr m5
#define dataStore m6
#define workerIdM1 m8

#define data a0:1
#define datai0 a0
#define datai1 a1
#define dataBHiLo a4:7
#define dataB a4:5
#define dataBHi a6:7
#define dataBi0 a4
#define dataBi1 a5
#define result a2:3
#define k a6

// scratch variables
#define mscratch m10
#define ascratch a7

.type VERTEX(float_half).kernel, @function

DEF_STACK_USAGE 0 VERTEX(float_half).kernel
.section .text.VERTEX(float_half).kernel, FUNCTION_IS_WORKER
.align 8
VERTEX(float_half).kernel:
.worker
  nop   //rpt alignment
VERTEX(half_scale).kernel:
  ld32 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
  {bri 1f
   f16tof32 $k,$k}
VERTEX(float_scale).kernel:
  ld32 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
1:
  // load vertex state
  ld32 $countD2, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
  ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4
  {
    ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4
    uput $FP_CLR, $ascratch
  }

  {
    get $workerIdM1, $WSR
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // process 2 at a time first as this is the optimal scenario
  shr $countD2, $countD2, 1

  // if worker id is less than the remainder this worker can process an extra 2.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD2, $countD2, $mscratch

  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld32step $azero, $mzero, $dataBPtr+=, $workerIdM1
  // If no loops to do, go check for a last one
  brz $countD2, .Lloop_epilogue

  mov $dataStore, $dataPtr
  // Pre-load and cast
  ld32step $dataBi0, $mzero, $dataBPtr+=, CTXT_WORKERS
  {ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  f16v2tof32 $dataB,$dataBi0}
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD2, -1
   f32v2axpy $azeros, $dataB, $data}

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {ld32step $dataBi0, $mzero, $dataBPtr+=, CTXT_WORKERS
   f32v2axpy $result, $azeros, $azeros}
  {ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
   f16v2tof32 $dataB,$dataBi0}
  {st64step $result, $mzero, $dataStore+=, CTXT_WORKERS
   f32v2axpy $azeros, $dataB, $data}
2:
  f32v2axpy $result, $azeros, $azeros
  st64step $result, $mzero, $dataStore+=, CTXT_WORKERS

.Lloop_epilogue:
  // at most one of our workers will have to do the remaining element. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 1 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lepilogue
  {brz $final, .Lepilogue
  zero $datai1}

  // scalar.
  // zero the top half of data and dataB so we can safely accumulate them
  {ldb16 $dataBi0, $mzero, $dataBPtr,0
   zero $dataBi1}
  {ld32 $datai0, $dataPtr, $mzero, 0
   f16tof32 $dataBi0, $dataBi0}

  f32v2axpy $azeros, $dataB, $data
  f32v2axpy $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

.Lepilogue:
  exitz $mzero

.size VERTEX(float_half).kernel, .-VERTEX(float_half).kernel

#endif // __IPU__
