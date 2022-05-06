// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "ScaledAddSupervisor.inc"
#include "CommonPoplibsMacros.h.S"
#include "workDivision.h.S"

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty

// vertex state (offsets in bytes)

//
// Vertex state
//
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_B_OFFSET 4
#define VERTEX_SCALE_OFFSET 8
#define VERTEX_PACKED_COUNT_OFFSET 10

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

FN_SECTION VERTEX(float_half).kernel 8
FN_EXPORT VERTEX(float_half_half_false)
  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHIFTS
  ldb16 $k, $mzero, $dataPtr, 0
  {bri  1f
   f16tof32 $k,$k}
FN_EXPORT VERTEX(float_half_float_false)
  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHIFTS
  ld32  $k, $mzero, $dataPtr, 0
1:
  // load vertex state
  ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  {
    ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
    uput $FP_CLR, $ascratch
  }

  {
    get $workerIdM1, $WSR
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $countD2, LOG2_FLOAT_ATOM_SIZE

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
  // Exit if no remainder
  and $mscratch, $remM1, 1
  brz $mscratch, .Lepilogue
  // There is a remainder to process if we get here.
  // Use the worker which has already got the same address as the remainder
  // to process it
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  andc $remM1, $remM1, 1
  ld32step $azero, $mzero, $mscratch+=, $remM1
  cmpeq $mscratch, $mscratch, $dataPtr
  {brz   $mscratch, .Lepilogue
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

FN_SIZE VERTEX(float_half).kernel

#endif // __IPU__
