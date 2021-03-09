// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of vertex for popnn::NonLinearityGrad2D for SWISH.

// Restrictions
//
//  * All input/output regions 8-byte aligned.
//  * Load up to 64-bits past the end of outGrad and out regions without exceptions.

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "NonLinearitySwishCommon.S"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGradSupervisor___half_popnn__NonLinearityType__SWISH
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGradSupervisor___float_popnn__NonLinearityType__SWISH


// Constants
#if defined(VECTOR_AVAIL_SCALED_PTR64)
#define OUTGRAD_PTR_VOFFSET 0
#define OUT_PTR_VOFFSET 2
#define INGRAD_PTR_VOFFSET 4
#define N_VOFFSET 6
#else
#define OUTGRAD_PTR_VOFFSET 0
#define OUT_PTR_VOFFSET 4
#define INGRAD_PTR_VOFFSET 8
#define N_VOFFSET 12
#endif

// Supervisor register aliases
#define SUPER_BASE m0
#define WORKER_ENTRY m1

// Worker register aliases
#define WORKER_ID m0
#define OUTGRAD_PTR m2
#define OUT_PTR m3
#define INGRAD_PTR m4
#define SIZE m5
#define REM m6
#define REM_64BIT m7
#define MSCRATCH m10
#define MSCRATCH2 m11


DEF_STACK_USAGE 0 HALF_SYMBOL

.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

.align 8
.supervisor

HALF_SYMBOL:
  setzi $WORKER_ENTRY, .Lhalf_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.align 8
.Lhalf_worker:
.worker
  ldz16 $MSCRATCH, $mvertex_base, $mzero, N_VOFFSET/2
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/2
  ldz16 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/2
  ldz16 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/2
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 3
  shl $OUT_PTR, $OUT_PTR, 3
  shl $INGRAD_PTR, $INGRAD_PTR, 3
#else
  ld32 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  {
    ld32 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/4
    fnop
  }
#endif

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  HALF_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  {
    get $WORKER_ID, $WSR
    f16v2exp $ONE, $azero
  }
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Add remaining 64-bit loads/stores to relevant workers
  shr $REM_64BIT, $REM, 2
  cmpult $MSCRATCH, $WORKER_ID, $REM_64BIT
  add $SIZE, $SIZE, $MSCRATCH

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $azeros, $mzero, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $OUT_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $INGRAD_PTR+=, $WORKER_ID
  
  // The rpt loop is pipelined, so decrement loop count by 1
  brz $SIZE, .Lhalf_32_bit_remainder
  add $SIZE, $SIZE, -1

  SwishGradLoopHalfV4 OUT_PTR OUTGRAD_PTR mzero INGRAD_PTR ONE SIZE CTXT_WORKERS

.Lhalf_32_bit_remainder:
  // Handle remaining element with a single worker. We pick the first
  // worker which didn't handle a remainder element.
  // $REM_64BIT = No. of remaining 64-bit loads possible = index to first
  // worker for which 64-bit load isn't possible.
  cmpeq $MSCRATCH, $WORKER_ID, $REM_64BIT
  brz $MSCRATCH, .Lhalf_end

  and $MSCRATCH, $REM, 0x2
  brz $MSCRATCH, .Lhalf_16_bit_remainder

  // Handle remaining 32-bit value
  ld32step $ACTS_0, $mzero, $OUT_PTR+=, 1
  ld32step $GRAD_0, $mzero, $OUTGRAD_PTR+=, 1
  SwishGradHalfV2 RESULT_0 ACTS_0 ASCRATCH GRAD_0 ONE
  st32step $RESULT_0, $mzero, $INGRAD_PTR+=, 1

.Lhalf_16_bit_remainder:
  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lhalf_end

  ldb16 $ACTS_0, $mzero, $OUT_PTR, 0

  // Handle remaining 16-bit value
  // Broadcasting lower 16-bits of remaining input words to
  // ensure no exceptions when calculating last gradient.
  ldb16 $GRAD_0, $mzero, $OUTGRAD_PTR, 0
  SwishGradHalfV2 RESULT_0 ACTS_0 ASCRATCH GRAD_0 ONE
  ldb16 $GRAD_1, $mzero, $INGRAD_PTR, 1
  sort4x16lo $RESULT_0, $RESULT_0, $GRAD_1
  st32 $RESULT_0, $mzero, $INGRAD_PTR, 0

.Lhalf_end:
  exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.supervisor
FLOAT_SYMBOL:
  setzi $WORKER_ENTRY, .Lfloat_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.align 8
.Lfloat_worker:
.worker
  ldz16 $MSCRATCH, $mvertex_base, $mzero, N_VOFFSET/2
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/2
  ldz16 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/2
  ldz16 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/2
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 3
  shl $OUT_PTR, $OUT_PTR, 3
  shl $INGRAD_PTR, $INGRAD_PTR, 3
#else
  ld32 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  {
    ld32 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/4
    fnop
  }
#endif

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  FLOAT_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Add remaining 64-bit loads/stores to relevant workers
  shr $REM_64BIT, $REM, 1
  cmpult $MSCRATCH, $WORKER_ID, $REM_64BIT

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $azeros, $mzero, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $OUT_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $INGRAD_PTR+=, $WORKER_ID

  {
    add $SIZE, $SIZE, $MSCRATCH
    f32exp $ONE, $azero
  }

  // inner loop is pipelined, so decrement loop count by 1
  brz $SIZE, .Lfloat_32_bit_remainder
  add $SIZE, $SIZE, -1

  SwishGradLoopFloatV2 OUT_PTR OUTGRAD_PTR mzero INGRAD_PTR ONE SIZE CTXT_WORKERS

.Lfloat_32_bit_remainder:
  // Handle remaining element with a single worker. We pick the first
  // worker which didn't handle a remainder element.
  // $REM_64BIT = No. of remaining 64-bit loads possible = index to first
  // worker for which 64-bit load isn't possible.
  cmpeq $MSCRATCH, $WORKER_ID, $REM_64BIT
  brz $MSCRATCH, .Lfloat_end

  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lfloat_end

  // Handle remaining 32-bit value
  ld32 $ACTS_0, $mzero, $OUT_PTR, 0
  ld32 $GRAD_0, $mzero, $OUTGRAD_PTR, 0
  SwishGradFloatV1 RESULT_0, ACTS_0, ASCRATCH, GRAD_0, ONE
  st32 $RESULT_0, $mzero, $INGRAD_PTR, 0

.Lfloat_end:
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
