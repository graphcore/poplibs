// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::NonLinearityGrad1D vertex template instantiations.

// Restrictions
//
//  * All input/output regions 8-byte aligned.
//  * Load up to 64-bits past the end of outGrad and out regions without exceptions.

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad1D___half_popnn__NonLinearityType__GELU
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad1D___float_popnn__NonLinearityType__GELU

// Force all accumulators to zero
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

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

#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define LOG2_24_OVER_3 3
#define LOG2_12_OVER_3 2

#define HALF_1_0 0x3C00
#define HALF_0_5 0x3800
#define HALF_MINUS_0_5 0xB800
#define HALF_ALPHA 0x3A62  // 0.7980
#define HALF_BETA 0x2989   // 0.0447
#define HALF_ALPHA_TIMES_BETA 0x2891 // 3.568E-2
#define HALF_6_0 0x4600
#define HALF_MINUS_6_0 0xC600

#define FLOAT_12_0 0x41400000
#define FLOAT_MINUS_12_0 0xC1400000
#define FLOAT_MINUS_0_5 0xBF000000
#define FLOAT_0_5 0x3F000000
#define FLOAT_ALPHA 0x3F4C422A  // sqrt(2 / PI) ~= 0.79788456
#define FLOAT_BETA 0x3D372713   // 0.044715
#define FLOAT_ALPHA_TIMES_BETA 0x3D122279   // 3.5677407E-2

// Scratch Offsets in 2xFloats
#define SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5 (0 / 2) // only for float vertices
#define SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA    (2 / 2) // all vertices
#define SCRATCH_OFFSET_FLOAT_CLAMP           (4 / 2) // only for float vertices
#define SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF  (6 / 2) // all vertices

// Scratch Offsets in 1xFloats
#define SCRATCH_OFFSET_CONST_HALF_HI_1_0_LO_0_5    0 // only for half vertices
#define SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5 1 // only for half vertices
#define SCRATCH_OFFSET_HALF_CLAMP_LIMITS          10 // only for half vertices
#define SCRATCH_OFFSET_CONST_M0_5                  8 // all vertices

#define SCRATCH_OFFSET_CONST_ALPHA_TIMES_BETA  (SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA * 2)
#define SCRATCH_OFFSET_CONST_ALPHA           ((SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA * 2) + 1)
#define SCRATCH_OFFSET_CONST_0_5             (SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5 * 2)
#define SCRATCH_OFFSET_CONST_1_0             ((SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5 * 2) + 1)

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

// Output gradient, which is an input to the subroutines in this file
#define OUTGRAD_PAR a1

// Result accumulator
// Accumulator also functions as factor0
#define ACC_0 a4
#define ACC_1 a5
#define ACC_PAIR a4:5

// Clamped version of activations
#define XCLAMPED_0 a6
#define XCLAMPED_1 a7
#define XCLAMPED_PAIR a6:7

// NOTE: Register definitions as well as Loop implementation are included here
#include "NonLinearityGradLoop_gelu.S"


DEF_STACK_USAGE 0 HALF_SYMBOL

.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

// All inputs must be separate registers
// Splits 64-bit chunks of n elements between workers.
// The result we want is n / (no. of worker contexts * elements per-64-bits).
// We achieve this by dividing by 3 first, by multiplying n by the reciprocal
// of 3 shifted left. This value is then shifted right by the same amount + any
// further division by powers of 2 to get the actual divisor we want.
// As an example, in this half case there are 4 halves per-64-bits and
// 6 worker contexts so the divisor we want is 24.
// (n / 3) / 8 = n / 24 so the extra divisor is 8, meaning an extra shift of 3.
.macro HALF_SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_24_OVER_3)
    mul \rem, \size, 24
    sub \rem, \n, \rem
.endm

.align 8

HALF_SYMBOL:

.worker
  ldz16 $MSCRATCH, $mvertex_base, $mzero, N_VOFFSET/2
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/2
  ldz16 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/2
  ldz16 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/2
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 3
  shl $OUT_PTR, $OUT_PTR, 3
  {
    shl $INGRAD_PTR, $INGRAD_PTR, 3
    zero $ACC_PAIR
  }
#else
  ld32 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  {
    ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
    fnop // rpt alignment
  }
  {
    ld32 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/4
    zero $ACC_PAIR
  }
#endif

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  HALF_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  {
    get $WORKER_ID, $WSR
    zero $XCLAMPED_PAIR
  }
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Add remaining 64-bit loads/stores to relevant workers
  shr $REM_64BIT, $REM, 2

  {
    cmpult $MSCRATCH, $WORKER_ID, $REM_64BIT
    setzi $ASCRATCH_0, ZAACC_BITMASK
  }

  // Force all accumulators to zero
  {
    add $SIZE, $SIZE, $MSCRATCH
    uput $FP_CLR, $ASCRATCH_0
  }

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $azeros, $mzero, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $OUT_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $INGRAD_PTR+=, $WORKER_ID

  // Load clamp values
  ldconst $HALF_CLAMP_LIMITS, (HALF_MINUS_6_0 | (HALF_6_0 << 16))
  st32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
  ldconst $ASCRATCH_0, (HALF_0_5) | (HALF_1_0 << 16)
  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_1_0_LO_0_5
  ldconst $ASCRATCH_0, (HALF_MINUS_0_5) | (HALF_ALPHA << 16)
  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5
  ldconst $ASCRATCH_0, (HALF_ALPHA_TIMES_BETA) | (HALF_ALPHA << 16)

  {
    brz $SIZE, .Lhalf_32_bit_remainder
    uput $TAS, $ASCRATCH_0
  }

  // Inner Loop kernel
  NONLINEARITY_GELU_HALF $SIZE $mzero $mzero $mzero CTXT_WORKERS

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
  ld32step $OUT_0, $mzero, $OUT_PTR+=, 1
  ld32step $OUTGRAD_PAR, $mzero, $OUTGRAD_PTR+=, 1
  call $MSCRATCH, calc_GELU_half
  st32step $INGRAD_0, $mzero, $INGRAD_PTR+=, 1

.Lhalf_16_bit_remainder:
  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lhalf_end

  ldb16 $OUT_0, $mzero, $OUT_PTR, 0

  // Handle remaining 16-bit value
  // Broadcasting lower 16-bits of remaining input words to
  // ensure no exceptions when calculating last gradient.
  ldb16 $OUTGRAD_PAR, $mzero, $OUTGRAD_PTR, 0
  call $MSCRATCH, calc_GELU_half
  ldb16 $INGRAD_1, $mzero, $INGRAD_PTR, 1
  sort4x16lo $INGRAD_0, $INGRAD_0, $INGRAD_1
  st32 $INGRAD_0, $mzero, $INGRAD_PTR, 0

.Lhalf_end:
  exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

// All inputs must be separate registers
// As described above in HALF_SPLIT_BETWEEN_WORKERS with different
// divisor.
.macro FLOAT_SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_12_OVER_3)
    mul \rem, \size, 12
    sub \rem, \n, \rem
.endm

.align 8

FLOAT_SYMBOL:
.worker
  ldz16 $MSCRATCH, $mvertex_base, $mzero, N_VOFFSET/2
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/2
  ldz16 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/2
  ldz16 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/2
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 3
  shl $OUT_PTR, $OUT_PTR, 3
  {
    shl $INGRAD_PTR, $INGRAD_PTR, 3
    zero $ACC_PAIR
  }
#else
  ld32 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  {
    ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
    fnop // rpt alignment
  }
  {
    ld32 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/4
    zero $ACC_PAIR
  }
#endif

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  FLOAT_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  {
    get $WORKER_ID, $WSR
    zero $XCLAMPED_PAIR
  }
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Add remaining 64-bit loads/stores to relevant workers
  {
    shr $REM_64BIT, $REM, 1
    setzi $ASCRATCH_0, ZAACC_BITMASK
  }

  {
    cmpult $MSCRATCH, $WORKER_ID, $REM_64BIT
    uput $FP_CLR, $ASCRATCH_0
  }

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $azeros, $mzero, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $azeros, $mzero, $OUT_PTR+=, $WORKER_ID

  // Load constants
  ldconst $ASCRATCH_0, FLOAT_ALPHA_TIMES_BETA
  ldconst $ASCRATCH_1, FLOAT_ALPHA

  {
    ld64step $azeros, $mzero, $INGRAD_PTR+=, $WORKER_ID
    uput $TAS, $ASCRATCH_1
  }

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA
    or      $ASCRATCH_0, $azero, FLOAT_0_5
  }

  {
    add $SIZE, $SIZE, $MSCRATCH
    f32exp  $ASCRATCH_1, $azero
  }

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5
    or      $ASCRATCH_0, $azero, FLOAT_MINUS_0_5
  }

  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_M0_5

  ldconst $FLOAT_CLAMP_LIMITS_0, FLOAT_MINUS_12_0
  ldconst $FLOAT_CLAMP_LIMITS_1, FLOAT_12_0
  st64    $FLOAT_CLAMP_LIMITS_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP

  brz $SIZE, .Lfloat_32_bit_remainder

  // Inner Loop kernel
  NONLINEARITY_GELU_FLOAT $SIZE $mzero $mzero $mzero CTXT_WORKERS

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
  ld32 $OUT_0, $mzero, $OUT_PTR, 0
  ld32 $OUTGRAD_PAR, $mzero, $OUTGRAD_PTR, 0
  call $MSCRATCH, calc_GELU_float
  st32 $INGRAD_0, $mzero, $INGRAD_PTR, 0

.Lfloat_end:
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
