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
  __runCodelet_popnn__NonLinearityGrad1D___half_popnn__NonLinearityType__\NL_TYPE
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad1D___float_popnn__NonLinearityType__\NL_TYPE

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

// Worker register aliases
#define WORKER_ID m0
#define OUTGRAD_PTR m2
#define OUT_PTR m3
#define INGRAD_PTR m4
#define SIZE m5
#define REM m6
#define REM_64BIT m7
#define MSCRATCH m10

#define OUTGRAD_0 a0
#define OUTGRAD_1 a1
#define OUTGRAD_PAIR a0:1
#define OUT_0 a2
#define OUT_1 a3
#define OUT_PAIR a2:3
#define INGRAD_0 a4
#define INGRAD_1 a5
#define INGRAD_PAIR a4:5
#define ONES_0 a6
#define ONES_1 a7
#define ONES a6:7

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

//------------------------------------------------------------------------------
// Handle remaining 32-bit value or 2x16 bit values.
// The maths in this case and the loops below is exactly the same.
// Each variant uses 3 instructions, and different combinations of operands
// to achieve the result - the same here as constructed using macro parameters
// in the loop body
.macro CALC_REMAINDER NL_TYPE INSTR_BASE
.ifc "\NL_TYPE","RELU"
  \INSTR_BASE\()cmpgt $INGRAD_0, $OUT_0, $azero
  \INSTR_BASE\()min $INGRAD_0, $INGRAD_0, $ONES_0
.endif

.ifc "\NL_TYPE","TANH"
  \INSTR_BASE\()mul $INGRAD_0, $OUT_0, $OUT_0
  \INSTR_BASE\()sub $INGRAD_0, $ONES_0, $INGRAD_0
.endif

.ifc "\NL_TYPE","SIGMOID"
  \INSTR_BASE\()mul $INGRAD_0, $OUT_0, $OUT_0
  \INSTR_BASE\()sub $INGRAD_0, $OUT_0, $INGRAD_0
.endif

  \INSTR_BASE\()mul $INGRAD_0, $INGRAD_0, $OUTGRAD_0
.endm

//------------------------------------------------------------------------------
.macro INSTANTIATE_HALF NL_TYPE OP1 OP1_OPERAND_C OP2 OP2_OPERAND_B OP2_OPERAND_C
DEF_STACK_USAGE 0 HALF_SYMBOL
.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

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
  shl $INGRAD_PTR, $INGRAD_PTR, 3
#else
  ld32 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  {
    ld32 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/4
    fnop // rpt alignment
  }
#endif

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  HALF_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  get $WORKER_ID, $WSR
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

  // Load inputs ahead, and generate ones if we need them
.ifnc "\NL_TYPE","SIGMOID"
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
    f16v2exp $ONES_0, $azero }
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
    f16v2exp $ONES_1, $azero }
.else
  ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
  ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
.endif
  brz $SIZE, .Lhalf_32_bit_remainder\@
  // Warm up the pipeline
  { add $SIZE, $SIZE, -1
    f16v4\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C}
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
    f16v4\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C}
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
    f16v4mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR}
  rpt $SIZE, (2f - 1f)/8 - 1
1:
  { st64step $INGRAD_PAIR, $mzero, $INGRAD_PTR+=, CTXT_WORKERS
    f16v4\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C}
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
    f16v4\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C}
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
    f16v4mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR}
2:
  // Handle last pipeline output
  st64step $INGRAD_PAIR, $mzero, $INGRAD_PTR+=, CTXT_WORKERS

.Lhalf_32_bit_remainder\@:
  // Handle remaining element with a single worker. We pick the first
  // worker which didn't handle a remainder element.
  // $REM_64BIT = No. of remaining 64-bit loads possible = index to first
  // worker for which 64-bit load isn't possible.
  cmpeq $MSCRATCH, $WORKER_ID, $REM_64BIT
  brz $MSCRATCH, .Lhalf_end\@

  and $MSCRATCH, $REM, 0x2
  brz $MSCRATCH, .Lhalf_16_bit_remainder\@

  CALC_REMAINDER \NL_TYPE f16v2

  // Store and move the upper word of remaining loaded values
  // down for use in the 16-bit remainder below
  { st32step $INGRAD_0, $mzero, $INGRAD_PTR+=, 1
    mov $OUTGRAD_0, $OUTGRAD_1 }
  mov $OUT_0, $OUT_1

.Lhalf_16_bit_remainder\@:
  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lhalf_end\@

  // Handle remaining 16-bit value
  // Broadcasting lower 16-bits of remaining input words to
  // ensure no exceptions when calculating last gradient.
  { ld32 $INGRAD_1, $mzero, $INGRAD_PTR, 0
    sort4x16lo $OUTGRAD_0, $OUTGRAD_0, $OUTGRAD_0 }
  sort4x16lo $OUT_0, $OUT_0, $OUT_0

  CALC_REMAINDER \NL_TYPE f16v2

  sort4x16hi $INGRAD_0, $INGRAD_0, $INGRAD_1
  st32 $INGRAD_0, $mzero, $INGRAD_PTR, 0

.Lhalf_end\@:
  exitz $mzero
.size HALF_SYMBOL, .-HALF_SYMBOL
.endm

//------------------------------------------------------------------------------
.macro INSTANTIATE_FLOAT NL_TYPE OP1 OP1_OPERAND_C OP2 OP2_OPERAND_B OP2_OPERAND_C
DEF_STACK_USAGE 0 FLOAT_SYMBOL

.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

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
  shl $INGRAD_PTR, $INGRAD_PTR, 3
#else
  ld32 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  {
    ld32 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET/4
    fnop // rpt alignment
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
  add $SIZE, $SIZE, $MSCRATCH

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, $WORKER_ID
  ld64step $INGRAD_PAIR, $mzero, $INGRAD_PTR+=, $WORKER_ID

  // Load inputs ahead, and generate ones if we need them
.ifnc "\NL_TYPE","SIGMOID"
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
    f32exp $ONES_0, $azero }
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
    f32exp $ONES_1, $azero }
.else
  ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
  ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
.endif
  brz $SIZE, .Lfloat_32_bit_remainder\@
  // Warm up the pipeline
  { add $SIZE, $SIZE, -1
    f32v2\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C}
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
    f32v2\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C}
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
    f32v2mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR}
  rpt $SIZE, (2f - 1f)/8 - 1
1:
  { st64step $INGRAD_PAIR, $mzero, $INGRAD_PTR+=, CTXT_WORKERS
    f32v2\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C}
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, CTXT_WORKERS
    f32v2\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C}
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, CTXT_WORKERS
    f32v2mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR}
2:
  // Handle last pipeline output
  st64step $INGRAD_PAIR, $mzero, $INGRAD_PTR+=, CTXT_WORKERS

.Lfloat_32_bit_remainder\@:
  // Handle remaining element with a single worker. We pick the first
  // worker which didn't handle a remainder element.
  // $REM_64BIT = No. of remaining 64-bit loads possible = index to first
  // worker for which 64-bit load isn't possible.
  cmpeq $MSCRATCH, $WORKER_ID, $REM_64BIT
  brz $MSCRATCH, .Lfloat_end\@

  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lfloat_end\@

  // Handle remaining 32-bit value
  CALC_REMAINDER \NL_TYPE f32

  st32step $INGRAD_0, $mzero, $INGRAD_PTR+=, 1

.Lfloat_end\@:
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL
.endm
//------------------------------------------------------------------------------
// Use the macros above to create each vertex for each type of non linearity
//
// Each specifies 2 instructions to use, followed by the register(s) to use for
// some of the operands of that instruction. Other operands are the same for all
// non linearities.
INSTANTIATE_HALF RELU cmpgt $azeros min $INGRAD_PAIR $ONES
INSTANTIATE_FLOAT RELU cmpgt $azeros min $INGRAD_PAIR $ONES

INSTANTIATE_HALF TANH mul $OUT_PAIR sub $ONES $INGRAD_PAIR
INSTANTIATE_FLOAT TANH mul $OUT_PAIR sub $ONES $INGRAD_PAIR

INSTANTIATE_HALF SIGMOID mul $OUT_PAIR sub $OUT_PAIR $INGRAD_PAIR
INSTANTIATE_FLOAT SIGMOID mul $OUT_PAIR sub $OUT_PAIR $INGRAD_PAIR

#endif // __IPU__
