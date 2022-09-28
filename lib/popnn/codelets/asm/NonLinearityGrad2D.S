// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::NonLinearityGrad2D vertex template instantiations.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.
//  * All input/output regions 8-byte aligned.

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___half_popnn__NonLinearityType__\NL_TYPE
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___float_popnn__NonLinearityType__\NL_TYPE

// Constants
#define OUTGRAD_PTR_VOFFSET 0
#define OUT_PTR_VOFFSET 4
#define INGRAD_BASE_AND_N0_VOFFSET 8
#define INGRAD_DELTAN_PTR_VOFFSET 12

#define PTR64_SHL_BITS 3

#define DELTAN_BASE_PTR_BITS 24
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)

// Worker register aliases
#define MASK m0
#define OUTGRAD_OUTER_PTR m1
#define OUTGRAD_PTR m2
#define OUT_OUTER_PTR m3
#define OUT_PTR m4
#define INGRAD_BASE_PTR m5
#define INGRAD_DELTAN_PTR m6
#define INGRAD_OFFSET m7
#define N0 m8
#define N0_B m9
#define N1 m9
#define N1_64BIT m10
#define MSCRATCH m11

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
  ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET/4

  // Unpack base pointer and n0, and generate ones if needed
  ldconst $MASK, DELTAN_BASE_PTR_MASK

.ifnc "\NL_TYPE","SIGMOID"
  { and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
    f16v2exp $ONES_0, $azero }
  { shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS
    f16v2exp $ONES_1, $azero }
.else
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS
.endif

  // DeltaN table pointer contains a 24 bit absolute address
  // followed by the upper 8 bits of N0. Combine with the
  // lower N0 bits which has alraedy been loaded from the
  // upper 8 bits of the Base pointer
  {
    ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/4
    fnop // rpt alignment
  }
  and $INGRAD_DELTAN_PTR, $MSCRATCH, $MASK
  shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
  shl $N0, $N0, 8
  or $N0, $N0, $N0_B

  setzi $MASK, DELTAN_OFFSET_MASK

  // Sub 1 to use post-decrementing brnzdec
  add $N0, $N0, -1
.Lhalf_n0_loop\@:
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  and $INGRAD_OFFSET, $MSCRATCH, $MASK
  shl $INGRAD_OFFSET, $INGRAD_OFFSET, PTR64_SHL_BITS
  ld32step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  ld32step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 2

  // Load inputs ahead
  ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, 1
  ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, 1
  brz $N1_64BIT, .Lhalf_32_bit_remainder\@
  // Warm up the pipeline
  { add $N1_64BIT, $N1_64BIT, -1
    f16v4\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C }
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, 1
    f16v4\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C }
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, 1
    f16v4mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR }
  rpt $N1_64BIT, (2f - 1f)/8 - 1
1:
  { st64step $INGRAD_PAIR, $INGRAD_BASE_PTR, $INGRAD_OFFSET+=, 1
    f16v4\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C }
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, 1
    f16v4\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C }
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, 1
    f16v4mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR }
2:
  // Handle last pipeline output
  st64step $INGRAD_PAIR, $INGRAD_BASE_PTR, $INGRAD_OFFSET+=, 1

.Lhalf_32_bit_remainder\@:
  and $MSCRATCH, $N1, 0x2
  brz $MSCRATCH, .Lhalf_16_bit_remainder\@

  // Handle remaining 32-bit value
  CALC_REMAINDER \NL_TYPE f16v2
  // Store and move the upper word of remaining loaded values
  // down for use in the 16-bit remainder below
  { st32step $INGRAD_0, $INGRAD_BASE_PTR, $INGRAD_OFFSET+=, 1
    mov $OUTGRAD_0, $OUTGRAD_1 }
  mov $OUT_0, $OUT_1

.Lhalf_16_bit_remainder\@:
  and $MSCRATCH, $N1, 0x1
  brz $MSCRATCH, .Lhalf_n0_loop_cond\@

  // Handle remaining 16-bit value
  // Broadcasting lower 16-bits of remaining input words to
  // ensure no exceptions when calculating last gradient.
  { ld32 $INGRAD_1, $INGRAD_BASE_PTR, $INGRAD_OFFSET, 0
    sort4x16lo $OUTGRAD_0, $OUTGRAD_0, $OUTGRAD_0 }
  sort4x16lo $OUT_0, $OUT_0, $OUT_0

  CALC_REMAINDER \NL_TYPE f16v2

  sort4x16hi $INGRAD_0, $INGRAD_0, $INGRAD_1
  st32 $INGRAD_0, $INGRAD_BASE_PTR, $INGRAD_OFFSET, 0

.Lhalf_n0_loop_cond\@:
  brnzdec $N0, .Lhalf_n0_loop\@
  exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function
.endm
//------------------------------------------------------------------------------
.macro INSTANTIATE_FLOAT NL_TYPE OP1 OP1_OPERAND_C OP2 OP2_OPERAND_B OP2_OPERAND_C
.align 8
FLOAT_SYMBOL:
  ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET/4

  // Unpack base pointer and n0, and generate ones if needed
  ldconst $MASK, DELTAN_BASE_PTR_MASK
.ifnc "\NL_TYPE","SIGMOID"
  { and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
    f32exp $ONES_0, $azero }
  { shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS
    f32exp $ONES_1, $azero }
.else
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS
.endif

  // DeltaN table pointer contains a 24 bit absolute address
  // followed by the upper 8 bits of N0. Combine with the
  // lower N0 bits which has alraedy been loaded from the
  // upper 8 bits of the Base pointer
  {
    ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/4
    fnop // rpt alignment
  }
  and $INGRAD_DELTAN_PTR, $MSCRATCH, $MASK
  shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
  shl $N0, $N0, 8
  or $N0, $N0, $N0_B

  setzi $MASK, DELTAN_OFFSET_MASK

  // Sub 1 to use post-decrementing brnzdec
  add $N0, $N0, -1
.Lfloat_n0_loop\@:
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  and $INGRAD_OFFSET, $MSCRATCH, $MASK
  shl $INGRAD_OFFSET, $INGRAD_OFFSET, PTR64_SHL_BITS
  ld32step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ld32step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 1

  // Load inputs ahead
  ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, 1
  ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, 1
  brz $N1_64BIT, .Lfloat_32_bit_remainder\@
  // Warm up the pipeline
  { add $N1_64BIT, $N1_64BIT, -1
    f32v2\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C }
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, 1
    f32v2\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C }
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, 1
    f32v2mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR }
  rpt $N1_64BIT, (2f - 1f)/8 - 1
1:
  { st64step $INGRAD_PAIR, $INGRAD_BASE_PTR, $INGRAD_OFFSET+=, 1
    f32v2\OP1 $INGRAD_PAIR, $OUT_PAIR, \OP1_OPERAND_C }
  { ld64step $OUT_PAIR, $mzero, $OUT_PTR+=, 1
    f32v2\OP2 $INGRAD_PAIR, \OP2_OPERAND_B, \OP2_OPERAND_C }
  { ld64step $OUTGRAD_PAIR, $mzero, $OUTGRAD_PTR+=, 1
    f32v2mul $INGRAD_PAIR, $INGRAD_PAIR, $OUTGRAD_PAIR }
2:
  // Handle last pipeline output
  st64step $INGRAD_PAIR, $INGRAD_BASE_PTR, $INGRAD_OFFSET+=, 1

.Lfloat_32_bit_remainder\@:
  and $MSCRATCH, $N1, 0x1
  brz $MSCRATCH, .Lfloat_n0_loop_condition\@

  // Handle 32-bit remainder
  CALC_REMAINDER \NL_TYPE f32

  st32step $INGRAD_0, $INGRAD_BASE_PTR, $INGRAD_OFFSET+=, 1

.Lfloat_n0_loop_condition\@:
  brnzdec $N0, .Lfloat_n0_loop\@
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL
.endm
//------------------------------------------------------------------------------
// Use the macros above to create each vertex for each type of non linearity.
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
