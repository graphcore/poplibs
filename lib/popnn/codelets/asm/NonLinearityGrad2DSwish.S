// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::NonLinearityGrad2D vertex for SWISH activations

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.
//  * All input/output regions 8-byte aligned.

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "NonLinearitySwishCommon.S"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___half_popnn__NonLinearityType__SWISH
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___float_popnn__NonLinearityType__SWISH

// Constants
#define OUTGRAD_PTR_VOFFSET         0
#define OUT_PTR_VOFFSET             4
#define INGRAD_BASE_AND_N0_VOFFSET  8
#define INGRAD_DELTAN_PTR_VOFFSET   12

#define PTR64_SHL_BITS 3

#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)

// Worker register aliases
#define MASK                    m0
#define OUTGRAD_OUTER_PTR       m1
#define OUTGRAD_PTR             m2
#define OUT_OUTER_PTR           m3
#define OUT_PTR                 m4
#define INGRAD_BASE_PTR         m5
#define INGRAD_DELTAN_PTR       m6
#define INGRAD_PTR              m7
#define N0                      m8
#define N0_B                    m9
#define N1                      m9
#define MSCRATCH                m10
#define N1_64BIT                m11

DEF_STACK_USAGE 0 HALF_SYMBOL

.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function
.align 8
HALF_SYMBOL:
  {
    ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
    f16v2exp $ONE, $azero
  }
  ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET/4

  // Unpack base pointer and n0
  ldconst $MASK, DELTAN_BASE_PTR_MASK
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

  // DeltaN table pointer contains a 24 bit absolute address
  // followed by the upper 8 bits of N0. Combine with the
  // lower N0 bits which has alraedy been loaded from the
  // upper 8 bits of the Base pointer
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/4
  and $INGRAD_DELTAN_PTR, $MSCRATCH, $MASK
  shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
  shl $N0, $N0, 8
  or $N0, $N0, $N0_B

  setzi $MASK, DELTAN_OFFSET_MASK
  // Sub 1 to use post-decrementing brnzdec
  add $N0, $N0, -1

.Lhalf_n0_loop:
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  and $INGRAD_PTR, $MSCRATCH, $MASK
  shl $INGRAD_PTR, $INGRAD_PTR, PTR64_SHL_BITS
  ld32step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ld32step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 2

  // decrement by 1 to use pipelined rpt loop
  brz $N1_64BIT, .Lhalf_32_bit_remainder
  {
    add $N1_64BIT, $N1_64BIT, -1
    fnop
  }
  SwishGradLoopHalfV4 OUT_PTR OUTGRAD_PTR INGRAD_BASE_PTR INGRAD_PTR ONE N1_64BIT 1

.Lhalf_32_bit_remainder:
  and $MSCRATCH, $N1, 0x2
  brz $MSCRATCH, .Lhalf_16_bit_remainder

  // Handle remaining 32-bit value
  ld32step $ACTS_0, $mzero, $OUT_PTR+=, 1
  ld32step $ASCRATCH, $mzero, $OUTGRAD_PTR+=, 1
  SwishGradHalfV2 RESULT_0 ACTS_0 SIGM_0 ASCRATCH ONE
  st32step $RESULT_0, $INGRAD_BASE_PTR, $INGRAD_PTR+=, 1

.Lhalf_16_bit_remainder:
  and $MSCRATCH, $N1, 0x1
  brz $MSCRATCH, .Lhalf_n0_loop_cond

  ldb16 $ACTS_0, $OUT_PTR, $mzero, 0
  ldb16 $ASCRATCH, $OUTGRAD_PTR, $mzero, 0
  SwishGradHalfV2 RESULT_0 ACTS_0 SIGM_0 ASCRATCH ONE
  ldb16 $GRAD_0, $INGRAD_BASE_PTR, $INGRAD_PTR, 1
  sort4x16lo $GRAD_0, $RESULT_0, $GRAD_0
  st32 $GRAD_0, $INGRAD_BASE_PTR, $INGRAD_PTR, 0

.Lhalf_n0_loop_cond:
  brnzdec $N0, .Lhalf_n0_loop
  exitz $mzero
.size HALF_SYMBOL, .-HALF_SYMBOL


DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.align 8
FLOAT_SYMBOL:
  ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
  ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET/4

  // Unpack base pointer and n0
  ldconst $MASK, DELTAN_BASE_PTR_MASK
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

  // DeltaN table pointer contains a 24 bit absolute address
  // followed by the upper 8 bits of N0. Combine with the
  // lower N0 bits which has alraedy been loaded from the
  // upper 8 bits of the Base pointer
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/4
  and $INGRAD_DELTAN_PTR, $MSCRATCH, $MASK
  shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
  shl $N0, $N0, 8
  or $N0, $N0, $N0_B

  {
    setzi $MASK, DELTAN_OFFSET_MASK
    f32exp $ONE, $azero
  }

  // Sub 1 to use post-decrementing brnzdec
  add $N0, $N0, -1

.Lfloat_n0_loop:
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  and $INGRAD_PTR, $MSCRATCH, $MASK
  shl $INGRAD_PTR, $INGRAD_PTR, PTR64_SHL_BITS
  ld32step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ld32step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 1

  // 1 less because of the pipelining in the rpt loop
  brz $N1_64BIT, .Lfloat_32_bit_remainder
  {
    add $N1_64BIT, $N1_64BIT, -1
    fnop
  }
  SwishGradLoopFloatV2 OUT_PTR OUTGRAD_PTR INGRAD_BASE_PTR INGRAD_PTR ONE N1_64BIT 1

.Lfloat_32_bit_remainder:
  and $MSCRATCH, $N1, 0x1
  brz $MSCRATCH, .Lfloat_n0_loop_condition

  // Handle 32-bit remainder
  ld32 $ACTS_0, $OUT_PTR, $azero, 0
  ld32 $ASCRATCH, $OUTGRAD_PTR, $mzero, 0
  SwishGradFloatV1 RESULT_0 ACTS_0 SIGM_0 ASCRATCH ONE
  st32 $RESULT_0, $INGRAD_BASE_PTR, $INGRAD_PTR, 0

.Lfloat_n0_loop_condition:
  brnzdec $N0, .Lfloat_n0_loop
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
