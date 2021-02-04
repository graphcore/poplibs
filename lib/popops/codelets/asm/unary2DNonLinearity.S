// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popops::UnaryOp2D[InPlace] vertices for the three
// non linearities SIGMOID RELU, TANH.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.
//
//  * The not-in-place variants require each vector (both input and output)
//    to be 8 byte aligned
//
//  * The in-place ones only require 'natural' alignment (2 bytes for half,
//    4 bytes for float)

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// This macro associates to the symbol 'label' a size defined as
// (Current_loc - label)
.macro FN_SIZE label
.size \label, . - \label
.endm

// Symbols
#define HALF_SYMBOL(IN_PLACE) \
  __runCodelet_popops__UnaryOp2D ## IN_PLACE ## ___popops__expr__UnaryOpType__\NL_TYPE\()_half

#define FLOAT_SYMBOL(IN_PLACE) \
  __runCodelet_popops__UnaryOp2D ## IN_PLACE ## ___popops__expr__UnaryOpType__\NL_TYPE\()_float

// Offsets for the vertex state parameters (fields), for the InPlace variants
#define BASE_AND_N0_VOFFSET 0
#define DELTAN_PTR_VOFFSET 4

// Offsets for the vertex state parameters (fields), for the not-InPlace variants
#define IN_ITERATOR_OFFS   0
#define OUT_ITERATOR_OFFS  4
#define N_VECT_2D_OFFS    8


#define PTR16_SHL_BITS 1
#define PTR32_SHL_BITS 2

#if defined(VECTORLIST_AVAIL_DELTAN)
#define DELTAN_BASE_PTR_BITS 20
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_HALF_OFFSET_BITS 18
#define DELTAN_HALF_OFFSET_MASK ((1 << DELTAN_HALF_OFFSET_BITS) - 1)
#define DELTAN_FLOAT_OFFSET_BITS 18
#define DELTAN_FLOAT_OFFSET_MASK ((1 << DELTAN_FLOAT_OFFSET_BITS) - 1)
#else
#define DELTAN_BASE_PTR_BITS 24
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_HALF_OFFSET_BITS 20
#define DELTAN_HALF_OFFSET_MASK ((1 << DELTAN_HALF_OFFSET_BITS) - 1)
#define DELTAN_FLOAT_OFFSET_BITS 19
#define DELTAN_FLOAT_OFFSET_MASK ((1 << DELTAN_FLOAT_OFFSET_BITS) - 1)
#endif

// Worker register aliases
#define MASK m0
#if defined(VECTORLIST_AVAIL_DELTAN)
#define MEMORY_BASE m1
#else
#define MEMORY_BASE mzero
#endif
#define BASE_PTR m2
#define N0 m3
#define N0_B m6
#define DELTAN_PTR m4
#define DATA_PTR m5
#define N1 m6
#define N1_64BIT m7
#define OUT_PTR m8
#define MSCRATCH m10

#define IN_ITERATOR   m0
#define OUT_ITERATOR  m1
#define N_VECT_2D     m2

#define ACTS_0 a0
#define ACTS_1 a1
#define ACTS_PAIR a0:1
#define RESULTS_0 a4
#define RESULTS_1 a5
#define RESULTS_PAIR a4:5
#define ASCRATCH a6
#define ASCRATCH_PAIR a6:7

//------------------------------------------------------------------------------
//                                  HALF CODE
//------------------------------------------------------------------------------

// Process one linear vector of HALF data, with the input and output vectors
// aligned on 64 bit boundary (8 bytes, 4 halves)
//
// MRF registers used in this macro (all are modified):
//
// $DATA_PTR : pointer to start of input vector
// $OUT_PTR  : pointer to start of output vector
// $N1       : number of elements
//
// $MSCRATCH,  N1_64BIT
.macro PROCESS_VECTOR_HALF OP OPTIONAL_OPERAND_C
    shr $N1_64BIT, $N1, 2

    brz $N1_64BIT, .Lhalf_32_bit_remainder\@
    add $N1_64BIT, $N1_64BIT, -1
    ld64step $ACTS_PAIR, $mzero, $DATA_PTR+=, 1
    {
      rpt $N1_64BIT, (2f - 1f) / 8 - 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
1:
    {
      ld64step $ACTS_PAIR, $mzero, $DATA_PTR+=, 1
      \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    }
    {
      st64step $RESULTS_PAIR, $mzero, $OUT_PTR+=, 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
2:
    \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    st64step $RESULTS_PAIR, $mzero, $OUT_PTR+=, 1

.Lhalf_32_bit_remainder\@:
    and $MSCRATCH, $N1, 0x2
    brz $MSCRATCH, .Lhalf_16_bit_remainder\@

    ld32step $ACTS_0, $mzero, $DATA_PTR+=, 1
    \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    st32step $RESULTS_0, $mzero, $OUT_PTR+=, 1

.Lhalf_16_bit_remainder\@:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lend_half_process_vector\@

    ldb16 $ACTS_0, $DATA_PTR, $mzero, 0
    {
      ldb16 $ASCRATCH, $OUT_PTR, $mzero, 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
    roll16 $RESULTS_0, $RESULTS_0, $ASCRATCH
    st32 $RESULTS_0, $mzero, $OUT_PTR, 0
.Lend_half_process_vector\@:
.endm

// Will generate all code required for one operation, both in-place and
// not-in-place, for half data
.macro INSTANTIATE_HALF NL_TYPE OP OPTIONAL_OPERAND_C=""

DEF_STACK_USAGE 0 HALF_SYMBOL()
.section .text.HALF_SYMBOL()
.globl HALF_SYMBOL()
.type HALF_SYMBOL(), @function

.align 8
    nop
HALF_SYMBOL():
    ld32 $IN_ITERATOR, $mvertex_base, $mzero, IN_ITERATOR_OFFS/4
    ld32 $OUT_ITERATOR, $mvertex_base, $mzero, OUT_ITERATOR_OFFS/4
    ld32 $N_VECT_2D, $mvertex_base, $mzero,N_VECT_2D_OFFS/4
    add  $N_VECT_2D, $N_VECT_2D, -1  // for brnzdec
.Lhalf_loop\@:
    ld32step $DATA_PTR, $mzero, $IN_ITERATOR+=, 1
    ld32step $OUT_PTR, $mzero, $OUT_ITERATOR+=, 1
    ld32step $N1, $mzero, $OUT_ITERATOR+=, 1

    PROCESS_VECTOR_HALF \OP "\OPTIONAL_OPERAND_C"

    brnzdec $N_VECT_2D, .Lhalf_loop\@
    exitz $mzero
FN_SIZE HALF_SYMBOL()


DEF_STACK_USAGE 0 HALF_SYMBOL(InPlace)
.section .text.HALF_SYMBOL(InPlace)
.globl HALF_SYMBOL(InPlace)
.type HALF_SYMBOL(InPlace), @function

.align 8
HALF_SYMBOL(InPlace):
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET/4

    // Unpack base pointer and n0
#if defined(VECTORLIST_AVAIL_DELTAN)
    setzi $MASK, DELTAN_BASE_PTR_MASK
#else
    ldconst $MASK, DELTAN_BASE_PTR_MASK
#endif
    and $BASE_PTR, $MSCRATCH, $MASK
    shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

#if defined(VECTORLIST_AVAIL_DELTAN)
    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/2
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, PTR32_SHL_BITS
#else
    // DeltaN table pointer contains a 24 bit absolute address
    // followed by the upper 8 bits of N0. Combine with the
    // lower N0 bits which has alraedy been loaded from the
    // upper 8 bits of the Base pointer
    ld32 $MSCRATCH, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/4
    and $DELTAN_PTR, $MSCRATCH, $MASK
    shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
    shl $N0, $N0, 8
    or $N0, $N0, $N0_B
#endif

    setzi $MASK, DELTAN_HALF_OFFSET_MASK

    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.Lhalf_n0_loop\@:
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $MASK
#if !defined(VECTORLIST_AVAIL_DELTAN)
    shl $DATA_PTR, $DATA_PTR, PTR16_SHL_BITS
#endif
    shr $N1, $MSCRATCH, DELTAN_HALF_OFFSET_BITS
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    and $MSCRATCH, $DATA_PTR, 0x3
    brz $MSCRATCH, .Lhalf_32_bit_aligned\@

    // Handle the first 16-bit element. We'll always have
    // at least 1 element here.
    andc $DATA_PTR, $DATA_PTR, 0x3
    ldb16 $ACTS_0, $DATA_PTR, $mzero, 1
    {
      ldb16 $ASCRATCH, $DATA_PTR, $mzero, 0
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
    roll16 $RESULTS_0, $ASCRATCH, $RESULTS_0
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -1
    brz $N1, .Lhalf_n0_loop_cond\@

.Lhalf_32_bit_aligned\@:
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lhalf_64_bit_aligned\@

    // Special case for a single 16-bit element at 32-bit
    // aligned address.
    cmpult $MSCRATCH, $N1, 2
    brnz $MSCRATCH, .Lhalf_16_bit_remainder_inplace\@

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    {
      add $N1, $N1, -2
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.Lhalf_64_bit_aligned\@:
    mov $OUT_PTR, $DATA_PTR
    PROCESS_VECTOR_HALF \OP "\OPTIONAL_OPERAND_C"

.Lhalf_n0_loop_cond\@:
    brnzdec $N0, .Lhalf_n0_loop\@
    exitz $mzero

// handling of last element, note that this is duplicated from the end of
// PROCESS_VECTOR_HALF
.Lhalf_16_bit_remainder_inplace\@:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lhalf_n0_loop_cond\@

    ldb16 $ACTS_0, $DATA_PTR, $mzero, 0
    {
      ldb16 $ASCRATCH, $DATA_PTR, $mzero, 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
    roll16 $RESULTS_0, $RESULTS_0, $ASCRATCH
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

    bri .Lhalf_n0_loop_cond\@

FN_SIZE HALF_SYMBOL(InPlace)
.endm


//------------------------------------------------------------------------------
//                                 FLOAT CODE
//------------------------------------------------------------------------------

// Process one linear vector of FLOAT data, with the input and output vectors
// aligned on 64 bit boundary (8 bytes, 2 floats)
//
// MRF registers used in this macro (all are modified):
//
// $DATA_PTR : pointer to start of input vector
// $OUT_PTR  : pointer to start of output vector
// $N1       : number of elements
//
// $MSCRATCH,  N1_64BIT
//
.macro PROCESS_VECTOR_FLOAT OP OPTIONAL_OPERAND_C
    shr $N1_64BIT, $N1, 1

    brz $N1_64BIT, .Lfloat_32_bit_remainder\@
    add $N1_64BIT, $N1_64BIT, -1
    ld64step $ACTS_PAIR, $mzero, $DATA_PTR+=, 1
    {
      rpt $N1_64BIT, (2f - 1f) / 8 - 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
1:
    {
      ld64step $ACTS_PAIR, $mzero, $DATA_PTR+=, 1
      \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    }
    {
      st64step $RESULTS_PAIR, $mzero, $OUT_PTR+=, 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
2:
    \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    st64step $RESULTS_PAIR, $mzero, $OUT_PTR+=, 1

.Lfloat_32_bit_remainder\@:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lend_float_process_vector\@

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    st32 $RESULTS_0, $mzero, $OUT_PTR, 0
.Lend_float_process_vector\@:
.endm

// Will generate all code required for one operation, both in-place and
// not-in-place, for float data
.macro INSTANTIATE_FLOAT NL_TYPE OP OPTIONAL_OPERAND_C=""

DEF_STACK_USAGE 0 FLOAT_SYMBOL()
.section .text.FLOAT_SYMBOL()
.globl FLOAT_SYMBOL()
.type FLOAT_SYMBOL(), @function

.align 8
    nop
FLOAT_SYMBOL():
    ld32 $IN_ITERATOR, $mvertex_base, $mzero, IN_ITERATOR_OFFS/4
    ld32 $OUT_ITERATOR, $mvertex_base, $mzero, OUT_ITERATOR_OFFS/4
    ld32 $N_VECT_2D, $mvertex_base, $mzero,N_VECT_2D_OFFS/4
    add  $N_VECT_2D, $N_VECT_2D, -1  // for brnzdec
.Lhalf_loop\@:
    ld32step $DATA_PTR, $mzero, $IN_ITERATOR+=, 1
    ld32step $OUT_PTR, $mzero, $OUT_ITERATOR+=, 1
    ld32step $N1, $mzero, $OUT_ITERATOR+=, 1

    PROCESS_VECTOR_FLOAT \OP "\OPTIONAL_OPERAND_C"

    brnzdec $N_VECT_2D, .Lhalf_loop\@
    exitz $mzero
FN_SIZE FLOAT_SYMBOL()

DEF_STACK_USAGE 0 FLOAT_SYMBOL(InPlace)
.section .text.FLOAT_SYMBOL(InPlace)
.globl FLOAT_SYMBOL(InPlace)
.type FLOAT_SYMBOL(InPlace), @function

.align 8
FLOAT_SYMBOL(InPlace):
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET/4

    // Unpack base pointer and n0
#if defined(VECTORLIST_AVAIL_DELTAN)
    setzi $MASK, DELTAN_BASE_PTR_MASK
#else
    ldconst $MASK, DELTAN_BASE_PTR_MASK
#endif
    and $BASE_PTR, $MSCRATCH, $MASK
    shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

#if defined(VECTORLIST_AVAIL_DELTAN)
    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/2
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, PTR32_SHL_BITS
#else
    // DeltaN table pointer contains a 24 bit absolute address
    // followed by the upper 8 bits of N0. Combine with the
    // lower N0 bits which has alraedy been loaded from the
    // upper 8 bits of the Base pointer
    ld32 $MSCRATCH, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/4
    and $DELTAN_PTR, $MSCRATCH, $MASK
    shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
    shl $N0, $N0, 8
    or $N0, $N0, $N0_B
#endif

    setzi $MASK, DELTAN_FLOAT_OFFSET_MASK

    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.Lfloat_n0_loop\@:
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $MASK
#if !defined(VECTORLIST_AVAIL_DELTAN)
    shl $DATA_PTR, $DATA_PTR, PTR32_SHL_BITS
#endif
    shr $N1, $MSCRATCH, DELTAN_FLOAT_OFFSET_BITS
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    // DATA_PTR and N1 give us the regions to actually loop
.Lfloat_32_bit_aligned\@:
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lfloat_64_bit_aligned\@

    // Handle the first 32-bit element. We'll always have
    // at least 1 element here.
    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    {
      add $N1, $N1, -1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.Lfloat_64_bit_aligned\@:
    mov $OUT_PTR, $DATA_PTR
    PROCESS_VECTOR_FLOAT \OP "\OPTIONAL_OPERAND_C"

.Lfloat_n0_loop_cond\@:
    brnzdec $N0, .Lfloat_n0_loop\@
    exitz $mzero

FN_SIZE FLOAT_SYMBOL(InPlace)
.endm

//------------------------------------------------------------------------------
// Use the macros above to create each vertex for each type of non linearity.
//
// Each specifies an instruction.  In the case of RELU that instruction has a
// third operand which must be passed in as well
INSTANTIATE_FLOAT TANH f32tanh
INSTANTIATE_HALF TANH f16v2tanh

INSTANTIATE_FLOAT RELU f32max ",$azero"
INSTANTIATE_HALF RELU f16v2max ",$azero"

INSTANTIATE_FLOAT SIGMOID f32sigm
INSTANTIATE_HALF SIGMOID f16v2sigm

#endif // __IPU__
