// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popops::UnaryOp1D[InPlace] vertices for
// the three non linearities SIGMOID RELU, TANH.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.
//
//  * The not-in-place variants require the input and output to be 8 byte
//    aligned
//
//  * The in-place ones only require 32 bit alignment, for both half and float

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "workDivision.h.S"
#include "CommonPoplibsMacros.h.S"

// Symbols
#define HALF_SYMBOL(IN_PLACE) \
  __runCodelet_popops__UnaryOp1D ## IN_PLACE ## ___popops__expr__UnaryOpType__\NL_TYPE\()_half

#define FLOAT_SYMBOL(IN_PLACE) \
  __runCodelet_popops__UnaryOp1D ## IN_PLACE ## ___popops__expr__UnaryOpType__\NL_TYPE\()_float

// Offsets for the vertex state parameters (fields), for the InPlace variants
#if defined(VECTOR_AVAIL_SCALED_PTR32)
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 2
#else
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 4
#endif

// Offsets for the vertex state parameters (fields), for the not-InPlace variants
#define IN_PTR_OFFS   0
#define OUT_PTR_OFFS  4
#define N_OFFS        8

// Worker register aliases
#define WORKER_ID m0
#ifdef VECTOR_AVAIL_SCALED_PTR32
#define BASE m1
#else
#define BASE mzero
#endif
#define DATA_PTR m2
#define OUT_PTR m7
#define SIZE m3
#define REM m4
#define REM_32BIT m5
#define REM_16BIT m6
#define MSCRATCH m10

#define ACTS_0 a0
#define ACTS_1 a1
#define ACTS_PAIR a0:1
#define RESULTS_0 a4
#define RESULTS_1 a5
#define RESULTS_PAIR a4:5
#define ASCRATCH a6
#define ASCRATCH_PAIR a6:7

//------------------------------------------------------------------------------
.macro INSTANTIATE_HALF NL_TYPE OP OPTIONAL_OPERAND_C=""
FN_WORKER_ENTRY_POINT HALF_SYMBOL()
    ld32 $DATA_PTR, $mvertex_base, $mzero, IN_PTR_OFFS/4
    ld32 $OUT_PTR, $mvertex_base, $mzero,  OUT_PTR_OFFS/4
    ld32 $MSCRATCH, $mvertex_base, $mzero, N_OFFS/4
#if defined(VECTOR_AVAIL_SCALED_PTR32)
    setzi $BASE, 0
#endif
    bri .Lhalf_do_work\@
FN_SIZE HALF_SYMBOL()

FN_WORKER_ENTRY_POINT HALF_SYMBOL(InPlace) 8
    ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2

    // Scaled pointer gives offset in 32-bit units from
    // TMEM_REGION0_BASE_ADDR
#if defined(VECTOR_AVAIL_SCALED_PTR32)
    ldz16 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/2
    shl $DATA_PTR, $DATA_PTR, 2
    setzi $BASE, TMEM_REGION0_BASE_ADDR
#else
    ld32 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
#endif
    mov $OUT_PTR, $DATA_PTR

.Lhalf_do_work\@:
    // $SIZE = No. of 64-bit elements each worker should process
    // $REM = No. of remaining elements between workers
    SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM 24


    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Check if address is 64-bit aligned. Note that for the not-in-place
    // variant $DATA_PTR and $OUT_PTR are always 64 bit aligned (guaranteed
    // by the vertex field definitions)
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lhalf_64_bit_aligned\@

.Lhalf_32_bit_aligned\@:
    // Catch special case for just 1 or 2 elements at a 32-bit aligned address.
    setzi $MSCRATCH, 2
    cmpult $MSCRATCH, $MSCRATCH, $REM
    or $MSCRATCH, $MSCRATCH, $SIZE
    brnz $MSCRATCH, .Lhalf_32_bit_lead\@


    shr $REM_32BIT, $REM, 1
    and $REM_16BIT, $REM, 0x1
    shr $REM_32BIT, $REM_32BIT, $WORKER_ID
    shr $REM_16BIT, $REM_16BIT, $WORKER_ID
    bri .Lhalf_32_bit_remainder\@

.Lhalf_32_bit_lead\@:
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .Lhalf_skip_32_bit_lead\@

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    st32 $RESULTS_0, $OUT_PTR, $BASE, 0

.Lhalf_skip_32_bit_lead\@:
    ld32step $ASCRATCH, $BASE, $DATA_PTR+=, 1
    ld32step $ASCRATCH, $BASE, $OUT_PTR+=, 1

    // Decrement remaining element count
    add $REM, $REM, -2
    brpos $REM, .Lhalf_64_bit_aligned\@
    add $REM, $REM, (CTXT_WORKERS * 4)
    add $SIZE, $SIZE, -1

.Lhalf_64_bit_aligned\@:
    // $REM_32BIT = Non-zero if a remaining 32-bit load
    // $REM_16BIT = Non-zero if a remaining 16-bit load
    // $REM = No. of remaining 64-bit loads
    and $REM_32BIT, $REM, 0x2
    and $REM_16BIT, $REM, 0x1
    shr $REM, $REM, 2

    // Add any remaining 64-bit loads/stores possible to relevant
    // workers
    cmpult $MSCRATCH, $WORKER_ID, $REM
    add $SIZE, $SIZE, $MSCRATCH

    // Offset each worker's pointers into data and output to interleave them.
    ld64step $ASCRATCH_PAIR, $BASE, $DATA_PTR+=, $WORKER_ID
    ld64step $ASCRATCH_PAIR, $BASE, $OUT_PTR+=, $WORKER_ID

    // Overlap 64-bit loads/stores with vector 2 half calculations
    brz $SIZE, .Lhalf_64_bit_loop_exit\@
    add $SIZE, $SIZE, -1
    ld64step $ACTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
    {
      rpt $SIZE, (2f - 1f) / 8 - 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
1:
    {
      ld64step $ACTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
      \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    }
    {
      st64step $RESULTS_PAIR, $BASE, $OUT_PTR+=, CTXT_WORKERS
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
2:
    \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    st64step $RESULTS_PAIR, $BASE, $OUT_PTR+=, CTXT_WORKERS
.Lhalf_64_bit_loop_exit\@:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    brz $MSCRATCH, .Lhalf_end\@

.Lhalf_32_bit_remainder\@:
    brz $REM_32BIT, .Lhalf_16_bit_remainder\@

    ld32step $ACTS_0, $BASE, $DATA_PTR+=, 1
    \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    st32step $RESULTS_0, $BASE, $OUT_PTR+=, 1

.Lhalf_16_bit_remainder\@:
    brz $REM_16BIT, .Lhalf_end\@

    // Load the first and second half in the word to store along
    // with the remaining
    ldb16 $ACTS_0, $DATA_PTR, $BASE, 0
    {
      ldb16 $ASCRATCH, $OUT_PTR, $BASE, 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
    roll16 $RESULTS_0, $RESULTS_0, $ASCRATCH
    st32 $RESULTS_0, $OUT_PTR, $BASE, 0

.Lhalf_end\@:
    exitz $mzero
FN_SIZE HALF_SYMBOL(InPlace)
.endm
//------------------------------------------------------------------------------
.macro INSTANTIATE_FLOAT NL_TYPE OP OPTIONAL_OPERAND_C=""

FN_WORKER_ENTRY_POINT FLOAT_SYMBOL()
    ld32 $DATA_PTR, $mvertex_base, $mzero, IN_PTR_OFFS/4
    ld32 $OUT_PTR, $mvertex_base, $mzero,  OUT_PTR_OFFS/4
    ld32 $MSCRATCH, $mvertex_base, $mzero, N_OFFS/4
#if defined(VECTOR_AVAIL_SCALED_PTR32)
    setzi $BASE, 0
#endif
    bri .Lfloat_do_work\@
FN_SIZE FLOAT_SYMBOL()

FN_WORKER_ENTRY_POINT FLOAT_SYMBOL(InPlace) 8
    ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2

    // Scaled pointer gives offset in 32-bit units from
    // TMEM_REGION0_BASE_ADDR
#if defined(VECTOR_AVAIL_SCALED_PTR32)
    ldz16 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/2
    shl $DATA_PTR, $DATA_PTR, 2
    setzi $BASE, TMEM_REGION0_BASE_ADDR
#else
    ld32 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
#endif

    mov $OUT_PTR, $DATA_PTR

.Lfloat_do_work\@:
    // $SIZE = No. of 64-bit elements each worker should process
    // $REM = No. of remaining elements between workers
    SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM 12
    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Check if address is 64-bit aligned. Note that for the not-in-place
    // variant $DATA_PTR and $OUT_PTR are always 64 bit aligned (guaranteed
    // by the vertex field definitions)
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lfloat_64_bit_aligned\@

.Lfloat_32_bit_aligned\@:
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .Lfloat_skip_32_bit_lead\@

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.Lfloat_skip_32_bit_lead\@:
    ld32step $ASCRATCH, $BASE, $DATA_PTR+=, 1
    ld32step $ASCRATCH, $BASE, $OUT_PTR+=, 1

    // Decrement remaining element count
    add $REM, $REM, -1
    brpos $REM, .Lfloat_64_bit_aligned\@
    add $REM, $REM, (CTXT_WORKERS * 2)
    add $SIZE, $SIZE, -1

.Lfloat_64_bit_aligned\@:
    // $SIZE = No. of 64-bit loads/stores possible
    // $REM_32BIT = No. of remaining 32-bit loads
    // $REM = No. of remaining 64-bit loads
    and $REM_32BIT, $REM, 0x1
    shr $REM, $REM, 1

    // Add any remaining 64-bit loads/stores possible to relevant
    // workers
    cmpult $MSCRATCH, $WORKER_ID, $REM
    add $SIZE, $SIZE, $MSCRATCH

    // Offset each worker's pointers into data and output to interleave them.
    ld64step $ASCRATCH_PAIR, $BASE, $DATA_PTR+=, $WORKER_ID
    ld64step $ASCRATCH_PAIR, $BASE, $OUT_PTR+=, $WORKER_ID

    // Overlap 64-bit loads/stores with float calculations
    brz $SIZE, .Lfloat_64_bit_loop_exit\@
    add $SIZE, $SIZE, -1
    ld64step $ACTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
    {
      rpt $SIZE, (2f - 1f) / 8 - 1
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
1:
    {
      ld64step $ACTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
      \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    }
    {
      st64step $RESULTS_PAIR, $BASE, $OUT_PTR+=, CTXT_WORKERS
      \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    }
2:
    \OP $RESULTS_1, $ACTS_1 \OPTIONAL_OPERAND_C
    st64step $RESULTS_PAIR, $BASE, $OUT_PTR+=, CTXT_WORKERS
.Lfloat_64_bit_loop_exit\@:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    and $MSCRATCH, $MSCRATCH, $REM_32BIT
    brz $MSCRATCH, .Lfloat_end\@

.Lfloat_32_bit_remainder\@:
    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    \OP $RESULTS_0, $ACTS_0 \OPTIONAL_OPERAND_C
    st32step $RESULTS_0, $BASE, $OUT_PTR+=, 1

.Lfloat_end\@:
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
