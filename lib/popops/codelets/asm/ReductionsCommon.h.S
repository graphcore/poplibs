// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
//******************************************************************************
// Functions used in Reduction.S and Reductions_4_vectorised.S
//******************************************************************************

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "CommonPoplibsMacros.h.S"

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
#define OUT_OFF          0
#define OUT_OFFSET       4
#define NUM_PART_OFF     6
#define IN_OFF           8
#define IN_OFFSET        12
#define SCALE_OFF        14

#define DELTAN_BASE_BITS   20
#define DELTAN_COUNT_BITS  12
#define DELTAN_OFFSET_MASK ((1 << DELTAN_BASE_BITS) - 1)
#define DELTAN_OFFSET_MASK_COMPL (~DELTAN_OFFSET_MASK)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_LENGTH_BITS 14

#else
#define OUT_OFF          0
#define OUT_OFFSET       4
#define NUM_PART_OFF     8
#define IN_OFF           12
#define IN_OFFSET        16
#define SCALE_OFF        20

#define DELTAN_BASE_BITS   24
#define DELTAN_COUNT_BITS  8
#define DELTAN_OFFSET_MASK ((1 << DELTAN_BASE_BITS) - 1)
#define DELTAN_OFFSET_MASK_COMPL (~DELTAN_OFFSET_MASK)
#define DELTAN_OFFSET_BITS 21
#define DELTAN_LENGTH_BITS 11

#endif

#define LDCONST_MASK     ((1<<20)-1)

// all scratch offsets given in words
#define REM_SCRATCH      0
#define IN_PTR_SCRATCH   1
#define BASE_SCRATCH     2
#define NP_PTR_SCRATCH   3
#define OUT_PTR_SCRATCH  4
#define NP_SCRATCH       5
#define OUT_j_SIZE_SCRATCH  6
#define FN_REDUCE_OUTER_LOOP_SCRATCH  7

#define NUM_ELEM        m0
#define OUT_i_PTR       m0
#define OUT_j_PTR       m1
#define IN_i_PTR        m2
#define IN_j_PTR        m3
#define OUT_i_SIZE      m4
#define OUT_j_SIZE      m5
#define OUT_BASE        m6
#define IN_j_DELTA      m6
#define NUM_PART_PTR    m7
#define SCRATCH2        m7
#define SCRATCH         m8
#define NUM_PART        m9
#define IN_BASE         m10
#define IN_j_SIZE       m11

#define VALUES_0        a0
#define VALUES_1        a1
#define VALUES_2        a2
#define VALUES_3        a3
#define VALUES_4        a4
#define ACC_0           a2
#define ACC_1           a3
#define ASCRATCH_0      a5
#define SCALE           a6
#define ZAACC           a7


//******************************************************************************
// Macro declarations
//******************************************************************************
//******************************************************************************
// Macro to create function which gets called at the top of each outer loop.
//******************************************************************************
.macro INSTANTIATE_REDUCE_OUTER_LOOP_SETUP FNAME LOG2_BYTE_ALIGNMENT

FN_WORKER_ENTRY_POINT \FNAME
// ************************************************* //
// unpack offset and size
// ************************************************* //
  ld32       $OUT_BASE, $mworker_base, $mzero, BASE_SCRATCH
  ld32       $OUT_i_PTR, $mworker_base, $mzero, OUT_PTR_SCRATCH
  ld32step   $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  st32       $OUT_i_PTR, $mworker_base, $mzero, OUT_PTR_SCRATCH
  shr        $OUT_j_SIZE, $OUT_j_PTR, DELTAN_OFFSET_BITS-\LOG2_BYTE_ALIGNMENT
  shl        $OUT_j_PTR, $OUT_j_PTR, DELTAN_LENGTH_BITS+\LOG2_BYTE_ALIGNMENT
  shr        $OUT_j_PTR, $OUT_j_PTR, DELTAN_LENGTH_BITS
  add        $OUT_j_PTR, $OUT_j_PTR, $OUT_BASE

// ************************************************* //
// going to do 8 at a time and store remainder to memory
// ************************************************* //
  ld32       $NUM_PART_PTR, $mworker_base, $mzero, NP_PTR_SCRATCH
  ldz16step  $NUM_PART, $mzero, $NUM_PART_PTR+=, 1
  st32       $NUM_PART_PTR, $mworker_base, $mzero, NP_PTR_SCRATCH
  add        $NUM_PART, $NUM_PART, -1
  st32       $NUM_PART, $mzero, $mworker_base, NP_SCRATCH

  setzi      $IN_j_DELTA, 0

  br		 $IN_j_SIZE
FN_SIZE \FNAME
.endm

//******************************************************************************
// Function declarations
//******************************************************************************

// Use a macro so that although this file is included in multiple files the
// shared functions can be assembled only once
.macro CREATE_COMMON_FUNCTIONS

//******************************************************************************
// Initial function to load and process vertex state
// before entering any loops.  Called by all vertex variants
//******************************************************************************
FN_WORKER_ENTRY_POINT _Reduce_load_state_process_common
// ************************************************* //
// Load vertex state - common code
// ************************************************* //
  ld32       $IN_BASE, $mvertex_base, $mzero, IN_OFF/4
  ld32       $OUT_BASE, $mvertex_base, $mzero, OUT_OFF/4
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  ldz16      $OUT_i_PTR, $mvertex_base, $mzero, OUT_OFFSET/2
  ldz16      $IN_i_PTR, $mvertex_base, $mzero, IN_OFFSET/2
  ldz16      $NUM_PART_PTR, $mvertex_base, $mzero, NUM_PART_OFF/2
#else
  ld32       $OUT_i_PTR, $mvertex_base, $mzero, OUT_OFFSET/4
  ld32       $IN_i_PTR, $mvertex_base, $mzero, IN_OFFSET/4
  ld32       $NUM_PART_PTR, $mvertex_base, $mzero, NUM_PART_OFF/4
 #endif
// ************************************************* //
// Useful constants
// ************************************************* //
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  setzi    $SCRATCH, TMEM_REGION0_BASE_ADDR
#endif

// ************************************************* //
// Unpack scaled pointers
// ************************************************* //
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  shl        $OUT_i_PTR, $OUT_i_PTR, 2
  shl        $IN_i_PTR, $IN_i_PTR, 2
  shl        $NUM_PART_PTR, $NUM_PART_PTR, 2
  add        $OUT_i_PTR, $OUT_i_PTR, $SCRATCH
  add        $IN_i_PTR, $IN_i_PTR, $SCRATCH
  add        $NUM_PART_PTR, $NUM_PART_PTR, $SCRATCH
#endif

// ************************************************* //
// Extract size and bases
// ************************************************* //
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  shr        $OUT_i_SIZE, $OUT_BASE, DELTAN_BASE_BITS
  setzi      $SCRATCH, DELTAN_OFFSET_MASK
  and        $OUT_BASE, $OUT_BASE, $SCRATCH
  and        $IN_BASE, $IN_BASE, $SCRATCH
#else
  shuf8x8hi  $OUT_i_SIZE, $OUT_i_PTR, $OUT_BASE
  shr        $OUT_i_SIZE, $OUT_i_SIZE, 16
  or         $SCRATCH, $mzero, DELTAN_OFFSET_MASK_COMPL
  andc       $OUT_BASE, $OUT_BASE, $SCRATCH
  andc       $IN_BASE, $IN_BASE, $SCRATCH
  andc       $OUT_i_PTR, $OUT_i_PTR, $SCRATCH
  andc       $IN_i_PTR, $IN_i_PTR, $SCRATCH
#endif

// ************************************************* //
// Start loops, store in_i_ptr as rest every loop
// ************************************************* //
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  st32       $OUT_BASE, $mworker_base, $mzero, BASE_SCRATCH
  st32       $NUM_PART_PTR, $mworker_base, $mzero, NP_PTR_SCRATCH
  {st32      $OUT_i_PTR, $mworker_base, $mzero, OUT_PTR_SCRATCH
   setzi     $ZAACC, ZAACC_BITMASK}
  brnzdec    $OUT_i_SIZE, 1f
  exitz      $mzero

1:
  br         $IN_j_SIZE

//******************************************************************************
// Function called within each vertex loop to fetch and process pointers
// Called multiple times by all vertex variants
//******************************************************************************
FN_EXPORT _Reduce_ptr_fetch
  ld32step   $IN_j_PTR, $mzero, $IN_i_PTR+=, 1
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  shr        $IN_j_SIZE, $IN_j_PTR, DELTAN_OFFSET_BITS
  shl        $IN_j_PTR, $IN_j_PTR, DELTAN_LENGTH_BITS
#else
  shr        $IN_j_SIZE, $IN_j_PTR, DELTAN_OFFSET_BITS-SCALED_PTR64_SHIFTS
  shl        $IN_j_PTR, $IN_j_PTR, DELTAN_LENGTH_BITS+SCALED_PTR64_SHIFTS
#endif
  shr        $IN_j_PTR, $IN_j_PTR, DELTAN_LENGTH_BITS
  add        $IN_j_PTR, $IN_BASE, $IN_j_PTR
  mov        $SCRATCH, $IN_j_DELTA
  br		 $SCRATCH2

FN_SIZE _Reduce_load_state_process_common

//******************************************************************************
// Function called at the top of each outer loop.
// Called by all vertex variants
//******************************************************************************
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
INSTANTIATE_REDUCE_OUTER_LOOP_SETUP _Reduce_outer_loop_setup 0
#else
INSTANTIATE_REDUCE_OUTER_LOOP_SETUP _Reduce_outer_loop_setup_out_align32 SCALED_PTR32_SHIFTS
INSTANTIATE_REDUCE_OUTER_LOOP_SETUP _Reduce_outer_loop_setup_out_align64 SCALED_PTR64_SHIFTS
#endif

// ************************************************* //
// half float and half half vertices may need to
// load inputs that are only 16 bit aligned.
// They share this subroutine, which is therefore in its own
// section.
// ************************************************* //

FN_SECTION _Reduce_ld128_MIS_2
FN_EXPORT _Reduce_ld128_MIS_2
  and $SCRATCH2, $SCRATCH, 0x7
  brz $SCRATCH2, _aligned_case
  and $SCRATCH2, $SCRATCH, 0x2
  brz $SCRATCH2, _4_misaligned

  ldb16step $VALUES_0, $IN_j_PTR, $SCRATCH+=, 1
  ld32step  $ASCRATCH_0, $IN_j_PTR, $SCRATCH+=, 1
 {ld32step  $VALUES_2, $IN_j_PTR, $SCRATCH+=, 1
  roll16    $VALUES_0, $VALUES_0, $ASCRATCH_0};
 {ld32step  $ASCRATCH_0, $IN_j_PTR, $SCRATCH+=, 1
  roll16    $VALUES_1, $ASCRATCH_0, $VALUES_2};
 {ldb16step $VALUES_3, $IN_j_PTR, $SCRATCH+=, -7
  roll16    $VALUES_2, $VALUES_2, $ASCRATCH_0};
 {br        $OUT_j_SIZE
  roll16    $VALUES_3, $ASCRATCH_0, $VALUES_3}

_4_misaligned:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  ld32 $VALUES_2, $IN_j_PTR, $SCRATCH, 2
  ld32 $VALUES_3, $IN_j_PTR, $SCRATCH, 3
  br $OUT_j_SIZE

_aligned_case:
  ld64 $VALUES_0:1, $IN_j_PTR, $SCRATCH, 0
  ld64 $VALUES_2:3, $IN_j_PTR, $SCRATCH, 1
  br $OUT_j_SIZE


FN_SIZE _Reduce_ld128_MIS_2

// ************************************************* //
// Small but often referenced zero and load subroutine.
// ************************************************* //
FN_SECTION _Reduce_zero_and_load
FN_EXPORT _Reduce_zero_and_load
   {
    ld32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
    zero       $VALUES_0:1
  }
  {
    ld32       $NUM_PART, $mzero, $mworker_base, NP_SCRATCH
    uput       $FP_CLR, $ZAACC  // arf
  }
  {
    br         $SCRATCH2
    zero       $VALUES_2:3
  }
 FN_SIZE Reductions_common_zero_and_load

.endm

#endif
