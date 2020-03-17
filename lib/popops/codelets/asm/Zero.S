// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Contains functions to zero vectors using zero and zero2d vertices.
// Supports half, float and int types
/* -------------------------------------------------------------------------- */
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
// Register aliases
#define mSCRATCH         m0
#define OUT_PTR          m1
#define OUT_SIZE         m2
#define COUNTER          m3
#define LOOP_COUNTER     m4

#define OUT_LIST_PTR     m6
#define OUT_LIST_COUNT   m7

#define LOG2_SIZEOF      m8

#define aSCRATCH         a0

// Mangled function names
#define ZERO2D_HALF_FUNC __runCodelet_popops__Zero2d___half
#define ZERO2D_FLOAT_FUNC __runCodelet_popops__Zero2d___float
#define ZERO2D_INT_FUNC __runCodelet_popops__Zero2d___int
#define ZERO2D_UINT_FUNC __runCodelet_popops__Zero2d___unsigned_int


#define ZERO_HALF_FUNC __runCodelet_popops__Zero___half
#define ZERO_FLOAT_FUNC __runCodelet_popops__Zero___float
#define ZERO_INT_FUNC __runCodelet_popops__Zero___int
#define ZERO_UINT_FUNC __runCodelet_popops__Zero___unsigned_int

/* Input vertex structure offsets */
#define VOFF_OUT_PTR             0
#define VOFF_OUT_SIZE            1



.globl ZERO2D_HALF_FUNC
.globl ZERO2D_FLOAT_FUNC
.globl ZERO2D_INT_FUNC
.globl ZERO2D_UINT_FUNC

.type ZERO2D_HALF_FUNC, @function
.type ZERO2D_FLOAT_FUNC, @function
.type ZERO2D_INT_FUNC, @function
.type ZERO2D_UINT_FUNC, @function

.globl ZERO_HALF_FUNC
.globl ZERO_FLOAT_FUNC
.globl ZERO_INT_FUNC
.globl ZERO_UINT_FUNC

.type ZERO_HALF_FUNC, @function
.type ZERO_FLOAT_FUNC, @function
.type ZERO_INT_FUNC, @function
.type ZERO_UINT_FUNC, @function

//******************************************************************************
// Zero 2d vertex entry point.
// Fetches the vertex state which is specific to Zero2d.  Then uses the same
// body of code as the zero vertex to zero memory.
// Organised so that the impact on memory and execution time for the zero
// vertex code relativelt unaffected.
//******************************************************************************
DEF_STACK_USAGE 0 .text.zero2d
.section .text.zero2d

  .align 4

__popops__Zero2d:
ZERO2D_FLOAT_FUNC:
ZERO2D_INT_FUNC:
ZERO2D_UINT_FUNC:
  setzi $LOG2_SIZEOF, 2
  bri 1f
ZERO2D_HALF_FUNC:
  setzi $LOG2_SIZEOF, 1

1:
  // Fetch vertex state: start and end of a list of pointers to data
  ld32    $OUT_LIST_PTR, $mvertex_base,  VOFF_OUT_PTR
  ld32    $OUT_LIST_COUNT, $mvertex_base,  VOFF_OUT_SIZE
  add     $OUT_LIST_COUNT, $OUT_LIST_COUNT, -1

  // Will loop back to this point for the next vector.
zero2d_loop:
  ld32step $OUT_PTR, $mzero, $OUT_LIST_PTR+=,1
  ld32step $OUT_SIZE, $mzero, $OUT_LIST_PTR+=,1
  bri      zero_2d_continue

.size __popops__Zero2d, .-__popops__Zero2d

//******************************************************************************
// Zero vertex entry
// Copes with data aligned to 2 byte boundary (half) as the minimum data size.
// Most of the code is shared with the zero2d vertex.
//******************************************************************************

DEF_STACK_USAGE 0 .text.zero
//Different section, enabling selection of the zero vertex only
.section .text.zero
 .align 8

__popops__Zero:
ZERO_FLOAT_FUNC:
ZERO_INT_FUNC:
ZERO_UINT_FUNC:
  setzi $LOG2_SIZEOF, 2
  bri 1f
ZERO_HALF_FUNC:
  setzi $LOG2_SIZEOF, 1

1:
  // Fetch pointers to  start and end of the one area to zero
  ld32     $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR
  ld32     $OUT_SIZE, $mzero, $mvertex_base, VOFF_OUT_SIZE
  setzi    $OUT_LIST_COUNT,0
  // Entry point common with zero 2d
zero_2d_continue:
  // Only 2 byte aligned?
  and      $mSCRATCH, $OUT_PTR,2
  brz      $mSCRATCH, 1f
  // Write a first half
  andc       $OUT_PTR, $OUT_PTR, 3
  ld32       $aSCRATCH, $mzero, $OUT_PTR,0
  {
    sub        $OUT_SIZE, $OUT_SIZE, 1
    sort4x16lo $aSCRATCH, $aSCRATCH, $azero
  }
  st32step   $aSCRATCH, $mzero, $OUT_PTR+=,1
1:
  // generate a byte counter, excluding the 1st item.
  shl      $COUNTER, $OUT_SIZE, $LOG2_SIZEOF

  // We could be done, or just 2 (4 byte aligned) bytes to write
  brz      $COUNTER, 3f
  cmpult   $mSCRATCH, $COUNTER, 4
  brnz     $mSCRATCH, .Llast_half

  // At least 4 bytes to write - are they 8 byte aligned?
  and      $mSCRATCH, $OUT_PTR,4
  brz      $mSCRATCH, 1f

  // Write 4 bytes to get to 64 bit alignment or as there are 4 left
  st32step $azero,$mzero,$OUT_PTR+=,1
  add      $COUNTER,$COUNTER,-4
1:
  // Run the loop, which writes 8 bytes/pass
  shr     $LOOP_COUNTER,$COUNTER,3

  {rpt    $LOOP_COUNTER, (2f - 1f)/8 -1
   fnop}
1:
  { st64step $azeros, $mzero, $OUT_PTR+=, 1
    fnop }
2:
  // 0 2 4 or 6 bytes left to process
  // In lsbs of counter despite it not having been modified in the modulo 8 loop
  and     $mSCRATCH,$COUNTER,4
  brz     $mSCRATCH, 1f

  // Write 4 bytes
  st32step $azero,$mzero,$OUT_PTR+=,1
1:
  // remaining 2 bytes?
  and        $mSCRATCH,$COUNTER,2
  brz        $mSCRATCH, 3f
.Llast_half:
  // Write the last 2 bytes
  ld32       $aSCRATCH, $mzero, $OUT_PTR,0
  sort4x16hi $aSCRATCH, $azero, $aSCRATCH
  st32step   $aSCRATCH, $mzero, $OUT_PTR+=,1

  // Loop back in the 2d case, count is zero for the zero case
3:
  brnzdec    $OUT_LIST_COUNT, zero2d_loop
  exitz      $mzero

.size __popops__Zero, .-__popops__Zero

#endif
