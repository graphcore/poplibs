// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Contains functions to fill vectors using fill and fill2d vertices.
// Supports half, float, int and unsigned int types
/* -------------------------------------------------------------------------- */
#include "poplar/TileConstants.hpp"
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
#define aFILL_VALUE      a2
#define aFILL_VALUE_HI   a3
#define aFILL_VALUE_64   a2:3

// Mangled function names
#define FILL2D_HALF_FUNC __runCodelet_popops__Fill2d___half
#define FILL2D_FLOAT_FUNC __runCodelet_popops__Fill2d___float
#define FILL2D_INT_FUNC __runCodelet_popops__Fill2d___int
#define FILL2D_UINT_FUNC __runCodelet_popops__Fill2d___unsigned_int


#define FILL_HALF_FUNC __runCodelet_popops__Fill___half
#define FILL_FLOAT_FUNC __runCodelet_popops__Fill___float
#define FILL_INT_FUNC __runCodelet_popops__Fill___int
#define FILL_UINT_FUNC __runCodelet_popops__Fill___unsigned_int

// Input vertex structure offsets
#define VOFF_IN                  0
#define VOFF_OUT_PTR             1
#define VOFF_OUT_SIZE            2



.globl FILL2D_HALF_FUNC
.globl FILL2D_FLOAT_FUNC
.globl FILL2D_INT_FUNC
.globl FILL2D_UINT_FUNC

.type FILL2D_HALF_FUNC, @function
.type FILL2D_FLOAT_FUNC, @function
.type FILL2D_INT_FUNC, @function
.type FILL2D_UINT_FUNC, @function

.globl FILL_HALF_FUNC
.globl FILL_FLOAT_FUNC
.globl FILL_INT_FUNC
.globl FILL_UINT_FUNC

.type FILL_HALF_FUNC, @function
.type FILL_FLOAT_FUNC, @function
.type FILL_INT_FUNC, @function
.type FILL_UINT_FUNC, @function

//******************************************************************************
// Fill 2d vertex entry point.
// Fetches the vertex state which is specific to Fill2d.  Then uses the same
// body of code as the fill vertex to fill the memory.
// Organised so that the impact on memory and execution time for the fill
// vertex code relatively unaffected.
//******************************************************************************
DEF_STACK_USAGE 0 .text.fill2d
.section .text.fill2d

  .align 4

__popops__Fill2d:
FILL2D_FLOAT_FUNC:
FILL2D_INT_FUNC:
FILL2D_UINT_FUNC:
  setzi $LOG2_SIZEOF, 2
  ld32 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN
  bri 1f
FILL2D_HALF_FUNC:
  setzi $LOG2_SIZEOF, 1
  ldb16 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN

1:
  // Fetch vertex state: start and end of a list of pointers to data
  ld32    $OUT_LIST_PTR, $mvertex_base,  VOFF_OUT_PTR
  ld32    $OUT_LIST_COUNT, $mvertex_base,  VOFF_OUT_SIZE
  add     $OUT_LIST_COUNT, $OUT_LIST_COUNT, -1

  // Will loop back to this point for the next vector.
fill2d_loop:
  ld32step $OUT_PTR, $mzero, $OUT_LIST_PTR+=,1
  ld32step $OUT_SIZE, $mzero, $OUT_LIST_PTR+=,1
  bri      fill_2d_continue

.size __popops__Fill2d, .-__popops__Fill2d

//******************************************************************************
// Fill vertex entry
// Copes with data aligned to 2 byte boundary (half) as the minimum data size.
// Most of the code is shared with the fill2d vertex.
//******************************************************************************

DEF_STACK_USAGE 0 .text.fill
//Different section, enabling selection of the fill vertex only
.section .text.fill
 .align 8

__popops__Fill:
FILL_FLOAT_FUNC:
FILL_INT_FUNC:
FILL_UINT_FUNC:
  setzi $LOG2_SIZEOF, 2
  ld32 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN
  bri 1f
FILL_HALF_FUNC:
  setzi $LOG2_SIZEOF, 1
  ldb16 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN

1:
  // Fetch pointers to the start and end of the one area to fill
  ld32     $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR
  ld32     $OUT_SIZE, $mzero, $mvertex_base, VOFF_OUT_SIZE
  setzi    $OUT_LIST_COUNT,0
  // Entry point common with fill 2d
fill_2d_continue:
  // Only 2 byte aligned?
  and      $mSCRATCH, $OUT_PTR,2
  brz      $mSCRATCH, 1f
  // Write a first half
  andc       $OUT_PTR, $OUT_PTR, 3
  ld32       $aSCRATCH, $mzero, $OUT_PTR,0
  {
    sub        $OUT_SIZE, $OUT_SIZE, 1
    sort4x16lo $aSCRATCH, $aSCRATCH, $aFILL_VALUE
  }
  st32step   $aSCRATCH, $mzero, $OUT_PTR+=,1
1:
  // Generate a counter of the remaining bytes
  shl      $COUNTER, $OUT_SIZE, $LOG2_SIZEOF

  // We could be done, or just 2 (4 byte aligned) bytes to write
  brz      $COUNTER, 3f
  cmpult   $mSCRATCH, $COUNTER, 4
  brnz     $mSCRATCH, .Llast_half

  // At least 4 bytes to write - are they 8 byte aligned?
  and      $mSCRATCH, $OUT_PTR,4
  brz      $mSCRATCH, 1f

  // Write 4 bytes to get to 64 bit alignment or as there are 4 left
  st32step $aFILL_VALUE,$mzero,$OUT_PTR+=,1
  add      $COUNTER,$COUNTER,-4
1:
  {
    // Run the loop, which writes 8 bytes/pass
    shr      $LOOP_COUNTER,$COUNTER,3
    // Prepare two adjacent registers for 8 byte vectorised writes.
    mov      $aFILL_VALUE_HI, $aFILL_VALUE
  }
  rpt      $LOOP_COUNTER, (2f - 1f)/8 -1
1:
  { st64step $aFILL_VALUE_64, $mzero, $OUT_PTR+=, 1
    fnop }
2:
  // 0 2 4 or 6 bytes left to process
  // In lsbs of counter despite it not having been modified in the modulo 8 loop
  and     $mSCRATCH,$COUNTER,4
  brz     $mSCRATCH, 1f

  // Write 4 bytes
  st32step $aFILL_VALUE,$mzero,$OUT_PTR+=,1
1:
  // remaining 2 bytes?
  and        $mSCRATCH,$COUNTER,2
  brz        $mSCRATCH, 3f
.Llast_half:
  // Write the last 2 bytes
  ld32       $aSCRATCH, $mzero, $OUT_PTR,0
  sort4x16hi $aSCRATCH, $aFILL_VALUE, $aSCRATCH
  st32step   $aSCRATCH, $mzero, $OUT_PTR+=,1

  // Loop back in the 2d case, count is zero for the zero case
3:
  brnzdec    $OUT_LIST_COUNT, fill2d_loop
  exitz      $mzero

.size __popops__Fill, .-__popops__Fill

#endif
