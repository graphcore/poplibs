// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Contains functions to fill vectors using fill and fill2d vertices.
// Supports half, float, int, unsigned int and all 8-bit types
/* -------------------------------------------------------------------------- */
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

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
#define FILL2D_BOOL_FUNC __runCodelet_popops__Fill2d___bool
#define FILL2D_CHAR_FUNC __runCodelet_popops__Fill2d___char
#define FILL2D_UCHAR_FUNC __runCodelet_popops__Fill2d___unsigned_char
#define FILL2D_SCHAR_FUNC __runCodelet_popops__Fill2d___signed_char
#define FILL2D_HALF_FUNC __runCodelet_popops__Fill2d___half
#define FILL2D_FLOAT_FUNC __runCodelet_popops__Fill2d___float
#define FILL2D_INT_FUNC __runCodelet_popops__Fill2d___int
#define FILL2D_UINT_FUNC __runCodelet_popops__Fill2d___unsigned_int
#define FILL2D_LONGLONG_FUNC __runCodelet_popops__Fill2d___long_long
#define FILL2D_ULONGLONG_FUNC __runCodelet_popops__Fill2d___unsigned_long_long


#define FILL_BOOL_FUNC __runCodelet_popops__Fill___bool
#define FILL_CHAR_FUNC __runCodelet_popops__Fill___char
#define FILL_UCHAR_FUNC __runCodelet_popops__Fill___unsigned_char
#define FILL_SCHAR_FUNC __runCodelet_popops__Fill___signed_char
#define FILL_HALF_FUNC __runCodelet_popops__Fill___half
#define FILL_FLOAT_FUNC __runCodelet_popops__Fill___float
#define FILL_INT_FUNC __runCodelet_popops__Fill___int
#define FILL_UINT_FUNC __runCodelet_popops__Fill___unsigned_int
#define FILL_ULONGLONG_FUNC __runCodelet_popops__Fill___unsigned_long_long
#define FILL_LONGLONG_FUNC __runCodelet_popops__Fill___long_long

// Input vertex structure offsets
#define VOFF_IN                  0
#define VOFF_OUT_PTR             1
#define VOFF_OUT_SIZE            2

// Input vertex structure offsets for 64-bit
#define VOFF_IN_64BIT            0
#define VOFF_OUT_PTR_64BIT       2
#define VOFF_OUT_SIZE_64BIT      3

//******************************************************************************
// Fill 2d vertex entry point for 16 and 32 bit types.
// Fetches the vertex state which is specific to Fill2d.  Then uses the same
// body of code as the fill vertex to fill the memory.
// Organised so that the impact on memory and execution time for the fill
// vertex code relatively unaffected.
//******************************************************************************
FN_WORKER_ENTRY_POINT FILL2D_FLOAT_FUNC
FN_EXPORT FILL2D_INT_FUNC
FN_EXPORT FILL2D_UINT_FUNC
  setzi $LOG2_SIZEOF, 2
  ld32 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN
  bri .Lfill_common
FN_SIZE FILL2D_FLOAT_FUNC


FN_WORKER_ENTRY_POINT FILL2D_HALF_FUNC
  setzi $LOG2_SIZEOF, 1
  ldb16 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN

.Lfill_common:
  // Fetch vertex state: start and end of a list of pointers to data
  ld32    $OUT_LIST_PTR, $mvertex_base,  VOFF_OUT_PTR
  ld32    $OUT_LIST_COUNT, $mvertex_base,  VOFF_OUT_SIZE
  add     $OUT_LIST_COUNT, $OUT_LIST_COUNT, -1

  // Will loop back to this point for the next vector.
fill2d_loop:
  ld32step $OUT_PTR, $mzero, $OUT_LIST_PTR+=,1
  ld32step $OUT_SIZE, $mzero, $OUT_LIST_PTR+=,1
  bri      fill_2d_continue
FN_SIZE FILL2D_HALF_FUNC

//******************************************************************************
// Fill vertex entry for 16 and 32 bit types.
// Copes with data aligned to 2 byte boundary (half) as the minimum data size.
// Most of the code is shared with the fill2d vertex.
//******************************************************************************

FN_WORKER_ENTRY_POINT FILL_FLOAT_FUNC
FN_EXPORT FILL_INT_FUNC
FN_EXPORT FILL_UINT_FUNC
  setzi $LOG2_SIZEOF, 2
  ld32 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN
  bri  .Lfill_fetch_pointers
FN_SIZE FILL_FLOAT_FUNC

FN_WORKER_ENTRY_POINT FILL_HALF_FUNC 8 nop
  setzi $LOG2_SIZEOF, 1
  ldb16 $aFILL_VALUE, $mzero, $mvertex_base, VOFF_IN

.Lfill_fetch_pointers:
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

FN_SIZE FILL_HALF_FUNC

//******************************************************************************
// inner loop used by 64-bit
//******************************************************************************
.macro INNER_LOOP_64BIT
rpt $OUT_SIZE, (2f - 1f) / 8 - 1
1:
  {
    st64step $aFILL_VALUE_64, $mzero, $OUT_PTR+=, 1
    fnop
  }
2:
.endm

//******************************************************************************
// Fill vertex entry for 64-bit types (ULONGLONG and LONGLONG)
//******************************************************************************

FN_WORKER_ENTRY_POINT FILL_ULONGLONG_FUNC 8
FN_EXPORT FILL_LONGLONG_FUNC

  ld64     $aFILL_VALUE_64, $mzero, $mvertex_base, VOFF_IN_64BIT
  ld32     $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_64BIT
  ld32     $OUT_SIZE, $mzero, $mvertex_base, VOFF_OUT_SIZE_64BIT
  INNER_LOOP_64BIT
  exitz      $mzero
FN_SIZE FILL_ULONGLONG_FUNC

//******************************************************************************
// Fill 2D vertex entry for 64-bit types (ULONGLONG and LONGLONG)
//******************************************************************************

FN_WORKER_ENTRY_POINT FILL2D_ULONGLONG_FUNC 8 nop
FN_EXPORT FILL2D_LONGLONG_FUNC
  ld64    $aFILL_VALUE_64, $mzero, $mvertex_base, VOFF_IN_64BIT
  ld32    $OUT_LIST_PTR, $mvertex_base, VOFF_OUT_PTR_64BIT
  ld32    $OUT_LIST_COUNT, $mvertex_base, VOFF_OUT_SIZE_64BIT
  add     $OUT_LIST_COUNT, $OUT_LIST_COUNT, -1

fill2d_64bit_loop:
  ld32step  $OUT_PTR, $mzero, $OUT_LIST_PTR+=, 1
  ld32step  $OUT_SIZE, $mzero, $OUT_LIST_PTR+=, 1
  INNER_LOOP_64BIT
  brnzdec $OUT_LIST_COUNT, fill2d_64bit_loop
  exitz      $mzero
FN_SIZE FILL2D_ULONGLONG_FUNC

//******************************************************************************
// Fill vertex entry for 8-bit types types (char and bool)
// The code is significantly bigger than 16/32 bit types to warrant separate
// sections.
// First, if necessary, we fill 1, 2 or 3 bytes, to advance the pointer to
// 32-bit (4-byte) alignment boundary, then (if output area is big enough so
// that we gain overall) to 64-bit, so that we can do 64 bit stores.
// Then we process the possible remainder (1 word and then 3,2,1 bytes)
//******************************************************************************
#define FILL_VALUE        m3
#define FILL_VALUE4       m8
#define DIFF              m9
#define REMAINDER         m9
#define REMAINDER_WORD    m10
#define MEM_VAL           m10
#define mSCRATCH2         m11
#define N_START_FILL      m11
#define MASK              m5

// One word in scratch area used to move from MFR reg to ARF reg
#define SCRATCH_AREA_TMP_OFFS   0

FN_WORKER_ENTRY_POINT FILL2D_BOOL_FUNC
FN_EXPORT FILL2D_CHAR_FUNC
FN_EXPORT FILL2D_UCHAR_FUNC
FN_EXPORT FILL2D_SCHAR_FUNC
  ld32    $OUT_LIST_PTR, $mvertex_base,  VOFF_OUT_PTR
  ld32    $OUT_LIST_COUNT, $mvertex_base,  VOFF_OUT_SIZE
  ldz8    $FILL_VALUE, $mzero, $mvertex_base, VOFF_IN
  add     $OUT_LIST_COUNT, $OUT_LIST_COUNT, -1

  // Will loop back to this point for the next vector.
fill2d_8bit_loop:
  ld32step $OUT_PTR, $mzero, $OUT_LIST_PTR+=,1
  ld32step $OUT_SIZE, $mzero, $OUT_LIST_PTR+=,1
  bri      fill_2d_8bit_continue
FN_SIZE FILL2D_BOOL_FUNC

FN_WORKER_ENTRY_POINT FILL_BOOL_FUNC 8
FN_EXPORT FILL_CHAR_FUNC
FN_EXPORT FILL_UCHAR_FUNC
FN_EXPORT FILL_SCHAR_FUNC
  ld32      $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR
  ld32      $OUT_SIZE, $mzero, $mvertex_base, VOFF_OUT_SIZE
  ldz8      $FILL_VALUE, $mzero, $mvertex_base, VOFF_IN
  setzi     $OUT_LIST_COUNT,0

fill_2d_8bit_continue:
  // Broadcast the fill value 4 times into a whole register
  shuf8x8lo   $FILL_VALUE4, $FILL_VALUE, $FILL_VALUE
  sort4x16lo  $FILL_VALUE4, $FILL_VALUE4, $FILL_VALUE4

  // Is destination aligned to 4 bytes boundary?
  and       $DIFF, $OUT_PTR, 3
  brz       $DIFF, .Ldst_aligned

  // Dst ptr NOT aligned: need to advance $DIFF bytes (1,2 or 3) until it is.

  sub       $OUT_PTR, $OUT_PTR, $DIFF // Align dstPtr to prev 4-byte boundary
  ld32      $MEM_VAL, $mzero, $OUT_PTR, 0 // Get word from memory at aligned ptr

  // Need to preserve $DIFF bytes (least significant) from memory word $MEM_VAL
  // and 'mix'' $N_START_FILL (3, 2 or 1) fill bytes (most significant)
  //
  // For instance, if $DIFF is 1, we need to mix 3 fill values with 1 byte from
  // $MEM_VAL:
  //
  //     Most sig.                 Least sig.
  //      +-------+-------+-------+-------+
  //      |  FILL |  FILL |  FILL |   m0  |
  //      +-------+-------+-------+-------+
  //          3       2       1       0
  //      \----------------------/ \------/
  //            N_START_FILL         DIFF
  // For this example we build a mask 0xFFFFFF00 to mask $FILL_VALUE4, then we
  // invert it to 0x000000FF to mask the 'm0' byte of $MEM_VAL, and then we
  // OR the two together

  // But if the total length (OUT_SIZE) is less than N_START_FILL, we need to
  // use OUT_SIZE, not N_START_FILL. For instance, if DIFF is 1, but OUT_SIZE
  // is 2, we need to use mask 0x00FFFF00 for $FILL_VALUE4 and 0xFF0000FF for
  // MEM_VAL
  //
  //      +-------+-------+-------+-------+
  //      |   m3  |  FILL |  FILL |   m0  |
  //      +-------+-------+-------+-------+
  //          3       2       1       0
  //              \---OUT_SIZE---/
  //
  //      \----------------------/ \------/
  //            N_START_FILL         DIFF

  // START_FILL = 4-DIFF
  // START_FILL = OUT_SIZE if OUT_SIZE < N_START_FILL
  sub       $N_START_FILL, 4, $DIFF
  cmpult    $mSCRATCH, $OUT_SIZE, $N_START_FILL
  movnz     $N_START_FILL, $mSCRATCH, $OUT_SIZE

  // Make a mask with 1s for the bytes to use from FILL_VALUE4
  add       $MASK, $mzero, -1   // set MASK to 0xffffffff
  // Build mask with N_START_FILL bytes 'at the right' set to 1s
  sub       $mSCRATCH, 4, $N_START_FILL
  mul       $mSCRATCH, $mSCRATCH, 8
  shr       $MASK, $MASK, $mSCRATCH
  // Move the mask $DIFF bytes to the left
  mul       $mSCRATCH, $DIFF, 8
  shl       $MASK, $MASK, $mSCRATCH

  // 'Mask in' all the required fill bytes
  and       $mSCRATCH, $FILL_VALUE4, $MASK

  // Now invert MASK and 'mask in' the bytes to preserve FROM MEM_VAL
  xnor      $MASK, $mzero, $MASK
  and       $MEM_VAL, $MEM_VAL, $MASK

  // Mix together and write
  or        $MEM_VAL, $MEM_VAL, $mSCRATCH
  st32step  $MEM_VAL, $mzero, $OUT_PTR+=, 1

  // Above we have already processed $N_START_FILL bytes
  sub       $OUT_SIZE, $OUT_SIZE, $N_START_FILL

.Ldst_aligned:
  // Here destination pointer is aligned to 4 byte, so we can write
  // 4 bytes to memory with a single st32

  // Keep track of how many 4-byte words we need to process and how many
  // bytes of REMAINDER (0,1,2,3) we have from that
  and         $REMAINDER, $OUT_SIZE, 3
  shr         $LOOP_COUNTER, $OUT_SIZE, 2
  brz         $LOOP_COUNTER, .Lprocess_REMAINDER

  // Can we do 64 bit stores? We must have at least 2 words, aligned to 64 bit,
  // but then there are several extra instructions of overhead so we want
  // to check that we have enough words so that we gain overall.
  cmpult      $mSCRATCH, $LOOP_COUNTER, 17
  brnz        $mSCRATCH, .Luse_32_bit_loop
  // enough words, we can use 64 bit stores

  // If not aligned to 64 bit boundary, store one word to get there
  and         $mSCRATCH, $OUT_PTR, 7
  brz         $mSCRATCH, 1f
  st32step    $FILL_VALUE4, $mzero, $OUT_PTR+=, 1
  add         $LOOP_COUNTER, $LOOP_COUNTER, -1
1:
  // Here, output pointer aligned to 64 bits
  and         $REMAINDER_WORD, $LOOP_COUNTER, 1
  shr         $LOOP_COUNTER, $LOOP_COUNTER, 1

  // Extend fill value to 64 bit (in ARF), using one word of scratch area
  st32        $FILL_VALUE4, $mzero, $mworker_base, SCRATCH_AREA_TMP_OFFS
  ld32        $aFILL_VALUE, $mzero, $mworker_base, SCRATCH_AREA_TMP_OFFS
  mov         $aFILL_VALUE_HI, $aFILL_VALUE
  rpt         $LOOP_COUNTER, 0
 {st64step    $aFILL_VALUE_64, $mzero, $OUT_PTR+=, 1
  fnop}
  // Do we have a single full 32 bit word to write?
  brz         $REMAINDER_WORD, .Lprocess_REMAINDER
  st32step    $FILL_VALUE4, $mzero, $OUT_PTR+=, 1
  bri         .Lprocess_REMAINDER

.Luse_32_bit_loop:
  // not enough words to justify overhead of setting up 64-bit loop.
  rpt         $LOOP_COUNTER, 0
 {st32step    $FILL_VALUE4, $mzero, $OUT_PTR+=, 1
  fnop}

.Lprocess_REMAINDER:
  // Here we have 0, 1, 2 or 3 bytes of (trailing) REMAINDER after 4- or
  // 8-byte main loop
  // For instance, if $REMAINDER is 2, we have to 'mix' doubled fill bytes
  // (least significant) with most significant 2 bytes from memory.
  //      +-------+-------+-------+-------+
  //      |   m3  |   m2  |  FILL |  FILL |
  //      +-------+-------+-------+-------+
  //          3       2       1       0
  //                       \--------------/
  //                          REMAINDER
  //
  // We'll build a mask to get the appropriate bytes from $FILL_VALUE4, then
  // invert it to get the appropriate bytes from memory
  brz       $REMAINDER, fill_2d_8bit_end

  ld32      $MEM_VAL, $mzero, $OUT_PTR, 0  // get word from memory

  add       $MASK, $mzero, -1  // set mask to 0xffffffff
  // Build mask with REMAINDER bytes 'at the right' set to all 1s
  sub       $mSCRATCH, 4, $REMAINDER
  mul       $mSCRATCH, $mSCRATCH, 8
  shr       $MASK, $MASK, $mSCRATCH

  // 'Mask in' all the required fill bytes
  and       $mSCRATCH, $FILL_VALUE4, $MASK

  // Now invert MASK and 'mask in' the bytes to preserve from MEM_VAL
  xnor      $MASK, $mzero, $MASK
  and       $MEM_VAL, $MEM_VAL, $MASK

  // Mix together and write
  or        $MEM_VAL, $MEM_VAL, $mSCRATCH
  st32      $MEM_VAL, $mzero, $OUT_PTR, 0
fill_2d_8bit_end:
  brnzdec    $OUT_LIST_COUNT, fill2d_8bit_loop
  exitz      $mzero
FN_SIZE FILL_BOOL_FUNC


#endif
