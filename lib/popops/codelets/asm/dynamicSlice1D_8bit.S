// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// -----------------------------------------------------------------------------
// Dynamic Slice and Dynamic Update Slice 1D MultiVertex code for 8-bit types
//
// Workers 0-4 will copy only a number of full 32-bit words, where the
// DESTINATION is 32-bit aligned (so that they can do st32 instructions).
// Each worker will process a word every 6 (STRIDE=6).
//
// Workers 5 is the one that (beside processing its share of full words) will
// also process any starting 1-3 bytes to align the destination to to 32-bit
// boundary, and also any remainder 1-3 bytes at the end of the set of full
// words.
// -----------------------------------------------------------------------------

// Worker Registers.
// Note that these register definitions must agree with those in
// "SliceCopyFunction_8bit.inc"
#define loopCount         m1
#define atomCountPtr      m2
#define workerId          m4
#define subRegion         m5
#define subElem           m6
#define baseSlice         m7
#define srcOffs           m8
#define dstOffs           m9
#define atomsPerWorker    m9
#define baseElem          m11
#define storeOffset       m11

// We use some ARF registers as storage for parameters
#define aSrc              a3
#define aDst              a4
#define aUpdate           a5

//****************************************************************************
// The input structure parameters. All offsets are in words
//****************************************************************************
#define VOFF_OFFSET      0
#define VOFF_BASET       1
#define VOFF_SUBT        2
#define VOFF_BASE_ELEM   3
#define VOFF_SUB_ELEM    4
#define VOFF_REGION_SIZE 5

//****************************************************************************
// Offsets for $mworker_base local variable storage (all in word offsets)
//****************************************************************************
// Four words where we store the size of whole words (atoms) that we have to
// copy in the case of 0, 1, 2 or 3 bytes of starting bytes, before the 32-bit
// boundary.
#define WOFF_ATOMS       0 // must be a multiple of 2 (i.e. multiple of 8 bytes)
#define WOFF_ATOMS_DIFF1 1
#define WOFF_ATOMS_DIFF2 2
#define WOFF_ATOMS_DIFF3 3

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "SliceCopyFunction_8bit.inc"
#include "workDivision.h.S"

#define DSS_BOOL_FUNC __runCodelet_popops__DynamicSlice1D___bool
#define DUSS_BOOL_FUNC __runCodelet_popops__DynamicUpdateSlice1D___bool

#define DSS_CHAR_FUNC __runCodelet_popops__DynamicSlice1D___char
#define DUSS_CHAR_FUNC __runCodelet_popops__DynamicUpdateSlice1d___char

#define DSS_SCHAR_FUNC __runCodelet_popops__DynamicSlice1D___signed_char
#define DUSS_SCHAR_FUNC __runCodelet_popops__DynamicUpdateSlice1D___signed_char

#define DSS_UCHAR_FUNC __runCodelet_popops__DynamicSlice1D___unsigned_char
#define DUSS_UCHAR_FUNC __runCodelet_popops__DynamicUpdateSlice1D___unsigned_char

.global DSS_BOOL_FUNC
.type DSS_BOOL_FUNC, @function

.global DUSS_BOOL_FUNC
.type DUSS_BOOL_FUNC, @function

.global DSS_CHAR_FUNC
.type DSS_CHAR_FUNC, @function

.global DUSS_CHAR_FUNC
.type DUSS_CHAR_FUNC, @function

.global DSS_UCHAR_FUNC
.type DSS_UCHAR_FUNC, @function

.global DUSS_UCHAR_FUNC
.type DUSS_UCHAR_FUNC, @function

.global DSS_SCHAR_FUNC
.type DSS_SCHAR_FUNC, @function

.global DUSS_SCHAR_FUNC
.type DUSS_SCHAR_FUNC, @function


DEF_STACK_USAGE 0 .text.DynamicSlice1d_8bit // No function here use any stack

.section .text.DynamicSlice1d_8bit

.align 8
.worker

DUSS_BOOL_FUNC:
DUSS_CHAR_FUNC:
DUSS_UCHAR_FUNC:
DUSS_SCHAR_FUNC:
  setzi     $aUpdate,1
  ld32      $aDst, $mzero, $mvertex_base, VOFF_BASET
  ld32      $aSrc, $mzero, $mvertex_base, VOFF_SUBT
  bri       .Lds_super_common
// ************************************************* //
// Non Update version: clear a flag for no update,
// load base, sub array pointers
// Src = Base
// Dst = Sub
// ************************************************* //
DSS_BOOL_FUNC:
DSS_CHAR_FUNC:
DSS_UCHAR_FUNC:
DSS_SCHAR_FUNC:
  setzi     $aUpdate,0
  ld32      $aSrc, $mzero, $mvertex_base, VOFF_BASET
  ld32      $aDst, $mzero, $mvertex_base, VOFF_SUBT


// --------- Common path for updating/non updating version ----------
.Lds_super_common:
  // Load sub elem, decrement for use as a brnzdec loop counter
  ld32      $subElem, $mzero, $mvertex_base, VOFF_SUB_ELEM
  add       $subElem, $subElem, -1

  // Fetch worker ID, masking it out: 0..5
  get       $workerId, $WSR
  and       $workerId, $workerId, CSR_W_WSR__CTXTID_M1__MASK

  // Compute the number of atoms (full words) to copy for this worker, for the
  // case where there will be 0, 1, 2 or 3 starting bytes (to align to 32 bit
  // boundary). They will be stored in 4 words starting at WOFF_ATOMS, with
  // offset $storeOffset (so later we can index using $diff)
  //
  //                 Start
  //  $loopCount     bytes    $mRegionSize    $storeOffset
  //      3            0           N               0
  //      2            1         N - 1             3
  //      1            2         N - 2             2
  //      0            3         N - 3             1
  setzi     $loopCount, 3
  add       $atomCountPtr, $mworker_base, WOFF_ATOMS*4
  ld32      $mRegionSize, $mzero, $mvertex_base, VOFF_REGION_SIZE
  // Default of all zeros
  st64      $azeros, $mzero, $mworker_base, WOFF_ATOMS/2
  st64      $azeros, $mzero, $mworker_base, WOFF_ATOMS/2 + 1
divide_work_loop:
  brz       $mRegionSize, 1f // if we went down zero, bail out
  // Compute:
  //  $mRegionSize / 24 ==> $atomsPerWorker (quotient)

  DIVIDE_BY_WORKER $mRegionSize $workerId $mScratch $atomsPerWorker LOG2_HALF_ATOM_SIZE
  // store in the right place
  add       $storeOffset, $loopCount, 1
  and       $storeOffset, $storeOffset, 0x3
  stm32     $atomsPerWorker, $atomCountPtr, $storeOffset

  add       $mRegionSize, $mRegionSize, -1
  brnzdec   $loopCount, divide_work_loop
1:

  // Setup baseSlice: it starts as (*OFFSET)
  ld32      $baseSlice, $mzero, $mvertex_base, VOFF_OFFSET
  ld32      $baseSlice, $mzero, $baseSlice, 0

  ld32      $mRegionSize, $mzero, $mvertex_base, VOFF_REGION_SIZE

  zero      $subRegion

  // ------- Loop repeated for each '$mRegionSize' copied -------
.Lsub_loop_8bit:
  // Make sure baseSlice is within range of baseElem; wrap around if needed
  ld32      $baseElem, $mzero, $mvertex_base, VOFF_BASE_ELEM
  cmpult    $mScratch, $baseSlice, $baseElem
  brnz      $mScratch, 1f
  zero      $baseSlice
1:

  // Calculate incrementing offsets for the copy
  mov   $dstOffs, $subRegion
  mul   $srcOffs, $mRegionSize, $baseSlice

  // Update/ non update versions require source/dest offs to be swapped
  atom  $mScratch, $aUpdate
  brz   $mScratch, 1f
  mov   $srcOffs, $subRegion
  mul   $dstOffs, $mRegionSize, $baseSlice
1:

  // Now we have:
  //  source data is at            : $aSrc + $srcOffs
  //  dest data is at              : $aDst + $dstOffs
  //  total size of data           : $mRegionSize
  //  number of full aligned words : stored at WOFF_ATOMS[]

  // prepare pointers as expected by rest of code
  atom      $mScratch, $aSrc
  add       $mSrcPtr, $mScratch, $srcOffs
  atom      $mScratch, $aDst
  add       $mDstPtr, $mScratch, $dstOffs

  // Is the destination aligned to 32 bits?
  and       $diff, $mDstPtr, 3

  // Is this worker 5 (the last worker?)
  cmpeq     $mScratch, $workerId, (CTXT_WORKERS-1)
  brnz      $mScratch, .L_is_worker_5

  // ------- Worker Id 0-4
  // If we were not 32-bit aligned to start with, need just to increment the
  // pointers to align
  brz       $diff, .Lout_32bit_aligned_char
  sub       $mScratch, 4, $diff
  add       $mSrcPtr, $mSrcPtr, $mScratch
  add       $mDstPtr, $mDstPtr, $mScratch
  bri       .Lout_32bit_aligned_char

.L_is_worker_5:
  // ------- Worker Id 5; If misaligned, copy any starting bytes (1-3)
  brz       $diff, .Lno_start_bytes
  SLICE_COPY_START_BYTES
.Lno_start_bytes:
  // Will there be remainder bytes (0-3) after copying the set of full words?
  and       $remainder, $mRegionSize, 3
  brz       $remainder, .Lout_32bit_aligned_char
  // There is a remainder. Prepare pointers for copying of remainder
  sub       $mRegionSize, $mRegionSize, $remainder
  add       $mSrcPtr, $mSrcPtr, $mRegionSize
  add       $mDstPtr, $mDstPtr, $mRegionSize
  SLICE_COPY_REMAINDER
  // Restore pointers to start of 32-bit aligned area. Note that $mSrcPtr
  // was left decremented by one extra byte by SLICE_COPY_REMAINDER
  sub       $mSrcPtr, $mSrcPtr, $mRegionSize
  add       $mSrcPtr, $mSrcPtr, 1
  sub       $mDstPtr, $mDstPtr, $mRegionSize

.Lout_32bit_aligned_char:
  // Destination is aligned to 32 bit

  // The number of words to copy for this worker is at WOFF_ATOMS[$diff]
  shl       $diff, $diff, 2
  ld32      $numWords, $diff, $mworker_base, WOFF_ATOMS
  brz       $numWords, .Lno_full_words

  // Advance src/dst pointers to the initial position based on worker id.
  // Note: cannot use dummy load for src pointer as it might be misaligned
  shl       $mScratch, $workerId, 2
  add       $mSrcPtr, $mSrcPtr, $mScratch
  add       $mDstPtr, $mDstPtr, $mScratch

  // ===== Copy the main bulk of the data =====
  SLICE_COPY_ALIGN32  CTXT_WORKERS
.Lno_full_words:

  // Advance base slice
  add       $baseSlice, $baseSlice, 1
  // The next sub region is just region $mRegionSize bytes afterwards
  ld32      $mRegionSize, $mzero, $mvertex_base, VOFF_REGION_SIZE
  add       $subRegion, $subRegion, $mRegionSize

  brnzdec   $subElem, .Lsub_loop_8bit

  exitz     $mzero

.size DynamicSlice1d_8bit, .-DSS_BOOL_FUNC

#endif
