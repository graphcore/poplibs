// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// Definitions for copying bytes (8bit types) used in dynamic slice/upadte, both
// 1d (Supervisor) and 2d, and multi slice/update code

#include "dynamicSlice.inc"

#define mVal1           m0
#define diff            m0
#define mMemVal         m1
#define mVal            m2
#define mVal0           m2
#define mask            m2
#define nStartBytes     m3
#define remainder       m3
#define numWords        m10
#define mVal2           m11
#define shifts          m11


//**************************************************************************
// Copies the starting 1-3 bytes required to align the destination pointer
// to 32 bit boundary.
//
// INPUT:
//   $mSrcPtr     : pointer to source data (can have any alignment)
//   $mDstPtr     : pointer to destination data
//   $mRegionSize : total number of bytes to copy
//   $diff        : offset (1-3) from 32 bit alignment of the dest pointer
//
// OUPTUT:
//   $mSrcPtr, $mDstPtr, $mRegionSize have been updated appropriately
//                                    (incr/decr) after copy
//**************************************************************************
.macro SLICE_COPY_START_BYTES
    // Need to preserve $diff bytes (least significant) from memory word
    // $mMemVal and 'mix'' $nStartBytes (i.e. 4 - $diff) bytes from source
    // (most significant)
    //
    // For instance, if $diff is 1, we need to mix 3 source bytes with 1 byte
    // from $mMemVal (note that addresses in memory are right to left below):
    //
    //     Most sig.                 Least sig.
    //      +-------+-------+-------+-------+
    //      | BYTE2 | BYTE1 | BYTE0 |   m0  |
    //      +-------+-------+-------+-------+
    //          3       2       1       0
    //      \----------------------/ \------/
    //            N_START_BYTES     *  DIFF
    //                              |
    //                              +--- $mDstPtr
    //
    // For this example we build a $mask 0x000000FF to mask the 'm0' byte of
    // $MEM_VAL, and then we put the three source BYTEx in the right position in
    // $MEM_VAL
    //
    // But if the total length ($mRegionSize) is less than $nStartBytes, we
    // need to use $mRegionSize, not (4 - $diff)). For instance, if $diff is
    // 1, but $mRegionSize is 2, we need to use $mask 0xFF0000FF for $mMemVal
    // to preserve 'm0' and 'm3':
    //
    //      +-------+-------+-------+-------+
    //      |   m3  | BYTE1 | BYTE0 |   m0  |
    //      +-------+-------+-------+-------+
    //          3       2       1       0
    //              \--mRegionSize--/
    //
    //      \----------------------/ \------/
    //           N_START_BYTES         DIFF

    sub         $mDstPtr, $mDstPtr, $diff // Align dstPtr to prev 4-b boundary
    ld32        $mMemVal, $mzero, $mDstPtr, 0  // get word from memory

    // nStartBytes = 4 - diff
    // nStartBytes = mRegionSize if mRegionSize < nStartBytes
    sub         $nStartBytes, 4, $diff
    cmpult      $mScratch, $mRegionSize, $nStartBytes
    movz        $nStartBytes, $mScratch, $mRegionSize

    // Make a mask with 1s for the bytes to preserve in $mMemVal
    add         $mask, $mzero, -1   // set MASK to 0xffffffff
    // Build mask with $nStartBytes bytes 'at the right' set to 1s
    sub         $mScratch, 4, $nStartBytes
    mul         $mScratch, $mScratch, 8
    shr         $mask, $mask, $mScratch
    // Move the mask $diff bytes to the left
    mul         $shifts, $diff, 8
    shl         $mask, $mask, $shifts

    // Now invert MASK and 'mask in' the bytes to preserve from MEM_VAL
    xnor        $mask, $mzero, $mask
    and         $mMemVal, $mMemVal, $mask

    // Here we are processing $nStartBytes bytes
    sub         $mRegionSize, $mRegionSize, $nStartBytes

    // Read $nStartBytes bytes form source and put them in the right place in
    // $mMemVal
    add         $nStartBytes, $nStartBytes, -1 // for brnzdec
.Loop_start_bytes\@:
    ldz8step    $mVal, $mzero, $mSrcPtr+=, 1
    shl         $mVal, $mVal, $shifts
    or          $mMemVal, $mMemVal, $mVal
    add         $shifts, $shifts, 8
    brnzdec     $nStartBytes, .Loop_start_bytes\@

    // Now in $mMemVal: the 1,2,3 copied bytes plus previous bytes from memory
    st32step    $mMemVal, $mzero, $mDstPtr+=, 1
.endm


//**************************************************************************
// Copies bytes from source to destination, using a loop that writes in the
// destination a full 4-byte word at a time (so destination nust be 32 bit
// aligned).
// Source can have any alignment.
//
// Note: need to make sure the first RPT in this macro is aligned
//
//   STRIDE       : either 1 or 6 for the 2d or 1d worker version
//
// INPUT:
//   $mSrcPtr     : pointer to source data (can have any alignment)
//   $mDstPtr     : pointer to destination data (32 bit aligned)
//   $numWords    : number of words (4 bytes) to copy
//
// OUPTUT:
//   $mSrcPtr     : updated (incremented) past the last byte copied
//   $mDstPtr     : updated (incremented) past the last byte copied
//**************************************************************************
.macro SLICE_COPY_ALIGN32  STRIDE

    // What is the alignment of src pointer?
    and         $diff, $mSrcPtr, 3
    brz         $diff, .Lsrc_4byte_aligned\@
    cmpeq       $mScratch, $diff, 2
    brnz        $mScratch , .Lsrc_2byte_aligned\@

    // =========== Src ptr aligned on 1 or 3 bytes boundary ==========
    // We can read: 1 byte, 1 halfword, 1 byte from source
    rpt         $numWords, (2f-1f)/8 - 1
1:
   {ldz8step    $mVal0, $mzero , $mSrcPtr+=, 1;fnop}
   {ldz16step   $mVal1, $mzero , $mSrcPtr+=, 1;fnop}
   {ldz8step    $mVal2, $mzero , $mSrcPtr+=, (4 * \STRIDE - 3);fnop}

   {sort4x16lo  $mVal1, $mVal1, $mVal2;fnop}
   {shl         $mVal0, $mVal0, 24;fnop}
   {roll8l      $mVal0, $mVal0, $mVal1;fnop}

   {st32step    $mVal0, $mzero, $mDstPtr+=, \STRIDE;fnop}
2:
    bri         .Lcopy_bulk_end\@

    nop
.Lsrc_2byte_aligned\@:
    // =========== Src ptr aligned on 2 bytes boundary ==========
    // We can read 2 halfwords from source

    add        $numWords, $numWords, -1  // One word done in prologue + epilogue

    // We can use ARF registers (VAL1, VAL2) for 16 bit loads
    ldb16step  $VAL1, $mzero , $mSrcPtr+=, 1;
    ldb16step  $VAL2, $mzero , $mSrcPtr+=, (2 * \STRIDE - 1);

    rpt        $numWords, (2f-1f)/8 -1
1:
   {ldb16step  $VAL1, $mzero , $mSrcPtr+=, 1
    sort4x16lo $VAL3, $VAL1, $VAL2}
   {ldb16step  $VAL2, $mzero , $mSrcPtr+=, (2 * \STRIDE - 1)
    fnop}
   {st32step   $VAL3, $mzero, $mDstPtr+=, \STRIDE
    fnop}
2:
    sort4x16lo $VAL3, $VAL1, $VAL2
    st32step   $VAL3, $mzero, $mDstPtr+=, \STRIDE

    bri         .Lcopy_bulk_end\@

.Lsrc_4byte_aligned\@:
    // =========== Src ptr aligned on 4 bytes boundary ==========
    // We can read source 1 word at at time

    rpt         $numWords, (2f-1f)/8 - 1
1:
   {ld32step    $mVal, $mzero, $mSrcPtr+=, \STRIDE; fnop}
   {st32step    $mVal, $mzero, $mDstPtr+=, \STRIDE; fnop}
2:
.Lcopy_bulk_end\@:
.endm


//**************************************************************************
// Copies the trailing 1-3 bytes after the main copy loop; The 0 remainder
// case is not handled here.
//
// INPUT:
//   $mSrcPtr    : pointer to source data (can have any alignment)
//   $mDstPtr    : pointer to destination data (32-bit aligned )
//   $remainder  : 1-3 bytes to copy
//
// OUPTUT:
//   $mSrcPtr     : modified to point one byte before its starting value
//   (NOTE that $mDstPtr IS NOT updated)
//**************************************************************************
.macro SLICE_COPY_REMAINDER
    // Here we have 1, 2 or 3 bytes of (trailing) remainder after 4- or
    // 8-byte main loop.
    // For instance, if $remainder is 3, we have to 'mix' 3 source bytes
    // (least significant) with 1 (most significant) byte from memory ($mMemVal)
    // Note that addresses in memory go from right to left below.
    //      +-------+-------+-------+-------+
    //      |   m3  | BYTE2 | BYTE1 | BYTE0 |
    //      +-------+-------+-------+-------+
    //          3       2       1       0
    //               \-------REMAINDER------/
    //
    // We'll build a mask to preserve the appropriate bytes of $mMemVal
    // (0xFF000000 in the example above) then add the source bytes

    ld32        $mMemVal, $mzero, $mDstPtr, 0  // get word from memory at dest.

    add         $mask, $mzero, -1  // set mask to 0xffffffff
    // Build mask with 4 - REMAINDER bytes 'at the left' set to all 1s
    mul         $mScratch, $remainder, 8
    shl         $mask, $mask, $mScratch
    // finally 'mask in' the bytes to preserve in $mMemVal
    and         $mMemVal, $mMemVal, $mask

    // Read $remainder bytes from source and accumulate them in the right
    // position inside $mScratch.
    // Need to read from src memory in inverse order, so we advance
    // $mSrcPtr += ($remainder-1) and then we'll post decrement at each read
    add         $mSrcPtr, $mSrcPtr, $remainder
    add         $mSrcPtr, $mSrcPtr, -1
    setzi       $mScratch, 0
    rpt         $remainder, ((2f - 1f) / 8) - 1
1:
   {shl         $mScratch, $mScratch, 8; fnop}
   {ldz8step    $mVal, $mzero, $mSrcPtr+=, -1; fnop}
   {or          $mScratch, $mScratch, $mVal; fnop}
2:
    // put together with $mMemVal and store
    or          $mMemVal, $mMemVal, $mScratch
    st32        $mMemVal, $mzero, $mDstPtr, 0
.endm

#endif // __IPU__
