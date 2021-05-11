// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "SliceCopyFunction_8bit.inc"

/* -------------------------------------------------------------------------- */
// Common copy function used bt next codelets:
// - DynamicSlice2d,
// - DynamicUpdateSlice2d
// - MultiSlice
// - MultiUpdate
/* -------------------------------------------------------------------------- */


//******************************************************************************
// Labels names for each variant
//******************************************************************************
.globl Slice_copy_function_8bit
.type Slice_copy_function_8bit, @function

.globl Slice_copy_function_half
.type Slice_copy_function_half, @function

.globl Slice_copy_function_float_int
.type Slice_copy_function_float_int @function

//******************************************************************************
// Copy function for 8 bit values
// Inputs:
//    mReturnAddress (m7)  - return address
//    mSrcPtr        (m8)  - source address
//    mDstPtr        (m9)  - destination address
//    mRegionSize    (m10) - region size in elements (bytes)
//
// This function is allowed to use/modify m8, m9, m10, m11 (mScratch)
// and all ARF registers but must preserve all other MRF registers (m0..m7).
//
// To save/restore the registers used here, this function uses a few words in
// the scratch area, at offsets WOFF_SAVE_REG1..n
//******************************************************************************
.section .text.Slice_copy_8bit
.align 8
Slice_copy_function_8bit:
    brz         $mRegionSize, Slice_copy_8bit_end

    // Save previous value of used registers in scratch area
    st32        $diff, $mzero, $mworker_base, WOFF_SAVE_REGS1
    st32        $mMemVal, $mzero, $mworker_base, WOFF_SAVE_REGS2
    st32        $mVal, $mzero, $mworker_base, WOFF_SAVE_REGS3
    st32        $remainder, $mzero, $mworker_base, WOFF_SAVE_REGS4
    // Is destination aligned to 4 bytes boundary?
    and         $diff, $mDstPtr, 3
    brz         $diff, .Ldst_4byte_aligned

    SLICE_COPY_START_BYTES

.Ldst_4byte_aligned:
    // How many 4-byte words do we need to process and how many bytes of
    // remainder (0,1,2,3)?
    and         $remainder, $mRegionSize, 3
    shr         $numWords, $mRegionSize, 2
    brz         $numWords, .Lzero_full_words

    // Here destination pointer is aligned to 4 bytes.
    SLICE_COPY_ALIGN32  1
.Lzero_full_words:

    // If 0 bytes of remainder, bail out
    brz         $remainder, .Lrestore_regs
    SLICE_COPY_REMAINDER

.Lrestore_regs:
    // Restore previous registers from scratch area
    ld32        $diff, $mzero, $mworker_base, WOFF_SAVE_REGS1
    ld32        $mMemVal, $mzero, $mworker_base, WOFF_SAVE_REGS2
    ld32        $mVal, $mzero, $mworker_base, WOFF_SAVE_REGS3
    ld32        $remainder, $mzero, $mworker_base, WOFF_SAVE_REGS4
Slice_copy_8bit_end:
    br          $mReturnAddress
.size Slice_copy_function_8bit, .-Slice_copy_function_8bit


//******************************************************************************
// Copy function for half
// Inputs (will/migth be reused inside copy function):
//         m7  - return address (from call m7, imm)
//         m8  - source address
//         m9  - destination address
//         m10 - region size in elements
// Register used by copy function:
//         m11 - scratch register
//         a0  - temp register
//         a1  - temp register
//         a2  - temp register
// This function is allowed to use/modify m8, m9, m10 (above), m11 (mScratch)
// and all ARF registers but must preserve all other MRF registers (m0..m7).
//******************************************************************************
.section .text.Slice_copy_half

.align 8
Slice_copy_half:
    // nop for rpt loop alignment
    nop
Slice_copy_function_half:
    // 32/16 alignment half copy begins

    // Aligned output?
    and      $mScratch, $mDstPtr, 2
    brz      $mScratch, 1f

    // Output not aligned -load/store to force it to be aligned for the loop
    ldb16step $VAL1,        $mzero,       $mSrcPtr+=, 1
    add       $mDstPtr,     $mDstPtr,     -2
    ldb16     $VAL2,        $mzero,       $mDstPtr, 0
    {add      $mRegionSize, $mRegionSize, -1;
     roll16   $VAL1,        $VAL2,        $VAL1}
    st32step  $VAL1,        $mzero,       $mDstPtr+=, 1
1:
    // input aligned?
    and     $mScratch, $mSrcPtr, 2
    brz     $mScratch, 5f

    // Case for misaligned input
    shr       $mScratch, $mRegionSize, 2
    ldb16step $VAL1,     $mzero,       $mSrcPtr+=, 1

    // Copy 2 pairs of items per loop, output is 32 bit aligned, input is not
    rpt $mScratch, ((2f - 1f) / 8) - 1
1:
    {ld32step $VAL2, $mzero, $mSrcPtr+=, 1; fnop}
    {ld32step $VAL3, $mzero, $mSrcPtr+=, 1; roll16 $VAL1, $VAL1,  $VAL2}
    {st32step $VAL1, $mzero, $mDstPtr+=, 1; roll16 $VAL2, $VAL2,  $VAL3}
    {st32step $VAL2, $mzero, $mDstPtr+=, 1; mov    $VAL1, $VAL3}
2:
    // Are there 32 bits or more bits left?
    and       $mScratch, $mRegionSize, 2
    brz       $mScratch, 1f

    // Store remaining 32 bits, fetch 16 more incase we have 1 more to store
    ldb16step $VAL2, $mzero, $mSrcPtr+=, 1
    roll16    $VAL1, $VAL1,  $VAL2
    st32step  $VAL1, $mzero, $mDstPtr+=, 1
    ldb16step $VAL1, $mzero, $mSrcPtr+=, 1
1:
    // Is there a last one?  We have already fetched it if so
    and     $mScratch, $mRegionSize,1
    brz     $mScratch, Slice_copy_half_end
    bri     3f
    // nop for rpt loop alignment
    nop
5:
    // Case for aligned input
    shr     $mScratch, $mRegionSize, 1

    // Copy pairs of items, both 32 bit aligned
    rpt $mScratch, ((2f - 1f) / 8) - 1
1:
    {ld32step    $VAL1, $mzero, $mSrcPtr+=, 1; fnop}
    {st32step    $VAL1, $mzero, $mDstPtr+=, 1; fnop}
2:
    // Is there a last one?
    and       $mScratch, $mRegionSize, 1
    brz       $mScratch, Slice_copy_half_end
    ldb16step $VAL1,     $mzero, $mSrcPtr+=, 1
3:
    // Write the 1st 2 bytes into the last 4 byte word - preserve bytes 3,4
    ldb16    $VAL2, $mzero, $mDstPtr, 1
    roll16   $VAL1, $VAL1,  $VAL2
    st32step $VAL1, $mzero, $mDstPtr+=, 1

Slice_copy_half_end:
    br       $mReturnAddress

.size Slice_copy_half, .-Slice_copy_half


//******************************************************************************
// Copy function for float and int
// Inputs (will/migth be reused inside copy function):
//         m7  - return address (from call m7, imm)
//         m8  - source address
//         m9  - destination address
//         m10 - region size in elements
// Register used by copy function:
//         m11 - scratch register
//         a0  - temp register
//         a1  - temp register
//         a2  - temp register
// This function is allowed to use/modify m8, m9, m10 (above), m11 (mScratch)
// and all ARF registers but must preserve all other MRF registers (m0..m7).
//******************************************************************************
.section .text.Slice_copy_float_int

.align 8
Slice_copy_float_int:
    // nop for rpt loop alignment
    nop
Slice_copy_function_float_int:
    // Aligned output?
    and      $mScratch, $mDstPtr, 4
    brz      $mScratch, 1f

    // Output not aligned -load/store a word to force it to be aligned
    // for the loop
    ld32step $VAL1, $mzero, $mSrcPtr+=, 1
    st32step $VAL1, $mzero, $mDstPtr+=, 1
    add      $mRegionSize,  $mRegionSize, -1
1:
    // input aligned?
    and      $mScratch, $mSrcPtr, 4
    brz      $mScratch, 5f
    shr      $mScratch, $mRegionSize, 1

    // Copy pairs of items - output is 64 bit aligned input is not
    rpt      $mScratch, ((2f - 1f) / 8) - 1
1:
    {ld32step $VAL1,  $mzero, $mSrcPtr+=,  1; fnop}
    {ld32step $VAL2,  $mzero, $mSrcPtr+=,  1; fnop}
    {st64step $VAL12, $mzero, $mDstPtr+=,  1; fnop}
2:
    and      $mScratch, $mRegionSize,1
    brz      $mScratch, Slice_copy_float_int_end
    bri      3f
    // nop for rpt loop alignment
    nop
5:
    // Copy pairs of items - both input and output are 64 bit aligned
    shr      $mScratch, $mRegionSize, 1
    rpt     $mScratch, ((2f - 1f) / 8) - 1
1:
    {ld64step $VAL12, $mzero, $mSrcPtr+=,  1; fnop}
    {st64step $VAL12, $mzero, $mDstPtr+=,  1; fnop}
2:
    // Is there a last one?
    and      $mScratch, $mRegionSize,1
    brz      $mScratch, Slice_copy_float_int_end
3:
    // Copy the last 32 bits
    ld32step $VAL1, $mzero, $mSrcPtr+=,  1
    st32step $VAL1, $mzero, $mDstPtr+=,  1

Slice_copy_float_int_end:
    br       $mReturnAddress

.size Slice_copy_float_int, .-Slice_copy_float_int

#endif
