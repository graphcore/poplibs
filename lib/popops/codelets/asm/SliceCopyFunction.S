// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

/* -------------------------------------------------------------------------- */
// Common copy function used bt next codelets:
// - DynamicSlice2d, 
// - DynamicUpdateSlice2d
// - MultiSlice
// - MultiUpdate
/* -------------------------------------------------------------------------- */
// Register aliases
#define mReturnAddress m7
#define mSrcPtr        m8
#define mDstPtr        m9
#define mRegionSize    m10
#define mScratch       m11

#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1
#define VAL3    a2

//******************************************************************************
// Labels names for each variant
//******************************************************************************
.globl Slice_copy_function_half
.type Slice_copy_function_half, @function

.globl Slice_copy_function_float_int
.type Slice_copy_function_float_int @function

//******************************************************************************
// Copy function for half
// Inputs (will/migth be reused insode copy function): 
//         m7  - return address (from call m7, imm)
//         m8  - source address
//         m9  - destination address
//         m10 - region size in elements
// Register used by copy function:
//         m11 - scratch register
//         a0  - temp register
//         a1  - temp register
//         a2  - temp register
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
// Inputs (will/migth be reused insode copy function): 
//         m7  - return address (from call m7, imm)
//         m8  - source address
//         m9  - destination address
//         m10 - region size in elements
// Register used by copy function:
//         m11 - scratch register
//         a0  - temp register
//         a1  - temp register
//         a2  - temp register
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
