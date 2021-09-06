// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "dynamicSlice.inc"
#include "MultiSliceUpdateCommon.h.S"

/* -------------------------------------------------------------------------- */
// Dynamic Slice and Dynamic Update Slice vertex code for int, half and float
// variants
/* -------------------------------------------------------------------------- */
// Register aliases
#define mCopyPtr           m0
#define mOffsetSize        m1
#define mOffsetPtr         m2
#define mBaseTPtr          m3
#define mSubTPtr           m4
#define mBaseElem          m5
#define mRegionSizeInBytes m6
#define mBaseIdx           m8 // Alias to mSrcPtr. Re-use m8
#define mSubIdx            m9 // Alias to mDstPtr. Re-use m9
#define mWkrStart          m4
#define mMaxElementsPerWorker m3
#define mWkrEnd            m6
#define mWorkerId          m2


#define STACK_SIZE         8
//****************************************************************************
// The input structure parameters:
// 32 bit offset vector
// 32 ptr baseT
// 32 ptr subT
// 32 bit baseOffset
// 32 bit numBaseElements
// 16 bit numregions
//****************************************************************************
#define VOFF_OFFSET_PTR      0
#define VOFF_OFFSET_SIZE     1
#define VOFF_BASET_PTR       2
#define VOFF_SUBT_PTR        3
#define VOFF_BASE_OFFSET     4
#define VOFF_BASE_ELEM       5
#define VOFF_REGION_SIZE    (6*2) // 16 bit - offsets for use with 16 bit loads
#define VOFF_ELEMS_PER_WORKER 7

//******************************************************************************
// Labels names for each variant
//******************************************************************************
#define MS_BOOL_FUNC __runCodelet_popops__MultiSlice___bool
#define MU_BOOL_FUNC __runCodelet_popops__MultiUpdate___bool

#define MS_CHAR_FUNC __runCodelet_popops__MultiSlice___char
#define MU_CHAR_FUNC __runCodelet_popops__MultiUpdate___char

#define MS_UCHAR_FUNC __runCodelet_popops__MultiSlice___unsigned_char
#define MU_UCHAR_FUNC __runCodelet_popops__MultiUpdate___unsigned_char

#define MS_SCHAR_FUNC __runCodelet_popops__MultiSlice___signed_char
#define MU_SCHAR_FUNC __runCodelet_popops__MultiUpdate___signed_char

#define MS_HALF_FUNC __runCodelet_popops__MultiSlice___half
#define MU_HALF_FUNC __runCodelet_popops__MultiUpdate___half

#define MS_FLOAT_FUNC __runCodelet_popops__MultiSlice___float
#define MU_FLOAT_FUNC __runCodelet_popops__MultiUpdate___float

#define MS_INT_FUNC __runCodelet_popops__MultiSlice___int
#define MU_INT_FUNC __runCodelet_popops__MultiUpdate___int

#define MS_UNSIGNED_FUNC __runCodelet_popops__MultiSlice___unsigned_int
#define MU_UNSIGNED_FUNC __runCodelet_popops__MultiUpdate___unsigned_int

.globl MU_BOOL_FUNC
.type MU_BOOL_FUNC, @function

.globl MS_BOOL_FUNC
.type MS_BOOL_FUNC, @function

.globl MU_CHAR_FUNC
.type MU_CHAR_FUNC, @function

.globl MS_CHAR_FUNC
.type MS_CHAR_FUNC, @function

.globl MU_UCHAR_FUNC
.type MU_UCHAR_FUNC, @function

.globl MS_UCHAR_FUNC
.type MS_UCHAR_FUNC, @function

.globl MU_SCHAR_FUNC
.type MU_SCHAR_FUNC, @function

.globl MS_SCHAR_FUNC
.type MS_SCHAR_FUNC, @function

.globl MU_HALF_FUNC
.type MU_HALF_FUNC, @function

.globl MS_HALF_FUNC
.type MS_HALF_FUNC, @function

.globl MU_FLOAT_FUNC
.type MU_FLOAT_FUNC, @function

.globl MS_FLOAT_FUNC
.type MS_FLOAT_FUNC, @function

.globl MU_INT_FUNC
.type MU_INT_FUNC, @function

.globl MS_INT_FUNC
.type MS_INT_FUNC, @function

.globl MU_UNSIGNED_FUNC
.type MU_UNSIGNED_FUNC, @function

.globl MS_UNSIGNED_FUNC
.type MS_UNSIGNED_FUNC, @function

//******************************************************************************
// Entry points for each function, each setting up a pointer to the copy
// function and then using common code
//******************************************************************************
DEF_STACK_USAGE STACK_SIZE .text.MultiSlice_8_entry
.section .text.MultiSlice_8_entry
.align 4
MultiSlice_8_entry:
MU_BOOL_FUNC:
MU_CHAR_FUNC:
MU_UCHAR_FUNC:
MU_SCHAR_FUNC:
   setzi   $mCopyPtr, copy_loop_MU_8
   setzi   $mRegionSize, 0
   bri     MultiSlice_common_u

MS_BOOL_FUNC:
MS_CHAR_FUNC:
MS_UCHAR_FUNC:
MS_SCHAR_FUNC:
   setzi   $mCopyPtr, copy_loop_MS_8
   setzi   $mRegionSize, 0
   bri     MultiSlice_common_s

.size MultiSlice_8_entry, .-MultiSlice_8_entry

DEF_STACK_USAGE STACK_SIZE .text.MultiSlice_16_entry
.section .text.MultiSlice_16_entry
MultiSlice_16_entry:
.align 4
MU_HALF_FUNC:
   setzi   $mCopyPtr, copy_loop_MU_16
   setzi   $mRegionSize, 1
   bri     MultiSlice_common_u

MS_HALF_FUNC:
   setzi   $mCopyPtr, copy_loop_MS_16
   setzi   $mRegionSize, 1
   bri     MultiSlice_common_s

.size MultiSlice_16_entry, .-MultiSlice_16_entry

DEF_STACK_USAGE STACK_SIZE .text.MultiSlice_32_entry
.section .text.MultiSlice_32_entry
.align 4
MultiSlice_32_entry:
MU_FLOAT_FUNC:
MU_INT_FUNC:
MU_UNSIGNED_FUNC:
   setzi   $mCopyPtr, copy_loop_MU_32
   setzi   $mRegionSize, 2
   bri     MultiSlice_common_u

MS_FLOAT_FUNC:
MS_INT_FUNC:
MS_UNSIGNED_FUNC:
   setzi   $mCopyPtr, copy_loop_MS_32
   setzi   $mRegionSize, 2
   bri     MultiSlice_common_s

.size MultiSlice_32_entry, .-MultiSlice_32_entry


//******************************************************************************
// Common code for fetching pointers and managing outer loop
//******************************************************************************
.section .text.MultiSlice_common
.align 4
MultiSlice_common:
MultiSlice_common_u:
  UPDATE_OFFSET_AND_SIZES $mBaseElem $mMaxElementsPerWorker $mWorkerId $mWkrStart $mWkrEnd $mBaseIdx (4*VOFF_BASE_ELEM) (4*VOFF_ELEMS_PER_WORKER) (4*VOFF_BASE_OFFSET)
  bri MultiSlice_common_check_no_elem
MultiSlice_common_s:
    // Load states
    ld32  $mBaseElem, $mzero, $mvertex_base, VOFF_BASE_ELEM
    ld32  $mBaseIdx, $mzero, $mvertex_base, VOFF_BASE_OFFSET
    setzi $mWkrStart, 0
    
MultiSlice_common_check_no_elem:
    ld32  $mOffsetSize, $mzero, $mvertex_base, VOFF_OFFSET_SIZE
    brnzdec $mOffsetSize, MultiSlice_common_start
    exitz   $mzero

MultiSlice_common_start:
    st32  $mBaseIdx, $mzero, $mworker_base, 0

    // Load region size and get it in bytes
    ldz16 $mScratch, $mzero, $mvertex_base, VOFF_REGION_SIZE
    shl   $mRegionSizeInBytes, $mScratch, $mRegionSize

    mul   $mWkrStart, $mWkrStart, $mRegionSizeInBytes

    // Load vectors pointers
    ld32  $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
    ld32  $mBaseTPtr, $mzero, $mvertex_base, VOFF_BASET_PTR
    add   $mBaseTPtr, $mBaseTPtr, $mWkrStart
    ld32  $mSubTPtr, $mzero, $mvertex_base, VOFF_SUBT_PTR


MultiSlice_common_outer_loop_begin:

    // Load next offset
    ld32step $mScratch,    $mzero, $mOffsetPtr+=, 1

    // Reload states
    ld32   $mBaseIdx, $mzero, $mworker_base, 0
    ldz16  $mRegionSize, $mzero, $mvertex_base, VOFF_REGION_SIZE

    // Check if offset is a part of BaseT;
    sub    $mBaseIdx, $mScratch, $mBaseIdx
    cmpult $mScratch, $mBaseElem, $mBaseIdx
    brnz   $mScratch, MultiSlice_common_outer_loop_resume

    // Substruct baseOffset and prepare BaseT pointer
    mul    $mBaseIdx, $mBaseIdx, $mRegionSizeInBytes

    // Execute region copy
    br     $mCopyPtr

MultiSlice_common_outer_loop_resume:
    // Update SubT pointer to next region
    add     $mSubTPtr,    $mSubTPtr, $mRegionSizeInBytes
    brnzdec $mOffsetSize, MultiSlice_common_outer_loop_begin

MultiSlice_common_exit:
    exitz   $mzero

.size MultiSlice_common, .-MultiSlice_common


//******************************************************************************
// Pre copy step for MultiSlice and MultiUpdate (bool, signed/unsigned char)
//******************************************************************************
.section .text.MultiSlice_common_copy_8
.align 4
MultiSlice_common_copy_8:
// Load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_MU_8:
    add      $mDstPtr, $mBaseIdx, $mBaseTPtr
    mov      $mSrcPtr, $mSubTPtr
    call     $mReturnAddress, Slice_copy_function_8bit
    bri      MultiSlice_common_outer_loop_resume

// Index to create the next BASET input pointer
// Load the next SUBT pointer to use as a destination
copy_loop_MS_8:
    add      $mSrcPtr, $mBaseIdx, $mBaseTPtr
    mov      $mDstPtr, $mSubTPtr
    call     $mReturnAddress, Slice_copy_function_8bit
    bri      MultiSlice_common_outer_loop_resume

.size MultiSlice_common_copy_8, .-MultiSlice_common_copy_8

//******************************************************************************
// Pre copy step for MultiSlice and MultiUpdate (half)
//******************************************************************************
.section .text.MultiSlice_common_copy_16
.align 4
MultiSlice_common_copy_16:
// Load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_MU_16:
    add      $mDstPtr, $mBaseIdx, $mBaseTPtr
    mov      $mSrcPtr, $mSubTPtr
    call     $mReturnAddress, Slice_copy_function_half
    bri      MultiSlice_common_outer_loop_resume

// Index to create the next BASET input pointer
// Load the next SUBT pointer to use as a destination
copy_loop_MS_16:
    add      $mSrcPtr, $mBaseIdx, $mBaseTPtr
    mov      $mDstPtr, $mSubTPtr
    call     $mReturnAddress, Slice_copy_function_half
    bri      MultiSlice_common_outer_loop_resume

.size MultiSlice_common_copy_16, .-MultiSlice_common_copy_16


//******************************************************************************
// Pre copy step for MultiSlice and MultiUpdate (float or int)
//******************************************************************************
.section .text.MultiSlice_common_copy_32
.align 4
MultiSlice_common_copy_32:
// Load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_MU_32:
    add      $mDstPtr,  $mBaseIdx, $mBaseTPtr
    mov      $mSrcPtr, $mSubTPtr
    call     $mReturnAddress, Slice_copy_function_float_int
    bri      MultiSlice_common_outer_loop_resume

// Index to create the next BASET input pointer
// Load the next SUBT pointer to use as a destination
copy_loop_MS_32:
    add      $mSrcPtr, $mBaseIdx, $mBaseTPtr
    mov      $mDstPtr,  $mSubTPtr
    call     $mReturnAddress, Slice_copy_function_float_int
    bri      MultiSlice_common_outer_loop_resume

.size MultiSlice_common_copy_32, .-MultiSlice_common_copy_32

#endif
