#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

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
#define mBaseIdx           m8
#define mSubIdx            m9

// Copy function arguments and registers
#define mReturnAddress     m7
#define mSrcPtr            m8 // Alias to mBaseIdx. Re-use m8
#define mDstPtr            m9 // Alias to mSubIdx. Re-use m9
#define mRegionSize        m10
#define mScratch           m11

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

//******************************************************************************
// Labels names for each variant
//******************************************************************************
#define MS_HALF_FUNC __runCodelet_popops__MultiSlice___half
#define MU_HALF_FUNC __runCodelet_popops__MultiUpdate___half

#define MS_FLOAT_FUNC __runCodelet_popops__MultiSlice___float
#define MU_FLOAT_FUNC __runCodelet_popops__MultiUpdate___float

#define MS_INT_FUNC __runCodelet_popops__MultiSlice___int
#define MU_INT_FUNC __runCodelet_popops__MultiUpdate___int

#define MS_UNSIGNED_FUNC __runCodelet_popops__MultiSlice___unsigned_int
#define MU_UNSIGNED_FUNC __runCodelet_popops__MultiUpdate___unsigned_int

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
.section .text.MultiSlice_16_entry
MultiSlice_16_entry:
.align 4
MU_HALF_FUNC:
   setzi   $mCopyPtr, copy_loop_MU_16
   setzi   $mRegionSize, 1
   bri     MultiSlice_common

MS_HALF_FUNC:
   setzi   $mCopyPtr, copy_loop_MS_16
   setzi   $mRegionSize, 1
   bri     MultiSlice_common

.size MultiSlice_16_entry, .-MultiSlice_16_entry

.section .text.MultiSlice_32_entry
.align 4
MultiSlice_32_entry:
MU_FLOAT_FUNC:
MU_INT_FUNC:
MU_UNSIGNED_FUNC:
   setzi   $mCopyPtr, copy_loop_MU_32
   setzi   $mRegionSize, 2
   bri     MultiSlice_common

MS_FLOAT_FUNC:
MS_INT_FUNC:
MS_UNSIGNED_FUNC:
   setzi   $mCopyPtr, copy_loop_MS_32
   setzi   $mRegionSize, 2
   bri     MultiSlice_common

.size MultiSlice_32_entry, .-MultiSlice_32_entry


//******************************************************************************
// Common code for fetching pointers and managing outer loop
//******************************************************************************
.section .text.MultiSlice_common
.align 4
MultiSlice_common:
    ld32  $mScratch, $mzero, $mvertex_base, VOFF_OFFSET_SIZE
    brz   $mScratch, MultiSlice_common_exit
    add   $mOffsetSize, $mScratch, -1

    // Load region size and get it in bytes
    ldz16 $mScratch, $mzero, $mvertex_base, VOFF_REGION_SIZE
    shl   $mRegionSizeInBytes, $mScratch, $mRegionSize

    // Load vectors pointers 
    ld32  $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
    ld32  $mBaseTPtr, $mzero, $mvertex_base, VOFF_BASET_PTR
    ld32  $mSubTPtr, $mzero, $mvertex_base, VOFF_SUBT_PTR
    
    // Load states
    ld32  $mBaseElem, $mzero, $mvertex_base, VOFF_BASE_ELEM

MultiSlice_common_outer_loop_begin:

    // Load next offset 
    ld32step $mScratch,    $mzero, $mOffsetPtr+=, 1

    // Reload states
    ld32   $mBaseIdx, $mzero, $mvertex_base, VOFF_BASE_OFFSET
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
