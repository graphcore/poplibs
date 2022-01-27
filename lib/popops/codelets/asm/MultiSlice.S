// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "dynamicSlice.inc"
#include "MultiSliceUpdateCommon.h.S"
#include "BinarySearch.h.S"
#include "CommonPoplibsMacros.h.S"

/* -------------------------------------------------------------------------- */
// Dynamic Slice and Dynamic Update Slice vertex code for int, half and float
// variants
/* -------------------------------------------------------------------------- */
// Register aliases
#define mCopyPtr           m0
#define mOffsetSize        m1
#define mElemSize          m1
#define mOffsetPtr         m2
#define mBaseTPtr          m3
#define mSubTPtr           m4
#define mBaseElem          m5
#define mRegionSizeInBytes m6
#define mBaseIdx           m8 // Alias to mSrcPtr. Re-use m8
#define mSubIdx            m9 // Alias to mDstPtr. Re-use m9
#define mWkrStart          m4
#define mWkrStartScaled    m2
#define mMaxElementsPerWorker m3
#define mWkrEnd            m6
#define mWkrElems          m3
#define mWorkerId          m2
#define mIndicesAreSorted  m6
#define mMaxOffsetsPerWorker m6
#define STACK_SIZE         24

//****************************************************************************
// The input structure parameters:
// 32 bit offset vector
// 32 ptr baseT
// 32 ptr subT
// 32 bit baseOffset
// 32 bit numBaseElements
// 16 bit numregions
// 8  bit indicesAreSorted
//****************************************************************************
#define VOFF_OFFSET_PTR      0
#define VOFF_OFFSET_SIZE     1
#define VOFF_BASET_PTR       2
#define VOFF_SUBT_PTR        3
#define VOFF_BASE_OFFSET     4
#define VOFF_BASE_ELEM       5
#define VOFF_REGION_SIZE    (6*2) // 16 bit - offsets for use with 16 bit loads
#define VOFF_INDICES_ARE_SORTED (6*4 + 2)
#define VOFF_SPLIT_SINGLE_REGION (6*4 + 3)
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

// Split work between workers
.macro DIVIDE_WORK ELEM_SIZE, MAX_ELEMS, WORKER_ELEM_SIZE 
  // divide the offsets dimension
  get $mWorkerId, $WSR
  and $mWorkerId, $mWorkerId, CSR_W_WSR__CTXTID_M1__MASK

  // the max to be allocated per worker is already computed and passed as
  // a vertex field
  ld32 \MAX_ELEMS, $mzero, $mvertex_base, VOFF_ELEMS_PER_WORKER
  
  // Find the start entry for this worker
  mul $mWkrStart, $mWorkerId, \MAX_ELEMS
  min $mWkrStart, $mWkrStart, \ELEM_SIZE

  // FInd the end entry for this worker
  add $mWkrEnd, $mWkrStart, \MAX_ELEMS
  min $mWkrEnd, $mWkrEnd, \ELEM_SIZE

  // Number of offset entries this worker will process
  sub \WORKER_ELEM_SIZE, $mWkrEnd, $mWkrStart
.endm


//******************************************************************************
// Entry points for each function, each setting up a pointer to the copy
// function and then using common code
//******************************************************************************
FN_WORKER_ENTRY_POINT MU_BOOL_FUNC 4 "" STACK_SIZE
FN_EXPORT MU_CHAR_FUNC
FN_EXPORT MU_UCHAR_FUNC
FN_EXPORT MU_SCHAR_FUNC
   setzi   $mCopyPtr, copy_loop_MU_8
   setzi   $mRegionSize, 0
   bri     MultiSlice_common_u
FN_SIZE MU_BOOL_FUNC

FN_WORKER_ENTRY_POINT MS_BOOL_FUNC
FN_EXPORT MS_CHAR_FUNC
FN_EXPORT MS_UCHAR_FUNC
FN_EXPORT MS_SCHAR_FUNC
   setzi   $mCopyPtr, copy_loop_MS_8
   setzi   $mRegionSize, 0
   bri     MultiSlice_common_s
FN_SIZE MS_BOOL_FUNC

FN_WORKER_ENTRY_POINT MU_HALF_FUNC 4 "" STACK_SIZE
   setzi   $mCopyPtr, copy_loop_MU_16
   setzi   $mRegionSize, 1
   bri     MultiSlice_common_u
FN_SIZE MU_HALF_FUNC

FN_WORKER_ENTRY_POINT MS_HALF_FUNC
   setzi   $mCopyPtr, copy_loop_MS_16
   setzi   $mRegionSize, 1
   bri     MultiSlice_common_s
FN_SIZE MS_HALF_FUNC

FN_WORKER_ENTRY_POINT MU_FLOAT_FUNC 4 "" STACK_SIZE
FN_EXPORT MU_INT_FUNC
FN_EXPORT MU_UNSIGNED_FUNC
   setzi   $mCopyPtr, copy_loop_MU_32
   setzi   $mRegionSize, 2
   bri     MultiSlice_common_u
FN_SIZE MU_FLOAT_FUNC


FN_WORKER_ENTRY_POINT MS_FLOAT_FUNC 4 "" STACK_SIZE
FN_EXPORT MS_INT_FUNC
FN_EXPORT MS_UNSIGNED_FUNC
   setzi   $mCopyPtr, copy_loop_MS_32
   setzi   $mRegionSize, 2
   bri     MultiSlice_common_s
FN_SIZE MS_FLOAT_FUNC


//******************************************************************************
// Common code for fetching pointers and managing outer loop
//******************************************************************************
FN_SECTION MultiSlice_common
MultiSlice_common:
MultiSlice_common_u:
  ldz8 $mScratch, $mvertex_base, VOFF_SPLIT_SINGLE_REGION
  brnz $mScratch, MultiSlice_region_split

  UPDATE_OFFSET_AND_SIZES $mBaseElem $mMaxElementsPerWorker $mWorkerId $mWkrStart $mWkrEnd $mBaseIdx (4*VOFF_BASE_ELEM) (4*VOFF_ELEMS_PER_WORKER) (4*VOFF_BASE_OFFSET)
  
  // save registers used by binary search
  ldz8 $mIndicesAreSorted, $mvertex_base, VOFF_INDICES_ARE_SORTED
  brz $mIndicesAreSorted, MultiSlice_common_ld_offsets_span

  // Do a binary search over the sorted entries. The lowest entry greater than
  // equal to the dictionary entry allocated to his tile and the highest entry
  // less than equal to largest dictionary entry is searched for. We use this
  // information to find the number of offset positions to process per worker and
  // adjust the offset pointers accordingly

  // store used registers on stack. These are used by the binary search function
  st32 $m0, $mzero, $mworker_base, 1
  st32 $m10, $mzero, $mworker_base, 2
  st32 $m4, $mzero, $mworker_base, 3
  st32 $m8, $mzero, $mworker_base, 4

  ld32 $mBS_indicesPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
  ld32 $mBS_numIndices, $mzero, $mvertex_base, VOFF_OFFSET_SIZE
  mov  $mBS_targetValue, $mBaseIdx

  // result in $mBS_startIndex
  call $mBS_retLr, lowerBinarySearch

  // $mBS_indicesPtr, $mBS_numIndices and  $mBS_targetValue are unmodified
  add $mBS_targetValue, $mBS_targetValue, $mBaseElem
  // result in $mBS_endIndex
  call $mBS_retLr, upperBinarySearch
  // new offset size
  sub $mOffsetSize, $mBS_endIndex, $mBS_startIndex
  brneg $mOffsetSize, MultiSlice_common_exit

  // adjust offset pointer
  ld32  $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
  ld32step $mzero, $mzero, $mOffsetPtr+=, $mBS_startIndex
  st32 $mBS_startIndex, $mworker_base, 5

  // restore registers from stack
  ld32 $m0, $mzero, $mworker_base, 1
  ld32 $m10, $mzero, $mworker_base, 2
  ld32 $m4, $mzero, $mworker_base, 3
  ld32 $m8, $mzero, $mworker_base, 4

  bri MultiSlice_common_check_no_elem

  ld32 $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
  bri MultiSlice_common_check_no_elem

MultiSlice_common_ld_offsets_span:
    st32  $mzero, $mworker_base, 5
    ld32  $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
    ld32  $mOffsetSize, $mzero, $mvertex_base, VOFF_OFFSET_SIZE

MultiSlice_common_check_no_elem:
    brnzdec $mOffsetSize, MultiSlice_common_start
    exitz   $mzero

MultiSlice_common_start:
    st32  $mBaseIdx, $mzero, $mworker_base, 0

    // Load region size and get it in bytes
    ldz16 $mScratch, $mzero, $mvertex_base, VOFF_REGION_SIZE
    shl   $mRegionSizeInBytes, $mScratch, $mRegionSize

    mul   $mWkrStart, $mWkrStart, $mRegionSizeInBytes

    // Load vectors pointers
    ld32  $mBaseTPtr, $mzero, $mvertex_base, VOFF_BASET_PTR
    add   $mBaseTPtr, $mBaseTPtr, $mWkrStart
    ld32  $mSubTPtr, $mzero, $mvertex_base, VOFF_SUBT_PTR
    ld32  $mScratch, $mworker_base, 5
    mul   $mScratch, $mScratch, $mRegionSizeInBytes
    add   $mSubTPtr,  $mSubTPtr, $mScratch

MultiSlice_common_outer_loop_begin:

    ldz16  $mRegionSize, $mzero, $mvertex_base, VOFF_REGION_SIZE

    // Reload states
    ld32   $mBaseIdx, $mzero, $mworker_base, 0

MultiSlice_common_region_copy_entry:
    // Load next offset
    ld32step $mScratch,    $mzero, $mOffsetPtr+=, 1


    // Check if offset is a part of BaseT;
    sub    $mBaseIdx, $mScratch, $mBaseIdx
    cmpult $mScratch, $mBaseIdx, $mBaseElem
    brz   $mScratch, MultiSlice_common_outer_loop_resume

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

MultiSlice_common_s:
  ldz8 $mScratch, $mvertex_base, VOFF_SPLIT_SINGLE_REGION
  brnz $mScratch, MultiSlice_region_split

  // divide the offsets dimension
  ld32 $mOffsetSize, $mzero, $mvertex_base, VOFF_OFFSET_SIZE
  DIVIDE_WORK $mOffsetSize $mMaxOffsetsPerWorker $mOffsetSize

  // Point to the entry in the offset vector this worker will start processing
  // from
  ld32  $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
  ld32step $mzero, $mzero, $mOffsetPtr+=, $mWkrStart
  
  // this is stored in stack where we jump
  ld32  $mBaseIdx, $mzero, $mvertex_base, VOFF_BASE_OFFSET

  // This is require to offset the subT pointer
  st32 $mWkrStart, $mworker_base, 5

  // This is required to offset the baseT pointer and we are not splitting the
  // slice dimension. Therefore must be 0
  setzi $mWkrStart, 0

  ld32  $mBaseElem, $mzero, $mvertex_base, VOFF_BASE_ELEM
  bri MultiSlice_common_check_no_elem

// Split region when there is only one offset
MultiSlice_region_split:

  // divide the region dimension
  ldz16 $mElemSize, $mzero, $mvertex_base, VOFF_REGION_SIZE
  DIVIDE_WORK $mElemSize $mMaxElementsPerWorker $mWkrElems

  // worker start offset in bytes
  shl $mWkrStartScaled, $mWkrStart, $mRegionSize
  shl $mRegionSizeInBytes, $mElemSize, $mRegionSize
  brz $mWkrElems, MultiSlice_common_exit

  // copy only part of the region in elements
  mov $mRegionSize, $mWkrElems

  ld32 $mBaseIdx, $mzero, $mvertex_base, VOFF_BASE_OFFSET
  ld32 $mBaseElem, $mzero, $mvertex_base, VOFF_BASE_ELEM

  // Load vectors pointers and offset to where workers process
  ld32 $mBaseTPtr, $mzero, $mvertex_base, VOFF_BASET_PTR
  add  $mBaseTPtr, $mBaseTPtr, $mWkrStartScaled
  ld32 $mSubTPtr, $mzero, $mvertex_base, VOFF_SUBT_PTR
  add  $mSubTPtr, $mSubTPtr, $mWkrStartScaled

  // Point to the entry in the offset vector this worker will start processing
  // from
  ld32  $mOffsetPtr, $mzero, $mvertex_base, VOFF_OFFSET_PTR
  
  // only one pass through the loop
  setzi $mOffsetSize, 0

  bri MultiSlice_common_region_copy_entry


FN_SIZE MultiSlice_common


//******************************************************************************
// Pre copy step for MultiSlice and MultiUpdate (bool, signed/unsigned char)
//******************************************************************************
FN_SECTION MultiSlice_common_copy_8
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

FN_SIZE MultiSlice_common_copy_8

//******************************************************************************
// Pre copy step for MultiSlice and MultiUpdate (half)
//******************************************************************************
FN_SECTION MultiSlice_common_copy_16
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

FN_SIZE MultiSlice_common_copy_16


//******************************************************************************
// Pre copy step for MultiSlice and MultiUpdate (float or int)
//******************************************************************************
FN_SECTION MultiSlice_common_copy_32
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

FN_SIZE MultiSlice_common_copy_32

#endif
