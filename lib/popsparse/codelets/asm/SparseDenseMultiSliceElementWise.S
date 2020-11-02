// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of  SparseDenseMultiSlice vertex template variations.

// Restrictions
//
//  * The subT that slices are written to must be must be 32-bit aligned and
//    have columns which are 32 bits in length so that each row is also 32-bit
//    aligned. (This constrains half vertices to have an even number of columns)

#include "SparseDenseMatMulStructs.h.S"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

#define INCLUDE_IN_ASSEMBLER
#include "../../SparseCodeletMetaInfoScale.hpp"

// Symbols
#define CODELET_SLICE_HALF __runCodelet_popsparse__SparseDenseMultiSliceElementWise___half
#define CODELET_SLICE_FLOAT __runCodelet_popsparse__SparseDenseMultiSliceElementWise___float


// Vertex state (Offsets in bytes)
#define VOFF_OFFSET_PTR           0
#define VOFF_BASET_NZ_PTR         4
#define VOFF_BASET_META_PTR       8 // Short span (20 lsbs address)
#define VOFF_SUBT_PTR             12
#define VOFF_NZ_SCALE_FACTOR      16
#define VOFF_ROWS_PER_PARTITION   18
#define VOFF_Y_PARTITION          20
#define VOFF_SUBT_COLUMNS         22
#define VOFF_OFFSET_SIZE          24

// Vertex state created by the supervisor on the stack and passed to workers
#define WOFF_NZ_PTR                0
#define WOFF_META_PTR              4
#define WOFF_OFFSET_PTR            8
#define WOFF_ROW_OFFSET            12
#define WOFF_NZ_SCALE_FACTOR       16
#define WOFF_SUBT_COLUMNS          20
#define WOFF_SUBT_PTR              24
#define WOFF_OFFSET_SIZE           28
#define WOFF_NUM_ROWS_M1           32
#define WOFF_SUBT_COLUMNSxWORKERS  36
#define WOFF_SUPER_LR_STORE        40
#define WOFF_SUPER_FP_STORE        44
#define STACK_SIZE                (WOFF_SUPER_FP_STORE+4)

// Worker stack
#define WSTACK_META_ROW           0
#define WSTACK_OFFSET_PTR         4
#define WSTACK_SUBT_PTR           8
#define WSTACK_OFFSET_LOOPS       12

// worker register aliases
#define w_subTStride              m0
#define w_nzScaleFactor           m0

#define w_nzPtr                   m1
#define w_metaPtr                 m2
#define w_offsetLoops             m3
#define w_subTPtr                 m4

#define w_metaRow                 m5
#define w_offset32Bit             m5

#define w_metaColumns             m6

#define w_numRowsM1               m7

#define w_offsetPtr               m8
#define w_temp                    m9
#define w_nzPtrTemp               m10
#define w_metaPtrTemp             m11
#define w_nzAndMetaPtrTemp        m10:11

// -----------------------------------------------------------------------------
// Preamble macro to pre-process worker state specific to each worker

// Registers used only in the preamble macro
#define w_id                      m0

.macro WORKER_PREAMBLE TYPE
  get           $w_id, $WSR
  and           $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
  // Load and offset the offsets and subT pointers for this worker's work
  ld32          $w_temp, $mvertex_base, WOFF_SUBT_COLUMNS/4
  mul           $w_temp, $w_temp, $w_id

  ld32          $w_offsetPtr, $mvertex_base, WOFF_OFFSET_PTR/4
  ld32          $w_subTPtr, $mvertex_base, WOFF_SUBT_PTR/4
  ld32step      $mzero, $mzero, $w_offsetPtr+=,$w_id

.ifc \TYPE, half
  ldz16step     $mzero, $mzero, $w_subTPtr+=,$w_temp
.else
  ld32step      $mzero, $mzero, $w_subTPtr+=,$w_temp
.endif

  st32          $w_offsetPtr, $mworker_base, WSTACK_OFFSET_PTR/4
  st32          $w_subTPtr, $mworker_base, WSTACK_SUBT_PTR/4

  // Find the number of loops for this worker:
  // For n with 0 <= n <= 65533 this does a division by 6 with the remainder
  // split amongst workers.
  ld32          $w_offsetLoops, $mvertex_base, WOFF_OFFSET_SIZE/4
  add           $w_offsetLoops, $w_offsetLoops, NUM_WORKERS
  sub           $w_offsetLoops, $w_offsetLoops, $w_id
  mul           $w_offsetLoops, $w_offsetLoops, 21845
  shr           $w_offsetLoops, $w_offsetLoops, 17

.endm
#undef w_id


//------------------------------------------------------------------------------
// Worker Half function
.section .text.worker_half
.align 8
.type worker_half, @function
.worker
  // Entry point to use after the 1st call the supervisor vertex makes to the
  // workers.  Worker specific work division has been done and the result stored
  // on each worker's stack, so no need to calculate it again.
worker_stack_retained_half:
  ld32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4
  brpos         $w_offsetLoops, worker_continue_half
  // Exit if nothing to do
  exitz         $mzero

  // First time entry point - so use of state (stack) retention
worker_half:
  WORKER_PREAMBLE half
  brnzdec       $w_offsetLoops, 1f
  // Exit if nothing to do for this worker. Store the loop count for when
  // the stack retained entry is used.  Subtract 1 so it is consistent with
  // what would have happened if the brnzdec branch was taken
  sub           $w_offsetLoops, $w_offsetLoops, 1
  st32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4
  exitz         $mzero
1:
  // Using loops - 1 as we use brnzdec in the loop later
  st32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4

worker_continue_half:
  ld32          $w_nzPtr, $mvertex_base, WOFF_NZ_PTR/4
  ld32          $w_metaPtr, $mvertex_base, WOFF_META_PTR/4
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNSxWORKERS/4

  ld32          $w_numRowsM1, $mvertex_base, WOFF_NUM_ROWS_M1/4
meta_info_row_loopH:
  // Load the row and add the offset
  ldz16step     $w_metaRow, $mzero, $w_metaPtr+=,1
  ld32          $w_temp, $mvertex_base, WOFF_ROW_OFFSET/4
  add           $w_metaRow, $w_metaRow, $w_temp

  // Load the loop count, offsets and subT pointers for this worker's work
  ld32          $w_offsetPtr, $mworker_base, WSTACK_OFFSET_PTR/4
  ld32          $w_subTPtr, $mworker_base, WSTACK_SUBT_PTR/4

  ld32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4
offsets_loopH:
  ld32step      $w_temp, $mzero, $w_offsetPtr+=,NUM_WORKERS
  cmpeq         $w_temp, $w_temp, $w_metaRow
  brz           $w_temp, next_offsetH
  // Found a row, read data and populate. Use temp ptrs as we may have to do this
  // for many slices with the same offset
  tapack        $w_nzAndMetaPtrTemp, $w_nzPtr, $w_metaPtr, $mzero
  // Store w_metaRow.  Even though this looks like we in the 2nd innermost loop
  // it's conditional that we get here and so it is likely that storing/restoring
  // here is faster
  st32          $w_metaRow, $mworker_base, WSTACK_META_ROW/4
  ld32          $w_nzScaleFactor, $mvertex_base, WOFF_NZ_SCALE_FACTOR/4

  // The number of columns in this row's metaData.
  // Subtract 1 as we are stuck with a brnzdec due to subWord write needing a
  // branch
  ldz16step     $w_metaColumns, $mzero, $w_metaPtrTemp+=,1
  sub           $w_metaColumns, $w_metaColumns, 1
meta_info_columns_loopH:
  // Load the column and apply scale to get the column number.  This is like a
  // divide which we can do by multiply, then shift right.  But we avoid the shift
  ldz16step     $w_temp, $mzero, $w_metaPtrTemp+=,1
  mul           $w_temp, $w_nzScaleFactor, $w_temp
  // At this moment we have column<<(RECIPROCAL_MUL_SHIFT+1).
  // We don't actually need the column
  // number so don't generate it, which saves an instruction
#define COLUMN_BIT_POSITION  (RECIPROCAL_MUL_SHIFT+YOffsetTypeDivShiftHalf)

  // Fetch the data to preserve in the subword write
  // *** As data is 32 bit aligned, and we have an even number of columns in SubT
  // w_subTPtr will always be 32 bit aligned.  This simplifes the following code

  // Due to *** The offset in 32 bit increments to the 32bit aligned location to
  // slice into is just given by column/2
  shr           $w_offset32Bit, $w_temp, (COLUMN_BIT_POSITION + 1)
  ld32          $a1, $mzero, $w_subTPtr, $w_offset32Bit
  // Fetch the data to write
  ldb16step     $a0, $mzero, $w_nzPtrTemp+=,1

  // Due to *** the choice of writing the new data into the upper or lower
  // half of the 32 bit word is just given by the column being odd or even, we
  // get the LSBit of the column into the sign bit by shifting

  shl           $w_temp, $w_temp, (32-COLUMN_BIT_POSITION-YOffsetTypeDivShiftHalf)
  {brpos        $w_temp, 1f
   sort4x16hi   $a2, $a0, $a1}
  sort4x16lo    $a2, $a1, $a0
1:
  st32          $a2, $mzero, $w_subTPtr, $w_offset32Bit
  brnzdec       $w_metaColumns, meta_info_columns_loopH

  // register restore
  ld32          $w_metaRow, $mworker_base, WSTACK_META_ROW/4
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNSxWORKERS/4
next_offsetH:
  // Advance to this worker's next SubT slice to populate
  ldz16step     $mzero, $mzero, $w_subTPtr+=, $w_subTStride
  brnzdec       $w_offsetLoops, offsets_loopH

  // Advance main pointers to to metadata and NZ data for the next row
  ldz16step     $w_metaColumns, $mzero, $w_metaPtr+=,1
  ldz16step     $mzero, $mzero, $w_metaPtr+=, $w_metaColumns
  ldb16step     $azero, $mzero, $w_nzPtr+=, $w_metaColumns

  brnzdec       $w_numRowsM1, meta_info_row_loopH

  exitz         $mzero

.size worker_half, . - worker_stack_retained_half

//------------------------------------------------------------------------------
// Worker float function
.section .text.worker_float
.align 8
.type worker_float, @function
.worker
  // Entry point to use after the 1st call the supervisor vertex makes to the
  // workers.  Worker specific work division has been done and the result stored
  // on each worker's stack, so no need to calculate it again.
worker_stack_retained_float:
  ld32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4
  brpos         $w_offsetLoops, worker_continue_float
  // Exit if nothing to do
  exitz         $mzero

 // First time entry point - so use of state (stack) retention
worker_float:
  WORKER_PREAMBLE float
  // Entry point to use after the 1st call the supervisor vertex makes to the
  // workers.  Worker specific work division has been done and the result stored
  // on each worker's stack, so no need to calculate it again.  We do need to
  // check if the worker has anything to do though.
  brnzdec       $w_offsetLoops, 1f
  // Exit if nothing to do for this worker. Store the loop count for when
  // the stack retained entry is used.  Subtract 1 so it is consistent with
  // what would have happened if the brnzdec branch was taken
  sub           $w_offsetLoops, $w_offsetLoops, 1
  st32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4
  exitz         $mzero
1:
  // Using loops - 1 as we use brnzdec in the loop later
  st32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4

worker_continue_float:

  ld32          $w_nzPtr, $mvertex_base, WOFF_NZ_PTR/4
  ld32          $w_metaPtr, $mvertex_base, WOFF_META_PTR/4
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNSxWORKERS/4

  ld32          $w_numRowsM1, $mvertex_base, WOFF_NUM_ROWS_M1/4
meta_info_row_loopF:
  // Load the row and add the offset
  ldz16step     $w_metaRow, $mzero, $w_metaPtr+=,1
  ld32          $w_temp, $mvertex_base, WOFF_ROW_OFFSET/4
  add           $w_metaRow, $w_metaRow, $w_temp

  // Load the loop count, offsets and subT pointers for this worker's work
  ld32          $w_offsetPtr, $mworker_base, WSTACK_OFFSET_PTR/4
  ld32          $w_subTPtr, $mworker_base, WSTACK_SUBT_PTR/4
  ld32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4

  // The number of columns that have NZ data in this metaData row
  ldz16step     $w_metaColumns, $mzero, $w_metaPtr+=,1

offsets_loopF:
  // Load the next offset and compare to the row found in the metaData
  ld32step      $w_temp, $mzero, $w_offsetPtr+=, NUM_WORKERS
  cmpeq         $w_temp, $w_temp, $w_metaRow
  brz           $w_temp, next_offsetF
  // Found a match, read data and populate. Use temp ptrs as we may have to do this
  // for many slices with the same offset.
  tapack        $w_nzAndMetaPtrTemp, $w_nzPtr, $w_metaPtr, $mzero
  ld32          $w_nzScaleFactor, $mvertex_base, WOFF_NZ_SCALE_FACTOR/4

  rpt           $w_metaColumns, (2f-1f)/8 -1
meta_info_columns_loopF:
1:
  // Load the column and apply scale to get the column number
  {ldz16step     $w_temp, $mzero, $w_metaPtrTemp+=,1
   fnop}
  {mul           $w_temp, $w_temp, $w_nzScaleFactor
   fnop}
  {shr           $w_temp, $w_temp, (RECIPROCAL_MUL_SHIFT+YOffsetTypeDivShiftFloat)
   fnop}
  // Fetch the NZ-data and write it to the subT slice result
  {ld32step     $a0, $mzero, $w_nzPtrTemp+=, 1
   fnop}
  {st32         $a0, $mzero, $w_subTPtr, $w_temp
   fnop}
2:
  // register restore
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNSxWORKERS/4

next_offsetF:
  // Advance to this worker's next SubT slice to populate
  ld32step     $mzero, $mzero, $w_subTPtr+=, $w_subTStride
  brnzdec      $w_offsetLoops, offsets_loopF

  // Advance main pointers to to metadata and NZ data for the next row
  ldz16step     $mzero, $mzero, $w_metaPtr+=, $w_metaColumns
  ld32step      $azero, $mzero, $w_nzPtr+=, $w_metaColumns

  brnzdec       $w_numRowsM1, meta_info_row_loopF

  exitz          $mzero
.size worker_float, . - worker_stack_retained_float
//------------------------------------------------------------------------------
// Supervisor functions to divide work and parse meta information, calling
// worker threads once information matching the subGroup ID is found
//------------------------------------------------------------------------------
// Supervisor register aliases
#define s_base                      m0
#define s_workerEntry               m1
#define s_requiredYPartition        m2
#define s_metaPtrPtr                m3
#define s_bucketCount               m4
#define s_nzPtrPtr                  m5
#define s_metaPtr                   m6

#define s_subgroupSparseElems       m7
#define s_subgroupID                m7
#define s_yPartition                m7
#define s_xPartition                m7

#define s_offsetToNextSubGroup      m8
#define s_temp                      m8

#define s_nzPtr                     m9
#define s_rowsPerPartition          m10

// Defines just used for copying vertex state
#define s_offsetSize                m2
#define s_subTColumns               m3
#define s_offsetPtr                 m4
#define s_nzScaleFactor             m5
#define s_subTPtr                   m6

#define SHORT_SPAN_ADDRESS_BITS    20

//------------------------------------------------------------------------------
// Macro to create entry points, supervisor code and call the workers
//------------------------------------------------------------------------------
.macro INSTANTIATE_SUPERVISOR NAME TYPE

DEF_STACK_USAGE  (STACK_SIZE)  \NAME

.section .text.\NAME
.align 4
.globl \NAME
.type \NAME, @function
\NAME:
.supervisor
  // Compute work division, compute useful constant values and copy vertex state
  // to the workers.  All this needs to be done just once so dealing with it up
  // front saves time.
  add       $sp, $sp, -STACK_SIZE
  ldz16     $s_subTColumns, $s_base, VOFF_SUBT_COLUMNS/2
  ld32      $s_offsetPtr, $s_base, VOFF_OFFSET_PTR/4
  ldz16     $s_nzScaleFactor, $s_base, VOFF_NZ_SCALE_FACTOR/2
  ld32      $s_subTPtr, $s_base, VOFF_SUBT_PTR/4
  ldz16     $s_offsetSize, $s_base, VOFF_OFFSET_SIZE/2

  st32      $lr, $sp, WOFF_SUPER_LR_STORE/4

  st32      $s_subTColumns, $sp, WOFF_SUBT_COLUMNS/4
  mul       $s_subTColumns, $s_subTColumns, NUM_WORKERS

  st32      $fp, $sp, WOFF_SUPER_FP_STORE/4

  st32      $s_offsetPtr, $sp, WOFF_OFFSET_PTR/4
  st32      $s_nzScaleFactor, $sp, WOFF_NZ_SCALE_FACTOR/4
  st32      $s_subTPtr, $sp, WOFF_SUBT_PTR/4
  st32      $s_offsetSize, $sp, WOFF_OFFSET_SIZE/4

  st32      $s_subTColumns, $sp, WOFF_SUBT_COLUMNSxWORKERS/4

  // The supervisor loops over buckets and subGroups found in the buckets.
  // When a subGroup is found with the required Y partition, we pass pointers to
  // the NZ and metadata for that subGroup to the worker along with row count,
  // and row offset
  ldz16     $s_requiredYPartition, $s_base, VOFF_Y_PARTITION/2
  ld32      $s_metaPtrPtr, $s_base, VOFF_BASET_META_PTR/4
  shr       $s_bucketCount, $s_metaPtrPtr, SHORT_SPAN_ADDRESS_BITS
  shl       $s_metaPtrPtr, $s_metaPtrPtr, (32-SHORT_SPAN_ADDRESS_BITS)
  shr       $s_metaPtrPtr, $s_metaPtrPtr, (32-SHORT_SPAN_ADDRESS_BITS)
  sub       $s_bucketCount, $s_bucketCount, 1

  ld32      $s_nzPtrPtr, $s_base, VOFF_BASET_NZ_PTR/4
  ldz16     $s_rowsPerPartition, $s_base, VOFF_ROWS_PER_PARTITION/2
  setzi     $s_workerEntry, worker_\TYPE
bucket_loop\@:
  ld32step   $s_metaPtr, $mzero, $s_metaPtrPtr+=, 1
  ld32step   $s_nzPtr, $mzero, $s_nzPtrPtr+=, 1
  bri        subgroup_loop_first_entry\@

subgroup_loop\@:
  // Advance through the bucket to the next sub group.  Use the size of the
  // metaInfo and NZ data to step
  ldz16      $s_subgroupSparseElems, $s_metaPtr, MetaInfoSubGroupEntry_sparseElementCount/2
  ldz16      $s_offsetToNextSubGroup, $s_metaPtr, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  ldz16step  $mzero, $mzero, $s_metaPtr+=, $s_offsetToNextSubGroup
.ifc \TYPE, half
  ldz16step  $mzero, $mzero, $s_nzPtr+=, $s_subgroupSparseElems
.else
  ld32step   $mzero, $mzero, $s_nzPtr+=, $s_subgroupSparseElems
.endif
subgroup_loop_first_entry\@:
  // Fetch pointer to the SubGroupEntry struct for the next subgroup
  ldz16      $s_subgroupID, $s_metaPtr, MetaInfoSubGroupEntry_id/2
  // ID = 0 means that's the last one in the bucket
  brz        $s_subgroupID, subgroup_loop_end\@
  // Otherwise, does the yPartition match the one we're looking for?
  ldz16      $s_yPartition, $s_metaPtr, MetaInfoSubGroupEntry_yPartition/2
  cmpeq      $s_yPartition, $s_yPartition, $s_requiredYPartition
  brz        $s_yPartition, subgroup_loop\@

  // If so, pass information to the workers and call them

  // Find the offset of the row information stored in the meta data and pass it
  // to the workers
  ldz16      $s_xPartition, $s_metaPtr, MetaInfoSubGroupEntry_xPartition/2
  mul        $s_xPartition, $s_xPartition, $s_rowsPerPartition

  // Pass pointers to meta info and the nz to the workers.
  ldz16      $s_temp, $s_metaPtr, MetaInfoSubGroupEntry_offsetToFirstOutputEntry/2
  shl        $s_temp, $s_temp, 1
  add        $s_temp, $s_temp, $s_metaPtr

  // The above supervisor code will have many pipline stalls
  // as we branch, and load/compute/store with register dependencies.
  // On all but the first pass we do some of that while workers are running
  // by syncing here instead of directly after runall: assuming 1 worker ends
  // before the others we can get branches and computation for free.  We have to
  // sync before we affect stacked vertex state for the workers to use though.
  sync       TEXCH_SYNCZONE_LOCAL
  st32       $s_temp, $sp, WOFF_META_PTR/4
  st32       $s_nzPtr, $sp, WOFF_NZ_PTR/4
  st32       $s_xPartition, $sp, WOFF_ROW_OFFSET/4
  // Pass the number of rows to the workers
  ldz16      $s_temp, $s_metaPtr, MetaInfoSubGroupEntry_numXm1/2
  st32       $s_temp, $sp, WOFF_NUM_ROWS_M1/4

  runall     $s_workerEntry, $sp, 0
  // For any worker execution other than the 1st it has already computed worker
  // specific values - so choose a fresh entry point that allows for use of those
  // values without calculating them again
  setzi      $s_workerEntry, worker_stack_retained_\TYPE
  bri        subgroup_loop\@
subgroup_loop_end\@:
  brnzdec   $s_bucketCount, bucket_loop\@

  // Restore and exit....
  ld32      $lr, $sp, WOFF_SUPER_LR_STORE/4
  ld32      $fp, $sp, WOFF_SUPER_FP_STORE/4
  // Last sync to ensure all workers are done before exit
  sync      TEXCH_SYNCZONE_LOCAL
  add       $sp, $sp, STACK_SIZE
  br        $lr

.size \NAME, . - \NAME
.endm
//------------------------------------------------------------------------------
INSTANTIATE_SUPERVISOR CODELET_SLICE_HALF half
INSTANTIATE_SUPERVISOR CODELET_SLICE_FLOAT float


#endif // __IPU__
