// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of  SparseDenseMultiUpdateAdd vertex template variations.

#include "SparseDenseMatMulStructs.h.S"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

#define INCLUDE_IN_ASSEMBLER
#include "../../SparseCodeletMetaInfoScale.hpp"

// Symbols
#define CODELET_SLICE_HALF __runCodelet_popsparse__SparseDenseMultiUpdateAddElementWise___half
#define CODELET_SLICE_FLOAT __runCodelet_popsparse__SparseDenseMultiUpdateAddElementWise___float


// Vertex state (Offsets in bytes)
#define VOFF_OFFSET_PTR           0
#define VOFF_BASET_NZ_PTR         4
#define VOFF_BASET_META_PTR       8 // Short span (20 lsbs address)
#define VOFF_SUBT_PTR             12
#define VOFF_NZ_SCALE_FACTOR      16
#define VOFF_ROWS_PER_PARTITION   18
#define VOFF_Y_PARTITION_PTR      20
#define VOFF_SUBT_COLUMNS         24
#define VOFF_OFFSET_SIZE          26
#define VOFF_SCALE_PTR            30

// Vertex state created by the supervisor on the stack and passed to workers
#define WOFF_NZ_PTR                0
#define WOFF_META_PTR              4
#define WOFF_OFFSET_PTR            8
#define WOFF_ROW_OFFSET            12
#define WOFF_NZ_SCALE_FACTOR       16
#define WOFF_SUBT_COLUMNS          20
#define WOFF_SUBT_PTR              24
#define WOFF_NUM_OFFSETS_M1        28
#define WOFF_SCALE                 32
#define WOFF_SUPER_LR_STORE        36
#define WOFF_SUPER_FP_STORE        40
#define STACK_SIZE                (WOFF_SUPER_FP_STORE+4)

// worker register aliases
#define w_nzScaleFactor           m0
#define w_id                      m0

#define w_numWorkers              m1
#define w_nzPtr                   m1

#define w_metaPtr                 m2
#define w_offsetLoopsM1           m3

#define w_subTPtr                 m4
#define w_offsetsYInSFirst        m4

#define w_metaRow                 m5
#define w_metaColumns             m6
#define w_totalNumY               m7
#define w_offsetPtr               m8
#define w_temp                    m9

#define w_nzPtrTemp               m10
#define w_rowOffset               m10

#define w_metaPtrTemp             m11
#define w_subTStride              m11

#define w_scale                   a7
///------------------------------------------------------------------------------
// Worker float, half function macro
.macro INSTANTIATE_WORKER NAME TYPE
.section .text.\NAME
.align 8
.type \NAME, @function
.worker
.ifc \TYPE, half
  nop  // rpt align
.endif
\NAME:
  // Load the pointer to the number of GradWWorkerEntry structs
  ld32          $w_metaPtr, $mvertex_base, WOFF_META_PTR/4
  // Load the count of the number of GradWWorkerEntry structs, point to the
  // first struct
  ldz16step     $w_numWorkers, $mzero, $w_metaPtr+=,1
  // Exit if there is nothing for this worker to do
  get           $w_id, $WSR
  and           $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
  add           $w_temp, $w_id, 1

  cmpult        $w_temp, $w_numWorkers, $w_temp
  brz           $w_temp, 1f
  // Exit if there is nothing to do
  exitz         $mzero
1:
  // Point to this worker's GradWWorkerEntry struct
  mul           $w_id, $w_id, Sizeof_MIGradWWorkerEntry
  add           $w_metaPtr, $w_metaPtr, $w_id
  // Get the count of number of columns in this worker's work and
  // an offset into the 1st row that the worker processes
  ldz16         $w_totalNumY, $w_metaPtr, MIGradWorkerEntry_totalNumY/2
  ldz16         $w_offsetsYInSFirst, $w_metaPtr, MIGradWorkerEntry_metaInfoOffsetToOffsetsYInSFirst/2

  // Point to the worker's meta data (row start) and nzData
  ld32          $w_nzPtr, $mvertex_base, WOFF_NZ_PTR/4
  ldz16         $w_temp, $w_metaPtr, MIGradWorkerEntry_sparseOffset/2
  ld32step      $azero, $mzero, $w_nzPtr+=,$w_temp
  ldz16         $w_temp, $w_metaPtr, MIGradWorkerEntry_metaInfoOffsetOutputEntry/2
  ldz16step     $mzero, $mzero, $w_metaPtr+=,$w_temp

  ld32          $w_nzScaleFactor, $mvertex_base, WOFF_NZ_SCALE_FACTOR/4
  ld32          $w_scale, $mvertex_base, WOFF_SCALE/4

  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNS/4

  // Load info about the 1st row the worker is to process.  This is a bit
  // different to other ones as this worker may not be assigned all the columns
  // within the first row

  // Load the first row and apply the row offset
  ldz16step     $w_metaRow, $mzero, $w_metaPtr+=,1
  ld32          $w_rowOffset, $mvertex_base, WOFF_ROW_OFFSET/4
  add           $w_metaRow, $w_metaRow, $w_rowOffset

  // Load the number of columns in this row, advance to this worker's work
  // allocation within the row.
  ldz16step     $w_metaColumns, $mzero, $w_metaPtr+=,1
  ldz16step     $mzero, $mzero, $w_metaPtr+=,$w_offsetsYInSFirst
  sub           $w_metaColumns, $w_metaColumns, $w_offsetsYInSFirst
  // Ensure that the total columns for this worker to process isn't exceeded, if
  // it is to process only part of this row
  min           $w_metaColumns, $w_metaColumns, $w_totalNumY
  bri           meta_info_row_loop_first_time\@

meta_info_row_loop\@:
  // Advance main pointers to to metadata and NZ data for the next row
  ldz16step     $mzero, $mzero, $w_metaPtr+=, $w_metaColumns
  ld32step      $azero, $mzero, $w_nzPtr+=, $w_metaColumns

  // Get the next row and apply the row offset
  ldz16step     $w_metaRow, $mzero, $w_metaPtr+=,1
  add           $w_metaRow, $w_metaRow, $w_rowOffset

  // Get the number of columns in this row, but this worker may only need to
  // process some of them
  ldz16step     $w_metaColumns, $mzero, $w_metaPtr+=,1
  min           $w_metaColumns, $w_metaColumns, $w_totalNumY

meta_info_row_loop_first_time\@:
  // Prepare to process the offsets
  ld32          $w_offsetPtr, $mvertex_base, WOFF_OFFSET_PTR/4
  ld32          $w_offsetLoopsM1, $mvertex_base, WOFF_NUM_OFFSETS_M1/4
  ld32          $w_subTPtr, $mvertex_base, WOFF_SUBT_PTR/4

offsets_loop\@:
  // Load the next row in the offsets, and compare it to that in the metainfo
  ld32step      $w_temp, $mzero, $w_offsetPtr+=,1
  cmpeq         $w_temp, $w_temp, $w_metaRow
  brz           $w_temp, next_offset\@
  // A row matches, so update it into the (always float) result.
  // Use temp pointers as we may come back and use the same NZ data and metaInfo
  // multiple times if multiple offsets match the row we are updating
  mov           $w_metaPtrTemp, $w_metaPtr
  mov           $w_nzPtrTemp, $w_nzPtr


.ifc \TYPE, half
  // Unroll the loop so we can bundle. Load the column number from the metadata
  // Use the divide by multiply and shift method to remove the scaling on the column
  // and find the offset in words
  ldz16step     $w_temp, $mzero, $w_metaPtrTemp+=,1
  mul           $w_temp, $w_temp, $w_nzScaleFactor
  rpt           $w_metaColumns, (2f-1f)/8-1
1:
  {shr          $w_temp, $w_temp, (RECIPROCAL_MUL_SHIFT+YOffsetTypeDivShiftHalf)
   fnop}
  // Fetch the subT data, cast to float, apply scale and add it to the NZ data
  {ldb16        $a0, $mzero, $w_subTPtr, $w_temp
   fnop}
  // Load the column number from the metadata for the next loop pass
  {ldz16step    $w_temp, $mzero, $w_metaPtrTemp+=,1
   f16tof32     $a0,$a0}
  {ld32         $a1, $mzero, $w_nzPtrTemp, 0
   f32mul       $a0, $a0, $w_scale}
  // On the last pass this will multiply with an overread.  That's OK with integer data
  {mul          $w_temp, $w_temp, $w_nzScaleFactor
   f32add       $a0, $a1,$a0}
 {st32step     $a0, $mzero, $w_nzPtrTemp+=, 1
  fnop}
2:
.else
  // Unroll the loop so we can bundle. Load the column number from the metadata
  ldz16step     $w_temp, $mzero, $w_metaPtrTemp+=,1
  rpt           $w_metaColumns, (2f-1f)/8-1
1:
  // Use the divide by multiply and shift method to remove the scaling on the column
  // and find the offset in words
  {mul          $w_temp, $w_temp, $w_nzScaleFactor
   fnop}
  {shr          $w_temp, $w_temp, (RECIPROCAL_MUL_SHIFT+YOffsetTypeDivShiftFloat)
   fnop}
  // Fetch the subT data, apply scale and add it to the NZ data
  {ld32         $a0, $mzero, $w_subTPtr, $w_temp
   fnop}
  {ld32         $a1, $mzero, $w_nzPtrTemp, 0
   f32mul       $a0, $a0, $w_scale}
  // Load the column number from the metadata for the next loop pass
  {ldz16step    $w_temp, $mzero, $w_metaPtrTemp+=,1
   f32add       $a0, $a1,$a0}
  {st32step     $a0, $mzero, $w_nzPtrTemp+=, 1
   fnop}
2:
.endif

  // Restore registers which contain constants but were overwritten by the
  // inner loop above
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNS/4
  ld32          $w_rowOffset, $mvertex_base, WOFF_ROW_OFFSET/4
next_offset\@:
  // Step the subT ptr to the next row
.ifc \TYPE, half
  ldz16step     $mzero, $mzero, $w_subTPtr+=,$w_subTStride
.else
  ld32step      $mzero, $mzero, $w_subTPtr+=,$w_subTStride
.endif
  brnzdec       $w_offsetLoopsM1, offsets_loop\@
  // Having dealt with this row, we have accounted for this many columns
  // so subtract and continue if there are more columns (in more rows to deal with)
  sub           $w_totalNumY, $w_totalNumY, $w_metaColumns
  brnz          $w_totalNumY, meta_info_row_loop\@

  exitz         $mzero

.size \NAME, . - \NAME
.endm
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
#define s_numWorkers                m7

#define s_offsetToNextSubGroup      m8
#define s_xPartition                m8

#define s_nzPtr                     m9
#define s_rowsPerPartition          m10

// Defines just used for copying vertex state
#define s_offsetSize                m2
#define s_subTColumns               m3
#define s_offsetPtr                 m4
#define s_nzScaleFactor             m5
#define s_subTPtr                   m6
#define s_scale                     m7

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
  // Mainly copy vertex state to the workers.  All this needs to be done just
  // once so dealing with it up front saves time.
  ldz16     $s_offsetSize, $s_base, VOFF_OFFSET_SIZE/2
  add       $sp, $sp, -STACK_SIZE
  ld32      $s_scale, $s_base, VOFF_SCALE_PTR/4
  ld32      $s_offsetPtr, $s_base, VOFF_OFFSET_PTR/4
  ldz16     $s_subTColumns, $s_base, VOFF_SUBT_COLUMNS/2
  ldz16     $s_nzScaleFactor, $s_base, VOFF_NZ_SCALE_FACTOR/2
  ld32      $s_subTPtr, $s_base, VOFF_SUBT_PTR/4
  ld32      $s_scale, $s_scale, 0

  // Workers need size-1
  sub       $s_offsetSize, $s_offsetSize, 1

  st32      $lr, $sp, WOFF_SUPER_LR_STORE/4
  st32      $fp, $sp, WOFF_SUPER_FP_STORE/4

  st32      $s_offsetPtr, $sp, WOFF_OFFSET_PTR/4
  st32      $s_subTColumns, $sp, WOFF_SUBT_COLUMNS/4
  st32      $s_nzScaleFactor, $sp, WOFF_NZ_SCALE_FACTOR/4
  st32      $s_subTPtr, $sp, WOFF_SUBT_PTR/4
  st32      $s_offsetSize, $sp, WOFF_NUM_OFFSETS_M1/4
  st32      $s_scale, $sp, WOFF_SCALE/4

  // The supervisor loops over buckets and subGroups found in the buckets.
  // When a subGroup is found with the required Y partition, we pass pointers to
  // the NZ and metadata for that subGroup to the worker along with row count,
  // and row offset
  ld32      $s_requiredYPartition, $s_base, VOFF_Y_PARTITION_PTR/4
  ldz16     $s_requiredYPartition, $s_requiredYPartition, 0
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
  // metaInfo and NZ data to step.
  ldz16      $s_subgroupSparseElems, $s_metaPtr, MetaInfoSubGroupEntry_sparseElementCount/2
  ldz16      $s_offsetToNextSubGroup, $s_metaPtr, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  ldz16step  $mzero, $mzero, $s_metaPtr+=, $s_offsetToNextSubGroup
  // NZ data is always float, so always using 32 loads with nzPtr
  ld32step   $mzero, $mzero, $s_nzPtr+=, $s_subgroupSparseElems

subgroup_loop_first_entry\@:
  // Fetch pointer to the SubGroupEntry struct for the next subgroup
  ldz16      $s_subgroupID, $s_metaPtr, MetaInfoSubGroupEntry_id/2
  // ID = 0 means that's the last one in the bucket
  brz        $s_subgroupID, subgroup_loop_end\@
  // Otherwise, does the yPartition match the one we're looking for?
  ldz16      $s_yPartition, $s_metaPtr, MetaInfoSubGroupEntry_yPartition/2
  cmpeq      $s_yPartition, $s_yPartition, $s_requiredYPartition
  brz        $s_yPartition, subgroup_loop\@

  // If so, pass information to the workers and call them.

  // Find the offset of the row information stored in the meta data and pass it
  // to the workers
  ldz16      $s_xPartition, $s_metaPtr, MetaInfoSubGroupEntry_xPartition/2
  mul        $s_xPartition, $s_xPartition, $s_rowsPerPartition

  // Pass pointers to meta info and the nz to the workers.
  // Currently pointing at the start of the SubGroup entry,
  // Advance the pointer by the size of the subgroup entry plus
  // MetaInfoSubGroupEntry_numWorkers *  sizeof_MetaInfoSubGroupEntry
  // To point to the number of GradWWorkerEntries - pass this to the workers
  ldz16      $s_numWorkers, $s_metaPtr, MetaInfoSubGroupEntry_numWorkers/2
  mul        $s_numWorkers, $s_numWorkers, Sizeof_MetaInfoWorkerEntry
  add        $s_numWorkers, $s_numWorkers, sizeof_MetaInfoSubGroupEntry
  add        $s_numWorkers, $s_metaPtr, $s_numWorkers

  // The above supervisor code will have many pipline stalls
  // as we branch, and load/compute/store with register dependencies.
  // On all but the first pass we do some of that while workers are running
  // by syncing here instead of directly after runall: assuming 1 worker ends
  // before the others we can get branches and computation for free.  We have to
  // sync before we affect stacked vertex state for the workers to use though.
  sync       TEXCH_SYNCZONE_LOCAL
  st32       $s_numWorkers, $sp, WOFF_META_PTR/4
  st32       $s_nzPtr, $sp, WOFF_NZ_PTR/4
  st32       $s_xPartition, $sp, WOFF_ROW_OFFSET/4

  runall     $s_workerEntry, $sp, 0

  bri        subgroup_loop\@

subgroup_loop_end\@:
  brnzdec   $s_bucketCount, bucket_loop\@

  // Restore and exit....
  ld32      $lr, $sp, WOFF_SUPER_LR_STORE/4
  ld32      $fp, $sp, WOFF_SUPER_FP_STORE/4
  // Last sync to ensure all workers are done before exit
  sync       TEXCH_SYNCZONE_LOCAL
  add       $sp, $sp, STACK_SIZE
  br        $lr

.size \NAME, . - \NAME
.endm

//------------------------------------------------------------------------------
// Use the macros to instantiate both the supervisor entry points and the worker
// functions

INSTANTIATE_SUPERVISOR CODELET_SLICE_HALF half
INSTANTIATE_SUPERVISOR CODELET_SLICE_FLOAT float

INSTANTIATE_WORKER worker_float float
INSTANTIATE_WORKER worker_half half

#endif // __IPU__
