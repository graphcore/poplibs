// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of  SparseDenseMultiSlice vertex template variations.

// Restrictions
//
//  * The subT that slices are written to must be must be 32-bit aligned and
//    have columns which are 32 bits in length so that each row is also 32-bit
//    aligned. (This constrains half vertices to have an even number of columns)

#include "BlockSparseMatMulStructs.h.S"
#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Symbols
#define VERTEX_NAME __runCodelet_popsparse__SparseDenseMultiSliceBlock___\TYPE\()_\BYTES_PER_VECTOR

// Size constants
#define LOG2_SIZEOF_FLOAT          2
#define LOG2_SIZEOF_HALF           1

// Vertex state (Offsets in bytes)
#define VOFF_OFFSET_PTR           0
#define VOFF_BASET_NZ_PTR         4
#define VOFF_BASET_META_PTR       8 // Short span (20 lsbs address)
#define VOFF_SUBT_PTR             12
#define VOFF_BLOCK_ROWS           16
#define VOFF_BLOCK_COLUMNS        18
#define VOFF_ROWS_PER_PARTITION   20
#define VOFF_Y_PARTITION          22
#define VOFF_SUBT_COLUMNS         24
#define VOFF_OFFSET_SIZE          26

// Vertex state created by the supervisor on the stack and passed to workers
#define WOFF_NZ_PTR                0
#define WOFF_META_PTR              4
#define WOFF_OFFSET_PTR            8
#define WOFF_ROW_OFFSET            12
#define WOFF_BLOCK_ROWS            16
#define WOFF_SUBT_COLUMNS          20
#define WOFF_SUBT_PTR              24
#define WOFF_OFFSET_SIZE           28
#define WOFF_NUM_ROWS_M1           32
#define WOFF_SUBT_COLUMNSxWORKERS  36
#define WOFF_BLOCK_COLUMN_LOOPS    40
#define WOFF_BLOCK_ELEMENTS        44
#define WOFF_BLOCK_ELEMS_MINUS_ROW 48
#define WOFF_BLOCK_BYTES_PER_ROW   52
#define WOFF_SUPER_LR_STORE        56
#define WOFF_SUPER_FP_STORE        60
#define STACK_SIZE                (WOFF_SUPER_FP_STORE+4)

// Worker stack
#define WSTACK_META_ROWS_M1       0
#define WSTACK_OFFSET_PTR         4
#define WSTACK_SUBT_PTR           8
#define WSTACK_OFFSET_LOOPS       12
#define WSTACK_META_ROW           16

// worker register aliases
#define w_subTStride              m0
#define w_subTColumnOffset        m0

#define w_nzPtr                   m1
#define w_metaPtr                 m2
#define w_offsetLoops             m3
#define w_subTPtr                 m4

#define w_metaRowStart            m5
#define w_columnsLoops            m5

#define w_metaColumns             m6

#define w_numRowsM1               m7
#define w_temp2                   m7

#define w_offsetPtr               m8

#define w_temp                    m9
#define w_columnCopyLoops         m9

#define w_nzPtrTemp               m10
#define w_metaPtrTemp             m11
#define w_metaRowsInBlock         m11
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
  add           $w_offsetLoops, $w_offsetLoops, CTXT_WORKERS
  sub           $w_offsetLoops, $w_offsetLoops, $w_id
  mul           $w_offsetLoops, $w_offsetLoops, 21845
  shr           $w_offsetLoops, $w_offsetLoops, 17

.endm
#undef w_id

//------------------------------------------------------------------------------
// Inner loop copy for 8 byte aligned, nx8 byte sized blocks
// Possible optimisation: If we constained subT to be in a different segment to
// the NZ data it would be possible to do this copy in 1 cycle. This would cost
// a few extra cycles to free up registers and tapack the addresses, so would
// only be useful for larger block sizes.  It could maybe be another specialisation.
.macro INNER_LOOP_BODY_8
  rpt            $w_columnCopyLoops, (2f-1f)/8 -1
1:
  {ld64step      $a0:1, $mzero, $w_nzPtrTemp+=,1
    fnop}
  {st64step      $a0:1, $w_subTPtr, $w_subTColumnOffset+=,1
    fnop}
2:
.endm
//------------------------------------------------------------------------------
// Inner loop copy for 4 byte aligned, nx4 byte sized blocks
.macro INNER_LOOP_BODY_4
  rpt           $w_columnCopyLoops, (2f-1f)/8 -1
1:
  {ld32step      $a0, $mzero, $w_nzPtrTemp+=,1
    fnop}
  {st32step      $a0, $w_subTPtr, $w_subTColumnOffset+=,1
    fnop}
2:
.endm
//------------------------------------------------------------------------------
// Inner loop copy for 2 byte aligned nx2 byte sized blocks
// As (probably) blocks are fairly small, and "nice" sized ones are dealt with in
// other cases it doesn't seem worth the complexity to target a 2-cycle loop which
// would be possible if the input(NZData) and the output(subT) are both aligned.
.macro INNER_LOOP_BODY_2
  // Data type is half.
  // subT is aligned (4-byte) and has an even number of columns so w_subTPtr will
  // always be aligned.  Therefore we only need to check the offset to see if we are
  // writing to an aligned or misaligned location
  and           $w_temp2, $w_subTColumnOffset, 2
  brnz          $w_temp2, not_align\@

align\@:
  // On a 4 byte boundary. There are always an odd number of columns in a block.
  // So we can write pairs of halves, and then a single one at the end
  ldb16step     $a0, $mzero, $w_nzPtrTemp+=,1
  rpt            $w_columnCopyLoops, (2f-1f)/8 -1
1:
  {ldb16step     $a1, $mzero, $w_nzPtrTemp+=,1
    fnop}
  {ldb16step     $a0, $mzero, $w_nzPtrTemp+=,1
   sort4x16hi    $a2, $a0, $a1}
  {st32step      $a2, $w_subTPtr, $w_subTColumnOffset+=,1
   fnop}
2:
  // Sub-word write for the last one - read one ahead to preserve
  ldb16          $a1, $w_subTPtr, $w_subTColumnOffset,1
  sort4x16hi     $a2, $a0, $a1
  st32step       $a2, $w_subTPtr, $w_subTColumnOffset+=,1
  bri continue\@

not_align\@:
  // Not on a 4-byte boundary.  Use the fact that the number of columns in a
  // block must be odd (Otherwise a different vertex variant would have been picked).
  // We can now write the individual misaligned half, then pairs of halves and
  // will have an odd number always.

  // Create a rounded down address, for a 4 byte boundary and sub-word write
  // for the first one
  andc           $w_subTColumnOffset, $w_subTColumnOffset, 2
  ldb16step      $a0, $mzero, $w_nzPtrTemp+=,1
  ldb16          $a1, $w_subTPtr, $w_subTColumnOffset, 0
  // Load a 1st one to unroll the loop
  {ldb16step     $a0, $mzero, $w_nzPtrTemp+=,1
   sort4x16hi    $a2, $a1, $a0}
  st32step       $a2, $w_subTPtr, $w_subTColumnOffset+=,1

  {rpt            $w_columnCopyLoops, (2f-1f)/8 -1
   fnop}
1:
  {ldb16step     $a1, $mzero, $w_nzPtrTemp+=,1
    fnop}
  {ldb16step     $a0, $mzero, $w_nzPtrTemp+=,1
   sort4x16hi    $a2, $a0, $a1}
  {st32step      $a2, $w_subTPtr, $w_subTColumnOffset+=,1
   fnop}
2:
  // Undo the increment from the over-read in the loop
  ldb16step      $azero, $mzero, $w_nzPtrTemp+=,-1

continue\@:
.endm

//------------------------------------------------------------------------------
// Worker float function
.macro INSTANTIATE_WORKER TYPE BYTES_PER_VECTOR LDINSTR

.section .text.worker_\TYPE\()_\BYTES_PER_VECTOR
.align 8
.type worker_\TYPE\()_\BYTES_PER_VECTOR, @function
.worker
  nop   // Rpt align
  // Entry point to use after the 1st call the supervisor vertex makes to the
  // workers.  Worker specific work division has been done and the result stored
  // on each worker's stack, so no need to calculate it again.
worker_stack_retained_\TYPE\()_\BYTES_PER_VECTOR:
  ld32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4
  brpos         $w_offsetLoops, worker_continue\@
  // Exit if nothing to do
  exitz         $mzero

 // First time entry point - no use of state (stack) retention
worker_\TYPE\()_\BYTES_PER_VECTOR:
  WORKER_PREAMBLE \TYPE
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

worker_continue\@:
  ld32          $w_nzPtr, $mvertex_base, WOFF_NZ_PTR/4
  ld32          $w_metaPtr, $mvertex_base, WOFF_META_PTR/4
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNSxWORKERS/4

  ld32          $w_numRowsM1, $mvertex_base, WOFF_NUM_ROWS_M1/4
  ld32          $w_metaRowsInBlock, $mvertex_base, WOFF_BLOCK_ROWS/4

meta_info_row_loop\@:
  // Load the row that the block starts at and add the offset
  ldz16step     $w_metaRowStart, $mzero, $w_metaPtr+=,1
  ld32          $w_temp, $mvertex_base, WOFF_ROW_OFFSET/4
  add           $w_metaRowStart, $w_metaRowStart, $w_temp

  // Load the loop count, offsets and subT pointers for this worker's work
  ld32          $w_offsetPtr, $mworker_base, WSTACK_OFFSET_PTR/4
  ld32          $w_subTPtr, $mworker_base, WSTACK_SUBT_PTR/4
  ld32          $w_offsetLoops, $mworker_base, WSTACK_OFFSET_LOOPS/4

  // The number of columns that have NZ data in this metaData row
  ldz16step     $w_metaColumns, $mzero, $w_metaPtr+=,1
  add           $w_metaColumns, $w_metaColumns, 1

offsets_loop\@:
  // Load the next offset and compare to the rows spanned by the block found in
  // the metaData.  Doing this using subtract not compare means we can retrieve
  // the result later from the w_temp register
  ld32step      $w_temp, $mzero, $w_offsetPtr+=, CTXT_WORKERS
  sub           $w_temp, $w_temp, $w_metaRowStart
  brneg         $w_temp, next_offset\@
  sub           $w_temp, $w_temp, $w_metaRowsInBlock
  brpos         $w_temp, next_offset\@
  // Found a match, read data and populate. Use temp ptrs as we may have to do this
  // for many slices with the same offset.
  // Register store - although this looks like an inner loop it is reached
  // conditionally so storing here is probably more efficient than storing in outer
  // loops
  st32          $w_metaRowStart, $mworker_base, WSTACK_META_ROW/4
  st32          $w_numRowsM1, $mworker_base, WSTACK_META_ROWS_M1/4

  // It's not necessarily the 1st row in the block that matched so offset the
  // NZ pointer by columns * (1stRowInBlock - row) elements.
  // This offset, applied once is applied through all blocks of columns in the block
  add           $w_temp, $w_temp, $w_metaRowsInBlock
  ld32          $w_nzPtrTemp, $mvertex_base, WOFF_BLOCK_BYTES_PER_ROW/4
  mul           $w_nzPtrTemp, $w_temp, $w_nzPtrTemp
  add           $w_nzPtrTemp, $w_nzPtr, $w_nzPtrTemp

  mov           $w_metaPtrTemp, $w_metaPtr

  sub           $w_columnsLoops, $w_metaColumns, 1
  ld32          $w_columnCopyLoops, $mvertex_base, WOFF_BLOCK_COLUMN_LOOPS/4
.ifc \BYTES_PER_VECTOR, 2
  shr            $w_columnCopyLoops, $w_columnCopyLoops, 1
.endif
column_block_loop\@:
  ldz16step     $w_subTColumnOffset, $mzero, $w_metaPtrTemp+=,1
.ifc \TYPE, float
  shl           $w_subTColumnOffset, $w_subTColumnOffset, LOG2_SIZEOF_FLOAT
.else
  shl           $w_subTColumnOffset, $w_subTColumnOffset, LOG2_SIZEOF_HALF
.endif

  // Do the actual copy of 1 row from the block
  INNER_LOOP_BODY_\BYTES_PER_VECTOR

  // Advance to the next block of NZ data
  ld32         $w_subTColumnOffset, $mvertex_base, WOFF_BLOCK_ELEMS_MINUS_ROW/4
  \LDINSTR     $a0, $mzero, $w_nzPtrTemp+=,$w_subTColumnOffset

  brnzdec      $w_columnsLoops, column_block_loop\@

  // register restore
  ld32         $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNSxWORKERS/4
  ld32         $w_metaRowsInBlock, $mvertex_base, WOFF_BLOCK_ROWS/4
  ld32         $w_metaRowStart, $mworker_base, WSTACK_META_ROW/4
  ld32         $w_numRowsM1, $mworker_base, WSTACK_META_ROWS_M1/4

next_offset\@:
  // Advance to this worker's next SubT slice to populate
  \LDINSTR     $azero, $mzero, $w_subTPtr+=, $w_subTStride
  brnzdec      $w_offsetLoops, offsets_loop\@

  // Advance main pointers to to metadata and NZ data for the next row
  ld32          $w_temp, $mvertex_base, WOFF_BLOCK_ELEMENTS/4
  mul           $w_temp, $w_temp, $w_metaColumns
  \LDINSTR      $azero, $mzero, $w_nzPtr+=, $w_temp
  ldz16step     $mzero, $mzero, $w_metaPtr+=, $w_metaColumns

  brnzdec       $w_numRowsM1, meta_info_row_loop\@

  exitz          $mzero
.size worker_\TYPE\()_\BYTES_PER_VECTOR, . - worker_stack_retained_\TYPE\()_\BYTES_PER_VECTOR
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
#define s_xPartition                m7

#define s_offsetToNextSubGroup      m8
#define s_temp                      m8

#define s_nzPtr                     m9
#define s_rowsPerPartition          m10

// Defines just used for copying vertex state
#define s_offsetSize                m2
#define s_subTColumns               m3
#define s_offsetPtr                 m4
#define s_blockRows                 m5
#define s_subTPtr                   m6
#define s_blockColumns              m7

#define SHORT_SPAN_ADDRESS_BITS    20

//------------------------------------------------------------------------------
// Macro to create entry points, supervisor code and call the workers
//------------------------------------------------------------------------------
.macro INSTANTIATE_SUPERVISOR TYPE LOG2_ELEMS_PER_VECTOR BYTES_PER_VECTOR

DEF_STACK_USAGE  (STACK_SIZE)  VERTEX_NAME

.section .text.VERTEX_NAME
.align 4
.globl VERTEX_NAME
.type VERTEX_NAME, @function
VERTEX_NAME:
.supervisor
  // Compute work division, compute useful constant values and copy vertex state
  // to the workers.  All this needs to be done just once so dealing with it up
  // front saves time.
  add       $sp, $sp, -STACK_SIZE
  ldz16     $s_subTColumns, $s_base, VOFF_SUBT_COLUMNS/2
  ld32      $s_offsetPtr, $s_base, VOFF_OFFSET_PTR/4
  ldz16     $s_blockRows, $s_base, VOFF_BLOCK_ROWS/2
  ldz16     $s_blockColumns, $s_base, VOFF_BLOCK_COLUMNS/2
  ld32      $s_subTPtr, $s_base, VOFF_SUBT_PTR/4
  ldz16     $s_offsetSize, $s_base, VOFF_OFFSET_SIZE/2

  st32      $s_subTColumns, $sp, WOFF_SUBT_COLUMNS/4
  mul       $s_subTColumns, $s_subTColumns, CTXT_WORKERS

  st32      $fp, $sp, WOFF_SUPER_FP_STORE/4
  st32      $lr, $sp, WOFF_SUPER_LR_STORE/4

.ifc \TYPE, half
  shl       $s_temp, $s_blockColumns, LOG2_SIZEOF_HALF
.else
  shl       $s_temp, $s_blockColumns, LOG2_SIZEOF_FLOAT
.endif

  st32      $s_offsetPtr, $sp, WOFF_OFFSET_PTR/4
  st32      $s_blockRows, $sp, WOFF_BLOCK_ROWS/4

  st32      $s_subTPtr, $sp, WOFF_SUBT_PTR/4
  st32      $s_offsetSize, $sp, WOFF_OFFSET_SIZE/4
  st32      $s_subTColumns, $sp, WOFF_SUBT_COLUMNSxWORKERS/4
  st32      $s_temp, $sp, WOFF_BLOCK_BYTES_PER_ROW/4

  mul       $s_blockRows, $s_blockColumns, $s_blockRows
  st32      $s_blockRows, $sp, WOFF_BLOCK_ELEMENTS/4
  sub       $s_blockRows, $s_blockRows, $s_blockColumns
  st32      $s_blockRows, $sp, WOFF_BLOCK_ELEMS_MINUS_ROW/4
.ifnc \LOG2_ELEMS_PER_VECTOR, 0
  shr       $s_blockColumns, $s_blockColumns, \LOG2_ELEMS_PER_VECTOR
.endif
  st32      $s_blockColumns, $sp, WOFF_BLOCK_COLUMN_LOOPS/4

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
  setzi     $s_workerEntry, worker_\TYPE\()_\BYTES_PER_VECTOR
bucket_loop\@:
  ld32step   $s_metaPtr, $mzero, $s_metaPtrPtr+=, 1
  ld32step   $s_nzPtr, $mzero, $s_nzPtrPtr+=, 1
  bri        subgroup_loop_first_entry\@

subgroup_loop\@:
  // Advance through the bucket to the next sub group.  Use the size of the
  // metaInfo and NZ data to step
  ldz16      $s_subgroupSparseElems, $s_metaPtr, MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries/2
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
  // As work is divided by offset, each worker processes all the metaInfo rows,
  // columns.  So pass a ptr to that
  ldz16      $s_temp, $s_metaPtr, MetaInfoSubGroupEntry_numGradWWorkers/2
  mul        $s_temp, $s_temp, sizeof_MetaInfoGradWWorkerEntry
  add        $s_temp, $s_temp, sizeof_MetaInfoSubGroupEntry
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
  setzi      $s_workerEntry, worker_stack_retained_\TYPE\()_\BYTES_PER_VECTOR
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

.size VERTEX_NAME, . - VERTEX_NAME
.endm
//------------------------------------------------------------------------------
// Use the macros above to instantiate vertices.
// We have float, half variants which are supported by worker functions that
// copy 2, 4 or 8 bytes in their inner loop.  Of course 2 bytes is for half only

INSTANTIATE_SUPERVISOR float 0 4
INSTANTIATE_SUPERVISOR float 1 8

INSTANTIATE_SUPERVISOR half 0 2
INSTANTIATE_SUPERVISOR half 1 4
INSTANTIATE_SUPERVISOR half 2 8

INSTANTIATE_WORKER float 4 ld32step
INSTANTIATE_WORKER float 8 ld32step

INSTANTIATE_WORKER half 2 ldb16step
INSTANTIATE_WORKER half 4 ldb16step
INSTANTIATE_WORKER half 8 ldb16step

#endif // __IPU__
