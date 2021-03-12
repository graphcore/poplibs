// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of  SparseDenseMultiUpdateAdd vertex template variations.

#include "BlockSparseMatMulStructs.h.S"
#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Symbols
#define VERTEX_NAME __runCodelet_popsparse__SparseDenseMultiUpdateAddBlock___\TYPE\()_\VECTORISE

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
#define VOFF_SUBT_COLUMNS         22
#define VOFF_Y_PARTITION_PTR      24
#define VOFF_SCALE_PTR            28
#define VOFF_OFFSET_SIZE          32

// Vertex state created by the supervisor on the stack and passed to workers
#define WOFF_NZ_PTR                0
#define WOFF_META_PTR              4
#define WOFF_OFFSET_PTR            8
#define WOFF_ROW_OFFSET            12
#define WOFF_BLOCK_ROWS            16
#define WOFF_SUBT_COLUMNS          20
#define WOFF_SUBT_PTR              24
#define WOFF_NUM_OFFSETS_M1        28
#define WOFF_SCALE                 32
#define WOFF_BLOCK_COPY_LOOPS      36
#define WOFF_BLOCK_ELEMS           40
#define WOFF_BLOCK_BYTES_PER_ROW   44
#define WOFF_BLOCK_ELEMS_MINUS_ROW 48
#define WOFF_SUPER_LR_STORE        52
#define WOFF_SUPER_FP_STORE        56
#define STACK_SIZE                (WOFF_SUPER_FP_STORE+4)

// Worker stack
#define WSTACK_META_BLOCKS_IN_ROW 0

// worker register aliases
#define w_subTColumnOffset        m0
#define w_id                      m0
#define w_metaRowsInBlock         m0
#define w_temp2                   m0

#define w_numWorkers              m1
#define w_nzPtr                   m1

#define w_metaPtr                 m2
#define w_offsetLoopsM1           m3

#define w_subTPtr                 m4
#define w_offsetsYInSFirst        m4

#define w_metaRowStart            m5
#define w_metaBlocksInRow         m6
#define w_totalNumY               m7
#define w_offsetPtr               m8
#define w_temp                    m9

#define w_nzPtrTemp               m10
#define w_rowOffset               m10

#define w_metaPtrTemp             m11
#define w_subTStride              m11

#define w_scale                   a7

//------------------------------------------------------------------------------
// Non-vectorised update of a block of columns for half.  Unrolled, so updates 1
// more than the loop count
.macro COPY_BLOCK_COLUMNS_half_false
  // Scale the offset based on the size of the type of subT
  shl           $w_subTColumnOffset, $w_subTColumnOffset, LOG2_SIZEOF_HALF

  // Update : baseT = baseT + subT * scale
  ldb16step     $a2, $w_subTPtr, $w_subTColumnOffset+=,1
  {ldb16step    $a0, $w_subTPtr, $w_subTColumnOffset+=,1
   f16tof32     $a2, $a2}
  {ld32         $a1, $mzero, $w_nzPtrTemp,0
   f32mul       $a2, $a2, $w_scale}

  {rpt          $w_temp, (2f-1f)/8-1
   f32add       $a2, $a1, $a2}
1:
  {st32step     $a2, $mzero, $w_nzPtrTemp+=,1
   f16tof32     $a2, $a0}
  {ld32         $a1, $mzero, $w_nzPtrTemp,0
   f32mul       $a2, $a2, $w_scale}
  // Reading the next subT element - an overread on the last pass
  {ldb16step    $a0, $w_subTPtr, $w_subTColumnOffset+=,1
   f32add       $a2, $a1, $a2}
2:
  st32step      $a2, $mzero, $w_nzPtrTemp+=,1
.endm

//------------------------------------------------------------------------------
// Vectorised update of a block of columns for half.  Unrolled, so updates 1
// more than the loop count.
// Takes 3 cycles for 2 elements.  If we were to vectorise to deal with 4 halves
// we could use ld64 BUT would need 2x f16v2fof32, 2x f32v2mul and 2x f32v2add
// instructions = 6 cycles for 4 elements which is no faster and can be used in
// fewer cases
//
// Possible optimisation: using tapack addresses,constraining nz data to
// interleaved memory and using f32v2axpy we can do the inner loop in 1 less
// cycle.  This costs setup cycles as per the comment for the float version but also
// the pipeline of the f32v2axpy means the loop would need to be 2x64 ahead for
// readPtr vs writePtr.  This doesn't work in interleaved memory!  Maybe a more
// elaborate loop could be devised but should be added as a specialisation if needed
.macro COPY_BLOCK_COLUMNS_half_true
  // Scale the offset based on the size of the type of subT
  shl           $w_subTColumnOffset, $w_subTColumnOffset, LOG2_SIZEOF_HALF

  ld32step      $a2, $w_subTPtr, $w_subTColumnOffset+=,1
  {ld32step     $a0, $w_subTPtr, $w_subTColumnOffset+=,1
   f16v2tof32   $a2:3, $a2}
  {ld64         $a2:3, $mzero, $w_nzPtrTemp,0
   f32v2mul     $a4:5, $w_scale:B, $a2:3}

  {rpt          $w_temp, (2f-1f)/8-1
   f32v2add     $a4:5, $a4:5, $a2:3}
1:
  {st64step     $a4:5, $mzero, $w_nzPtrTemp+=,1
   f16v2tof32   $a0:1, $a0}
  {ld64         $a2:3, $mzero, $w_nzPtrTemp,0
   f32v2mul     $a4:5, $w_scale:B, $a0:1}
  // Reading the next subT element - an overread on the last pass
  {ld32step     $a0, $w_subTPtr, $w_subTColumnOffset+=,1
   f32v2add     $a4:5, $a4:5, $a2:3}
2:
  st64step      $a4:5, $mzero, $w_nzPtrTemp+=,1
.endm

//------------------------------------------------------------------------------
// Non-vectorised update of a block of columns for float.
.macro COPY_BLOCK_COLUMNS_float_false
  // Scale the offset based on the size of the type of subT
  shl           $w_subTColumnOffset, $w_subTColumnOffset, LOG2_SIZEOF_FLOAT

  // Read the first subT element before the loop
  ld32step      $a0, $w_subTPtr, $w_subTColumnOffset+=,1
  rpt           $w_temp, (2f-1f)/8-1
1:
  {ld32         $a1, $mzero, $w_nzPtrTemp,0
   f32mul       $a2, $a0, $w_scale}
  // Reading the next subT element - an overread on the last pass
  {ld32step     $a0, $w_subTPtr, $w_subTColumnOffset+=,1
   f32add       $a2, $a1, $a2}
  {st32step     $a2, $mzero, $w_nzPtrTemp+=,1
   fnop}
2:
.endm

//------------------------------------------------------------------------------
// Vectorised update of a block of columns for float.
// Possible optimisation: using tapack addresses and constraining nz data to
// interleaved memory we can do the inner loop in 1 less cycle.  This costs 4
// cycles around the loop to produce offset pointers, tapack and extract the
// pointer to advance to the next block.  Only worthwhile for larger blocks,
// and therefore adding a specialisation to do this would be the right approach.
.macro COPY_BLOCK_COLUMNS_float_true
  // Scale the offset based on the size of the type of subT
  shl           $w_subTColumnOffset, $w_subTColumnOffset, LOG2_SIZEOF_FLOAT

  // Read the first subT element before the loop
  ld64step      $a0:1, $w_subTPtr, $w_subTColumnOffset+=,1
  rpt           $w_temp, (2f-1f)/8-1
1:
  {ld64         $a2:3, $mzero, $w_nzPtrTemp,0
   f32v2mul     $a4:5, $w_scale:B, $a0:1}
  // Reading the next subT element - an overread on the last pass
  {ld64step     $a0:1, $w_subTPtr, $w_subTColumnOffset+=,1
   f32v2add     $a4:5, $a2:3, $a4:5}
  {st64step     $a4:5, $mzero, $w_nzPtrTemp+=,1
   fnop}
2:
.endm

//------------------------------------------------------------------------------
// Worker float, half function macro
.macro INSTANTIATE_WORKER TYPE VECTORISE
.section .text.worker_\TYPE\()_\VECTORISE
.align 8
.type worker_\TYPE\()_\VECTORISE, @function
.worker
.ifc \TYPE, float
  nop  // rpt align
.endif
worker_\TYPE\()_\VECTORISE:
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
  mul           $w_id, $w_id, sizeof_MetaInfoGradWWorkerEntry
  add           $w_metaPtr, $w_metaPtr, $w_id
  // Get the count of number of blocks in this worker's work and
  // an offset into the 1st row that the worker processes
  ldz16         $w_totalNumY, $w_metaPtr, MetaInfoGradWWorkerEntry_totalNumY/2
  ldz16         $w_offsetsYInSFirst, $w_metaPtr, MetaInfoGradWWorkerEntry_metaInfoOffsetToOffsetsYInSFirst/2

  // Point to the worker's meta data (block start) and nzData
  ld32          $w_nzPtr, $mvertex_base, WOFF_NZ_PTR/4
  ldz16         $w_temp, $w_metaPtr, MetaInfoGradWWorkerEntry_sparseOffset/2
  ld32step      $azero, $mzero, $w_nzPtr+=,$w_temp

  ldz16         $w_temp, $w_metaPtr, MetaInfoGradWWorkerEntry_metaInfoOffsetOutputEntry/2
  ldz16step     $mzero, $mzero, $w_metaPtr+=,$w_temp

  ld32          $w_scale, $mvertex_base, WOFF_SCALE/4
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNS/4

  // Load info about the 1st row of blocks the worker is to process.  This is a bit
  // different to other ones as this worker may not be assigned all the blocks
  // within the first row

  // Load the first row and apply the row offset
  ldz16step     $w_metaRowStart, $mzero, $w_metaPtr+=,1
  ld32          $w_rowOffset, $mvertex_base, WOFF_ROW_OFFSET/4
  add           $w_metaRowStart, $w_metaRowStart, $w_rowOffset

  // Load the number of blocks in this row, advance to this worker's work
  // allocation within the row.
  ldz16step     $w_metaBlocksInRow, $mzero, $w_metaPtr+=,1
  ldz16step     $mzero, $mzero, $w_metaPtr+=,$w_offsetsYInSFirst
  add           $w_metaBlocksInRow, $w_metaBlocksInRow, 1
  sub           $w_metaBlocksInRow, $w_metaBlocksInRow, $w_offsetsYInSFirst
  // Ensure that the total columns for this worker to process isn't exceeded, if
  // it is to process only part of this row
  min           $w_metaBlocksInRow, $w_metaBlocksInRow, $w_totalNumY
  st32          $w_metaBlocksInRow, $mworker_base, WSTACK_META_BLOCKS_IN_ROW/4

  ld32          $w_metaRowsInBlock, $mvertex_base, WOFF_BLOCK_ROWS/4
  bri           meta_info_row_loop_first_time\@

meta_info_row_loop\@:
  // Advance main pointers to to metadata and NZ data for the next row
  // NZ pointer advances by 1 whole block * w_metaBlocksInRow
  ldz16step     $mzero, $mzero, $w_metaPtr+=, $w_metaBlocksInRow
  ld32          $w_temp, $mvertex_base, WOFF_BLOCK_ELEMS/4
  mul           $w_temp, $w_temp, $w_metaBlocksInRow
  ld32step      $azero, $mzero, $w_nzPtr+=, $w_temp

  // Get the next row and apply the row offset
  ldz16step     $w_metaRowStart, $mzero, $w_metaPtr+=,1
  add           $w_metaRowStart, $w_metaRowStart, $w_rowOffset

  // Get the number of columns in this row, but this worker may only need to
  // process some of them
  ldz16step     $w_metaBlocksInRow, $mzero, $w_metaPtr+=,1
  add           $w_metaBlocksInRow, $w_metaBlocksInRow, 1
  min           $w_metaBlocksInRow, $w_metaBlocksInRow, $w_totalNumY
  st32          $w_metaBlocksInRow, $mworker_base, WSTACK_META_BLOCKS_IN_ROW/4

meta_info_row_loop_first_time\@:
  // Prepare to process the offsets
  ld32          $w_offsetPtr, $mvertex_base, WOFF_OFFSET_PTR/4
  ld32          $w_offsetLoopsM1, $mvertex_base, WOFF_NUM_OFFSETS_M1/4
  ld32          $w_subTPtr, $mvertex_base, WOFF_SUBT_PTR/4

offsets_loop\@:
  // Load the next row in the offsets, and compare it to the range of rows
  // covered by the block found in the metainfo
  // Comparing using sub saves actually creating metaRowEnd and doesn't destroy
  // the row loaded so we save registers (We can get it back by adding again)
  ld32step      $w_temp, $mzero, $w_offsetPtr+=,1
  sub           $w_temp, $w_temp, $w_metaRowStart
  brneg         $w_temp, next_offset\@
  sub           $w_temp, $w_temp, $w_metaRowsInBlock
  brpos         $w_temp, next_offset\@

  // A row matches, so update it into the (always float) result.
  // Use temp pointers as we may come back and use the same NZ data and metaInfo
  // multiple times if multiple offsets match the row we are updating
  mov           $w_metaPtrTemp, $w_metaPtr

  // It's not necessarily the 1st row in the block that matched so offset the
  // NZ pointer by columns * (1stRowInBlock - row) elements.
  // This offset, applied once is applied through all blocks of columns in the block
  add           $w_temp, $w_temp, $w_metaRowsInBlock
  ld32          $w_nzPtrTemp, $mvertex_base, WOFF_BLOCK_BYTES_PER_ROW/4
  mul           $w_nzPtrTemp, $w_temp, $w_nzPtrTemp
  add           $w_nzPtrTemp, $w_nzPtr, $w_nzPtrTemp

  // Using brnzdec so sub 1 to loop
  sub           $w_metaBlocksInRow, $w_metaBlocksInRow, 1
  // Load the inner loop count, which was calculated by the supervisor to the
  // requirements of the combination of "type, vectorised"
  ld32          $w_temp, $mvertex_base, WOFF_BLOCK_COPY_LOOPS/4
column_blocks_loop\@:
  // Prepare to copy a block of columns
  ldz16step     $w_subTColumnOffset, $mzero, $w_metaPtrTemp+=,1

  // Inner loop to copy a block of columns
  COPY_BLOCK_COLUMNS_\TYPE\()_\VECTORISE

  // We have advanced the pointer by 1 row, so complete the advance of 1 block
  // by advancing (1 block - 1 row) more elements
  ld32          $w_temp2, $mvertex_base, WOFF_BLOCK_ELEMS_MINUS_ROW/4
  ld32step      $mzero, $mzero, $w_nzPtrTemp+=, $w_temp2
  brnzdec       $w_metaBlocksInRow, column_blocks_loop\@

  // Restore registers which contain constants but were overwritten by the
  // inner loop above
  ld32          $w_subTStride, $mvertex_base, WOFF_SUBT_COLUMNS/4
  ld32          $w_rowOffset, $mvertex_base, WOFF_ROW_OFFSET/4
  ld32          $w_metaRowsInBlock, $mvertex_base, WOFF_BLOCK_ROWS/4
  // And those that were stored above
  ld32          $w_metaBlocksInRow, $mworker_base, WSTACK_META_BLOCKS_IN_ROW/4
next_offset\@:
  // Step the subT ptr to the next row
.ifc \TYPE, half
  ldz16step     $mzero, $mzero, $w_subTPtr+=,$w_subTStride
.else
  ld32step      $mzero, $mzero, $w_subTPtr+=,$w_subTStride
.endif
  brnzdec       $w_offsetLoopsM1, offsets_loop\@
  // Having dealt with this row, we have accounted for this many blocks of columns
  // so subtract and continue if there are more blocks of columns
  // (in other blocks) to deal with
  sub           $w_totalNumY, $w_totalNumY, $w_metaBlocksInRow
  brnz          $w_totalNumY, meta_info_row_loop\@

  exitz         $mzero

.size worker_\TYPE\()_\VECTORISE, . - worker_\TYPE\()_\VECTORISE
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
#define s_blockCopyLoops            m1
#define s_offsetSize                m2
#define s_subTColumns               m3
#define s_offsetPtr                 m4
#define s_blockRows                 m5
#define s_subTPtr                   m6
#define s_scale                     m7
#define s_blockColumns              m8
#define s_blockSize                 m9

#define SHORT_SPAN_ADDRESS_BITS    20
//------------------------------------------------------------------------------
// Macro to create entry points, supervisor code and call the workers
//------------------------------------------------------------------------------
.macro INSTANTIATE_SUPERVISOR TYPE VECTORISE

DEF_STACK_USAGE  (STACK_SIZE)  VERTEX_NAME

.section .text.VERTEX_NAME
.align 4
.globl VERTEX_NAME
.type VERTEX_NAME, @function
VERTEX_NAME:
.supervisor
  // Mainly copy vertex state to the workers, plus some calculation of constants.
  // All this needs to be done just once so dealing with it up front saves time.
  ld32      $s_scale, $s_base, VOFF_SCALE_PTR/4
  ldz16     $s_offsetSize, $s_base, VOFF_OFFSET_SIZE/2
  add       $sp, $sp, -STACK_SIZE
  ld32      $s_offsetPtr, $s_base, VOFF_OFFSET_PTR/4
  ldz16     $s_subTColumns, $s_base, VOFF_SUBT_COLUMNS/2
  ldz16     $s_blockColumns, $s_base, VOFF_BLOCK_COLUMNS/2
  ldz16     $s_blockRows, $s_base, VOFF_BLOCK_ROWS/2
  ld32      $s_scale, $s_scale, 0

  st32      $lr, $sp, WOFF_SUPER_LR_STORE/4
  st32      $fp, $sp, WOFF_SUPER_FP_STORE/4
  mul       $s_blockSize, $s_blockRows, $s_blockColumns

// Compute the inner loop count - loops to update a block of columns.
// If vectorised we update 2 elements per pass, so have to divide by 2.
// In addition, the loops processing half data are unrolled so need 1 less loop pass
.ifc \TYPE, half
  .ifc \VECTORISE, true
    shr       $s_blockCopyLoops, $s_blockColumns, 1
    sub       $s_blockCopyLoops, $s_blockCopyLoops,1
  .else
    sub       $s_blockCopyLoops, $s_blockColumns,1
  .endif
.else
  .ifc \VECTORISE, true
    shr       $s_blockCopyLoops, $s_blockColumns, 1
  .else
    mov       $s_blockCopyLoops, $s_blockColumns
  .endif
.endif

  // Workers need size-1
  sub       $s_offsetSize, $s_offsetSize, 1
  st32      $s_offsetPtr, $sp, WOFF_OFFSET_PTR/4
  st32      $s_subTColumns, $sp, WOFF_SUBT_COLUMNS/4
  st32      $s_blockCopyLoops, $sp, WOFF_BLOCK_COPY_LOOPS/4
  ld32      $s_subTPtr, $s_base, VOFF_SUBT_PTR/4

  st32      $s_blockSize, $sp, WOFF_BLOCK_ELEMS/4
  sub       $s_blockSize, $s_blockSize, $s_blockColumns
  shl       $s_blockColumns, $s_blockColumns, LOG2_SIZEOF_FLOAT

  st32      $s_blockRows, $sp, WOFF_BLOCK_ROWS/4
  st32      $s_offsetSize, $sp, WOFF_NUM_OFFSETS_M1/4
  st32      $s_scale, $sp, WOFF_SCALE/4
  st32      $s_subTPtr, $sp, WOFF_SUBT_PTR/4
  st32      $s_blockSize, $sp, WOFF_BLOCK_ELEMS_MINUS_ROW/4
  st32      $s_blockColumns, $sp, WOFF_BLOCK_BYTES_PER_ROW/4

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

  // Setup worker entry point
  setzi     $s_workerEntry, worker_\TYPE\()_\VECTORISE

bucket_loop\@:
  ld32step   $s_metaPtr, $mzero, $s_metaPtrPtr+=, 1
  ld32step   $s_nzPtr, $mzero, $s_nzPtrPtr+=, 1
  bri        subgroup_loop_first_entry\@

subgroup_loop\@:
  // Advance through the bucket to the next sub group.  Use the size of the
  // metaInfo and NZ data to step.
  ldz16      $s_subgroupSparseElems, $s_metaPtr, MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries/2
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
  // Advance the pointer by the size of the subgroup entry - 1 element
  // To point to the number of GradWWorkerEntries - pass this to the workers
  add        $s_numWorkers, $s_metaPtr, sizeof_MetaInfoSubGroupEntry - 2

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
done\@:
  // Restore and exit....
  ld32      $lr, $sp, WOFF_SUPER_LR_STORE/4
  ld32      $fp, $sp, WOFF_SUPER_FP_STORE/4
  // Last sync to ensure all workers are done before exit
  sync       TEXCH_SYNCZONE_LOCAL
  add       $sp, $sp, STACK_SIZE
  br        $lr

.size VERTEX_NAME, . - VERTEX_NAME
.endm

//------------------------------------------------------------------------------
// Use the macros to instantiate both the supervisor entry points and the worker
// functions

INSTANTIATE_SUPERVISOR half true
INSTANTIATE_SUPERVISOR float true
INSTANTIATE_SUPERVISOR half false
INSTANTIATE_SUPERVISOR float false

INSTANTIATE_WORKER float true
INSTANTIATE_WORKER half true
INSTANTIATE_WORKER float false
INSTANTIATE_WORKER half false

#endif // __IPU__
