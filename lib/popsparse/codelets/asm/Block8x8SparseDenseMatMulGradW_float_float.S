// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// 
// Performs a matrix multiplication R = Q * S where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 8x8
//

#ifdef __IPU__
#include "BlockSparseMatMulGradW.h.S"
#include "BlockSparseMatMulStructs.h.S"
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================

#define WORKER_SYMBOL Block8x8SparseDenseMultiplyGradWFF

// =============================================================================
// Worker

#define w_id m0
#define w_metaInfoPtr m1
#define w_sparseOffset m2
#define w_metaInfoOffsetOutputEntry m3
#define w_metaInfoOffsetToOffsetsYInSFirst m4
#define w_numYRemaining m5
#define w_numWorkerEntries m6
#define w_numZ m7
#define w_zStrideInQ m8
#define w_zStrideInS m9
#define w_column_counter m10
#define w_row_counter m11

#define WORKER_STACK_OFFSET_sBase 0
#define WORKER_STACK_OFFSET_sBase_innerLoop 4
#define WORKER_STACK_OFFSET_qGradBase 8
#define WORKER_STACK_OFFSET_qGradBase_innerLoop 12
#define WORKER_STACK_SIZE 16

DEF_STACK_USAGE WORKER_STACK_SIZE WORKER_SYMBOL
.section .text.WORKER_SYMBOL, FUNCTION_IS_WORKER
.align 8
.type WORKER_SYMBOL, @function
.globl WORKER_SYMBOL
.worker
// Align for rpt
WORKER_SYMBOL:
  get $w_id, $WSR
  and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

  ld32 $w_metaInfoPtr, $mvertex_base, W_METAINFO/4
  ldz16step $w_numWorkerEntries, $mzero, $w_metaInfoPtr+=, 1

  cmpult $w_numWorkerEntries, $w_id, $w_numWorkerEntries
  brz $w_numWorkerEntries, 0f
//================================
#undef w_numWorkerEntries
//================================
#define w_offsetToWorkerEntry m0
//================================
  mul $w_offsetToWorkerEntry, $w_id, sizeof_MetaInfoGradWWorkerEntry
//================================
#undef w_id
//================================
  add $w_metaInfoPtr, $w_metaInfoPtr, $w_offsetToWorkerEntry
//================================
#undef w_offsetToWorkerEntry
//================================

  ldz16 $w_numYRemaining, $w_metaInfoPtr, $mzero, MetaInfoGradWWorkerEntry_totalNumY/2
  // NOTE: Assumption numY for a worker is never 0 and the worker's entry
  // would not be present if this were the case
  add $w_numYRemaining, $w_numYRemaining, -1

  ldz16 $w_metaInfoOffsetOutputEntry, $w_metaInfoPtr, $mzero, MetaInfoGradWWorkerEntry_metaInfoOffsetOutputEntry/2
  ldz16 $w_metaInfoOffsetToOffsetsYInSFirst, $w_metaInfoPtr, $mzero, MetaInfoGradWWorkerEntry_metaInfoOffsetToOffsetsYInSFirst/2
#if MetaInfoGradWWorkerEntry_sparseOffset != 0
  #error Relying on sparseOffset being first in worker entries
#endif
  ldz16step $w_sparseOffset, $mzero, $w_metaInfoPtr+=, $w_metaInfoOffsetOutputEntry
//================================
#undef w_metaInfoOffsetOutputEntry
//================================
#define w_rGradBase m0
//================================

  // Load and offset sparse entries
  ld32 $w_rGradBase, $mvertex_base, $mzero, W_RGRAD_BASE/4
  ld32step $mzero, $mzero, $w_rGradBase+=, $w_sparseOffset
//================================
#undef w_sparseOffset
//================================
#define w_offsetXOrY m2
#define w_numYThisXm1 m3
//================================
  ldz16step $w_offsetXOrY, $mzero, $w_metaInfoPtr+=, 1
  ldz16step $w_numYThisXm1, $mzero, $w_metaInfoPtr+=, 1

  // First time into the loop over Y we have an offset
  sub $w_numYThisXm1, $w_numYThisXm1, $w_metaInfoOffsetToOffsetsYInSFirst
  ldz16step $mzero, $mzero, $w_metaInfoPtr+=, $w_metaInfoOffsetToOffsetsYInSFirst
//================================
#undef w_metaInfoOffsetToOffsetsYInSFirst
//================================
#define w_sBase m4
#define w_qGradBase m6
//================================

  ld32 $w_numZ, $mvertex_base, $mzero, W_NUM_Z/4
  // 1 less because of pipeline around rpt loop
  add $w_numZ, $w_numZ, -1
  ld32 $w_zStrideInQ, $mvertex_base, $mzero, W_ZSTRIDE_IN_Q/4
  ld32 $w_zStrideInS, $mvertex_base, $mzero, W_ZSTRIDE_IN_S/4

  // We load 2x 64-bits of S at a time hence take 1 64-bit multiple off
  // the stride to reach the next batch.
  { add $w_zStrideInS, $w_zStrideInS, -1
    setzi $a0, 1 << CSR_W_FP_CLR__ZAACC__SHIFT }
  // We load 32-bits of Q at a time hence multiply by 2 to give correct offset
  // in Z
  { shl $w_zStrideInQ, $w_zStrideInQ, 1
    uput $FP_CLR, $a0 }

  bri .LProcessFirstOutputEntry
.LProcessOutputEntry:
    ldz16step $w_offsetXOrY, $mzero, $w_metaInfoPtr+=, 1
    ldz16step $w_numYThisXm1, $mzero, $w_metaInfoPtr+=, 1

.LProcessFirstOutputEntry:
    min $w_numYThisXm1, $w_numYThisXm1, $w_numYRemaining

    ld32 $w_qGradBase, $mvertex_base, $mzero, W_QGRAD_BASE/4
    ld32step $azero, $mzero, $w_qGradBase+=, $w_offsetXOrY
    // Store away starting pointer for qGrad for this block
    st32 $w_qGradBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_qGradBase/4

.LProcessY:
      // Process an entire 8x8 block.
      // We iterate columns in the inner loop and rows in the outer loop
      // meaning we load a single element of qGrad and multiply/accumulate
      // with 4 elements of s, twice, before moving to the next row (next
      // element of qGrad).
      
      ldz16step $w_offsetXOrY, $mzero, $w_metaInfoPtr+=, 1
      ld32 $w_sBase, $mvertex_base, $mzero, W_S_BASE/4
      ld32step $azero, $mzero, $w_sBase+=, $w_offsetXOrY
      // Store away starting pointer for s for this block
      st32 $w_sBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_sBase/4

      setzi $w_row_counter, 7
.LProcessRow:
        // Save qGrad pointer as it will be used to iterate over numZ
        st32 $w_qGradBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_qGradBase_innerLoop/4
        setzi $w_column_counter, 1
.LProcess4Columns:
          // Save s pointer as it will be used to iterate over numZ
          st32 $w_sBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_sBase_innerLoop/4

          ld64 $a4:5, $w_rGradBase, $mzero, 0
          ld64 $a6:7, $w_rGradBase, $mzero, 1
          ld32step $a0, $mzero, $w_qGradBase+=, $w_zStrideInQ
          ld64step $a2:3, $mzero, $w_sBase+=, 1
          { rpt $w_numZ, (2f - 1f) / 8 - 1
            f32v4acc $a4:7 }
1:
          { ld64step $a2:3, $mzero, $w_sBase+=, $w_zStrideInS
            f32v2mul $a4:5, $a0:B, $a2:3 }
          { ld64step $a2:3, $mzero, $w_sBase+=, 1
            f32v2mul $a6:7, $a0:B, $a2:3 }
          { ld32step $a0, $mzero, $w_qGradBase+=, $w_zStrideInQ
            f32v4acc $a4:7 }
2:
          { ld64 $a2:3, $w_sBase, $mzero, 0
            f32v2mul $a4:5, $a0:B, $a2:3 }
          { ld32 $w_sBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_sBase_innerLoop/4
            f32v2mul $a6:7, $a0:B, $a2:3 }
          { ld64step $azeros, $mzero, $w_sBase+=, 2
            f32v4acc $a4:7 }
          { ld32 $w_qGradBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_qGradBase_innerLoop/4
            f32v2gina $a4:5, $azeros, 0 }
          { st64step $a4:5, $mzero, $w_rGradBase+=, 1
            f32v2gina $a4:5, $azeros, 0 }
          st64step $a4:5, $mzero, $w_rGradBase+=, 1
          brnzdec $w_column_counter, .LProcess4Columns
        
        ld32 $w_sBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_sBase/4
        ld32step $azero, $mzero, $w_qGradBase+=, 1
        brnzdec $w_row_counter, .LProcessRow
      ld32 $w_qGradBase, $mworker_base, $mzero, WORKER_STACK_OFFSET_qGradBase/4
      add $w_numYRemaining, $w_numYRemaining, -1
      brnzdec $w_numYThisXm1, .LProcessY
    brpos $w_numYRemaining, .LProcessOutputEntry
0:
  exitz $mzero

.size WORKER_SYMBOL, . - WORKER_SYMBOL

#endif // __IPU__
