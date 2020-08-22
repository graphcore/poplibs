// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication R = Q * S' Where
// Q and S are dense matrices and R is a sparse matrix
//
// This serves the purpose of computing the entries of the
// sparse gradients with respect to weights.

#ifdef __IPU__
#include "BlockSparseMatMulGradW.h.S"
#include "BlockSparseMatMulStructs.h.S"
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================

// worker registers
#define w_metaInfo                         m0
#define w_rGradBase                        m1
#define w_numZ                             m4
#define w_qGradBase                        m2

#define w_numWorkers                       m5
#define w_id                               m6
#define w_processWork                      m7
#define w_wkrInfoOffset                    m5
#define w_sparseOffset                     m5
#define w_metaInfoOffsetOutputEntry        m6
#define w_offsetY                          m7

#define w_totalNumY                        m8

#define w_numYm1                           m6
#define w_qGradBaseLoop                    m7
#define w_qGradBaseLoop1                   m5
#define w_sBaseLoop                        m11
#define w_offInSAndQ                       m10
#define w_loopCnt                          m2 
#define w_zStrideInS                       m9
#define w_zStrideInQ                       m3

#define fp_clr_reg                         a1

#define W_QGRAD_BASE_LOOP                  0
#define W_S_BASE_LOOP                      4
#define STACK_SIZE_WORKER                  (W_S_BASE_LOOP + 4)

DEF_STACK_USAGE STACK_SIZE_WORKER Block4x4SparseDenseMultiplyGradWHF
.section ".text.Block4x4SparseDenseMultiplyGradWHF", FUNCTION_IS_WORKER
.type Block4x4SparseDenseMultiplyGradWHF, @function
.globl Block4x4SparseDenseMultiplyGradWHF
.align 8
.worker
// worker code

Block4x4SparseDenseMultiplyGradWHF:
ld32              $w_metaInfo, $mvertex_base, W_METAINFO/4
ld32              $w_rGradBase, $mvertex_base, W_RGRAD_BASE/4
ld32              $w_numZ, $mvertex_base, W_NUM_Z/4
add               $w_numZ, $w_numZ, -1
ld32              $w_zStrideInS, $mvertex_base, W_ZSTRIDE_IN_S/4
ld32              $w_zStrideInQ, $mvertex_base, W_ZSTRIDE_IN_Q/4
// ld32step is used for Q
shl               $w_zStrideInQ, $w_zStrideInQ, 1

// The number of workers is the first field
// w_metaInfo -> worker entries
ldz16step         $w_numWorkers, $mzero, $w_metaInfo+=, 1
get               $w_id, $WSR
and               $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

// There are a max of worker entries as there are number of workers
cmpult            $w_processWork, $w_id, $w_numWorkers
brz               $w_processWork, LEndWorker

// point to this worker entry
// w_metaInfo -> &metaInfo->workerEntries[wid]
mul               $w_wkrInfoOffset, $w_id, sizeof_MetaInfoGradWWorkerEntry
add               $w_metaInfo, $w_metaInfo, $w_wkrInfoOffset

ldz16             $w_metaInfoOffsetOutputEntry, $w_metaInfo, MetaInfoGradWWorkerEntry_metaInfoOffsetOutputEntry/2
ldz16             $w_offsetY, $w_metaInfo, MetaInfoGradWWorkerEntry_metaInfoOffsetToOffsetsYInSFirst/2
ldz16             $w_totalNumY, $w_metaInfo, MetaInfoGradWWorkerEntry_totalNumY/2

// !!! Assumption here that sparse offset is the first entry in the table
ldz16step         $w_sparseOffset, $mzero, $w_metaInfo+=, $w_metaInfoOffsetOutputEntry
// dummy load to move to gradient base for this worker
ld32step          $mzero, $mzero, $w_rGradBase+=, $w_sparseOffset

ldz16step         $w_offInSAndQ, $mzero, $w_metaInfo+=, 1 
ldz16step         $w_numYm1, $mzero, $w_metaInfo+=, 1 
{
  add               $w_numYm1, $w_numYm1, 1
  setzi             $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}

// move meta info pointer by doing a dummy load
{
  ldz16step         $mzero, $mzero, $w_metaInfo+=, $w_offsetY
  uput              $FP_CLR, $fp_clr_reg 
}
sub               $w_numYm1, $w_numYm1, $w_offsetY
min               $w_numYm1, $w_numYm1, $w_totalNumY

LMainLoopX:
add               $w_numYm1, $w_numYm1, -1
ld32              $w_qGradBase, $mvertex_base, W_QGRAD_BASE/4
ldb16step         $azero, $mzero, $w_qGradBase+=, $w_offInSAndQ
st32              $w_qGradBase, $mworker_base, W_QGRAD_BASE_LOOP/4
ldz16step         $w_offInSAndQ, $mzero, $w_metaInfo+=, 1

LMainLoopY:
  ld32              $w_sBaseLoop, $mvertex_base, W_S_BASE/4
  ldz16step         $mzero, $mzero, $w_sBaseLoop+=, $w_offInSAndQ
  st32              $w_sBaseLoop, $mworker_base, W_S_BASE_LOOP/4
  ld32              $w_qGradBaseLoop1, $mworker_base, W_QGRAD_BASE_LOOP/4
  mov               $w_qGradBaseLoop, $w_qGradBaseLoop1
  setzi             $w_loopCnt, 1

LqRowLoop:
    ld64              $a4:5, $mzero, $w_rGradBase, 2
    {
      ld64              $a4:5, $mzero, $w_rGradBase, 3
      f32v2gina         $azeros, $a4:5, 0
    }
    {
      ld64              $a4:5, $mzero, $w_rGradBase, 0
      f32v2gina         $azeros, $a4:5, 0
    }
    ld64              $a6:7, $mzero, $w_rGradBase, 1
    {
      ld64step          $a0:1, $mzero, $w_sBaseLoop+=, $w_zStrideInS
      fnop
    }
    {
      ld32step          $a2, $mzero, $w_qGradBaseLoop+=, $w_zStrideInQ
      f32v4acc          $a4:7
    }
    {
      rpt               $w_numZ, 2
      f16v4mul          $a4:5, $a2:BL, $a0:1
    }
      {
        ld64step          $a0:1, $mzero, $w_sBaseLoop+=, $w_zStrideInS
        f16v4mul          $a6:7, $a2:BU, $a0:1
      }
      {
        ld32step          $a2, $mzero, $w_qGradBaseLoop+=, $w_zStrideInQ
        f16v8acc          $a4:7
      }
      {
        nop
        f16v4mul          $a4:5, $a2:BL, $a0:1
      }
    {
      ld32              $w_sBaseLoop,  $mworker_base, W_S_BASE_LOOP/4
      f16v4mul          $a6:7, $a2:BU, $a0:1
    }
    {
      add               $w_qGradBaseLoop1, $w_qGradBaseLoop1, 4
      f16v8acc          $a4:7
    }
    {
      mov               $w_qGradBaseLoop, $w_qGradBaseLoop1
      f32v2gina         $a0:1, $azeros, 0
    }
    {
      st64step          $a0:1, $mzero, $w_rGradBase+=, 1
      f32v2gina         $a0:1, $azeros, 0
    }
    {
      st64step          $a0:1, $mzero, $w_rGradBase+=, 1
      f32v2gina         $a0:1, $azeros, 0
    }
    {
      st64step          $a0:1, $mzero, $w_rGradBase+=, 1
      f32v2gina         $a0:1, $azeros, 0
    }
    st64step          $a0:1, $mzero, $w_rGradBase+=, 1
    brnzdec           $w_loopCnt, LqRowLoop

  add               $w_totalNumY, $w_totalNumY, -1
  ldz16step         $w_offInSAndQ, $mzero, $w_metaInfo+=, 1
  brnzdec           $w_numYm1, LMainLoopY

ldz16step         $w_numYm1, $mzero, $w_metaInfo+=, 1
add               $w_numYm1, $w_numYm1, 1
min               $w_numYm1, $w_numYm1, $w_totalNumY
brnz              $w_numYm1, LMainLoopX

LEndWorker:
exitz             $mzero

.size Block4x4SparseDenseMultiplyGradWHF, . - Block4x4SparseDenseMultiplyGradWHF

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
