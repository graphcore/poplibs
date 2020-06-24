// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication R = Q * S' Where
// Q and S are dense matrices and R is a sparse matrix
//
// This serves the purpose of computing the entries of the
// sparse gradients with respect to weights.

#ifdef __IPU__
#include "SparseDenseMatMulGradWElementWise.h"
#include "SparseDenseMatMulStructs.h"
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================

// worker registers
#define w_metaInfo                         m0
#define w_rGradBase                        m1
#define w_qGradBase                        m2
#define w_sBase                            m3
#define w_numZ                             m4
#define w_numWorkers                       m5
#define w_id                               m6
#define w_processWork                      m7
#define w_wkrInfoOffset                    m5
#define w_sparseOffset                     m5
#define w_metaInfoOffsetOutputEntry        m6
#define w_offsetY                          m7
#define w_totalNumY                        m8

#define w_offXInQ                          m5
#define w_numY                             m6
#define w_zEq2                             m7
#define w_zEq4                             m7
#define w_numZDiv2                         m7
#define w_numRem                           m4
#define w_qGradBaseLoop                    m10
#define w_sBaseLoop                        m11
#define w_packedPtr                        m10:11

#define w_offYInS                          m9

#define fp_clr_reg                         a1

DEF_STACK_USAGE 0 elemwiseSparseDenseMultiplyGradFF
.section ".text.elemwiseSparseDenseMultiplyGradFF", FUNCTION_IS_WORKER
.type elemwiseSparseDenseMultiplyGradFF, @function
.globl elemwiseSparseDenseMultiplyGradFF
.align 8
.worker
// worker code

elemwiseSparseDenseMultiplyGradFF:
ld32              $w_metaInfo, $mvertex_base, W_METAINFO/4
ld32              $w_rGradBase, $mvertex_base, W_RGRAD_BASE/4
ld32              $w_qGradBase, $mvertex_base, W_QGRAD_BASE/4
ld32              $w_sBase, $mvertex_base, W_S_BASE/4
ld32              $w_numZ, $mvertex_base, W_NUM_Z/4

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
mul               $w_wkrInfoOffset, $w_id, Sizeof_MIGradWWorkerEntry
add               $w_metaInfo, $w_metaInfo, $w_wkrInfoOffset


ldz16             $w_metaInfoOffsetOutputEntry, $w_metaInfo, MIGradWorkerEntry_metaInfoOffsetOutputEntry/2
ldz16             $w_offsetY, $w_metaInfo, MIGradWorkerEntry_metaInfoOffsetToOffsetsYInSFirst/2
ldz16             $w_totalNumY, $w_metaInfo, MIGradWorkerEntry_totalNumY/2

// !!! Assumption here that sparse offset is the first entry in the table
ldz16step         $w_sparseOffset, $mzero, $w_metaInfo+=, $w_metaInfoOffsetOutputEntry
// dummy load to move to gradient base for this worker
ld32step          $mzero, $mzero, $w_rGradBase+=, $w_sparseOffset

ldz16step         $w_offXInQ, $mzero, $w_metaInfo+=, 1 
ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1 

// move meta info pointer by doing a dummy load
ldz16step         $mzero, $mzero, $w_metaInfo+=, $w_offsetY
sub               $w_numY, $w_numY, $w_offsetY
min               $w_numY, $w_numY, $w_totalNumY

{
  cmpeq             $w_zEq4, $w_numZ, 4
  setzi             $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  brnz              $w_zEq4, LZEq4Main
  uput              $FP_CLR, $fp_clr_reg 
}
cmpeq             $w_zEq2, $w_numZ, 2
brnz              $w_zEq2, LZEq2Main

shr               $w_numZDiv2, $w_numZ, 1
and               $w_numRem, $w_numZ, 0x1 
shl               $w_offXInQ, $w_offXInQ, 2

LMainLoopX:
add               $w_numY, $w_numY, -1
ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1

LMainLoopY:

// Modify pointers to use pace instructions: stride increments are 1
{
  add               $w_qGradBaseLoop, $w_qGradBase, $w_offXInQ
  mov               $a6:7, $azeros
}
{
  add               $w_sBaseLoop, $w_sBase, $w_offYInS
  mov               $a5, $azero
}
ld32              $a4,  $mzero, $w_rGradBase, 0
{
  ld2x64pace        $a0:1, $a2:3, $w_packedPtr+=, $mzero, 0
  fnop
}
{
  rpt $w_numZDiv2, 0
  f32v4acc            $a4:7
}
  {
    ld2x64pace        $a0:1, $a2:3, $w_packedPtr+=, $mzero, 0
    f32v2mac          $a0:1, $a2:3
  }
brz               $w_numRem, LFinalAdd
f32mac            $a0, $a2

LFinalAdd:
{
  add               $w_totalNumY, $w_totalNumY, -1
  f32v2gina         $a0:1, $azeros, 0
}
{ 
  ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
  f32add            $a0, $a0, $a1
}

st32step          $a0, $mzero, $w_rGradBase+=, 1
brnzdec           $w_numY, LMainLoopY

ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1
min               $w_numY, $w_numY, $w_totalNumY
shl               $w_offXInQ, $w_offYInS, 2
brnz              $w_numY, LMainLoopX
LEndWorker:
exitz             $mzero

// -----------------------------------------------------------------------------
// processes z=4
LZEq4Main:            
ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
shl               $w_offXInQ, $w_offXInQ, 2
sub               $w_totalNumY, $w_totalNumY, $w_numY
add               $w_numY, $w_numY, -1
ld64              $a0:1, $w_offXInQ, $w_qGradBase, 0
ld64              $a2:3, $w_offXInQ, $w_qGradBase, 1
ld64              $a4:5, $w_offYInS, $w_sBase, 0
{
  ld64              $a4:5, $w_offYInS, $w_sBase, 1
  f32v2mac          $a0:1, $a4:5
}
{
  rpt               $w_numY, 4
  f32v2mac          $a2:3, $a4:5
}
  {
    ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
    f32v2gina         $a6:7, $azeros, 0
  }
  {
    ld64              $a4:5, $w_offYInS, $w_sBase, 0
    f32add            $a6, $a6, $a7
  }
  {
    ld32              $a7, $mzero, $w_rGradBase, 0
    f32v2mac          $a0:1, $a4:5
  } 
  {
    ld64              $a4:5, $w_offYInS, $w_sBase, 1
    f32add            $a6, $a6, $a7
  }
  {
    st32step          $a6, $mzero, $w_rGradBase+=, 1
    f32v2mac          $a2:3, $a4:5
  }

{
  ldz16step         $w_offXInQ, $mzero, $w_metaInfo+=, 1
  f32v2gina         $a6:7, $azeros, 0
}
{
  ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1
  f32add            $a6, $a6, $a7
}
ld32              $a7, $mzero, $w_rGradBase, 0
{
  min               $w_numY, $w_totalNumY, $w_numY
  f32add            $a6, $a6, $a7
}
st32step          $a6, $mzero, $w_rGradBase+=, 1
brnz              $w_numY, LZEq4Main
exitz             $mzero

// -----------------------------------------------------------------------------
// processes z=2

LZEq2Main:
ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
sub               $w_totalNumY, $w_totalNumY, $w_numY
add               $w_numY, $w_numY, -1
shl               $w_offXInQ, $w_offXInQ, 2
ld64              $a0:1, $w_offXInQ, $w_qGradBase, 0
ld64              $a2:3, $w_offYInS, $w_sBase, 0
{
  rpt               $w_numY, 3
  f32v2mul          $a4:5, $a0:1, $a2:3
}
  {
    ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
    f32add            $a6, $a4, $a5
  }
  {
    ld32              $a7, $mzero, $w_rGradBase, 0
    fnop
  }
  {
    ld64              $a2:3, $w_offYInS, $w_sBase, 0
    f32add            $a6, $a6, $a7
  }
  {
    st32step          $a6, $mzero, $w_rGradBase+=, 1
    f32v2mul          $a4:5, $a0:1, $a2:3
  }
{
  ldz16step         $w_offXInQ, $mzero, $w_metaInfo+=, 1
  f32add            $a6, $a4, $a5
}
ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1
ld32              $a7, $mzero, $w_rGradBase, 0
{
  min               $w_numY, $w_totalNumY, $w_numY
  f32add            $a6, $a6, $a7
}
st32step          $a6, $mzero, $w_rGradBase+=, 1
brnz              $w_numY, LZEq2Main
exitz             $mzero

.size elemwiseSparseDenseMultiplyGradFF, . - elemwiseSparseDenseMultiplyGradFF

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
