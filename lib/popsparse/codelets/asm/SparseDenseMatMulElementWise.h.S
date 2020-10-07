// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Sparse-Dense matrix multiplication for Fwd asm codelets

#ifndef _SparseDenseMatMulElementWise_h_
#define _SparseDenseMatMulElementWise_h_

#include "SparseDenseMatMulStructs.h.S"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define LOG2_SIZEOF_OUT_ATOM 2
// Short span
#define SHORT_SPAN_ADDRESS_BITS      20

// =============================================================================

//// Supervisor vertex state
#define SUP_VBASE_R_BASE_LIST        0    // one pointer
#define SUP_VBASE_META_LIST_PTR      4    // short span (20 lsbs address)
#define SUP_VBASE_S_BASE             8    // one pointer
#define SUP_VBASE_Q_BASE             12   // one pointer
#define SUP_VBASE_PN_SUBGROUP_ID     16   // ushort
#define SUP_VBASE_ZERO_INFO          18   // ushort

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////
#define W_S_BASE                        0
#define W_Q_BASE                        4
#define W_R_BASE                        8
#define W_METAINFO                      12
#define W_NUMZ                          16
#define STACK_SIZE                      (W_NUMZ + 8)


.macro ELEM_SPARSE_MATMUL CNAME, TYPE

// Registers allocation
#define s_vertexBase                 m0
#define s_sBase                      m1
#define s_rBaseList                  m2
#define s_pnSubgroupId               m3
#define s_qBase                      m4
#define s_numBuckets                 m5
#define s_metaInfoList               m6
#define s_metaInfo                   m1
#define s_rBase                      m4
#define s_subgroupId                 m7
#define s_offsetToNextSubgroup       m8
#define s_numZ                       m9
#define s_subgroupSparseElems        m9
#define s_match                      m7
#define s_zeroWkrFunction            m7
#define s_wTemp                      m7
#define s_wkrFunction                m10

// supervisor base is $m0 - passed to this function
DEF_STACK_USAGE  (STACK_SIZE + 8) \CNAME\()

.section .text.\CNAME\()
.align 4
.globl \CNAME\()
.type \CNAME\(), @function
\CNAME\():
.supervisor


ld32                   $s_metaInfoList, $s_vertexBase, SUP_VBASE_META_LIST_PTR/4

add                    $sp, $sp, -STACK_SIZE-8
// &S[0] is common to all the metaInformation tables
ld32                   $s_sBase, $s_vertexBase, SUP_VBASE_S_BASE/4

// &R[0] is common to all the metaInformation tables
ld32                   $s_rBaseList, $s_vertexBase, SUP_VBASE_R_BASE_LIST/4

// This is the subgroup ID the PN has to process
ldz16                  $s_pnSubgroupId, $s_vertexBase, SUP_VBASE_PN_SUBGROUP_ID/2

// &Q[0] is common for the all the metInfo tables
ld32                   $s_qBase, $s_vertexBase, SUP_VBASE_Q_BASE/4


shr                    $s_numBuckets, $s_metaInfoList, SHORT_SPAN_ADDRESS_BITS

setzi                  $s_zeroWkrFunction, zeroDenseOutFloat
st32                   $m9, $sp, STACK_SIZE/4
st32                   $m10, $sp, STACK_SIZE/4 + 1

// Push into worker vertex state because same pointer is used for buckets
st32                   $s_sBase, $sp, W_S_BASE/4
st32                   $s_qBase, $sp, W_Q_BASE/4

// why is short span not encoded with address as msbs??
shl                    $s_metaInfoList, $s_metaInfoList, (32 - SHORT_SPAN_ADDRESS_BITS)

// extract number of buckets
add                    $s_numBuckets, $s_numBuckets, -1
runall                 $s_zeroWkrFunction, $s_vertexBase, 0

setzi                  $s_wkrFunction, elemwiseSparseDenseMultiply
shr                    $s_metaInfoList, $s_metaInfoList, (32 - SHORT_SPAN_ADDRESS_BITS)

LmetaInfoListLoop_\TYPE\():  
  ld32step               $s_rBase, $mzero, $s_rBaseList+=, 1
  ld32step               $s_metaInfo, $mzero, $s_metaInfoList+=, 1
LsubgroupLoop_\TYPE\():  
  ldz16                  $s_subgroupId, $s_metaInfo, MetaInfoSubGroupEntry_id/2
  ldz16                  $s_numZ, $s_metaInfo, MetaInfoSubGroupEntry_numZ/2

  // s_metaInfo is at exactly where the numWorkers is so that it can be extracted 
  // by the worker (must be last field)
  ldz16                  $s_offsetToNextSubgroup, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  
  // Need to sync before writing state which worker may read 
  sync                   TEXCH_SYNCZONE_LOCAL
  // base address for R for this PN
  st32                   $s_rBase, $sp, W_R_BASE/4
  // numZ for this tile
  st32                   $s_numZ, $sp, W_NUMZ/4

  // The pointer to sparse R Is offset
  ldz16                  $s_subgroupSparseElems, $s_metaInfo, MetaInfoSubGroupEntry_sparseElementCount/2
  
  // If subgroup is 0 there is nothing to do
  brz                    $s_subgroupId, LendMetaInfoList_\TYPE\()

  // Check if any work to be done by the PN
  cmpeq                  $s_match, $s_subgroupId, $s_pnSubgroupId
  brz                    $s_match, LnextSubgroup_\TYPE\()
  add                    $s_wTemp, $s_metaInfo, sizeof_MetaInfoSubGroupEntry - sizeof_metaInfoEntry
  
  // pointer to worker meta info
  st32                   $s_wTemp, $sp, W_METAINFO/4
  runall                 $s_wkrFunction, $sp, 0
  
LnextSubgroup_\TYPE\():
  // dummy load to move pointer to next subgroup
.ifc \TYPE, float
  ld32step               $mzero, $mzero, $s_rBase+=, $s_subgroupSparseElems
.else
  ldz16step              $mzero, $mzero, $s_rBase+=, $s_subgroupSparseElems
.endif
  ldz16step              $mzero, $mzero, $s_metaInfo+=, $s_offsetToNextSubgroup
  bri                    LsubgroupLoop_\TYPE\()
  
LendMetaInfoList_\TYPE\(): 
  brnzdec                $s_numBuckets, LmetaInfoListLoop_\TYPE\()
LendMetaInfoLoop_\TYPE\():
ld32                   $m9, $sp, STACK_SIZE/4
ld32                   $m10, $sp, STACK_SIZE/4 + 1
add                    $sp, $sp, STACK_SIZE + 8
sync                   TEXCH_SYNCZONE_LOCAL
br                     $lr

.size \CNAME\(), . - \CNAME\()
.endm

// =============================================================================
#endif // #define _SparseDenseMatMulElementWise_h_
// =============================================================================
