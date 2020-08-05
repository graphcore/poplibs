// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Sparse-Dense Block matrix multiplication for Fwd asm codelets

#ifndef _BlockSparseDenseMatMul_h_
#define _BlockSparseDenseMatMul_h_

#include "BlockSparseMatMulStructs.h.S"
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
#define SUP_VBASE_WORKLIST           20   // one pointer
#define SUP_VBASE_ZSTRIDEINQ         24   // ushort
#define SUP_VBASE_ZSTRIDEINS         26   // ushort

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////
#define W_S_BASE                        0
#define W_Q_BASE                        4
#define W_R_BASE                        8
#define W_WORKLIST                      12
#define W_ZSTRIDEINQ                    16
#define W_ZSTRIDEINS                    20
#define W_XOFFSET                       24
#define W_YOFFSET                       28
#define W_NUM_BUCKETS                   32
#define W_RBASELIST                     36
#define STACK_SIZE                      (W_RBASELIST + 4)


.macro BLOCK_SPARSE_MATMUL CNAME, TYPE, POSTFIX

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
#define s_subgroupSparseElems        m9
#define s_match                      m7
#define s_zeroWkrFunction            m10
#define s_wTemp                      m7
#define s_wkrFunction                m2
#define s_workList                   m10
#define s_zStrideInQ                 m7
#define s_zStrideInS                 m9
#define s_clrCWEI                    m9
#define s_xOffsetInQ                 m3
#define s_yOffsetInS                 m3
#define s_numXm1                     m10
#define s_numYm1                     m5

// supervisor base is $m0 - passed to this function
DEF_STACK_USAGE  (STACK_SIZE + 8) \CNAME\()

.section .text.\CNAME\()
.align 4
.globl \CNAME\()
.type \CNAME\(), @function
\CNAME\():
.supervisor
setzi                  $s_clrCWEI, (1 << 2)
ld32                   $s_metaInfoList, $s_vertexBase, SUP_VBASE_META_LIST_PTR/4

add                    $sp, $sp, -STACK_SIZE-8
// &S[0] is common to all the metaInformation tables
ld32                   $s_sBase, $s_vertexBase, SUP_VBASE_S_BASE/4

// &R[0] is common to all the metaInformation tables
ld32                   $s_rBaseList, $s_vertexBase, SUP_VBASE_R_BASE_LIST/4

// &Q[0] is common for the all the metaInfo tables
ld32                   $s_qBase, $s_vertexBase, SUP_VBASE_Q_BASE/4
shr                    $s_numBuckets, $s_metaInfoList, SHORT_SPAN_ADDRESS_BITS

st32                   $m9, $sp, STACK_SIZE/4
st32                   $m10, $sp, STACK_SIZE/4 + 1
ld32                   $s_workList, $s_vertexBase, SUP_VBASE_WORKLIST/4
ldz16                  $s_zStrideInQ, $s_vertexBase, SUP_VBASE_ZSTRIDEINQ/2
ldz16                  $s_zStrideInS, $s_vertexBase, SUP_VBASE_ZSTRIDEINS/2

// Push into worker vertex state because same pointer is used for buckets
st32                   $s_sBase, $sp, W_S_BASE/4
st32                   $s_qBase, $sp, W_Q_BASE/4
put                    $CR, $s_clrCWEI
st32                   $s_workList, $sp, W_WORKLIST/4
setzi                  $s_zeroWkrFunction, zeroDenseOutFloat
st32                   $s_zStrideInQ, $sp, W_ZSTRIDEINQ/4
st32                   $s_zStrideInS, $sp, W_ZSTRIDEINS/4

// why is short span not encoded with address as msbs??
shl                    $s_metaInfoList, $s_metaInfoList, (32 - SHORT_SPAN_ADDRESS_BITS)

// extract number of buckets
add                    $s_numBuckets, $s_numBuckets, -1

// load rBase before the call to zero to use the bubble created by it
ld32step               $s_rBase, $mzero, $s_rBaseList+=, 1

runall                 $s_zeroWkrFunction, $s_vertexBase, 0

shr                    $s_metaInfoList, $s_metaInfoList, (32 - SHORT_SPAN_ADDRESS_BITS)

LmetaInfoListLoop_\TYPE\():  
  st32                   $s_rBaseList, $sp, W_RBASELIST/4
  setzi                  $s_wkrFunction, blockSparseDenseMultiply_\POSTFIX\()   
  ld32step               $s_metaInfo, $mzero, $s_metaInfoList+=, 1
  st32                   $s_numBuckets, $sp, W_NUM_BUCKETS/4
LsubgroupLoop_\TYPE\():  
  // This is the subgroup ID the PN has to process
  ldz16                  $s_pnSubgroupId, $s_vertexBase, SUP_VBASE_PN_SUBGROUP_ID/2
  ldz16                  $s_subgroupId, $s_metaInfo, MetaInfoSubGroupEntry_id/2

  // s_metaInfo is at exactly where the numWorkers is so that it can be extracted 
  // by the worker (must be last field)
  ldz16                  $s_offsetToNextSubgroup, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  
  // The pointer to sparse R Is offset
  ldz16                  $s_subgroupSparseElems, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries/2
  
  // If subgroup is 0 there is nothing to do
  brz                    $s_subgroupId, LendMetaInfoList_\TYPE\()

  // Check if any work to be done by the PN
  cmpeq                  $s_match, $s_subgroupId, $s_pnSubgroupId
  brz                    $s_match, LnextSubgroup_\TYPE\()
  ldz16                  $s_numXm1, $mzero, $s_metaInfo, MetaInfoSubGroupEntry_numXm1/2
  add                    $s_wTemp, $s_metaInfo, sizeof_MetaInfoSubGroupEntry
  // We let CCCSLOAD to increment between blocks rather then tracking it using
  // a general register
  put                    $CCCSLOAD, $s_rBase
  sync                   TEXCH_SYNCZONE_LOCAL

  // load entries from output
Ls_XLoop_\TYPE\():
    ldz16step              $s_xOffsetInQ, $mzero, $s_wTemp+=, 1
    st32                   $s_xOffsetInQ, $sp, W_XOFFSET/4
    ldz16step              $s_numYm1, $mzero, $s_wTemp+=, 1

Ls_YLoop_\TYPE\():
      ldz16step              $s_yOffsetInS, $mzero, $s_wTemp+=, 1
      ld64putcs              0x0 // CCCS_CWEI_0_0
      ld64putcs              0x1 // CCCS_CWEI_0_1
      ld64putcs              0x4 // CCCS_CWEI_1_0
      ld64putcs              0x5 // CCCS_CWEI_1_1
      ld64putcs              0x8 // CCCS_CWEI_2_0
      st32                   $s_yOffsetInS, $sp, W_YOFFSET/4
      ld64putcs              0x9 // CCCS_CWEI_2_1
      ld64putcs              0xC // CCCS_CWEI_3_0
      ld64putcs              0xD // CCCS_CWEI_3_1
      runall                 $s_wkrFunction, $sp, 0

      setzi                  $s_wkrFunction, blockSparseDenseMultiply_\POSTFIX\()_retained   
      sync                   TEXCH_SYNCZONE_LOCAL
      brnzdec                $s_numYm1, Ls_YLoop_\TYPE\()
    brnzdec                $s_numXm1, Ls_XLoop_\TYPE\()
  
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
  ld32                   $s_rBaseList, $sp, W_RBASELIST/4
  ld32                   $s_numBuckets, $sp, W_NUM_BUCKETS/4
  ld32step               $s_rBase, $mzero, $s_rBaseList+=, 1
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
#endif // #define _BlockSparseDenseMatMul_h_
// =============================================================================
