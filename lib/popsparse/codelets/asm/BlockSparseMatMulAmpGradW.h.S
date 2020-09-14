// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Common function that is the supervisor function that performs sparse matrix 
// multiplication R = Q * S' Where Q and S are dense matrices and R is a
// sparse matrix. Supported block sizes are 4x4, 8x8 and 16x16 and use AMP.
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

.extern zeroDenseOutput64GradW

// =============================================================================

// =============================================================================

// Registers allocation
#define s_vertexBase                 m0
#define s_metaInfo                   m1
#define s_pnSubgroupId               m2
#define s_rGradBase                  m3
#define s_subgroupId                 m4
#define s_numZ                       m5
#define s_offsetToNextSubgroup       m6
#define s_subgroupSparseElems        m7
#define s_match                      m9
#define s_temp                       m7
#define s_zeroWkrFunction            m8
#define s_wkrFunction                m10
#define s_clrCWEI                    m2


#define s_numX                       m4
#define s_offXInQ                    m2
#define s_offYinS                    m2
#define s_numY                       m6
#define s_qGradThisX                 m6
#define s_sThisY                     m7
#define s_offZ                       m8
#define s_numWorkers                 m8
#define s_sLoadCwei                  m9
#define s_yTempOffset                m9
#define s_stride                     m6


// macro to load a block of
//   - 4x16 half or 2x8 float coefficients 
//   - 8x16 half coefficients, or 8x8 float coefficients
// in CWEI
.macro LOAD_BLOCK_8x16HALF_OR_8x8FLOAT_IN_CWEI TYPE, FULL
  ld128putcs             0
  ld128putcs             2
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             4
  ld128putcs             6
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             8
  ld128putcs             10
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             12
  ld128putcs             14
.ifc \FULL, 1
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             16
  ld128putcs             18
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             20
  ld128putcs             22
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             24
  ld128putcs             26
  LOAD_POSTINC           \TYPE $mzero $mzero $s_sLoadCwei $s_numZ
  put                    $CCCSLOAD, $s_sLoadCwei  
  ld128putcs             28
  ld128putcs             30
.endif
.endm

// macro to load a block of 16x16 coefficients in CWEI for MK1
.macro LOAD_BLOCK16x16_HALF_IN_CWEI PTYPE, WKR_FUNC
  LOAD_BLOCK_8x16HALF_OR_8x8FLOAT_IN_CWEI half 1
  LOAD_POSTINC           half $mzero $mzero $s_sLoadCwei $s_numZ
  runall                 $s_wkrFunction, $sp, 0
  setzi                  $s_wkrFunction, \WKR_FUNC\()_retained_zOff
  put                    $CCCSLOAD, $s_sLoadCwei  

  // increment pointer to move to next half of rGrad
  INCR_PTR               \PTYPE $s_rGradBase 8
  sync                   TEXCH_SYNCZONE_LOCAL
  st32                   $s_rGradBase, $sp, W_AMP_RGRAD_BASE_BLOCK/4 
  DECR_PTR               \PTYPE $s_rGradBase 8
  LOAD_BLOCK_8x16HALF_OR_8x8FLOAT_IN_CWEI half 1
.endm

// Macro to update S pointer after a each iteration of Z
.macro UPDATE_S_PTR INTYPE
.ifc \INTYPE, float
  DECR_PTR               float $s_sThisY 8
.else
  DECR_PTR               half $s_sThisY 16
.endif
.endm

// macro to create supervisor code for AMP
.macro BLOCK_SPARSE_MATMUL_GRADW_AMP_SUP CNAME, INTYPE, PTYPE, BLOCKSIZE, GRADW_FN

// supervisor base is $m0 - passed to this function
DEF_STACK_USAGE  (STACK_SIZE_AMP + 8) \CNAME\()

.section .text.\CNAME\()
.align 4
.globl \CNAME\()
.type \CNAME\(), @function
\CNAME\():
.supervisor


setzi                  $s_clrCWEI, (1 << 2)
ldz16                  $s_numZ, $s_vertexBase, SUP_VBASE_AMP_NUM_Z/2
setzi                  $s_zeroWkrFunction, zeroDenseOutput64GradW
ld32                   $s_rGradBase, $s_vertexBase, SUP_VBASE_RGRAD_BASE/4
// allocate stack
add                    $sp, $sp, -STACK_SIZE_AMP-8

// This is the subgroup ID the PN has to process
ld32                   $s_pnSubgroupId, $s_vertexBase, SUP_VBASE_AMP_PN_SUBGROUP_ID/4
// number of blocks of 16/8 (half/float) batch elements to process
.ifc \INTYPE, float     
shr                    $s_temp, $s_numZ, 3
shr                    $s_stride, $s_numZ, 1
.else
shr                    $s_temp, $s_numZ, 4
shr                    $s_stride, $s_numZ, 2
.endif

runall                 $s_zeroWkrFunction, $s_vertexBase, 0
add                    $s_stride, $s_stride, -3
// Decrement by 1 to use in brnzdec
add                    $s_temp, $s_temp, -1
ldz16                  $s_pnSubgroupId, $mzero, $s_pnSubgroupId, 0
st32                   $m9, $sp, STACK_SIZE_AMP/4
st32                   $m10, $sp, STACK_SIZE_AMP/4 + 1
st32                   $s_numZ, $sp, W_AMP_NUM_Z/4
.ifc \INTYPE, half     
shl                    $s_stride, $s_stride, 10
.endif
setzi                  $s_wkrFunction, \GRADW_FN
st32                   $s_temp, $sp, W_AMP_NUM_X16_BATCHES/4
st32                   $s_pnSubgroupId, $sp, W_AMP_PN_SUBGROUP_ID/4
put                    $CR, $s_clrCWEI
ld32                   $s_metaInfo, $s_vertexBase, SUP_VBASE_AMP_META_BASE/4
st32                   $s_stride, $sp, W_AMP_STRIDES/4

LsubgroupLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\(): 
  ldz16                  $s_subgroupId, $s_metaInfo, MetaInfoSubGroupEntry_id/2
  st32                   $s_metaInfo, $sp, W_AMP_METAINFO/4
  st32                   $s_rGradBase, $sp, W_AMP_RGRAD_BASE/4
  ld32                   $s_sThisY, $s_vertexBase, SUP_VBASE_AMP_S_BASE/4
  ld32                   $s_qGradThisX, $s_vertexBase, SUP_VBASE_AMP_QGRAD_BASE/4
  ldz16                  $s_numWorkers, $s_metaInfo, MetaInfoSubGroupEntry_numGradWWorkers/2

  // If subgroup is 0 there is nothing to do
  brz                    $s_subgroupId, LendMetaInfoLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\()
  // Check if any work to be done by the PN
  cmpeq                  $s_match, $s_subgroupId, $s_pnSubgroupId
  brz                    $s_match, LnextSubgroup_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\()

  ldz16                  $s_numX, $s_metaInfo, MetaInfoSubGroupEntry_numXm1/2
  nop
  mul                    $s_numWorkers, $s_numWorkers, sizeof_MetaInfoGradWWorkerEntry
  add                    $s_metaInfo, $s_metaInfo, $s_numWorkers
  add                    $s_metaInfo, $s_metaInfo, sizeof_MetaInfoSubGroupEntry

  ldz16step              $s_offXInQ, $mzero, $s_metaInfo+=, 1
  mul                    $s_offXInQ, $s_offXInQ, $s_numZ


LXLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\():
    LOAD_POSTINC           \INTYPE $mzero $mzero $s_qGradThisX $s_offXInQ
    sync                   TEXCH_SYNCZONE_LOCAL
    st32                   $s_qGradThisX, $sp, W_QGRAD_BASE/4
    ldz16step              $s_numY, $mzero, $s_metaInfo+=, 1 

LYLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\():
      // The number of times to loop in blocks of 16/8 batch elements
      ld32                   $s_offZ, $sp, W_AMP_NUM_X16_BATCHES/4
.ifc \INTYPE, half
      sub                    $s_yTempOffset, $s_numZ, 16 
.else
      sub                    $s_yTempOffset, $s_numZ, 8
.endif
      ldz16step              $s_offYinS, $mzero, $s_metaInfo+=, 1 
      mul                    $s_offYinS, $s_offYinS, $s_numZ
      add                    $s_offYinS, $s_offYinS, $s_yTempOffset
      LOAD_POSTINC           \INTYPE $mzero $mzero $s_sThisY $s_offYinS

LZLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\():
        // Need to sync because workers may  active and we touch common vertex state
        sync                   TEXCH_SYNCZONE_LOCAL
        st32                   $s_rGradBase, $sp, W_AMP_RGRAD_BASE_BLOCK/4 
        st32                   $s_offZ, $sp, W_AMP_OFFSET_Z/4
        mov                    $s_sLoadCwei, $s_sThisY
        put                    $CCCSLOAD, $s_sThisY
        // load S into $CWEI
.ifc \BLOCKSIZE, 16
.ifc \INTYPE, half
.ifc \PTYPE, half
#if __IPU_ARCH_VERSION__ < 2 
        LOAD_BLOCK16x16_HALF_IN_CWEI \PTYPE \GRADW_FN
#else
        ld128putcs             0
        ld128putcs             2
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             4
        ld128putcs             6
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             32
        ld128putcs             34
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             36
        ld128putcs             38
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             8
        ld128putcs             10
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             12
        ld128putcs             14
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             40
        ld128putcs             42
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             44
        ld128putcs             46
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             16
        ld128putcs             18
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             20
        ld128putcs             22
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             48
        ld128putcs             50
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             52
        ld128putcs             54
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             24
        ld128putcs             26
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             28
        ld128putcs             30
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             56
        ld128putcs             58
        LOAD_POSTINC           \INTYPE $mzero $mzero $s_sLoadCwei $s_numZ
        put                    $CCCSLOAD, $s_sLoadCwei  
        ld128putcs             60
        ld128putcs             62
#endif // __IPU_ARCH_VERSION__ < 2
.endif // .ifc \PTYPE, half

.ifc \PTYPE, float
        LOAD_BLOCK16x16_HALF_IN_CWEI \PTYPE \GRADW_FN
.endif // .ifc \PTYPE, float
.endif // .ifc \INTYPE, half
.endif // .ifc \BLOCKSIZE, 16

.ifc \BLOCKSIZE, 8
        LOAD_BLOCK_8x16HALF_OR_8x8FLOAT_IN_CWEI \INTYPE 1
.endif // .ifc \BLOCKSIZE, 8

.ifc \BLOCKSIZE, 4
        LOAD_BLOCK_8x16HALF_OR_8x8FLOAT_IN_CWEI \INTYPE 0
.endif // .ifc \BLOCKSIZE, 4
        UPDATE_S_PTR           \INTYPE
        runall                 $s_wkrFunction, $sp, 0
        setzi                  $s_wkrFunction, \GRADW_FN\()_retained_zOff 
        brnzdec                $s_offZ, LZLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\()
        
      ld32                   $s_sThisY, $s_vertexBase, SUP_VBASE_AMP_S_BASE/4
      setzi                  $s_wkrFunction, \GRADW_FN\()_retained 
      INCR_PTR               \PTYPE $s_rGradBase \BLOCKSIZE*\BLOCKSIZE
      brnzdec                $s_numY, LYLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\()
    
    ld32                   $s_qGradThisX, $s_vertexBase, SUP_VBASE_AMP_QGRAD_BASE/4
    ldz16step              $s_offXInQ, $mzero, $s_metaInfo+=, 1
    mul                    $s_offXInQ, $s_offXInQ, $s_numZ
    brnzdec                $s_numX, LXLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\()
  
LnextSubgroup_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\():
  ld32                   $s_rGradBase, $sp, W_AMP_RGRAD_BASE/4
  // load pointers to be updated for the next subgroup
  ld32                   $s_metaInfo, $sp, W_AMP_METAINFO/4
  ldz16                  $s_offsetToNextSubgroup, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  // The pointer to sparse R is offset by this
  ldz16                  $s_subgroupSparseElems, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries/2
  ld32                   $s_pnSubgroupId, $sp, W_AMP_PN_SUBGROUP_ID/4

  // dummy load to move pointer to next subgroup
  ldz16step              $mzero, $mzero, $s_metaInfo+=, $s_offsetToNextSubgroup
  LOAD_POSTINC           \PTYPE $mzero $mzero $s_rGradBase $s_subgroupSparseElems
  bri                    LsubgroupLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\()
  
LendMetaInfoLoop_\INTYPE\()_\PTYPE\()_\BLOCKSIZE\():
ld32                   $m9, $sp, STACK_SIZE_AMP/4
ld32                   $m10, $sp, STACK_SIZE_AMP/4 + 1
add                    $sp, $sp, STACK_SIZE_AMP + 8
sync                   TEXCH_SYNCZONE_LOCAL
br                     $lr

.size \CNAME\(), . - \CNAME\()
.endm

// =============================================================================

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
