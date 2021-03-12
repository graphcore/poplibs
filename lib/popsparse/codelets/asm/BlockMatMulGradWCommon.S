// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Common function that is the supervisor function that performs sparse matrix 
// multiplication R = Q * S' Where Q and S are dense matrices and R is a
// sparse matrix
// This serves the purpose of computing the entries of the
// sparse gradients with respect to weights.

#ifdef __IPU__
#include "BlockSparseMatMulGradW.h.S"
#include "BlockSparseMatMulStructs.h.S"
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================

DEF_STACK_USAGE 0 zeroDenseOutput64GradW
.section ".text.zeroDenseOutput64GradW", FUNCTION_IS_WORKER
.type zeroDenseOutput64GradW, @function
.globl zeroDenseOutput64GradW
.align 8

#define wkr_id_zv                       m0
#define zero_info_zv                    m1
#define zero_info_div_12_zv             m2
#define outchan_ptr_zv                  m3
zeroDenseOutput64GradW:
get           $wkr_id_zv, $WSR
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
ldz16         $zero_info_zv, $mvertex_base, SUP_VBASE_ZERO_INFO/2

// we could get zero information as this vertex could be called multiple times
// but zero infor field must be zero only in the first call
brz           $zero_info_zv, Loop_end_zero_64

// For n with 0 <= n <= 65533 this does a division by 6 with the remainder
// split amongst workers.
add           $zero_info_div_12_zv, $zero_info_zv, 6
sub           $zero_info_div_12_zv, $zero_info_div_12_zv, $wkr_id_zv
mul           $zero_info_div_12_zv, $zero_info_div_12_zv, 21845
shr           $zero_info_div_12_zv, $zero_info_div_12_zv, 17

ld32          $outchan_ptr_zv, $mvertex_base, SUP_VBASE_RGRAD_BASE/4
ld64step      $azeros, $mzero, $outchan_ptr_zv+=, $wkr_id_zv

{
  rpt           $zero_info_div_12_zv, (Loop_end_zero_64 - Loop_start_zero_64)/8 - 1
  fnop
}

Loop_start_zero_64:
  {
    st64step      $azeros, $mzero, $outchan_ptr_zv+=, 6
    fnop
  }
Loop_end_zero_64:
exitz         $mzero

.size zeroDenseOutput64GradW, . - zeroDenseOutput64GradW

// =============================================================================

// =============================================================================
// Supervisor codelet which launches the zeroing of the output RGrad matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. Meta information for a subgroup Id is expected
// to be found only once.

// Registers allocation
#define s_vertexBase                 m0
#define s_sBase                      m1
#define s_metaInfo                   m9
#define s_pnSubgroupId               m3
#define s_qGradBase                  m4
#define s_rGradBase                  m5
#define s_subgroupId                 m7
#define s_numZ                       m4
#define s_offsetToNextSubgroup       m1
#define s_subgroupSparseElems        m4
#define s_match                      m7
#define s_wkrFunction                m6 // is used as a parameter to this fn
#define s_zeroWkrFunction            m8
#define s_temp                       m7
#define s_zStrideInQ                 m7
#define s_zStrideInS                 m1
#define s_rGradLd32Flag              m2 // is used as a parameter to this fn.
                                        // set to 1 for float rGrad else 0.

#define COMMON_FN BlockGradWCommonSupervisor

// supervisor base is $m0 - passed to this function
// s_wkrFunction - the worker function to call
// s_rGradLd32Flag - set to 1 for float rGrad, else 0
DEF_STACK_USAGE  (STACK_SIZE + 4) COMMON_FN

.section .text.COMMON_FN
.align 4
.type COMMON_FN, @function
COMMON_FN:
.supervisor

// allocate stack
add                    $sp, $sp, -STACK_SIZE-4
ldz16                  $s_numZ, $s_vertexBase, SUP_VBASE_NUM_Z/2

// This is the subgroup ID the PN has to process
ld32                   $s_pnSubgroupId, $s_vertexBase, SUP_VBASE_PN_SUBGROUP_ID/4
setzi                  $s_zeroWkrFunction, zeroDenseOutput64GradW

// &S[0] is common to all the metaInformation tables
ld32                   $s_sBase, $s_vertexBase, SUP_VBASE_S_BASE/4

// &R[0] is common to all the metaInformation tables
ld32                   $s_rGradBase, $s_vertexBase, SUP_VBASE_RGRAD_BASE/4

st32                   $m9, $sp, STACK_SIZE/4
ld32                   $s_metaInfo, $s_vertexBase, SUP_VBASE_META_BASE/4
st32                   $s_numZ, $sp, W_NUM_Z/4

ldz16                  $s_zStrideInQ, $s_vertexBase, SUP_VBASE_ZSTRIDE_IN_Q/2
st32                   $s_sBase, $sp, W_S_BASE/4
ldz16                  $s_zStrideInS, $s_vertexBase, SUP_VBASE_ZSTRIDE_IN_S/2

// &Q[0] is common for the all the metaInfo tables
ld32                   $s_qGradBase, $s_vertexBase, SUP_VBASE_QGRAD_BASE/4
runall                 $s_zeroWkrFunction, $s_vertexBase, 0
st32                   $s_qGradBase, $sp, W_QGRAD_BASE/4
setzi                  $s_subgroupSparseElems, 0
st32                   $s_zStrideInQ, $sp, W_ZSTRIDE_IN_Q/4
st32                   $s_zStrideInS, $sp, W_ZSTRIDE_IN_S/4
ldz16                  $s_pnSubgroupId, $mzero, $s_pnSubgroupId, 0
st32                   $s_rGradBase, $sp, W_RGRAD_BASE/4

LsubgroupLoopLd32:
  // At entry, this shouldn't change the pointer as s_subgroupSparseElems = 0
  ld32step               $mzero, $mzero, $s_rGradBase+=, $s_subgroupSparseElems  
LsubgroupLoop:  
  ldz16                  $s_subgroupId, $s_metaInfo, MetaInfoSubGroupEntry_id/2

  ldz16                  $s_offsetToNextSubgroup, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  
  // The pointer to sparse R is offset by this
  ldz16                  $s_subgroupSparseElems, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries/2
  
  // If subgroup is 0 there is nothing to do
  brz                    $s_subgroupId, LendMetaInfoLoop

  // Check if any work to be done by the PN
  cmpeq                  $s_match, $s_subgroupId, $s_pnSubgroupId
  brz                    $s_match, LnextSubgroup

  // Last entry in subgroup is always the number of entries for GradW.
  // We could remove it from the subgroup entry and place it immediately
  // after it
  add                    $s_temp, $s_metaInfo, MetaInfoSubGroupEntry_numGradWWorkers

  // Need to sync because workers may  active and we touch common vertex state
  sync                   TEXCH_SYNCZONE_LOCAL

  // pointer to worker meta info
  st32                   $s_rGradBase, $sp, W_RGRAD_BASE/4
  st32                   $s_temp, $sp, W_METAINFO/4
  runall                 $s_wkrFunction, $sp, 0
  
LnextSubgroup:
  // dummy load to move pointer to next subgroup
  ldz16step              $mzero, $mzero, $s_metaInfo+=, $s_offsetToNextSubgroup
  brnz                   $s_rGradLd32Flag, LsubgroupLoopLd32
  ldz16step              $mzero, $mzero, $s_rGradBase+=, $s_subgroupSparseElems
  bri                    LsubgroupLoop
  
LendMetaInfoLoop:
ld32                   $m9, $sp, STACK_SIZE/4
add                    $sp, $sp, STACK_SIZE + 4
sync                   TEXCH_SYNCZONE_LOCAL
br                     $lr

.size COMMON_FN, . - COMMON_FN

// =============================================================================

#define FF4x4_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___float_float_4_4

DEF_STACK_USAGE (STACK_SIZE + 4) FF4x4_CODELET_NAME

.section .text.FF4x4_CODELET_NAME
.align 4
.globl FF4x4_CODELET_NAME
.type FF4x4_CODELET_NAME, @function
FF4x4_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block4x4SparseDenseMultiplyGradWFF
setzi             $s_rGradLd32Flag, 1
bri               COMMON_FN

.size FF4x4_CODELET_NAME, . - FF4x4_CODELET_NAME

// =============================================================================

#define HF4x4_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___half_float_4_4

DEF_STACK_USAGE (STACK_SIZE + 4) HF4x4_CODELET_NAME

.section .text.HF4x4_CODELET_NAME
.align 4
.globl HF4x4_CODELET_NAME
.type HF4x4_CODELET_NAME, @function
HF4x4_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block4x4SparseDenseMultiplyGradWHF
setzi             $s_rGradLd32Flag, 1
bri               COMMON_FN

.size HF4x4_CODELET_NAME, . - HF4x4_CODELET_NAME

// =============================================================================

#define HH4x4_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___half_half_4_4

DEF_STACK_USAGE (STACK_SIZE + 4) HH4x4_CODELET_NAME

.section .text.HH4x4_CODELET_NAME
.align 4
.globl HH4x4_CODELET_NAME
.type HH4x4_CODELET_NAME, @function
HH4x4_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block4x4SparseDenseMultiplyGradWHH
setzi             $s_rGradLd32Flag, 0
bri               COMMON_FN

.size HH4x4_CODELET_NAME, . - HH4x4_CODELET_NAME

// =============================================================================

#define FF8x8_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___float_float_8_8

DEF_STACK_USAGE (STACK_SIZE + 4) FF8x8_CODELET_NAME

.section .text.FF8x8_CODELET_NAME
.align 4
.globl FF8x8_CODELET_NAME
.type FF8x8_CODELET_NAME, @function
FF8x8_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block8x8SparseDenseMultiplyGradWFF
setzi             $s_rGradLd32Flag, 1
bri               COMMON_FN

.size FF8x8_CODELET_NAME, . - FF8x8_CODELET_NAME

// =============================================================================

#define HF8x8_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___half_float_8_8

DEF_STACK_USAGE (STACK_SIZE + 4) HF8x8_CODELET_NAME

.section .text.HF8x8_CODELET_NAME
.align 4
.globl HF8x8_CODELET_NAME
.type HF8x8_CODELET_NAME, @function
HF8x8_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block8x8SparseDenseMultiplyGradWHF
setzi             $s_rGradLd32Flag, 1
bri               COMMON_FN

.size HF8x8_CODELET_NAME, . - HF8x8_CODELET_NAME

// =============================================================================

#define HH8x8_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___half_half_8_8

DEF_STACK_USAGE (STACK_SIZE + 4) HH8x8_CODELET_NAME

.section .text.HH8x8_CODELET_NAME
.align 4
.globl HH8x8_CODELET_NAME
.type HH8x8_CODELET_NAME, @function
HH8x8_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block8x8SparseDenseMultiplyGradWHH
setzi             $s_rGradLd32Flag, 0
bri               COMMON_FN

.size HH8x8_CODELET_NAME, . - HH8x8_CODELET_NAME

// =============================================================================

#define HF16x16_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___half_float_16_16

DEF_STACK_USAGE (STACK_SIZE + 4) HF16x16_CODELET_NAME

.section .text.HF16x16_CODELET_NAME
.align 4
.globl HF16x16_CODELET_NAME
.type HF16x16_CODELET_NAME, @function
HF16x16_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block16x16SparseDenseMultiplyGradWHF
setzi             $s_rGradLd32Flag, 1
bri               COMMON_FN

.size HF16x16_CODELET_NAME, . - HF16x16_CODELET_NAME

// =============================================================================

#define HH16x16_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradW___half_half_16_16

DEF_STACK_USAGE (STACK_SIZE + 4) HH16x16_CODELET_NAME

.section .text.HH16x16_CODELET_NAME
.align 4
.globl HH16x16_CODELET_NAME
.type HH16x16_CODELET_NAME, @function
HH16x16_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, Block16x16SparseDenseMultiplyGradWHH
setzi             $s_rGradLd32Flag, 0
bri               COMMON_FN

.size HH16x16_CODELET_NAME, . - HH16x16_CODELET_NAME

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
