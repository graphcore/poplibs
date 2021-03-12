// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication R = Q * S' Where
// Q and S are dense matrices and R is a sparse matrix
//
// This serves the purpose of computing the entries of the
// sparse gradients with respect to weights.

#ifdef __IPU__
#include "SparseDenseMatMulGradWElementWise.h.S"
#include "SparseDenseMatMulStructs.h.S"
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================

#define HF_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulGradWElementWise___half_float
#define FF_CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulGradWElementWise___float_float

// =============================================================================

.extern elemwiseSparseDenseMultiplyGradFF

// =============================================================================

DEF_STACK_USAGE 0 zeroDenseOutFloatGrad
.section ".text.zeroDenseOutFloatGrad", FUNCTION_IS_WORKER
.type zeroDenseOutFloatGrad, @function
.align 8
nop

#define wkr_id_zv                       m0
#define zero_info_zv                    m1
#define outchan_ptr_zv                  m2
zeroDenseOutFloatGrad:
get           $wkr_id_zv, $WSR
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
ldz16         $zero_info_zv, $mvertex_base, SUP_VBASE_ZERO_INFO/2

// we could get zero information as this vertex could be called multiple times
// but zero infor field must be zero only in the first call
brz           $zero_info_zv, Loop_end_zero_64

// For n with 0 <= n <= 65533 this does a division by 6 with the remainder
// split amongst workers.
add           $zero_info_zv, $zero_info_zv, 6
sub           $zero_info_zv, $zero_info_zv, $wkr_id_zv
mul           $zero_info_zv, $zero_info_zv, 21845
shr           $zero_info_zv, $zero_info_zv, 17

// Minus 1 so we can quickly store the last element below
ld32          $outchan_ptr_zv, $mvertex_base, SUP_VBASE_RGRAD_BASE/4
ld64step      $azeros, $mzero, $outchan_ptr_zv+=, $wkr_id_zv

rpt           $zero_info_zv, (Loop_end_zero_64 - Loop_start_zero_64)/8 - 1
Loop_start_zero_64:
  {
    st64step      $azeros, $mzero, $outchan_ptr_zv+=, 6
    fnop
  }
Loop_end_zero_64:
exitz         $mzero

.size zeroDenseOutFloatGrad, . - zeroDenseOutFloatGrad

// =============================================================================

// worker registers
#define w_metaInfo                         m0
#define w_rGradBase                        m1
#define w_qGradBase                        m2
#define w_sBase                            m3
#define w_numZ                             m4
#define w_numWorkers                       m5
#define w_id                               m6
#define w_wkrInfoOffset                    m7
#define w_processWork                      m7
#define w_metaInfoOffsetOutputEntry        m8
#define w_offsetY                          m9
#define w_totalNumY                        m10
#define w_sparseOffset                     m11
#define w_zEq8                             m5
#define w_zEq4                             m6
#define w_numY                             m8
#define w_numZDiv4                         m3
#define w_numZRem2                         m5
#define w_numRem                           m4
#define w_qGradBaseLoop                    m6
#define w_sBaseLoop                        m7
#define w_packedPtr                        m6:7
#define w_numZMul2                         m7
#define w_offXInQ                          m11
#define w_offYInS                          m9

#define fp_clr_reg                         a1

// worker stack
#define w_StackEntry_numZMul2              0
#define w_StackSize                        (w_StackEntry_numZMul2 + 8)


DEF_STACK_USAGE w_StackSize elemwiseSparseDenseMultiplyGradHF
.section ".text.elemwiseSparseDenseMultiplyGradHF", FUNCTION_IS_WORKER
.type elemwiseSparseDenseMultiplyGradHF, @function
.align 8
.worker
// worker code
elemwiseSparseDenseMultiplyGradHF:

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

shl               $w_numZMul2, $w_numZ, 1
st32              $w_numZMul2, $mworker_base, w_StackEntry_numZMul2/4

ldz16step         $w_offXInQ, $mzero, $w_metaInfo+=, 1 
mul               $w_offXInQ, $w_offXInQ, $w_numZMul2
ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1 

// move meta info pointer by doing a dummy load
ldz16step         $mzero, $mzero, $w_metaInfo+=, $w_offsetY
sub               $w_numY, $w_numY, $w_offsetY
min               $w_numY, $w_numY, $w_totalNumY

{
  cmpeq             $w_zEq8, $w_numZ, 8
  setzi             $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  brnz              $w_zEq8, LZEq8Main
	uput              $FP_CLR, $fp_clr_reg 
}
cmpeq             $w_zEq4, $w_numZ, 4
brnz              $w_zEq4, LZEq4Main

shr               $w_numZDiv4, $w_numZ, 2
and               $w_numZRem2, $w_numZ, 0x2
and               $w_numRem, $w_numZ, 0x1 

LMainLoopX:
add               $w_numY, $w_numY, -1
ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1

LMainLoopY:
// Modify pointers to use pace instructions: stride increments are 1
{
  ld32              $w_sBaseLoop, $mvertex_base, W_S_BASE/4
  fnop
}
{
  add               $w_qGradBaseLoop, $w_qGradBase, $w_offXInQ
  mov               $a6:7, $azeros
}
{
  add               $w_sBaseLoop, $w_sBaseLoop, $w_offYInS
  mov               $a5, $azero
}

ld2x64pace        $a0:1, $a2:3, $w_packedPtr+=, $mzero, 0
ld32              $a4, $mzero, $w_rGradBase, 0
{
  rpt $w_numZDiv4,  0
  f32v4acc          $a4:7
}
  {
    ld2x64pace        $a0:1, $a2:3, $w_packedPtr+=, $mzero, 0
    f16v4cmac         $a0:1, $a2:3
  }

brz               $w_numZRem2, LRemLast
f16v2cmac         $a0, $a2
mov               $a0, $a1
mov               $a2, $a3

LRemLast:
brz               $w_numRem, LFinalAdd
sort4x16lo        $a0, $a0, $azero 
sort4x16lo        $a2, $a2, $azero 
f16v2cmac         $a0, $a2

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
ld32              $w_numZMul2, $mworker_base, w_StackEntry_numZMul2/4
mul               $w_offXInQ, $w_offYInS, $w_numZMul2
brnz              $w_numY, LMainLoopX
LEndWorker:
exitz             $mzero

// -----------------------------------------------------------------------------
// processes z=8
.align 8
LZEq8Main:
ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
sub               $w_totalNumY, $w_totalNumY, $w_numY
add               $w_numY, $w_numY, -1

ld64              $a0:1, $w_offXInQ, $w_qGradBase, 0
ld64              $a2:3, $w_offXInQ, $w_qGradBase, 1
ld64              $a4:5, $w_offYInS, $w_sBase, 0
{
  ld64              $a4:5, $w_offYInS, $w_sBase, 1
  f16v4cmac         $a0:1, $a4:5
}
{
  rpt               $w_numY, 4
  f16v4cmac         $a2:3, $a4:5
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
    f16v4cmac         $a0:1, $a4:5
  } 
  {
    ld64              $a4:5, $w_offYInS, $w_sBase, 1    
    f32add            $a6, $a6, $a7
  }
  {
    st32step          $a6, $mzero, $w_rGradBase+=, 1
    f16v4cmac         $a2:3, $a4:5
  }
{
  ldz16step         $w_offXInQ, $mzero, $w_metaInfo+=, 1
  f32v2gina         $a6:7, $azeros, 0
}
{
  ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1
  f32add            $a6, $a6, $a7
}
mul               $w_offXInQ, $w_offXInQ, $w_numZMul2
ld32              $a7, $mzero, $w_rGradBase, 0
{
  min               $w_numY, $w_totalNumY, $w_numY
  f32add            $a6, $a6, $a7
}
st32step          $a6, $mzero, $w_rGradBase+=, 1
brnz              $w_numY, LZEq8Main
exitz             $mzero

// -----------------------------------------------------------------------------
// processes z=4

LZEq4Main:
ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
sub               $w_totalNumY, $w_totalNumY, $w_numY
add               $w_numY, $w_numY, -1
ld64              $a0:1, $w_offXInQ, $w_qGradBase, 0
ld64              $a4:5, $w_offYInS, $w_sBase, 0
{
  rpt               $w_numY, 3
  f16v4cmac         $a0:1, $a4:5
}
  {
    ldz16step         $w_offYInS, $mzero, $w_metaInfo+=, 1
    f32v2gina         $a6:7, $azeros, 0
  }
  {
    ld32              $a7, $mzero, $w_rGradBase, 0
    f32add            $a6, $a6, $a7
  }
  {
    ld64              $a4:5, $w_offYInS, $w_sBase, 0
    f32add            $a6, $a6, $a7
  }
  {
    st32step          $a6, $mzero, $w_rGradBase+=, 1
    f16v4cmac         $a0:1, $a4:5
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
mul               $w_offXInQ, $w_offXInQ, $w_numZMul2
{  
  min               $w_numY, $w_totalNumY, $w_numY
  f32add            $a6, $a6, $a7
}
st32step          $a6, $mzero, $w_rGradBase+=, 1
brnz              $w_numY, LZEq4Main
exitz             $mzero
.size elemwiseSparseDenseMultiplyGradHF, . - elemwiseSparseDenseMultiplyGradHF

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. Meta information for a subgroup Id is expected
// to be found only once.

// Registers allocation

// Registers allocation
#define s_vertexBase                 m0
#define s_sBase                      m1
#define s_metaInfo                   m2
#define s_pnSubgroupId               m3
#define s_qGradBase                  m4
#define s_rGradBase                  m5
#define s_subgroupId                 m7
#define s_numZ                       m4
#define s_offsetToNextSubgroup       m1
#define s_subgroupSparseElems        m4
#define s_match                      m7
#define s_wkrFunction                m6
#define s_zeroWkrFunction            m8
#define s_numFwdWorkers              m8
#define s_offset                     m8
#define s_temp                       m7

#define COMMON_FN CommonSupervisor

// supervisor base is $m0 - passed to this function
DEF_STACK_USAGE  (STACK_SIZE) COMMON_FN

.section .text.COMMON_FN
.align 4
.type COMMON_FN, @function
COMMON_FN:
.supervisor

// allocate stack
add                    $sp, $sp, -STACK_SIZE
ldz16                  $s_numZ, $s_vertexBase, SUP_VBASE_NUM_Z/2

// This is the subgroup ID the PN has to process
ld32                   $s_pnSubgroupId, $s_vertexBase, SUP_VBASE_PN_SUBGROUP_ID/4
setzi                  $s_zeroWkrFunction, zeroDenseOutFloatGrad

// &S[0] is common to all the metaInformation tables
ld32                   $s_sBase, $s_vertexBase, SUP_VBASE_S_BASE/4

// &R[0] is common to all the metaInformation tables
ld32                   $s_rGradBase, $s_vertexBase, SUP_VBASE_RGRAD_BASE/4

ld32                   $s_metaInfo, $s_vertexBase, SUP_VBASE_META_BASE/4
st32                   $s_numZ, $sp, W_NUM_Z/4

// &Q[0] is common for the all the metaInfo tables
ld32                   $s_qGradBase, $s_vertexBase, SUP_VBASE_QGRAD_BASE/4
ldz16                  $s_pnSubgroupId, $mzero, $s_pnSubgroupId, 0
runall                 $s_zeroWkrFunction, $s_vertexBase, 0
st32                   $s_sBase, $sp, W_S_BASE/4
st32                   $s_rGradBase, $sp, W_RGRAD_BASE/4
st32                   $s_qGradBase, $sp, W_QGRAD_BASE/4

LsubgroupLoop:  
  ldz16                  $s_subgroupId, $s_metaInfo, MetaInfoSubGroupEntry_id/2
  ldz16                  $s_numFwdWorkers, $s_metaInfo, MetaInfoSubGroupEntry_numWorkers/2

  // s_metaInfo is at exactly where the numWorkers is so that it can be extracted 
  // by the worker (must be last field)
  ldz16                  $s_offsetToNextSubgroup, $s_metaInfo, MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo/2
  
 
  // The pointer to sparse R Is offset
  ldz16                  $s_subgroupSparseElems, $s_metaInfo, MetaInfoSubGroupEntry_sparseElementCount/2
  
  // If subgroup is 0 there is nothing to do
  brz                    $s_subgroupId, LendMetaInfoLoop
  mul                    $s_offset, $s_numFwdWorkers, Sizeof_MIFwdWorkerEntry

  // Check if any work to be done by the PN
  cmpeq                  $s_match, $s_subgroupId, $s_pnSubgroupId
  brz                    $s_match, LnextSubgroup

  // load number of workers for fwd as the number of workers for GradW is 
  // immediately after
  add                    $s_temp, $s_offset, $s_metaInfo
  add                    $s_temp, $s_temp, sizeof_MetaInfoSubGroupEntry

  // Need to sync because workers may  active and we touch common vertex state
  sync                   TEXCH_SYNCZONE_LOCAL

  // pointer to worker meta info
  st32                   $s_rGradBase, $sp, W_RGRAD_BASE/4
  st32                   $s_temp, $sp, W_METAINFO/4
  runall                 $s_wkrFunction, $sp, 0
  
LnextSubgroup:
  // dummy load to move pointer to next subgroup
  ldz16step              $mzero, $mzero, $s_metaInfo+=, $s_offsetToNextSubgroup
  ld32step               $mzero, $mzero, $s_rGradBase+=, $s_subgroupSparseElems
  bri                    LsubgroupLoop
  

LendMetaInfoLoop:
add                    $sp, $sp, STACK_SIZE
sync                   TEXCH_SYNCZONE_LOCAL
br                     $lr

.size COMMON_FN, . - COMMON_FN

// =============================================================================


DEF_STACK_USAGE (STACK_SIZE) HF_CODELET_NAME

.section .text.HF_CODELET_NAME
.align 4
.globl HF_CODELET_NAME
.type HF_CODELET_NAME, @function
HF_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, elemwiseSparseDenseMultiplyGradHF
bri               COMMON_FN

.size HF_CODELET_NAME, . - HF_CODELET_NAME

// =============================================================================

DEF_STACK_USAGE (STACK_SIZE) FF_CODELET_NAME

.section .text.FF_CODELET_NAME
.align 4
.globl FF_CODELET_NAME
.type FF_CODELET_NAME, @function
FF_CODELET_NAME:
.supervisor

setzi             $s_wkrFunction, elemwiseSparseDenseMultiplyGradFF
bri               COMMON_FN

.size FF_CODELET_NAME, . - FF_CODELET_NAME

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
