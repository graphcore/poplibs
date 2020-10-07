// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R' * S Where
// Q and S are dense matrices and R is a sparse matrix
//
// The meta information and NZ values for R are stored
// amenable for implementation of the forward pass and
// hence the transposition operation is implicitly done
// without the need for separate information

#ifdef __IPU__
#include "SparseDenseMatMulTranspElementWise.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulElementWiseTranspose___float_float

// =============================================================================

.extern zeroDenseOutFloatT

// =============================================================================

// worker stack
#define w_StackEntry_qBase                 0
#define w_StackEntry_numZDiv2              4
#define w_StackEntry_sBase                 8
#define w_StackEntry_wIdx2                 12
#define w_StackEntry_6mwId                 16
#define w_StackEntry_numXm1                20
#define w_StackEntry_wIdx4                 24
#define w_StackEntry_numZMul4              28
#define w_StackSize                        (w_StackEntry_numZMul4 + 4)

// worker registers
#define w_metaInfo                         m0
#define w_rBase                            m1
#define w_qBase                            m2
#define w_sBase                            m3
#define w_id                               m4
#define w_idx2                             m5
#define w_idx4                             m5
#define w_idm6                             m4
#define w_numXm1                           m4
#define w_numZ                             m5
#define w_temp                             m5
#define w_offsetXInQ                       m6
#define w_numY                             m7

#define w_qBaseLoop                        m4
#define w_sBaseLoop                        m3
#define w_rBaseLoop                        m8
#define w_numZMul4                         m8

#define w_deltaPtr                         m9
#define w_delta                            m10

#define w_numZDiv2                         m2
#define w_Zeq4                             m11
#define w_numZRem                          m11

#define fp_clr_reg                         a1

DEF_STACK_USAGE w_StackSize elemwiseSparseDenseMultiplyTransposeFF
.section ".text.elemwiseSparseDenseMultiplyTransposeFF", FUNCTION_IS_WORKER
.type elemwiseSparseDenseMultiplyTransposeFF, @function
.align 8
.worker
// worker code

elemwiseSparseDenseMultiplyTransposeFF:
ld32              $w_metaInfo, $mvertex_base, W_METAINFO/4
ld32              $w_rBase, $mvertex_base, W_R_BASE/4
ld32              $w_qBase, $mvertex_base, W_Q_BASE/4
ld32              $w_sBase, $mvertex_base, W_S_BASE/4

// We need simple functions of the worker id
// 1. sizeof(half) * worker_id to offset into the sparse entry and column entry
// 2. 6 - worker_id used in the division of work.
get               $w_id, $WSR
and               $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
mul               $w_idx2, $w_id, 2
st32              $w_idx2, $mworker_base, w_StackEntry_wIdx2/4
mul               $w_idx4, $w_idx2, 2
st32              $w_idx4, $mworker_base, w_StackEntry_wIdx4/4
sub               $w_idm6, 6, $w_id
st32              $w_idm6, $mworker_base, w_StackEntry_6mwId/4

{
  ld32              $w_numZ, $mvertex_base, W_NUM_Z/4
  setzi             $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  ld32              $w_numXm1, $mvertex_base, W_NUM_XM1/4
  uput              $FP_CLR, $fp_clr_reg 
}

st32              $w_qBase, $mworker_base, w_StackEntry_qBase/4
st32              $w_sBase, $mworker_base, w_StackEntry_sBase/4

shl               $w_numZMul4, $w_numZ, 2
st32              $w_numZMul4, $mworker_base, w_StackEntry_numZMul4/4

// We process  entries at a time if any and process the remainder if any.
shr               $w_numZDiv2, $w_numZ, 1
add               $w_numZDiv2, $w_numZDiv2, -1
st32              $w_numZDiv2, $mworker_base, w_StackEntry_numZDiv2/4
and               $w_numZRem, $w_numZ, 0x1

// A sync mechanism is used in the worker where workers are synchronised for
// each X. This is required because the same elements by be read/written to
// for different X. The work is split between workers in such a way that
// worker 0 always has at least as much work to do as the others. And worker 0
// is responsible for setting the sync.
LxLoop: 
#undef w_flag
#define  w_flag   w_numY

CheckFlag:
  ld32              $w_flag, $mvertex_base, W_FLAG/4
  brz               $w_flag, CheckFlag

  st32              $w_numXm1, $mworker_base, w_StackEntry_numXm1/4
  // Load output entries for this output row (x dimension). 
  ldz16step         $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  ld32              $w_numZMul4, $mworker_base, w_StackEntry_numZMul4/4
  mul               $w_offsetXInQ, $w_offsetXInQ, $w_numZMul4
  ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1

  // We divide each row in the FWD amongst available workers. And this means we
  // cannot control the number of 
  mov               $w_deltaPtr, $w_metaInfo

  ld32              $w_temp, $mworker_base, w_StackEntry_wIdx4/4
  add               $w_rBaseLoop, $w_rBase, $w_temp

  // move pointer to next output entry
  ldz16step         $mzero, $mzero, $w_metaInfo+=, $w_numY
  // move pointer to NZ values of next output entry
  ld32step          $mzero, $mzero, $w_rBase+=, $w_numY

  // divide work for each fwd row 
  ld32              $w_temp, $mworker_base, w_StackEntry_6mwId/4
  add               $w_numY, $w_numY, $w_temp
  mul               $w_numY, $w_numY, 21845
  shr               $w_numY, $w_numY, 17

  st32              $mzero, $mvertex_base, W_FLAG/4

  // some workers may not have anything to do
  brz               $w_numY, LRestoreUpdateXState
  add               $w_numY, $w_numY, -1
  ld32              $w_temp, $mworker_base, w_StackEntry_wIdx2/4

LyLoop:

    // metaInfo -> offset of column entries in 'y' dimension 
    ld32              $w_qBaseLoop, $mworker_base, w_StackEntry_qBase/4
    ld32              $w_sBaseLoop, $mworker_base, w_StackEntry_sBase/4

    // the Yoffsets are stored for half but used here for partials of type float
    ldz16step         $w_delta, $w_temp, $w_deltaPtr+=, 6
    ld32step          $a3, $mzero, $w_rBaseLoop+=, 6

    // Check if there are any multiples of 8 to process. If not, jump straight to
    // process remainder.
    ld32              $w_numZDiv2, $mworker_base, w_StackEntry_numZDiv2/4
    brneg             $w_numZDiv2, LzRem

    ld64step          $a0:1, $w_offsetXInQ, $w_qBaseLoop+=, 1
    {
      ld64              $a4:5, $w_delta, $w_sBaseLoop, 0
      f32v2mul          $a6:7, $a3:B, $a0:1
    }
    rpt               $w_numZDiv2, (LoopZEnd2 - LoopZStart2)/8 - 1
LoopZStart2:
      {
        ld64step          $a0:1, $w_offsetXInQ, $w_qBaseLoop+=, 1
        f32v2add          $a6:7, $a6:7, $a4:5
      }
      {
        st64step          $a6:7, $w_delta, $w_sBaseLoop+=, 1
        f32v2mul          $a6:7, $a3:B, $a0:1
      }
      {
        ld64              $a4:5, $w_delta, $w_sBaseLoop, 0
        fnop
      }
LoopZEnd2:
    f32v2add          $a6:7, $a6:7, $a4:5
    st64step          $a6:7, $w_delta, $w_sBaseLoop+=, 1
LzRem:
    brz               $w_numZRem, LEndY      
      
    ld32step          $a0, $w_offsetXInQ, $w_qBaseLoop+=, 1
    {
      ld32              $a4, $w_delta, $w_sBaseLoop, 0
      f32mul            $a6, $a3, $a0
    }
    f32add            $a6, $a6, $a4
    st32step          $a6, $w_delta, $w_sBaseLoop+=, 1

LEndY:
    brnzdec           $w_numY, LyLoop

    brnz              $w_temp, LxCheck
    st32              $w_metaInfo, $mvertex_base, W_FLAG/4
LxCheck:

LRestoreUpdateXState:
  ld32              $w_numXm1, $mworker_base, w_StackEntry_numXm1/4
  brnzdec           $w_numXm1, LxLoop
exitz             $mzero

.size elemwiseSparseDenseMultiplyTransposeFF, . - elemwiseSparseDenseMultiplyTransposeFF

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

ELEM_SPARSE_MATMUL_TRANSP CODELET_NAME float elemwiseSparseDenseMultiplyTransposeFF

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
