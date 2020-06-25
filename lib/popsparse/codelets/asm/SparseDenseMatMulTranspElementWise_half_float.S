// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication S = R' * Q Where
// Q and S are dense matrices and R is a sparse matrix
//
// The meta information and NZ values for R are stored in a form
// amenable for implementation of the forward pass and
// hence the transposition operation is implicitly done
// without the need for separate information

#ifdef __IPU__
#include "SparseDenseMatMulStructs.h.S"
#include "SparseDenseMatMulTranspElementWise.h.S"
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define LOG2_SIZEOF_OUT_ATOM 2
#define SHORT_SPAN_ADDRESS_BITS 20

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulElementWiseTranspose___half_float

// =============================================================================

// Zero output/partials
//
// Performance: 14 + num_samples / 2

DEF_STACK_USAGE 0 zeroDenseOutFloatT
.section ".text.zeroDenseOutFloatT", FUNCTION_IS_WORKER
.type zeroDenseOutFloatT, @function
.globl zeroDenseOutFloatT
.align 8

#define wkr_id_zv                       m0
#define zero_info_zv                    m1
#define zero_info_div_12_zv             m2
// Registers above must be retained between calls
#define outchan_ptr_zv                  m3
zeroDenseOutFloatT:
get           $wkr_id_zv, $WSR
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
ldz16         $zero_info_zv, $mvertex_base, SUP_VBASE_ZERO_INFO/2

// we could get zero information as this vertex could be called multiple times
// but zero infor field must be zero only in the first call
brz           $zero_info_zv, Loop_end_zero_64
shr           $zero_info_div_12_zv, $zero_info_zv, (3 - LOG2_SIZEOF_OUT_ATOM)

// For n with 0 <= n <= 65533 this does a division by 6 with the remainder
// split amongst workers.
add           $zero_info_div_12_zv, $zero_info_div_12_zv, 6
sub           $zero_info_div_12_zv, $zero_info_div_12_zv, $wkr_id_zv
mul           $zero_info_div_12_zv, $zero_info_div_12_zv, 21845
shr           $zero_info_div_12_zv, $zero_info_div_12_zv, 17

// Minus 1 so we can quickly store the last element below
add           $zero_info_zv, $zero_info_zv, -1
ld32          $outchan_ptr_zv, $mvertex_base, SUP_VBASE_S_BASE/4
// Unconditionally write the last element in all the workers
st32          $azero, $outchan_ptr_zv, $mzero, $zero_info_zv
ld64step      $azeros, $mzero, $outchan_ptr_zv+=, $wkr_id_zv

rpt           $zero_info_div_12_zv, (Loop_end_zero_64 - Loop_start_zero_64)/8 - 1

Loop_start_zero_64:
  {
    st64step      $azeros, $mzero, $outchan_ptr_zv+=, 6
    fnop
  }
Loop_end_zero_64:
exitz         $mzero

.size zeroDenseOutFloatT, . - zeroDenseOutFloatT

// =============================================================================

// worker stack
#define w_StackEntry_qBase                 0
#define w_StackEntry_numZDiv4              4
#define w_StackEntry_sBase                 8
#define w_StackEntry_wIdx2                 12
#define w_StackEntry_6mwId                 16
#define w_StackEntry_numXm1                20
#define w_StackSize                        (w_StackEntry_numXm1 + 4)

// worker registers
#define w_metaInfo                         m0
#define w_rBase                            m1
#define w_qBase                            m2
#define w_sBase                            m3
#define w_id                               m4
#define w_idx2                             m5
#define w_idm6                             m4
#define w_numXm1                           m4
#define w_numZ                             m5
#define w_temp                             m5
#define w_offsetXInQ                       m6
#define w_numY                             m7

#define w_delta1                           m11
#define w_qBaseLoop                        m4
#define w_sBaseLoop                        m3

// Note!! w_rBaseLoop and w_deltaPtr must be together as TAPACK
// is used
#define w_rBaseLoop                        m8
#define w_deltaPtr                         m9
#define w_biaddr                           m8:9

#define w_delta                            m10

#define w_numZDiv4                         m2
#define w_Zeq8                             m11
#define w_Zeq4                             m11
#define w_numZRem                          m11

#define fp_clr_reg                         a1

DEF_STACK_USAGE w_StackSize elemwiseSparseDenseMultiplyTransposeHF
.section ".text.elemwiseSparseDenseMultiplyTransposeHF", FUNCTION_IS_WORKER
.type elemwiseSparseDenseMultiplyTransposeHF, @function
.align 8
.worker
// worker code

elemwiseSparseDenseMultiplyTransposeHF:
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
sub               $w_idm6, 6, $w_id
st32              $w_idm6, $mworker_base, w_StackEntry_6mwId/4

ld32              $w_numZ, $mvertex_base, W_NUM_Z/4
ld32              $w_numXm1, $mvertex_base, W_NUM_XM1/4

// specialisations for numZ = 8 and numZ = 4
{
  cmpeq             $w_Zeq8, $w_numZ, 8
  setzi             $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  brnz              $w_Zeq8, LSpZEq8
  uput              $FP_CLR, $fp_clr_reg 
}
cmpeq             $w_Zeq4, $w_numZ, 4
brnz              $w_Zeq4, LSpZEq4

st32              $w_qBase, $mworker_base, w_StackEntry_qBase/4
st32              $w_sBase, $mworker_base, w_StackEntry_sBase/4

// We process 4 entries at a time if any and process the remainder if any.
shr               $w_numZDiv4, $w_numZ, 2
add               $w_numZDiv4, $w_numZDiv4, -1
st32              $w_numZDiv4, $mworker_base, w_StackEntry_numZDiv4/4
and               $w_numZRem, $w_numZ, 0x3
add               $w_numZRem, $w_numZRem, -1

// A sync mechanism is used in the worker where workers are synchronised for
// each X. This is required because the same elements by be read/written to
// for different X. The work is split between workers in such a way that
// worker 0 always has at least as much work to do as the others. And worker 0
// is responsible for setting the sync.

LxLoop: 
#undef w_flag
#define  w_flag   w_numY

CheckFlag:
  ld32                  $w_flag, $mvertex_base, W_FLAG/4
  brz                   $w_flag, CheckFlag

  st32              $w_numXm1, $mworker_base, w_StackEntry_numXm1/4
  // Load output entries for this output row (x dimension). 
  ldz16step         $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1

  // We divide each row in the FWD amongst available workers. And this means we
  // cannot control the number of 
  // mov               $w_deltaPtr, $w_metaInfo
  // mov               $w_rBaseLoop, $w_rBase
  tapack            $w_biaddr, $w_rBase, $w_metaInfo, $mzero

  // move pointer to next output entry
  ldz16step         $mzero, $mzero, $w_metaInfo+=, $w_numY
  // move pointer to NZ values of next output entry
  ldz16step         $mzero, $mzero, $w_rBase+=, $w_numY

  // divide work for each fwd row 
  ld32              $w_temp, $mworker_base, w_StackEntry_6mwId/4
  add               $w_numY, $w_numY, $w_temp
  mul               $w_numY, $w_numY, 21845
  shr               $w_numY, $w_numY, 17

  st32                  $mzero, $mvertex_base, W_FLAG/4

  // some workers may not have anything to do
  // TODO: optimise this
  brz               $w_numY, LRestoreUpdateXState
  add               $w_numY, $w_numY, -1
  ld32              $w_temp, $mworker_base, w_StackEntry_wIdx2/4

LyLoop:

    // metaInfo -> offset of column entries in 'y' dimension 
    ld32              $w_qBaseLoop, $mworker_base, w_StackEntry_qBase/4
    ld32              $w_sBaseLoop, $mworker_base, w_StackEntry_sBase/4

    // the Yoffsets are stored for half but used here for partials of type float
    ldz16step         $w_delta, $w_temp, $w_deltaPtr+=, 6
    shl               $w_delta, $w_delta, 1

    ldb16step         $a3, $w_temp, $w_rBaseLoop+=, 6

    // Check if there are any multiples of 8 to process. If not, jump straight to
    // process remainder.
    ld32              $w_numZDiv4, $mworker_base, w_StackEntry_numZDiv4/4
    brneg             $w_numZDiv4, LzRem
    {
      ld64step          $a0:1, $w_offsetXInQ, $w_qBaseLoop+=, 1
      fnop
    }
    {
      ld64              $a4:5, $w_delta, $w_sBaseLoop, 0
      f16v4mul          $a0:1, $a3:BL, $a0:1
    }
    {
      rpt               $w_numZDiv4, (LoopZEnd4 - LoopZStart4)/8 - 1
      f16v2tof32        $a6:7, $a0
    }
LoopZStart4:
      {
        ld64              $a6:7, $w_delta, $w_sBaseLoop, 1
        f32v2add          $a4:5, $a4:5, $a6:7
      }
      {
        st64step          $a4:5, $w_delta, $w_sBaseLoop+=, 1
        f16v2tof32        $a4:5, $a1
      }
      {
        ld64step          $a0:1, $w_offsetXInQ, $w_qBaseLoop+=, 1
        f32v2add          $a4:5, $a4:5, $a6:7
      }
      {
        st64step          $a4:5, $w_delta, $w_sBaseLoop+=, 1  
        f16v4mul          $a0:1, $a3:BL, $a0:1
      }
      {
        ld64              $a4:5, $w_delta, $w_sBaseLoop, 0
        f16v2tof32        $a6:7, $a0
      }
LoopZEnd4:
    {
      ld64              $a6:7, $w_delta, $w_sBaseLoop, 1
      f32v2add          $a4:5, $a4:5, $a6:7
    }
    {
      st64step          $a4:5, $w_delta, $w_sBaseLoop+=, 1
      f16v2tof32        $a4:5, $a1
    }
    f32v2add          $a4:5, $a4:5, $a6:7
    st64step          $a4:5, $w_delta, $w_sBaseLoop+=, 1
LzRem:
    brneg             $w_numZRem, LEndY
    ldb16step         $a0, $w_offsetXInQ, $w_qBaseLoop+=, 1
    {
      ld32              $a4, $w_delta, $w_sBaseLoop, 0
      f16v2mul          $a0, $a3, $a0
    }
    {
      rpt                 $w_numZRem, (LoopZ1End - LoopZ1Start)/8 - 1
      f16tof32            $a6, $a0
    }
LoopZ1Start:
      {
        ldb16step           $a0, $w_offsetXInQ, $w_qBaseLoop+=, 1
        f32add              $a6, $a4, $a6
      }
      {
        st32step            $a6, $w_delta, $w_sBaseLoop+=, 1
        f16v2mul            $a0, $a3, $a0
      }
      {
        ld32                $a4, $w_delta, $w_sBaseLoop, 0
        f16tof32            $a6, $a0
      }
LoopZ1End:
    f32add            $a6, $a4, $a6
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

// Specialisation for Z=8
LSpZEq8:
ld32                  $w_temp, $mworker_base, w_StackEntry_wIdx2/4
// Load output entries for this output row (x dimension)  
ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1

LxLoopSp8:  
#undef w_flag
#define  w_flag   w_delta1
#undef w_numY_loop
#define w_temp_6mwId w_delta1

CheckFlagSp8:
  ld32                  $w_flag, $mvertex_base, W_FLAG/4
  brz                   $w_flag, CheckFlagSp8

  // mov  $w_deltaPtr, $w_metaInfo
  // mov  $w_rBaseLoop, $w_rBase
  tapack                $w_biaddr, $w_rBase, $w_metaInfo, $mzero

  // move metaInfo pointer to next output entry
  ldz16step             $mzero, $mzero, $w_metaInfo+=, $w_numY
  // move nz weights pointer to next row entry
  ldz16step             $mzero, $mzero, $w_rBase+=, $w_numY

  // divide work for each fwd row 
  ld32                  $w_temp_6mwId, $mworker_base, w_StackEntry_6mwId/4
  add                   $w_numY, $w_numY, $w_temp_6mwId
  mul                   $w_numY, $w_numY, 21845
  shr                   $w_numY, $w_numY, 17
  st32                  $mzero, $mvertex_base, W_FLAG/4  
  brnzdec               $w_numY, ContColSp8

  // Load output entries for this output row (x dimension)  
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1

  // Ideally worker 0 shouldn't reach here. So this check should be
  // redundant.

  brnz                  $w_temp, LxCheckSp8_0
  st32                  $w_metaInfo, $mvertex_base, W_FLAG/4

LxCheckSp8_0:
  brnzdec                $w_numXm1, LxLoopSp8
exitz                 $mzero

ContColSp8:
  ld64                  $a4:5, $w_offsetXInQ, $w_qBase, 0  
  ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 1  
  ldz16step             $w_delta1, $w_temp, $w_deltaPtr+=, 6
  ldb16step             $a3, $w_temp, $w_rBaseLoop+=, 6

  // We can use ld128 because numZ is a multiple of 4
  {
    rpt                  $w_numY, (LyLoopEndSp8 - LyLoopStartSp8)/8 - 1
    f16v4mul             $a0:1, $a3:BL, $a4:5
  }
LyLoopStartSp8:
    {
      // the Yoffsets are stored for half but used here for partials of type 
      // float.
      shl               $w_delta, $w_delta1, 1
      f16v4mul          $a2:3, $a3:BL, $a6:7    
    }
    {   
      ld128             $a0:3, $w_delta, $w_sBase, 0
      f16v8acc          $a0:3     
    }
    {
      ldz16step         $w_delta1, $w_temp, $w_deltaPtr+=, 6
      f32v4acc          $a0:3
    }
    {
      ld128             $a0:3, $w_delta, $w_sBase, 1  
      f32v2gina         $a6:7, $azeros, 0
    }
    {
      st64              $a6:7, $w_delta, $w_sBase, 0
      f32v2gina         $a6:7, $azeros, 0
    }
    {
      st64              $a6:7, $w_delta, $w_sBase, 1
      f32v4acc          $a0:3
    }
    {
      ld64              $a6:7, $w_offsetXInQ, $w_qBase, 1
      f32v2gina         $a0:1, $azeros, 0
    }
    {
      st64              $a0:1, $w_delta, $w_sBase, 2
      f32v2gina         $a0:1, $azeros, 0
    }
    {
      ldb16step             $a3, $w_temp, $w_rBaseLoop+=, 6
      fnop
    }
    {
      st64                $a0:1, $w_delta, $w_sBase, 3
      f16v4mul            $a0:1, $a3:BL, $a4:5    
    }
LyLoopEndSp8:
  {
    // the Yoffsets are stored for half but used here for partials of type 
    // float.
    shl               $w_delta, $w_delta1, 1
    f16v4mul          $a2:3, $a3:BL, $a6:7    
  }
  {   
    ld128             $a0:3, $w_delta, $w_sBase, 0
    f16v8acc          $a0:3     
  }
  {
    ldz16step         $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
    f32v4acc          $a0:3
  }
  {
    ld128             $a0:3, $w_delta, $w_sBase, 1  
    f32v2gina         $a6:7, $azeros, 0
  }
  {
    st64              $a6:7, $w_delta, $w_sBase, 0
    f32v2gina         $a6:7, $azeros, 0
  }
  {
    st64              $a6:7, $w_delta, $w_sBase, 1
    f32v4acc          $a0:3
  }
  {
    ldz16step         $w_numY, $mzero, $w_metaInfo+=, 1
    f32v2gina         $a0:1, $azeros, 0
  }
  {
    st64              $a0:1, $w_delta, $w_sBase, 2
    f32v2gina         $a0:1, $azeros, 0
  }
  st64                $a0:1, $w_delta, $w_sBase, 3

  brnz                   $w_temp, LxCheckSp8
  st32                   $w_metaInfo, $mvertex_base, W_FLAG/4

LxCheckSp8:
  brnzdec                $w_numXm1, LxLoopSp8
exitz                 $mzero


// Specialisation of Z=4
LSpZEq4:
ld32                  $w_temp, $mworker_base, w_StackEntry_wIdx2/4

LxLoopSp4:  
#undef w_flag
#define w_flag w_numY
#undef w_temp_6mwId
#define w_temp_6mwId w_delta1

CheckFlagSp4:
  ld32                  $w_flag, $mvertex_base, W_FLAG/4
  brz                   $w_flag, CheckFlagSp4

  // Load output entries for this output row (x dimension)  
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1

  // mov  $w_deltaPtr, $w_metaInfo
  // mov  $w_rBaseLoop, $w_rBase
  tapack                $w_biaddr, $w_rBase, $w_metaInfo, $mzero

  // move metaInfo pointer to next output entry
  ldz16step             $mzero, $mzero, $w_metaInfo+=, $w_numY
   // move nz weights pointer to next row entry
  ldz16step             $mzero, $mzero, $w_rBase+=, $w_numY  

  // divide work for each fwd row 
  ld32                  $w_temp_6mwId, $mworker_base, w_StackEntry_6mwId/4
  add                   $w_numY, $w_numY, $w_temp_6mwId
  mul                   $w_numY, $w_numY, 21845
  shr                   $w_numY, $w_numY, 17

  ld64                  $a4:5, $w_offsetXInQ, $w_qBase, 0  
  ldb16step             $a3, $w_temp, $w_rBaseLoop+=, 6
  ldz16step             $w_delta1, $w_temp, $w_deltaPtr+=, 6

  st32                  $mzero, $mvertex_base, W_FLAG/4

  // We can use ld128 because numZ is a multiple of 4
  rpt                   $w_numY, (LyLoopEndSp4 - LyLoopStartSp4)/8 - 1
LyLoopStartSp4:
    {
      // the Yoffsets are stored for half but used here for partials of type 
      // float.
      shl               $w_delta, $w_delta1, 1
      f16v4mul          $a4:5, $a3:BL, $a4:5
    }
    {
      ld128             $a0:3, $w_delta, $w_sBase, 0
      fnop
    }
    {   
      ldz16step         $w_delta1, $w_temp, $w_deltaPtr+=, 6
      f16v4acc          $a4:5     
    }
    {
      ldb16step         $a3, $w_temp, $w_rBaseLoop+=, 6
      f32v4acc          $a0:3
    }
    {
      ld64              $a4:5, $w_offsetXInQ, $w_qBase, 0
      f32v2gina         $a6:7, $azeros, 0
    }
    {
      st64              $a6:7, $w_delta, $w_sBase, 0
      f32v2gina         $a6:7, $azeros, 0
    }
    {
      st64              $a6:7, $w_delta, $w_sBase, 1
      fnop
    }
LyLoopEndSp4:

  brnz                   $w_temp, LxCheckSp4
  st32                   $w_metaInfo, $mvertex_base, W_FLAG/4

LxCheckSp4:
  brnzdec                $w_numXm1, LxLoopSp4
exitz                 $mzero


.size elemwiseSparseDenseMultiplyTransposeHF, . - elemwiseSparseDenseMultiplyTransposeHF

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id.

ELEM_SPARSE_MATMUL_TRANSP CODELET_NAME half elemwiseSparseDenseMultiplyTransposeHF

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
