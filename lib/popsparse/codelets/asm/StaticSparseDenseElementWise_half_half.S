// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
//

#ifdef __IPU__
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"


// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseElementWise___half_half

// =============================================================================

#define W_METAINFO  12
#define W_R_BASE    8  
#define W_Q_BASE    0
#define W_S_BASE    4
#define W_NUM_Z     16

// =============================================================================
// byte offsets into header
#define WKR_HDR_NUM_ROWS        0
#define WKR_HDR_WKR_OFFSET      2
#define WKR_HDR_SPARSE_OFFSET   4
#define WKR_HDR_NUMZ            6
#define WKR_HDR_OFFSET_Z        8
#define WKR_HDR_ROW_OFFSET      10
#define WKR_HDR_NUM_SIZE        (WKR_HDR_ROW_OFFSET + 2)

// =============================================================================

// worker stack
#define w_StackEntry_rBase                 0
#define w_StackEntry_numZDiv8              4
#define w_StackEntry_sBase                 8
#define w_StackEntry_numZMul2              12
#define w_StackSize                        (w_StackEntry_numZMul2 + 4)

// worker registers
#define w_metaInfo                         m0
#define w_rBase                            m1
#define w_qBase                            m2
#define w_sBase                            m3
#define w_id                               m5
#define w_xOff                             m4
#define w_worklistOff                      m6
#define w_nzOffset                         m4
#define w_offsetZ                          m4 
#define w_numXm1                           m5
#define w_numZ                             m7
#define w_sBaseLoop                        m4
#define w_offsetXInQ                       m6
#define w_numYm1                           m8
#define w_qBaseLoop                        m9
#define w_rLoop                            m10
#define w_deltaPtr                         m1
#define w_delta                            m11
#define w_zEq8                             m4
#define w_zEq4                             m4

#define w_numZMul2                         m9
#define w_xOffIncr                         m9
#define w_numZDiv8                         m3
#define w_numZRem                          m7
#define w_numZDiv4                         m3
#define w_finalRem                         m3

#define w_rData                            a0
#define w_sDataL                           a2
#define w_sData                            a2:3
#define fp_clr_reg                         a1

DEF_STACK_USAGE w_StackSize CODELET_NAME
.section .text.CODELET_NAME
.type CODELET_NAME, @function
.globl CODELET_NAME
.align 8
.worker
// worker code

CODELET_NAME:
ld32                  $w_metaInfo, $mvertex_base, W_METAINFO/4
ld32                  $w_rBase, $mvertex_base, W_R_BASE/4
ld32                  $w_qBase, $mvertex_base, W_Q_BASE/4
ld32                  $w_sBase, $mvertex_base, W_S_BASE/4

// This is the numZ per tile. The numZ for a thread may differ from this
ld32                  $w_numZ, $mvertex_base, W_NUM_Z/4
mul                   $w_numZMul2, $w_numZ, 2
st32                  $w_numZMul2, $mworker_base, w_StackEntry_numZMul2/4

// The number of workers is the first field
// w_metaInfo -> worker entries
get                   $w_id, $WSR
and                   $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
mul                   $w_id, $w_id, WKR_HDR_NUM_SIZE

ldz16                 $w_worklistOff, $w_id, $w_metaInfo, WKR_HDR_WKR_OFFSET/2
ldz16                 $w_nzOffset, $w_id, $w_metaInfo, WKR_HDR_SPARSE_OFFSET/2
ldz16step             $mzero, $mzero, $w_rBase+=, $w_nzOffset
ldz16                 $w_offsetZ, $w_id, $w_metaInfo, WKR_HDR_OFFSET_Z/2
ldz16step             $mzero, $mzero, $w_qBase+=,  $w_offsetZ
ldz16step             $mzero, $mzero, $w_sBase+=,  $w_offsetZ
ldz16                 $w_numZ, $w_id, $w_metaInfo, WKR_HDR_NUMZ/2
ldz16                 $w_xOff, $w_id, $w_metaInfo, WKR_HDR_ROW_OFFSET/2
ldz16                 $w_numXm1, $w_id, $w_metaInfo, 0

brnzdec               $w_numXm1, LWorkerCont
exitz                 $mzero

LWorkerCont:
ldz16step             $mzero, $mzero, $w_metaInfo+=, $w_worklistOff

// move to correct offset into X: X is always linearly allocated to workers
mul                   $w_offsetXInQ, $w_numZMul2, $w_xOff

// branch to specialisations
{
  cmpeq                 $w_zEq8, $w_numZ, 8
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  brnz                  $w_zEq8, LZEq8Sp
  uput                  $FP_CLR, $fp_clr_reg 
}
cmpeq                 $w_zEq4, $w_numZ, 4
brnz                  $w_zEq4, LZEq4Sp

// save &r[sparseOffset] and &s[offsetZ] on stack. These will be update
// for different 'x' entries in the loop.
st32                  $w_rBase, $mworker_base, w_StackEntry_rBase/4
st32                  $w_sBase, $mworker_base, w_StackEntry_sBase/4

// We process 8 entries at a time if possible and handle the remaining quad
// if any.
shr                   $w_numZDiv8, $w_numZ, 3

// use of brnzdec, so subtract by 1.
add                   $w_numZDiv8,  $w_numZDiv8, -1

// we only need to know if there is a remainder. An and by 0x7 is sufficient
and                   $w_numZRem, $w_numZ, 0x7

// save on stack to avoid recomputing in loop.
st32                  $w_numZDiv8, $mworker_base, w_StackEntry_numZDiv8/4

LxLoop: 
  // Each output row in has entries which always offset from the &s[offsetZ].
  ld32                  $w_sBaseLoop, $mworker_base, w_StackEntry_sBase/4

  // Load output entries for this output row (x dimension). 
  ldz16step             $w_numYm1, $mzero, $w_metaInfo+=, 1
  // metaInfo -> offset of column entries in 'y' dimension 
  mov                   $w_qBaseLoop, $w_qBase

  // Check if there are any multiples of 8 to process. If not, jump straight to
  // process remainder.
  ld32                  $w_numZDiv8, $mworker_base, w_StackEntry_numZDiv8/4
  brneg                 $w_numZDiv8, LzRem

LzLoop8:      
    // we need to reuse the same entries in R for all the same output row
    // and for any z dimension. So reload pointers to R and offsets in S.
    ld32                  $w_rLoop, $mworker_base, w_StackEntry_rBase/4
    mov                   $w_deltaPtr, $w_metaInfo
    
    // we need to multply the whole Z dimension entries by the same sparse
    // entry in R
    {
      ldb16step             $w_rData, $mzero, $w_rLoop+=, 1
      fnop
    }
    {
      ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
       mov                  $a4:5, $azeros
    }
    // delta's are byte offsets and as we are processing 8 columns of S at
    // at time load the second quad first.
    {
      rpt                   $w_numYm1, (LEndYLoop8 - LStartYLoop8) / 8 - 1
      mov                   $a6:7, $azeros
    }  
LStartYLoop8:         
      { 
        ld64                  $w_sData, $w_delta, $w_sBaseLoop, 1
        f16v8acc              $a4:7 
      }
      { 
        ldd16a64              $w_sData, $w_deltaPtr++, $w_sBaseLoop, $w_delta@ 
        f16v4mul              $a6:7, $w_rData:BL, $w_sData 
      }
      { 
        ldb16step             $w_rData, $mzero, $w_rLoop+=, 1
        f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
      }
LEndYLoop8: 
    { 
      ld64                  $w_sData, $w_delta, $w_sBaseLoop, 1
      f16v8acc              $a4:7 
    }
    { 
      ldd16a64              $w_sData, $w_deltaPtr++, $w_sBaseLoop, $w_delta@ 
      f16v4mul              $a6:7, $w_rData:BL, $w_sData 
    }
    { 
      ldb16step             $w_rData, $mzero, $w_rLoop+=, 1
      f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
    }
    f16v8acc              $a4:7 
    { 
      // We have used up 8 halves of s. move to next set of columns.
      add                   $w_sBaseLoop, $w_sBaseLoop, 16
      f16v2gina             $a6, $azero, 0
    }
    { 
      st32step              $a6, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f16v2gina             $a6, $azero, 0 
    }
    { 
      st32step              $a6, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f16v4gacc             $a6:7 
    }
    { 
      st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      // to propagate zeros through the acc chain
      f16v2gina             $azero, $azero, 0 
    }
    {
      brnzdec               $w_numZDiv8, LzLoop8
      // to propagate zeros through the acc chain
      f16v2gina             $azero, $azero, 0 
    }

LzRem:  
    brz                   $w_numZRem, LRestoreUpdateXState  
    ld32                  $w_rLoop, $mworker_base, w_StackEntry_rBase/4
    ldb16step             $w_rData, $mzero, $w_rLoop+=, 1
    {
      ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1
      fnop
    }
    {
      rpt                   $w_numYm1, (LEndYLoop4 - LStartYLoop4) / 8 - 1
      mov                   $a6:7, $azeros
    }
LStartYLoop4:         
      { 
        ldd16a64              $w_sData, $w_metaInfo++, $w_sBaseLoop, $w_delta@
        f16v4acc              $a6:7  
      }
      { 
        ldb16step             $w_rData, $mzero, $w_rLoop+=, 1
        f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
      }
LEndYLoop4: 
    { 
      ldd16a64              $w_sData, $w_metaInfo++, $w_sBaseLoop, $w_delta@
      f16v4acc              $a6:7  
    }
    { 
      ldb16step             $w_rData, $mzero, $w_rLoop+=, 1
      f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
    }
    {
      add                   $w_sBaseLoop, $w_sBaseLoop, 8
      f16v4acc              $a6:7  
    }
    f16v4gacc             $a6:7
    {
      st64step            $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      // to flush pipeline
      f16v2gina           $azero, $azero, 0
    }
  
LRestoreUpdateXState: 
  ld32                  $w_numZMul2, $mworker_base, w_StackEntry_numZMul2/4
  add                   $w_offsetXInQ, $w_offsetXInQ, $w_numZMul2

  // we use the update w_deltaPtr to keep track of the metaInfo pointer. There
  // is an extra load for which we compensate by -2. 
  // metaInfo -> next output row entry
  {
    add                   $w_metaInfo, $w_deltaPtr, -2
    // to flush pipeline
    f16v2gina           $azero, $azero, 0
  }
  add                   $w_rLoop, $w_rLoop, -2
  st32                  $w_rLoop, $mworker_base, w_StackEntry_rBase/4
  brnzdec               $w_numXm1, LxLoop

LEndWorker:
exitz                 $mzero

// Specialisation for z = 8
LZEq8Sp:
ldz16step             $w_numYm1, $mzero, $w_metaInfo+=, 1
{
  ldz16step           $w_delta, $mzero, $w_metaInfo+=, 1
  uput                  $FP_CLR, $fp_clr_reg 
}
{
  ldb16step             $w_rData, $mzero, $w_rBase+=, 1
  mov                   $a4:5, $azeros
}
// numZMul2 / 8 - 1 is needed as an increment. numZMul2 can be divided by
// 8 because numZ must be a multiple of 4
shr                   $w_xOffIncr, $w_numZMul2, 3
add                   $w_xOffIncr, $w_xOffIncr, -1

LxLoop8Sp: 

  // delta's are byte offsets and as we are processing 8 columns of S at
  // at time load the second quad first.
  {
    rpt                   $w_numYm1, (LEndYLoop8Sp - LStartYLoop8Sp) / 8 - 1
    mov                   $a6:7, $azeros
  }  
LStartYLoop8Sp:         
    { 
      ld64                  $w_sData, $w_delta, $w_sBase, 1
      f16v8acc              $a4:7 
    }
    { 
      ldd16a64              $w_sData, $w_metaInfo++, $w_sBase, $w_delta@ 
      f16v4mul              $a6:7, $w_rData:BL, $w_sData 
    }
    { 
      ldb16step             $w_rData, $mzero, $w_rBase+=, 1
      f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
    }
LEndYLoop8Sp: 
  { 
    ld64                  $w_sData, $w_delta, $w_sBase, 1
    f16v8acc              $a4:7 
  }
  { 
    ld64                  $w_sData, $w_delta, $w_sBase, 0 
    f16v4mul              $a6:7, $w_rData:BL, $w_sData 
  }
  {
    ldz16step             $w_numYm1, $mzero, $w_metaInfo+=, 1
    f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
  }
  {
    ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1
    f16v8acc              $a4:7 
  }
  f16v2gina             $a6, $azero, 0
  {
    ldb16step             $w_rData, $mzero, $w_rBase+=, 1
    f16v2gina             $a7, $azero, 0
  }
  {
    st64step              $a6:7, $w_qBase, $w_offsetXInQ+=, 1
    f16v4gacc             $a6:7
  }
  {
    st64step              $a6:7, $w_qBase, $w_offsetXInQ+=, $w_xOffIncr
    uput                  $FP_CLR, $fp_clr_reg 
  }
  { 
    brnzdec              $w_numXm1, LxLoop8Sp
    mov                   $a4:5, $azeros
  }
  exitz                $mzero

LZEq4Sp:
ldz16step             $w_numYm1, $mzero, $w_metaInfo+=, 1
ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1
ldb16step             $w_rData, $mzero, $w_rBase+=, 1
// Can shift right by 3 because numZ is always a multiple of 4
shr                   $w_xOffIncr, $w_numZMul2, 3

LxLoop4Sp:
  // Load output entries for this output row (x dimension). 
  {
    rpt                   $w_numYm1, (LEndYLoop4Sp - LStartYLoop4Sp) / 8 - 1
    mov                   $a6:7, $azeros
  }
LStartYLoop4Sp:         
    { 
      ldd16a64              $w_sData, $w_metaInfo++, $w_sBase, $w_delta@
      f16v4acc              $a6:7  
    }
    { 
      ldb16step             $w_rData, $mzero, $w_rBase+=, 1
      f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
    }
LEndYLoop4Sp: 
  { 
    ld64                  $w_sData, $w_delta, $w_sBase, 0
    f16v4acc              $a6:7  
  }
  {
    ldz16step             $w_numYm1, $mzero, $w_metaInfo+=, 1
    f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
  }
  {
    ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1
    f16v4acc              $a6:7  
  }
  {
    ldb16step             $w_rData, $mzero, $w_rBase+=, 1
    f16v4gacc             $a6:7
  }
  {
    st64step              $a6:7, $w_qBase, $w_offsetXInQ+=, $w_xOffIncr
    uput                  $FP_CLR, $fp_clr_reg 
  }
  brnzdec              $w_numXm1, LxLoop4Sp
  exitz                $mzero

.size CODELET_NAME, . - CODELET_NAME


// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
