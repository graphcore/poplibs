// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
//

#ifdef __IPU__
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "poplar/TileConstants.hpp"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__StaticSparseDenseElementWise___float_float

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

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////

// =============================================================================

// worker stack
#define w_StackEntry_rBase                 0
#define w_StackEntry_numZDiv4              4
#define w_StackEntry_sBase                 8
#define w_StackEntry_numZMul4              12
#define w_StackSize                        (w_StackEntry_numZMul4 + 4)

// worker registers
#define w_metaInfo                         m0
#define w_rBase                            m1
#define w_qBase                            m2
#define w_sBase                            m3
#define w_nzOffset                         m4
#define w_id                               m5
#define w_worklistOff                      m6
#define w_wkrInfoOffset                    m5
#define w_offsetZ                          m4 
#define w_zOff                             m4
#define w_numXm1                           m5
#define w_numZ                             m7
#define w_sBaseLoop                        m4
#define w_offsetXInQ                       m6
#define w_numY                             m8
#define w_qBaseLoop                        m9
#define w_rLoop                            m10
#define w_deltaPtr                         m1
#define w_delta                            m11

#define w_zEq4                             m10
#define w_zEq2                             m10

#define w_numZRem                          m7
#define w_numZTemp                         m3
#define w_numZDiv4                         m3
#define w_numZMul4                         m9
#define w_xOffIncr                         m9


#define w_rDataL                           a0
#define w_rDataH                           a1
#define w_rData                            a0:1

#define w_sDataL                           a2
#define w_sData                            a2:3

#define fp_clr_reg                         a1

DEF_STACK_USAGE w_StackSize CODELET_NAME
.section .text.CODELET_NAME
.type CODELET_NAME, @function
.globl CODELET_NAME
.align 8
.worker

CODELET_NAME:
ld32                  $w_metaInfo, $mvertex_base, W_METAINFO/4
ld32                  $w_rBase, $mvertex_base, W_R_BASE/4
ld32                  $w_qBase, $mvertex_base, W_Q_BASE/4
ld32                  $w_sBase, $mvertex_base, W_S_BASE/4

// This is the numZ per tile. The numZ for a thread may differ from this
ld32                  $w_numZ, $mvertex_base, W_NUM_Z/4
mul                   $w_numZMul4, $w_numZ, 4
st32                  $w_numZMul4, $mworker_base, w_StackEntry_numZMul4/4

// The number of workers is the first field
// w_metaInfo -> worker entries
get                   $w_id, $WSR
and                   $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
mul                   $w_id, $w_id, WKR_HDR_NUM_SIZE

ldz16                 $w_worklistOff, $w_id, $w_metaInfo, WKR_HDR_WKR_OFFSET/2
ldz16                 $w_nzOffset, $w_id, $w_metaInfo, WKR_HDR_SPARSE_OFFSET/2
ld32step              $mzero, $mzero, $w_rBase+=, $w_nzOffset
ldz16                 $w_offsetZ, $w_id, $w_metaInfo, WKR_HDR_OFFSET_Z/2
ld32step              $mzero, $mzero, $w_qBase+=, $w_offsetZ
ld32step              $mzero, $mzero, $w_sBase+=, $w_offsetZ
ldz16                 $w_numZ, $w_id, $w_metaInfo, WKR_HDR_NUMZ/2
ldz16                 $w_zOff, $w_id,  $w_metaInfo, WKR_HDR_ROW_OFFSET/2
ldz16                 $w_numXm1, $w_id, $w_metaInfo, WKR_HDR_NUM_ROWS/2

brnzdec               $w_numXm1, LWorkerCont
exitz                 $mzero

LWorkerCont:
ldz16step             $mzero, $mzero, $w_metaInfo+=, $w_worklistOff
mul                   $w_offsetXInQ, $w_zOff, $w_numZMul4

{
  cmpeq                 $w_zEq4, $w_numZ, 4
  setzi                 $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  brnz                  $w_zEq4, LZEq4Sp
  uput                  $FP_CLR, $fp_clr_reg 
}
cmpeq                 $w_zEq2, $w_numZ, 2
brnz                  $w_zEq2, LZEq2Sp


// save &r[sparseOffset] and &s[offsetZ] on stack. These will be update
// for different 'x' entries in the loop.
st32                  $w_rBase, $mworker_base, w_StackEntry_rBase/4
st32                  $w_sBase, $mworker_base, w_StackEntry_sBase/4

// We process 4 entries at a time if possible and handle the remaining if any.
shr                   $w_numZDiv4, $w_numZ, 2

// use of brnzdec, so subtract by 1.
add                   $w_numZDiv4,  $w_numZDiv4, -1

// we only need to know if there is a remainder. An and by 0x3 is sufficient
and                   $w_numZRem, $w_numZ, 0x3

// save on stack to avoid recomputing in loop.
st32                  $w_numZDiv4, $mworker_base, w_StackEntry_numZDiv4/4

LxLoop:	
  // Each output row in has entries which always offset from the &s[offsetZ].
  ld32                  $w_sBaseLoop, $mworker_base, w_StackEntry_sBase/4

  // Load output entries for this output row (x dimension). 
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1
  // metaInfo -> offset of column entries in 'y' dimension 
  mov                   $w_qBaseLoop, $w_qBase

  // Check if there are any multiples of 4 to process. If not, jump straight to
  // process remainder.
  ld32                  $w_numZDiv4, $mworker_base, w_StackEntry_numZDiv4/4
  brneg                 $w_numZDiv4, LzRem

LzLoop4:	    
    // we need to reuse the same entries in R for all the same output row
    // and for any z dimension. So reload pointers to R and offsets in S.
    ld32                  $w_rLoop, $mworker_base, w_StackEntry_rBase/4
    {
      mov                   $w_deltaPtr, $w_metaInfo
      fnop
    }
    // we need to multply the whole Z dimension entries by the same sparse
    // entry in R
    ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
    {
      ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
       mov                  $a4:5, $azeros
    }
    {
      rpt                   $w_numY, (LEndYLoop4 - LStartYLoop4) / 8 - 1
      mov                   $a6:7, $azeros
    }
LStartYLoop4:	       
      {
        ld64                  $a6:7, $w_delta, $w_sBaseLoop, 1
        f32v4acc              $a4:7
      } 
      {
        ldd16a64              $a4:5, $w_deltaPtr++, $w_sBaseLoop, $w_delta@ 
        f32v2mul              $a6:7, $w_rDataL:B, $a6:7
      }  
      {
        ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
        f32v2mul              $a4:5, $w_rDataL:B, $a4:5
      }
LEndYLoop4:	
    f32v4acc                $a4:7
    { 
      // We have used up 4 elements of s. move to next set of columns.
      add                   $w_sBaseLoop, $w_sBaseLoop, 16
      f32v2gina             $a6:7, $azeros, 0
    }
    {
      st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f32v2gina             $a6:7, $azeros, 0
    }
    st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
    brnzdec               $w_numZDiv4, LzLoop4

LzRem:	
    // At this point we could have a maximum of 3 elements to process. Quick
    // exit if 0.
    brz                   $w_numZRem, LRestoreUpdateXState
    and                   $w_numZTemp, $w_numZRem, 0x2
    brz                   $w_numZTemp, LzRemFinal

    // process 2 columns in dimension z
    ld32                  $w_rLoop, $mworker_base, w_StackEntry_rBase/4
    mov                   $w_deltaPtr, $w_metaInfo
    // we need to multply the whole Z dimension entries by the same sparse
    // entry in R
    ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
    // delta's are byte offsets and as we are processing 8 columns of S at
    // at time load the second quad first.
    {
      rpt                   $w_numY, (LEndYLoop2 - LStartYLoop2) / 8 - 1
      fnop
    }
LStartYLoop2:	        
      {
        ldd16a64              $w_sData, $w_deltaPtr++, $w_sBaseLoop, $w_delta@ 
        mov                   $w_rDataH, $w_rDataL
      }  
      {
        ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
        f32v2mac              $w_sData, $w_rData
      }
LEndYLoop2:	
    { 
      // We have used up 2 elements of s. move to next set of columns.
      add                   $w_sBaseLoop, $w_sBaseLoop, 8
      f32v2gina             $a6:7, $azeros, 0
    }
    st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1

LzRemFinal:
    and                   $w_numZTemp, $w_numZRem, 0x1
    brz                   $w_numZTemp, LRestoreUpdateXState
    // only one remaining
    ld32                  $w_rLoop, $mworker_base, w_StackEntry_rBase/4
    mov                   $w_deltaPtr, $w_metaInfo
    ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
    rpt                   $w_numY, (LEndYLoopRem - LStartYLoopRem) / 8 - 1
LStartYLoopRem:	        
      { 
        ldd16a32              $w_sDataL, $w_deltaPtr++, $w_sBaseLoop, $w_delta@
        fnop
      }
      { 
        ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
        f32mac                $w_sDataL, $w_rDataL
      }
LEndYLoopRem:	
    {
      nop
      f32v2gina             $a6:7, $azeros, 0
    }
    f32add                $a6, $a6, $a4
    st32step              $a6, $w_offsetXInQ, $w_qBaseLoop+=, 1

LRestoreUpdateXState:	
  ld32                  $w_numZMul4, $mworker_base, w_StackEntry_numZMul4/4
  add                   $w_offsetXInQ, $w_offsetXInQ, $w_numZMul4

  // we use the update w_deltaPtr to keep track of the metaInfo pointer. There
  // is an extra load for which we compensate by -2. 
  // metaInfo -> next output row entry
  add                   $w_metaInfo, $w_deltaPtr, -2
  add                   $w_rLoop, $w_rLoop, -4
  st32                  $w_rLoop, $mworker_base, 0
  brnzdec               $w_numXm1, LxLoop

LEndWorker:
exitz                 $mzero


// Specialisation for z = 4
LZEq4Sp: 
// numZ is guaranteed to be a multiple of 2
shr                   $w_xOffIncr, $w_numZMul4, 3 
add                   $w_xOffIncr, $w_xOffIncr, -1
ld32step              $w_rDataL, $mzero, $w_rBase+=, 1
{
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1
  mov                   $a4:5, $azeros
}
{
  ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1
  fnop
}
LZEq4SpLoop:
  // we need to multiply the whole Z dimension entries by the same sparse
  // entry in R
  {
    rpt                   $w_numY, (LEndYLoop4Sp - LStartYLoop4Sp) / 8 - 1
    mov                   $a6:7, $azeros
  }
LStartYLoop4Sp:        
    {
      ld64                  $a6:7, $w_delta, $w_sBase, 1
      f32v4acc              $a4:7
    } 
    {
      ldd16a64              $a4:5, $w_metaInfo++, $w_sBase, $w_delta@ 
      f32v2mul              $a6:7, $w_rDataL:B, $a6:7
    }  
    {
      ld32step              $w_rDataL, $mzero, $w_rBase+=, 1
      f32v2mul              $a4:5, $w_rDataL:B, $a4:5
    }
LEndYLoop4Sp: 
  {
    mov                      $w_numY, $w_delta
    f32v4acc                 $a4:7
  }
  {
    ldz16step               $w_delta, $mzero, $w_metaInfo+=, 1
    f32v2gina               $a6:7, $azeros, 0
  }
  {
    st64step                $a6:7, $w_qBase,  $w_offsetXInQ+=, 1
    f32v2gina               $a6:7, $azeros, 0
  }
  {
    st64step                $a6:7, $w_qBase, $w_offsetXInQ+=, $w_xOffIncr  
    mov                     $a4:5, $azeros
  }
  brnzdec                 $w_numXm1, LZEq4SpLoop
  exitz                   $mzero


// specialisation for z = 2
LZEq2Sp:
// numZ is guaranteed to be a multiple of 2
shr                    $w_xOffIncr, $w_numZMul4, 3    
ld32step               $w_rDataL, $mzero, $w_rBase+=, 1
ldz16step              $w_numY, $mzero, $w_metaInfo+=, 1

LZEq2SpLoop:
  // Load output entries for this output row (x dimension). 
  // we need to multply the whole Z dimension entries by the same sparse
  // entry in R
  ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1
  {
    rpt                   $w_numY, (LEndYLoop2Sp - LStartYLoop2Sp) / 8 - 1
    fnop
  }
LStartYLoop2Sp:         
    {
      ldd16a64              $w_sData, $w_metaInfo++, $w_sBase, $w_delta@ 
      mov                   $w_rDataH, $w_rDataL
    }  
    {
      ld32step              $w_rDataL, $mzero, $w_rBase+=, 1
      f32v2mac              $w_sData, $w_rData
    }
LEndYLoop2Sp: 
  {
    // move the over-read meta-info to numY
    mov                   $w_numY, $w_delta
    f32v2gina             $a6:7, $azeros, 0
  }  
  st64step              $a6:7, $w_qBase, $w_offsetXInQ+=, $w_xOffIncr
  brnzdec               $w_numXm1, LZEq2SpLoop
  exitz                 $mzero


.size CODELET_NAME, . - CODELET_NAME

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
