// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
//

#ifdef __IPU__
#include "SparseDenseMatMulElementWise.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulElementWise___float_float

// =============================================================================

.extern zeroDenseOutFloat

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
#define w_StackSize                        (w_StackEntry_sBase + 8)

// worker registers
#define w_metaInfo                         m0
#define w_rBase                            m1
#define w_qBase                            m2
#define w_sBase                            m3
#define w_numWorkers                       m4
#define w_id                               m5
#define w_processWork                      m6
#define w_wkrInfoOffset                    m5
#define w_offsetZ                          m4 
#define w_numXm1                           m5
#define w_metaInfoOffset                   m6
#define w_numZ                             m7
#define w_sparseOffset                     m6
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

#define w_rDataL                           a0
#define w_rDataH                           a1
#define w_rData                            a0:1

#define w_sDataL                           a2
#define w_sData                            a2:3

#define fp_clr_reg                         a1

DEF_STACK_USAGE w_StackSize elemwiseSparseDenseMultiply
.section ".text.elemwiseSparseMultiply", FUNCTION_IS_WORKER
.type elemwiseSparseDenseMultiply, @function
.align 8
.worker
// worker code

elemwiseSparseDenseMultiply:
ld32                  $w_metaInfo, $mvertex_base, W_METAINFO/4
ld32                  $w_rBase, $mvertex_base, W_R_BASE/4
ld32                  $w_qBase, $mvertex_base, W_Q_BASE/4
ld32                  $w_sBase, $mvertex_base, W_S_BASE/4

// The number of workers is the first field
// w_metaInfo -> worker entries
ldz16step             $w_numWorkers, $mzero, $w_metaInfo+=, 1
get                   $w_id, $WSR
and                   $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

// There are a max of worker entries as there are number of workers
cmpult                $w_processWork, $w_id, $w_numWorkers
brz                   $w_processWork, LEndWorker

// point to this worker entry 
// w_metaInfo -> &metaInfo->workerEntries[wid]
mul                   $w_wkrInfoOffset, $w_id, Sizeof_MetaInfoWorkerEntry
add                   $w_metaInfo, $w_metaInfo, $w_wkrInfoOffset

// load worker information
ldz16                 $w_offsetZ, $w_metaInfo, MetaInfoWorkerEntry_offsetZ/2
ldz16                 $w_numXm1, $w_metaInfo, MetaInfoWorkerEntry_numXm1/2
ldz16                 $w_metaInfoOffset, $w_metaInfo, MetaInfoWorkerEntry_metaInfoOffset/2
ldz16                 $w_numZ, $w_metaInfo, MetaInfoWorkerEntry_numZ/2

// Note: metaInfoOffset points to the start of output entries reserved for this
//       worker. Utilise the fact that sparseOffset is the first entry in the
//       worker table so that we can directly jump to the worker information.
ldz16step             $w_sparseOffset, $mzero, $w_metaInfo+=, $w_metaInfoOffset

// update pointer start offsets for this worker
// The data types for r and s are the same whereas q is of accum type
ld32step              $mzero, $mzero, $w_rBase+=, $w_sparseOffset
ld32step              $mzero, $mzero, $w_sBase+=, $w_offsetZ
ld32step              $mzero, $mzero, $w_qBase+=, $w_offsetZ

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
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  // X offset in in elements
  shl                   $w_offsetXInQ, $w_offsetXInQ, 3
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
    mov                   $w_deltaPtr, $w_metaInfo

    // we need to multply the whole Z dimension entries by the same sparse
    // entry in R
    ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1

    // Load and feed partials into accumulator
    ld64                  $a4:5, $w_offsetXInQ, $w_qBaseLoop, 0
    ld64                  $a6:7, $w_offsetXInQ, $w_qBaseLoop, 1

    {
      rpt                   $w_numY, (LEndYLoop4 - LStartYLoop4) / 8 - 1
      fnop
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
    {
      mov                   $w_deltaPtr, $w_metaInfo
      mov                   $a6:7, $azeros
    }
    // we need to multply the whole Z dimension entries by the same sparse
    // entry in R
    ld32step              $w_rDataL, $mzero, $w_rLoop+=, 1
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
    ld64                  $a4:5, $w_offsetXInQ, $w_qBaseLoop, 0
    // delta's are byte offsets and as we are processing 8 columns of S at
    // at time load the second quad first.
    {
      rpt                   $w_numY, (LEndYLoop2 - LStartYLoop2) / 8 - 1
      f32v4acc              $a4:7
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
      ld32                  $a4, $w_offsetXInQ, $w_qBaseLoop, 0
      f32v2gina             $a6:7, $azeros, 0
    }
    f32add                $a6, $a6, $a4
    st32step              $a6, $w_offsetXInQ, $w_qBaseLoop+=, 1

LRestoreUpdateXState:	
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
  // Load output entries for this output row (x dimension). 
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  shl                   $w_offsetXInQ, $w_offsetXInQ, 3
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1


  // we need to multply the whole Z dimension entries by the same sparse
  // entry in R
  ld32step              $w_rDataL, $mzero, $w_rBase+=, 1
  ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1

  // Load and feed partials into accumulator
  ld64                  $a4:5, $w_offsetXInQ, $w_qBase, 0
  ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 1

  {
    rpt                   $w_numY, (LEndYLoop4Sp - LStartYLoop4Sp) / 8 - 1
    fnop
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
    add                      $w_metaInfo, $w_metaInfo, -2
    f32v4acc                 $a4:7
  }
  {
    add                     $w_rBase, $w_rBase, -4
    f32v2gina               $a6:7, $azeros, 0
  }
  {
    st64                    $a6:7, $w_offsetXInQ, $w_qBase, 0
    f32v2gina               $a6:7, $azeros, 0
  }
  st64                    $a6:7, $w_offsetXInQ, $w_qBase, 1
  brnzdec                 $w_numXm1, LZEq4Sp
  exitz                   $mzero


  // specialisation for z = 2
  LZEq2Sp:
  // Load output entries for this output row (x dimension). 
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  shl                   $w_offsetXInQ, $w_offsetXInQ, 3
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1

  // we need to multply the whole Z dimension entries by the same sparse
  // entry in R
  ld32step              $w_rDataL, $mzero, $w_rBase+=, 1
  ldz16step             $w_delta, $mzero, $w_metaInfo+=, 1 
  {
    ld64                  $a4:5, $w_offsetXInQ, $w_qBase, 0
    mov                   $a6:7, $azeros
  }  
  {
    rpt                   $w_numY, (LEndYLoop2Sp - LStartYLoop2Sp) / 8 - 1
    f32v4acc              $a4:7
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
    add                   $w_metaInfo, $w_metaInfo, -2
    f32v2gina             $a6:7, $azeros, 0
  }
  
  st64                  $a6:7, $w_offsetXInQ, $w_qBase, 0
  add                   $w_rBase, $w_rBase, -4
  brnzdec               $w_numXm1, LZEq2Sp
  exitz                 $mzero


.size elemwiseSparseDenseMultiply, . - elemwiseSparseDenseMultiply

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
ELEM_SPARSE_MATMUL CODELET_NAME float

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
