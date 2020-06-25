// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix.
// 
// The data R is not linearly accessed and is accessed via
// a table of entries
//
// This is used to compute the Gradients wrt activations
// where the transpose is implicitly done during the 
// computation of the sparse-dense matrix multiplication.

#ifdef __IPU__
#include "SparseDenseMatMulGradAElementWise.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulGradAElementWise___half_float

// =============================================================================

.extern zeroDenseOutFloat

// =============================================================================

// worker stack
#define w_StackEntry_numZDiv8              0
#define w_StackEntry_sBase                 4
#define w_StackSize                        (w_StackEntry_sBase + 4)

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
#define w_zEq8                             m4
#define w_zEq4                             m4

#define w_numZDiv8                         m3
#define w_numZRem                          m7
#define w_numZDiv4                         m3
#define w_finalRem                         m3

#define w_rData                            a0
#define w_sDataL                           a2
#define w_sData                            a2:3

#define fp_clr_reg                         a1

DEF_STACK_USAGE w_StackSize elemwiseSparseDenseMultiplyGradA
.section ".text.elemwiseSparseMultiplyGradA", FUNCTION_IS_WORKER
.type elemwiseSparseDenseMultiplyGradA, @function
.align 8
.worker
// worker code

elemwiseSparseDenseMultiplyGradA:
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
ldz16step             $mzero, $mzero, $w_sBase+=, $w_offsetZ
ld32step              $mzero, $mzero, $w_qBase+=, $w_offsetZ

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

ld32                  $w_rLoop, $mvertex_base, W_R_BASE/4

LxLoop:	
  // Each output row in has entries which always offset from the &s[offsetZ].
  ld32                  $w_sBaseLoop, $mworker_base, w_StackEntry_sBase/4

  // Load output entries for this output row (x dimension). 
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 1
  shl                   $w_offsetXInQ, $w_offsetXInQ, 1
  ldz16step             $w_numY, $mzero, $w_metaInfo+=, 1
  add                   $w_numY, $w_numY, -1
  // metaInfo -> offset of column entries in 'y' dimension 
  mov                   $w_qBaseLoop, $w_qBase

  // Check if there are any multiples of 8 to process. If not, jump straight to
  // process remainder.
  ld32                  $w_numZDiv8, $mworker_base, w_StackEntry_numZDiv8/4
  brneg                 $w_numZDiv8, LzRem

LzLoop8:	    
    // we need to reuse the same entries in R for all the same output row
    // and for any z dimension. So reload pointer to offsets in S.
    mov                   $w_deltaPtr, $w_metaInfo
    // we need to multply the whole Z dimension entries by the same sparse
    // entry in R
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
    {
      ldd16b16              $w_rData, $w_deltaPtr++, $w_rLoop, $w_delta@ 
      mov                   $a4:5, $azeros
    }
    // delta's are byte offsets and as we are processing 8 columns of S at
    // at time load the second quad first.  
    {
      rpt                   $w_numY, (LEndYLoop8 - LStartYLoop8) / 8 - 1
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
        ldd16b16              $w_rData, $w_deltaPtr++, $w_rLoop, $w_delta@ 
        f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
      }
LEndYLoop8:	
    { 
      ld64                  $w_sData, $w_delta, $w_sBaseLoop, 1
      f16v8acc              $a4:7 
    }
    { 
      ld64                  $w_sData, $w_delta, $w_sBaseLoop, 0 
      f16v4mul              $a6:7, $w_rData:BL, $w_sData 
    }
    {
      ld64                  $a0:1, $w_offsetXInQ, $w_qBaseLoop, 0
      f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
    }
    {
      ld64                  $a2:3, $w_offsetXInQ, $w_qBaseLoop, 1
      f16v8acc              $a4:7 
    }
    {
      ld64                  $a0:1, $w_offsetXInQ, $w_qBaseLoop, 2
      f32v4acc              $a0:3
    }
    { 
      ld64                  $a2:3, $w_offsetXInQ, $w_qBaseLoop, 3
      f32v2gina             $a6:7, $azeros, 0
    }
    { 
      st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f32v2gina             $a6:7, $azeros, 0 
    }
    {
      // We have used up 8 halves of s. move to next set of columns.
      add                   $w_sBaseLoop, $w_sBaseLoop, 16
      f32v4acc              $a0:3
    }
    { 
      st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f32v2gina             $a6:7, $azeros, 0 
    }
    { 
      st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f32v2gina             $a6:7, $azeros, 0 
    }
    st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
    brnzdec               $w_numZDiv8, LzLoop8

LzRem:	
    brz                   $w_numZRem, LRestoreUpdateXState
    shr                   $w_numZDiv4, $w_numZRem, 2
    brz                   $w_numZDiv4, LzRemLt4
	
    mov                   $w_deltaPtr, $w_metaInfo
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
    ldd16b16              $w_rData, $w_deltaPtr++, $w_rLoop, $w_delta@ 
    {
      rpt                   $w_numY, (LEndYLoop4 - LStartYLoop4) / 8 - 1
      mov                   $a6:7, $azeros
    }
LStartYLoop4:	        
      { 
        ldd16a64              $w_sData, $w_deltaPtr++, $w_sBaseLoop, $w_delta@
        f16v4acc              $a6:7  
      }
      { 
        ldd16b16              $w_rData, $w_deltaPtr++, $w_rLoop, $w_delta@ 
        f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
      }
LEndYLoop4:	
    { 
      ld64                  $w_sData, $w_delta, $w_sBaseLoop, 0
      f16v4acc              $a6:7  
    }
    {
      ld64                  $a0:1, $w_offsetXInQ, $w_qBaseLoop, 0
      f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
    }
    {
      ld64                  $a2:3, $w_offsetXInQ, $w_qBaseLoop, 1
      f16v4acc              $a6:7  
    }
    {
      add                   $w_sBaseLoop, $w_sBaseLoop, 8
      f32v4acc              $a0:3
    }
    f32v2gina             $a6:7, $azeros, 0
    { 
      st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f32v2gina             $a6:7, $azeros, 0 
    }
    st64step              $a6:7, $w_offsetXInQ, $w_qBaseLoop+=, 1
	
LzRemLt4:
    and                   $w_finalRem, $w_numZRem, 0x3
    brz                   $w_finalRem, LRestoreUpdateXState
    add                   $w_finalRem, $w_finalRem, -1

LzRemLoop:
    mov                   $w_deltaPtr, $w_metaInfo
    ldz16step             $w_delta, $mzero, $w_deltaPtr+=, 1
    ldd16b16              $w_rData, $w_deltaPtr++, $w_rLoop, $w_delta@ 
    {
      rpt                   $w_numY, (LEndYLoopRem - LStartYLoopRem) / 8 - 1
      mov                   $a6:7, $azeros
    }
LStartYLoopRem:	        
      { 
        ldd16b16              $w_sDataL, $w_deltaPtr++, $w_sBaseLoop, $w_delta@
        f16v4acc              $a6:7
      }
      { 
        ldd16b16              $w_rData, $w_deltaPtr++, $w_rLoop, $w_delta@ 
        f16v2mul              $a6,  $w_rData:BL, $w_sDataL 
      }
LEndYLoopRem:	
    {
      ldb16              $w_sDataL, $w_delta, $w_sBaseLoop, 0
      f16v4acc              $a6:7
    }
    {
      add                   $w_sBaseLoop, $w_sBaseLoop, 2
      f16v2mul              $a6,  $w_rData:BL, $w_sDataL 
    }
    {
      ld32                  $a0, $w_offsetXInQ, $w_qBaseLoop, 0
      f16v4acc              $a6:7
    }
    f32v2gina             $a6:7, $azeros, 0
    f32add               $a6, $a6, $a0
    {
      st32step              $a6, $w_offsetXInQ, $w_qBaseLoop+=, 1
      f32v2gina             $a6:7, $azeros, 0
    }
    brnzdec               $w_finalRem, LzRemLoop

LRestoreUpdateXState:	
  // we use the update w_deltaPtr to keep track of the metaInfo pointer. There
  // is an extra load for which we compensate by -2. 
  // metaInfo -> next output row entry
  mov                   $w_metaInfo, $w_deltaPtr
  brnzdec               $w_numXm1, LxLoop

LEndWorker:
exitz                 $mzero

// Specialisation for z = 8
// TODO: We could potentially save by keeping numY - 1 in the output
// entries and unrolling the loop below. This needs to be balanced against
// the requirements for the GradW pass.
// Also we could use the unused slots in the loop (just as is done for the
// z = 4 specialisation)
LZEq8Sp:
  ldz16                 $w_numY, $mzero, $w_metaInfo, 1
  add                   $w_numY, $w_numY, -1
  ldz16                 $w_delta, $mzero, $w_metaInfo, 2

LxLoop8Sp: 
  // Load output entries for this output row (x dimension). 
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 3
  {
    shl                   $w_offsetXInQ, $w_offsetXInQ, 1
    fnop
  }
  {
    ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 0
    mov                   $a4:5, $azeros
  }
  {
    ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 1
    f32v2gina             $azeros, $a6:7, 0
  }
  {
    ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 2
    f32v2gina             $azeros, $a6:7, 0
  }
  {
    ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 3
    f32v2gina             $azeros, $a6:7, 0
  }
  {
    // metaInfo -> offset of column entries in 'y' dimension 
    ldd16b16             $w_rData, $w_metaInfo++, $w_rBase, $w_delta@ 
    f32v2gina             $azeros, $a6:7, 0
  }
  // delta's are byte offsets and as we are processing 8 columns of S at
  // at time load the second quad first.
  {
    rpt                   $w_numY, (LEndYLoop8Sp - LStartYLoop8Sp) / 8 - 1
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
      ldd16b16             $w_rData, $w_metaInfo++, $w_rBase, $w_delta@ 
      f16v4mul             $a4:5,  $w_rData:BL, $w_sData 
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
    ldz16                 $w_numY, $mzero, $w_metaInfo, 1
    f16v4mul              $a4:5,  $w_rData:BL, $w_sData 
  }
  {
    add                   $w_numY, $w_numY, -1
    f16v8acc              $a4:7 
  }
  {
    ldz16                 $w_delta, $mzero, $w_metaInfo, 2
    f32v2gina             $a0:1, $azeros, 0
  }
  {
    st64                  $a0:1, $w_offsetXInQ, $w_qBase, 0
    f32v2gina             $a0:1, $azeros, 0
  }
  {
    st64                  $a0:1, $w_offsetXInQ, $w_qBase, 1
    f32v2gina             $a0:1, $azeros, 0
  }
  {
    st64                  $a0:1, $w_offsetXInQ, $w_qBase, 2
    f32v2gina             $a0:1, $azeros, 0
  }
  st64                 $a0:1, $w_offsetXInQ, $w_qBase, 3
  brnzdec              $w_numXm1, LxLoop8Sp
  exitz                $mzero


// TODO: We could potentially save by keeping numY - 1 in the output
// entries and unrolling the loop below. This needs to be balanced against
// the requirements for the GradW pass.

LZEq4Sp:
  // Load output entries for this output row (x dimension). 
  ldz16                 $w_numY, $mzero, $w_metaInfo, 1
  add                   $w_numY, $w_numY, -1
  ldz16                 $w_delta, $mzero, $w_metaInfo, 2
LxLoop4Sp:
  ldz16step             $w_offsetXInQ, $mzero, $w_metaInfo+=, 3
  shl                   $w_offsetXInQ, $w_offsetXInQ, 1
  ld64                  $a4:5, $w_offsetXInQ, $w_qBase, 0
  ld64                  $a6:7, $w_offsetXInQ, $w_qBase, 1
  {
    ldd16b16              $w_rData, $w_metaInfo++, $w_rBase, $w_delta@
    f32v4acc              $a4:7
  }
  {
    rpt                   $w_numY, (LEndYLoop4Sp - LStartYLoop4Sp) / 8 - 1
    mov                   $a6:7, $azeros
  }
LStartYLoop4Sp:         
    { 
      ldd16a64              $w_sData, $w_metaInfo++, $w_sBase, $w_delta@
      f16v4acc              $a6:7  
    }
    { 
      ldd16b16              $w_rData, $w_metaInfo++, $w_rBase, $w_delta@
      f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
    }
LEndYLoop4Sp: 
  { 
    ld64                  $w_sData, $w_delta, $w_sBase, 0
    f16v4acc              $a6:7  
  }
  {
    // overread
    ldz16                 $w_numY, $mzero, $w_metaInfo, 1
    f16v4mul              $a6:7,  $w_rData:BL, $w_sData 
  }
  {
    add                   $w_numY, $w_numY, -1
    f16v4acc              $a6:7  
  }
  {
    // overread
    ldz16                 $w_delta, $mzero, $w_metaInfo, 2
    f32v2gina             $a6:7, $azeros, 0
  }
  {
    st64                  $a6:7, $w_offsetXInQ, $w_qBase, 0
    f32v2gina             $a6:7, $azeros, 0
  }
  st64                 $a6:7, $w_offsetXInQ, $w_qBase, 1
  brnzdec              $w_numXm1, LxLoop4Sp
  exitz                $mzero

.size elemwiseSparseDenseMultiplyGradA, . - elemwiseSparseDenseMultiplyGradA

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id.

// Instantiate supervisor codelet
SPARSE_MATMUL_ELEM_SUPERVISOR CODELET_NAME half elemwiseSparseDenseMultiplyGradA

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
