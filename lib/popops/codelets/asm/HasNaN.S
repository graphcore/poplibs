// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"
#include "poplibs_support/TileConstants.hpp"

// -----------------------------------------------------------------------------

#define HAS_NAN_HALF __runCodelet_popops__HasNaN___half
#define HAS_NAN_FLOAT __runCodelet_popops__HasNaN___float

// Constants: Vertex state
#define V_IN_ADDR      0
#define V_IN_SIZE      4

// Register aliases    m0
#define w_inVectors    m1
#define w_numVectors   m2
#define w_inPtr        m3
#define w_numValues    m4
#define w_numx8        m5
#define w_numx4        m6
#define w_numx2        m6
#define w_retVal       m7

#define fp_ctl_reg     a7
#define fp_clr_reg     a6

// -----------------------------------------------------------------------------

// propagate NaNs and return non-zero in $w_retVal if detected
.macro PROPAGATE_NAN TYPE
// propagate NaNs
f32v2gina       $a0:1, $azeros, 0
// In the worst case we could have +Inf in the output and addition won't change
// that
f32v2gina       $a2:3, $azeros, 0
f32v2add        $a0:1, $a0:1, $a2:3
.ifc \TYPE, half
f32v2gina       $a2:3, $azeros, 0
f32v2add        $a0:1, $a0:1, $a2:3
f32v2gina       $a2:3, $azeros, 0
f32v2add        $a0:1, $a0:1, $a2:3
.endif
f32add          $a0, $a0, $a1
f32class        $a0, $a0
atom            $w_retVal, $a0
cmpeq           $w_retVal, $w_retVal, 2
.endm

// pre-vector loop: load vertex state and set FP_CTL
.macro LOAD_VERTEX_STATE_AND_SET_FPCTL SUFFIX
// Load the vertex state.
{
  ld32            $w_inVectors, $mvertex_base, $mzero, V_IN_ADDR/4
  uget            $fp_ctl_reg, $FP_CTL
}
{
  ld32            $w_numVectors, $mvertex_base, $mzero, V_IN_SIZE/4
  setzi           $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  mov             $w_retVal, $mzero
  uput            $FP_CLR, $fp_clr_reg
}
brz             $w_numVectors, LExit\SUFFIX\()
{
  add             $w_numVectors, $w_numVectors, -1
  // clear exceptions
  uput            $FP_CTL, $azero
}
.endm

// -----------------------------------------------------------------------------

// If interleave memory constraints are imposed then we could use f16v8absacc.
// The code is structured to move to using ld128 with minor changes even though
// we could get the same performance using a single cycle innermost loop.
.globl HAS_NAN_HALF
.type HAS_NAN_HALF, @function

DEF_STACK_USAGE 0 HAS_NAN_HALF
.section .text.HAS_NAN_HALF
.align 8

HAS_NAN_HALF:

LOAD_VERTEX_STATE_AND_SET_FPCTL H

LVectorsLoopH:
  ld32step        $w_inPtr, $mzero, $w_inVectors+=, 1
  ld32step        $w_numValues, $mzero, $w_inVectors+=, 1
  shr             $w_numx8, $w_numValues, 3
  and             $w_numValues, $w_numValues, 0x7
  ld64step        $a0:1, $mzero, $w_inPtr+=, 1
  {
    rpt             $w_numx8, 1
    fnop
  }
    {
      ld64step        $a2:3, $mzero, $w_inPtr+=, 1
      fnop
    }
    {
      ld64step        $a0:1, $mzero, $w_inPtr+=, 1
      f16v8absacc     $a0:3
    }

  shr             $w_numx4, $w_numValues, 2
  brz             $w_numx4, LLt4H
  {
    ld64step        $a0:1, $mzero, $w_inPtr+=, 1
    f16v4absacc     $a0:1
  }
  add             $w_numValues, $w_numValues, -4
LLt4H:
  {
    shr             $w_numx2, $w_numValues, 1
    mov             $a2:3, $azeros
  }
  brz             $w_numx2, LLastH
  mov             $a2, $a0
  {
    add             $w_numValues, $w_numValues, -2
    mov             $a0, $a1
  }
LLastH:
  {
    brz             $w_numValues, LDecrCountH
    // We can unconditionally add $a2:3 it is either zero, or is set correctly
    f16v4absacc     $a2:3
  }
  sort4x16lo      $a2, $a0, $a0
  f16v4absacc     $a2:3

LDecrCountH:
  brnzdec         $w_numVectors, LVectorsLoopH

PROPAGATE_NAN half

LExitH:
uput            $FP_CTL, $fp_ctl_reg
exitz           $w_retVal 
.size HAS_NAN_HALF, .-HAS_NAN_HALF

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// If we impose interleave memory constraints then we could use f32v4absacc.
// The code is structured to use ld128 with minor changes even though same
// performance could be acheived by using a single cycle innermost loop.

.globl HAS_NAN_FLOAT
.type HAS_NAN_FLOAT, @function

DEF_STACK_USAGE 0 HAS_NAN_FLOAT
.section .text.HAS_NAN_FLOAT
.align 8

HAS_NAN_FLOAT:

// Load vertex state and set up control
LOAD_VERTEX_STATE_AND_SET_FPCTL F

LVectorsLoopF:
  ld32step        $w_inPtr, $mzero, $w_inVectors+=, 1
  ld32step        $w_numValues, $mzero, $w_inVectors+=, 1
  shr             $w_numx4, $w_numValues, 2
  and             $w_numValues, $w_numValues, 0x3
  ld64step        $a0:1, $mzero, $w_inPtr+=, 1
  {
    rpt             $w_numx4, 1
    fnop
  }
    {
      ld64step        $a2:3, $mzero, $w_inPtr+=, 1
      fnop
    }
    {
      ld64step        $a0:1, $mzero, $w_inPtr+=, 1
      f32v4absacc     $a0:3
    }      
  {
    shr             $w_numx2, $w_numValues, 1 
    mov             $a2:3, $azeros
  }
  brz             $w_numx2, LLastF
  {
    ld64step        $a0:1, $mzero, $w_inPtr+=, 1
    f32v4absacc     $a0:3
  }
  add             $w_numValues, $w_numValues, -2 
LLastF:
  {
    brz             $w_numValues, LDecrCountF
    mov             $a1, $azero             
  }
  f32v4absacc     $a0:3
LDecrCountF:
  brnzdec         $w_numVectors, LVectorsLoopF

PROPAGATE_NAN float

LExitF:
uput            $FP_CTL, $fp_ctl_reg
exitz           $w_retVal 
.size HAS_NAN_FLOAT, .-HAS_NAN_FLOAT

#endif
