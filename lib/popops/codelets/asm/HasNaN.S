// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// 1D supervisor and 2D vertices for NaN checking for data type half and float

#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"
#include "poplibs_support/TileConstants.hpp"

// -----------------------------------------------------------------------------

#define HAS_NAN_HALF __runCodelet_popops__HasNaN___half
#define HAS_NAN_FLOAT __runCodelet_popops__HasNaN___float
#define HAS_NAN_HALF_SUP __runCodelet_popops__HasNaNSupervisor___half
#define HAS_NAN_FLOAT_SUP __runCodelet_popops__HasNaNSupervisor___float

// Constants: Vertex state
// The first two are used for 2D and all are used for 1D
#define V_IN_ADDR        0  // word
#define V_IN_SIZE        4  // ushort/uint for 1D/2D 
#define V_LAST_WORKER    6  // uchar for 1D
#define V_EXTRAS         7  // uchar for 1D

// Register aliases    m0
#define w_inVectors    m1
#define w_numVectors   m2
#define w_inPtr        m3
#define w_numValues    m4
#define w_numx8        m5
#define w_numx4        m6
#define w_numx2        m6
#define w_retVal       m7
#define w_stride       m8
#define w_addRem       m9
#define w_lastWorker   m10
#define w_wkrId        m11
#define w_tmp          m10

#define fp_ctl_reg     a7
#define fp_clr_reg     a6


// -----------------------------------------------------------------------------


// 1D worker size calculation
.macro CALC_1D_SIZE_AND_SETUP_POINTERS  TYPE
get             $w_wkrId, $WSR
{
  and             $w_wkrId, $w_wkrId, CSR_W_WSR__CTXTID_M1__MASK
  uget            $fp_ctl_reg, $FP_CTL
}
{
  ld32            $w_inPtr, $mvertex_base, V_IN_ADDR/4
  setzi           $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT 
}
{
  ldz16           $w_numValues, $mvertex_base, V_IN_SIZE/2
  uput            $FP_CLR, $fp_clr_reg
}
{
  ldz8            $w_lastWorker, $mvertex_base, V_LAST_WORKER
  // clear exceptions
  uput            $FP_CTL, $azero  
}
cmpult          $w_addRem, $w_wkrId, $w_lastWorker
add             $w_numValues, $w_numValues, $w_addRem
ldz8            $w_addRem, $mvertex_base, V_EXTRAS
.ifc \TYPE, half
shl             $w_numValues, $w_numValues, 2
.else
shl             $w_numValues, $w_numValues, 1
.endif
cmpeq           $w_tmp, $w_wkrId, $w_lastWorker
brz             $w_tmp, LLoopStart1D_\TYPE\()
add             $w_numValues, $w_numValues, $w_addRem
LLoopStart1D_\TYPE\():
setzi           $w_stride, NUM_WORKERS
ld64step        $azeros, $mzero, $w_inPtr+=, $w_wkrId
.endm

// -----------------------------------------------------------------------------

// Instantiate a 1D worker
.macro INSTANTIATE_1D_WORKER TYPE, NAME

DEF_STACK_USAGE 0 \NAME\()
.section .text.\NAME\(), FUNCTION_IS_WORKER
.type \NAME\(), @function
.globl \NAME\()
.align 8
.worker
\NAME\():
CALC_1D_SIZE_AND_SETUP_POINTERS \TYPE
.ifc \TYPE, half
INNER_LOOP_HALF 1D
.else
INNER_LOOP_FLOAT 1D
.endif
PROPAGATE_NAN \TYPE
// restore FPCTL
uput           $FP_CTL, $fp_ctl_reg
exitz          $w_retVal
.size \NAME\(), .-\NAME\()
.endm

// -----------------------------------------------------------------------------

// instantiate a 1D supervisor
.macro INSTANTIATE_1D_SUPERVISOR CODELET_NAME, WORKER_FN

DEF_STACK_USAGE 0 \CODELET_NAME\()
.section .text.\CODELET_NAME\()
.align 4
.globl \CODELET_NAME\()
.type \CODELET_NAME\(), @function
\CODELET_NAME\():
.supervisor
setzi           $m1, \WORKER_FN\()
runall          $m1, $m0, 0
sync            TEXCH_SYNCZONE_LOCAL
br              $lr
.size \CODELET_NAME\(), . - \CODELET_NAME\()
.endm

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

// -----------------------------------------------------------------------------

// pre-vector loop: load vertex state and set FP_CTL
.macro LOAD_VERTEX_STATE_AND_SET_FPCTL TYPE_SUFFIX DIM_SUFFIX
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
brz             $w_numVectors, LExit\TYPE_SUFFIX\()_\DIM_SUFFIX\()
{
  add             $w_numVectors, $w_numVectors, -1
  // clear exceptions
  uput            $FP_CTL, $azero
}
setzi             $w_stride, 1
.endm

// -----------------------------------------------------------------------------

// code fragment that processes the inner loop for halves
.macro INNER_LOOP_HALF DIM_SUFFIX
shr             $w_numx8, $w_numValues, 3
and             $w_numValues, $w_numValues, 0x7
ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
rpt             $w_numx8, 1
  {
    ld64step        $a2:3, $mzero, $w_inPtr+=, $w_stride
    fnop
  }
  {
    ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
    f16v8absacc     $a0:3  
  }

shr             $w_numx4, $w_numValues, 2
brz             $w_numx4, LLt4H_\DIM_SUFFIX\()
{
  ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
  f16v4absacc     $a0:1
}
add             $w_numValues, $w_numValues, -4
LLt4H_\DIM_SUFFIX\():
{
  shr             $w_numx2, $w_numValues, 1
  mov             $a2:3, $azeros
}
brz             $w_numx2, LLastH_\DIM_SUFFIX\()
mov             $a2, $a0
{
  add             $w_numValues, $w_numValues, -2
  mov             $a0, $a1
}
LLastH_\DIM_SUFFIX\():
{
  brz             $w_numValues, LDecrCountH_\DIM_SUFFIX\()
  // We can unconditionally add $a2:3 it is either zero, or is set correctly
  f16v4absacc     $a2:3
}
sort4x16lo      $a2, $a0, $a0
f16v4absacc     $a2:3
LDecrCountH_\DIM_SUFFIX\():
.endm

// -----------------------------------------------------------------------------

// code fragment that processes the inner loop for halves
.macro INNER_LOOP_FLOAT DIM_SUFFIX
shr             $w_numx4, $w_numValues, 2
and             $w_numValues, $w_numValues, 0x3
ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
rpt             $w_numx4, 1
  {
    ld64step        $a2:3, $mzero, $w_inPtr+=, $w_stride
    fnop
  }
  {
    ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
    f32v4absacc     $a0:3
  }      
{
  shr             $w_numx2, $w_numValues, 1 
  mov             $a2:3, $azeros
}
brz             $w_numx2, LLastF_\DIM_SUFFIX\()
{
  ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
  f32v4absacc     $a0:3
}
add             $w_numValues, $w_numValues, -2 
LLastF_\DIM_SUFFIX\():
{
  brz             $w_numValues, LDecrCountF_\DIM_SUFFIX\()
  mov             $a1, $azero             
}
f32v4absacc     $a0:3
LDecrCountF_\DIM_SUFFIX\():
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

LOAD_VERTEX_STATE_AND_SET_FPCTL H 2D

LVectorsLoopH:
  ld32step        $w_inPtr, $mzero, $w_inVectors+=, 1
  ld32step        $w_numValues, $mzero, $w_inVectors+=, 1
  INNER_LOOP_HALF 2D
  brnzdec         $w_numVectors, LVectorsLoopH

PROPAGATE_NAN half

LExitH_2D:
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
LOAD_VERTEX_STATE_AND_SET_FPCTL F 2D

LVectorsLoopF:
  ld32step        $w_inPtr, $mzero, $w_inVectors+=, 1
  ld32step        $w_numValues, $mzero, $w_inVectors+=, 1
  INNER_LOOP_FLOAT 2D
  brnzdec         $w_numVectors, LVectorsLoopF

PROPAGATE_NAN float

LExitF_2D:
uput            $FP_CTL, $fp_ctl_reg
exitz           $w_retVal 
.size HAS_NAN_FLOAT, .-HAS_NAN_FLOAT

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// Instantiate 1D workers
INSTANTIATE_1D_WORKER half HasNaN1DHalfWorker
INSTANTIATE_1D_WORKER float HasNaN1DFloatWorker

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// Instantiate 1D supervisors
INSTANTIATE_1D_SUPERVISOR HAS_NAN_HALF_SUP HasNaN1DHalfWorker
INSTANTIATE_1D_SUPERVISOR HAS_NAN_FLOAT_SUP HasNaN1DFloatWorker

#endif
