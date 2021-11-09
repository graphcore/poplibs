// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// 1D supervisor and 2D vertices for NaN checking for data type half and float

#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"
#include "poplar/TileConstants.hpp"
#include "CommonPoplibsMacros.h.S"

// -----------------------------------------------------------------------------

#define HAS_NAN_HALF_2D __runCodelet_popops__HasNaNOrInf2D___half_false
#define HAS_NAN_FLOAT_2D __runCodelet_popops__HasNaNOrInf2D___float_false
#define HAS_NAN_HALF_1D __runCodelet_popops__HasNaNOrInf1D___half_false
#define HAS_NAN_FLOAT_1D __runCodelet_popops__HasNaNOrInf1D___float_false
#define HAS_NANORINF_HALF_2D __runCodelet_popops__HasNaNOrInf2D___half_true
#define HAS_NANORINF_FLOAT_2D __runCodelet_popops__HasNaNOrInf2D___float_true
#define HAS_NANORINF_HALF_1D __runCodelet_popops__HasNaNOrInf1D___half_true
#define HAS_NANORINF_FLOAT_1D __runCodelet_popops__HasNaNOrInf1D___float_true

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
.macro CALC_1D_SIZE_AND_SETUP_POINTERS  TYPE, CHECK_INF
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
brz             $w_tmp, LLoopStart1D_\TYPE\()_\CHECK_INF\()
add             $w_numValues, $w_numValues, $w_addRem
LLoopStart1D_\TYPE\()_\CHECK_INF\():
setzi           $w_stride, CTXT_WORKERS
ld64step        $azeros, $mzero, $w_inPtr+=, $w_wkrId
.endm

// -----------------------------------------------------------------------------

// Instantiate a 1D worker
.macro INSTANTIATE_1D_WORKER TYPE, NAME, CHECK_INF
FN_WORKER_ENTRY_POINT \NAME 8

CALC_1D_SIZE_AND_SETUP_POINTERS \TYPE \CHECK_INF
.ifc \TYPE, half
INNER_LOOP_HALF 1D \CHECK_INF
.else
INNER_LOOP_FLOAT 1D \CHECK_INF
.endif
PROPAGATE_NAN \TYPE \CHECK_INF
// restore FPCTL
uput           $FP_CTL, $fp_ctl_reg
exitz          $w_retVal
FN_SIZE \NAME
.endm


// -----------------------------------------------------------------------------

// propagate NaNs and return non-zero in $w_retVal if detected
.macro PROPAGATE_NAN TYPE, CHECK_INF
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

.ifc \CHECK_INF, true
// If NaN is present we should get a NaN. Otherwise we should get an Inf if
// at least one of the data was Inf. Convert it to a NaN by subtracting by
// itself so that we just have to deal with NaN in the check below.
f32sub          $a0, $a0, $a0
.endif

f32class        $a0, $a0
atom            $w_retVal, $a0
cmpeq           $w_retVal, $w_retVal, 2
.endm

// -----------------------------------------------------------------------------

// pre-vector loop: load vertex state and set FP_CTL
.macro LOAD_VERTEX_STATE_AND_SET_FPCTL TYPE_SUFFIX, DIM_SUFFIX, CHECK_INF
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
brz             $w_numVectors, LExit\TYPE_SUFFIX\()_\DIM_SUFFIX\()_\CHECK_INF\()
{
  add             $w_numVectors, $w_numVectors, -1
  // clear exceptions
  uput            $FP_CTL, $azero
}
setzi             $w_stride, 1
.endm

// -----------------------------------------------------------------------------

// code fragment that processes the inner loop for halves
// We don't do anything special for HALF for detecting Infs because
// we can never get an overflow by adding halves on the IPU given limited
// memory
.macro INNER_LOOP_HALF DIM_SUFFIX, CHECK_INF
.ifc \CHECK_INF, true
shr             $w_numx8, $w_numValues, 3
and             $w_numValues, $w_numValues, 0x7
ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
// Infs become NaNs and fp16 infs get converted to NaNs and are hence
// propagated.
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
brz             $w_numx4, LLt4H_\DIM_SUFFIX\()_\CHECK_INF\()
{
  ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
  f16v4absacc     $a0:1
}
add             $w_numValues, $w_numValues, -4
LLt4H_\DIM_SUFFIX\()_\CHECK_INF\():
{
  shr             $w_numx2, $w_numValues, 1
  mov             $a2:3, $azeros
}
brz             $w_numx2, LLastH_\DIM_SUFFIX\()_\CHECK_INF\()
mov             $a2, $a0
{
  add             $w_numValues, $w_numValues, -2
  mov             $a0, $a1
}
LLastH_\DIM_SUFFIX\()_\CHECK_INF\():
{
  brz             $w_numValues, LDecrCountH_\DIM_SUFFIX\()_\CHECK_INF\()
  // We can unconditionally add $a2:3 it is either zero, or is set correctly
  f16v4absacc     $a2:3
}
sort4x16lo      $a2, $a0, $a0
f16v4absacc     $a2:3

.else

// As we want to exclude Infs, we must convert first into f32.
shr             $w_numx4, $w_numValues, 2
and             $w_numValues, $w_numValues, 0x3
ld64step        $a4:5, $mzero, $w_inPtr+=, $w_stride
rpt             $w_numx4, 2
  {
    nop
    f16v2tof32      $a0:1, $a4
  }
  {
    nop
    f16v2tof32      $a2:3, $a5
  }
  {
    ld64step        $a4:5, $mzero, $w_inPtr+=, $w_stride
    f32v4absacc     $a0:3
  }
{
  shr             $w_numx2, $w_numValues, 1
  mov             $a2:3, $azeros
}
{
  brz             $w_numx2, LLastH_\DIM_SUFFIX\()_\CHECK_INF\()
  f16v2tof32      $a0:1, $a4
}
f32v4absacc     $a0:3
{
  add             $w_numValues, $w_numValues, -2
  f16v2tof32      $a0:1, $a5
}
LLastH_\DIM_SUFFIX\()_\CHECK_INF\():
{
  brz             $w_numValues, LDecrCountH_\DIM_SUFFIX\()_\CHECK_INF\()
  mov             $a1, $azero
}
f32v4absacc     $a0:3
.endif
LDecrCountH_\DIM_SUFFIX\()_\CHECK_INF\():
.endm

// -----------------------------------------------------------------------------

// code fragment that processes the inner loop for halves
.macro INNER_LOOP_FLOAT DIM_SUFFIX, CHECK_INF
shr             $w_numx4, $w_numValues, 2
{
  and             $w_numValues, $w_numValues, 0x3
  // very small value in the order of 1e-38
  // this is strictly not required when Inf are not
  // checked for.
  or              $a5, $azero, 0x00800000
}
{
  ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
  fnop
}
.ifc \CHECK_INF, true
rpt             $w_numx4, 2
  {
    ld64step        $a2:3, $mzero, $w_inPtr+=, $w_stride
    f32v2mul        $a0:1, $a5:B, $a0:1
  }
  {
    nop
    f32v2mul        $a2:3, $a5:B, $a2:3
  }
  {
    ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
    f32v4absacc     $a0:3
  }
.else
rpt             $w_numx4, 1
  {
    ld64step        $a2:3, $mzero, $w_inPtr+=, $w_stride
    fnop
  }
  {
    ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
    f32v4absacc     $a0:3
  }
.endif
{
  shr             $w_numx2, $w_numValues, 1
  mov             $a2:3, $azeros
}
{
  brz             $w_numx2, LLastF_\DIM_SUFFIX\()_\CHECK_INF\()
  f32v2mul        $a0:1, $a5:B, $a0:1
}
{
  ld64step        $a0:1, $mzero, $w_inPtr+=, $w_stride
  f32v4absacc     $a0:3
}
{
  add             $w_numValues, $w_numValues, -2
  f32v2mul        $a0:1, $a5:B, $a0:1
}
LLastF_\DIM_SUFFIX\()_\CHECK_INF\():
{
  brz             $w_numValues, LDecrCountF_\DIM_SUFFIX\()_\CHECK_INF\()
  mov             $a1, $azero
}
f32v4absacc     $a0:3
LDecrCountF_\DIM_SUFFIX\()_\CHECK_INF\():
.endm

// -----------------------------------------------------------------------------

// If interleave memory constraints are imposed then we could use f16v8absacc.
// The code is structured to move to using ld128 with minor changes even though
// we could get the same performance using a single cycle innermost loop.
.macro HAS_NAN_OR_INF_HALF NAME, CHECK_INF
FN_WORKER_ENTRY_POINT \NAME\() 8

LOAD_VERTEX_STATE_AND_SET_FPCTL H 2D \CHECK_INF

LVectorsLoopH_\CHECK_INF\():
  ld32step        $w_inPtr, $mzero, $w_inVectors+=, 1
  ld32step        $w_numValues, $mzero, $w_inVectors+=, 1
  INNER_LOOP_HALF 2D \CHECK_INF
  brnzdec         $w_numVectors, LVectorsLoopH_\CHECK_INF\()

PROPAGATE_NAN half \CHECK_INF

LExitH_2D_\CHECK_INF\():
uput            $FP_CTL, $fp_ctl_reg
exitz           $w_retVal
FN_SIZE \NAME
.endm

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// If we impose interleave memory constraints then we could use f32v4absacc.
// The code is structured to use ld128 with minor changes even though same
// performance could be acheived by using a single cycle innermost loop.

.macro HAS_NAN_OR_INF_FLOAT NAME, CHECK_INF
FN_WORKER_ENTRY_POINT \NAME 8

// Load vertex state and set up control
LOAD_VERTEX_STATE_AND_SET_FPCTL F 2D \CHECK_INF

LVectorsLoopF_\CHECK_INF\():
  ld32step        $w_inPtr, $mzero, $w_inVectors+=, 1
  ld32step        $w_numValues, $mzero, $w_inVectors+=, 1
  INNER_LOOP_FLOAT 2D \CHECK_INF
  brnzdec         $w_numVectors, LVectorsLoopF_\CHECK_INF\()

PROPAGATE_NAN float \CHECK_INF

LExitF_2D_\CHECK_INF\():
uput            $FP_CTL, $fp_ctl_reg
exitz           $w_retVal
FN_SIZE \NAME
.endm

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// Instantiate 2D workers
HAS_NAN_OR_INF_HALF HAS_NAN_HALF_2D false
HAS_NAN_OR_INF_FLOAT HAS_NAN_FLOAT_2D false
HAS_NAN_OR_INF_HALF HAS_NANORINF_HALF_2D true
HAS_NAN_OR_INF_FLOAT HAS_NANORINF_FLOAT_2D true


// Instantiate 1D MultiVertex
INSTANTIATE_1D_WORKER half HAS_NAN_HALF_1D false
INSTANTIATE_1D_WORKER float HAS_NAN_FLOAT_1D false
INSTANTIATE_1D_WORKER half HAS_NANORINF_HALF_1D true
INSTANTIATE_1D_WORKER float HAS_NANORINF_FLOAT_1D true

#endif
