// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// poprand::Normal

#include "poprandCommon.inc"

#define poprandNormalSvF32     __runCodelet_poprand__NormalSupervisor___float
#define poprandNormalSvF16     __runCodelet_poprand__NormalSupervisor___half

.globl poprandNormalSvF32
.type poprandNormalSvF32, @function

.globl poprandNormalSvF16
.type poprandNormalSvF16, @function

.section .text.poprandNormalSvF32
.align 4
.supervisor
poprandNormalSvF32:
  setzi       $mWorkerEntry  , poprandNormalF32
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.align 8
.worker
poprandNormalF32:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
  ld32        $biasOut, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
  {
    rpt         $mCount, ((.LpoprandNormalF32_end - .LpoprandNormalF32_start)/8) - 1
    f32v2grand  $randOut
  }
.LpoprandNormalF32_start:
  {
    nop
    f32v2mul    $randOut, $scaleOut:B, $randOut
  }
  {
    nop
    f32v2add    $randOut, $biasOut:B, $randOut
  }
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    f32v2grand  $randOut
  }
.LpoprandNormalF32_end:
  brz         $mRemainder, .LpoprandNormalF32_epilog
  f32mul      $randOut_0, $scaleOut, $randOut_0
  f32add      $randOut_0, $biasOut, $randOut_0
  st32step    $randOut_0, $mzero, $mBaseOut+=, 1
.LpoprandNormalF32_epilog:
  exitz       $mzero
.size poprandNormalSvF32   , .-poprandNormalSvF32

.section .text.poprandNormalSvF16
.align 4
.supervisor
poprandNormalSvF16:
  setzi       $mWorkerEntry, poprandNormalF16
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.align 8
.worker
poprandNormalF16:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
  {
    ld32        $biasOut, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
    f32tof16    $scaleOut, $scaleOut
  }
  f32tof16    $biasOut, $biasOut
  {
    rpt         $mCount, ((.LpoprandNormalF16_end - .LpoprandNormalF16_start)/8) - 1
    f16v2grand  $randOut_0
  }
.LpoprandNormalF16_start:
  {
    nop
    f16v2grand  $randOut_1
  }
  {
    nop
    f16v4mul    $randOut, $scaleOut:BL, $randOut
  }
  {
    nop
    f16v4add    $randOut, $biasOut:BL, $randOut
  }
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    f16v2grand  $randOut_0
  }
.LpoprandNormalF16_end:
  {
    brz         $mRemainder, .LpoprandNormalF16_epilog
    f16v2grand  $randOut_1
  }
  f16v4mul    $randOut, $scaleOut:BL, $randOut
  f16v4add    $randOut, $biasOut:BL, $randOut
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandNormalF16_epilog:
  exitz       $mzero
.size poprandNormalSvF16, .-poprandNormalSvF16

#endif
