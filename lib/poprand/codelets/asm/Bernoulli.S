// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// poprand::Bernoulli

#include "poprandCommon.inc"

#define poprandBernoulliSvF32     __runCodelet_poprand__BernoulliSupervisor___float
#define poprandBernoulliSvF16     __runCodelet_poprand__BernoulliSupervisor___half
#define poprandBernoulliSvInt     __runCodelet_poprand__BernoulliSupervisor___int

.globl poprandBernoulliSvF32
.type poprandBernoulliSvF32, @function

.globl poprandBernoulliSvF16
.type poprandBernoulliSvF16, @function

.globl poprandBernoulliSvInt
.type poprandBernoulliSvInt, @function

.section .text.poprandBernoulliSvF32
.align 4
.supervisor

poprandBernoulliSvF32:
  setzi       $mWorkerEntry, poprandBernoulliF32
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.worker
poprandBernoulliF32:
  {
    bri         poprandBernoulli32
    or          $fpOne0, $azero, 0x3F800000
  }
.size poprandBernoulliSvF32, .-poprandBernoulliSvF32

.section .text.poprandBernoulliSvF16
.align 4
.supervisor

poprandBernoulliSvF16:
  setzi       $mWorkerEntry, poprandBernoulliF16
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.align 8
poprandBernoulliF16Aligned:
.worker
nop
poprandBernoulliF16:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  {
    ld32        $probOut, $mvertex_base, $mzero, VBASE_PROB_OFFSET
    setzi       $fpOne0, 0x3C00
  }
  f16v4add    $fpOneVec, $fpOne0:BL, $azeros
  {
    rpt         $mCount, ((.LpoprandBernoulliF16_end - .LpoprandBernoulliF16_start)/8) - 1
    f16v4rmask   $randOut, $fpOneVec, $probOut
  }
.LpoprandBernoulliF16_start:
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    f16v4rmask  $randOut, $fpOneVec, $probOut
  }
.LpoprandBernoulliF16_end:
  brz         $mRemainder, .LpoprandBernoulliF16_epilog
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandBernoulliF16_epilog:
  exitz       $mzero
.size poprandBernoulliSvF16, .-poprandBernoulliSvF16

.section .text.poprandBernoulliSvInt
.align 4
.supervisor
poprandBernoulliSvInt:
  setzi       $mWorkerEntry, poprandBernoulliInt
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

poprandBernoulliInt:
.worker
  {
    bri         poprandBernoulli32
    setzi       $fpOne0, 0x1
  }
.size poprandBernoulliSvInt, .-poprandBernoulliSvInt

.section .text.poprandBernoulli32
.align 8
.worker
poprandBernoulli32:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  {
    ld32        $probOut, $mvertex_base, $mzero, VBASE_PROB_OFFSET
    or          $fpOne1, $fpOne0, $azero
  }
  {
    rpt         $mCount, ((.LpoprandBernoulli32_end - .LpoprandBernoulli32_start)/8) - 1
    f32v2rmask  $randOut, $fpOneVec, $probOut
  }
.LpoprandBernoulli32_start:
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    f32v2rmask  $randOut, $fpOneVec, $probOut
  }
.LpoprandBernoulli32_end:
  brz         $mRemainder, .LpoprandBernoulli32_epilog
  st32step    $randOut_0, $mzero, $mBaseOut+=, 1
.LpoprandBernoulli32_epilog:
  exitz       $mzero
.size poprandBernoulli32, .-poprandBernoulli32

#endif
