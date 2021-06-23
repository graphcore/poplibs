// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// poprand::Bernoulli

#include "poprandCommon.inc"

#define poprandBernoulliF32     __runCodelet_poprand__Bernoulli___float
#define poprandBernoulliF16     __runCodelet_poprand__Bernoulli___half
#define poprandBernoulliInt     __runCodelet_poprand__Bernoulli___int

.globl poprandBernoulliF32
.type poprandBernoulliF32, @function

.globl poprandBernoulliF16
.type poprandBernoulliF16, @function

.globl poprandBernoulliInt
.type poprandBernoulliInt, @function

DEF_STACK_USAGE 0 poprandBernoulliF32
.section .text.poprandBernoulliF32
.align 4

.worker
poprandBernoulliF32:
  {
    bri         poprandBernoulli32
    or          $fpOne0, $azero, 0x3F800000
  }
.size poprandBernoulliF32, .-poprandBernoulliF32

DEF_STACK_USAGE 0 poprandBernoulliF16
.section .text.poprandBernoulliF16

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
.size poprandBernoulliF16, .-poprandBernoulliF16

DEF_STACK_USAGE 0 poprandBernoulliInt
.section .text.poprandBernoulliInt
.align 4


poprandBernoulliInt:
.worker
  {
    bri         poprandBernoulli32
    setzi       $fpOne0, 0x1
  }
.size poprandBernoulliInt, .-poprandBernoulliInt

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
