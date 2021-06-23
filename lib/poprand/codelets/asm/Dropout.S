// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// poprand::Dropout
#include "poprandCommon.inc"
#include "poplar/AvailableVTypes.h"

#define poprandDropoutF32     __runCodelet_poprand__Dropout___float
#define poprandDropoutF16     __runCodelet_poprand__Dropout___half

.globl poprandDropoutF32
.type poprandDropoutF32, @function

.globl poprandDropoutF16
.type poprandDropoutF16, @function

DEF_STACK_USAGE 0 poprandDropoutF32
.section .text.poprandDropoutF32

.align 8
.worker
poprandDropoutF32:
#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16       $mBaseIn, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_BASE_OFFSET/2
  shl         $mBaseIn, $mBaseIn, 3   // from scaled64 pointer to full pointer
  ldz16       $mBaseOut, $mzero, $mvertex_base, VBASE_DROPOUT_OUTPUT_BASE_OFFSET/2
  shl         $mBaseOut, $mBaseOut, 3 // from scaled64 pointer to full pointer
#else
  ld32        $mBaseIn, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_BASE_OFFSET/4
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_DROPOUT_OUTPUT_BASE_OFFSET/4
#endif
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_SIZE_OFFSET/4
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld64step    $randOut1, $mzero, $mBaseIn+=, $mWorkerIdx
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ldb16       $probOut, $mvertex_base, $mzero, VBASE_DROPOUT_PROB_OFFSET_FLOAT/2
  {
    ld32        $scaleOut, $mvertex_base, $mzero, VBASE_DROPOUT_SCALE_OFFSET/4
    sort4x16lo  $probOut, $probOut, $azero
  }
  ld64step    $randOut1, $mzero, $mBaseIn+=, 6
  {
    rpt         $mCount, ((.LpoprandDropout32_end - .LpoprandDropout32_start)/8) - 1
    f32v2rmask   $randOut0, $randOut1, $probOut
  }
.LpoprandDropout32_start:
  {
    ld64step    $randOut1, $mzero, $mBaseIn+=, 6
    f32v2mul    $randOut0, $scaleOut:B, $randOut0
  }
  {
    st64step    $randOut0, $mzero, $mBaseOut+=, 6
    f32v2rmask  $randOut0, $randOut1, $probOut
  }
.LpoprandDropout32_end:
  // Remainder is 0, or 1
  brz         $mRemainder, .LpoprandDropoutF32_epilog
  f32mul      $randOut_0, $scaleOut, $randOut_0
  st32step    $randOut_0, $mzero, $mBaseOut+=, 6
.LpoprandDropoutF32_epilog:
  exitz       $mzero
.size poprandDropoutF32, .-poprandDropoutF32

DEF_STACK_USAGE 0 poprandDropoutF16
.section .text.poprandDropoutF16

.align 8
.worker
poprandDropoutF16:
#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16       $mBaseIn, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_BASE_OFFSET/2
  shl         $mBaseIn, $mBaseIn, 3 // convert to full pointer
  ldz16       $mBaseOut, $mzero, $mvertex_base, VBASE_DROPOUT_OUTPUT_BASE_OFFSET/2
  shl         $mBaseOut, $mBaseOut, 3 // convert to full pointer
#else
  ld32        $mBaseIn, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_BASE_OFFSET/4
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_DROPOUT_OUTPUT_BASE_OFFSET/4
#endif
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_SIZE_OFFSET/4
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2

  ld64step    $randOut1, $mzero, $mBaseIn+=, $mWorkerIdx
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ldb16       $scaleOut, $mvertex_base, $mzero, VBASE_DROPOUT_SCALE_OFFSET/2
  ldb16       $probOut, $mvertex_base, $mzero, VBASE_DROPOUT_PROB_OFFSET_HALF/2
  {
    ld64step    $randOut1, $mzero, $mBaseIn+=, 6
    sort4x16lo  $probOut, $probOut, $azero
  }
  {
    rpt         $mCount, ((.LpoprandDropoutF16_end - .LpoprandDropoutF16_start)/8) - 1
    f16v4rmask  $randOut0, $randOut1, $probOut
  }
.LpoprandDropoutF16_start:
  {
    ld64step    $randOut1, $mzero, $mBaseIn+=, 6
    f16v4mul    $randOut0, $scaleOut:BL, $randOut0
  }
  {
    st64step    $randOut0, $mzero, $mBaseOut+=, 6
    f16v4rmask  $randOut0, $randOut1, $probOut
  }
.LpoprandDropoutF16_end:
  // Remainder is 0, 1, 2 or 3
  // When non-zero must mask randOut_1 to avoid FP exceptions on past-end values
  brnz        $mRemainder, 1f // exit now when no remainder
  exitz  $mzero
1: // $mRemainder is [1,2,3]
  {
    add $mCount, $mRemainder, -1
    mov $randOut1, $azeros
  }
  {
    brnzdec     $mCount, .LtwoOrThree // branch when 2/3 valid elements
    sort4x16lo  $randOut1_0, $randOut_0, $azero // randOut1=[a0,00]
  }
.LdoFinalF16dmul:
  // $randOutO must contain only valid values
  f16v4mul      $randOut0, $scaleOut:BL, $randOut1
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
  exitz         $mzero

.LtwoOrThree:
  { // $mCount is rem-2 [0 1], randout1=[a0,00]
    brz  $mCount, .LdoFinalF16dmul // branch when two valid
    mov           $randOut1_0, $randOut_0         // randOut1=[ab:00]
  }
  // randOut1=[ab,00] when we fall through to here
  {
    bri .LdoFinalF16dmul
    sort4x16lo    $randOut1_1, $randOut_1, $azero // randOut1=[ab:c0]
  }

.size poprandDropoutF16, .-poprandDropoutF16

#endif
