#ifdef __IPU__

// poprand::Dropout
#include "poprandCommon.inc"

#define poprandDropoutSvF32     __runCodelet_poprand__DropoutSupervisor___float
#define poprandDropoutSvF16     __runCodelet_poprand__DropoutSupervisor___half

.globl poprandDropoutSvF32
.type poprandDropoutSvF32, @function

.globl poprandDropoutSvF16
.type poprandDropoutSvF16, @function

.section .text.poprandDropoutSvF32
.align 4
.supervisor
poprandDropoutSvF32:
  setzi       $mWorkerEntry, poprandDropoutF32
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.align 8
.worker
poprandDropoutF32Aligned:
  nop
poprandDropoutF32:
  ld32        $mBaseIn, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_BASE_OFFSET
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_DROPOUT_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld64step    $randOut1, $mzero, $mBaseIn+=, $mWorkerIdx
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $scaleOut, $mvertex_base, $mzero, VBASE_DROPOUT_SCALE_OFFSET
  ld32        $probOut, $mvertex_base, $mzero, VBASE_DROPOUT_PROB_OFFSET
  ld64step    $randOut1, $mzero, $mBaseIn+=, 6
  {
    rpt         $mCount, ((.LpoprandDropout32_end - .LpoprandDropout32_start)/8) - 1
    f32v2rmask   $randOut, $randOut1, $probOut
  }
.LpoprandDropout32_start:
  {
    ld64step    $randOut1, $mzero, $mBaseIn+=, 6
    f32v2mul    $randOut, $scaleOut:B, $randOut
  }
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    f32v2rmask  $randOut, $randOut1, $probOut
  }
.LpoprandDropout32_end:
  // Remainder is 0, or 1
  brz         $mRemainder, .LpoprandDropoutF32_epilog
  f32mul      $randOut_0, $scaleOut, $randOut_0
  st32step    $randOut_0, $mzero, $mBaseOut+=, 6
.LpoprandDropoutF32_epilog:
  exitz       $mzero
.size poprandDropoutSvF32, .-poprandDropoutSvF32

.section .text.poprandDropoutSvF16
.align 4
.supervisor
poprandDropoutSvF16:
  setzi       $mWorkerEntry, poprandDropoutF16
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.align 8
.worker
poprandDropoutF16Aligned:
  nop
poprandDropoutF16:
  ld32        $mBaseIn, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_BASE_OFFSET
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_DROPOUT_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_DROPOUT_INPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld64step    $randOut1, $mzero, $mBaseIn+=, $mWorkerIdx
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $scaleOut, $mvertex_base, $mzero, VBASE_DROPOUT_SCALE_OFFSET
  ld32        $probOut, $mvertex_base, $mzero, VBASE_DROPOUT_PROB_OFFSET
  ld64step    $randOut1, $mzero, $mBaseIn+=, 6
  {
    rpt         $mCount, ((.LpoprandDropoutF16_end - .LpoprandDropoutF16_start)/8) - 1
    f16v4rmask  $randOut, $randOut1, $probOut
  }
.LpoprandDropoutF16_start:
  {
    ld64step    $randOut1, $mzero, $mBaseIn+=, 6
    f16v4mul    $randOut, $scaleOut:BL, $randOut
  }
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    f16v4rmask  $randOut, $randOut1, $probOut
  }
.LpoprandDropoutF16_end:
  // Remainder is 0, 1, 2 or 3
  // When non-zero must mask randOut_1 to avoid FP exceptions on past-end values
  {
    brnzdec     $mRemainder, 1f // exit now when no remainder
    mov $randOut, $azeros
  }
  exitz  $mzero
1: // $mRemainder is rem-1, randOut1=[00:00]
  {
    brnzdec     $mRemainder, .LfinalF16dropout // branch when one valid element
    sort4x16lo  $randOut1_0, $randOut_0, $azero // randOut1=[a0:00]
  }
  {
    brnzdec     $mRemainder, .LfinalF16dropout // branch when two valid
    mov         $randOut1_0, $randOut_0         // randOut1=[ab:00]
  }
  sort4x16lo    $randOut1_1, $randOut_1, $azero // randOut1=[ab:c0]
.LfinalF16dropout:
  f16v4mul    $randOut, $scaleOut:BL, $randOut1
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
  exitz       $mzero

.size poprandDropoutSvF16, .-poprandDropoutSvF16

#endif
