#ifdef __IPU__

// poprand::TruncNormal

#include "poprandCommon.inc"

#define poprandTruncNormalSvF32  __runCodelet_poprand__TruncatedNormalSupervisor___float
#define poprandTruncNormalSvF16  __runCodelet_poprand__TruncatedNormalSupervisor___half

.globl poprandTruncNormalSvF32
.type poprandTruncNormalSvF32, @function

.globl poprandTruncNormalSvF16
.type poprandTruncNormalSvF16, @function
.worker
.macro POPRAND_TRUNCATED_NORMAL_F32
  {
    ld64        $alphaV2, $mworker_base, $mzero, (ALPHA_STACK_OFFSET/2)
    andc64      $randOut, $randOut, $maskOut
  }
  f32v2clamp  $clampOut, $randOut, $alphaV2
  f32v2cmpeq  $clampOut, $clampOut, $randOut
  and64       $randOut, $randOut, $clampOut
  {
    ld64        $trncNorm, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
    or64        $maskOut, $maskOut, $clampOut
  }
  {
    atom        $maskOut_0, $maskOut0
    f32v2mul    $randOut, $scaleOut:BL, $randOut
  }
  {
    atom        $maskOut_1, $maskOut1
    or64        $trncNorm, $trncNorm, $randOut
  }
  st64        $trncNorm, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
  {
    and         $maskOut_0, $maskOut_0, $maskOut_1
    f32v2grand  $randOut
  }
.endm

.macro POPRAND_TRUNCATED_NORMAL_F16
  {
    ld32        $alpha, $mworker_base, $mzero, ALPHA_STACK_OFFSET
    andc64      $randOut, $randOut, $maskOut
  }
  f16v4clamp  $clampOut, $randOut, $alpha
  f16v4cmpeq  $clampOut, $clampOut, $randOut
  and64       $randOut, $randOut, $clampOut
  {
    ld64        $trncNorm, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
    or64        $maskOut, $maskOut, $clampOut
  }
  {
    atom        $maskOut_0, $maskOut0
    f16v4mul    $randOut, $scaleOut:BL, $randOut
  }
  {
    atom        $maskOut_1, $maskOut1
    or64        $trncNorm, $trncNorm, $randOut
  }
  and         $maskOut_0, $maskOut_0, $maskOut_1
  {
    xnor        $maskOut_0, $maskOut_0, $mzero
    f16v2grand  $randOut_0
  }
  {
    st64        $trncNorm, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
    f16v2grand  $randOut_1
  }
.endm

.section .text.poprandTruncNormalSvF32
.align 4
.supervisor
poprandTruncNormalSvF32:
  setzi       $mWorkerEntry  , poprandTruncNormalF32
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

poprandTruncNormalF32:
.worker
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $biasOut, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
  ld32        $alpha, $mvertex_base, $mzero, VBASE_ALPHA_OFFSET;
  {
    st64        $azeros, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
    f32sub      $minusAlpha, $azero, $alpha
  }
  {
    ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
    and64       $maskOut, $maskOut, $azeros    
  }
  {
    ld32        $nIter, $mvertex_base, $mzero, VBASE_NUM_ITER_OFFSET
    f32v2grand  $randOut
  }
  st64        $alphaV2, $mworker_base, $mzero, (ALPHA_STACK_OFFSET/2)
  brz         $mCount, .LpoprandTruncatedNormalF32_start
  add         $mCount, $mCount, (-1)
.Ltruncated_normal_f32_start:
  POPRAND_TRUNCATED_NORMAL_F32
  brnz        $maskOut_0, .Ltruncated_normal_f32_save
  brnzdec     $nIter, .Ltruncated_normal_f32_start
.Ltruncated_normal_f32_save:
  {
    ld32        $nIter, $mvertex_base, $mzero, VBASE_NUM_ITER_OFFSET
    f32v2add    $trncNorm, $biasOut:B, $trncNorm
  }
  st64step    $trncNorm, $mzero, $mBaseOut+=, 6
  st64        $azeros, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
  {
    brnzdec     $mCount, .Ltruncated_normal_f32_start
    and64       $maskOut, $maskOut, $azeros
  }
.LpoprandTruncatedNormalF32_start:
  brz         $mRemainder, .LpoprandTruncatedNormalF32_epilog
  POPRAND_TRUNCATED_NORMAL_F32
  brz         $maskOut_0, .LpoprandTruncatedNormalF32_store
  brnzdec     $nIter, .LpoprandTruncatedNormalF32_start
.LpoprandTruncatedNormalF32_store:
  f32v2add    $randOut, $biasOut:B, $trncNorm
  st32step    $randOut_0, $mzero, $mBaseOut+=, 1
.LpoprandTruncatedNormalF32_epilog:
  exitz       $mzero
.size poprandTruncNormalSvF32, .-poprandTruncNormalSvF32

.section .text.poprandTruncNormalSvF16
.align 4
.supervisor
poprandTruncNormalSvF16:
  setzi       $mWorkerEntry  , poprandTruncNormalF16
  runall      $mWorkerEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

poprandTruncNormalF16:
.worker
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $alpha, $mvertex_base, $mzero, VBASE_ALPHA_OFFSET
  f16v2sub    $minusAlpha, $azero, $alpha
  {
    ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
    sort4x16hi  $alpha, $minusAlpha, $alpha
  }
  {
    st32        $alpha, $mworker_base, $mzero, (ALPHA_STACK_OFFSET)
    and64       $maskOut, $maskOut, $azeros
  }
  {
    ld32        $biasOut, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
    f32tof16    $scaleOut, $scaleOut
  }
  f32tof16    $biasOut, $biasOut
  {
    ld32        $nIter, $mvertex_base, $mzero, VBASE_NUM_ITER_OFFSET
    f16v2grand  $randOut_0
  }
  {
    st64        $azeros, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
    f16v2grand  $randOut_1
  }
  brz         $mCount, .LpoprandTruncatedNormalF16_start
  add         $mCount, $mCount, (-1)
.Ltruncated_normal_f16_start:
  POPRAND_TRUNCATED_NORMAL_F16
  brz         $maskOut_0, .Ltruncated_normal_f16_save
  brnzdec     $nIter, .Ltruncated_normal_f16_start
.Ltruncated_normal_f16_save:
  {
    ld32        $nIter, $mvertex_base, $mzero, VBASE_NUM_ITER_OFFSET
    f16v4add    $trncNorm, $biasOut:BL, $trncNorm
  }
  st64step    $trncNorm, $mzero, $mBaseOut+=, 6
  st64        $azeros, $mworker_base, $mzero, (TRUNC_NORMAL_STACK_OFFSET/2)
  {
    brnzdec     $mCount, .Ltruncated_normal_f16_start
    and64       $maskOut, $maskOut, $azeros
  }
.LpoprandTruncatedNormalF16_start:
  brz         $mRemainder, .LpoprandTruncatedNormalF16_epilog
  POPRAND_TRUNCATED_NORMAL_F16
  brz         $maskOut_0, .LpoprandTruncatedNormalF16_store
  brnzdec     $nIter, .LpoprandTruncatedNormalF16_start
.LpoprandTruncatedNormalF16_store:
  f16v4add    $randOut, $biasOut:BL, $trncNorm
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandTruncatedNormalF16_epilog:
  exitz       $mzero
.size poprandTruncNormalSvF16, .-poprandTruncNormalSvF16

#endif
