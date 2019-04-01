#ifdef __IPU__

// poprand::Dropout
#include "poprandConst.h"

#define poprandDropoutSvF32     __runCodelet_poprand__DropoutSupervisor___float
#define poprandDropoutSvF16     __runCodelet_poprand__DropoutSupervisor___half

.macro POPRAND_DROPOUT_LOAD_ARGS
  ld32         $mBaseIn       , $mzero                , $mvertex_base     , VBASE_DROPOUT_INPUT_BASE_OFFSET;
  ld32         $mBaseOut      , $mzero                , $mvertex_base     , VBASE_DROPOUT_OUTPUT_BASE_OFFSET;
  ld32         $mInSize       , $mzero                , $mvertex_base     , VBASE_DROPOUT_INPUT_SIZE_OFFSET;
.endm

.globl poprandDropoutSvF32
.type poprandDropoutSvF32      , @function

.globl poprandDropoutSvF16
.type poprandDropoutSvF16      , @function

.section .text.poprandDropoutSvF32
.align 4
poprandDropoutSvF32:
  setzi        $mWorkerEntry  , poprandDropoutF32
  POPRAND_SUPERVISOR $mWorkerEntry
poprandDropoutF32:
  POPRAND_DROPOUT_LOAD_ARGS
  {
    ld64step $randOut1    , $mzero, $mBaseIn+=       , $mWorkerIdx;
    urand64  $azeros
  }
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld32         $scaleOut      , $mvertex_base         , $mzero            , VBASE_DROPOUT_SCALE_OFFSET
  ld32         $probOut       , $mvertex_base         , $mzero            , VBASE_DROPOUT_PROB_OFFSET
  ld64step     $randOut       , $mzero                , $mBaseIn+=        , 6;
.align 8
  {
    rpt          $mCount        , ((.LpoprandDropout32_end - .LpoprandDropout32_start)/8) - 1;
    f32v2mul     $randOut       , $scaleOut:B           , $randOut
  }
.LpoprandDropout32_start:
  {
    ld64step     $randOut1      , $mzero                , $mBaseIn+=        , 6;
    f32v2rmask   $randOut       , $randOut              , $probOut
  }
  {
    st64step     $randOut       , $mzero                , $mBaseOut+=       , 6;
    f32v2mul     $randOut       , $scaleOut:B           , $randOut1
  }
.LpoprandDropout32_end:
  // Remainder is 0, 1, 2 or 3
  {
    brz          $mRemainder    , .LpoprandDropoutF32_epilog
    f32v2rmask   $randOut       , $randOut              , $probOut
  }
  st32step     $randOut_0     , $mzero                , $mBaseOut+=       , 6
.LpoprandDropoutF32_epilog:
  exitz        $mzero
.size poprandDropoutSvF32   , .-poprandDropoutSvF32

.section .text.poprandDropoutSvF16
.align 4
poprandDropoutSvF16:
  setzi        $mWorkerEntry  , poprandDropoutF16
  POPRAND_SUPERVISOR $mWorkerEntry
poprandDropoutF16:
  POPRAND_DROPOUT_LOAD_ARGS
  {
    ld64step $randOut1    , $mzero, $mBaseIn+=       , $mWorkerIdx;
    urand64  $azeros
  }
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld32         $scaleOut      , $mvertex_base         , $mzero            , VBASE_DROPOUT_SCALE_OFFSET
  ld32         $probOut       , $mvertex_base         , $mzero            , VBASE_DROPOUT_PROB_OFFSET
  ld64step     $randOut       , $mzero                , $mBaseIn+=        , 6;
.align 8
  {
    rpt          $mCount        , ((.LpoprandDropoutF16_end - .LpoprandDropoutF16_start)/8) - 1;
    f16v4mul     $randOut       , $scaleOut:BL          , $randOut
  }
.LpoprandDropoutF16_start:
  {
    ld64step     $randOut1      , $mzero                , $mBaseIn+=        , 6;
    f16v4rmask   $randOut       , $randOut              , $probOut
  }
  {
    st64step     $randOut       , $mzero                , $mBaseOut+=       , 6;
    f16v4mul     $randOut       , $scaleOut:BL          , $randOut1
  }
.LpoprandDropoutF16_end:
  {
    brz          $mRemainder    , .LpoprandDropoutF16_epilog
    f16v4rmask   $randOut       , $randOut              , $probOut
  }
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandDropoutF16_epilog:
  exitz          $mzero
.size poprandDropoutSvF16   , .-poprandDropoutSvF16

#endif
