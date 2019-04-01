#ifdef __IPU__

// poprand::Uniform
// Instead of y=a*U~[0,1]+b we do y=a*U~[-1/2,1/2]+(b+a/2)

#define poprandUniformF32     __runCodelet_poprand__Uniform___float
#define poprandUniformF16     __runCodelet_poprand__Uniform___half
#define poprandUniformInt     __runCodelet_poprand__Uniform___int

#include "poprandConst.h"

#define poprandUniformSvF32     __runCodelet_poprand__UniformSupervisor___float
.globl poprandUniformSvF32
.type poprandUniformSvF32      , @function

.section .text.poprandUniformSvF32
.align 4

poprandUniformSvF32:
  setzi        $mWorkerEntry  , poprandUniformF32
  POPRAND_SUPERVISOR $mWorkerEntry
poprandUniformF32:
  POPRAND_LOAD_OUTPUT_BASE $mBaseOut $mInSize
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  ld32           $scaleOut      , $mvertex_base         , $mzero            , VBASE_SCALE_OFFSET
  ld32           $biasOut       , $mvertex_base         , $mzero            , VBASE_OFFSET_OFFSET
.align 8
  {
    rpt            $mCount        , ((.LpoprandUniformF32_end - .LpoprandUniformF32_start)/8) - 1;
    urand64        $randOut
  }
.LpoprandUniformF32_start:
  {
    nop;
    f32v2sufromui  $randOut       , $randOut
  }
  {
    nop;
    f32v2mul       $randOut       , $scaleOut:B           , $randOut
  }
  {
    nop;
    f32v2add       $randOut       , $biasOut:B            , $randOut
  }
  {
    st64step       $randOut       , $mzero                , $mBaseOut+=       , 6;
    urand64        $randOut
  }
.LpoprandUniformF32_end:
  brz            $mRemainder    , .LpoprandUniformF32_epilog
  f32v2sufromui  $randOut       , $randOut
  f32v2mul       $randOut       , $scaleOut:B           , $randOut
  f32v2add       $randOut       , $biasOut:B            , $randOut
  st32step       $randOut_0     , $mzero                , $mBaseOut+=       , 1;
.LpoprandUniformF32_epilog:
  exitz          $mzero
.size poprandUniformSvF32   , .-poprandUniformSvF32

#define poprandUniformSvF16     __runCodelet_poprand__UniformSupervisor___half
.globl poprandUniformSvF16
.type poprandUniformSvF16      , @function

.section .text.poprandUniformSvF16
.align 4

poprandUniformSvF16:
  setzi        $mWorkerEntry  , poprandUniformF16
  POPRAND_SUPERVISOR $mWorkerEntry
poprandUniformF16:
  POPRAND_LOAD_OUTPUT_BASE $mBaseOut $mInSize
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld32           $scaleOut      , $mvertex_base         , $mzero            , VBASE_SCALE_OFFSET;
  {
    ld32           $biasOut       , $mvertex_base         , $mzero            , VBASE_OFFSET_OFFSET;
    f32tof16       $scaleOut      , $scaleOut
  }
  f32tof16       $biasOut       , $biasOut
.align 8
  {
    rpt            $mCount        , ((.LpoprandUniformF16_end - .LpoprandUniformF16_start)/8) - 1;
    urand64        $randOut
  }
.LpoprandUniformF16_start:
  {
    nop;
    f16v4sufromui  $randOut      , $randOut
  }
  {
    nop;
    f16v4mul       $randOut      , $scaleOut:BL          , $randOut
  }
  {
    nop;
    f16v4add       $randOut      , $biasOut:BL           , $randOut
  }
  {
    st64step       $randOut      , $mzero                , $mBaseOut+=       , 6;
    urand64        $randOut
  }
.LpoprandUniformF16_end:
  brz            $mRemainder    , .LpoprandUniformF16_epilog
  f16v4sufromui  $randOut      , $randOut
  f16v4mul       $randOut      , $scaleOut:BL          , $randOut
  f16v4add       $randOut      , $biasOut:BL           , $randOut
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandUniformF16_epilog:
  exitz          $mzero
.size poprandUniformSvF16   , .-poprandUniformSvF16

#define poprandUniformSvInt     __runCodelet_poprand__UniformSupervisor___int
.globl poprandUniformSvInt
.type poprandUniformSvInt      , @function

.section .text.poprandUniformSvInt
.align 4

poprandUniformSvInt:
  setzi        $mWorkerEntry  , poprandUniformInt
  POPRAND_SUPERVISOR $mWorkerEntry
.align 8
poprandUniformInt:
  POPRAND_LOAD_OUTPUT_BASE $mBaseOut $mInSize
  ld32         $mScale        , $mvertex_base         , $mzero            , VBASE_SCALE_OFFSET
  brz          $mScale        , .LpoprandUniformMaxRangeInt
  {
    ld32step $maskOut0      , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 0
  {
    ld32         $mOffset       , $mvertex_base         , $mzero            , VBASE_OFFSET_OFFSET;
    urand64      $azeros
  }
  {
    ld32         $mOutShift     , $mvertex_base         , $mzero            , VBASE_SHIFT_OFFSET
    urand64      $azeros
  }
  bri          .LpoprandUniformInt_loop
.LpoprandUniformMaxRangeInt:
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
.align 8
  {
    rpt          $mCount        , ((.LpoprandUniformMaxRangeInt_end - .LpoprandUniformMaxRangeInt_start)/8) - 1;
    urand64      $randOut
  }
.LpoprandUniformMaxRangeInt_start:
  {
    st64step     $randOut       , $mzero                , $mBaseOut+=       , 6;
    urand64      $randOut
  }
.LpoprandUniformMaxRangeInt_end:
  brz            $mRemainder    , .LpoprandUniformInt_end
  st32step       $randOut_0     , $mzero                , $mBaseOut+=       , 1;
  bri          .LpoprandUniformInt_end
.LpoprandUniformInt_loop:
.align 8
  {
    rpt          $mCount        , ((.LpoprandUniformInt_end - .LpoprandUniformInt_start)/8) - 1;
    urand32      $randOut_0
  }
.LpoprandUniformInt_start:
  {
    atom         $mRandOut      , $randOut_0;
    fnop
  }
  {
    shr          $mRandOut      , $mRandOut             , 8;
    fnop
  }
  {
    mul          $mRandOut      , $mRandOut             , $mScale;
    fnop
  }
  {
    shr          $mRandOut      , $mRandOut             , $mOutShift;
    fnop
  }
  {
    add          $mRandOut      , $mRandOut             , $mOffset;
    fnop
  }
  {
    st32step     $mRandOut      , $mzero                , $mBaseOut+=       , 6;
    urand32      $randOut_0
  }
.LpoprandUniformInt_end:
  exitz        $mzero
.size poprandUniformSvInt   , .-poprandUniformSvInt

#endif
