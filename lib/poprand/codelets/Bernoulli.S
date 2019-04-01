#ifdef __IPU__

// poprand::Bernoulli

#include "poprandConst.h"

#define poprandBernoulliSvF32     __runCodelet_poprand__BernoulliSupervisor___float
#define poprandBernoulliSvF16     __runCodelet_poprand__BernoulliSupervisor___half
#define poprandBernoulliSvInt     __runCodelet_poprand__BernoulliSupervisor___int

.globl poprandBernoulliSvF32
.type poprandBernoulliSvF32      , @function

.globl poprandBernoulliSvF16
.type poprandBernoulliSvF16      , @function

.globl poprandBernoulliSvInt
.type poprandBernoulliSvInt      , @function

.section .text.poprandBernoulliSvF32
.align 4

poprandBernoulliSvF32:
  setzi        $mWorkerEntry  , poprandBernoulliF32
  POPRAND_SUPERVISOR $mWorkerEntry
poprandBernoulliF32:
  {
    bri          poprandBernoulli32;
    or           $fpOne0        , $azero                , 0x3F800000
  }
.size poprandBernoulliSvF32   , .-poprandBernoulliSvF32

.section .text.poprandBernoulliSvF16
.align 4

poprandBernoulliSvF16:
  setzi        $mWorkerEntry  , poprandBernoulliF16
  POPRAND_SUPERVISOR $mWorkerEntry
poprandBernoulliF16:
  POPRAND_LOAD_OUTPUT_BASE $mBaseOut $mInSize
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  {
    ld32         $probOut       , $mvertex_base         , $mzero            , VBASE_PROB_OFFSET;
    setzi        $fpOne0        , 0x3C00
  }
  f16v4add     $fpOneVec      , $fpOne0:BL            , $azeros
.align 8
  {
    rpt          $mCount        , ((.LpoprandBernoulliF16_end - .LpoprandBernoulliF16_start)/8) - 1;
    f16v4rmask   $randOut       , $fpOneVec             , $probOut
  }
.LpoprandBernoulliF16_start:
  {
    st64step     $randOut       , $mzero                , $mBaseOut+=       , 6;
    f16v4rmask   $randOut       , $fpOneVec             , $probOut
  }
.LpoprandBernoulliF16_end:
  brz          $mRemainder    , .LpoprandBernoulliF16_epilog
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandBernoulliF16_epilog:
  exitz          $mzero
.size poprandBernoulliSvF16   , .-poprandBernoulliSvF16

.section .text.poprandBernoulliSvInt
.align 4
poprandBernoulliSvInt:
  setzi        $mWorkerEntry  , poprandBernoulliInt
  POPRAND_SUPERVISOR $mWorkerEntry
poprandBernoulliInt:
  {
    bri          poprandBernoulli32;
    setzi        $fpOne0        , 0x1
  }
.size poprandBernoulliSvInt   , .-poprandBernoulliSvInt

.section .text.poprandBernoulli32
poprandBernoulli32:
  POPRAND_LOAD_OUTPUT_BASE $mBaseOut $mInSize
  {
    ld64step $randOut1    , $mzero, $mBaseOut+=      , $mWorkerIdx;
    urand64  $azeros
  }
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  {
    ld32         $probOut       , $mvertex_base         , $mzero            , VBASE_PROB_OFFSET;
    or           $fpOne1        , $fpOne0               , $azero
  }
.align 8
  {
    rpt          $mCount        , ((.LpoprandBernoulli32_end - .LpoprandBernoulli32_start)/8) - 1;
    f32v2rmask   $randOut       , $fpOneVec             , $probOut
  }
.LpoprandBernoulli32_start:
  {
    st64step     $randOut       , $mzero                , $mBaseOut+=       , 6;
    f32v2rmask   $randOut       , $fpOneVec             , $probOut
  }
.LpoprandBernoulli32_end:
  brz          $mRemainder    , .LpoprandBernoulli32_epilog
  st32step     $randOut_0     , $mzero                , $mBaseOut+=       , 1;
.LpoprandBernoulli32_epilog:
  exitz          $mzero
.size poprandBernoulli32, .-poprandBernoulli32

#endif
