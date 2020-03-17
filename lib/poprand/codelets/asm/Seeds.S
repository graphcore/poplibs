// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// poprand::SetSeedSupervisor

#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define poprandSvSetSeed     __runCodelet_poprand__SetSeedSupervisor

// SetSeed : Word offsets
#define VBASE_SEED_OFFSET                0   // ONE_PTR
#define VBASE_SEED_MODIFIER_USER_OFFSET  1
#define VBASE_SEED_MODIFIER_HW_OFFSET    2
#define MTOA_SCRATCH_OFFSET              0

// Number of 64-bits read from PRNG for warm-up after seed is set
#define WARMUP_RPT_COUNT                 10

//Supervisor registers
#define mWrkrEntry  m1

// Worker registers
#define mModifier   m4
#define mWorkerIdx  m5
#define mPrngState  m9

#define prngState0  a0
#define prngState1  a1
#define prngState   a0:1

DEF_STACK_USAGE 0 poprandSvSetSeed
.globl poprandSvSetSeed
.type poprandSvSetSeed, @function

.section .text.poprandSvSetSeed

.align 8
.worker
poprandSetSeed:
  ld32        $mModifier, $mzero, $mvertex_base, VBASE_SEED_MODIFIER_HW_OFFSET
  get         $mWorkerIdx, $WSR
  and         $mWorkerIdx, $mWorkerIdx, CSR_W_WSR__CTXTID_M1__MASK
  or          $mModifier, $mModifier, $mWorkerIdx
  ld32        $prngState1, $mzero, $mvertex_base, VBASE_SEED_MODIFIER_USER_OFFSET
  st32        $mModifier, $mworker_base, $mzero, MTOA_SCRATCH_OFFSET
  {
    ld32        $prngState0, $mworker_base, $mzero, MTOA_SCRATCH_OFFSET
    uput        $PRNG_1_1, $prngState1
  }
  {
    ld32        $mPrngState, $mvertex_base, VBASE_SEED_OFFSET
    uput        $PRNG_1_0, $prngState0
  }
  ld64        $prngState, $mzero, $mPrngState, 0
  uput        $PRNG_0_0, $prngState0
  {
    rpt         WARMUP_RPT_COUNT, ((.LpoprandSetSeed_warmup_end - .LpoprandSetSeed_warmup_start)/8) - 1
    uput        $PRNG_0_1, $prngState1
  }
.LpoprandSetSeed_warmup_start:
    {
      nop
      urand64   $azeros
    }
.LpoprandSetSeed_warmup_end:
  exitz       $mzero


poprandSvSetSeed:
  .supervisor
  setzi       $mWrkrEntry, poprandSetSeed
  runall      $mWrkrEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.size poprandSvSetSeed, .-poprandSvSetSeed

#endif
