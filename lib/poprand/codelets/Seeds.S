#ifdef __IPU__
// poprand::SetSeedSupervisor

#include "poplibs_support/TileConstants.hpp"

#define poprandSvSetSeed     __runCodelet_poprand__SetSeedSupervisor
#define poprandSvGetHwSeeds  __runCodelet_poprand__GetHwSeedsSupervisor
#define poprandSvSetHwSeeds  __runCodelet_poprand__SetHwSeedsSupervisor

// SetSeed : Word offsets
#define VBASE_SEED_OFFSET                0   // ONE_PTR
#define VBASE_SEED_MODIFIER_USER_OFFSET  1
#define VBASE_SEED_MODIFIER_HW_OFFSET    2
#define MTOA_SCRATCH_OFFSET              0

// GetHwSeeds : Word offsets
#define VBASE_OUT_PTR_OFFSET             0   // ONE_PTR
// SetHwSeeds : Word offsets
#define VBASE_IN_PTR_OFFSET              0   // ONE_PTR

// Number of 64-bits read from PRNG for warm-up after seed is set
#define WARMUP_RPT_COUNT                 10

//Supervisor registers
#define mWrkrEntry  m1

// Worker registers
#define mModifier   m4
#define mWorkerIdx  m5
#define mPrngState  m9
#define outPtr      m0
#define inPtr       m0

#define prngState0  a0
#define prngState1  a1
#define prngState   a0:1

.globl poprandSvSetSeed
.type poprandSvSetSeed, @function

.section .text.poprandSvSetSeed

.align 8
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
  setzi       $mWrkrEntry, poprandSetSeed
  runall      $mWrkrEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.size poprandSvSetSeed, .-poprandSvSetSeed

.globl poprandSvGetHwSeeds
.type poprandSvGetHwSeeds, @function

.section .text.poprandSvGetHwSeeds
.align 4

// Supervisor code to read Hw seeds to a vertex edge on a tile
poprandSvGetHwSeeds:
  setzi       $mWrkrEntry, poprandGetSeeds
  runall      $mWrkrEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

// Worker code
poprandGetSeeds:
  get         $mWorkerIdx, $WSR
  {
    and         $mWorkerIdx, $mWorkerIdx, CSR_W_WSR__CTXTID_M1__MASK
    uget        $a0, $PRNG_0_0
  }
  {
    shl         $mWorkerIdx, $mWorkerIdx, 4
    uget        $a1, $PRNG_0_1
  }
  {
    ld32        $outPtr, $mvertex_base, $mzero, VBASE_OUT_PTR_OFFSET
    uget        $a2, $PRNG_1_0
  }
  {
    st64        $a0:1, $outPtr, $mWorkerIdx, 0
    uget        $a3, $PRNG_1_1
  }
  st64        $a2:3, $outPtr, $mWorkerIdx, 1
  exitz       $mzero
.size poprandSvGetHwSeeds, .-poprandSvGetHwSeeds


.globl poprandSvSetHwSeeds
.type poprandSvSetHwSeeds, @function

.section .text.poprandSvSetHwSeeds
.align 4

// Supervisor code to set Hw seeds on a tile
poprandSvSetHwSeeds:
  setzi       $mWrkrEntry, poprandSetSeeds
  runall      $mWrkrEntry, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

// Worker code
poprandSetSeeds:
  get         $mWorkerIdx, $WSR
  and         $mWorkerIdx, $mWorkerIdx, CSR_W_WSR__CTXTID_M1__MASK
  shl         $mWorkerIdx, $mWorkerIdx, 4
  ld32        $inPtr, $mvertex_base, $mzero, VBASE_IN_PTR_OFFSET
  ld64        $a0:1, $inPtr, $mWorkerIdx, 0
  {
    ld64        $a2:3, $inPtr, $mWorkerIdx, 1
    uput        $PRNG_0_0, $a0
  }
  uput        $PRNG_0_1, $a1
  uput        $PRNG_1_0, $a2
  uput        $PRNG_1_1, $a3
  exitz       $mzero
.size poprandSvSetHwSeeds, .-poprandSvSetHwSeeds

#endif
