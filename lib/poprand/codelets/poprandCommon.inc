#ifdef __IPU__
#ifndef POPRAND_CONST_H
#define POPRAND_CONST_H

#include "poplibs_support/TileConstants.hpp"

#define VBASE_OUTPUT_BASE_OFFSET          0
#define VBASE_OUTPUT_SIZE_OFFSET          1
#define VBASE_PROB_OFFSET                 2
#define VBASE_OFFSET_OFFSET               2
#define VBASE_SCALE_OFFSET                3
#define VBASE_SHIFT_OFFSET                4
#define VBASE_ALPHA_OFFSET                4
#define VBASE_NUM_ITER_OFFSET             5

// All as byte offsets
#define VBASE_DROPOUT_INPUT_BASE_OFFSET   0  // scaled64
#define VBASE_DROPOUT_OUTPUT_BASE_OFFSET  2  // scaled64
#define VBASE_DROPOUT_INPUT_SIZE_OFFSET   4  // input and output size bytes
#define VBASE_DROPOUT_PROB_OFFSET_HALF    10  // probability (unsigned short)
#define VBASE_DROPOUT_SCALE_OFFSET_HALF   8

#define VBASE_DROPOUT_PROB_OFFSET_FLOAT   12  // probability (unsigned short)
#define VBASE_DROPOUT_SCALE_OFFSET_FLOAT  8   // scale

#define ALPHA_STACK_OFFSET                0
#define TRUNC_NORMAL_STACK_OFFSET         2

// worker variables
#define mBaseOut               m0

#define mInSize                m1

#define mRemainder             m2

#define mCount                 m3

#define mModifier              m4
#define mOutShiftR             m4

#define mWorkerIdx             m5
#define mRandOut               m5

#define mScale                 m6
#define nIter                  m6

#define mLastWorker            m7
#define mOffset                m7
#define maskOut_0              m7

#define mOutShiftL             m8
#define maskOut_1              m8

#define mLoadCount             m9
#define mPrngState             m9
#define mOutShift              m9

#define mBaseIn                m10
#define mWarmUp                m10

#define mQuotient              m11
#define mShiftCorr             m11

#define prngState0             a0
#define randOut_0              a0

#define prngState1             a1
#define randOut_1              a1

#define prngState              a0:1
#define randOut                a0:1
#define randOut0               a0:1

#define minusAlpha             a2
#define prngStateV             a2
#define randOut1_0             a2

#define alpha                  a3
#define randOut1_1             a3

#define clampOut               a2:3
#define alphaV2                a2:3
#define randOut1               a2:3
#define trncNorm               a2:3

#define scaleOut               a4

#define biasOut                a5
#define probOut                a5

#define fpOne0                 a6
#define fpHalf                 a6
#define maskOut0               a6

#define fpOne1                 a7
#define maskOut1               a7

#define fpOneVec               a6:7
#define maskOut                a6:7

// supervisor variables
#define mWorkerEntry           m6


// Split work such that a grain of 8 bytes is assigned to a worker. Grains are
// interleaved between workers
.macro POPRAND_GET_INTERLEAVED_WORK_SPLIT inSize count remainder sh
  setzi       $mQuotient, 0xAAAB;
  mul         $mQuotient, $mQuotient, \inSize
  shr         $mQuotient, $mQuotient, (18 + \sh)
  shr         \count, \inSize, \sh   // Total number of loads
  mul         \remainder, $mQuotient, CTXT_WORKERS
  sub         \remainder, \count, \remainder
  get         $mWorkerIdx, $WSR
  and         $mWorkerIdx, $mWorkerIdx, CSR_W_WSR__CTXTID_M1__MASK
  cmpult      $mLoadCount, $mWorkerIdx, \remainder
  add         \count, $mLoadCount, $mQuotient
  cmpeq       $mLoadCount, $mWorkerIdx, \remainder
  and         \remainder, \inSize, ((1 << \sh) - 1)
  mul         \remainder, $mLoadCount, \remainder
.endm


// Deal with a remainder which is less than a grain size of 8 bytes
.macro POPRAND_STORE_LAST_WORKER_F16 remainder
  cmpult      $mLoadCount, \remainder, 3
  brnz        $mLoadCount, .LpoprandF16_v2_store
  st32step    $randOut_0, $mzero, $mBaseOut+=, 1;
  {
    bri         .LpoprandF16_store16;
    or          $randOut_0, $randOut_1, $azero
  }
.LpoprandF16_v2_store:
  cmpult      $mLoadCount, \remainder, 2
  brz         $mLoadCount, .LpoprandF16_store32
.LpoprandF16_store16:
  ldb16       $randOut_1, $mzero, $mBaseOut, 1
  sort4x16lo  $randOut_0, $randOut_0, $randOut_1
.LpoprandF16_store32:
  st32step     $randOut_0, $mzero, $mBaseOut+=, 1
.endm

#endif
#endif
