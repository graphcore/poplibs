// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// poprand::Uniform
// Instead of y=a*U~[0,1]+b we do y=a*U~[-1/2,1/2]+(b+a/2)

#include "poprandCommon.inc"

#define poprandUniformF32   __runCodelet_poprand__Uniform___float
.globl poprandUniformF32
.type poprandUniformF32, @function

DEF_STACK_USAGE 0 poprandUniformF32
.section .text.poprandUniformF32

.align 8
.worker
poprandUniformF32:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET

  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  // dummy load for post-increment
  ld64step    $randOut, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
  ld32        $biasOut, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
  {
    rpt         $mCount, ((.LpoprandUniformF32_end - .LpoprandUniformF32_start)/8) - 1
    urand64     $randOut
  }
.LpoprandUniformF32_start:
  {
    nop
    f32v2sufromui $randOut, $randOut
  }
  {
    nop
    f32v2mul    $randOut, $scaleOut:B, $randOut
  }
  {
    nop
    f32v2add    $randOut, $biasOut:B, $randOut
  }
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    urand64     $randOut
  }
.LpoprandUniformF32_end:
  brz         $mRemainder, .LpoprandUniformF32_epilog
  f32v2sufromui $randOut, $randOut
  f32v2mul    $randOut, $scaleOut:B, $randOut
  f32v2add    $randOut, $biasOut:B, $randOut
  st32step    $randOut_0, $mzero, $mBaseOut+=, 1
.LpoprandUniformF32_epilog:
  exitz       $mzero
.size poprandUniformF32, .-poprandUniformF32

#define poprandUniformF16     __runCodelet_poprand__Uniform___half
.globl poprandUniformF16
.type poprandUniformF16, @function

DEF_STACK_USAGE 0 poprandUniformF16
.section .text.poprandUniformF16

.align 8
.worker
poprandUniformF16:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 2
  ld64step    $randOut1, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
  {
    ld32        $biasOut, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
    f32tof16    $scaleOut, $scaleOut
  }
  f32tof16    $biasOut, $biasOut
  {
    rpt         $mCount, ((.LpoprandUniformF16_end - .LpoprandUniformF16_start)/8) - 1
    urand64     $randOut
  }
.LpoprandUniformF16_start:
  {
    nop
    f16v4sufromui $randOut, $randOut
  }
  {
    nop
    f16v4mul    $randOut, $scaleOut:BL, $randOut
  }
  {
    nop
    f16v4add    $randOut, $biasOut:BL, $randOut
  }
  {
    st64step    $randOut, $mzero, $mBaseOut+=, 6
    urand64     $randOut
  }
.LpoprandUniformF16_end:
  brz         $mRemainder, .LpoprandUniformF16_epilog
  f16v4sufromui $randOut, $randOut
  f16v4mul    $randOut, $scaleOut:BL, $randOut
  f16v4add    $randOut, $biasOut:BL, $randOut
  POPRAND_STORE_LAST_WORKER_F16 $mRemainder
.LpoprandUniformF16_epilog:
  exitz       $mzero
.size poprandUniformF16, .-poprandUniformF16

#define poprandUniformInt     __runCodelet_poprand__Uniform___int
.globl poprandUniformInt
.type poprandUniformInt, @function

DEF_STACK_USAGE 0 poprandUniformInt
.section .text.poprandUniformInt

.align 8
.worker
poprandUniformIntAligned:
poprandUniformInt:
  ld32        $mBaseOut, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32        $mInSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  ld32        $mScale, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
  or          $twoPow24, $mzero, 0x1000000
// Note that the max range case occurs when scale = 0 and 0 < 2^24 so we continue on after this check
  cmpult      $twoPow24, $mScale, $twoPow24
  brz         $twoPow24, .LpoprandUniformLargeRangeInt
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 1
  {
    brnz      $mScale, .LpoprandUniformInt24Bit
    or        $twoPowMinus32, $azero, 0x2f800000
  }
  ld64step    $azeros, $mzero, $mBaseOut+=, $mWorkerIdx
.align 8
.LpoprandUniformMaxRangeInt:
// Generate random integers in the full range of an int32
  {
    rpt         $mCount, ((.LpoprandUniformMaxRangeInt_end - .LpoprandUniformMaxRangeInt_start)/8) - 1
    urand64     $randOut
  }
.LpoprandUniformMaxRangeInt_start:
  {
    st64step    $randOut, $mzero, $mBaseOut+=, CTXT_WORKERS
    urand64     $randOut
  }
.LpoprandUniformMaxRangeInt_end:
  brz         $mRemainder, .LpoprandUniformMaxRangeInt_epilog
  st32step    $randOut_0, $mzero, $mBaseOut+=, 1
.LpoprandUniformMaxRangeInt_epilog:
  exitz         $mzero
.LpoprandUniformInt24Bit:
// Generate random integers with less than 24 bits of range
  {
    ld32        $scaleOut, $mvertex_base, $mzero, VBASE_SCALE_OFFSET
    urand64     $randOut
  }
  {
    ld32step    $azero, $mzero, $mBaseOut+=, $mWorkerIdx
    f32fromui32 $randOut_0, $randOut_0
  }
  f32fromui32 $randOut_1, $randOut_1
  f32fromui32 $scaleOut, $scaleOut
  {
    ld32        $mOffset, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
    f32mul      $scaleOut, $scaleOut, $twoPowMinus32
  }
  f32v2mul      $randOut, $scaleOut:B, $randOut
  f32int        $randOut_0, $randOut_0, 2
  f32int        $randOut_1, $randOut_1, 2
  f32toui32     $randOut1_1, $randOut_1
  {
    atom        $mRandOut, $randOut1_1
    f32toui32   $randOut1_0, $randOut_0
  }
  add         $mRandOut, $mRandOut, $mOffset
.align 8
.LpoprandUniformInt24Bit_loop:
  {
    rpt         $mCount, ((.LpoprandUniformInt24Bit_end - .LpoprandUniformInt24Bit_start)/8) - 1
    urand64     $randOut
  }
.LpoprandUniformInt24Bit_start:
  {
    st32step      $mRandOut, $mzero, $mBaseOut+=, 1
    f32fromui32   $randOut_0, $randOut_0
  }
  {
    atom          $mRandOut, $randOut1_0
    f32fromui32   $randOut_1, $randOut_1
  }
  {
    add           $mRandOut, $mRandOut, $mOffset
    f32v2mul      $randOut, $scaleOut:B, $randOut
  }
  // Note: These 2 rounding instructions are necessary because setting the
  // rounding mode to negative infinity is not supported by the ipu
  {
    nop
    f32int        $randOut_0, $randOut_0, 2
  }
  {
    nop
    f32int        $randOut_1, $randOut_1, 2
  }
  {
    st32step      $mRandOut, $mzero, $mBaseOut+=, 2 * CTXT_WORKERS - 1
    f32toui32     $randOut1_1, $randOut_1
  }
  {
    atom          $mRandOut, $randOut1_1
    f32toui32     $randOut1_0, $randOut_0
  }
  {
    add           $mRandOut, $mRandOut, $mOffset
    urand64       $randOut
  }
.LpoprandUniformInt24Bit_end:
  brz           $mRemainder, .LpoprandUniformInt24Bit_epilog
  st32          $mRandOut, $mzero, $mBaseOut, 1
.LpoprandUniformInt24Bit_epilog:
  exitz         $mzero
.LpoprandUniformLargeRangeInt:
// Generate random integers with range in [2^24, 2^32)
  POPRAND_GET_INTERLEAVED_WORK_SPLIT $mInSize $mCount $mRemainder 0
  setzi     $mMask, 0xFFFF
  shr       $mHighScale, $mScale, 16
  and       $mLowScale, $mScale, $mMask
  ld32step  $azero, $mzero, $mBaseOut+=, $mWorkerIdx
  ld32      $mOffset, $mvertex_base, $mzero, VBASE_OFFSET_OFFSET
// Skip to the check at the end of the loop before doing any work to avoid generating an extra element
  bri       .LpoprandUniformLargeRangeInt_end
.LpoprandUniformLargeRangeInt_start:
  atom        $mRandOut, $randOut_0
  shr         $mHighRandOut, $mRandOut, 16
  and         $mLowRandOut, $mRandOut, $mMask
  mul         $mHHOut, $mHighRandOut, $mHighScale
  mul         $mHLOut, $mHighRandOut, $mLowScale
  shr         $mHLOut, $mHLOut, 16
  mul         $mLHOut, $mLowRandOut, $mHighScale
  shr         $mLHOut, $mLHOut, 16
  add         $mHHOut, $mHHOut, $mLHOut
  add         $mHHOut, $mHHOut, $mHLOut
  add         $mHHOut, $mHHOut, $mOffset
  st32step  $mHHOut, $mzero, $mBaseOut+=, CTXT_WORKERS
.LpoprandUniformLargeRangeInt_end:
  {
    brnzdec     $mCount, .LpoprandUniformLargeRangeInt_start
    urand32     $randOut_0
  }
  exitz       $mzero
.size poprandUniformInt, .-poprandUniformInt

#endif
