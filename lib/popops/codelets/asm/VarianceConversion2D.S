// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

// vertex state (byte increments)
#define VERTEX_BROADCAST_IN1_PTR_OFFSET 0
#define VERTEX_BROADCAST_IN1_COUNT_OFFSET 4
#define VERTEX_BROADCAST_OUT_PTR_OFFSET 8
#define VERTEX_BROADCAST_IN2_PTR_OFFSET 12

#define VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET 4
#define VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET 8

// Register aliases

// integer variables
#define in1Ptr m0
#define outPtr m1
#define in1Count m2
#define in1 m3
#define out m4
#define mLoops m5
#define in2Ptr m6
#define dataLength m7
#define outIncr m8

// float variables
#define inData a0
#define inData2 a1
#define inPair a0:1
#define epsilonPair a2:3
#define epsilon a2
#define epsilon2 a3
#define outData a4
#define outData2 a5
#define outPair a4:5

// Name mangling for use in macros
#define VERTEX __runCodelet_popops__BroadcastScalar\EXT\()___popops__expr__BroadcastOpType__\OP_TYPES
#define VERTEX_INPLACE __runCodelet_popops__BroadcastScalar\EXT\()InPlace___popops__expr__BroadcastOpType__\OP_TYPES

//******************************************************************************
// Macro to read half, compute INV_STD_DEV_TO_VARAINCE and output as either
// float or half.  All arithmetic to be done as float
.macro INSTANTIATE_VERTEX_ISD_TO_VAR EXT OP_TYPES FLOAT_OUT

.global VERTEX
.type VERTEX, @function
.section .text.VERTEX

.align 8
 nop //Alignment for the repeat body below
VERTEX:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4
  // To share code, set an increment used when fetching out - which is a ONE_PTR
  // in this case - using 1 * 32 bits
  setzi $outIncr, 1

// If float is being output there is no inPlace option
.ifeq \FLOAT_OUT
  bri common\@

.global VERTEX_INPLACE
.type VERTEX_INPLACE, @function

VERTEX_INPLACE:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
  // To share code, set an increment used when fetching inOut - which is a SPAN
  // in this case - using 2 * 32 bits
  mov $outPtr, $in1Ptr
  setzi $outIncr, 2
.endif

common\@:
  ldb16 $epsilon, $mzero, $in2Ptr, 0
  // Decrement as using brnzdec
  {sub  $in1Count,$in1Count, 1
   f16tof32 $epsilon, $epsilon}
   mov $epsilon2, $epsilon

outerLoop\@:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $dataLength, $mzero, $in1Ptr+=, 1
  ld32step $out, $mzero, $outPtr+=, $outIncr
  // process 2 at once in the loop
  shr $mLoops, $dataLength, 1
  // If no pairs, then go to check if there is a last one
  brz $mLoops, finalCheck\@

  ld32step $inData, $mzero, $in1+=, 1
  // One less, as the loop is unrolled to help with bundling
  {sub $mLoops, $mLoops, 1
   f16v2tof32 $inPair, $inData}
  // Calculate the first result without - epsilon or casting to half for the output
  // pre -loop:
  // (1 / (in * in)) - epsilon
  f32v2mul $inPair, $inPair, $inPair
  f32oox $inData, $inData

  {rpt $mLoops, (2f - 1f) / 8 - 1
   f32oox $inData2, $inData2}
1:
.if \FLOAT_OUT
  {ld32step $inData, $mzero, $in1+=, 1
   f32v2sub $outPair, $inPair, $epsilonPair}
  {st64step $outPair, $mzero, $out+=, 1
   f16v2tof32 $inPair, $inData}
.else
  {nop
   f32v2sub $outPair, $inPair, $epsilonPair}
  {ld32step $inData, $mzero, $in1+=, 1
   f32v2tof16 $outData, $outPair}
  {st32step $outData, $mzero, $out+=, 1
   f16v2tof32 $inPair, $inData}
.endif

  {nop
   f32v2mul $inPair, $inPair, $inPair}
  {nop
   f32oox $inData, $inData}
  {nop
   f32oox $inData2, $inData2}
2:
  f32v2sub $outPair, $inPair, $epsilonPair
.if \FLOAT_OUT
  st64step $outPair, $mzero, $out+=, 1
.else
  // Cast and store the final loop output
  f32v2tof16 $outData, $outPair
  st32step $outData, $mzero, $out+=, 1
.endif
finalCheck\@:
  // Is there a final 1 ?
  and $dataLength, $dataLength, 1
  brz $dataLength, 3f
  // Deal with the last 1
  // (1 / (in * in)) - epsilon
  ldb16 $inData, $mzero, $in1, 0
  f16tof32 $inData, $inData
  f32mul $inData, $inData, $inData
  f32oox $inData, $inData
  f32sub $outData, $inData, $epsilon
.ifeq \FLOAT_OUT
  {ldb16 $inData2, $mzero, $out, 1
   f32tof16 $outData, $outData}
  // Combine with 16 bits of data in the output to write back
  roll16 $outData, $outData, $inData2
.endif
  st32 $outData, $mzero, $out, 0
3:
  brnzdec $in1Count, outerLoop\@

  exitz $mzero

.size VERTEX, .-VERTEX

.endm

//******************************************************************************
// Macro to read float, compute VARIANCE_TO_INV_STD_DEV and output half.
// All arithmetic to be done as float.
.macro INSTANTIATE_VERTEX_VAR_TO_ISD EXT OP_TYPES

.global VERTEX
.type VERTEX, @function
.global VERTEX_INPLACE
.type VERTEX, @function
.section .text.VETEX_INPLACE

.align 8
VERTEX:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4

  ld32 $epsilon, $mzero, $in2Ptr, 0
  // Decrement as using brnzdec
  {sub  $in1Count,$in1Count, 1
   mov $epsilon2, $epsilon}

outerLoop\@:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $dataLength, $mzero, $in1Ptr+=, 1
  ld32step $out, $mzero, $outPtr+=, 1
  // process 2 at once in the loop
  shr $mLoops, $dataLength, 1
  // If no pairs, then go to check if there is a last one
  brz $mLoops, finalCheck\@

  ld64step $inPair, $mzero, $in1+=, 1

  // Calculate the first result without casting to half for the output
  // pre -loop:
  // (1 / sqrt(in + epsilon))

  // One less, as the loop is unrolled to help with bundling
  {sub $mLoops, $mLoops, 1
   f32v2add $inPair, $inPair, $epsilonPair}
  f32oorx $outData, $inData

  {rpt $mLoops, (2f - 1f) / 8 - 1
   f32oorx $outData2, $inData2}
1:
  {ld64step $inPair, $mzero, $in1+=, 1
   f32v2tof16 $outData, $outPair}
  {st32step $outData, $mzero, $out+=, 1
   f32v2add $inPair, $inPair, $epsilonPair}
  {nop
   f32oorx $outData, $inData}
  {nop
   f32oorx $outData2, $inData2}
2:
  // Cast and store the final loop output
  f32v2tof16 $outData, $outPair
  st32step $outData, $mzero, $out+=, 1

finalCheck\@:
  // Is there a final 1 ?
  and $dataLength, $dataLength, 1
  brz $dataLength, 3f
  // Deal with the last 1
  // (1 / (in * in)) - epsilon
  ld32 $inData, $mzero, $in1, 0
  f32add $inData, $inData, $epsilon
  f32oorx $outData, $inData
  {ldb16 $inData2, $mzero, $out, 1
   f32tof16 $outData, $outData}
  // Combine with 16 bits of data in the output to write back
  roll16 $outData, $outData, $inData2
  st32 $outData, $mzero, $out, 0

3:
  brnzdec $in1Count, outerLoop\@

  exitz $mzero

.size VERTEX, .-VERTEX

.endm
//******************************************************************************
// Create the vertices from the macros above

INSTANTIATE_VERTEX_ISD_TO_VAR 2DData INV___STD___DEV___TO___VARIANCE_half 0
INSTANTIATE_VERTEX_ISD_TO_VAR 2Types2DData INV___STD___DEV___TO___VARIANCE_half_float 1


INSTANTIATE_VERTEX_VAR_TO_ISD 2Types2DData VARIANCE___TO___INV___STD___DEV_float_half

#endif
