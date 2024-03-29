// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "workDivision.h.S"
#include "CommonPoplibsMacros.h.S"

// vertex state (byte increments)
#define VERTEX_BROADCAST_IN1_PTR_OFFSET 0
#define VERTEX_BROADCAST_IN1_COUNT_OFFSET 4
#define VERTEX_BROADCAST_OUT_PTR_OFFSET 8
#define VERTEX_BROADCAST_IN2_PTR_OFFSET 12

#define VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET 4
#define VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET 8

// Register aliases

// Worker registers
// integer variables
#define mscratch m0
#define in1Count m2
#define in1 m3
#define out m4
#define mLoops m5
#define in2 m6
#define outIncr m8
#define workerIdM1 m9
#define outPtrConst m10

// float variables
#define inData a0
#define inData2 a1
#define inPair a0:1

#define epsilon a2
#define epsilon2 a3
#define epsilonPair a2:3

#define outData a4
#define outData2 a5
#define outPair a4:5

// Name mangling for use in macros
#define VERTEX __runCodelet_popops__BroadcastScalar\EXT\()___popops__expr__BinaryOpType__\OP_TYPES
#define VERTEX_INPLACE __runCodelet_popops__BroadcastScalar\EXT\()InPlace___popops__expr__BinaryOpType__\OP_TYPES

//******************************************************************************
// Compute macro for inverse standard deviation to variance, with half in,
// half or float out
.macro DO_INV_STD_DEV_TO_VARIANCE FLOAT_OUT LABEL_NUM
  {ld32step $inData, $mzero, $in1+=, 6
   mov $epsilon2, $epsilon}
  f16v2tof32 $inPair, $inData
  // Calculate the first result without -eplison or casting to half for the output
  // pre -loop:
  // out = (1 / (in * in)) - epsilon
  f32v2mul $inPair, $inPair, $inPair
  f32oox $inData, $inData

  {rpt $mLoops, (2f - 1f) / 8 - 1
   f32oox $inData2, $inData2}
1:
.if \FLOAT_OUT
  {ld32step $inData, $mzero, $in1+=, 6
   f32v2sub $outPair, $inPair, $epsilonPair}
  {st64step $outPair, $mzero, $out+=, 6
   f16v2tof32 $inPair, $inData}
 .else
  {nop
   f32v2sub $outPair, $inPair, $epsilonPair}
  {ld32step $inData, $mzero, $in1+=, 6
   f32v2tof16 $outData, $outPair}
  {st32step $outData, $mzero, $out+=, 6
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
  st64step $outPair, $mzero, $out+=, 6
.else
  // Cast and store the final loop output
  f32v2tof16 $outData, $outPair
  st32step $outData, $mzero, $out+=, 6
.endif

.LfinalCheck\LABEL_NUM:
  // Here we have done all groups of 2 halves for every worker, no overread.
  // Use the worker which is pointing to the next output to process the last 1
  // (if needed).

   // Is there a final 1 ?
  and $mscratch, $in1Count, 1
  brz $mscratch, 3f

  // Advance outPtrConst and compare to this worker's pointer to decide if it
  // needs to do the last one
  andc $mscratch, $in1Count, 1
.ifeq \FLOAT_OUT
  ldb16step $azero, $mzero, $outPtrConst+=, $mscratch
.else
  ld32step $azero, $mzero, $outPtrConst+=, $mscratch
.endif
  cmpeq $mscratch, $out, $outPtrConst
  brz $mscratch, 3f

  // Deal with the last 1
  // out = (1 / (in * in)) - epsilon
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
.endm

//******************************************************************************
// Compute macro for variance to inverse standard deviation with float in,
// half out
.macro DO_VARIANCE_TO_INV_STD_DEV LABEL_NUM
  {ld64step $inPair, $mzero, $in1+=, 6
   mov $epsilon2, $epsilon}
  // Calculate the first result without casting to half for the output
  // pre -loop:
  // out = (1 / sqrt(in + epsilon))
  f32v2add $inPair, $inPair, $epsilonPair
  f32oorx $outData, $inData

  {rpt $mLoops, (2f - 1f) / 8 - 1
   f32oorx $outData2, $inData2}
1:
  {ld64step $inPair, $mzero, $in1+=, 6
   f32v2tof16 $outData, $outPair}
  {st32step $outData, $mzero, $out+=, 6
   f32v2add $inPair, $inPair, $epsilonPair}
  {nop
   f32oorx $outData, $inData}
  {nop
   f32oorx $outData2, $inData2}
2:
  // Cast and store the final loop output
  f32v2tof16 $outData, $outPair
  st32step $outData, $mzero, $out+=, 6

.LfinalCheck\LABEL_NUM:
  // Here we have done all groups of 2 halves for every worker, no overread.
  // Use the worker which is pointing to the next half to process the last 1
  // (if needed).
  // Is there a final 1 ?
  and $mscratch, $in1Count, 1
  brz $mscratch, 3f
  // Advance outPtrConst and compare to this worker's pointer to decide if it
  // needs to do the last one
  andc $mscratch, $in1Count, 1
  ldb16step $azero, $mzero, $outPtrConst+=, $mscratch
  cmpeq $mscratch, $out, $outPtrConst
  brz $mscratch, 3f

  // Deal with the last 1
  // out = (1 / (in * in)) - epsilon
  ld32 $inData, $mzero, $in1, 0
  f32add $inData, $inData, $epsilon
  f32oorx $outData, $inData
  {ldb16 $inData2, $mzero, $out, 1
   f32tof16 $outData, $outData}
  // Combine with 16 bits of data in the output to write back
  roll16 $outData, $outData, $inData2
  st32 $outData, $mzero, $out, 0

3:
.endm


//******************************************************************************
// Macro to read vertex state, provide MultiVertex work division, and then call
// variance to inverse standard deviation or inverse standard deviaiton to variance
.macro INSTANTIATE_VERTEX EXT OP_TYPES FLOAT_IN FLOAT_OUT TO_VARIANCE

FN_WORKER_ENTRY_POINT VERTEX_INPLACE

// If float is being output there is no inPlace option
.ifeq \FLOAT_OUT

  ld32 $in1, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET/4
  ld32 $in2, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
  mov $out, $in1
  bri common\@
.endif
FN_SIZE VERTEX_INPLACE

FN_WORKER_ENTRY_POINT VERTEX 8
  ld32 $in1, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $out, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4

common\@:
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // Loops for this worker: divide by 12 and round up based on workerID
  DIVIDE_BY_WORKER $in1Count $workerIdM1 $mscratch $mLoops LOG2_FLOAT_ATOM_SIZE
  // Keep the unmodified out pointer to use to decide which worker does the
  // remainder elements
  mov $outPtrConst, $out

  // Dummy loads to advance the pointers for each worker.  32 bit incr for
  // a half operand, 64 bit incr for a float operand (input or output)
  // epsilon is the same type as the input but may need conversion to float
.if \FLOAT_IN
  ld64step $inPair, $mzero, $in1+=, $workerIdM1
  ld32 $epsilon, $mzero, $in2, 0
.else
  ldb16 $epsilon, $mzero, $in2, 0
  {ld32step $inData, $mzero, $in1+=, $workerIdM1
   f16tof32 $epsilon, $epsilon}
.endif

.if \FLOAT_OUT
  ld64step $inPair, $mzero, $out+=, $workerIdM1
.else
  ld32step $inData, $mzero, $out+=, $workerIdM1
.endif
  // Decrement as the loop is unrolled, and branch if no loops needed
  brnzdec $mLoops, 1f
  bri .LfinalCheck\@
1:
  // Insert a macro to do the maths and loop
  // The strange \@ parameter is to create labels in that macro with the same
  // macro index as this "calling" macro

.if \TO_VARIANCE
  DO_INV_STD_DEV_TO_VARIANCE \FLOAT_OUT \@
.else
  DO_VARIANCE_TO_INV_STD_DEV \@
.endif

  exitz $mzero

FN_SIZE VERTEX

.endm
//******************************************************************************
// Create the vertices from the macros above

INSTANTIATE_VERTEX 1D INV___STD___DEV___TO___VARIANCE_half 0 0 1

INSTANTIATE_VERTEX 2Types1D INV___STD___DEV___TO___VARIANCE_half_float 0 1 1
INSTANTIATE_VERTEX 2Types1D VARIANCE___TO___INV___STD___DEV_float_half 1 0 0

#endif
