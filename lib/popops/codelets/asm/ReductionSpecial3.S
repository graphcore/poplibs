// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Specialisation 3 STRIDED_REDUCE - Overview:
// `partials` is a single edge
// `out` is a single edge
// The vertex treats partials as a 2D array, size {`numPartials`, `partialsWidth`}
// Eg, for partialsWidth = 12, numPartials = 3
// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
// 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
//
// The output will be the sum of each 'column', computed for `numOutputs` only,
// eg for `numOutputs` = 8
// 3, 6, 9, 12, 15, 18, 21, 24
//
// Constraints:
// The output must be 32bit aligned, partials must be 64bit aligned.
// Num outputs must be based on a 64-bit partials input width:
// reduce float -> half  : 2 half outputs
// reduce float -> float : 2 float outputs
// reduce half  -> half  : 4 half outputs
// reduce half  -> float : 4 float outputs
//
// Operation/speed:
// Accumulate down columns, 64 bits at once (2 floats or 4 halves), using a
// stride to skip to the next row.
// This results in an inner loop that takes 1 cycle per 64 bits processed
// (2 floats or 4 halves).

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "MathConstants.S"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

// Vertex state
#ifdef VECTOR_AVAIL_SCALED_PTR32
#define OUT_OFFSET                  0
#define IN_OFFSET                   2
#define COUNTS_AND_STRIDES_OFFSET   4
#define SCALE_OFFSET                8   
#else
#define OUT_OFFSET                  0
#define IN_OFFSET                   4
#define COUNTS_AND_STRIDES_OFFSET   8
#define SCALE_OFFSET                12   
#endif

// Offsets into the COUNTS_AND_STRIDES tensor
#define NUM_OUTPUTS_OFFSET   0
#define NUM_PARTIALS_OFFSET  2
#define PARTIALS_WIDTH_OFFSET 4
// The STRIDED_REDUCE vertex omits the NUM_OUTER_STRIDES and OUTER_STRIDE parameters.
#define NUM_OUTER_STRIDES_OFFSET 6
#define OUTER_STRIDE_OFFSET  8

// Defines when using the worker's pre-allocated scratch area
#define SCRATCH_RESULT_UPPER 0

// Register definitions
#define NUM_OUTER_STRIDES_CONST m2
#define COUNTS_AND_STRIDES      m4
#define SCRATCH                 m5
#define STRIDE                  m7
#ifdef VECTOR_AVAIL_SCALED_PTR32
#define BASE                    m8
#else
#define BASE                    mzero
#endif
#define OUT_COUNT               m9
#define NUM_OUTER_STRIDES       m10
#define OUTER_STRIDE            m11

#define RESULT_LO       a2
#define RESULT_HI       a3
#define RESULT          a2:3
#define f32v4RESULT     a0:3
#define ZAACC           a4
#define SCALE           a6
#define CONST           a7

// Name mangling
#define REDUCE_FLOAT(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_float_\OUT_TYPE\()_\UPDATE\()_popops__ReductionSpecialisation__##specialisation

#define REDUCE_HALF(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_half_\OUT_TYPE\()_\UPDATE\()_popops__ReductionSpecialisation__##specialisation


//******************************************************************************
// Macros for log - add arithmetic
 // Add input to the result with:
 // max = max(input,result) and min = min(input,result)
 // result = max + std::log(1 + std::exp(min - max))

 // B_PRE, B_POST (BUNDLE_PRE, BUNDLE_POST) can be used to bundle by
 // passing "{nop;" "};" or omitted to not bundle
.macro f32v2LOGADD B_PRE  B_POST
  \B_PRE              f32v2max $a4:5, $RESULT, $a0:1
  \B_POST \B_PRE      f32v2min $a0:1, $RESULT, $a0:1
  \B_POST \B_PRE      f32v2sub $a0:1, $a0:1, $a4:5
  \B_POST \B_PRE      f32exp $a0,$a0
  \B_POST \B_PRE      f32exp $a1,$a1
  \B_POST \B_PRE      f32v2add $a0:1, $CONST:B, $a0:1
  \B_POST \B_PRE      f32ln $a0,$a0
  \B_POST \B_PRE      f32ln $a1,$a1
  \B_POST
// The last  "f32v2add $RESULT, $a4:5, $a0:1" Is omitted so it can be bundled
.endm

.macro f16v4LOGADD B_PRE  B_POST
  \B_PRE              f16v4max $a4:5, $RESULT, $a0:1
  \B_POST \B_PRE      f16v4min $a0:1, $RESULT, $a0:1
  \B_POST \B_PRE      f16v4sub $a0:1, $a0:1, $a4:5
  \B_POST \B_PRE      f16v2exp $a0,$a0
  \B_POST \B_PRE      f16v2exp $a1,$a1
  \B_POST \B_PRE      f16v4add $a0:1, $CONST:BL, $a0:1
  \B_POST \B_PRE      f16v2ln $a0,$a0
  \B_POST \B_PRE      f16v2ln $a1,$a1
  \B_POST
// The last  "f16v4add $RESULT, $a4:5, $a0:1" Is omitted so it can be bundled
.endm

.macro f16v2LOGADD B_PRE  B_POST
  \B_PRE              f16v2max $a4, $RESULT_LO, $a0
  \B_POST \B_PRE      f16v2min $a0, $RESULT_LO, $a0
  \B_POST \B_PRE      f16v2sub $a0, $a0, $a4
  \B_POST \B_PRE      f16v2exp $a0,$a0
  \B_POST \B_PRE      f16v2add $a0, $CONST:BL, $a0
  \B_POST \B_PRE      f16v2ln $a0,$a0
  \B_POST
// The last  "f16v2add $RESULT_LO, $a4, $a0" Is omitted so it can be bundled
.endm

//******************************************************************************
// Macro to create vertex code for float input variants
.macro INSTANTIATE_REDUCE_FLOAT OUT_TYPE OP INSTRUCTION OP_IMPL CONST_VALUE UPDATE
.equ LOG2_PARTIAL_SIZE, 2

FN_WORKER_ENTRY_POINT REDUCE_FLOAT(Reduce,STRIDED___REDUCE)
#ifdef VECTOR_AVAIL_SCALED_PTR32
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
#endif
.ifc "\OP_IMPL","log_add"
  // No scale implemented by adding 0 in log-mul op
  {bri         StridedReduceFloatCommon\@
    or         $SCALE, $azero, $azero}
.else
  {bri         StridedReduceFloatCommon\@
    or         $SCALE, $azero, FLOAT_1_0}
.endif
FN_SIZE REDUCE_FLOAT(Reduce,STRIDED___REDUCE)

FN_WORKER_ENTRY_POINT REDUCE_FLOAT(Reduce,STRIDED___REDUCE___OUTER)
#ifdef VECTOR_AVAIL_SCALED_PTR32
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
#endif
  ld32       $COUNTS_AND_STRIDES, $mvertex_base, $mzero, COUNTS_AND_STRIDES_OFFSET/4
  ldz16      $OUTER_STRIDE, $COUNTS_AND_STRIDES, $mzero, OUTER_STRIDE_OFFSET/2
  ldz16      $NUM_OUTER_STRIDES_CONST, $COUNTS_AND_STRIDES, $mzero, NUM_OUTER_STRIDES_OFFSET/2
.ifc "\OP_IMPL","log_add"
  // No scale implemented by adding 0 in log-mul op
  {bri         StridedReduceFloatCommonOuter\@
    or         $SCALE, $azero, $azero}
.else
  {bri         StridedReduceFloatCommonOuter\@
    or         $SCALE, $azero, FLOAT_1_0}
.endif
FN_SIZE REDUCE_FLOAT(Reduce,STRIDED___REDUCE___OUTER)

FN_WORKER_ENTRY_POINT REDUCE_FLOAT(ScaledReduce,STRIDED___REDUCE)
  // As scale uses a SCALED_PTR32 there is no downside to having partials also
  // use SCALED_PTR32 as we can load and use the same base address
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/2 // load scale
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $BASE, $mzero, $SCRATCH
#else
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4 // load scale
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif
  bri        StridedReduceFloatCommon\@
FN_SIZE REDUCE_FLOAT(ScaledReduce,STRIDED___REDUCE)


FN_WORKER_ENTRY_POINT REDUCE_FLOAT(ScaledReduce,STRIDED___REDUCE___OUTER)
  // As scale uses a SCALED_PTR32 there is no downside to having partials also
  // use SCALED_PTR32 as we can load and use the same base address
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/2 // load scale
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $BASE, $mzero, $SCRATCH
#else
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4 // load scale
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif
  ld32       $COUNTS_AND_STRIDES, $mvertex_base, $mzero, COUNTS_AND_STRIDES_OFFSET/4
  ldz16      $OUTER_STRIDE, $COUNTS_AND_STRIDES, $mzero, OUTER_STRIDE_OFFSET/2
  ldz16      $NUM_OUTER_STRIDES_CONST, $COUNTS_AND_STRIDES, $mzero, NUM_OUTER_STRIDES_OFFSET/2
  bri        StridedReduceFloatCommonOuter\@
FN_SIZE REDUCE_FLOAT(ScaledReduce,STRIDED___REDUCE___OUTER)

FN_SECTION REDUCE_FLOAT(ScaledReduceFloatCommon,STRIDED___REDUCE) 8 nop
StridedReduceFloatCommon\@:
  ld32 $COUNTS_AND_STRIDES, $mvertex_base, $mzero, COUNTS_AND_STRIDES_OFFSET/4
  mov  $NUM_OUTER_STRIDES_CONST, $mzero
StridedReduceFloatCommonOuter\@:
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16 $m0, $mvertex_base, $mzero, OUT_OFFSET/2 // load output pointer
  ldz16 $m1, $mvertex_base, $mzero, IN_OFFSET/2  // load partials pointer
  // keep $m0 and $m1 as byte offsets, using $BASE to hold memory base for
  // offset addressing below
  shl $m0, $m0, 2
  shl $m1, $m1, 2
#else
  ld32 $m0, $mvertex_base, $mzero, OUT_OFFSET/4 // load output pointer
  ld32 $m1, $mvertex_base, $mzero, IN_OFFSET/4  // load partials pointer
#endif
  // Ensure that the inputs being accumulated (that weren't loaded) are legal
  {ldz16 $OUT_COUNT, $COUNTS_AND_STRIDES, $mzero, NUM_OUTPUTS_OFFSET/2 // load numOutputs
   mov $RESULT, $azeros}
  {ldz16 $m3, $COUNTS_AND_STRIDES, $mzero, NUM_PARTIALS_OFFSET/2 // load numPartials
  setzi $ZAACC, ZAACC_BITMASK}

  ldz16 $STRIDE, $COUNTS_AND_STRIDES, $mzero, PARTIALS_WIDTH_OFFSET/2
  {mov $m6, $m1 // Load working partials ptr
   uput  $FP_CLR, $ZAACC}

.ifc "\OP_IMPL","log_add"
  // log_add uses an add instead of a load of the last in group with a stride
  // so manipulate the passed value
  sub $OUTER_STRIDE, $OUTER_STRIDE, $STRIDE
  shl $OUTER_STRIDE, $OUTER_STRIDE, 3
.endif

///////////////////// Assemble one of 3 nested loops
float_loop\@:
  // $m6 points to the first partial to be accumulated for this output
  // $STRIDE is the step between consecutive partials for the same output

.ifc "\OP_IMPL","acc"
  // Reading the second set of accumulators clears them for the next pass
 {mov $NUM_OUTER_STRIDES, $NUM_OUTER_STRIDES_CONST
  f32v2gina $RESULT, $azeros, 0}
float_outer_stride_loop\@:
  // First input is unused, so zero it
   {rpt $m3, (3f-2f)/8-1
    mov $a0:1, $azeros}  // Load zero into 1st input
2:
  {ld64step $a0:1, $BASE, $m6+=, $STRIDE
   \INSTRUCTION $f32v4RESULT}
3:
  // Read the last in a group with the outer stride
  {ld64step $a0:1, $BASE, $m6+=, $OUTER_STRIDE
  \INSTRUCTION $f32v4RESULT}
  {brnzdec $NUM_OUTER_STRIDES, float_outer_stride_loop\@
  \INSTRUCTION $f32v4RESULT}
  // advance to first partial for the next output
  // Get the result from the accumulators
 {add $m1, $m1, 8
  f32v2gina $RESULT, $azeros, 0}
.endif

.ifc "\OP_IMPL","max_min"
// Const value used to load the initial result
 {mov $NUM_OUTER_STRIDES, $NUM_OUTER_STRIDES_CONST
  or $RESULT_HI, $azero, \CONST_VALUE}
  {add $m1, $m1, 8 // advance to first partial for the next output
   mov $RESULT_LO,$RESULT_HI}
float_outer_stride_loop\@:
  // 1st input is unused - 1st pass: set to max or min, which is in result
  // Other passes - load the result, which has no effect: a = max(a,a)
  {rpt $m3, (3f-2f)/8-1
   mov $a0:1, $RESULT}
2:
  {ld64step $a0:1, $BASE, $m6+=, $STRIDE
   \INSTRUCTION $RESULT, $RESULT, $a0:1}
3:
  // Read the last one with the outer stride
 {ld64step $a0:1, $BASE, $m6+=, $OUTER_STRIDE
  \INSTRUCTION $RESULT, $RESULT, $a0:1}
  {brnzdec $NUM_OUTER_STRIDES, float_outer_stride_loop\@
  \INSTRUCTION $RESULT, $RESULT, $a0:1}
.endif

.ifc "\OP_IMPL","log_add"
  // For log-add the constant value is just needed as a constant for calculation
  // in a register.
  {ld64step $a0:1, $BASE, $m6+=, $STRIDE
   or  $CONST, $azero, \CONST_VALUE}
  {mov $NUM_OUTER_STRIDES, $NUM_OUTER_STRIDES_CONST
    mov $a4:5, $azeros}
   // Null $a4:5 so the 1st add produces the initial value
  {bri 1f
   fnop} // RPT alignment
float_outer_stride_loop\@:
  ld64step $a0:1, $BASE, $m6+=, $STRIDE
  f32v2LOGADD
1:
  rpt $m3, (3f-2f)/8-1
2:
  // Load the next input, complete the previous log-add
  {ld64step $a0:1, $BASE, $m6+=, $STRIDE
  f32v2add $RESULT, $a4:5, $a0:1}
  f32v2LOGADD "{nop;" "};"
3:
  // Use an add to advance the pointer to avoid overread with stride
  // Read the last one with the outer stride
  // Complete the log-add operation
  {add $m6, $m6, $OUTER_STRIDE
  f32v2add $RESULT, $a4:5, $a0:1}
  brnzdec $NUM_OUTER_STRIDES, float_outer_stride_loop\@

  // advance to first partial for the next output
  add $m1, $m1, 8
.endif

// Apply scale.  A log-mul is implemented with an add
.ifc "\OP_IMPL","log_add"
  {mov $m6, $m1
  f32v2add  $RESULT, $SCALE:B, $RESULT}
.else
  {mov $m6, $m1
  f32v2mul  $RESULT, $SCALE:B, $RESULT}
.endif

// Update and store - float
.ifc "\OUT_TYPE","float"
  .ifc "\UPDATE","true"
    ld32   $a0, $BASE, $m0, 0
    ld32   $a1, $BASE, $m0, 1
    .ifc "\OP_IMPL","log_add"
      f32v2LOGADD
      f32v2add $RESULT, $a4:5, $a0:1
    .else
      f32v2add $RESULT, $a0:1, $RESULT
    .endif
  .endif
  st32step $RESULT_LO, $BASE, $m0+=, 1
  st32step $RESULT_HI, $BASE, $m0+=, 1
.endif

// Update and store - half
.ifc "\OUT_TYPE","half"
  f32v2tof16 $RESULT_LO, $RESULT
  .ifc "\UPDATE","true"
    .ifc "\OP_IMPL","log_add"
      {ld32 $a0, $BASE, $m0, 0
       setzi $CONST, HALF_1_0}
      f16v2LOGADD
      f16v2add $RESULT_LO, $a4, $a0
      // Restore the constant needed for float log-add
      or $CONST, $azero, \CONST_VALUE
    .else
      ld32 $a0, $BASE, $m0, 0
      f16v2add $RESULT_LO, $RESULT_LO, $a0
    .endif
  .endif
  st32step $RESULT_LO, $BASE, $m0+=, 1
.endif

4:
  brnzdec $OUT_COUNT, float_loop\@
  exitnz $mzero
FN_SIZE REDUCE_FLOAT(ScaledReduceFloatCommon,STRIDED___REDUCE)
.endm

//******************************************************************************
// Macro to extract accumulators results, apply scale, update and store as float
.macro STORE_FLOAT_RESULT_FROM_ACCUMULATORS UPDATE
.ifc "\UPDATE","true"
  // Float output, with update
  {ld32   $RESULT_LO, $BASE, $m0, 0
   f32v2mul  $a0:1, $SCALE:B, $a0:1}
  ld32   $RESULT_HI, $BASE, $m0, 1
  f32v2add $a0:1, $a0:1, $RESULT
  {st32step $a0, $BASE, $m0+=, 1
   f32v2gina $RESULT, $azeros, 0}
  {st32step $a1, $BASE, $m0+=, 1
   f32v2mul  $RESULT, $SCALE:B, $RESULT}
  ld32   $a0, $BASE, $m0, 0
  ld32   $a1, $BASE, $m0, 1
  {mov $m6, $m1 // Load working ptr
   f32v2add $RESULT, $a0:1, $RESULT}
  st32step $RESULT_LO, $BASE, $m0+=, 1
  st32step $RESULT_HI, $BASE, $m0+=, 1
.else
  // float output, no update
  {mov $m6, $m1 // Load working ptr
   f32v2mul  $a0:1, $SCALE:B, $a0:1}
  {st32step $a0, $BASE, $m0+=, 1
   f32v2gina $RESULT, $azeros, 0}
  {st32step $a1, $BASE, $m0+=, 1
   f32v2mul  $RESULT, $SCALE:B, $RESULT}
  st32step $RESULT_LO, $BASE, $m0+=, 1
  st32step $RESULT_HI, $BASE, $m0+=, 1
.endif
.endm

//******************************************************************************
// Macro to extract accumulators results, apply scale, update and store as half
.macro STORE_HALF_RESULT_FROM_ACCUMULATORS UPDATE
.ifc "\UPDATE","true"
  // Half output, with update
  // Scale and update the 1st pair of outputs
  {ld32   $RESULT_LO, $BASE, $m0, 0
   f32v2mul  $a0:1, $SCALE:B, $a0:1}
  f32v2tof16 $a0, $a0:1
  f16v2add $a0, $a0, $RESULT_LO
  {st32step $a0, $BASE, $m0+=, 1
  f32v2gina $a0:1, $azeros, 0}
  // Scale and update the 2nd pair of outputs
  {ld32   $RESULT_LO, $BASE, $m0, 0
   f32v2mul  $a0:1, $SCALE:B, $a0:1}
  f32v2tof16 $a0, $a0:1
  {mov $m6, $m1 // Load working ptr
  f16v2add $a0, $a0, $RESULT_LO}
  st32step $a0, $BASE, $m0+=, 1
.else
  // Half output, no update
  f32v2mul  $a0:1, $SCALE:B, $a0:1
  f32v2tof16 $a0, $a0:1
  {st32step $a0, $BASE, $m0+=, 1
  f32v2gina $RESULT, $azeros, 0}
  f32v2mul  $RESULT, $SCALE:B, $RESULT
  {mov $m6, $m1 // Load working ptr
  f32v2tof16 $RESULT_LO, $RESULT}
  st32step $RESULT_LO, $BASE, $m0+=, 1
.endif
.endm
//******************************************************************************
// Macro to create vertex code for variants with a half input

.macro INSTANTIATE_REDUCE_HALF OUT_TYPE OP INSTRUCTION OP_IMPL CONST_VALUE UPDATE
.equ LOG2_PARTIAL_SIZE, 1

FN_WORKER_ENTRY_POINT REDUCE_HALF(Reduce,STRIDED___REDUCE)
#ifdef VECTOR_AVAIL_SCALED_PTR32
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
#endif
.ifc "\OP_IMPL","log_add"
  // No scale implemented by adding 0 in log-mul op
  { bri        ScaledReduceHalfCommon\@
    or         $SCALE, $azero, $azero}
.else
  { bri        ScaledReduceHalfCommon\@
    or         $SCALE, $azero, FLOAT_1_0}
.endif
FN_SIZE REDUCE_HALF(Reduce,STRIDED___REDUCE)

FN_WORKER_ENTRY_POINT REDUCE_HALF(Reduce,STRIDED___REDUCE___OUTER)
#ifdef VECTOR_AVAIL_SCALED_PTR32
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
#endif
  ld32       $COUNTS_AND_STRIDES, $mvertex_base, $mzero, COUNTS_AND_STRIDES_OFFSET/4
  ldz16      $OUTER_STRIDE, $COUNTS_AND_STRIDES, $mzero, OUTER_STRIDE_OFFSET/2
  ldz16      $NUM_OUTER_STRIDES_CONST, $COUNTS_AND_STRIDES, $mzero, NUM_OUTER_STRIDES_OFFSET/2
.ifc "\OP_IMPL","log_add"
  // No scale implemented by adding 0 in log-mul op
  { bri        ScaledReduceHalfCommonOuter\@
    or         $SCALE, $azero, $azero}
.else
  { bri        ScaledReduceHalfCommonOuter\@
    or         $SCALE, $azero, FLOAT_1_0}
.endif
FN_SIZE REDUCE_HALF(Reduce,STRIDED___REDUCE___OUTER)


FN_WORKER_ENTRY_POINT REDUCE_HALF(ScaledReduce,STRIDED___REDUCE)
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/2 // load scale
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $BASE, $mzero, $SCRATCH
#else
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4 // load scale
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif
  bri        ScaledReduceHalfCommon\@
FN_SIZE REDUCE_HALF(ScaledReduce,STRIDED___REDUCE)


FN_WORKER_ENTRY_POINT REDUCE_HALF(ScaledReduce,STRIDED___REDUCE___OUTER)
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/2 // load scale
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $BASE, $mzero, $SCRATCH
#else
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4 // load scale
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif
  ld32       $COUNTS_AND_STRIDES, $mvertex_base, $mzero, COUNTS_AND_STRIDES_OFFSET/4
  ldz16      $OUTER_STRIDE, $COUNTS_AND_STRIDES, $mzero, OUTER_STRIDE_OFFSET/2
  ldz16      $NUM_OUTER_STRIDES_CONST, $COUNTS_AND_STRIDES, $mzero, NUM_OUTER_STRIDES_OFFSET/2
  bri        ScaledReduceHalfCommonOuter\@
FN_SIZE REDUCE_HALF(ScaledReduce,STRIDED___REDUCE___OUTER)


FN_SECTION ScaledReduceHalfCommon\@ 8
ScaledReduceHalfCommon\@:
  ld32       $COUNTS_AND_STRIDES, $mvertex_base, $mzero, COUNTS_AND_STRIDES_OFFSET/4
  mov        $NUM_OUTER_STRIDES_CONST, $mzero
ScaledReduceHalfCommonOuter\@:
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16 $m0, $mvertex_base, $mzero, OUT_OFFSET/2 // load output pointer
  ldz16 $m1, $mvertex_base, $mzero, IN_OFFSET/2 // load partials pointer
  // keep $m0 and $m1 as byte offsets, using $BASE to hold memory base for
  // offset addressing below
  shl $m0, $m0, 2
  shl $m1, $m1, 2
#else
  ld32 $m0, $mvertex_base, $mzero, OUT_OFFSET/4 // load output pointer
  ld32 $m1, $mvertex_base, $mzero, IN_OFFSET/4 // load partials pointer
#endif
  {ldz16 $OUT_COUNT, $COUNTS_AND_STRIDES, $mzero, NUM_OUTPUTS_OFFSET/2  
  setzi $ZAACC, ZAACC_BITMASK}

  // Ensure that the inputs being accumulated (that weren't loaded) are legal.
  {ldz16 $m3, $COUNTS_AND_STRIDES, $mzero, NUM_PARTIALS_OFFSET/2 // load numPartials
  mov $RESULT, $azeros}
  {ldz16 $STRIDE, $COUNTS_AND_STRIDES, $mzero, PARTIALS_WIDTH_OFFSET/2
   setzi  $CONST,  \CONST_VALUE}
  {mov $m6, $m1 // Load working pointer
  uput  $FP_CLR, $ZAACC}

.ifc "\OP_IMPL","log_add"
  // log_add uses an add instead of a load of the last in group with a stride
  // so manipulate the passed value
  sub $OUTER_STRIDE, $OUTER_STRIDE, $STRIDE
  shl $OUTER_STRIDE, $OUTER_STRIDE, 3
.endif

///////////////////// Assemble one of 3 nested loops
half_loop\@:
  // $m6 points to the first partial to be accumulated for this output
  // $STRIDE is the step between consecutive partials for the same output

.ifc "\OP_IMPL","acc"
  // Reading the second set of accumulators clears them for the next pass
  {mov $NUM_OUTER_STRIDES, $NUM_OUTER_STRIDES_CONST
   f32v2gina $RESULT, $azeros, 0}
half_outer_stride_loop\@:
  // First input is unused, so zero it
  {rpt $m3, (3f-2f)/8-1
   mov $a0:1, $azeros}
2:
  // There is no f16v4sqadd, so use f16v8 for all types
 {ld64step $a0:1, $BASE, $m6+=, $STRIDE
  \INSTRUCTION $f32v4RESULT}
3:
  // Read the last in a group with the outer stride
  {ld64step $a0:1, $BASE, $m6+=, $OUTER_STRIDE
  \INSTRUCTION $f32v4RESULT}
  {brnzdec $NUM_OUTER_STRIDES, half_outer_stride_loop\@
   \INSTRUCTION $f32v4RESULT}

 // advance to first partial for the next output
 // If using the acc the result is in float so we apply the float scale and
 // cast to half if required
 {add $m1, $m1, 8
  f32v2gina $a0:1, $azeros, 0}
  // Macro to store will also load the working pointer
  .ifc "\OUT_TYPE","float"
    STORE_FLOAT_RESULT_FROM_ACCUMULATORS \UPDATE
  .endif
  .ifc "\OUT_TYPE","half"
    STORE_HALF_RESULT_FROM_ACCUMULATORS \UPDATE
  .endif
.endif

.ifc "\OP_IMPL","max_min"
  // Const value used to load the initial result
  {mov $NUM_OUTER_STRIDES, $NUM_OUTER_STRIDES_CONST
   sort4x16lo $RESULT_LO, $CONST, $CONST}
  {add $m1, $m1, 8 // advance to first partial for the next output
   sort4x16lo $RESULT_HI, $CONST, $CONST}
  // 1st input is unused - 1st pass: set to max or min, which is in result
  // Other passes - load the result, which has no effect: a = max(a,a)
half_outer_stride_loop\@:
  {rpt $m3, (3f-2f)/8-1
   mov $a0:1, $RESULT}
2:
 {ld64step $a0:1, $BASE, $m6+=, $STRIDE
  \INSTRUCTION $RESULT,$a0:1,$RESULT}
3:
  // Read the last in a group with the outer stride
  {ld64step $a0:1, $BASE, $m6+=, $OUTER_STRIDE
  \INSTRUCTION $RESULT,$a0:1,$RESULT}
  {brnzdec $NUM_OUTER_STRIDES, half_outer_stride_loop\@
   \INSTRUCTION $RESULT,$a0:1,$RESULT}

  // The result is in half. Cast so that we can apply scale, and then cast back.
  // Here we will always output half for max, min
  f16v2tof32 $a0:1, $RESULT_LO
  {mov $m6,$m1  // Load working pointer
  f16v2tof32 $RESULT, $RESULT_HI}

  f32v2mul  $RESULT, $SCALE:B, $RESULT
  f32v2mul  $a0:1, $SCALE:B, $a0:1
  f32v4tof16 $RESULT, $f32v4RESULT
  .ifc "\UPDATE","true"
    ld32 $a0, $BASE, $m0, 0
    ld32 $a1, $BASE, $m0, 1
    f16v4add $RESULT, $a0:1, $RESULT
  .endif
  st32step $RESULT_LO, $BASE, $m0+=, 1
  st32step $RESULT_HI, $BASE, $m0+=, 1
.endif

.ifc "\OP_IMPL","log_add"
  // For log-add the constant value is just needed as a constant for calculation
  // in a register.
  {mov $NUM_OUTER_STRIDES, $NUM_OUTER_STRIDES_CONST
   sort4x16lo $RESULT_LO, $CONST, $CONST}
  // Null $a4:5 so the 1st add produces the initial value
  {ld64step $a0:1, $BASE, $m6+=, $STRIDE
   mov $a4:5, $azeros}
   {bri  1f
    fnop} //RPT alignment
half_outer_stride_loop\@:
  ld64step $a0:1, $BASE, $m6+=, $STRIDE
  f16v4LOGADD
1:
  rpt $m3, (3f-2f)/8-1
2:
  // Load the next input, complete the previous log-add
 {ld64step $a0:1, $BASE, $m6+=, $STRIDE
  f16v4add $RESULT, $a4:5, $a0:1}
  f16v4LOGADD "{nop;" "};"
3:
  // Use an add to advance the pointer to avoid overread with stride
  // Complete the log-add
  {add $m6, $m6, $OUTER_STRIDE
  f16v4add $RESULT, $a4:5, $a0:1}
  brnzdec $NUM_OUTER_STRIDES, half_outer_stride_loop\@

  // advance to first partial for the next output
  add $m1, $m1, 8

  // The result is in half. Cast so that we can apply scale
  .ifc "\OUT_TYPE","half"
    {mov $m6,$m1  // Load working pointer
    f16v2tof32 $a0:1, $RESULT_LO}
    f16v2tof32 $RESULT, $RESULT_HI
    f32v2add  $a0:1, $SCALE:B, $a0:1
    f32v2add  $RESULT, $SCALE:B, $RESULT
    f32v4tof16 $RESULT, $f32v4RESULT
    .ifc "\UPDATE","true"
      ld32 $a0, $BASE, $m0, 0
      ld32 $a1, $BASE, $m0, 1
      f16v4LOGADD
      f16v4add $RESULT, $a4:5, $a0:1
    .endif
    st32step $RESULT_LO, $BASE, $m0+=, 1
    st32step $RESULT_HI, $BASE, $m0+=, 1
  .else

    {st32 $RESULT_HI, $mzero, $mworker_base, SCRATCH_RESULT_UPPER
    f16v2tof32 $RESULT, $RESULT_LO}
    // Apply scale (logMul = add)
    {mov $m6,$m1  // Load working pointer
     f32v2add  $RESULT, $SCALE:B, $RESULT}
    .ifc "\UPDATE","true"
      // Updating, a float result
      ld32 $a0, $BASE, $m0, 0
      // Constant needed as we are now doing float log-add not half
      {ld32 $a1, $BASE, $m0, 1
       or  $CONST, $azero, FLOAT_1_0}
      f32v2LOGADD
      {ld32 $RESULT_HI, $mzero, $mworker_base, SCRATCH_RESULT_UPPER
       f32v2add $a0:1, $a0:1, $a4:5}
      st32step $a0, $BASE, $m0+=, 1
      st32step $a1, $BASE, $m0+=, 1

      // 2nd pair of results
      {ld32 $a0, $BASE, $m0, 0
       f16v2tof32 $RESULT, $RESULT_HI}
      {ld32 $a1, $BASE, $m0, 1
       f32v2add  $RESULT, $SCALE:B, $RESULT}
      f32v2LOGADD
      f32v2add $RESULT, $a0:1, $a4:5
      // Store the last results, restore the constant for fp16 log-add
      {st32step $RESULT_LO, $BASE, $m0+=, 1
       setzi  $CONST, \CONST_VALUE}
      st32step $RESULT_HI, $BASE, $m0+=, 1
    .else
      // Store the 1st 2 values, cast the 2nd two, apply scale and store
      ld32 $a0, $mzero, $mworker_base, SCRATCH_RESULT_UPPER
      {st32step $RESULT_LO, $BASE, $m0+=, 1
       f16v2tof32 $a0:1, $a0}
      {st32step $RESULT_HI, $BASE, $m0+=, 1
       f32v2add  $a0:1, $SCALE:B, $a0:1}
      st32step $a0, $BASE, $m0+=, 1
      st32step $a1, $BASE, $m0+=, 1
    .endif
  .endif
.endif
4:
  brnzdec $OUT_COUNT, half_loop\@

  exitnz $mzero
FN_SIZE REDUCE_HALF(ScaledReduceHalfCommon,STRIDED___REDUCE)
.endm

//******************************************************************************
// Use macros to instantiate vertices

// It is useful to have add, squareAdd vertices which cast all 4 combinations of
// half to/from float, as per the logic in the reduction library
// (reduce add, squareAdd ops keep better range and precision with
// intermediate values kept as float).
.macro INSTANTIATE_ADD_SQUARE_ADD UPDATE
  INSTANTIATE_REDUCE_FLOAT float ReduceAdd f32v4acc acc 0 \UPDATE
  INSTANTIATE_REDUCE_FLOAT float ReduceSquareAdd f32v4sqacc acc 0 \UPDATE
  INSTANTIATE_REDUCE_FLOAT half ReduceAdd f32v4acc acc 0 \UPDATE
  INSTANTIATE_REDUCE_FLOAT half ReduceSquareAdd f32v4sqacc acc 0 \UPDATE

  INSTANTIATE_REDUCE_HALF float ReduceAdd f16v8acc acc 0 \UPDATE
  INSTANTIATE_REDUCE_HALF float ReduceSquareAdd f16v8sqacc acc 0 \UPDATE
  INSTANTIATE_REDUCE_HALF half ReduceAdd f16v8acc acc 0 \UPDATE
  INSTANTIATE_REDUCE_HALF half ReduceSquareAdd f16v8sqacc acc 0 \UPDATE
.endm

INSTANTIATE_ADD_SQUARE_ADD true
INSTANTIATE_ADD_SQUARE_ADD false

// It is useful to have max, min vertices which maintain the type, there is no
// point in casting.
.macro INSTANTIATE_MAX_MIN UPDATE
  INSTANTIATE_REDUCE_FLOAT float ReduceMax f32v2max max_min MIN_FLOAT \UPDATE
  INSTANTIATE_REDUCE_FLOAT float ReduceMin f32v2min max_min MAX_FLOAT \UPDATE

  INSTANTIATE_REDUCE_HALF half ReduceMax f16v4max max_min MIN_HALF \UPDATE
  INSTANTIATE_REDUCE_HALF half ReduceMin f16v4min max_min MAX_HALF \UPDATE
.endm

INSTANTIATE_MAX_MIN true
INSTANTIATE_MAX_MIN false

.macro INSTANTIATE_LOG_ADD UPDATE
  // We don't use an initial value here, pass in 1.0 in the correct type
  // instead which is needed
  INSTANTIATE_REDUCE_FLOAT float ReduceLogAdd void log_add FLOAT_1_0 \UPDATE
  INSTANTIATE_REDUCE_FLOAT half ReduceLogAdd void log_add FLOAT_1_0 \UPDATE

  INSTANTIATE_REDUCE_HALF float ReduceLogAdd void log_add HALF_1_0 \UPDATE
  INSTANTIATE_REDUCE_HALF half ReduceLogAdd void log_add HALF_1_0 \UPDATE
.endm

INSTANTIATE_LOG_ADD true
INSTANTIATE_LOG_ADD false

#endif
