// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__

// Specialisation 3 SINGLE_OUTPUT_REGION - Overview:
// `partials` is a single edge
// `out` is a single edge
// The vertex treats partials as a 2D array, size {`numPartials`, `numOutputs`}
// Eg, for numOutputs = 12, numPartials = 3
// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
// 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
//
// The output will be the sum of each 'column':
// 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36,
//
// Constraints:
// The output must be 32bit aligned, partials must be 64bit aligned.
// Num outputs must result in a 64-bit multiple (ie - 2 floats or 4 halves)
//
// Operation/speed:
// Accumulate down columns, 64 bits at once (2 floats or 4 halves), using a
// stride to skip to the next row.
// This results in an inner loop that takes 1 cycle per 64 bits processed (2 floats or 4 halves).

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "MathConstants.S"

#ifdef VECTOR_AVAIL_SCALED_PTR32
#define OUT_OFFSET           0
#define IN_OFFSET            2
#define NUM_OUTPUTS_OFFSET   4
#define NUM_PARTIALS_OFFSET  6
#define SCALE_OFFSET         8
#else
#define OUT_OFFSET           0
#define IN_OFFSET            4
#define NUM_OUTPUTS_OFFSET   8
#define NUM_PARTIALS_OFFSET  10
#define SCALE_OFFSET         12
#endif

#define PTR32_SHL_BITS   2
#define PTR64_SHL_BITS   3

// Register definitions
#define SCRATCH2        m7
#ifdef VECTOR_AVAIL_SCALED_PTR32
#define BASE            m8
#else
#define BASE            mzero
#endif

#define ZAACC           a4
#define SCALE           a6

// Constants
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// Name mangling
#define REDUCE_FLOAT_FLOAT(prefix, specialisation) __runCodelet_popops__##prefix##___popops__\OP\()_float_float_false_##specialisation
#define REDUCE_HALF(prefix, specialisation) __runCodelet_popops__##prefix##___popops__\OP\()_half_\OUT_TYPE\()_false_##specialisation

//******************************************************************************
// Macro to create vertex code for float float
.macro INSTANTIATE_REDUCE_FLOAT_FLOAT OP INSTRUCTION OP_USES_ACC INITIAL_VALUE
.equ LOG2_PARTIAL_SIZE, 2

.globl REDUCE_FLOAT_FLOAT(Reduce,3)
.type REDUCE_FLOAT_FLOAT(Reduce,3), @function
.globl REDUCE_FLOAT_FLOAT(ScaledReduce,3)
.type REDUCE_FLOAT_FLOAT(ScaledReduce,3), @function

.section .text.REDUCE_FLOAT_FLOAT(Reduce,3common), "ax"
.align 8

REDUCE_FLOAT_FLOAT(Reduce,3common):
REDUCE_FLOAT_FLOAT(Reduce,3):
setzi $BASE, TMEM_REGION0_BASE_ADDR
{
  bri        1f
  or         $SCALE, $azero, FLOAT_1_0
}
REDUCE_FLOAT_FLOAT(ScaledReduce,3):
  // As scale uses a SCALED_PTR32 there is no downside to having partials also
  // use SCALED_PTR32 as we can load and use the same base address
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16      $SCRATCH2, $mvertex_base, $mzero, SCALE_OFFSET/2 // load scale
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $BASE, $mzero, $SCRATCH2
#else
  ld32       $SCRATCH2, $mvertex_base, $mzero, SCALE_OFFSET/4 // load scale
  {
    ld32       $SCALE, $mzero, $SCRATCH2, 0
    fnop  // rpt alignment
  }
#endif
1:
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
  ldz16 $m2, $mvertex_base, $mzero, NUM_OUTPUTS_OFFSET/2 // load numOutputs
  {ldz16 $m3, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2 // load numPartials
  setzi $ZAACC, ZAACC_BITMASK}

  // $m4 = numOutputs/4; 2 elements per loop for float partials
  and $m4, $m2, 1<<(3-LOG2_PARTIAL_SIZE)-1
  brz $m4, 9f // branch if outputs a multiple of 8 bytes
  ld32 $m0, $mzero, $mzero, 0 // issue a null address read to stop
9:
  shr $m4, $m2, 3-LOG2_PARTIAL_SIZE
  // $m6 holds the rewind in bytes from the final partial back to the first for
  // the next output loop. Note the core loop makes an extra stride
  add $m6, $m3, 1
  mul $m6, $m6, $m2
  shl $m6, $m6, LOG2_PARTIAL_SIZE // float partials
  sub $m6, $m6, 8
  // $m7 = 64bit offset between consecutive partials for the same output
 {shr $m7, $m2, 3-LOG2_PARTIAL_SIZE
  uput  $FP_CLR, $ZAACC}
  bri 4f
1:
  // $m1 points to the first partial to be accumulated for this output
  // $m7 is the step between consecutive partials for the same output
  {ld64step $a0:1, $BASE, $m1+=, $m7
  or $a2, $azero, \INITIAL_VALUE}
  {rpt $m3, (3f-2f)/8-1
  mov $a3,$a2}
  // last pass will overread 8 bytes
2:
.ifc "\OP_USES_ACC","true"
 {ld64step $a0:1, $BASE, $m1+=, $m7
  \INSTRUCTION $a0:3}
.else
 {ld64step $a0:1, $BASE, $m1+=, $m7
  \INSTRUCTION $a2:3, $a2:3, $a0:1}
.endif
3:
.ifc "\OP_USES_ACC","true"
  // Get the result from the accumulators if they were used
  f32v2gina $a2:3, $azeros, 0
.endif
 {sub $m1, $m1, $m6 // advance to first partial for the next output
  f32v2mul  $a2:3, $SCALE:B, $a2:3}
  st32step $a2, $BASE, $m0+=, 1
  st32step $a3, $BASE, $m0+=, 1
4:
  brnzdec $m4, 1b
  exitnz $mzero
.size REDUCE_FLOAT_FLOAT(Reduce,3), .-REDUCE_FLOAT_FLOAT(Reduce,3)
.endm

//******************************************************************************
// Macro to create vertex code for half float

.macro INSTANTIATE_REDUCE_HALF OUT_TYPE OP INSTRUCTION OP_USES_ACC INITIAL_VALUE
.equ LOG2_PARTIAL_SIZE, 1

.globl REDUCE_HALF(Reduce,3)
.type REDUCE_HALF(Reduce,3), @function
.globl REDUCE_HALF(ScaledReduce,3)
.type REDUCE_HALF(ScaledReduce,3), @function

.section .text.REDUCE_HALF(Reduce,3common), "ax"
.align 4

REDUCE_HALF(Reduce,3common):
REDUCE_HALF(Reduce,3):
setzi $BASE, TMEM_REGION0_BASE_ADDR
{
  bri        1f
  or         $SCALE, $azero, FLOAT_1_0
}
REDUCE_HALF(ScaledReduce,3):
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16      $SCRATCH2, $mvertex_base, $mzero, SCALE_OFFSET/2 // load scale
  setzi      $BASE, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $BASE, $mzero, $SCRATCH2
#else
  ld32       $SCRATCH2, $mvertex_base, $mzero, SCALE_OFFSET/4 // load scale
  ld32       $SCALE, $mzero, $SCRATCH2, 0
#endif
1:
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16 $m0, $mvertex_base, $mzero, OUT_OFFSET/2 // load output pointer
  ldz16 $m1, $mvertex_base, $mzero, IN_OFFSET/2 // load partials pointer
  ldz16 $m2, $mvertex_base, $mzero, NUM_OUTPUTS_OFFSET/2  // load numOutputs
  ldz16 $m3, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2 // load numPartials
  // keep $m0 and $m1 as byte offsets, using $BASE to hold memory base for
  // offset addressing below
  {shl $m0, $m0, 2
  setzi $ZAACC, ZAACC_BITMASK}
.ifc "\OP_USES_ACC","true"
  {shl $m1, $m1, 2
   or $a7, $azero, \INITIAL_VALUE}
.else
  shl $m1, $m1, 2
  ldconst $a7,\INITIAL_VALUE
.endif
#else
  ld32 $m0, $mvertex_base, $mzero, OUT_OFFSET/4 // load output pointer
  ld32 $m1, $mvertex_base, $mzero, IN_OFFSET/4 // load partials pointer
  {ldz16 $m2, $mvertex_base, $mzero, NUM_OUTPUTS_OFFSET/2  // load numOutputs
  setzi $ZAACC, ZAACC_BITMASK}
.ifc "\OP_USES_ACC","true"
  {ldz16 $m3, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2 // load numPartials
   or $a7, $azero, \INITIAL_VALUE}
.else
  ldz16 $m3, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2 // load numPartials
  ldconst $a7,\INITIAL_VALUE
.endif
#endif
  // $m4 = numOutputs/4; 2 elements per loop for float partials
  {and $m4, $m2, 1<<(3-LOG2_PARTIAL_SIZE)-1
   sort4x16lo $a7,$a7,$a7}
  brz $m4, 9f // branch if outputs a multiple of 8 bytes
  ld32 $m0, $mzero, $mzero, 0 // issue a null address read to stop
9:
  shr $m4, $m2, 3-LOG2_PARTIAL_SIZE
  // $m6 holds the rewind in bytes from the final partial back to the first for
  // the next output loop. Note the core loop makes an extra stride
  add $m6, $m3, 1
  mul $m6, $m6, $m2
  shl $m6, $m6, LOG2_PARTIAL_SIZE // half partials
  sub $m6, $m6, 8
  // $m7 = 64bit offset between consecutive partials for the same output
 {shr $m7, $m2, 3-LOG2_PARTIAL_SIZE
  uput  $FP_CLR, $ZAACC}
  bri 4f

.align 8 //Repeat alignment given variable code size above
1:
  // $m1 points to the first partial to be accumulated for this output
  // $m7 is the step between consecutive partials for the same output
  {ld64step $a0:1, $BASE, $m1+=, $m7
   mov $a2,$a7}
  {rpt $m3, (3f-2f)/8-1
   mov $a3,$a7}
  // last pass will overread 8 bytes
  // There is no f16v4sqadd, so use f16v8 for all types
2:
.ifc "\OP_USES_ACC","true"
 {ld64step $a0:1, $BASE, $m1+=, $m7
  \INSTRUCTION $a0:3}
.else
 {ld64step $a0:1, $BASE, $m1+=, $m7
  \INSTRUCTION $a2:3,$a0:1,$a2:3}
.endif
3:
// Extract result from accumulators if necessary, apply scale and store
.ifc "\OP_USES_ACC","true"
 // If using the acc the result is in float so we apply the float scale and
 // cast to half if required
 {sub $m1, $m1, $m6 // advance to first partial for the next output
  f32v2gina $a0:1, $azeros, 0}
  f32v2mul  $a0:1, $SCALE:B, $a0:1
  .ifc "\OUT_TYPE","float"
    {st32step $a0, $BASE, $m0+=, 1
    f32v2gina $a2:3, $azeros, 0}
    {st32step $a1, $BASE, $m0+=, 1
    f32v2mul  $a2:3, $SCALE:B, $a2:3}
    st32step $a2, $BASE, $m0+=, 1
    st32step $a3, $BASE, $m0+=, 1
  .else
    f32v2tof16 $a0, $a0:1
    {st32step $a0, $BASE, $m0+=, 1
    f32v2gina $a2:3, $azeros, 0}
    f32v2mul  $a2:3, $SCALE:B, $a2:3
    f32v2tof16 $a2, $a2:3
    st32step $a2, $BASE, $m0+=, 1
  .endif
.else
  // If not using the acc then the result is in half. Cast so that we can
  // apply scale, and then cast back
  {sub $m1, $m1, $m6 // advance to first partial for the next output
  f16v2tof32 $a0:1, $a2}
  f16v2tof32 $a2:3, $a3
  f32v2mul  $a2:3, $SCALE:B, $a2:3
  f32v2mul  $a0:1, $SCALE:B, $a0:1
  f32v4tof16 $a0:1, $a0:3
  st32step $a0, $BASE, $m0+=, 1
  st32step $a1, $BASE, $m0+=, 1

.endif
4:
  brnzdec $m4, 1b
  exitnz $mzero
.size REDUCE_HALF(Reduce,3common), .-REDUCE_HALF(Reduce,3common)
.endm

//******************************************************************************
// Use macros to instantiate vertices

// It is useful to have add, squareAdd vertices which cast half->float or
// maintain float->float, as per the logic in the reduction library
// (reduce add, squareAdd ops keep better range and precision with
// intermediate values kept as float).
// The last stage cast of float->half is done as a separate cast at the moment,
// this could be changed which would make float->half useful
INSTANTIATE_REDUCE_FLOAT_FLOAT ReduceAdd f32v4acc true 0
INSTANTIATE_REDUCE_FLOAT_FLOAT ReduceSquareAdd f32v4sqacc true 0

INSTANTIATE_REDUCE_HALF float ReduceAdd f16v8acc true 0
INSTANTIATE_REDUCE_HALF float ReduceSquareAdd f16v8sqacc true 0

// It is useful to have max, min vertices which maintain the type, there is no
// point in casting.
INSTANTIATE_REDUCE_FLOAT_FLOAT ReduceMax f32v2max false MIN_FLOAT
INSTANTIATE_REDUCE_FLOAT_FLOAT ReduceMin f32v2min false MAX_FLOAT

INSTANTIATE_REDUCE_HALF half ReduceMax f16v4max false MIN_HALF
INSTANTIATE_REDUCE_HALF half ReduceMin f16v4min false MAX_HALF

#endif
