// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
// Partials Equal Size Reduce (PARTIALS_EQUAL_SIZE) Overview:
// `partials` is a vectorlist (DELTAN)
// `out` is a single edge
// Each partial is treated as a row, so for example with
// `partialsSizeM1` = 2 we have partials with length 2*grainsize*(2+1),
// with (grainsize=4 for half, 2 for float). The example is for float.
//
//       'column'     0  1  2  3  4  5  6  7  8  9  10  11
// vectorlist ptr 0-> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
// vectorlist ptr 1-> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12,
// vectorlist ptr 2-> 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12, 13,
//
// We reduce to `outCount` * 2 * grainSize output words, so if
// outCount = 1 the output width would be 1 * 2 * 2 = 4 (floats)
// Partials 'fold' to the output width so:
// The first output is the sum of columns 0, 4, 8
// Second output is the sum of columns 1, 5, 9
// etc...
//
// To implement the above efficiently, all partials must be the same size,
// all a multiple of 128 bits in length.  The output must also be a multiple of 128
// bits in length.  Finally the partials must be a multiple of the output in
// length.
// Everything is 64-bit aligned.
//
// The result is that there is no need to deal with any odd number of partials
// or outputs.  The inner loop indexes through each partial one by one, in
// other words down the matrix as described above.  This means that it has to
// interpret the DELTAN vectorlist and we take 4 cycles to process 128 bits
// of data (8 halves, or 4 floats).
//
// Note - It is the overhead of having to unpack the DELTAN ptrs in the inner
//        loop that has led to the decision to process 128, not 64 bits
//        per loop pass. Without this, 64 bits would be a better choice.
// Note - Of course the items pointed to in a vectorlist can be anywhere in
//        memory.  Probably often they are in the same region and spaced
//        equally.  In this case a different vertex could deal with them.
//        It would only need a single edge for the input, a number of partials
//        parameter and a stride parameter.  The result would be a more efficient
//        inner loop and a similar result achieved given the constraints of
//        consistent strides within one region.


#include "poplibs_support/TileConstants.hpp"

// vertex state, all offsets are 16-bit
#define VERTEX_OUT_PTR_OFFSET 0
#define VERTEX_OUT_COUNT_OFFSET 1
#define VERTEX_PARTIALS_BASE_OFFSET 2
#define VERTEX_PARTIALS_DELTA_PTR_OFFSET 4
#define VERTEX_PARTIALS_SIZE_OFFSET 5
#define VERTEX_SCALE_OFFSET 6

// stack state, all offsets are 32-bit
#define STACK_PARTIALS_DELTA_PTR_OFFSET 0

// constants
#define LDCONST_MASK ((1<<20)-1)
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define SCALED_PTR32_SHL_BITS 2
#define SCALED_PTR64_SHL_BITS 3
#define VECTOR_LIST_BASE_BITS 20
#define VECTOR_LIST_COUNT_BITS 12
#define DELTAN_LENGTH_BITS 14
#define DELTAN_ADDRESS_MASK 0x3ffff

#define FLOAT_CONST_ONE 0x3f800000
#define FLOAT_GRAIN_SIZE 4
#define HALF_GRAIN_SIZE 8
#define BYTES_PER_GRAIN 16

// integer variables
#define outPtr m0
#define outCount m1
#define partialsSizeCounter m2
#define partialsBase m3
#define partialsDeltaPtr m4
#define numPartials m5
#define partialsSize m6
#define partialsElementPtr m7
#define partialsBaseWorking m8
#define mscratch m9
#define partialsStride m10
#define base m11

// floating point variables
#define scale a7

#define VERTEX __runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_\UPDATE
#define VERTEX_SCALED __runCodelet_popops__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_\UPDATE

//******************************************************************************
// Macro to extract the output, as float
.macro OUTPUT_FLOATS UPDATE GRAIN_SIZE
  // Read, scale and store the output
  // offset by grain size 128 bit words for the next output group
  {
    add $partialsBase, $partialsBase, BYTES_PER_GRAIN
    f32v2gina $a0:1, $azeros, 0
  }

.ifc "\UPDATE","true"
  {
    rpt \GRAIN_SIZE/2-1, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld64 $a2:3, $mzero, $outPtr, 0
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
  {
    nop
    f32v2add $a0:1, $a0:1, $a2:3
  }
  {
    st64step $a0:1, $mzero, $outPtr+=, 1
    f32v2gina $a0:1, $azeros, 0
  }
2:
  {
    ld64 $a2:3, $mzero, $outPtr, 0
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
  f32v2add $a0:1, $a0:1, $a2:3

.else

  {
    rpt \GRAIN_SIZE/2-1, (2f-1f)/8-1
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
1:
  {
    st64step $a0:1, $mzero, $outPtr+=, 1
    f32v2gina $a0:1, $azeros, 0
  }
  {
    nop
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
2:
.endif

  st64step $a0:1, $mzero, $outPtr+=, 1

.endm

//******************************************************************************
// Macro to extract the output, as half
.macro OUTPUT_HALFS UPDATE GRAIN_SIZE
  // Read, scale and store the output
  // offset by grain size 128 bit words for the next output group
  {
    add $partialsBase, $partialsBase, BYTES_PER_GRAIN
    f32v2gina $a0:1, $azeros, 0
  }

.ifc "\UPDATE","true"
  {
    rpt \GRAIN_SIZE/2-1, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld32 $a2, $mzero, $outPtr, 0
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
  {
    nop
    f32v2tof16 $a0, $a0:1
  }
  {
    nop
    f16v2add $a0, $a0, $a2
  }
  {
    st32step $a0, $mzero, $outPtr+=, 1
    f32v2gina $a0:1, $azeros, 0
  }
2:
  {
    ld32 $a2, $mzero, $outPtr, 0
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
  f32v2tof16 $a0, $a0:1
  f16v2add $a0, $a0, $a2

.else

  {
    rpt \GRAIN_SIZE/2-1, (2f-1f)/8-1
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
1:
  {
    nop
    f32v2tof16 $a0, $a0:1
  }
  {
    st32step $a0, $mzero, $outPtr+=, 1
    f32v2gina $a0:1, $azeros, 0
  }
  {
    nop
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
2:
  f32v2tof16 $a0, $a0:1
.endif

  st32step $a0, $mzero, $outPtr+=, 1

.endm

//******************************************************************************
.macro MAKE_REDUCE_VERTEX OP PARTIALS_TYPE OUT_TYPE UPDATE ACC

// *********** Setup some stuff depending on the operation selected ***********
.ifc "\PARTIALS_TYPE","half"
  .equ GRAIN_SIZE, HALF_GRAIN_SIZE
.else
  .equ GRAIN_SIZE, FLOAT_GRAIN_SIZE
.endif

// ************* Macro code ****************

.globl VERTEX
.type VERTEX @function

.section .text.VERTEX
.align 4
VERTEX:
  setzi $base, TMEM_REGION0_BASE_ADDR
  {
    bri common_\@
    or $scale, $azero, FLOAT_CONST_ONE & ~LDCONST_MASK
  }

.size VERTEX, .-VERTEX

.globl VERTEX_SCALED
.type VERTEX_SCALED @function

.section .text.VERTEX_SCALED
.align 8
  nop     // Rpt alignment

VERTEX_SCALED:
  ldz16 $mscratch, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET
  setzi $base, TMEM_REGION0_BASE_ADDR
  ld32  $scale, $base, $mzero, $mscratch

common_\@:
  ldz16 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET
  ldz16 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET
  ld32  $partialsBase, $mzero, $mvertex_base, VERTEX_PARTIALS_BASE_OFFSET/2
  ldz16 $partialsDeltaPtr, $mzero, $mvertex_base, VERTEX_PARTIALS_DELTA_PTR_OFFSET

  // unpack pointers and sizes from the vertex state.
  shl $outPtr, $outPtr, SCALED_PTR64_SHL_BITS
  shr $numPartials, $partialsBase, VECTOR_LIST_BASE_BITS
  shl $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shr $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shl $partialsDeltaPtr, $partialsDeltaPtr, SCALED_PTR32_SHL_BITS
  // store the partialsDeltaPtr onto the stack as we reset it each iteration of
  // the middle loop.
  st32 $partialsDeltaPtr, $mzero, $mworker_base, STACK_PARTIALS_DELTA_PTR_OFFSET
  // Setup for addressing partials
  shl $partialsStride, $outCount, 4
  setzi $mscratch, DELTAN_ADDRESS_MASK

  sub $outCount, $outCount, 1

out_loop_\@:
  // reload the partials base pointer and reset numPartialsM1
  {
    ldz16 $partialsSizeCounter, $mzero, $mvertex_base, VERTEX_PARTIALS_SIZE_OFFSET
    // zero the accumulators.
    setzi $a0, ZAACC_BITMASK
  }
  {
    mov $partialsBaseWorking, $partialsBase
    uput $FP_CLR, $a0
  }
partial_column_loop_\@:

  // Loop though all the partials, reducing our grain size columns each loop
  ld32 $partialsDeltaPtr, $mzero, $mworker_base, STACK_PARTIALS_DELTA_PTR_OFFSET
  ld32step $partialsElementPtr, $base, $partialsDeltaPtr+=, 1

    rpt $numPartials, (2f-1f)/8-1
1:
  {
    and $partialsElementPtr, $partialsElementPtr, $mscratch
    fnop
  }
  {
    ld64 $a0:1, $partialsBaseWorking, $partialsElementPtr, 0
    fnop
  }
  {
    ld64 $a2:3, $partialsBaseWorking, $partialsElementPtr, 1
    fnop
  }
  {
    // Over read of a pointer
    ld32step $partialsElementPtr, $base, $partialsDeltaPtr+=, 1
    \ACC $a0:3
  }
2:
  // Stride to the next grain size column group that needs to be reduced into
  // the current set of outputs
  add $partialsBaseWorking, $partialsBaseWorking, $partialsStride
  brnzdec $partialsSizeCounter, partial_column_loop_\@

// Pick an output macro based on output type
.ifc "\OUT_TYPE", "float"
  OUTPUT_FLOATS \UPDATE GRAIN_SIZE
.endif
.ifc "\OUT_TYPE", "half"
  OUTPUT_HALFS \UPDATE GRAIN_SIZE
.endif

  brnzdec $outCount, out_loop_\@

  exitz $mzero

.size VERTEX_SCALED, .-VERTEX_SCALED

.endm

MAKE_REDUCE_VERTEX ReduceAdd half float false f16v8acc
MAKE_REDUCE_VERTEX ReduceAdd float float false f32v4acc
MAKE_REDUCE_VERTEX ReduceSquareAdd half float false f16v8sqacc
MAKE_REDUCE_VERTEX ReduceSquareAdd float float false f32v4sqacc

MAKE_REDUCE_VERTEX ReduceAdd half half false f16v8acc
MAKE_REDUCE_VERTEX ReduceAdd float half false f32v4acc
MAKE_REDUCE_VERTEX ReduceSquareAdd half half false f16v8sqacc
MAKE_REDUCE_VERTEX ReduceSquareAdd float half false f32v4sqacc

MAKE_REDUCE_VERTEX ReduceAdd half float true f16v8acc
MAKE_REDUCE_VERTEX ReduceAdd float float true f32v4acc
MAKE_REDUCE_VERTEX ReduceSquareAdd half float true f16v8sqacc
MAKE_REDUCE_VERTEX ReduceSquareAdd float float true f32v4sqacc

MAKE_REDUCE_VERTEX ReduceAdd half half true f16v8acc
MAKE_REDUCE_VERTEX ReduceAdd float half true f32v4acc
MAKE_REDUCE_VERTEX ReduceSquareAdd half half true f16v8sqacc
MAKE_REDUCE_VERTEX ReduceSquareAdd float half true f32v4sqacc

#endif // __IPU__
