// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__

// Specialisation 2 SCALAR_OUTPUT_SINGLE_INPUT - Overview:
// `partials` is a single edge
// `out` is a single edge and is a scalar
// The vertex treats partials as a 1D array, size {`numPartials`}
// Eg, for numPartials = 16
// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//
// The output will be the sum of all the partials:
// 121,
//
// Constraints:
// Output has no constraints. Partials must be 64bit aligned.
// NumPartials is a 16 bit field
//
// Operation/speed:
// Simply accumulate all of the data, deal with remaining values after
// accumulating in the inner loop.
// This results in an inner loop that takes 2 cycles per 128 bits processed (4 floats or 8 halves).

#include "poplibs_support/TileConstants.hpp"
// Registers
#define SCRATCH         m8

#define ZAACC           a4

// defines
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// Name mangling
#define REDUCE_FLOAT_FLOAT(prefix, specialisation) __runCodelet_popops__##prefix##___popops__\OP\()_float_float_false_##specialisation
#define REDUCE_FLOAT_HALF(prefix, specialisation) __runCodelet_popops__##prefix##___popops__\OP\()_float_half_false_##specialisation

// *****************************************************************************
// Macro describing the float, float and float, half versions of the operation

.macro INSTANTIATE_REDUCE OP INSTRUCTION
.type REDUCE_FLOAT_FLOAT(Reduce,2common), @function
.section .text.REDUCE_FLOAT_FLOAT(Reduce,2common), "ax"
.align 8
 nop
// accumulate values from a single edge
// This code is only used for acc and sqadd, so it's OK to accumulate extra
// zeros
REDUCE_FLOAT_FLOAT(Reduce,2common):
  //$m0: dest offset from MEMBASE in f32's
  //$m1: src offset from MEMBASE in bytes
  //$a0:3 accumulators
  //$a4 preamble/remainders
  //load *src, *dest, n
  ld32 $m0, $mvertex_base, $m15, 0 // load output pointer
  {ldz16 $m1, $mvertex_base, $m15, 2
   setzi $ZAACC, ZAACC_BITMASK}
  {shl $m1, $m1, 3 //
   f32v2add $a0:1, $azeros, $azeros}
  {
  ldz16 $m2, $mvertex_base, $m15, 3
  f32v2add $a2:3, $azeros, $azeros
  }
  {
  brz $m0, 9f
  uput  $FP_CLR, $ZAACC
  }
  // m1 is 64bit aligned
  shr   $m3, $m2, 2
1:// outer loop when rpt count is too small
  // consume quads of aligned elements
#if 1 // remove these 4 lines when T14680 is landed
#undef CSR_W_REPEAT_COUNT__VALUE__MASK // force the mk1 value
#define CSR_W_REPEAT_COUNT__VALUE__MASK 0x0fff
#endif
  min   $m4, $m3, CSR_W_REPEAT_COUNT__VALUE__MASK
  sub   $m3, $m3, $m4
  rpt   $m4, (3f-2f)/8-1
2:
    {
    ld64step $a2:3, $mzero, $m1+=, 1
    \INSTRUCTION $a0:3
    }
    {
    ld64step $a0:1, $mzero, $m1+=, 1
    fnop
    }
3:
  brnz $m3, 1b
  {
  and $m2, $m2, 0x3 // remainder
  \INSTRUCTION $a0:3
  }
  // consume remainder - 0-3 floats possible
  // arrive here by normal fall-through with partial sum in $a0:3, or in the
  // <5 entries case in which case $a0:3 are all zero
  mov $a1, $azero
  // all further accumulation is into the first accumulator
  {
  ld32step $a0, $mzero, $m1+=, 1
  f32v2gina $a4:5, $azeros, 0
  }
  {
  rpt $m2, (2f-1f)/8 - 1
  f32v2add $a2:3, $azeros, $azeros
  }
1:
    {
    ld32step $a0, $mzero, $m1+=, 1
    \INSTRUCTION $a0:3
    }
2:
  f32v2gina $a0:1, $azeros, 0
9:
  f32v2add $a0:1, $a0:1, $a4:5
  {
  br $m10
  f32add $a0, $a0, $a1
  }
.size REDUCE_FLOAT_FLOAT(Reduce,2common), .-REDUCE_FLOAT_FLOAT(Reduce,2common)

.globl REDUCE_FLOAT_FLOAT(Reduce,2)
.type REDUCE_FLOAT_FLOAT(Reduce,2), @function
.section .text.REDUCE_FLOAT_FLOAT(Reduce,2), "ax"
.align 4
REDUCE_FLOAT_FLOAT(Reduce,2):
  call $m10, REDUCE_FLOAT_FLOAT(Reduce,2common)
  st32 $a0, $m0, $mzero
  exitnz $mzero
.size REDUCE_FLOAT_FLOAT(Reduce,2), .-REDUCE_FLOAT_FLOAT(Reduce,2)

.globl REDUCE_FLOAT_HALF(Reduce,2)
.type REDUCE_FLOAT_HALF(Reduce,2), @function
.section .text.REDUCE_FLOAT_HALF(Reduce,2), "ax"
.align 4
REDUCE_FLOAT_HALF(Reduce,2):
  call $m10, REDUCE_FLOAT_FLOAT(Reduce,2common)
  and $m1, $m0, 0x2
  andc $m0, $m0, 0x2
  {
  ld32 $a1, $m0, $mzero
  f32tof16 $a0, $a0
  }
  {
  brz $m1, 1f
  sort4x16hi $a2, $a0, $a1 // a2 = a1[16:31] | a0 [16:31]
  }
  sort4x16lo $a2, $a1, $a0 // a3 = a0[0:15]  | a1[0:15]
1:
  st32 $a2, $m0, $mzero
  exitnz $mzero
.size REDUCE_FLOAT_HALF(Reduce,2), .-REDUCE_FLOAT_HALF(Reduce,2)
.endm
//******************************************************************************
// Use the macro to instantiate the vertices
// Each creates a float,float and float,half version of the operation

INSTANTIATE_REDUCE ReduceAdd f32v4acc
INSTANTIATE_REDUCE ReduceSquareAdd f32v4sqacc


#endif
