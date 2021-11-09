// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#ifdef VECTOR_AVAIL_SCALED_PTR64
#define OUT_OFFSET           0
#define IN_OFFSET            4
#define NUM_PARTIALS_OFFSET  6
#else
#define OUT_OFFSET           0
#define IN_OFFSET            4
#define NUM_PARTIALS_OFFSET  8
#endif

// Registers
#define SCRATCH         m8

#define ZAACC           a4

// Name mangling

#define REDUCE_FLOAT_FLOAT(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_float_float_false_popops__ReductionSpecialisation__##specialisation

#define REDUCE_FLOAT_HALF(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_float_half_false_popops__ReductionSpecialisation__##specialisation

// *****************************************************************************
// Macro describing the float, float and float, half versions of the operation
#ifdef VECTOR_AVAIL_SCALED_PTR64
#define NOP_FOR_RPT  nop
#else
#define NOP_FOR_RPT
#endif

.macro INSTANTIATE_REDUCE OP INSTRUCTION

// accumulate values from a single edge
// This code is only used for acc and sqadd, so it's OK to accumulate extra
// zeros
FN_WORKER_ENTRY_POINT REDUCE_FLOAT_FLOAT(Reduce,2common) 8 NOP_FOR_RPT
  //$m0: dest offset from MEMBASE in f32's
  //$m1: src offset from MEMBASE in bytes
  //$a0:3 accumulators
  //$a4 preamble/remainders
  //load *src, *dest, n
  {ld32 $m0, $mvertex_base, $m15, OUT_OFFSET/4 // load output pointer
   setzi $ZAACC, ZAACC_BITMASK}
#ifdef VECTOR_AVAIL_SCALED_PTR64
  {ldz16 $m1, $mvertex_base, $m15, IN_OFFSET/2
   f32v2add $a0:1, $azeros, $azeros}
  shl $m1, $m1, SCALED_PTR64_SHIFTS
#else
  {ld32 $m1, $mvertex_base, $m15, IN_OFFSET/4
   f32v2add $a0:1, $azeros, $azeros}
#endif
  {
  ldz16 $m2, $mvertex_base, $m15, NUM_PARTIALS_OFFSET/2
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
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
  // m3 TOTAL ELEMS, from m2 >> 2, m2 from ldz16 (unsigned short)
  // m4 RPT SIZE
  min   $m4, $m3, CSR_W_REPEAT_COUNT__VALUE__MASK
  sub   $m3, $m3, $m4
  rpt   $m4, (3f-2f)/8-1
#else
  // Since m3 (TOTAL ELEMS), can't be larger than 16 bit
  // We are safe to do a rpt loop without checking the mask
  rpt   $m3, (3f-2f)/8-1
#endif
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
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
  brnz $m3, 1b
#else
  nop // for rpt align
#endif
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
FN_SIZE REDUCE_FLOAT_FLOAT(Reduce,2common)

FN_WORKER_ENTRY_POINT REDUCE_FLOAT_FLOAT(Reduce,SCALAR___OUTPUT___SINGLE___INPUT)
  call $m10, REDUCE_FLOAT_FLOAT(Reduce,2common)
  st32 $a0, $m0, $mzero
  exitnz $mzero
FN_SIZE REDUCE_FLOAT_FLOAT(Reduce,SCALAR___OUTPUT___SINGLE___INPUT)

FN_WORKER_ENTRY_POINT REDUCE_FLOAT_HALF(Reduce,SCALAR___OUTPUT___SINGLE___INPUT)
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
FN_SIZE REDUCE_FLOAT_HALF(Reduce,SCALAR___OUTPUT___SINGLE___INPUT)
.endm
//******************************************************************************
// Use the macro to instantiate the vertices
// Each creates a float,float and float,half version of the operation

INSTANTIATE_REDUCE ReduceAdd f32v4acc
INSTANTIATE_REDUCE ReduceSquareAdd f32v4sqacc


#endif
