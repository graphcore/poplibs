// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Worker2D vertex code for Histogram vertices
//
// General maths: When comparing float or half to another float or half using the
// comparison instructions we get false = 0, true (half) = 0xffff
// true (float) = 0xffffffff
// If we AND this with 1.0 in either format we get 1.0 when true and 0.0
// when false.  This can be vectorised.
// Then accumulating the results gives the number of elements < any upper limit.
// This would more naturally be kept as an unsigned integer but on tile the
// exact integer range of float is plenty to keep track (Half is not so accumulate
// into the ACCUMULATOR registers) and avoids casting.
// The vertex outputs float, but the exact integer range of floats
// may not always be enough to then accumulate over tiles.  When building this
// into a library function we may decide to cast to unsigned on the output of this
// vertex or cast using existing poplibs functions.  A reduction will follow...
/* -------------------------------------------------------------------------- */
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "poplar/AvailableVTypes.h"
#include "MathConstants.S"
#include "CommonPoplibsMacros.h.S"
#include "histogramCommon.S"

// vertex state, all offsets are 8-bit
#ifdef VECTOR_AVAIL_SCALED_PTR32
  #define VERTEX_DATA_SPAN_OFFSET 0
  #define VERTEX_LIMITS_PTR_OFFSET 4
  #define VERTEX_HISTOGRAM_PTR_OFFSET 6
  #define VERTEX_HISTOGRAM_COUNT_OFFSET 8
#else
  #define VERTEX_DATA_SPAN_OFFSET 0
  #define VERTEX_LIMITS_PTR_OFFSET 4
  #define VERTEX_HISTOGRAM_PTR_OFFSET 8
  #define VERTEX_HISTOGRAM_COUNT_OFFSET 12
#endif
// Register aliases
#define dataPtr m0
#define base m0

#define histPtr m1
#define limPtr m2
#define limCount m3

#define dataCount m4
#define dataCountDiv m5
#define ptrMask m6
#define mscratch m7

#define dataVectCountConst m8
#define dataVectPtrConst m9
#define dataVectCount m10
#define dataVectPtr m11

#define previous a2
#define upper a3
#define one01 a4:5
#define one0 a4
#define one1 a5
#define acc01 a6:7
#define acc0 a6
#define acc1 a7

// Naming / name mangling
#define MANGLE_STR(SUFFIX) __runCodelet_popops__Histogram2D___##SUFFIX

//******************************************************************************
.macro LOAD_VERTEX_STATE
  // Load and unpack 1st level of 2D vector pointer
  ld32 $dataVectPtrConst, $mzero, $mvertex_base, VERTEX_DATA_SPAN_OFFSET/4
  shr $dataVectCountConst, $dataVectPtrConst, SHORT_SPAN_PTR_SIZE
  sub $dataVectCountConst, $dataVectCountConst, 1
  setzi $ptrMask, SHORT_SPAN_PTR_MASK
  // Initialise the previous sum to zero
  {and $dataVectPtrConst, $dataVectPtrConst, $ptrMask
   mov $previous, $azero}

#ifdef VECTOR_AVAIL_SCALED_PTR32
  // Load and unpack limits, histogram pointers
  ldz16 $limPtr, $mzero, $mvertex_base, VERTEX_LIMITS_PTR_OFFSET/2
  shl $limPtr, $limPtr, 2
  ldz16 $histPtr, $mzero, $mvertex_base, VERTEX_HISTOGRAM_PTR_OFFSET/2
  shl $histPtr, $histPtr, 2
  // Add base as we have better uses for the register
  setzi $base, TMEM_REGION0_BASE_ADDR
  add $histPtr, $histPtr, $base
  add $limPtr, $limPtr, $base
#else
  ld32 $limPtr, $mzero, $mvertex_base, VERTEX_LIMITS_PTR_OFFSET/4
  ld32 $histPtr, $mzero, $mvertex_base, VERTEX_HISTOGRAM_PTR_OFFSET/4
#endif
.endm
//******************************************************************************
// macro for HALF
//******************************************************************************

.macro INSTANTIATE_HISTOGRAM_HALF IS_ABS
FN_WORKER_ENTRY_POINT MANGLE_STR(half_\IS_ABS) 8
  LOAD_VERTEX_STATE
  // Setup working pointer, clear accumulators. (In the loop, reading the
  // accumulators clears them for the next pass)
  {mov $dataVectCount, $dataVectCountConst
   setzi $a6, ZAACC_BITMASK}
  {mov $dataVectPtr, $dataVectPtrConst
   uput  $FP_CLR,$a6}

  // Setup a broadcast constant == {1.0h, 1.0h, 1.0h, 1.0h}
  {ldz16 $limCount, $mzero, $mvertex_base, VERTEX_HISTOGRAM_COUNT_OFFSET/2
   setzi $one0, HALF_1_0}
  {add $limCount, $limCount, -2
   sort4x16lo $one0, $one0, $one0}
  // load the first limit, used as an upper limit
  {ldb16step $upper, $mzero, $limPtr+=, 1
   mov $one1, $one0}

limit_loop\@:
data_loop\@:
  ld32step $dataPtr, $mzero, $dataVectPtr+=, 1
  shr $dataCount, $dataPtr, SHORT_SPAN_PTR_SIZE
  and $dataPtr, $dataPtr, $ptrMask

  // Deal with misaligned 0, 1, 2 elements - will be gathered in $a6:7
  and $mscratch, $dataPtr, 2
  {brz $mscratch, check_misaligned_two\@
   mov $a6:7, $azeros}
  // Load 1st misaligned 1, clear half of it incase there is only 1
  ldb16step $a0, $mzero, $dataPtr+=,1
  sort4x16lo $a0, $a0, $azero
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a6, $upper, $a0
  {sub $dataCount, $dataCount,1
   and $a6, $a6, $one0}
  // Result is in a6, and the count is adjusted
  // Note that if we branch when checking misaligned 2 the result of misaligned 1
  // is in a6 and will survive until the end where we will accumulate it
  sort4x16lo $a6, $a6, $azero

check_misaligned_two\@:
  and $mscratch, $dataPtr, 4
  brz $mscratch, data_loop_continue\@
  // misaligned two halves, load them but there may only be 0 or 1 to process
  ld32step $a0, $mzero, $dataPtr+=,1
  cmpult $mscratch, $dataCount, 2
  brnz $mscratch, check_one\@

  CONDITIONAL_ABSv2HALF \IS_ABS
  {sub $dataCount, $dataCount,2
  f16v2cmpgt $a7, $upper, $a0}
  // Result is in a7, and the count is adjusted
  and $a7, $a7, $one0
data_loop_continue\@:
  // Div by 4: 4 halves per loop, acc the misaligned entries
  {shr $dataCountDiv, $dataCount, 2
   f16v4acc $a6:7}
  ld64step $a0:1, $mzero, $dataPtr+=,1

  RPT_ALIGNED_TO MANGLE_STR(half_\IS_ABS) $dataCountDiv
1:
  CONDITIONAL_ABSv4HALF_BUNDLE \IS_ABS
  {nop
   f16v4cmpgt $a6:7, $upper:BL, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a6:7, $a6:7, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,1
   f16v4acc $a6:7}
2:
  // Deal with trailing 0, 1, 2 elements - will be gathered in $a6:7
  and $mscratch, $dataCount, 2
  {brz $mscratch, check_one\@
   mov $a6:7, $azeros}
  // last 2, already in $a0
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a6, $upper, $a0
  and $a6, $a6, $one0
  // Shuffle a1 into a0 in case there is a last one
  mov $a0,$a1

check_one\@:
  and $mscratch, $dataCount, 1
  {brz $mscratch, data_loop_end\@
   sort4x16lo $a0, $a0, $azero}
  //last one, with zero in the other half so safe to compare
  // Note this needs to not use a6 as it could come from the misaligned 1 above
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a7, $upper, $a0
  and $a7, $a7, $one0
  sort4x16lo $a7, $a7, $azero

data_loop_end\@:
  // Loop and accumulate the trailing 1, 2 elements if any
  {brnzdec $dataVectCount, data_loop\@
   f16v4acc $a6:7}
  // Extract and store the less than result
  // Load the dataVector pointer, count for next time
  {mov $dataVectCount, $dataVectCountConst
   f32v2gina $a0:1, $azeros, 0}
  {mov $dataVectPtr, $dataVectPtrConst
   f32v2gina $a6:7, $azeros, 0}
  // load the next limit, used as an upper limit
  {ldb16step $upper, $mzero, $limPtr+=, 1
   f32v2add $a0:1, $a0:1, $a6:7}
  f32add $a0, $a0, $a1

  // Subtract the previous sum
  f32sub $a1, $a0, $previous

  {st32step $a1, $mzero, $histPtr+=,1
   mov $previous, $a0}

  brnzdec $limCount, limit_loop\@

  bri MANGLE_STR(top_entry)

FN_SIZE MANGLE_STR(half_\IS_ABS)
.endm

//******************************************************************************
// Worker code in a macro for FLOAT
//******************************************************************************

.macro INSTANTIATE_HISTOGRAM_FLOAT IS_ABS
FN_WORKER_ENTRY_POINT MANGLE_STR(float_\IS_ABS) 8

  LOAD_VERTEX_STATE

  // Setup a broadcast constant == {1.0f, 1.0f}
  {ldz16 $limCount, $mzero, $mvertex_base, VERTEX_HISTOGRAM_COUNT_OFFSET/2
   or $one0, $azero, FLOAT_1_0}
  {add $limCount, $limCount, -2
   mov $one1, $one0}

  // Setup working pointer
  mov $dataVectCount, $dataVectCountConst
  mov $dataVectPtr, $dataVectPtrConst

limit_loop\@:
  // load the next limit, used as an upper limit, clear accumulation registers.
  {ld32step $upper, $mzero, $limPtr+=, 1
   mov $acc01, $azeros}

data_loop\@:
  ld32step $dataPtr, $mzero, $dataVectPtr+=, 1
  shr $dataCount, $dataPtr, SHORT_SPAN_PTR_SIZE
  and $dataPtr, $dataPtr, $ptrMask

  // Misaligned 1st element - will be in $a0 if processed
  and $mscratch, $dataPtr, 4
  {brz $mscratch, 1f
   mov $a0, $azero}
  ld32step $a0, $mzero, $dataPtr+=,1
  CONDITIONAL_ABSv1FLOAT \IS_ABS
  {sub $dataCount, $dataCount, 1
   f32cmpgt $a0, $upper, $a0}
   and $a0, $a0, $one0
1:
  // Div by 2: 2 floats per loop
  shr $dataCountDiv, $dataCount, 1
  // Load 1st loop input, acc the misaligned entry
  {ld64step $a0:1, $mzero, $dataPtr+=,1
   f32add $acc0, $acc0, $a0}

  RPT_ALIGNED_TO MANGLE_STR(float_\IS_ABS) $dataCountDiv
1:
  CONDITIONAL_ABSv2FLOAT_BUNDLE \IS_ABS
  {nop
   f32v2cmpgt $a0:1, $upper:B, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,1
   f32v2add $acc01, $acc01, $a0:1}
2:

check_one\@:
  and $mscratch, $dataCount, 1
  {brz $mscratch, data_loop_end\@
   mov $a1, $azero}
  //last one
  CONDITIONAL_ABSv1FLOAT \IS_ABS
  f32cmpgt $a0, $upper, $a0
  and $a1, $a0, $one0

data_loop_end\@:
  // The result from the last one is in $a1, if applicable, 0 otherwise
  {brnzdec $dataVectCount, data_loop\@
   f32add $acc0, $acc0, $a1}

  // Combine and store the result, load the pointer for next time
  {mov $dataVectCount, $dataVectCountConst
   f32add $a0, $acc0, $acc1}

  // Subtract the previous sum
  {mov $dataVectPtr, $dataVectPtrConst
   f32sub $a1, $a0, $previous}
  {st32step $a1, $mzero, $histPtr+=,1
   mov $previous, $a0}
  brnzdec $limCount, limit_loop\@

  bri MANGLE_STR(top_entry)

FN_SIZE MANGLE_STR(float_\IS_ABS)
.endm

//******************************************************************************
// A common function to find the top entry and then exit
//******************************************************************************

#define elemCount m6

FN_WORKER_ENTRY_POINT MANGLE_STR(top_entry) 8 nop
  // Initialise element count, accumulate all data sizes.  Quicker here
  // as although we could hand in onto the main loop it would be calculated
  // repeatedly
  // Count was already predecremented so fetch the first to initialise
  // elemCount, and then it is correct for use with rpt
  ld32step $dataPtr, $mzero, $dataVectPtrConst+=, 1
  shr $elemCount, $dataPtr, SHORT_SPAN_PTR_SIZE
  rpt $dataVectCountConst, (2f-1f)/8 -1
1:
  {ld32step $dataPtr, $mzero, $dataVectPtrConst+=, 1
   fnop}
  {shr $dataCount, $dataPtr, SHORT_SPAN_PTR_SIZE
   fnop}
  // Accumulate up total elements across all data vectors
  {add $elemCount, $elemCount, $dataCount
   fnop}
2:
  // Store then load the total number of elements as we have an atom
  // instruction but no mtoa. Just using the eventual result location to do this
  // as we know it's available
  st32 $elemCount, $mzero, $histPtr, 0
  // Load the unsigned count, cast to float
  ld32 $a0, $mzero, $histPtr, 0
  f32fromui32 $a0, $a0
  // Subtract previous "less than" sum and store as the final result, overwriting
  // the temp elemCount
  f32sub  $a0, $a0, $previous
  st32 $a0, $mzero, $histPtr, 0

  exitz $mzero


FN_SIZE MANGLE_STR(top_entry)


//******************************************************************************
// Use the macros above to create vertex code
//******************************************************************************
INSTANTIATE_HISTOGRAM_HALF true
INSTANTIATE_HISTOGRAM_HALF false
INSTANTIATE_HISTOGRAM_FLOAT true
INSTANTIATE_HISTOGRAM_FLOAT false

#endif
/* -------------------------------------------------------------------------- */
