// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Supervisor vertex code for Histogram vertices
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
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "poplar/AvailableVTypes.h"
#include "histogramCommon.S"

// vertex state, all offsets are 8-bit
#ifdef VECTOR_AVAIL_SCALED_PTR32
  #define VERTEX_DATA_PTR_OFFSET 0
  #define VERTEX_DATA_COUNT_OFFSET 4
  #define VERTEX_LIMITS_PTR_OFFSET 8
  #define VERTEX_HISTOGRAM_PTR_OFFSET 10
  #define VERTEX_HISTOGRAM_COUNT_OFFSET 12
#else
  #define VERTEX_DATA_PTR_OFFSET 0
  #define VERTEX_DATA_COUNT_OFFSET 4
  #define VERTEX_LIMITS_PTR_OFFSET 8
  #define VERTEX_HISTOGRAM_PTR_OFFSET 12
  #define VERTEX_HISTOGRAM_COUNT_OFFSET 16
#endif

// Register aliases
#define dataPtr m0
#define dataCount m1
#define limPtr m2
#define limCount m3

#define histPtr m4
#define dataCountDiv m5
#define mlink m6

#define histWritePtr m7
#define mscratch m7

#define mloops m8
#define remainder m9

#define workerIdM1 m10

#ifdef VECTOR_AVAIL_SCALED_PTR32
  #define base m11
#else
  #define base mzero
#endif

#define upper a3
#define one01 a4:5
#define one0 a4
#define one1 a5
#define acc01 a6:7
#define acc0 a6
#define acc1 a7

// Naming / name mangling
#define MANGLE_STR(SUFFIX) __runCodelet_popops__HistogramSupervisor___##SUFFIX

//******************************************************************************
// Supervisor entry, worker code in one macro for HALF
//******************************************************************************

.macro INSTANTIATE_HISTOGRAM IS_ABS
DEF_STACK_USAGE 0 MANGLE_STR(half_\IS_ABS\()_true)
.section .text.MANGLE_STR(half_\IS_ABS\()_true)
.global MANGLE_STR(half_\IS_ABS\()_true)
.type MANGLE_STR(half_\IS_ABS\()_true), @function
.align 4
.supervisor

MANGLE_STR(half_\IS_ABS\()_true):

  setzi       $m1, MANGLE_STR(worker_half_\IS_ABS\()_true)
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.worker
.align 8
  nop //rpt alignment
 MANGLE_STR(worker_half_\IS_ABS\()_true):
   {call  $mlink, MANGLE_STR(state_divide_work6)
    setzi $one0, HALF_ONE}

  // Vector width/type specific code - not in the shared function
  // Offset pointer for this worker's work
  {ldb16step $a0,$base, $limPtr+=, $workerIdM1
   sort4x16lo $one0, $one0, $one0}

  // Load limit
  {ldb16step $upper, $base, $limPtr+=, NUM_WORKERS
   mov $one1, $one0}

limit_loop\@:
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
  // misaligned two halves, load them but there may only be 1 to process
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

  // If the repeat count range is < 16 bits, wrap the repeat loop in another loop
  // to increase that range. As we aren't splitting the data size by worker,
  // this limit is quite easily hit. If repeat count range is > 16 bits
  // multiple vertices should be used
3:
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
  min $mscratch, $dataCountDiv, CSR_W_REPEAT_COUNT__VALUE__MASK
  sub $dataCountDiv, $dataCountDiv, $mscratch
  rpt $mscratch, (2f-1f)/8 -1
#else
  rpt $dataCountDiv, (2f-1f)/8 -1
#endif
1:
  CONDITIONAL_ABSv4HALF_BUNDLE \IS_ABS
  {nop
   f16v4cmpgt $a0:1, $upper:BL, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,1
   f16v4acc $a0:1}
2:
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
  brnz $dataCountDiv, 3b
#endif
  // Deal with trailing 0, 1, 2 elements - will be gathered in $a6:7
  {and $mscratch, $dataCount, 2
   mov $a6:7, $azeros}
  brz $mscratch, check_one\@
  // last 2, already in $a0
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a6, $upper, $a0
  and $a6, $a6, $one0
  // Shuffle a1 into a0 in case there is a last one
  mov $a0,$a1

check_one\@:
  and $mscratch, $dataCount, 1
  {brz $mscratch, extract\@
   sort4x16lo $a0, $a0, $azero}
  //last one, with zero in the other half so safe to compare
  // Note this needs to not use a6 as it could come from the misaligned 1 above
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a7, $upper, $a0
  and $a7, $a7, $one0
  sort4x16lo $a7, $a7, $azero

extract\@:
  //accumulate the trailing 1, 2 elements if any
  f16v4acc $a6:7
  // Extract the result from accumulators and store.
  // Load data pointer, next limit etc for the next loop pass
  {ld32 $dataPtr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
   f32v2gina $a0:1, $azeros, 0}
   f32v2gina $a2:3, $azeros, 0
  {ldb16step $upper, $base, $limPtr+=, NUM_WORKERS
   f32v2add $a6:7, $a0:1, $a2:3}
  {ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
   f32add $a6, $a6, $a7}
  st32step $a6, $base, $histPtr+=,NUM_WORKERS
  brnzdec $mloops, limit_loop\@

  brz $workerIdM1, MANGLE_STR(worker_postprocess)
  exitz $mzero

.size MANGLE_STR(half_\IS_ABS\()_true), . -MANGLE_STR(half_\IS_ABS\()_true)
.endm
//******************************************************************************
// Supervisor entry, worker code in one macro for FLOAT
//******************************************************************************

.macro INSTANTIATE_HISTOGRAM_FLOAT IS_ABS
DEF_STACK_USAGE 0 MANGLE_STR(float_\IS_ABS\()_true)
.section .text.MANGLE_STR(float_\IS_ABS\()_true)
.global MANGLE_STR(float_\IS_ABS\()_true)
.type MANGLE_STR(float_\IS_ABS\()_true), @function
.align 4
.supervisor

MANGLE_STR(float_\IS_ABS\()_true):

  setzi       $m1, MANGLE_STR(worker_float_\IS_ABS\()_true)
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.worker
.align 8
MANGLE_STR(worker_float_\IS_ABS\()_true):
   {call  $mlink, MANGLE_STR(state_divide_work6)
    or $one0, $azero, FLOAT_ONE}

  // Vector width/type specific code - not in the shared function
  // Offset pointer for this worker's work
  {ld32step $a0,$base, $limPtr+=, $workerIdM1
   mov $one1, $one0}

  ld32step $upper, $base, $limPtr+=, NUM_WORKERS
limit_loop\@:
  {ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
   mov $acc01, $azeros}
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
  // Div by 2: 2 floats per loop, acc the misaligned one
  {shr $dataCountDiv, $dataCount, 1
   f32add $acc0, $acc0, $a0}
  ld64step $a0:1, $mzero, $dataPtr+=,1
  // If the repeat count range is < 16 bits, wrap the repeat loop in another loop
  // to increase that range. As we aren't splitting the data size by worker,
  // this limit is quite easily hit.  If repeat count range is > 16 bits
  // multiple vertices should be used
3:
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
  min $mscratch, $dataCountDiv, CSR_W_REPEAT_COUNT__VALUE__MASK
  sub $dataCountDiv, $dataCountDiv, $mscratch
  RPT_ALIGNED_TO MANGLE_STR(worker_float_\IS_ABS\()_true) $mscratch
#else
  RPT_ALIGNED_TO MANGLE_STR(worker_float_\IS_ABS\()_true) $dataCountDiv
#endif
1:
  CONDITIONAL_ABSv2FLOAT_BUNDLE \IS_ABS
  {nop
   f32v2cmpgt $a0:1, $upper:B, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,1
   f32v2add $acc01,$acc01, $a0:1}
2:
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
  brnz $dataCountDiv, 3b
#endif

check_one\@:
  and $mscratch, $dataCount, 1
  {brz $mscratch, extract\@
   mov $a1, $azero}
  //last one (Already loaded in the loop)
  CONDITIONAL_ABSv1FLOAT \IS_ABS
  f32cmpgt $a0, $upper, $a0
  and $a1, $a0, $one0

extract\@:
  // The result from the last one is in $a1, if applicable, 0 otherwise
  {ld32 $dataPtr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
   f32add $acc0, $acc0, $a1}
  {ld32step $upper, $base, $limPtr+=, NUM_WORKERS
   f32add $acc0, $acc0, $acc1}
  st32step $acc0, $base, $histPtr+=,NUM_WORKERS
  brnzdec $mloops, limit_loop\@

  brz $workerIdM1, MANGLE_STR(worker_postprocess)
  exitz $mzero

.size MANGLE_STR(float_\IS_ABS\()_true), . -MANGLE_STR(float_\IS_ABS\()_true)
.endm
//******************************************************************************
// Vertex state load function to call on entry
// Plus post process function
//******************************************************************************
.section .text.MANGLE_STR(post_process)
.global MANGLE_STR(post_process)
.type MANGLE_STR(post_process), @function
.align 4
.worker

MANGLE_STR(state_divide_work6):
 // load histogram count from vertex state, divide by num workers
  ldz16 $limCount, $mzero, $mvertex_base, VERTEX_HISTOGRAM_COUNT_OFFSET/2
  add $limCount, $limCount, -1
  // Extract worker ID
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // Loops for this worker: divide by NUM_WORKERS find remainder
  setzi $mscratch, 0xAAAB
  mul $mscratch, $limCount, $mscratch
  shr $mscratch, $mscratch, 18
  mul $remainder, $mscratch, NUM_WORKERS

  // Compare remainder to total number of items to process
  sub $remainder, $limCount, $remainder

  // add 1 if < remainder
  cmpult $mloops, $workerIdM1, $remainder
  add $mloops, $mscratch, $mloops

  // Decrement and exit if no work to do
  brnzdec $mloops, 1f
  exitz $mzero
1:
  // Load and interpret pointers.  Clear the accumulators once, although this
  // is only needed for the half vertex.  Reading them clears them on each
  // loop pass, for the next pass
#ifdef VECTOR_AVAIL_SCALED_PTR32
  {ldz16 $limPtr, $mzero, $mvertex_base, VERTEX_LIMITS_PTR_OFFSET/2
   setzi $a6, ZAACC_BITMASK}
  {shl $limPtr, $limPtr, 2
   uput  $FP_CLR,$a6}
  ldz16 $histPtr, $mzero, $mvertex_base, VERTEX_HISTOGRAM_PTR_OFFSET/2
  shl $histPtr, $histPtr, 2
  setzi $base, TMEM_REGION0_BASE_ADDR
#else
  {ld32 $limPtr, $mzero, $mvertex_base, VERTEX_LIMITS_PTR_OFFSET/4
   setzi $a6, ZAACC_BITMASK}
  {ld32 $histPtr, $mzero, $mvertex_base, VERTEX_HISTOGRAM_PTR_OFFSET/4
   uput  $FP_CLR,$a6}
#endif
  ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4

  // Offset pointer for this worker's work
  ld32step $a0,$base, $histPtr+=, $workerIdM1
  ld32 $dataPtr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
  br  $mlink
//**************************************************************
// Post process - Done by worker zero as it will take the longest
// and we need the whole histogram to be written before starting.
// So far we have found < each limit (except for the top one)
// we need to do hist[N] = hist[N] - hist[N-1]
// But hist[0] is fine, and
// the last one is given by dataCount - hist[N-1]
//
// Work sharing may be possible but seems intricate as each worker would need to read
// any input before it is overwritten.  Also with relatively
// few histogram entries the additional overhead may actually be slower.

.align 8
#ifdef VECTOR_AVAIL_SCALED_PTR32
 nop    // rpt alignment
#endif

MANGLE_STR(worker_postprocess):
  // Reload count and pointer (We don't know where worker 0 ended up pointing)
  ldz16 $limCount, $mzero, $mvertex_base, VERTEX_HISTOGRAM_COUNT_OFFSET/2
  add $limCount, $limCount, -2
#ifdef VECTOR_AVAIL_SCALED_PTR32
  ldz16 $histPtr, $mzero, $mvertex_base, VERTEX_HISTOGRAM_PTR_OFFSET/2
  shl $histPtr, $histPtr, 2
#else
  ld32 $histPtr, $mzero, $mvertex_base, VERTEX_HISTOGRAM_PTR_OFFSET/4
#endif
  // Dummy load to point to the last one, duplicate to get a trailing write ptr
  ld32step $a1, $base, $histPtr+=,$limCount
  add $histWritePtr, $histPtr, 4

  // Cast the data-size and advance the read pointer
  // Read and process the 2nd so that we are ready to write in the loop
  ld32 $a0, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4

  {ld32step $a1, $base, $histPtr+=,-1
   f32fromui32 $a0, $a0}
  {rpt $limCount, (2f-1f)/8 -1
   f32sub $a2, $a0, $a1}
1:
  {ld32step $a1, $base, $histPtr+=,-1
   mov $a0,$a1}
  {st32step $a2, $base, $histWritePtr+=,-1
   f32sub $a2, $a0, $a1}
 2:
  st32step $a2, $base, $histWritePtr+=,-1
  exitz $m0

.size MANGLE_STR(state_divide_work6), . -MANGLE_STR(state_divide_work6)

//******************************************************************************
// Use the macros above to create vertex code
//******************************************************************************
INSTANTIATE_HISTOGRAM true
INSTANTIATE_HISTOGRAM false
INSTANTIATE_HISTOGRAM_FLOAT true
INSTANTIATE_HISTOGRAM_FLOAT false


#endif
/* -------------------------------------------------------------------------- */
