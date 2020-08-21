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
//
// The upper unbounded histogram entry is computed by:
//  "num elements" - "num elements less than upper bound"
//
// Each worker processes a portion of the data.  worker 0 deals with any
// misaligned and trailing elements.  The output is a [workers][histogramSize]
// array, where the final result is in the 1st row.
//
// Workers 1-5 use a row of the output to store their results.  In the last
//             column they put "num elements < upper bound"
//
// Worker 0 Accumulates results of comparisons and writes the final result within
//          its loop.  It post processes to find the last histogram result
//          using its "num elements < upper bound", those created by the other
//          workers and the total number of elements.
//
// In the above there is potential for race conditions - see comments in the code
// and be careful optimising. In particular the decision to make worker 0 deal with
// misaligned AND trailing elements ensures that it is slow enough that other
// workers have finished before it reads their results.  It is tempting to try to
// share either the in-loop summing of other worker's results or the post processing
// between workers but this results in some scenarios where worker 0 is either
// faster or very close to faster than the other workers.
//
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
#define histStride m5
#define mlink m6
#define dataPtrConst m6

#define mscratch m7

#define mloops m8
#define remainder m9
#define dataEndPtr m9

#define workerIdM1 m10

#ifdef VECTOR_AVAIL_SCALED_PTR32
  #define base m11
#else
  #define base mzero
#endif

#define previous a2
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
DEF_STACK_USAGE 0 MANGLE_STR(half_\IS_ABS)
.section .text.MANGLE_STR(half_\IS_ABS)
.global MANGLE_STR(half_\IS_ABS)
.type MANGLE_STR(half_\IS_ABS), @function
.align 4
.supervisor

MANGLE_STR(half_\IS_ABS):

  setzi       $m1, MANGLE_STR(worker_half_\IS_ABS)
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.worker
.align 8

 MANGLE_STR(worker_half_\IS_ABS):
   {call  $mlink, MANGLE_STR(state_divide_work_half)
    setzi $one0, HALF_ONE}

  // Load limit
  {ldb16step $upper, $base, $limPtr+=, 1
   sort4x16lo $one0, $one0, $one0}
  {brz $workerIdM1, worker0\@
   mov $one1, $one0}
  // ****IMPORTANT****
  // From this point onwards, worker 0 and the others have a different path.
  // See notes below

  // Produce this worker's pointer: the next 8 byte boundary rounded up, and then
  // offset by the worker ID
  add $dataPtr, $dataPtr, 7
  andc $dataPtr, $dataPtr, 7
  ld64step $azeros,$mzero, $dataPtr+=, $workerIdM1
  {mov $dataPtrConst, $dataPtr
   mov $previous, $azero}

limit_loop\@:
  brneg $mloops, extract\@

  ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS

  RPT_ALIGNED_TO MANGLE_STR(worker_half_\IS_ABS) $mloops
1:
  CONDITIONAL_ABSv4HALF_BUNDLE \IS_ABS
  {nop
   f16v4cmpgt $a0:1, $upper:BL, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS
   f16v4acc $a0:1}
2:
  CONDITIONAL_ABSv4HALF \IS_ABS
  f16v4cmpgt $a0:1, $upper:BL, $a0:1
  and64 $a0:1, $a0:1, $one01
  f16v4acc $a0:1

extract\@:
   // Extract the result from accumulators and store.
  // Load data pointer, next limit etc for the next loop pass
  {mov $dataPtr, $dataPtrConst
   f32v2gina $a0:1, $azeros, 0}
   f32v2gina $a6:7, $azeros, 0
  {ldb16step $upper, $base, $limPtr+=, 1
   f32v2add $a6:7, $a0:1, $a6:7}
  f32add $a6, $a6, $a7
  f32sub $a7, $a6, $previous
  {st32step $a7, $base, $histPtr+=,1
   mov $previous, $a6}
  brnzdec $limCount, limit_loop\@

  // Store "elements < upper limit" for worker 0 to pick up on
  st32 $previous, $base, $histPtr, 0
  exitz $mzero
//////////////////////
// Worker 0 - half

// **IMPORTANT**
// worker 0 will read the results written by each of the other workers
// to combine them and tidy up the last histogram entry at the end.  This
// avoids a separate reduction step.
// This means that worker 0 MUST be slower at reaching the point where it
// reads a result from any other worker.
//
// Workers 1-5 will always do at least the same number of loops as worker 0
// but there can be more instructions to prepare for the loop.
//

// Slowest worker 1-5 : 4 cycles before the loop
//                      7 cycles in the limit loop including the one that writes (abs option)
//
//
// Fastest worker 0:    3 cycles before the loop
//                      4 cycles - one more data vector to process
//                      11 cycles in the limit loop before the one that reads other workers results

// Conclusion : Worker 0 will always be slowest which is OK

.align 8
worker0\@:
  // Create a pointer or zero to compare to so we can decide if to deal with
  // trailing items.
  // Compare to the adjusted dataCount (accounting for misaligned start) to decide
  // if there are trailing ones,  but use the actual data count to find the address
  // of the last one
  and $dataEndPtr, $dataCount, 3
  ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
  {brz $dataEndPtr, 1f
   mov $previous, $azero}
   // Subtract so we point back at the first remainder
  sub $mscratch, $dataCount, $dataEndPtr
  mov  $dataEndPtr, $dataPtr
  ldb16step $azero, $mzero, $dataEndPtr+=, $mscratch
1:

limit_loop_w0\@:
  // Deal with misaligned 0, 1, 2 elements - will be gathered in $a6:7
  and $mscratch, $dataPtr, 2
  {brz $mscratch, check_misaligned_two_w0\@
   mov $a6:7, $azeros}
  // Load 1st misaligned 1, clear half of it incase there is only 1
  ldb16step $a0, $mzero, $dataPtr+=,1
  sort4x16lo $a0, $a0, $azero
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a6, $upper, $a0
  and $a6, $a6, $one0
  // Result is in a6, and the count is adjusted
  // Note that if we branch when checking misaligned 2 the result of misaligned 1
  // is in a6 and will survive until the end where we will accumulate it
  {sub $dataCount, $dataCount, 1
   sort4x16lo $a6, $a6, $azero}

check_misaligned_two_w0\@:
  and $mscratch, $dataPtr, 4
  brz $mscratch, data_loop_continue_w0\@
  // misaligned two halves, but there may be 0 or 1 to process
  cmpult $mscratch, $dataCount, 2
  brnz $mscratch, check_one_w0\@
  ld32step $a0, $mzero, $dataPtr+=,1
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a7, $upper, $a0
  // Result is in a7, and the count is adjusted
  {sub $dataCount, $dataCount, 2
   and $a7, $a7, $one0}
data_loop_continue_w0\@:
  //  acc the misaligned entries
  {brneg $mloops, check_two_w0\@
   f16v4acc $a6:7}
  ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS

  RPT_ALIGNED_TO worker0\@ $mloops
1:
  CONDITIONAL_ABSv4HALF_BUNDLE \IS_ABS
  {nop
   f16v4cmpgt $a0:1, $upper:BL, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS
   f16v4acc $a0:1}
2:
  CONDITIONAL_ABSv4HALF \IS_ABS
  f16v4cmpgt $a0:1, $upper:BL, $a0:1
  and64 $a0:1, $a0:1, $one01
  f16v4acc $a0:1

check_two_w0\@:
  // Deal with trailing 0, 1, 2 elements - will be gathered in $a6:7
  {brz $dataEndPtr, extract_w0\@
   mov $a6:7, $azeros}
  and $mscratch, $dataCount, 2
  mov $dataPtr, $dataEndPtr
  brz $mscratch, check_one_w0\@
  // last 2
  ld32step $a0, $mzero, $dataPtr+=, 1
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a6, $upper, $a0
check_one_w0\@:
  and $mscratch, $dataCount, 1
  brz $mscratch, extract_accumulate_w0\@
  // last one
  // Note this needs to not use a6 as it could come from the misaligned 1 above
  ldb16 $a0, $mzero, $dataPtr, 0
  CONDITIONAL_ABSv2HALF \IS_ABS
  f16v2cmpgt $a7, $upper, $a0
extract_accumulate_w0\@:
  and64 $a6:7, $a6:7, $one01
  sort4x16lo $a7, $a7, $azero

extract_w0\@:
  //accumulate the trailing 1, 2 elements if any
  {mov $mscratch, $histPtr
   f16v4acc $a6:7}
  // Extract the result from accumulators and store.
  // Load data pointer, next limit etc for the next loop pass
  {ld32 $dataPtr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
   f32v2gina $a0:1, $azeros, 0}
   // Advance pointer to the next row of results ready to combine them
  {ld32step $azero, $base, $mscratch+=, $histStride
   f32v2gina $a6:7, $azeros, 0}
  {ldb16step $upper, $base, $limPtr+=, 1
   f32v2add $a6:7, $a0:1, $a6:7}
  {ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
   f32add $a6, $a6, $a7}

  // **IMPORTANT**
  // At this point we are reading results produced by other workers.

  // Accumulate the values produced by the other workers along with worker 0
  // result to reduce and provide all but the last histogram entry
  {ld32step $a0, $base, $mscratch+=, $histStride
   f32sub $a7, $a6, $previous}

  RPT_ALIGNED_TO 2b 4
1:
  {ld32step $a0, $base, $mscratch+=, $histStride
   f32add $acc1, $acc1, $a0}
2:
  f32add $acc1, $acc1, $a0
  {st32step $acc1, $base, $histPtr+=,1
   mov $previous, $acc0}
   brnzdec $limCount, limit_loop_w0\@

  bri MANGLE_STR(post_process)

.size MANGLE_STR(half_\IS_ABS), . -MANGLE_STR(half_\IS_ABS)
.endm
//******************************************************************************
// Supervisor entry, worker code in one macro for FLOAT
//******************************************************************************

.macro INSTANTIATE_HISTOGRAM_FLOAT IS_ABS
DEF_STACK_USAGE 0 MANGLE_STR(float_\IS_ABS)
.section .text.MANGLE_STR(float_\IS_ABS)
.global MANGLE_STR(float_\IS_ABS)
.type MANGLE_STR(float_\IS_ABS), @function
.align 4
.supervisor

MANGLE_STR(float_\IS_ABS):

  setzi       $m1, MANGLE_STR(worker_float_\IS_ABS)
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr

.worker
.align 8
MANGLE_STR(worker_float_\IS_ABS):
   {call  $mlink, MANGLE_STR(state_divide_work_float)
    or $one0, $azero, FLOAT_ONE}

  {ld32step $upper, $base, $limPtr+=, 1
   mov $one1, $one0}
  {brz $workerIdM1, worker0\@
   mov $previous, $azero}
  // ****IMPORTANT****
  // From this point onwards, worker 0 and the others have a different path.
  // See notes below

  // Produce this worker's pointer: the next 8 byte boundary rounded up, and then
  // offset by the worker ID
  add $dataPtr, $dataPtr, 7
  andc $dataPtr, $dataPtr, 7
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  mov $dataPtrConst, $dataPtr

limit_loop\@:
  {brneg $mloops, extract\@
   mov $acc01, $azeros}
  ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS
  RPT_ALIGNED_TO MANGLE_STR(worker_float_\IS_ABS) $mloops
1:
  CONDITIONAL_ABSv2FLOAT_BUNDLE \IS_ABS
  {nop
   f32v2cmpgt $a0:1, $upper:B, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS
   f32v2add $acc01,$acc01, $a0:1}
2:
  CONDITIONAL_ABSv2FLOAT \IS_ABS
  f32v2cmpgt $a0:1, $upper:B, $a0:1
  and64 $a0:1, $a0:1, $one01
  f32v2add $acc01,$acc01, $a0:1

extract\@:
  {ld32step $upper, $base, $limPtr+=, 1
   f32add $acc0, $acc0, $acc1}
  {mov $dataPtr, $dataPtrConst
   f32sub $acc1, $acc0, $previous}
  {mov $mscratch, $histPtr
   mov $previous, $acc0}

  st32step $acc1, $base, $histPtr+=,1
  brnzdec $limCount, limit_loop\@

  // Store "elements < upper limit" for worker 0 to pick up on
  st32 $previous, $base, $histPtr, 0
  exitz $mzero

///////////////////////
// Worker 0 - float

// **IMPORTANT**
// worker 0 will read the results written by each of the other workers
// to combine them and tidy up the last histogram entry at the end.  This
// avoids a separate reduction step.
// This means that worker 0 MUST be slower at reaching the point where it
// reads a result from any other worker.
//
// Workers 1-5 will always do at least the same number of loops as worker 0
// but there can be more instructions to prepare for the loop.
//

// Slowest worker 1-5 : 4 cycles before the loop
//                      3 cycles in the limit loop including the one that writes (abs option)
//
//
// Fastest worker 0:    2 cycles before the loop
//                      4 cycles - one more data vector to process
//                      8 cycles in the limit loop before the one that reads other workers results

// Conclusion : Worker 0 will always be slowest which is OK

.align 8
worker0\@:

  // Create a pointer or zero to compare to, so we pick up a trailing element.
  // Compare to the adjusted dataCount (accounting for misaligned start) to decide
  // if there is a trailing one,  but use the actual data count to find the address
  // of the last one
  and $dataEndPtr, $dataCount, 1
  brz $dataEndPtr, 1f
  ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
  mov  $dataEndPtr, $dataPtr
  ld32step $azero, $mzero, $dataEndPtr+=, $dataCount
  sub $dataEndPtr, $dataEndPtr, 4
1:

limit_loop_w0\@:
  // Misaligned 1st element - will be in $a0 if processed
  {and $mscratch, $dataPtr, 4
  mov $acc01, $azeros}
  {brz $mscratch, 1f
   mov $a0, $azero}
  ld32step $a0, $mzero, $dataPtr+=,1
  CONDITIONAL_ABSv1FLOAT \IS_ABS
  f32cmpgt $a0, $upper, $a0
  and $a0, $a0, $one0
1:
  {brneg $mloops, check_one_w0\@
   f32add $acc0, $acc0, $a0}
  ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS
  RPT_ALIGNED_TO worker0\@ $mloops
1:
  CONDITIONAL_ABSv2FLOAT_BUNDLE \IS_ABS
  {nop
   f32v2cmpgt $a0:1, $upper:B, $a0:1} //(upper > data) == (data < upper)
  {nop
   and64 $a0:1, $a0:1, $one01}
  {ld64step $a0:1, $mzero, $dataPtr+=,NUM_WORKERS
   f32v2add $acc01,$acc01, $a0:1}
2:
  CONDITIONAL_ABSv2FLOAT \IS_ABS
  f32v2cmpgt $a0:1, $upper:B, $a0:1 //(upper > data) == (data < upper)
  and64 $a0:1, $a0:1, $one01
  f32v2add $acc01,$acc01, $a0:1

check_one_w0\@:
  {brz $dataEndPtr, extract_w0\@
   mov $a1, $azero}
  //last one
  ld32 $a0, $mzero, $dataEndPtr, 0
  CONDITIONAL_ABSv1FLOAT \IS_ABS
  f32cmpgt $a0, $upper, $a0
  and $a1, $a0, $one0

extract_w0\@:
  // The result from the last one is in $a1, if applicable, 0 otherwise
  {ld32 $dataPtr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
   f32add $acc0, $acc0, $a1}
  {ld32step $upper, $base, $limPtr+=, 1
   f32add $acc0, $acc0, $acc1}

  {mov $mscratch, $histPtr
   f32sub $acc1, $acc0, $previous}
  // Advance pointer to the next row of outputs: the next worker's result
  {ld32step $a0, $base, $mscratch+=, $histStride
   mov $previous, $acc0}

  // **IMPORTANT**
  // At this point we are reading results produced by other workers.

  // Accumulate the values produced by the other workers along with worker 0
  // result to reduce and provide all but the last histogram entry
  ld32step $a0, $base, $mscratch+=, $histStride

  RPT_ALIGNED_TO 2b 4
1:
  {ld32step $a0, $base, $mscratch+=, $histStride
   f32add $acc1, $acc1, $a0}
2:
  f32add $acc1, $acc1, $a0
  st32step $acc1, $base, $histPtr+=,1
  brnzdec $limCount, limit_loop_w0\@

  bri MANGLE_STR(post_process)

.size MANGLE_STR(float_\IS_ABS), . -MANGLE_STR(float_\IS_ABS)
.endm

//******************************************************************************
// Vertex state load function to call on entry
// Plus post process function
//******************************************************************************
.macro DIVIDE_WORK TYPE
.section .text.MANGLE_STR(divide_work_\TYPE\())
.global MANGLE_STR(divide_work_\TYPE\())
.type MANGLE_STR(divide_work_\TYPE\()), @function
.align 4
.worker
.ifc "\TYPE", "half"
  .equ SHIFTS_TO_DIV, 20
  .equ DIVISOR, 24
  .equ SHIFTS_FOR_GRAINSIZE, 2
  .equ SHIFTS_FOR_REMAINDER, 1

.else
  .equ SHIFTS_TO_DIV, 19
  .equ DIVISOR, 12
  .equ SHIFTS_FOR_GRAINSIZE, 1
  .equ SHIFTS_FOR_REMAINDER, 2

.endif

MANGLE_STR(state_divide_work_\TYPE\()):
 // load data count from vertex state, divide to share work
  ld32 $dataCount, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
  // Extract worker ID
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // Loops for this worker:
  // Ignore any misaligned elements as they aren't dealt with in a loop
  ld32 $dataPtr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
  and $mscratch, $dataPtr, 0x7
  shr $mscratch, $mscratch, SHIFTS_FOR_REMAINDER

.ifc "\TYPE", "half"
  // half only - misalignment 0 : we will consume 0 in processing the misaligned ones
  // misalign 3 -> consume 1
  // misalign 2 -> consume 2
  // misalign 1 -> consume 3
  // This can end up negative, if so skip the work division as it will fail
  brz $mscratch, 1f
  sub $mscratch, 4, $mscratch
1:
  sub $dataCount, $dataCount, $mscratch
  brneg $dataCount, 1f
.else
 sub $dataCount, $dataCount, $mscratch
.endif

  // Loops for this worker: divide by 12 or 24, find remainder
  setzi $mscratch, 0xAAAB
  mul $mscratch, $dataCount, $mscratch
  shr $mscratch, $mscratch, SHIFTS_TO_DIV
  mul $remainder, $mscratch, DIVISOR

  // Compare remainder to total number of items to process
  sub $remainder, $dataCount, $remainder
  shr $remainder, $remainder, SHIFTS_FOR_GRAINSIZE
  // add 1 if < remainder
  cmpult $mloops, $workerIdM1, $remainder
  add $mloops, $mscratch, $mloops
  // Minus 1 as we unroll the loop and don't want any overreads
  sub $mloops, $mloops, 1
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
  ldz16 $histStride, $mzero, $mvertex_base, VERTEX_HISTOGRAM_COUNT_OFFSET/2
  // Offset pointer for this worker's results
  mul $mscratch, $workerIdM1, $histStride
  ld32step $mzero, $base, $histPtr+=, $mscratch

  // -1 as limit count = histogram count -1, another -1 as using brnzdec
  add $limCount, $histStride, -2

  br  $mlink

.size MANGLE_STR(divide_work_\TYPE\()), . -MANGLE_STR(state_divide_work_\TYPE\())
.endm


.section .text.MANGLE_STR(post_process)
.global MANGLE_STR(post_process)
.type MANGLE_STR(post_process), @function
.align 4
.worker
MANGLE_STR(post_process):
  // Find the last histogram entry.  Each worker stored its `previous` result
  // in the last entry.  Previous == number of elements < highest limit.
  // The last histogram result = total elems - sum(previous for all workers)
  // $previous still contains this worker's result

  // Cast the total data count to a float
  ld32 $a0, $mzero, $mvertex_base, VERTEX_DATA_COUNT_OFFSET/4
  {mov $mscratch, $histPtr
   f32fromui32 $acc0, $a0}
  // Dummy read to offset to the next workers row
  {ld32step $a1, $base, $mscratch+=, $histStride
   f32sub $acc0, $acc0, $previous}
  // **IMPORTANT** This is the first read of another worker's result, but
  //               worker 0 is slower so this is OK
  ld32step $a1, $base, $mscratch+=, $histStride
  {ld32step $a1, $base, $mscratch+=, $histStride
   f32sub $acc0, $acc0, $a1}
  {ld32step $a1, $base, $mscratch+=, $histStride
   f32sub $acc0, $acc0, $a1}
  {ld32step $a1, $base, $mscratch+=, $histStride
   f32sub $acc0, $acc0, $a1}
  {ld32step $a1, $base, $mscratch+=, $histStride
   f32sub $acc0, $acc0, $a1}
  f32sub $acc0, $acc0, $a1
  st32 $acc0, $base, $histPtr, 0

  exitz $mzero

.size MANGLE_STR(post_process), . -MANGLE_STR(post_process)
//******************************************************************************
// Use the macros above to create vertex code
//******************************************************************************
DIVIDE_WORK float
DIVIDE_WORK half

INSTANTIATE_HISTOGRAM true_false
INSTANTIATE_HISTOGRAM false_false
INSTANTIATE_HISTOGRAM_FLOAT true_false
INSTANTIATE_HISTOGRAM_FLOAT false_false


#endif
/* -------------------------------------------------------------------------- */
