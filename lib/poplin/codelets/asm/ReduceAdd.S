// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
//****************************************************************************
// poplin ReduceAdd codelets
//
// Overview:
// We expect to be passed an array of pointers to partials.  Each of these is in
// practice often a long vector. We could think of each partial pointer as
// pointing to a row of an array.  The ReduceAdd function sums the values in
// each column of the array.
//
// partials[0][NUM ELEMS] = a0 b0 c0 d0 e0
// partials[1][NUM ELEMS] = a1 b1 c1 d1 e1
// partials[2][NUM ELEMS] = a2 b2 c1 d2 e2
//
// Output[NUM_ELEMS] = (a0+a1+a2), (b0+b1+b2), (c0+c1+c2)
//
// Workers will sum a group of 4 (float input) or 8 (half input)
// columns each, being assigned:
// worker0 : columns 0-3
// worker1 : columns 4-7
// .....
// worker5 : columns 20-23
// worker0 : columns 24-27
// ...
//
// Higher numbered workers are therefore likely to finish 1st (or all at about
// the same time), so higher numbered workers deal with the remaining 1-7
// columns.
//****************************************************************************
#ifdef __IPU__
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
#define COMPACT_VECTOR_TYPES_AVAILABLE 1
#else
#define COMPACT_VECTOR_TYPES_AVAILABLE 0
#endif

//****************************************************************************
// Supervisor registers
#define S_VERTEX_BASE   m0
#define S_WORKER_ENTRY  m4
//****************************************************************************
// Worker Registers
#define PARTIALS    m0
#define ELEMS_LOOPS m1

#define NUM_PARTIALS m2
#define NUM_ELEMS    m3
#define OUT_PTR      m4
#define PARTIALS_PTR m5
#define BASE         m6
#if COMPACT_VECTOR_TYPES_AVAILABLE
#define MEM_BASE     m7
#else
#define MEM_BASE mzero
#endif
#define OUT_PTR_STORE m8
#define SCRATCH2      m9
#define WKR_ID       m10
#define SCRATCH m11

#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1

#define VAL34   a2:3
#define VAL3    a2
#define VAL4    a3

#define VAL14   a0:3
#define ZAACC   a4

#if COMPACT_VECTOR_TYPES_AVAILABLE
//****************************************************************************
// The input structure parameters:
// partials :   16 bit SCALEDPTR32, pointing to the array of pointers to each partials vector
// out:         16 bit SCALEDPTR64, pointing to the output array
// numPartials: 16 bit partials count (number of partials vectors)
// numElems:    16 bit elems count (number of elements in each partial=number of outputs)
//****************************************************************************
#define VOFF_PARTIALS     0
#define VOFF_OUT          (2)
#define VOFF_NUM_PARTIALS (4)
#define VOFF_NUM_ELEMS    (6)
#else
//****************************************************************************
// The input structure parameters:
// partials :   32 bit ONE_PTR, pointing to the array of pointers to each partials vector
// out:         32 bit ONE_PTR, pointing to the output array
// numPartials: 16 bit partials count (number of partials vectors)
// numElems:    16 bit elems count (number of elements in each partial=number of outputs)
//****************************************************************************
#define VOFF_PARTIALS     0
#define VOFF_OUT          (4)
#define VOFF_NUM_PARTIALS (8)
#define VOFF_NUM_ELEMS    (10)
#endif

//****************************************************************************
// Constants
//****************************************************************************
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define STACK_SIZE (0)

//****************************************************************************
// Work division
//****************************************************************************
#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define LOG2_24_OVER_3 3
#define LOG2_12_OVER_3 2

#define DIV_6_SHIFT 18

.macro DIVIDE_BY_WORKER mIN mWORKER mSCRATCH mOUT SHIFTS_FOR_GRAINSIZE
  shr \mOUT, \mIN, \SHIFTS_FOR_GRAINSIZE
  add \mOUT, \mOUT, CTXT_WORKERS-1
  sub \mOUT, \mOUT, \mWORKER
  setzi \mSCRATCH, RECIPROCAL_3_SHL17
  mul \mOUT, \mOUT, \mSCRATCH
  shr \mOUT, \mOUT, DIV_6_SHIFT
.endm

//****************************************************************************
// Macro to load vertex state
//****************************************************************************

.macro WORKER_PREAMBLE
  ldz16 $NUM_ELEMS, $mzero, $mvertex_base, VOFF_NUM_ELEMS/2
  ldz16 $NUM_PARTIALS, $mzero, $mvertex_base, VOFF_NUM_PARTIALS/2

#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16  $PARTIALS, $mvertex_base, $mzero, VOFF_PARTIALS/2
  ldz16  $OUT_PTR, $mvertex_base, $mzero, VOFF_OUT/2
  shl   $OUT_PTR, $OUT_PTR, 3
  shl   $PARTIALS, $PARTIALS, 2
  {setzi $MEM_BASE, TMEM_REGION0_BASE_ADDR
   fnop} // rpt align
#else
  ld32 $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT/4
  ld32 $PARTIALS, $mzero, $mvertex_base, VOFF_PARTIALS/4
#endif

  {mov   $OUT_PTR_STORE, $OUT_PTR
   setzi $ZAACC, ZAACC_BITMASK}

  // Fetch worker ID, masking it out: 0..5
  get   $WKR_ID, $WSR
  {and  $WKR_ID, $WKR_ID, CSR_W_WSR__CTXTID_M1__MASK
   uput $FP_CLR, $ZAACC}
  // (init accumulators)
.endm

.worker

//****************************************************************************
// Reduce add float_half. ( Partials = half Output = float )
//****************************************************************************

#define ReduceAdd_Float_Half __runCodelet_poplin__ReduceAdd___float_half_false_false

DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Float_Half
.section .text.Reduce_Float_Half, "ax"
.globl ReduceAdd_Float_Half
.type ReduceAdd_Float_Half, @function
.align 8
  fnop        //For repeat alignment below
ReduceAdd_Float_Half:
  WORKER_PREAMBLE

  // Each worker will start writing strided 8 floats (32 bytes) apart
  // Each worker will start calculating strided 8 halves (16 bytes) apart
  shl   $BASE, $WKR_ID, 5
  add   $OUT_PTR, $OUT_PTR, $BASE
  shl   $BASE, $WKR_ID, 4

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_24_OVER_3

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 8 to process
  brnzdec $ELEMS_LOOPS, .Lelems_loop8_fh
  bri     .Lno_loop8_fh

.Lelems_loop8_fh:
  mov      $SCRATCH, $PARTIALS
  ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1

  rpt     $NUM_PARTIALS, (2f-1f)/8 -1

1:
  {ld64     $VAL12, $BASE, $PARTIALS_PTR,0
   fnop}
  {ld64     $VAL34, $BASE, $PARTIALS_PTR, 1
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1
   f16v8acc $VAL14}
2:
  // Store the results from the independent sums of 8 columns found in the loop
  {add         $BASE, $BASE, 16*6
   f32v2gina   $VAL12, $azeros, 0}
  {st64step    $VAL12, $mzero, $OUT_PTR+=,1
   f32v2gina   $VAL12, $azeros, 0}
  {st64step    $VAL12, $mzero, $OUT_PTR+=,1
   f32v2gina   $VAL12, $azeros, 0}
  {st64step    $VAL12, $mzero, $OUT_PTR+=,1
   f32v2gina   $VAL12, $azeros, 0}
  st64step    $VAL12, $mzero, $OUT_PTR+=,1+(5*4)
  brnzdec     $ELEMS_LOOPS, .Lelems_loop8_fh

  // Groups of 8 columns complete.  Now 0-7 columns left
.Lno_loop8_fh:
  // As we've dealt with groups of 8, it's simple to find the remaining elements
  // quick exit if none
  and         $SCRATCH, $NUM_ELEMS, 7
  brz         $SCRATCH, .Lexit_fh

  // from the loop above, higher numbered workers are likely to finish earlier
  // so let them do the job. Also, we need to write pairs of halves, processed
  // by a single worker to avoid write clashes. Choose the method that involves
  // the least workers, so that there is a greater chance of them getting
  // done before the worker with the most 8x loops completes.
  // If 1 remaining, use worker 5 to process a single element
  // If 2 remaining, use worker 5 to process a pair of elements
  // If 3 remaining, use worker 5 to process a pair, and worker 4 to process the last element
  // If 4 remaining, use worker 5 to process all 4
  // If 5 remaining, use worker 5 to process 4, and worker 4 to process the last element
  // If 6 remaining, use worker 5 to process 4, and worker 4 to process 2
  // If 7 remaining, use worker 5 to process 4, worker 4 to process 2 and worker 3 to process 1

  // Comparisons, decisions on worker / elements remaining
  add         $SCRATCH2, $WKR_ID, -4
  brz         $SCRATCH2, .Lworker4_fh
  brpos       $SCRATCH2, .Lworker5_fh
  cmpeq       $SCRATCH2, $WKR_ID, 3
  brz         $SCRATCH2, .Lexit_fh

.Lworker3_fh:
  // Worker 3 - if there are 7 elements to work on, process the last one
  cmpeq       $SCRATCH2, $SCRATCH, 7
  brnz        $SCRATCH2, .Lprocess1_fh

  exitz       $mzero
//******************************************************************************
.Lworker4_fh:
  // Worker 4 -if there are 3 elements to work on, process the last one
  cmpeq       $SCRATCH2, $SCRATCH, 3
  brnz        $SCRATCH2, .Lprocess1_fh
  // If there are 5, process the last one
  add         $SCRATCH2, $SCRATCH, -5
  brz         $SCRATCH2, .Lprocess1_fh
  // If there are more than 5, process 2 but adjust scratch
  add         $SCRATCH, $SCRATCH, -4
  brpos       $SCRATCH2, .Lprocess2_fh
  exitz       $mzero

//******************************************************************************
.Lworker5_fh:
  // Worker 5 - process 4, 2 or 1 in a priority order
  and         $SCRATCH2, $SCRATCH, 4
  brnz        $SCRATCH2, .Lprocess4_fh

  and         $SCRATCH2, $SCRATCH, 2
  brnz        $SCRATCH2, .Lprocess2_fh

.Lprocess1_fh:
  // base column for the last input half
  add         $BASE, $NUM_ELEMS, -1
  shl         $BASE, $BASE, 1

  ld32step    $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1

  {rpt        $NUM_PARTIALS, (2f-1f)/8 -1
   zero       $VAL1}
1:
  {ldb16      $VAL3, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step   $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f16v2add   $VAL1, $VAL1, $VAL3}
2:
  // Store the results. Base will now index to floats, not halves so double it
  {shl        $BASE, $BASE, 1
   f16v2tof32 $VAL12, $VAL1}
  st32        $VAL1, $BASE, $OUT_PTR_STORE, 0
  exitz       $mzero

//******************************************************************************
.Lprocess2_fh:
  // base column for the pair of halves to process, offset from the end by the number of
  // elements remaining
  sub         $BASE, $NUM_ELEMS, $SCRATCH
  shl         $BASE, $BASE, 1

  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   zero     $VAL1}

  {rpt    $NUM_PARTIALS, (2f-1f)/8 -1
   fnop}
1:
  {ld32     $VAL3, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f16v2add $VAL1, $VAL1, $VAL3}
2:
  // Store the results. Base will now index to floats, not halves so double it
  {shl         $BASE, $BASE, 1
  f16v2tof32 $VAL12, $VAL1}
  st64      $VAL12, $BASE, $OUT_PTR_STORE,0
  exitz     $mzero
//******************************************************************************
.Lprocess4_fh:
  // base column for the pair to process, offset from the end by the number of
  // elements remaining
  sub         $BASE, $NUM_ELEMS, $SCRATCH
  shl         $BASE, $BASE, 1

  ld32step    $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1

  rpt   $NUM_PARTIALS, (2f-1f)/8 -1
1:
  {ld64     $VAL12, $BASE, $PARTIALS_PTR,0
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f16v4acc    $VAL12}
2:
  // Store the results. Base will now index to floats, not halves so double it
  {shl        $BASE, $BASE, 1
   f32v2gina   $VAL12, $azeros, 0}
  {st64       $VAL12, $BASE, $OUT_PTR_STORE,0
   f32v2gina  $VAL12, $azeros, 0}
  st64        $VAL12, $BASE, $OUT_PTR_STORE,1

.Lexit_fh:
  exitz $mzero

.size ReduceAdd_Float_Half, .-ReduceAdd_Float_Half
//****************************************************************************
// Reduce add half_half. ( Partials = half Output = half )
//****************************************************************************

#define ReduceAdd_Half_Half __runCodelet_poplin__ReduceAdd___half_half_false_false
DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Half_Half
.section .text.Reduce_Half_Half, "ax"
.globl ReduceAdd_Half_Half
.type ReduceAdd_Half_Half, @function
.align 8

  nop          // For repeat alignment
ReduceAdd_Half_Half:
  WORKER_PREAMBLE
  // Each worker will start writing strided 8 halves (16 bytes) apart
  // Each worker will start calculating strided 8 halves (16 bytes) apart
  shl   $BASE, $WKR_ID, 4
  add   $OUT_PTR, $OUT_PTR, $BASE

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_24_OVER_3

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 8 to process
  brnzdec $ELEMS_LOOPS, .Lelems_loop8_hh_1st
  bri     .Lno_loop8_hh

.Lelems_loop8_hh:
  // Store with a stride so that we progress to this worker's next output:
  // Write and step our own output (1) PLUS
  // the other 5 worker's work: 5 workers with 2x64 bits each = (5*2)
  st64step    $VAL12, $mzero, $OUT_PTR+=,1+(5*2)
.Lelems_loop8_hh_1st:
  mov      $SCRATCH, $PARTIALS
  ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1
  rpt      $NUM_PARTIALS, (2f-1f)/8 -1

1:
  {ld64     $VAL12, $BASE, $PARTIALS_PTR,0
   fnop}
  {ld64     $VAL34, $BASE, $PARTIALS_PTR, 1
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1
   f16v8acc $VAL14}
2:
  // Store the results from the independent sums of 8 columns found in the loop
  {add         $BASE, $BASE, 16*6
   f16v2gina   $VAL1, $azero, 0}
  f16v2gina   $VAL2, $azero, 0
  {st64step    $VAL12, $mzero, $OUT_PTR+=,1
   f16v2gina   $VAL1, $azero, 0}

  {brnzdec     $ELEMS_LOOPS, .Lelems_loop8_hh
   f16v2gina   $VAL2, $azero, 0}
  // Store the last one
  // Store with a stride so that we progress to this worker's next output:
  // Write and step our own output (1) PLUS
  // the other 5 worker's work: 5 workers with 2x64 bits each = (5*2)
  st64step    $VAL12, $mzero, $OUT_PTR+=,1+(5*2)

  // Groups of 8 columns complete.  Now 0-7 columns left
.Lno_loop8_hh:
  // As we've dealt with groups of 8, it's simple to find the remaining elements
  // quick exit if none
  and         $SCRATCH, $NUM_ELEMS, 7
  brz         $SCRATCH, .Lexit_hh

  // from the loop above, higher numbered workers are likely to finish earlier
  // so let them do the job. Also, we need to write pairs of halves, processed
  // by a single worker to avoid write clashes. Choose the method that involves
  // the least workers, so that there is a greater chance of them getting
  // done before the worker with the most 8x loops completes.
  // If 1 remaining, use worker 5 to process a single element
  // If 2 remaining, use worker 5 to process a pair of elements
  // If 3 remaining, use worker 5 to process a pair, and worker 4 to process the last element
  // If 4 remaining, use worker 5 to process all 4
  // If 5 remaining, use worker 5 to process 4, and worker 4 to process the last element
  // If 6 remaining, use worker 5 to process 4, and worker 4 to process 2
  // If 7 remaining, use worker 5 to process 4, worker 4 to process 2 and worker 3 to process 1

  // Comparisons, decisions on worker / elements remaining
  add         $SCRATCH2, $WKR_ID, -4
  brz         $SCRATCH2, .Lworker4_hh
  brpos       $SCRATCH2, .Lworker5_hh
  cmpeq       $SCRATCH2, $WKR_ID, 3
  brz         $SCRATCH2, .Lexit_hh

.Lworker3_hh:
  // Worker 3 - if there are 7 elements to work on, process the last one
  cmpeq       $SCRATCH2, $SCRATCH, 7
  brnz        $SCRATCH2, .Lprocess1_hh

  exitz       $mzero
//******************************************************************************
.Lworker4_hh:
  // Worker 4 -if there are 3 elements to work on, process the last one
  cmpeq       $SCRATCH2, $SCRATCH, 3
  brnz        $SCRATCH2, .Lprocess1_hh
  // If there are 5, process the last one
  add         $SCRATCH2, $SCRATCH, -5
  brz         $SCRATCH2, .Lprocess1_hh
  // If there are more than 5, process 2 but adjust scratch so we process
  // the last aligned pair
  add         $SCRATCH, $SCRATCH, -4
  brpos       $SCRATCH2, .Lprocess2_hh
  exitz       $mzero

//******************************************************************************
.Lworker5_hh:
  // Worker 5 - process 4, 2 or 1 in a priority order
  and         $SCRATCH2, $SCRATCH, 4
  brnz        $SCRATCH2, .Lprocess4_hh

  and         $SCRATCH2, $SCRATCH, 2
  brnz        $SCRATCH2, .Lprocess2_hh

.Lprocess1_hh:
  // base column for the last input
  add         $BASE, $NUM_ELEMS, -1
  shl         $BASE, $BASE, 1

  ld32step    $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1

  {rpt        $NUM_PARTIALS, (2f-1f)/8 -1
   zero        $VAL1}
1:
  {ldb16      $VAL3, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step   $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f16v2add     $VAL1, $VAL1, $VAL3}
2:
  ldb16       $VAL2, $BASE, $OUT_PTR_STORE, 1
  roll16      $VAL1, $VAL1, $VAL2
  st32        $VAL1, $BASE, $OUT_PTR_STORE, 0
  exitz       $mzero

//******************************************************************************
.Lprocess2_hh:
  // base column for the pair to process, offset from the end by the number of
  // elements remaining
  sub         $BASE, $NUM_ELEMS, $SCRATCH
  shl         $BASE, $BASE, 1

  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   zero     $VAL1}

  {rpt    $NUM_PARTIALS, (2f-1f)/8 -1
   fnop}
1:
  {ld32     $VAL3, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f16v2add $VAL1, $VAL1, $VAL3}
2:
  st32      $VAL1, $BASE, $OUT_PTR_STORE,0
  exitz     $mzero
//******************************************************************************
.Lprocess4_hh:
  // base column for the pair to process, offset from the end by the number of
  // elements remaining
  sub         $BASE, $NUM_ELEMS, $SCRATCH
  shl         $BASE, $BASE, 1

  ld32step    $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1

  rpt   $NUM_PARTIALS, (2f-1f)/8 -1
1:
  {ld64     $VAL12, $BASE, $PARTIALS_PTR,0
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f16v4acc    $VAL12}
2:
  // Store the results
  f16v2gina   $VAL1, $azero, 0
  f16v2gina   $VAL2, $azero, 0
  st64        $VAL12, $BASE, $OUT_PTR_STORE,0

.Lexit_hh:
  exitz $mzero

.size ReduceAdd_Half_Half, .-ReduceAdd_Half_Half

//****************************************************************************
// Reduce add half_float. ( Partials = float Output = half )
//****************************************************************************

#define ReduceAdd_Half_Float __runCodelet_poplin__ReduceAdd___half_float_false_false
DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Half_Float
.section .text.Reduce_Half_Float, "ax"
.globl ReduceAdd_Half_Float
.type ReduceAdd_Half_Float, @function
.align 8

//****************************************************************************
ReduceAdd_Half_Float:
  WORKER_PREAMBLE

  // Each worker will start writing strided 4 halves (8 bytes) apart
  // Each worker will start calculating strided 4 floats (16 bytes) apart
  shl   $BASE, $WKR_ID, 3
  add   $OUT_PTR, $OUT_PTR, $BASE
  shl   $BASE, $WKR_ID, 4

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_12_OVER_3

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 4 to copy
  brnzdec   $ELEMS_LOOPS, .Lelems_loop4_hf_1st
  bri       .Lno_loop4_hf

.Lelems_loop4_hf:
  // Store the result from the previous pass
  st64step    $VAL12, $mzero, $OUT_PTR+=,6
.Lelems_loop4_hf_1st:
  mov      $SCRATCH, $PARTIALS
  ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1

  rpt     $NUM_PARTIALS, (2f-1f)/8 -1

1:
  {ld64     $VAL12, $BASE, $PARTIALS_PTR,0
   fnop}
  {ld64     $VAL34, $BASE, $PARTIALS_PTR, 1
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1
   f32v4acc    $VAL14}
2:
  // Store the results from the independent sums of 4 columns found in the loop
  {add         $BASE, $BASE, 16*6
   f16v2gina   $VAL1, $azero, 0}
  {brnzdec     $ELEMS_LOOPS, .Lelems_loop4_hf
   f16v2gina   $VAL2, $azero, 0}
  // Store the last one
  st64step    $VAL12, $mzero, $OUT_PTR+=,6

  // Groups of 4 columns complete.  Now 0-3 columns left
.Lno_loop4_hf:
  // As we've dealt with groups of 4, it's simple to find the remaining elements
  // quick exit if none
  and         $SCRATCH, $NUM_ELEMS, 3
  brz         $SCRATCH, .Lexit_hf

  // from the loop above, higher numbered workers are likely to finish earlier
  // so let them do the remainder. Also, we need to write pairs of halves, processed
  // by a single worker to avoid write clashes. So:
  // If 1 remaining, use worker 5 to process a single element
  // If 2 remaining, use worker 5 to process a pair of elements
  // If 3 remaining, use worker 5 to process a pair, and worker 4 to process the last element

  // Comparisons, decisions on worker / elements remaining
  cmpeq       $SCRATCH2, $WKR_ID, 5
  brnz        $SCRATCH2, .Lworker5_hf
  cmpeq       $SCRATCH2, $WKR_ID, 4
  brz         $SCRATCH2, .Lexit_hf

.Lworker4_hf:
  // Worker 4 - only process if there are 3 elements to work on
  cmpeq       $SCRATCH2, $SCRATCH, 3
  brnz        $SCRATCH2, .Lprocess1_hf
  exitz       $mzero

//******************************************************************************
.Lworker5_hf:
  // Worker 5 - process 1 if 1 left, or 2 if 2 or 3 left
  cmpeq       $SCRATCH2, $SCRATCH, 1
  brz         $SCRATCH2, .Lprocess2_hf

.Lprocess1_hf:
  // base column for the last input
  add         $BASE, $NUM_ELEMS, -1
  shl         $BASE, $BASE, 2

  {ld32step    $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   fnop}       // For loop alignment
  {rpt        $NUM_PARTIALS, (2f-1f)/8 -1
   zero       $VAL1}
1:
  {ld32       $VAL3, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step   $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f32add     $VAL1, $VAL1, $VAL3}
2:
  // Output base is in halves, not floats so needs to be half the size
  {shr        $BASE, $BASE, 1
   f32tof16   $VAL1, $VAL1}
  ldb16       $VAL2, $BASE, $OUT_PTR_STORE, 1
  roll16      $VAL1, $VAL1, $VAL2
  st32        $VAL1, $BASE, $OUT_PTR_STORE, 0
  exitz       $mzero
//******************************************************************************
.Lprocess2_hf:
  // base column for the pair to process, offset from the end by the number of
  // elements remaining
  sub       $BASE, $NUM_ELEMS, $SCRATCH
  shl       $BASE, $BASE, 2

  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   zero     $VAL12}

  {rpt      $NUM_PARTIALS, (2f-1f)/8 -1
   fnop}
1:
  {ld64     $VAL34, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f32v2add $VAL12, $VAL12, $VAL34}
2:
  // Store result.
  // Output base is in halves, not floats so needs to be half the size
  {shr        $BASE, $BASE, 1
   f32v2tof16 $VAL1, $VAL12}
  st32        $VAL1, $BASE, $OUT_PTR_STORE,0

.Lexit_hf:
  exitz $mzero

.size ReduceAdd_Half_Float, .-ReduceAdd_Half_Float


//****************************************************************************
// Reduce add float_float. ( Partials = float Output = float )
//****************************************************************************

#define ReduceAdd_Float_Float __runCodelet_poplin__ReduceAdd___float_float_false_false
DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Float_Float
.section .text.Reduce_Float_Float, "ax"
.globl ReduceAdd_Float_Float
.type ReduceAdd_Float_Float, @function
.align 8

ReduceAdd_Float_Float:
  WORKER_PREAMBLE

  // Each worker will start writing strided 4 floats (16 bytes) apart
  // Each worker will start calculating strided 4 floats (16 bytes) apart
  shl   $BASE, $WKR_ID, 4
  add   $OUT_PTR, $OUT_PTR, $BASE

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_12_OVER_3

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 4 to copy
  brnzdec   $ELEMS_LOOPS, .Lelems_loop4_ff
  bri       no_loop4_ff

.Lelems_loop4_ff:
  mov       $SCRATCH, $PARTIALS
  ld32step  $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1

  rpt       $NUM_PARTIALS, (2f-1f)/8 -1

1:
  {ld64     $VAL12, $BASE, $PARTIALS_PTR,0
   fnop}
  {ld64     $VAL34, $BASE, $PARTIALS_PTR, 1
   fnop}
  {ld32step $PARTIALS_PTR, $MEM_BASE, $SCRATCH+=,1
   f32v4acc $VAL14}
2:
  // Store the results from the independent sums of 4 columns found in the loop
  {add        $BASE, $BASE, 16*6
   f32v2gina  $VAL12, $azeros, 0}
  {st64step   $VAL12, $mzero, $OUT_PTR+=,1
   f32v2gina  $VAL34, $azeros, 0}
  // Store with a stride so that we progress to this worker's next output:
  // Write and step our own output (1) PLUS
  // the other 5 worker's work: 5 workers with 2x64 bits each = (5*2)
  st64step    $VAL34, $mzero, $OUT_PTR+=,1+(5*2)
  brnzdec     $ELEMS_LOOPS, .Lelems_loop4_ff

  // Groups of 4 columns complete.  Now 0-3 columns left
no_loop4_ff:
  // As we've dealt with groups of 4, it's simple to find the remaining elements
  // quick exit if none
  and         $SCRATCH, $NUM_ELEMS, 3
  brz         $SCRATCH, .Lexit_ff

  // from the loop above, higher numbered workers are likely to finish earlier
  // so let them do the remainder. if( wkr_id + remaining columns >= 6 ) there is
  // work to do for this worker (3,4 or 5 only)
  add         $SCRATCH, $SCRATCH, $WKR_ID
  cmpult      $SCRATCH, $SCRATCH, 6
  brnz        $SCRATCH, .Lexit_ff

  // Find in/out base: The column that this worker is to process
  add         $BASE, $NUM_ELEMS, $WKR_ID
  add         $BASE, $BASE, -6
  shl         $BASE, $BASE, 2

  {ld32step   $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   fnop}      // For loop alignment
  {rpt        $NUM_PARTIALS, (2f-1f)/8 -1
   zero       $VAL1}
1:
  {ld32      $VAL3, $BASE, $PARTIALS_PTR, 0
   fnop}
  {ld32step   $PARTIALS_PTR, $MEM_BASE, $PARTIALS+=,1
   f32add     $VAL1, $VAL1, $VAL3}
2:
  // Store the result
  st32       $VAL1, $BASE, $OUT_PTR_STORE, 0
.Lexit_ff:
  exitz $mzero


.size ReduceAdd_Float_Float, .-ReduceAdd_Float_Float


#endif // __IPU__
