// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//****************************************************************************
// poplin ReduceAdd codelets
//
// Overview:
// We expect to be passed two pointers to partials.
// We could think of each partial as being a row of an array.  The initialPartial is
// the first row and the `partials` pointer points to numPartials rows, all
// contiguous in memory.  The ReduceAdd function sums the values in
// each column of the array.
//
// initialPartial[NUM ELEMS] = a0 b0 c0 d0 e0
// partials[NUM ELEMS * NUM_PARTIALS] = a1 b1 c1 d1 e1
//                                      a2 b2 c1 d2 e2
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
// Constraints:
// The vertex is intended as a fast path, to be used when the partials sizes allow
// but achieves maximum throughput.  So numElems must result in partials with a
// 128 bit width and that data must be appropriately aligned in memorys
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
#define INITIAL_PARTIALS  m0
#define ELEMS_LOOPS       m1

#define NUM_PARTIALS m2
#define NUM_ELEMS    m3
#define OUT_PTR      m4
#define PARTIALS_PTR m5
#define BASE         m6

#define SCRATCH2     m9
#define WKR_ID       m10
#define SCRATCH      m11

#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1
#define VAL34   a2:3
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

// The singleInput version uses another SCALEDPTR64 at the end for compatibility
// with the above. In addition VOFF_PARTIALS will be a SCALEDPTR64
#define VOFF_INITIAL_PARTIAL 8

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

// The singleInput version uses another ONE_PTR at the end for compatibility
// with the above
#define VOFF_INITIAL_PARTIAL 12
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
  ldz16 $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT/2
  ldz16 $PARTIALS_PTR, $mzero, $mvertex_base, VOFF_PARTIALS/2
  ldz16 $INITIAL_PARTIALS, $mzero, $mvertex_base, VOFF_INITIAL_PARTIAL/2
  shl   $OUT_PTR, $OUT_PTR, 3
  shl   $PARTIALS_PTR, $PARTIALS_PTR, 3
  shl   $INITIAL_PARTIALS, $INITIAL_PARTIALS, 3
#else
  ld32 $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT/4
  ld32 $PARTIALS_PTR, $mzero, $mvertex_base, VOFF_PARTIALS/4
  ld32 $INITIAL_PARTIALS, $mzero, $mvertex_base, VOFF_INITIAL_PARTIAL/4
#endif
  // Fetch worker ID, masking it out: 0..5
  {get  $WKR_ID, $WSR
  setzi $ZAACC, ZAACC_BITMASK}
  {and  $WKR_ID, $WKR_ID, CSR_W_WSR__CTXTID_M1__MASK
   uput $FP_CLR, $ZAACC}
  // (init accumulators)
.endm

.worker

//****************************************************************************
// Reduce add float_half. ( Partials = half Output = float )
//****************************************************************************

#define ReduceAdd_Float_Half(FLAGS) __runCodelet_poplin__ReduceAdd___float_half_##FLAGS

.macro REDUCEADD_FLOAT_HALF partialsMemConstraints

DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Float_Half(true_\partialsMemConstraints)
.section .text.Reduce_Float_Half(true_\partialsMemConstraints), "ax"
.globl ReduceAdd_Float_Half(true_\partialsMemConstraints)
.type ReduceAdd_Float_Half(true_\partialsMemConstraints), @function
.align 4

ReduceAdd_Float_Half(true_\partialsMemConstraints):
  WORKER_PREAMBLE

  // Each worker will start writing strided 8 floats (32 bytes) apart
  // Each worker will start calculating strided 8 halves (16 bytes) apart
  shl   $BASE, $WKR_ID, 5
  add   $OUT_PTR, $OUT_PTR, $BASE
  shl   $BASE, $WKR_ID, 4

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_24_OVER_3

.ifc "\partialsMemConstraints","true"
  // Compute a stride for the partials - stride over a row in units of 128 bits
  shr   $SCRATCH, $NUM_ELEMS, 3
.else
  // Compute a stride for the partials = stride over a row in units of 64 bits
  // Minus one as we have already strided in our first ld64step instruction
  shr   $SCRATCH, $NUM_ELEMS, 2
  sub   $SCRATCH, $SCRATCH, 1
.endif

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 8 to process
  mov $SCRATCH2, $PARTIALS_PTR
  brnzdec $ELEMS_LOOPS, .Lelems_loop8_fh\@
  bri     .Lno_loop8_fh\@

.align 8
  nop
.Lelems_loop8_fh\@:
  ld64  $VAL12, $BASE, $INITIAL_PARTIALS, 0
  ld64  $VAL34, $BASE, $INITIAL_PARTIALS, 1

  rpt     $NUM_PARTIALS, (2f-1f)/8 -1
1:
.ifc "\partialsMemConstraints","true"
  {ld128step $VAL14, $BASE, $SCRATCH2+=,$SCRATCH
   f16v8acc $VAL14}
.else
  {ld64step $VAL12, $BASE, $SCRATCH2+=,1
   f16v8acc $VAL14}
  {ld64step $VAL34, $BASE, $SCRATCH2+=,$SCRATCH
   fnop}
.endif
2:
  {mov $SCRATCH2, $PARTIALS_PTR
   f16v8acc $VAL14}
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
  brnzdec     $ELEMS_LOOPS, .Lelems_loop8_fh\@

.Lno_loop8_fh\@:
  exitz $mzero

.size ReduceAdd_Float_Half(true_\partialsMemConstraints), .-ReduceAdd_Float_Half(true_\partialsMemConstraints)
.endm

REDUCEADD_FLOAT_HALF true
REDUCEADD_FLOAT_HALF false

//****************************************************************************
// Reduce add half_half. ( Partials = half Output = half )
//****************************************************************************

#define ReduceAdd_Half_Half(FLAGS) __runCodelet_poplin__ReduceAdd___half_half_##FLAGS

.macro REDUCEADD_HALF_HALF partialsMemConstraints


DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Half_Half(true_\partialsMemConstraints)
.section .text.Reduce_Half_Half(true_\partialsMemConstraints), "ax"
.globl ReduceAdd_Half_Half(true_\partialsMemConstraints)
.type ReduceAdd_Half_Half(true_\partialsMemConstraints), @function

  .align 8
ReduceAdd_Half_Half(true_\partialsMemConstraints):
  WORKER_PREAMBLE
  // Each worker will start writing strided 8 halves (16 bytes) apart
  // Each worker will start calculating strided 8 halves (16 bytes) apart
  shl   $BASE, $WKR_ID, 4
  add   $OUT_PTR, $OUT_PTR, $BASE

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_24_OVER_3

.ifc "\partialsMemConstraints","true"
  // Compute a stride for the partials = stride over a row in units of 128 bits
  shr   $SCRATCH, $NUM_ELEMS, 3
.else
  // Compute a stride for the partials = stride over a row in units of 64 bits
  // Minus one as we have already strided in our first ld64step instruction
  shr   $SCRATCH, $NUM_ELEMS, 2
  sub   $SCRATCH, $SCRATCH, 1
.endif
  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 8 to process
  mov      $SCRATCH2, $PARTIALS_PTR
  brnzdec $ELEMS_LOOPS, .Lelems_loop8_hh_1st\@
  bri     .Lno_loop8_hh\@

.align 8
.Lelems_loop8_hh\@:
  // Store with a stride so that we progress to this worker's next output:
  // Write and step our own output (1) PLUS
  // the other 5 worker's work: 5 workers with 2x64 bits each = (5*2)
  st64step    $VAL12, $mzero, $OUT_PTR+=,1+(5*2)
.Lelems_loop8_hh_1st\@:
  ld64     $VAL12, $BASE, $INITIAL_PARTIALS, 0
  ld64     $VAL34, $BASE, $INITIAL_PARTIALS, 1

  rpt      $NUM_PARTIALS, (2f-1f)/8 -1
1:
.ifc "\partialsMemConstraints","true"
  {ld128step     $VAL14, $BASE, $SCRATCH2+=,$SCRATCH
   f16v8acc $VAL14}
.else
  {ld64step $VAL12, $BASE, $SCRATCH2+=,1
    f16v8acc $VAL14}
  {ld64step $VAL34, $BASE, $SCRATCH2+=,$SCRATCH
    fnop}
.endif
2:
  {mov      $SCRATCH2, $PARTIALS_PTR
   f16v8acc $VAL14}
  // Store the results from the independent sums of 8 columns found in the loop
  {add         $BASE, $BASE, 16*6
   f16v2gina   $VAL1, $azero, 0}
  f16v2gina   $VAL2, $azero, 0
  {st64step    $VAL12, $mzero, $OUT_PTR+=,1
   f16v2gina   $VAL1, $azero, 0}

  {brnzdec     $ELEMS_LOOPS, .Lelems_loop8_hh\@
   f16v2gina   $VAL2, $azero, 0}
   // Store the last one
   st64        $VAL12, $mzero, $OUT_PTR, 0

.Lno_loop8_hh\@:
  exitz $mzero

.size ReduceAdd_Half_Half(true_\partialsMemConstraints), .-ReduceAdd_Half_Half(true_\partialsMemConstraints)
.endm


REDUCEADD_HALF_HALF true
REDUCEADD_HALF_HALF false
//****************************************************************************
// Reduce add half_float. ( Partials = float Output = half )
//****************************************************************************

#define ReduceAdd_Half_Float(FLAGS) __runCodelet_poplin__ReduceAdd___half_float_##FLAGS

.macro REDUCEADD_HALF_FLOAT partialsMemConstraints

DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Half_Float(true_\partialsMemConstraints)
.section .text.Reduce_Half_Float(true_\partialsMemConstraints), "ax"
.globl ReduceAdd_Half_Float(true_\partialsMemConstraints)
.type ReduceAdd_Half_Float(true_\partialsMemConstraints), @function
.align 4

ReduceAdd_Half_Float(true_\partialsMemConstraints):
  WORKER_PREAMBLE

  // Each worker will start writing strided 4 halves (8 bytes) apart
  // Each worker will start calculating strided 4 floats (16 bytes) apart
  shl   $BASE, $WKR_ID, 3
  add   $OUT_PTR, $OUT_PTR, $BASE
  shl   $BASE, $WKR_ID, 4

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_12_OVER_3

.ifc "\partialsMemConstraints","true"
 // Compute a stride for the partials = stride over a row in units of 128 bits
  shr   $SCRATCH, $NUM_ELEMS, 2
.else
  // Compute a stride for the partials = stride over a row in units of 64 bits
  // Minus one as we have already strided in our first ld64step instruction
  shr   $SCRATCH, $NUM_ELEMS, 1
  sub   $SCRATCH, $SCRATCH, 1
.endif

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 4 to copy
  mov       $SCRATCH2, $PARTIALS_PTR
  brnzdec   $ELEMS_LOOPS, .Lelems_loop4_hf_first\@
  bri       .Lno_loop4_hf\@

.align 8
.Lelems_loop4_hf\@:
  st64step    $VAL12, $mzero, $OUT_PTR+=,6
.Lelems_loop4_hf_first\@:
  ld64  $VAL12, $BASE, $INITIAL_PARTIALS, 0
  ld64  $VAL34, $BASE, $INITIAL_PARTIALS, 1

  rpt     $NUM_PARTIALS, (2f-1f)/8 -1
1:
.ifc "\partialsMemConstraints","true"
  {ld128step    $VAL14, $BASE, $SCRATCH2+=,$SCRATCH
   f32v4acc    $VAL14}
.else
  {ld64step    $VAL12, $BASE, $SCRATCH2+=,1
   f32v4acc    $VAL14}
  {ld64step    $VAL34, $BASE, $SCRATCH2+=,$SCRATCH
   fnop}
.endif
2:
  {mov       $SCRATCH2, $PARTIALS_PTR
   f32v4acc    $VAL14}
  // Store the results from the independent sums of 4 columns found in the loop
  {add         $BASE, $BASE, 16*6
   f16v2gina   $VAL1, $azero, 0}
  {brnzdec     $ELEMS_LOOPS, .Lelems_loop4_hf\@
   f16v2gina   $VAL2, $azero, 0}
  // Store the last one
  st64        $VAL12, $mzero, $OUT_PTR, 0

.Lno_loop4_hf\@:
  exitz $mzero

.size ReduceAdd_Half_Float(true_\partialsMemConstraints), .-ReduceAdd_Half_Float(true_\partialsMemConstraints)
.endm


REDUCEADD_HALF_FLOAT true
REDUCEADD_HALF_FLOAT false
//****************************************************************************
// Reduce add float_float. ( Partials = float Output = float )
//****************************************************************************

#define ReduceAdd_Float_Float(FLAGS) __runCodelet_poplin__ReduceAdd___float_float_##FLAGS

.macro REDUCEADD_FLOAT_FLOAT partialsMemConstraints

DEF_STACK_SIZE_OWN STACK_SIZE ReduceAdd_Float_Float(true_\partialsMemConstraints)
.section .text.Reduce_Float_Float(true_\partialsMemConstraints), "ax"
.globl ReduceAdd_Float_Float(true_\partialsMemConstraints)
.type ReduceAdd_Float_Float(true_\partialsMemConstraints), @function
.align 4

ReduceAdd_Float_Float(true_\partialsMemConstraints):
  WORKER_PREAMBLE

  // Each worker will start writing strided 4 floats (16 bytes) apart
  // Each worker will start calculating strided 4 floats (16 bytes) apart
  shl   $BASE, $WKR_ID, 4
  add   $OUT_PTR, $OUT_PTR, $BASE

  DIVIDE_BY_WORKER $NUM_ELEMS $WKR_ID $SCRATCH $ELEMS_LOOPS LOG2_12_OVER_3

.ifc "\partialsMemConstraints","true"
  // Compute a stride for the partials = stride over a row in units of 128 bits
  shr   $SCRATCH, $NUM_ELEMS, 2
.else
  // Compute a stride for the partials = stride over a row in units of 64 bits
  // Minus one as we have already strided in our first ld64step instruction
  shr   $SCRATCH, $NUM_ELEMS, 1
  sub   $SCRATCH, $SCRATCH, 1
.endif

  // Decrement loop count as we use brnzdec, and skip loop if this worker
  // has no groups of 4 to copy
  mov       $SCRATCH2, $PARTIALS_PTR
  brnzdec   $ELEMS_LOOPS, .Lelems_loop4_ff\@
  bri       no_loop4_ff\@

.align 8
  nop
.Lelems_loop4_ff\@:
  ld64  $VAL12, $BASE, $INITIAL_PARTIALS, 0
  ld64  $VAL34, $BASE, $INITIAL_PARTIALS, 1

  rpt       $NUM_PARTIALS, (2f-1f)/8 -1
1:
.ifc "\partialsMemConstraints","true"
  {ld128step     $VAL14, $BASE, $SCRATCH2+=,$SCRATCH
   f32v4acc     $VAL14}
.else
  {ld64step     $VAL12, $BASE, $SCRATCH2+=,1
   f32v4acc     $VAL14}
  {ld64step     $VAL34, $BASE, $SCRATCH2+=,$SCRATCH
   fnop}
.endif
2:
  {mov      $SCRATCH2, $PARTIALS_PTR
   f32v4acc $VAL14}
  // Store the results from the independent sums of 4 columns found in the loop
  {add        $BASE, $BASE, 16*6
   f32v2gina  $VAL12, $azeros, 0}
  {st64step   $VAL12, $mzero, $OUT_PTR+=,1
   f32v2gina  $VAL34, $azeros, 0}
  // Store with a stride so that we progress to this worker's next output:
  // Write and step our own output (1) PLUS
  // the other 5 worker's work: 5 workers with 2x64 bits each = (5*2)
  st64step    $VAL34, $mzero, $OUT_PTR+=,1+(5*2)
  brnzdec     $ELEMS_LOOPS, .Lelems_loop4_ff\@

no_loop4_ff\@:
  exitz $mzero

.size ReduceAdd_Float_Float(true_\partialsMemConstraints), .-ReduceAdd_Float_Float(true_\partialsMemConstraints)
.endm

REDUCEADD_FLOAT_FLOAT true
REDUCEADD_FLOAT_FLOAT false

#endif // __IPU__
