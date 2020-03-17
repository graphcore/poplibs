// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Compact and efficient implementation of the common VectorOuter operations in
// assembler.
// Half, float, inPlace and non-inplace versions
/* -------------------------------------------------------------------------- */

#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// Registers for each of the passed parameters
// vertex state, all offsets are 8-bit
#define VERTEX_DATA_PTR_OFFSET 0
#define VERTEX_OUT_PTR_OFFSET 4
#define VERTEX_B_PTR_OFFSET 8
#define VERTEX_B_LENGTH_OFFSET 12
#define VERTEX_COLUMNS_OFFSET 16
#define VERTEX_ROWS_OFFSET 18


#define VERTEX_INPLACE_DATA_PTR_OFFSET 0
#define VERTEX_INPLACE_B_PTR_OFFSET 4
#define VERTEX_INPLACE_B_LENGTH_OFFSET 8
#define VERTEX_INPLACE_COLUMNS_OFFSET 12
#define VERTEX_INPLACE_ROWS_OFFSET 14

// Register aliases

// integer variables
#define outPtr m0
#define in1Ptr m1
#define in2Ptr m2
#define in2Length m3
#define rows m4

#define workerOffset m5
#define workerIdM1 m6
#define stride m7
#define mloops m8
#define functionPtr m9
#define remainder m10
#define mscratch m11

#define stride2 m5
#define inOutStride m10
#define in2Stride m11

#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__VectorOuter_common##SUFFIX

// Strings for mangled version of the entry points - presently only the version
// that divides work by row, and does not deal with misaligned data.

#define MANGLE_STR_FLOAT __runCodelet_popops__\INPLACE_STR\()___popops__expr__BroadcastOpType__\NAME_STR\()_float_false
#define MANGLE_STR_HALF __runCodelet_popops__\INPLACE_STR\()___popops__expr__BroadcastOpType__\NAME_STR\()_half_false

//******************************************************************************
// Macros for float/binary entry - inplace, non inplace
//******************************************************************************

.macro VECTOR_OUTER_OP_FLOAT_ENTRY OPERATION INPLACE_STR NAME_STR BY_STR
DEF_STACK_USAGE 0 MANGLE_STR_FLOAT FUNCTION_IS_WORKER
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type MANGLE_STR_FLOAT, @function
.align 4
  // load vertex state
MANGLE_STR_FLOAT:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_B_PTR_OFFSET/4
  ld32 $in2Length, $mzero, $mvertex_base, VERTEX_B_LENGTH_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
  ldz16 $rows, $mzero, $mvertex_base, VERTEX_ROWS_OFFSET/2
  ldz16 $mloops, $mzero, $mvertex_base, VERTEX_COLUMNS_OFFSET/2
  // Setup loop function pointer, and jump to work division code
  setzi $functionPtr, outerLoop_float_\BY_STR\()_\OPERATION
  bri  divide_work_by_\BY_STR\()_float
.size MANGLE_STR_FLOAT, . -MANGLE_STR_FLOAT
.endm

.macro VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR BY_STR
DEF_STACK_USAGE 0 MANGLE_STR_FLOAT FUNCTION_IS_WORKER
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type MANGLE_STR_FLOAT, @function
.align 4
  // load vertex state
MANGLE_STR_FLOAT:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_DATA_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_B_PTR_OFFSET/4
  ld32 $in2Length, $mzero, $mvertex_base, VERTEX_INPLACE_B_LENGTH_OFFSET/4
  ldz16 $rows, $mzero, $mvertex_base, VERTEX_INPLACE_ROWS_OFFSET/2
  ldz16 $mloops, $mzero, $mvertex_base, VERTEX_INPLACE_COLUMNS_OFFSET/2
  mov $outPtr, $in1Ptr
  // Setup loop function pointer, and jump to work division code
  setzi $functionPtr, outerLoop_float_\BY_STR\()_\OPERATION
  bri  divide_work_by_\BY_STR\()_float
.size MANGLE_STR_FLOAT, . -MANGLE_STR_FLOAT
.endm

//******************************************************************************
// Macros for half/binary entry - inplace, non inplace
//******************************************************************************

.macro VECTOR_OUTER_OP_HALF_ENTRY OPERATION INPLACE_STR NAME_STR BY_STR
DEF_STACK_USAGE 0 MANGLE_STR_HALF FUNCTION_IS_WORKER
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4
  // load vertex state
MANGLE_STR_HALF:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_DATA_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_B_PTR_OFFSET/4
  ld32 $in2Length, $mzero, $mvertex_base, VERTEX_B_LENGTH_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
  ldz16 $rows, $mzero, $mvertex_base, VERTEX_ROWS_OFFSET/2
  ldz16 $mloops, $mzero, $mvertex_base, VERTEX_COLUMNS_OFFSET/2
  // Setup loop function pointer, and jump to work division code
  setzi $functionPtr, outerLoop_half_\BY_STR\()_\OPERATION
  bri  divide_work_by_\BY_STR\()_half
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm

.macro VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR BY_STR
DEF_STACK_USAGE 0 MANGLE_STR_HALF FUNCTION_IS_WORKER
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4
  // load vertex state
MANGLE_STR_HALF:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_DATA_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_B_PTR_OFFSET/4
  ld32 $in2Length, $mzero, $mvertex_base, VERTEX_INPLACE_B_LENGTH_OFFSET/4
  ldz16 $rows, $mzero, $mvertex_base, VERTEX_INPLACE_ROWS_OFFSET/2
  ldz16 $mloops, $mzero, $mvertex_base, VERTEX_INPLACE_COLUMNS_OFFSET/2
  mov $outPtr, $in1Ptr
  // Setup loop function pointer, and jump to work division code
  setzi $functionPtr, outerLoop_half_\BY_STR\()_\OPERATION
  bri  divide_work_by_\BY_STR\()_half
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm
//******************************************************************************
// Macro for half or float/vectorOuter loop implementation - inplace, non inplace
// Parameterised so it'll work for work divided by column or by row.
//******************************************************************************
.macro VECTOR_OUTER_OP INSTRUCTION OPERATION SHIFT
.section .text.MANGLE_STR_COMMON(\@)
.align 8

.if \SHIFT == 1

outerLoop_half_row_\OPERATION\():
  // Offset into the B vector in bytes, for the divide work by row case
  shl $workerIdM1, $workerIdM1, \SHIFT
  // B vector length in bytes for pointer/wraparound calculation
  shl $in2Length, $in2Length, \SHIFT
  // Use brnzdec to check for no work and to decrement the loop count
  brnzdec $rows, outer_loop\@
  exitz $mzero
outerLoop_half_column_\OPERATION\():
  // B vector length in bytes for pointer/wraparound calculation
  shl $in2Length, $in2Length, \SHIFT
  // Use brnzdec to decrement the loop count and skip the first check on in2 ptr wraparound
  brnzdec $rows, 4f
  // Should never reach here

.else

outerLoop_float_row_\OPERATION\():
  // Offset into the B vector in bytes, for the divide work by row case
  shl $workerIdM1, $workerIdM1, \SHIFT
  // B vector length in bytes for pointer/wraparound calculation
  shl $in2Length, $in2Length, \SHIFT
  // Use brnzdec to check for no work and to decrement the loop count
  brnzdec $rows, outer_loop\@
  exitz $mzero
outerLoop_float_column_\OPERATION\():
  // B vector length in bytes for pointer/wraparound calculation
  shl $in2Length, $in2Length, \SHIFT
  // Use brnzdec to decrement the loop count and skip the first check on in2 ptr wraparound
  brnzdec $rows, 4f
  // Should never reach here

.endif

outer_loop\@:
  // Deal with wraparound of the in2 pointer if needed.  Could be needed due to the
  // initial worker's stride when dividing work by row so do this first
3:
  cmpult $functionPtr, $workerIdM1, $in2Length
  brnz $functionPtr, 4f
  sub $workerIdM1, $workerIdM1, $in2Length
  bri 3b
4:
  // Load and broadcast the next in2 element for this row.
.if \SHIFT == 1
  // Half case
  ldb16step $a2, $in2Ptr, $workerIdM1+=, $in2Stride
.else
  // Float case
  ld32step $a2, $in2Ptr, $workerIdM1+=, $in2Stride
.endif
  // Pre load so we can pipeline the loop
  {ld64step $a0:1, $mzero, $in1Ptr+=, $inOutStride
   mov $a3, $a2}

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, $inOutStride
   \INSTRUCTION\()\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, $inOutStride
   fnop}
2:
  {add $in1Ptr, $in1Ptr, $stride
   \INSTRUCTION\()\OPERATION $a4:5, $a0:1, $a2:3}
  st64step $a4:5, $mzero, $outPtr+=, $stride2
  brnzdec $rows, outer_loop\@

  exitz $mzero
.size MANGLE_STR_COMMON(\@), . -outerLoop_\OPERATION
.endm


//******************************************************************************
// Common code, mainly to divide work by row
//******************************************************************************
.section .text.MANGLE_STR_COMMON(divide_work_by_row)
.align 4
divide_work_by_row_half:
  shr $mloops, $mloops, 1
  // Falls through and uses the shr below, as half requires >>2, float >>1
divide_work_by_row_float:
  shr $mloops, $mloops, 1
  // Extract worker ID
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // Dividing work by row
  // Loops for this worker: divide by 6 find remainder
  setzi $mscratch, 0xAAAB
  mul $mscratch, $rows, $mscratch
  shr $mscratch, $mscratch, 18
  mul $remainder, $mscratch, NUM_WORKERS

  // Compare remainder to total number of items to process
  sub $remainder, $rows, $remainder
  // add 1 if < remainder
  cmpult $rows, $workerIdM1, $remainder
  add $rows, $mscratch, $rows

 // Our initial row = workerIdM1 * rowLength
  mul $stride, $mloops, NUM_WORKERS-1
  mul $workerOffset, $mloops, $workerIdM1
  // Dummy load to setup worker offsets.
  ld64step $a0:1, $mzero, $outPtr+=, $workerOffset
  ld64step $a0:1, $mzero, $in1Ptr+=, $workerOffset
  // One less as internal loop is unrolled
  add $mloops, $mloops, -1
  // Strides so the inner loop code can be shared
  setzi $inOutStride, 1
  setzi $in2Stride, NUM_WORKERS
  // Stride for use as a ptr increment in the last store
  add $stride2, $stride, 1
  shl $stride, $stride, 3
  br $functionPtr
.size MANGLE_STR_COMMON(divide_work_by_row), . -divide_work_by_row_half

//******************************************************************************
// Common code, mainly to divide work by column
//******************************************************************************
.section .text.MANGLE_STR_COMMON(divide_work_by_column)
.align 4
divide_work_by_column_half:
  // Falls through and uses the shr below, as half requires >>2, float >>1
  shr   $mloops, $mloops, 1
divide_work_by_column_float:
  shr   $mloops, $mloops, 1
  // Extract worker ID
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // Dividing work by column
  // Loops for this worker: divide by 24 or 12 (for halves or floats), find remainder
  setzi $mscratch, 0xAAAB
  mul $mscratch, $mloops, $mscratch
  // Total shift of 20 (for half results in divide by 24)
  // or 19 (for float results in divide by 12)
  shr $mscratch, $mscratch, 18
  mul $remainder, $mscratch, NUM_WORKERS

  // Unadjusted mloops used in the stride calculation below
  mov $stride, $mloops
  // Compare remainder to total number of items to process
  sub $remainder, $mloops, $remainder

  // add 1 if < remainder
  cmpult $mloops, $workerIdM1, $remainder
  add $mloops, $mscratch, $mloops

  // Our initial column = workerIdM1 * 8
  // Dummy load to setup worker offsets.
  ld64step $a0:1, $mzero, $outPtr+=, $workerIdM1
  ld64step $a0:1, $mzero, $in1Ptr+=, $workerIdM1

  // Compute a Stride to move to the next row.  Calculation based on reverting the strides
  // through the loop and adding 1 row length to adjust the pointer
  // to the next row.
  // 8 * (unadjusted mloops - NUM_WORKERS * $mloops)
  mul $mscratch, $mloops, -NUM_WORKERS
  add $stride, $stride, $mscratch
  // Stride for use as a ptr increment in the last store
  add $stride2, $stride, NUM_WORKERS
  shl $stride, $stride, 3

  // Strides so the inner loop code can be shared
  setzi $inOutStride, NUM_WORKERS
  setzi $in2Stride, 1
   // Repurposed as the in2 Index
  setzi $workerIdM1, 0
  // One less as internal loop is unrolled, exit if zero, which is possible for some workers
  brnzdec $mloops, 1f
  exitz $mzero
1:
  br $functionPtr
.size MANGLE_STR_COMMON(divide_work_by_column), . -divide_work_by_column_float

//******************************************************************************
// Use the macros to create inplace and non inplace, float and half entry points
// for each VectorOuter op.
//******************************************************************************

// By Row variants
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY add BroadcastVectorOuterByRowInPlace ADD row
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY sub BroadcastVectorOuterByRowInPlace SUBTRACT row
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY mul BroadcastVectorOuterByRowInPlace MULTIPLY row

  VECTOR_OUTER_OP_FLOAT_ENTRY add BroadcastVectorOuterByRow ADD row
  VECTOR_OUTER_OP_FLOAT_ENTRY sub BroadcastVectorOuterByRow SUBTRACT row
  VECTOR_OUTER_OP_FLOAT_ENTRY mul BroadcastVectorOuterByRow MULTIPLY row

  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY add BroadcastVectorOuterByRowInPlace ADD row
  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY sub BroadcastVectorOuterByRowInPlace SUBTRACT row
  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY mul BroadcastVectorOuterByRowInPlace MULTIPLY row

  VECTOR_OUTER_OP_HALF_ENTRY add BroadcastVectorOuterByRow ADD row
  VECTOR_OUTER_OP_HALF_ENTRY sub BroadcastVectorOuterByRow SUBTRACT row
  VECTOR_OUTER_OP_HALF_ENTRY mul BroadcastVectorOuterByRow MULTIPLY row


// By Column variants
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY add BroadcastVectorOuterByColumnInPlace ADD column
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY sub BroadcastVectorOuterByColumnInPlace SUBTRACT column
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY mul BroadcastVectorOuterByColumnInPlace MULTIPLY column

  VECTOR_OUTER_OP_FLOAT_ENTRY add BroadcastVectorOuterByColumn ADD column
  VECTOR_OUTER_OP_FLOAT_ENTRY sub BroadcastVectorOuterByColumn SUBTRACT column
  VECTOR_OUTER_OP_FLOAT_ENTRY mul BroadcastVectorOuterByColumn MULTIPLY column

  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY add BroadcastVectorOuterByColumnInPlace ADD column
  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY sub BroadcastVectorOuterByColumnInPlace SUBTRACT column
  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY mul BroadcastVectorOuterByColumnInPlace MULTIPLY column

  VECTOR_OUTER_OP_HALF_ENTRY add BroadcastVectorOuterByColumn ADD column
  VECTOR_OUTER_OP_HALF_ENTRY sub BroadcastVectorOuterByColumn SUBTRACT column
  VECTOR_OUTER_OP_HALF_ENTRY mul BroadcastVectorOuterByColumn MULTIPLY column

  // Use macros to create loop body for each operation and data type
  VECTOR_OUTER_OP f16v4 add 1
  VECTOR_OUTER_OP f16v4 sub 1
  VECTOR_OUTER_OP f16v4 mul 1

  VECTOR_OUTER_OP f32v2 add 2
  VECTOR_OUTER_OP f32v2 sub 2
  VECTOR_OUTER_OP f32v2 mul 2

#endif
/* -------------------------------------------------------------------------- */
