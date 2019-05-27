#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Compact and efficient implementation of the common VectorOuter operations in
// assembler.
// Half, float, inPlace and non-inplace versions
/* -------------------------------------------------------------------------- */

#include "poplibs_support/TileConstants.hpp"

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
#define stride2 m9
#define remainder m10
#define mscratch m11

#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__VectorOuter_common##SUFFIX

// Strings for mangled version of the entry points - presently only the version
// that divides work by row, and does not deal with misaligned data.

#define MANGLE_STR_FLOAT __runCodelet_popops__\INPLACE_STR\()___popops__expr__BroadcastOpType__\NAME_STR\()_float_false
#define MANGLE_STR_HALF __runCodelet_popops__\INPLACE_STR\()___popops__expr__BroadcastOpType__\NAME_STR\()_half_false

//******************************************************************************
// Macros for float/binary entry - inplace, non inplace
//******************************************************************************

.macro VECTOR_OUTER_OP_FLOAT_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type _MANGLE_STR_FLOAT, @function
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
  setzi $functionPtr, outerLoop_float_\OPERATION
  bri  divide_work_float
.size _MANGLE_STR_FLOAT, . -_MANGLE_STR_FLOAT
.endm

.macro VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type _MANGLE_STR_FLOAT, @function
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
  setzi $functionPtr, outerLoop_float_\OPERATION
  bri  divide_work_float
.size _MANGLE_STR_FLOAT, . -_MANGLE_STR_FLOAT
.endm

//******************************************************************************
// Macros for half/binary entry - inplace, non inplace
//******************************************************************************

.macro VECTOR_OUTER_OP_HALF_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type _MANGLE_STR_HALF, @function
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
  setzi $functionPtr, outerLoop_half_\OPERATION
  bri  divide_work_half
.size _MANGLE_STR_HALF, . -_MANGLE_STR_HALF
.endm

.macro VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type _MANGLE_STR_HALF, @function
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
  setzi $functionPtr, outerLoop_half_\OPERATION
  bri  divide_work_half
.size _MANGLE_STR_HALF, . -_MANGLE_STR_HALF
.endm
//******************************************************************************
// Macro for half or float/vectorOuter loop implementation - inplace, non inplace
//******************************************************************************
.macro VECTOR_OUTER_OP INSTRUCTION OPERATION SHIFT
.section .text.MANGLE_STR_COMMON(\@)
.align 8
.if \SHIFT == 1
outerLoop_half_\OPERATION\():
.else
outerLoop_float_\OPERATION\():
.endif
  // Offset into the B vector in bytes, and B vector length in bytes
  // for pointer/wraparound calculation
  shl $workerIdM1, $workerIdM1, \SHIFT
  shl $in2Length, $in2Length, \SHIFT

  // Stride in units of 8 bytes to adjust the outPtr in the last store operation
  shr $stride2, $stride, 3
  add $stride2, $stride2, 1

  // Use brnzdec to check for no work and to decement the loop count
  brnzdec $rows, outer_loop\@
  exitz $mzero

outer_loop\@:
  // Deal with wraparound of the in2 pointer if needed.  Could be needed due to the
  // initial worker's stride so do this first
3:
  cmpult $mscratch, $workerIdM1, $in2Length
  brnz $mscratch, 4f
  sub $workerIdM1, $workerIdM1, $in2Length
  bri 3b
4:
  // Load and broadcast the next in2 element for this row.
.if \SHIFT == 1
  // Half case
  ldb16step $a2, $in2Ptr, $workerIdM1+=, NUM_WORKERS
.else
  // Float case
  ld32step $a2, $in2Ptr, $workerIdM1+=, NUM_WORKERS
.endif
  // Pre load so we can pipeline the loop
  {ld64step $a0:1, $workerOffset, $in1Ptr+=, 1
   mov $a3, $a2}
  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $workerOffset, $in1Ptr+=, 1
   \INSTRUCTION\()\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $workerOffset, $outPtr+=, 1
   fnop}
2:
  {add $in1Ptr, $in1Ptr, $stride
   \INSTRUCTION\()\OPERATION $a4:5, $a0:1, $a2:3}
  st64step $a4:5, $workerOffset, $outPtr+=, $stride2
loop_end\@:
  brnzdec $rows, outer_loop\@

  exitz $mzero
.size MANGLE_STR_COMMON(\@), . -outerLoop_\OPERATION
.endm

//******************************************************************************
// Common code, mainly to divide work
//******************************************************************************
.section .text.MANGLE_STR_COMMON(divide_work)
.align 4
divide_work_half:
  shr $mloops, $mloops, 1
  // Falls through and uses the shr below, as half requires >>2, float >>1
divide_work_float:
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
  shl $workerOffset, $mloops, 3
  mul $stride, $workerOffset, NUM_WORKERS-1
  mul $workerOffset, $workerOffset, $workerIdM1
  // One less as internal loop is unrolled
  add $mloops, $mloops, -1

  br $functionPtr
.size MANGLE_STR_COMMON(divide_work), . -divide_work_half

//******************************************************************************
// Use the macros to create inplace and non inplace, float and half entry points
// for each VectorOuter op.
//******************************************************************************

  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY add BroadcastVectorOuterByRowInPlace ADD
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY sub BroadcastVectorOuterByRowInPlace SUBTRACT
  VECTOR_OUTER_OP_FLOAT_IN_PLACE_ENTRY mul BroadcastVectorOuterByRowInPlace MULTIPLY

  VECTOR_OUTER_OP_FLOAT_ENTRY add BroadcastVectorOuterByRow ADD
  VECTOR_OUTER_OP_FLOAT_ENTRY sub BroadcastVectorOuterByRow SUBTRACT
  VECTOR_OUTER_OP_FLOAT_ENTRY mul BroadcastVectorOuterByRow MULTIPLY

  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY add BroadcastVectorOuterByRowInPlace ADD
  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY sub BroadcastVectorOuterByRowInPlace SUBTRACT
  VECTOR_OUTER_OP_HALF_IN_PLACE_ENTRY mul BroadcastVectorOuterByRowInPlace MULTIPLY

  VECTOR_OUTER_OP_HALF_ENTRY add BroadcastVectorOuterByRow ADD
  VECTOR_OUTER_OP_HALF_ENTRY sub BroadcastVectorOuterByRow SUBTRACT
  VECTOR_OUTER_OP_HALF_ENTRY mul BroadcastVectorOuterByRow MULTIPLY

  // Use macros to create loop body for each operation and data type
  VECTOR_OUTER_OP f16v4 add 1
  VECTOR_OUTER_OP f16v4 sub 1
  VECTOR_OUTER_OP f16v4 mul 1

  VECTOR_OUTER_OP f32v2 add 2
  VECTOR_OUTER_OP f32v2 sub 2
  VECTOR_OUTER_OP f32v2 mul 2


#endif
/* -------------------------------------------------------------------------- */
