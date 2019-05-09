#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Supervisor vertex code for Binary and Scalar broadcast ops
/* -------------------------------------------------------------------------- */
#include "poplibs_support/TileConstants.hpp"

// Registers for each of the passed parameters
// vertex state, all offsets are 8-bit
#define VERTEX_IN1_PTR_OFFSET 0
#define VERTEX_IN2_PTR_OFFSET 4
#define VERTEX_OUT_PTR_OFFSET 8
#define VERTEX_OUT_COUNT_OFFSET 12

// In place version
#define VERTEX_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_INPLACE_OUT_COUNT_OFFSET 4
#define VERTEX_INPLACE_IN2_PTR_OFFSET 8

// Broadcasting scalar version
#define VERTEX_BROADCAST_IN1_PTR_OFFSET 0
#define VERTEX_BROADCAST_IN1_COUNT_OFFSET 4
#define VERTEX_BROADCAST_OUT_PTR_OFFSET 8
#define VERTEX_BROADCAST_IN2_PTR_OFFSET 12

// Broadcasting In place version
#define VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_BROADCAST_INPLACE_OUT_COUNT_OFFSET 4
#define VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET 8

#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2

// Register aliases
#define outPtr m0
#define in1Ptr m1
#define in2Ptr m2
#define outCount m3

#define mlink m4
#define mloopFn m5
#define mprocess1Fn m6
#define mloops m8
#define remainder m9
#define mprocess2Fn m4

#define workerIdM1 m10
#define mscratch m11

// Naming / name mangling
#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__BinaryOpCommon_##SUFFIX
#define MANGLE_STR_FLOAT __runCodelet_popops__\INPLACE_STR\()___popops__expr__\TYPE_STR\()Type__\NAME_STR\()_float
#define MANGLE_STR_HALF __runCodelet_popops__\INPLACE_STR\()___popops__expr__\TYPE_STR\()Type__\NAME_STR\()_half

//******************************************************************************
// Entry stub macros, one per operation: combination of
// float/half,
// broadcast/binary
// inplace/non-inplace
// add/sub/mul
//******************************************************************************
.macro INSTANTIATE_BROADCAST_OP_HALF OPERATION VERTEX_STATE_STUB INPLACE_STR TYPE_STR NAME_STR
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4

MANGLE_STR_HALF:
  call  $mlink, broadcast_fn_load_vertex_state_\VERTEX_STATE_STUB
  call  $mlink, fn_divide_work24
  setzi $mloopFn, broadcast_op_loop_half_\OPERATION
  setzi $mprocess1Fn, broadcast_op_process_1_half_\OPERATION
  setzi $mprocess2Fn, broadcast_op_process_2_half_\OPERATION
  bri broadcast_op_worker_half_framework
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm
//******************************************************************************
.macro INSTANTIATE_BROADCAST_OP_FLOAT OPERATION VERTEX_STATE_STUB INPLACE_STR TYPE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type MANGLE_STR_FLOAT, @function
.align 4

MANGLE_STR_FLOAT:
  call  $mlink, broadcast_fn_load_vertex_state_\VERTEX_STATE_STUB
  call  $mlink, fn_divide_work12
  setzi $mloopFn, broadcast_op_loop_float_\OPERATION
  setzi $mprocess1Fn, broadcast_op_process_1_float_\OPERATION
  bri broadcast_op_worker_float_framework
.size MANGLE_STR_FLOAT, . -MANGLE_STR_FLOAT
.endm

//******************************************************************************
.macro INSTANTIATE_BINARY_OP_HALF OPERATION VERTEX_STATE_STUB INPLACE_STR TYPE_STR NAME_STR
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4

MANGLE_STR_HALF:
  call  $mlink, fn_load_vertex_state_\VERTEX_STATE_STUB
  call  $mlink, fn_divide_work_binary24
  setzi $mloopFn, binary_op_loop_half_\OPERATION
  setzi $mprocess1Fn, binary_op_process_1_half_\OPERATION
  setzi $mprocess2Fn, binary_op_process_2_half_\OPERATION
  bri binary_op_worker_half_framework
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm

//******************************************************************************
.macro INSTANTIATE_BINARY_OP_FLOAT OPERATION VERTEX_STATE_STUB INPLACE_STR TYPE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type MANGLE_STR_FLOAT, @function
.align 4

MANGLE_STR_FLOAT:
  call  $mlink, fn_load_vertex_state_\VERTEX_STATE_STUB
  call  $mlink, fn_divide_work_binary12
  setzi $mloopFn, binary_op_loop_float_\OPERATION
  setzi $mprocess1Fn, binary_op_process_1_float_\OPERATION
  bri binary_op_worker_float_framework
.size MANGLE_STR_FLOAT, . -MANGLE_STR_FLOAT
.endm


//******************************************************************************
// Code stubs to load vertex state.
// inplace/non inplace
// broadcast/binary
//******************************************************************************

.macro BROADCAST_OP_LOAD_VERTEX_STATE
.section .text.MANGLE_STR_COMMON(\@)
.align 4
broadcast_fn_load_vertex_state_non_in_place:
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4
  br   $mlink
.size MANGLE_STR_COMMON(\@), . -broadcast_fn_load_vertex_state_non_in_place
.endm
//******************************************************************************

.macro BROADCAST_OP_LOAD_VERTEX_STATE_IN_PLACE
.section .text.MANGLE_STR_COMMON(\@)
.align 4
broadcast_fn_load_vertex_state_in_place:
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_OUT_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
  mov  $outPtr, $in1Ptr
  br   $mlink
.size MANGLE_STR_COMMON(\@), . -broadcast_fn_load_vertex_state_in_place
.endm
//******************************************************************************

.macro BINARY_OP_LOAD_VERTEX_STATE
.section .text.MANGLE_STR_COMMON(\@)
.align 4
fn_load_vertex_state_non_in_place:
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_IN1_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/4
  br   $mlink
.size MANGLE_STR_COMMON(\@), . -fn_load_vertex_state_non_in_place
.endm

//******************************************************************************

.macro BINARY_OP_LOAD_VERTEX_STATE_IN_PLACE
.section .text.MANGLE_STR_COMMON(\@)
.align 4
fn_load_vertex_state_in_place:
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_IN2_PTR_OFFSET/4
  mov  $outPtr, $in1Ptr
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_INPLACE_OUT_COUNT_OFFSET/4
  br   $mlink
.size MANGLE_STR_COMMON(\@), . -fn_load_vertex_state_in_place
.endm

//******************************************************************************
.macro BINARY_OP_DIVIDE_WORK SHIFTS_TO_DIV DIVISOR SHIFTS_FOR_GRAINSIZE
.section .text.MANGLE_STR_COMMON(\@)
.align 4
fn_divide_work_binary\DIVISOR\():
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
  ld64step $a0:1, $mzero, $in2Ptr+=, $workerIdM1
fn_divide_work\DIVISOR\():
  // Extract worker ID (intentional repeated load if the entry point above is
  // used as we have 2 entry points for the two op types)
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
  // Dummy reads to increment pointers for the initial worker stride
  ld64step $a0:1, $mzero, $in1Ptr+=, $workerIdM1
  ld64step $a0:1, $mzero, $outPtr+=, $workerIdM1

  // Loops for this worker: divide by 12 or 24, find remainder
  setzi $mscratch, 0xAAAB
  mul $mscratch, $outCount, $mscratch
  shr $mscratch, $mscratch, \SHIFTS_TO_DIV
  mul $remainder, $mscratch, \DIVISOR

  // Compare remainder to total number of items to process
  sub $remainder, $outCount, $remainder

  shr $remainder, $remainder, \SHIFTS_FOR_GRAINSIZE
  // add 1 if < remainder
  cmpult $mloops, $workerIdM1, $remainder
  add $mloops, $mscratch, $mloops
  br  $mlink
.size MANGLE_STR_COMMON(\@), . -fn_divide_work_binary\DIVISOR\()
.endm


//******************************************************************************
// General processing structure for float
//******************************************************************************

.section .text.MANGLE_STR_COMMON(float_processing)
.align 4
broadcast_op_worker_float_framework:
  ld32 $a2, $mzero, $in2Ptr, 0
  mov  $a3, $a2
binary_op_worker_float_framework:
  // Don't use the inner loop section of code at all if the result isn't needed
  // it will do a strided overread which must be avoided
  // As we will process pair with no loop decement.  Also skip loop if nothing to do
  // This way is fast if we are going to use the inner loop
  brnzdec $mloops, 1f
  bri inner_loop_float_return
1:
  br $mloopFn

inner_loop_float_return:
  // Here we have done all groups of 2 floats for every worker, no overread.
  // Use the worker which is pointing to the last float to process the last float
  // (if there is one).  This is simple, but would using the last worker
  //  be faster on average?
  and $mscratch, $outCount, 1
  brz $mscratch, 3f
  // All workers with id < remainder did one more loop, so the one that
  // has id == remainder must be pointing at the next piece of work to do
  cmpeq $mscratch, $remainder, $workerIdM1
  brz $mscratch, 3f

  ld32 $a0, $mzero, $in1Ptr, 0
  br $mprocess1Fn

process_1_float_return:
  st32 $a0, $mzero, $outPtr, 0
3:
  exitz $mzero
.size MANGLE_STR_COMMON(float_processing), . -broadcast_op_worker_float_framework
//******************************************************************************
// General processing structure for half
//******************************************************************************
.section .text.MANGLE_STR_COMMON(half_processing)
.align 4
broadcast_op_worker_half_framework:
  ldb16 $a2, $mzero, $in2Ptr, 0
  mov $a3, $a2
binary_op_worker_half_framework:
  // Don't use the inner loop section of code at all if the result isn't needed
  // it will do a strided overread which must be avoided
  // As we will process 64 bits with no loop, decrement the count.
  // Also skip loop if nothing to do
  // This way is fast if we are going to use the inner loop
  brnzdec $mloops, 1f
  bri inner_loop_half_return
1:
  br $mloopFn

inner_loop_half_return:
  // Here we have done all groups of 3 halves for every worker, no overread.
  // Use the worker which is pointing to the next half to process the last 3
  // (if needed).  This is simple, but would using the last worker
  //  be faster on average?

  // All workers with id < remainder did one more loop, so the one that
  // has id == remainder must be pointing at the next piece of work to do
  cmpeq $mscratch, $remainder, $workerIdM1
  brz $mscratch, 3f

  and $mscratch, $outCount, 2
  brz $mscratch, 4f
  // Process a remaining pair
  ld32step $a0, $mzero, $in1Ptr+=,1
  br $mprocess2Fn

process_2_half_return:
  st32step $a0, $mzero, $outPtr+=, 1
4:
  and $mscratch, $outCount, 1
  brz $mscratch, 3f
  // Process the last one
  ldb16 $a0, $mzero, $in1Ptr, 0
  br $mprocess1Fn

process_1_half_return:
  sort4x16lo $a0, $a0, $a1
  st32 $a0, $mzero, $outPtr, 0
3:
  exitz $mzero
.size MANGLE_STR_COMMON(half_processing), . -broadcast_op_worker_half_framework

//******************************************************************************
// General processing structure for float
//******************************************************************************
.macro INSTANTIATE_BINARY_OP_FLOAT_PROCESSING OPERATION
.section .text.MANGLE_STR_COMMON(float_loop1)
.align 8
// Loop for binary variant
binary_op_loop_float_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
  ld64step $a2:3, $mzero, $in2Ptr+=, NUM_WORKERS

  {rpt $mloops, (2f - 1f ) /8 - 1
  fnop}
1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
   f32v2\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
   fnop}
  {ld64step $a2:3, $mzero, $in2Ptr+=, NUM_WORKERS
   fnop}
2:
  f32v2\OPERATION $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
  bri inner_loop_float_return
.size MANGLE_STR_COMMON(float_loop1), . -binary_op_loop_float_\OPERATION\()

.section .text.MANGLE_STR_COMMON(float_loop2)
.align 8
// Loop for broadcast scalar variant
broadcast_op_loop_float_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS

  rpt $mloops, (2f - 1f ) /8 - 1

1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
   f32v2\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
   fnop}
2:
  f32v2\OPERATION $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
  bri inner_loop_float_return
.size MANGLE_STR_COMMON(float_loop), . -broadcast_op_loop_float_\OPERATION\()

.section .text.MANGLE_STR_COMMON(float_instr)
.align 4
// Single trailing item instruction
binary_op_process_1_float_\OPERATION\():
  ld32 $a2, $mzero, $in2Ptr, 0
broadcast_op_process_1_float_\OPERATION\():
  f32\OPERATION $a0, $a0, $a2
  bri process_1_float_return
.size MANGLE_STR_COMMON(float_instr), . -binary_op_process_1_float_\OPERATION\()
.endm
//******************************************************************************
.macro INSTANTIATE_BINARY_OP_HALF_PROCESSING OPERATION
.section .text.MANGLE_STR_COMMON(half_loop1)
.align 8
binary_op_loop_half_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
  ld64step $a2:3, $mzero, $in2Ptr+=, NUM_WORKERS

  {rpt $mloops, (2f - 1f ) /8 - 1
  fnop}
1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
   f16v4\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
   fnop}
  {ld64step $a2:3, $mzero, $in2Ptr+=, NUM_WORKERS
   fnop}
2:
  f16v4\OPERATION $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
  bri inner_loop_half_return
.size MANGLE_STR_COMMON(half_loop1), . -binary_op_loop_half_\OPERATION\()

.section .text.MANGLE_STR_COMMON(half_loop2)
.align 8
broadcast_op_loop_half_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
  rpt $mloops, (2f - 1f ) /8 - 1

1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, NUM_WORKERS
   f16v4\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
   fnop}
2:
  f16v4\OPERATION $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, NUM_WORKERS
  bri inner_loop_half_return
.size MANGLE_STR_COMMON(half_loop2), . -broadcast_op_loop_half_\OPERATION\()

.section .text.MANGLE_STR_COMMON(half_instr)
.align 4
binary_op_process_2_half_\OPERATION\():
  ld32step $a2, $mzero, $in2Ptr+=, 1
broadcast_op_process_2_half_\OPERATION\():
  f16v2\OPERATION $a0, $a0, $a2
  bri process_2_half_return

binary_op_process_1_half_\OPERATION\():
  ldb16 $a2, $mzero, $in2Ptr, 0
broadcast_op_process_1_half_\OPERATION\():
  {ldb16 $a1, $mzero, $outPtr, 1
   f16v2\OPERATION $a0, $a0, $a2}
   bri process_1_half_return
.size MANGLE_STR_COMMON(half_instr), . -binary_op_process_2_half_\OPERATION\()
.endm

//******************************************************************************
// Use the macros above to create vertex entry points
//******************************************************************************

// add half
INSTANTIATE_BINARY_OP_HALF add in_place BinaryOp1DInPlace BinaryOp ADD
INSTANTIATE_BROADCAST_OP_HALF add in_place BroadcastScalar1DInPlace BroadcastOp ADD
INSTANTIATE_BINARY_OP_HALF add non_in_place BinaryOp1D BinaryOp ADD
INSTANTIATE_BROADCAST_OP_HALF add non_in_place BroadcastScalar1D BroadcastOp ADD

// sub half
INSTANTIATE_BINARY_OP_HALF sub in_place BinaryOp1DInPlace BinaryOp SUBTRACT
INSTANTIATE_BROADCAST_OP_HALF sub in_place BroadcastScalar1DInPlace BroadcastOp SUBTRACT
INSTANTIATE_BINARY_OP_HALF sub non_in_place BinaryOp1D BinaryOp SUBTRACT
INSTANTIATE_BROADCAST_OP_HALF sub non_in_place BroadcastScalar1D BroadcastOp SUBTRACT

// mul half
INSTANTIATE_BINARY_OP_HALF mul in_place BinaryOp1DInPlace BinaryOp MULTIPLY
INSTANTIATE_BROADCAST_OP_HALF mul in_place BroadcastScalar1DInPlace BroadcastOp MULTIPLY
INSTANTIATE_BINARY_OP_HALF mul non_in_place BinaryOp1D BinaryOp MULTIPLY
INSTANTIATE_BROADCAST_OP_HALF mul non_in_place BroadcastScalar1D BroadcastOp MULTIPLY

// add float
INSTANTIATE_BINARY_OP_FLOAT add in_place BinaryOp1DInPlace BinaryOp ADD
INSTANTIATE_BROADCAST_OP_FLOAT add in_place BroadcastScalar1DInPlace BroadcastOp ADD
INSTANTIATE_BINARY_OP_FLOAT add non_in_place BinaryOp1D BinaryOp ADD
INSTANTIATE_BROADCAST_OP_FLOAT add non_in_place BroadcastScalar1D BroadcastOp ADD

// sub float
INSTANTIATE_BINARY_OP_FLOAT sub in_place BinaryOp1DInPlace BinaryOp SUBTRACT
INSTANTIATE_BROADCAST_OP_FLOAT sub in_place BroadcastScalar1DInPlace BroadcastOp SUBTRACT
INSTANTIATE_BINARY_OP_FLOAT sub non_in_place BinaryOp1D BinaryOp SUBTRACT
INSTANTIATE_BROADCAST_OP_FLOAT sub non_in_place BroadcastScalar1D BroadcastOp SUBTRACT

// mul float
INSTANTIATE_BINARY_OP_FLOAT mul in_place BinaryOp1DInPlace BinaryOp MULTIPLY
INSTANTIATE_BROADCAST_OP_FLOAT mul in_place BroadcastScalar1DInPlace BroadcastOp MULTIPLY
INSTANTIATE_BINARY_OP_FLOAT mul non_in_place BinaryOp1D BinaryOp MULTIPLY
INSTANTIATE_BROADCAST_OP_FLOAT mul non_in_place BroadcastScalar1D BroadcastOp MULTIPLY

//******************************************************************************
// Use the macros above to create shared code
//******************************************************************************

BINARY_OP_LOAD_VERTEX_STATE
BINARY_OP_LOAD_VERTEX_STATE_IN_PLACE
BROADCAST_OP_LOAD_VERTEX_STATE
BROADCAST_OP_LOAD_VERTEX_STATE_IN_PLACE


BINARY_OP_DIVIDE_WORK 19 12 LOG2_FLOAT_ATOM_SIZE
BINARY_OP_DIVIDE_WORK 20 24 LOG2_HALF_ATOM_SIZE

//******************************************************************************
// Use the macros above to create each individual operation code
//******************************************************************************

INSTANTIATE_BINARY_OP_HALF_PROCESSING add
INSTANTIATE_BINARY_OP_FLOAT_PROCESSING add

INSTANTIATE_BINARY_OP_HALF_PROCESSING sub
INSTANTIATE_BINARY_OP_FLOAT_PROCESSING sub

INSTANTIATE_BINARY_OP_HALF_PROCESSING mul
INSTANTIATE_BINARY_OP_FLOAT_PROCESSING mul

#endif
/* -------------------------------------------------------------------------- */
