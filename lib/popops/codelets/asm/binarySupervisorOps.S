// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Supervisor vertex code for Binary and Scalar broadcast ops
/* -------------------------------------------------------------------------- */
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

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
#define in1Ptr m0
#define in2Ptr m1
#define in12Ptr m0:1
#define outPtr m2
#define outCount m3

#define mlink m4
#define mprocess2Fn m4
#define mloopFn m5
#define mprocess1Fn m6
#define in2PtrTmp m7

#define mloops m8
#define remainder m9
#define workerIdM1 m10
#define mscratch m11

// Naming / name mangling
#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__BinaryOpCommon_##SUFFIX
#define MANGLE_STR __runCodelet_popops__\VERTEX_NAME\()___popops__expr__BinaryOpType__\OPERATION\()_\TYPE

// Macro to unpack tapack triple address packed ptrs
.macro UNPACK_PTRS IN1 IN2 OUT
  shr \OUT, \IN1, TMEM_BYTE_MAX_ADDRESS_WIDTH
  shr \IN2, \IN2, TMEM_BYTE_MAX_ADDRESS_WIDTH
  shl \IN2, \IN2, (32 - TMEM_BYTE_MAX_ADDRESS_WIDTH)
  or  \OUT, \IN2, \OUT

  shl \IN1, \IN1, (32 - TMEM_BYTE_MAX_ADDRESS_WIDTH)
  shr \IN1, \IN1, (32 - TMEM_BYTE_MAX_ADDRESS_WIDTH)
.endm

//******************************************************************************
// Entry stub macros, use to create one per operation: combination of
// float/half,
// broadcast/binary
// inplace/non-inplace
// add/sub/mul
// And likewise for short/unsigned short with and/or
//******************************************************************************
.macro INSTANTIATE_BROADCAST_OP TYPE VERTEX_STATE_STUB VERTEX_NAME OPERATION OP_PROCESSING_TYPE
DEF_STACK_USAGE 0 MANGLE_STR
.section .text.MANGLE_STR
.global MANGLE_STR
.type MANGLE_STR, @function
.align 4

MANGLE_STR:
  call  $mlink, broadcast_fn_load_vertex_state_\VERTEX_STATE_STUB
.ifc "\TYPE", "float"
  call  $mlink, fn_divide_work12
.else
  call  $mlink, fn_divide_work24
.endif
  // Select fast or normal loop based on the returned value, 0 = fast, 1 = normal
  setzi $mloopFn, broadcast_op_loop_\OP_PROCESSING_TYPE\()_fast_\OPERATION
  brz   $mscratch, 1f
  setzi $mloopFn, broadcast_op_loop_\OP_PROCESSING_TYPE\()_\OPERATION
1:
  mov $in2Ptr, $in2PtrTmp
  setzi $mprocess1Fn, broadcast_op_process_1_\OP_PROCESSING_TYPE\()_\OPERATION
.ifc "\TYPE", "float"
  bri broadcast_op_worker_float_framework
.else
  setzi $mprocess2Fn, broadcast_op_process_2_\OP_PROCESSING_TYPE\()_\OPERATION
  bri broadcast_op_worker_4x16bit_framework
.endif

.size MANGLE_STR, . -MANGLE_STR
.endm

//******************************************************************************
.macro INSTANTIATE_BINARY_OP TYPE VERTEX_STATE_STUB VERTEX_NAME OPERATION OP_PROCESSING_TYPE
DEF_STACK_USAGE 0 MANGLE_STR
.section .text.MANGLE_STR
.global MANGLE_STR
.type MANGLE_STR, @function
.align 4

MANGLE_STR:
  call  $mlink, fn_load_vertex_state_\VERTEX_STATE_STUB
.ifc "\TYPE", "float"
  call  $mlink, fn_divide_work12
.else
  call  $mlink, fn_divide_work24
.endif
  // Select fast or normal loop based on the returned value, 0 = fast, 1 = normal
  setzi $mloopFn, binary_op_loop_\OP_PROCESSING_TYPE\()_fast_\OPERATION
  brz   $mscratch, 1f
  setzi $mloopFn, binary_op_loop_\OP_PROCESSING_TYPE\()_\OPERATION
1:
  setzi $mprocess1Fn, binary_op_process_1_\OP_PROCESSING_TYPE\()_\OPERATION
.ifc "\TYPE", "float"
  bri binary_op_worker_float_framework
.else
  setzi $mprocess2Fn, binary_op_process_2_\OP_PROCESSING_TYPE\()_\OPERATION
  bri binary_op_worker_4x16bit_framework
.endif
.size MANGLE_STR, . -MANGLE_STR
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
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4
  ld32 $in2PtrTmp, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  // Put outptr into in2Ptr for a comparison of array start addresses, to lead to
  // a fast path through the code
  mov $in2Ptr, $outPtr

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
  mov  $outPtr, $in1Ptr
  // This case cannot lead to a fast path - as it doesn't exist.
  // Signal this by making in2Ptr = 0 as it will get loaded later anyhow
  ld32 $in2PtrTmp, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
  mov  $in2Ptr, $mzero
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
fn_divide_work\DIVISOR\():
  // Extract worker ID
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

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

// Consider using a faster loop for the operation, relying on fetching the 2
// input operands at the same time.   To help, here's a summary of the memory map,
// based on B0 architecture:
//
// 0x40000 +------------------+
// 0x44000 |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
// 0x60000 +------------------+
//         |                  |
// 0x68000 +------------------+
//         |                  |
//         +------------------+
//         |                  |
//         +------------------+
//         |                  |
//         +------------------+
//   0x80000
//
// Memory consists of 2 regions, 0x40000 to 0x60000 and 0x60000 to 0x80000.
// They are subdivided into memory elements, each of which can be accessed once
// per cycle.  The first region consists of 8 elements mapped linearly.
// The second region also has 8 elements, but they are combined into
// pairs as 4 interleaved element pairs. In interleaved memory "odd 64 bit"
// addresses belong to one element, "even 64 bit" belong to another.
// An element is 0x4000 in length, an element pair is 0x8000 in length.
//
// We have 2 non-overlapping arrays to access and have the start address of each
// and the length, which is the same for both.  To decide if we can access the
// whole array with a ld2x64pace instruction, one of the following can be true:
//
// 1. If min(end1, end2) > 0x60000 AND start1, start 2 differ by >0x8000.
// 2. If min(end1, end2) < 0x60000 AND start1, start2 differ by >0x4000
// 3. If max(start1, start2) >=0x60000 AND one start is odd, the other even
// 4. If max(end1, end2) < 0x60000 AND lower array end is in a lower element than the higher array start
// 5. If max(start1, start2) >= 0x60000 AND lower array end is in a lower element pair than the higher array start
// 6. Some side cases where one array is non-interleaved memory, another is not - covering
//    Odd/even addresses or the array ending before the start of interleaved memory
//
// All that logic would slow things down a lot, so being pragmatic we
// just check if the 2 addresses are 32k (1 interleaved element-pair) apart plus some padding

  // Continue in the case of broadcastOp inPlace which doesn't have a fast path
  setzi $mscratch, 1
  brz $in2Ptr, 2f
  // If greater than elementSize apart we can use the fast loop
  sub $mscratch, $in1Ptr, $in2Ptr
  abs $mscratch, $mscratch
  // Allow for the 2 strided pre-reads of 64 bits (6 * 8 bytes) done before
  // starting the RPT loop in 'broadcast_op_loop_float_fast_\OPERATION\()' and
  // 'broadcast_op_loop_\TYPE\()_fast_\OPERATION\()'.
  // Add an extra 1 because we use cmpult to do a 'COMPARE LESS OR EQUAL'
  cmpult $mscratch, $mscratch, (TMEM_ELEMSIZE * 2) + (2 * 6 * 8) + 1
 2:
 // Returning 0 = fast, 1 = normal
  br  $mlink
.size MANGLE_STR_COMMON(\@), . -fn_divide_work\DIVISOR\()
.endm


//******************************************************************************
// General processing structure for float
//******************************************************************************

.section .text.MANGLE_STR_COMMON(float_processing)
.align 4
broadcast_op_worker_float_framework:
  ld32 $a2, $mzero, $in2Ptr, 0
  // Jump so that the dummy read to increment the pointer below can't cause
  // an exception
  {bri 1f
   mov  $a3, $a2}
binary_op_worker_float_framework:
  // Compute address offset for each data access, given 64 bit stride between
  // work assigned to each worker
  ld64step $a0:1, $mzero, $in2Ptr+=, $workerIdM1
1:
  ld64step $a0:1, $mzero, $in1Ptr+=, $workerIdM1
  ld64step $a0:1, $mzero, $outPtr+=, $workerIdM1
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
// General processing structure for 4x16bit vectors: half, short, unsigned short
//******************************************************************************
.section .text.MANGLE_STR_COMMON(4x16bit_processing)
.align 4
broadcast_op_worker_4x16bit_framework:
  ldb16 $a2, $mzero, $in2Ptr, 0
  // Jump so that the dummy read to increment the pointer below can't cause
  // an exception
  {bri 1f
   mov  $a3, $a2}
binary_op_worker_4x16bit_framework:
  // Compute address offset for each data access, given 64 bit stride between
  // work assigned to each worker
  ld64step $a0:1, $mzero, $in2Ptr+=, $workerIdM1
1:
  ld64step $a0:1, $mzero, $in1Ptr+=, $workerIdM1
  ld64step $a0:1, $mzero, $outPtr+=, $workerIdM1
  // Don't use the inner loop section of code at all if the result isn't needed
  // it will do a strided overread which must be avoided
  // As we will process 64 bits with no loop, decrement the count.
  // Also skip loop if nothing to do
  // This way is fast if we are going to use the inner loop
  brnzdec $mloops, 1f
  bri inner_loop_4x16bit_return
1:
  br $mloopFn

inner_loop_4x16bit_return:
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

process_2_4x16bit_return:
  st32step $a4, $mzero, $outPtr+=, 1
4:
  and $mscratch, $outCount, 1
  brz $mscratch, 3f
  // Process the last one
  ldb16 $a0, $mzero, $in1Ptr, 0
  br $mprocess1Fn

process_1_4x16bit_return:
  sort4x16lo $a0, $a4, $a1
  st32 $a0, $mzero, $outPtr, 0
3:
  exitz $mzero
.size MANGLE_STR_COMMON(4x16bit_processing), . -broadcast_op_worker_4x16bit_framework

//******************************************************************************
// Loops and single element processing for float
//******************************************************************************
.macro INSTANTIATE_BINARY_OP_float_PROCESSING OPERATION INSTRUCTIONv2 INSTRUCTIONv1
.section .text.MANGLE_STR_COMMON(float_loop1)
.align 8
  nop //Repeat body alignment
// Loop for binary variant
binary_op_loop_float_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
  ld64step $a2:3, $mzero, $in2Ptr+=, CTXT_WORKERS

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
   fnop}
  {ld64step $a2:3, $mzero, $in2Ptr+=, CTXT_WORKERS
   fnop}
2:
  \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
  bri inner_loop_float_return

//******************************************************
// Loop for binary variant
binary_op_loop_float_fast_\OPERATION\():
  // Pre load so we can pipeline the loop
  setzi $mscratch, CTXT_WORKERS
  ld2x64pace $a0:1, $a2:3, $in12Ptr+=, $mscratch, 0b0101

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld2x64pace $a0:1, $a2:3, $in12Ptr+=, $mscratch, 0b0101
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
   fnop}
2:
  \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
  bri inner_loop_float_return
.size MANGLE_STR_COMMON(float_loop1), . -binary_op_loop_float_\OPERATION\()

//******************************************************
.section .text.MANGLE_STR_COMMON(float_loop2)
.align 8
// Loop for broadcast scalar variant
broadcast_op_loop_float_\OPERATION\():
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
   fnop}
2:
  \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
  bri inner_loop_float_return

//******************************************************
.align 8

// Loop for broadcast scalar variant
broadcast_op_loop_float_fast_\OPERATION\():
  // fewer loops as it's more unrolled.
  sub $mloopFn, $mloops, 1
  // Revert back to the slower path if this is a problem
  brneg $mloopFn, broadcast_op_loop_float_\OPERATION\()
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
  setzi $mscratch, CTXT_WORKERS
  {ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  tapack $in12Ptr, $in1Ptr, $mzero, $outPtr

  rpt $mloopFn, (2f - 1f ) /8 - 1
1:
  {ldst64pace $a0:1, $a4:5, $in12Ptr+=, $mscratch, 0b0101
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
2:
  {st64pace $a4:5, $in12Ptr+=, $mscratch, 0b01
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  st64pace $a4:5, $in12Ptr+=, $mscratch, 0b01
  // Restore the (incremented in the loop) normal pointers for dealing with
  // the remaining items (if any)
  UNPACK_PTRS $in1Ptr $in2Ptr $outPtr

  bri inner_loop_float_return
.size MANGLE_STR_COMMON(float_loop), . -broadcast_op_loop_float_\OPERATION\()
//******************************************************

.section .text.MANGLE_STR_COMMON(float_instr)
.align 4
// Single trailing item instruction
binary_op_process_1_float_\OPERATION\():
  ld32 $a2, $mzero, $in2Ptr, 0
broadcast_op_process_1_float_\OPERATION\():
  \INSTRUCTIONv1 $a0, $a0, $a2
  bri process_1_float_return
.size MANGLE_STR_COMMON(float_instr), . -binary_op_process_1_float_\OPERATION\()
.endm
//******************************************************************************
// Loops and single element processing for half/short/unsigned short
//******************************************************************************
.macro INSTANTIATE_BINARY_OP_4x16bit_PROCESSING TYPE OPERATION INSTRUCTIONv4 INSTRUCTIONv2="void"
.section .text.MANGLE_STR_COMMON(\@)
.align 8
  nop // Repeat body alignment
binary_op_loop_\TYPE\()_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
  ld64step $a2:3, $mzero, $in2Ptr+=, CTXT_WORKERS

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
   fnop}
  {ld64step $a2:3, $mzero, $in2Ptr+=, CTXT_WORKERS
   fnop}
2:
  \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
  bri inner_loop_4x16bit_return


//******************************************************
binary_op_loop_\TYPE\()_fast_\OPERATION\():
  // Pre load so we can pipeline the loop
  setzi $mscratch, CTXT_WORKERS
  ld2x64pace $a0:1, $a2:3, $in12Ptr+=, $mscratch, 0b0101

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld2x64pace $a0:1, $a2:3, $in12Ptr+=, $mscratch, 0b0101
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
   fnop}
2:
  \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
  bri inner_loop_4x16bit_return

.size MANGLE_STR_COMMON(\@), . -binary_op_loop__\TYPE\()_\OPERATION\()

//******************************************************
.section .text.MANGLE_STR_COMMON(\@)
.align 8
broadcast_op_loop_\TYPE\()_\OPERATION\():
 // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
  rpt $mloops, (2f - 1f ) /8 - 1

1:
  {ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
   fnop}
2:
  \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3
  st64step $a4:5, $mzero, $outPtr+=, CTXT_WORKERS
  bri inner_loop_4x16bit_return
//******************************************************
.align 8
broadcast_op_loop_\TYPE\()_fast_\OPERATION\():
  // fewer loops as it's more unrolled.
  sub $mloopFn, $mloops, 1
  // Revert back to the slower path if this is a problem
  brneg $mloopFn, broadcast_op_loop_\TYPE\()_\OPERATION\()
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
  setzi $mscratch, CTXT_WORKERS
  {ld64step $a0:1, $mzero, $in1Ptr+=, CTXT_WORKERS
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  tapack $in12Ptr, $in1Ptr, $mzero, $outPtr

  rpt $mloopFn, (2f - 1f ) /8 - 1
1:
  {ldst64pace $a0:1, $a4:5, $in12Ptr+=, $mscratch, 0b0101
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
2:
  // Continue to use the packed addresses for 64 bit writes
  {st64pace $a4:5, $in12Ptr+=, $mscratch, 0b01
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  st64pace $a4:5, $in12Ptr+=, $mscratch, 0b01
  // But restore the (incremented in the loop) normal pointers for dealing with
  // the remaining items (if any)
  UNPACK_PTRS $in1Ptr $in2Ptr $outPtr

  bri inner_loop_4x16bit_return
.size MANGLE_STR_COMMON(\@), . -broadcast_op_loop_\TYPE\()_\OPERATION\()
//******************************************************
.align 8

.section .text.MANGLE_STR_COMMON(\@)
.align 4
binary_op_process_2_\TYPE\()_\OPERATION\():
  ld32step $a2, $mzero, $in2Ptr+=, 1
broadcast_op_process_2_\TYPE\()_\OPERATION\():
.ifc "\INSTRUCTIONv2","void"
  // Some instructions have no v2 equivalent.  At the moment this is only used for
  // bitwise ops which don't cause exceptions, but beware of the overprocessing!
  \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3
.else
  \INSTRUCTIONv2 $a4, $a0, $a2
.endif
  bri process_2_4x16bit_return

binary_op_process_1_\TYPE\()_\OPERATION\():
  ldb16 $a2, $mzero, $in2Ptr, 0
broadcast_op_process_1_\TYPE\()_\OPERATION\():
.ifc "\INSTRUCTIONv2","void"
  // Some instructions have no v2 equivalent.  At the moment this is only used for
  // bitwise ops which don't cause exceptions, but beware of the overprocessing!
  {ldb16 $a1, $mzero, $outPtr, 1
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
.else
  {ldb16 $a1, $mzero, $outPtr, 1
   \INSTRUCTIONv2 $a4, $a0, $a2}
.endif
   bri process_1_4x16bit_return
.size MANGLE_STR_COMMON(\@), . -binary_op_process_2__\TYPE\()_\OPERATION\()
.endm

//******************************************************************************
// Use the macros above to create vertex entry points
//******************************************************************************

.macro INSTANTIATE_ENTRY_POINTS TYPE OPERATION OP_PROCESSING_TYPE
  INSTANTIATE_BINARY_OP \TYPE in_place BinaryOp1DInPlace \OPERATION \OP_PROCESSING_TYPE
  INSTANTIATE_BROADCAST_OP \TYPE in_place BroadcastScalar1DInPlace \OPERATION \OP_PROCESSING_TYPE
  INSTANTIATE_BINARY_OP \TYPE non_in_place BinaryOp1D \OPERATION \OP_PROCESSING_TYPE
  INSTANTIATE_BROADCAST_OP \TYPE non_in_place BroadcastScalar1D \OPERATION \OP_PROCESSING_TYPE
.endm

INSTANTIATE_ENTRY_POINTS half ADD half
INSTANTIATE_ENTRY_POINTS half SUBTRACT half
INSTANTIATE_ENTRY_POINTS half MULTIPLY half

INSTANTIATE_ENTRY_POINTS float ADD float
INSTANTIATE_ENTRY_POINTS float SUBTRACT float
INSTANTIATE_ENTRY_POINTS float MULTIPLY float

INSTANTIATE_ENTRY_POINTS short BITWISE___AND short
INSTANTIATE_ENTRY_POINTS short BITWISE___OR short

// Note - the `OP_PROCESSING_TYPE` parameter exists because unsigned_short can use
// an identical processing loop to short for these 2 operations
INSTANTIATE_ENTRY_POINTS unsigned_short BITWISE___AND short
INSTANTIATE_ENTRY_POINTS unsigned_short BITWISE___OR short

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

INSTANTIATE_BINARY_OP_4x16bit_PROCESSING half ADD f16v4add f16v2add
INSTANTIATE_BINARY_OP_float_PROCESSING ADD f32v2add f32add

INSTANTIATE_BINARY_OP_4x16bit_PROCESSING half SUBTRACT f16v4sub f16v2sub
INSTANTIATE_BINARY_OP_float_PROCESSING SUBTRACT f32v2sub f32sub

INSTANTIATE_BINARY_OP_4x16bit_PROCESSING half MULTIPLY f16v4mul f16v2mul
INSTANTIATE_BINARY_OP_float_PROCESSING MULTIPLY f32v2mul f32mul

// Short and unsigned short
INSTANTIATE_BINARY_OP_4x16bit_PROCESSING short BITWISE___AND and64
INSTANTIATE_BINARY_OP_4x16bit_PROCESSING short BITWISE___OR or64

#endif
/* -------------------------------------------------------------------------- */
