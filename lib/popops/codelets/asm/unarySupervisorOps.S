// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Supervisor vertex code for Unary ops
// Based on a simplified version of the framework used for broadcast and binary ops.
// Unlike the binary ops implemented so far it seems likely that unary ops will need
// completely individual operation stub functions.  When we implement more than
// one the framework can be evolved to suit this requirement
/* -------------------------------------------------------------------------- */
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "workDivision.h.S"

// Registers for each of the passed parameters
// vertex state, all offsets are 8-bit

// Non in place version
#define VERTEX_IN_PTR_OFFSET 0
#define VERTEX_OUT_PTR_OFFSET 4
#define VERTEX_OUT_COUNT_OFFSET 8

// In place version
#define VERTEX_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_INPLACE_INOUT_COUNT_OFFSET 4

// Register aliases
#define inPtr m0
#define outPtr m2
#define outCount m3

#define mlink m4
#define mprocess2Fn m4
#define mloopFn m5
#define mprocess1Fn m6

#define mloops m8
#define outPtrConst m9
#define workerIdM1 m10
#define mscratch m11

// Naming / name mangling
#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__UnaryOpCommon_##SUFFIX
#define MANGLE_STR __runCodelet_popops__\VERTEX_NAME\()___popops__expr__UnaryOpType__\OPERATION\()_\TYPE

// The function stubs for the actual operations
#include "unaryOpsOps.h.S"

//******************************************************************************
// Entry stub macros, one per operation:
// inplace/non-inplace
//******************************************************************************
.macro INSTANTIATE_UNARY_OP TYPE VARIANT VERTEX_NAME OPERATION
DEF_STACK_USAGE 0 MANGLE_STR
.section .text.MANGLE_STR
.global MANGLE_STR
.type MANGLE_STR, @function
.align 4

MANGLE_STR:
  // load vertex state
.ifc "\VERTEX_NAME", "UnaryOp1DInPlace"
  ld32 $inPtr, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_COUNT_OFFSET/4
  mov  $outPtr, $inPtr
.else
  ld32 $inPtr, $mzero, $mvertex_base, VERTEX_IN_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
.endif
   call  $mlink, fn_divide_work24

  // Load pointers to the main loop, 1 of and 2 of functions
.ifc "\OPERATION","SIGNUM"
  LOAD_FUNCTION_PTRS_AND_CONSTANTS_SIGNUM \VARIANT \OPERATION
.else
  LOAD_FUNCTION_PTRS \VARIANT \OPERATION
.endif
  bri unary_op_worker_4x16bit_framework
.size MANGLE_STR, . -MANGLE_STR
.endm

//******************************************************************************
.macro UNARY_OP_DIVIDE_WORK DIVISOR SHIFTS_FOR_GRAINSIZE
.section .text.MANGLE_STR_COMMON(\@)
.align 4
fn_divide_work\DIVISOR\():
  // Extract worker ID
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // Loops for this worker: divide by 12 or 24, rounding up according to the workerId
  DIVIDE_BY_WORKER $outCount $workerIdM1 $mscratch $mloops \SHIFTS_FOR_GRAINSIZE
  // Hold onto the outPtr to use to decide which worker to process the remainder
  // results
  mov $outPtrConst, $outPtr

  br  $mlink
.size MANGLE_STR_COMMON(\@), . -fn_divide_work\DIVISOR\()
.endm

//******************************************************************************
// General processing structure for vectorised 4x16bit data types
//******************************************************************************
.section .text.MANGLE_STR_COMMON(4x16bit_processing)
.align 4
unary_op_worker_4x16bit_framework:
  ld64step $a0:1, $mzero, $inPtr+=, $workerIdM1
  ld64step $a0:1, $mzero, $outPtr+=, $workerIdM1
  // Don't use the inner loop section of code at all if the result isn't needed
  // it will do a strided overread which must be avoided
  // As we will process 64 bits with no loop, decrement the count.
  // Also skip loop if nothing to do
  // This way is fast if we are going to use the inner loop
  brnzdec $mloops, 1f
  bri 2f
1:
  br $mloopFn

inner_loop_4x16bit_return_1D:
  st64step $a2:3, $mzero, $outPtr+=, CTXT_WORKERS
2:
  // Here we have done all groups of 3 halves for every worker, no overread.
  // Use the worker which is pointing to the next half to process the last 3
  // (if needed).

  // Advance outPtrConst to point of the next output if we ignore the remainder,
  // the worker with that pointer address will do the remainders if any
  andc $mscratch, $outCount, 3
  ldb16step $azero, $mzero, $outPtrConst+=, $mscratch
  cmpeq $mscratch, $outPtrConst, $outPtr
  brz $mscratch, 3f

  and $mscratch, $outCount, 2
  brz $mscratch, 4f
  // Process a remaining pair
  ld32step $a0, $mzero, $inPtr+=,1
  br $mprocess2Fn

process_2_4x16bit_return_1D:
  st32step $a0, $mzero, $outPtr+=, 1
4:
  and $mscratch, $outCount, 1
  brz $mscratch, 3f
  // Process the last one
  ldb16 $a0, $mzero, $inPtr, 0
  br $mprocess1Fn

process_1_4x16bit_return_1D:
  sort4x16lo $a0, $a0, $a1
  st32 $a0, $mzero, $outPtr, 0
3:
  exitz $mzero
.size MANGLE_STR_COMMON(4x16bit_processing), . -unary_op_worker_4x16bit_framework

//******************************************************************************
// Use the macros above to create vertex entry points
//******************************************************************************

INSTANTIATE_UNARY_OP half 1D UnaryOp1D SIGNUM
INSTANTIATE_UNARY_OP half 1D UnaryOp1DInPlace SIGNUM

INSTANTIATE_UNARY_OP short 1D UnaryOp1D BITWISE___NOT
INSTANTIATE_UNARY_OP short 1D UnaryOp1DInPlace BITWISE___NOT
INSTANTIATE_UNARY_OP unsigned_short 1D UnaryOp1D BITWISE___NOT
INSTANTIATE_UNARY_OP unsigned_short 1D UnaryOp1DInPlace BITWISE___NOT
//******************************************************************************
// Use the macros above to create code that can be shared between different
// opertations using this framework
//******************************************************************************

UNARY_OP_DIVIDE_WORK 24 LOG2_HALF_ATOM_SIZE

//******************************************************************************
// Use the macros above to create each individual operation code
//******************************************************************************

// Signum specific
INSTANTIATE_UNARY_OP_HALF_PROCESSING_SIGNUM SIGNUM 1D CTXT_WORKERS

// General - just NOT at the momnent, supporting short and unsigned_short
INSTANTIATE_UNARY_OP_4x16bit_PROCESSING BITWISE___NOT not64 1D CTXT_WORKERS
#endif
/* -------------------------------------------------------------------------- */
