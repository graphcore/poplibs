// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Worker-2D vertex code for Unary ops
// Based on a simplified version of the framework used for broadcast and binary ops.
// Unlike the binary ops implemented so far it seems likely that unary ops will need
// completely individual operation stub functions.  When we implement more than
// one the framework can be evolved to suit this requirement
/* -------------------------------------------------------------------------- */
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// Registers for each of the passed parameters
// vertex state, all offsets are 8-bit

// Non in place version
#define VERTEX_IN_PTR_PTR_OFFSET 0
#define VERTEX_OUT_PTR_PTR_OFFSET 4
#define VERTEX_OUT_COUNT_OFFSET 8

// In place version
#define VERTEX_INPLACE_INOUT_PTR_PTR_OFFSET 0
#define VERTEX_INPLACE_INOUT_COUNT_OFFSET 4

#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2
#define LOG2_HALF_VECTOR_WIDTH 2

// Register aliases
#define inPtr m0
#define inPtrPtr m1
#define outPtr m2
#define outCount m3

#define mprocess2Fn m4
#define mloopFn m5
#define mprocess1Fn m6

#define m2DVectorLoop m7
#define mloops m8
#define outPtrPtr m9
#define outerLoops m10
#define mscratch m11

// Naming / name mangling
#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__UnaryOpCommon_##SUFFIX
#define MANGLE_STR __runCodelet_popops__\VERTEX_NAME\()___popops__expr__UnaryOpType__\OPERATION\()_\TYPE

// The function stubs for the actual operations
#include "unaryOpsOps.h.S"

///******************************************************************************
// Entry stub macros, one per operation:
// inplace/non-inplace
//******************************************************************************
.macro INSTANTIATE_2D TYPE VARIANT VERTEX_NAME OPERATION
DEF_STACK_USAGE 0 MANGLE_STR
.section .text.MANGLE_STR
.global MANGLE_STR
.type MANGLE_STR, @function
.align 4

MANGLE_STR:
  // load vertex state
.ifc "\VERTEX_NAME", "UnaryOp2DInPlace"
  ld32 $outPtrPtr, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_PTR_PTR_OFFSET/4
  ld32 $outerLoops, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_COUNT_OFFSET/4
.else
  ld32 $inPtrPtr, $mzero, $mvertex_base, VERTEX_IN_PTR_PTR_OFFSET/4
  ld32 $outerLoops, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/4
  ld32 $outPtrPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_PTR_OFFSET/4
.endif

  // Load pointers to the main loop, 1 of and 2 of functions
.ifc "\OPERATION","SIGNUM"
  LOAD_FUNCTION_PTRS_AND_CONSTANTS_SIGNUM \VARIANT \OPERATION
.else
  LOAD_FUNCTION_PTRS \VARIANT \OPERATION
.endif
  // We need a slightly different 2D vector loop for inplace and non-inplace versions,
  // so set up a function pointer for that.
  setzi $m2DVectorLoop, unary_op_worker_2D_4x16bit_framework_\VERTEX_NAME
  brnzdec $outerLoops, unary_op_worker_2D_4x16bit_framework
  exitz $mzero
.size MANGLE_STR, . -MANGLE_STR
.endm

//******************************************************************************
// General processing structure for half - 2D worker vertices
//******************************************************************************
.section .text.MANGLE_STR_COMMON(4x16bit_processing_2D)
.align 4
unary_op_worker_2D_4x16bit_framework:
  // Load pointers to the inner 2D output (or in-out) vectors and associated
  // counter
  ld32step $outPtr, $mzero, $outPtrPtr+=, 1
  ld32step $outCount, $mzero, $outPtrPtr+=, 1
  br       $m2DVectorLoop
unary_op_worker_2D_4x16bit_framework_UnaryOp2DInPlace:
  mov      $inPtr, $outPtr
  bri      1f
unary_op_worker_2D_4x16bit_framework_UnaryOp2D:
  ld32step $inPtr, $mzero, $inPtrPtr+=, 1
1:
  shr      $mloops, $outCount, LOG2_HALF_VECTOR_WIDTH
  // Don't use the inner loop section of code at all if the result isn't needed
  // As we will process 64 bits with no loop, decrement the count.
  // Also skip loop if nothing to do
  brnzdec $mloops, 1f
  // Continue as there may be trailing 1-of or 2-of even if no inner loop is run
  bri 2f
1:
  // Call the inner loop
  br $mloopFn

inner_loop_4x16bit_return_2D:
  st64step $a2:3, $mzero, $outPtr+=, 1
2:
  and $mscratch, $outCount, 2
  brz $mscratch, 4f
  // Process a remaining pair
  ld32step $a0, $mzero, $inPtr+=,1
  br $mprocess2Fn

process_2_4x16bit_return_2D:
  st32step $a0, $mzero, $outPtr+=, 1
4:
  and $mscratch, $outCount, 1
  brz $mscratch, 3f
  // Process the last one
  ldb16 $a0, $mzero, $inPtr, 0
  br $mprocess1Fn

process_1_4x16bit_return_2D:
  sort4x16lo $a0, $a0, $a1
  st32 $a0, $mzero, $outPtr, 0
3:
  brnzdec $outerLoops, unary_op_worker_2D_4x16bit_framework
  exitz $mzero
.size MANGLE_STR_COMMON(4x16bit_processing_2D), . -unary_op_worker_2D_4x16bit_framework

//******************************************************************************
// Use the macros above to create vertex entry points
//******************************************************************************

INSTANTIATE_2D half 2D UnaryOp2D SIGNUM
INSTANTIATE_2D half 2D UnaryOp2DInPlace SIGNUM

INSTANTIATE_2D short 2D UnaryOp2D BITWISE___NOT
INSTANTIATE_2D short 2D UnaryOp2DInPlace BITWISE___NOT
INSTANTIATE_2D unsigned_short 2D UnaryOp2D BITWISE___NOT
INSTANTIATE_2D unsigned_short 2D UnaryOp2DInPlace BITWISE___NOT
//******************************************************************************
// Use the macros above to create each individual operation code
//******************************************************************************

// Signum specific
INSTANTIATE_UNARY_OP_HALF_PROCESSING_SIGNUM SIGNUM 2D 1

// General - just NOT at the momnent, supporting short and unsigned_short
INSTANTIATE_UNARY_OP_4x16bit_PROCESSING BITWISE___NOT not64 2D 1

#endif
/* -------------------------------------------------------------------------- */
