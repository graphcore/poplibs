// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Supervisor vertex code for Unary ops
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
#define VERTEX_IN_PTR_OFFSET 0
#define VERTEX_OUT_PTR_OFFSET 4
#define VERTEX_OUT_COUNT_OFFSET 8

// In place version
#define VERTEX_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_INPLACE_INOUT_COUNT_OFFSET 4

#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2

// Register aliases
#define inPtr m0
#define outPtr m2
#define outCount m3

#define mlink m4
#define mprocess2Fn m4
#define mloopFn m5
#define mprocess1Fn m6

#define mloops m8
#define remainder m9
#define workerIdM1 m10
#define mscratch m11

// Naming / name mangling
#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__UnaryOpCommon_##SUFFIX
#define MANGLE_STR_FLOAT __runCodelet_popops__\INPLACE_STR\()___popops__expr__UnaryOpType__\OPERATION\()_float
#define MANGLE_STR_HALF __runCodelet_popops__\INPLACE_STR\()___popops__expr__UnaryOpType__\OPERATION\()_half

// Constants
#define HALF_MINUS_ONE 0xbc00
//******************************************************************************
// Entry stub macros, one per operation:
// inplace/non-inplace
//******************************************************************************
.macro INSTANTIATE_UNARY_OP_HALF INPLACE_STR OPERATION
DEF_STACK_USAGE 0 MANGLE_STR_HALF
.section .text.MANGLE_STR_HALF
.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4

MANGLE_STR_HALF:
  // load vertex state
.ifc "\INPLACE_STR", "UnaryOp1DInPlace"
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
  // Load a SIGNUM specific constant - operation described in the SIGNUM loop
  {setzi $mloopFn, unary_op_loop_half_\OPERATION
   setzi $a6, HALF_MINUS_ONE}

  {setzi $mprocess1Fn, unary_op_process_1_half_\OPERATION
   sort4x16lo $a6,$a6,$a6}
  {setzi $mprocess2Fn, unary_op_process_2_half_\OPERATION
   mov  $a7,$a6}
.else
  // If we use this framework for other operations, there may be no constant
  // load required
  setzi $mloopFn, unary_op_loop_half_\OPERATION
  setzi $mprocess1Fn, unary_op_process_1_half_\OPERATION
  setzi $mprocess2Fn, unary_op_process_2_half_\OPERATION
.endif
  bri unary_op_worker_half_framework
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm

//******************************************************************************
.macro UNARY_OP_DIVIDE_WORK SHIFTS_TO_DIV DIVISOR SHIFTS_FOR_GRAINSIZE
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

  br  $mlink
.size MANGLE_STR_COMMON(\@), . -fn_divide_work\DIVISOR\()
.endm

//******************************************************************************
// General processing structure for half
//******************************************************************************
.section .text.MANGLE_STR_COMMON(half_processing)
.align 4
unary_op_worker_half_framework:
  ld64step $a0:1, $mzero, $inPtr+=, $workerIdM1
  ld64step $a0:1, $mzero, $outPtr+=, $workerIdM1
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
  ld32step $a0, $mzero, $inPtr+=,1
  br $mprocess2Fn

process_2_half_return:
  st32step $a0, $mzero, $outPtr+=, 1
4:
  and $mscratch, $outCount, 1
  brz $mscratch, 3f
  // Process the last one
  ldb16 $a0, $mzero, $inPtr, 0
  br $mprocess1Fn

process_1_half_return:
  sort4x16lo $a0, $a0, $a1
  st32 $a0, $mzero, $outPtr, 0
3:
  exitz $mzero
.size MANGLE_STR_COMMON(half_processing), . -unary_op_worker_half_framework

//******************************************************************************
// Loops and single element processing for half
//******************************************************************************
.macro INSTANTIATE_UNARY_OP_HALF_PROCESSING_SIGNUM
  // SIGNUM : x < 0  : -1
  //          x == 0 : 0
  //          x >  0 : 1
  // cmplt gives TFPU_FP16_TRUE: 0xffff or TFPU_FP16_FALSE: 0x0000
  // So
  //         | cmplt         | and64 0xbc00 == -1       | sub
  //         | $a2:3  $a4:5  | $a2:3       $a4:5        | $a2:3  $a4:5
  // if(x<0) | 0xffff 0x0000 | 0xbc00 = -1 0x0000 = 0   | -1     -0    = -1
  // if(x==0)| 0x0000 0x0000 | 0x0000 = 0  0x0000 = 0   | 0      - 0   = 0
  // if(x>0) | 0x0000 0xffff | 0x0000 = 0  0xbc00 = -1  | 0      -(-1) = +1

.section .text.MANGLE_STR_COMMON(half_loop)
.align 8
unary_op_loop_half_SIGNUM:
   // Pre load and process we can pipeline the loop
  ld64step $a0:1, $mzero, $inPtr+=, NUM_WORKERS
  f16v4cmplt $a2:3, $a0:1, $azeros
  f16v4cmplt $a4:5, $azeros, $a0:1
  and64 $a2:3, $a2:3, $a6:7

  {rpt $mloops, (2f - 1f ) /8 - 1
   and64 $a4:5, $a4:5, $a6:7}
1:
    {ld64step $a0:1, $mzero, $inPtr+=, NUM_WORKERS
    f16v4sub $a2:3,$a2:3,$a4:5}
    {st64step $a2:3, $mzero, $outPtr+=, NUM_WORKERS
    f16v4cmplt $a2:3, $a0:1, $azeros}
    {nop
     f16v4cmplt $a4:5, $azeros, $a0:1}
    {nop
     and64 $a2:3, $a2:3, $a6:7}
    {nop
     and64 $a4:5, $a4:5, $a6:7}
2:
  // Complete processing the last one and store
  f16v4sub $a2:3,$a2:3,$a4:5
  st64step $a2:3, $mzero, $outPtr+=, NUM_WORKERS
  bri inner_loop_half_return
.size MANGLE_STR_COMMON(half_loop), . -unary_op_loop_half_SIGNUM
//******************************************************

.section .text.MANGLE_STR_COMMON(half_instr)
.align 4
unary_op_process_2_half_SIGNUM:
  f16v2cmplt $a2,$a0,$azero
  f16v2cmplt $a4, $azero, $a0
  // and64 is just a logical operation and doesn't produce exceptions so no
  // need to initialise $a3 and $a5 which won't always have been used by the
  // time we get here
  and64 $a2:3, $a2:3, $a6:7
  and64 $a4:5, $a4:5, $a6:7
  {bri process_2_half_return
   f16v2sub $a0, $a2, $a4}

unary_op_process_1_half_SIGNUM:
  // Load value to combine and store after returning
  // Input is a single half but broadcast so v2 ops are not overprocessing
  {ldb16 $a1, $mzero, $outPtr, 1
   f16v2cmplt $a2,$a0,$azero}
  f16v2cmplt $a4, $azero, $a0
  // and64 is just a logical operation and doesn't produce exceptions so no
  // need to initialise $a3 and $a5 which won't always have been used by the
  // time we get here
  and64 $a2:3, $a2:3, $a6:7
  and64 $a4:5, $a4:5, $a6:7
  {bri process_1_half_return
   f16v2sub $a0, $a2, $a4}
.size MANGLE_STR_COMMON(half_instr), . -binary_op_process_2_half_SIGNUM
.endm

//******************************************************************************
// Use the macros above to create vertex entry points
//******************************************************************************

INSTANTIATE_UNARY_OP_HALF UnaryOp1D SIGNUM
INSTANTIATE_UNARY_OP_HALF UnaryOp1DInPlace SIGNUM

//******************************************************************************
// Use the macros above to create code that can be shared between different
// opertations using this framework
//******************************************************************************

UNARY_OP_DIVIDE_WORK 20 24 LOG2_HALF_ATOM_SIZE

//******************************************************************************
// Use the macros above to create each individual operation code
//******************************************************************************

// Just a signum function for now, but other operations can be supported by the
// framework
INSTANTIATE_UNARY_OP_HALF_PROCESSING_SIGNUM

#endif
/* -------------------------------------------------------------------------- */
