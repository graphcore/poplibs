// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Macros to create the actual operations for unary Ops, in a header so that
// they can be shared between Supervisor and 2DWorker vertices.
// They rely on register aliases within those 2 files which should probably be
// kept consistent.
//
// The general structure for calling these uses function pointers, there is no
// "indirect call" instruction. So we need a fixed return address which is
// specific to where the functions were called from.  This all leads to these
// being macros and not subroutines shared between Supervisor and 2DWorker
/* -------------------------------------------------------------------------- */

// Naming / name mangling
#define MANGLE_STR_MACRO(SUFFIX) __runCodelet_popops__UnaryOpCommon_\NAME_STR\()_##SUFFIX

// Constants
#define HALF_MINUS_ONE 0xbc00

//******************************************************************************
// Macros for half SIGNUM
//******************************************************************************
// Load SIGNUM specific constants - operation described in the SIGNUM loop
// Can be bundled with the load of function pointers for loop, 1-of and 2-of
// processing
.macro   LOAD_FUNCTION_PTRS_AND_CONSTANTS_SIGNUM NAME_STR OPERATION
  {setzi $mloopFn, unary_op_loop_half_\NAME_STR\()_\OPERATION
   setzi $a6, HALF_MINUS_ONE}

  {setzi $mprocess1Fn, unary_op_process_1_half_\NAME_STR\()_\OPERATION
   sort4x16lo $a6,$a6,$a6}
  {setzi $mprocess2Fn, unary_op_process_2_half_\NAME_STR\()_\OPERATION
   mov  $a7,$a6}
.endm

// Actual loop, 1-of and 2-of processign functions for SIGNUM
.macro INSTANTIATE_UNARY_OP_HALF_PROCESSING_SIGNUM NAME_STR STRIDE
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

.section .text.MANGLE_STR_MACRO(half_loop)
.align 8
unary_op_loop_half_\NAME_STR\()_SIGNUM:
   // Pre load and process we can pipeline the loop
  ld64step $a0:1, $mzero, $inPtr+=, \STRIDE
  f16v4cmplt $a2:3, $a0:1, $azeros
  f16v4cmplt $a4:5, $azeros, $a0:1
  and64 $a2:3, $a2:3, $a6:7

  {rpt $mloops, (2f - 1f ) /8 - 1
   and64 $a4:5, $a4:5, $a6:7}
1:
    {ld64step $a0:1, $mzero, $inPtr+=, \STRIDE
    f16v4sub $a2:3,$a2:3,$a4:5}
    {st64step $a2:3, $mzero, $outPtr+=, \STRIDE
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
  st64step $a2:3, $mzero, $outPtr+=, \STRIDE
  bri inner_loop_half_return_\NAME_STR
.size MANGLE_STR_MACRO(half_loop), . -unary_op_loop_half_\NAME_STR\()_SIGNUM
//******************************************************

.section .text.MANGLE_STR_MACRO(half_instr)
.align 4
unary_op_process_2_half_\NAME_STR\()_SIGNUM:
  f16v2cmplt $a2,$a0,$azero
  f16v2cmplt $a4, $azero, $a0
  // and64 is just a logical operation and doesn't produce exceptions so no
  // need to initialise $a3 and $a5 which won't always have been used by the
  // time we get here
  and64 $a2:3, $a2:3, $a6:7
  and64 $a4:5, $a4:5, $a6:7
  {bri process_2_half_return_\NAME_STR
   f16v2sub $a0, $a2, $a4}

unary_op_process_1_half_\NAME_STR\()_SIGNUM:
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
  {bri process_1_half_return_\NAME_STR
   f16v2sub $a0, $a2, $a4}
.size MANGLE_STR_MACRO(half_instr), . -unary_op_process_2_half_\NAME_STR\()_SIGNUM
.endm

//******************************************************************************
// Macros for generic functions - if no constants to be loaded this can
// used to set up the pointers to loop, 2-of and 1-of functions
//******************************************************************************
.macro   LOAD_FUNCTION_PTRS NAME_STR OPERATION
  // If we use this framework for other operations, there may be no constant
  // load required
  setzi $mloopFn, unary_op_loop_half_\NAME_STR\()_\OPERATION
  setzi $mprocess1Fn, unary_op_process_1_half_\NAME_STR\()_\OPERATION
  setzi $mprocess2Fn, unary_op_process_2_half_\NAME_STR\()_\OPERATION
.endm

#endif
/* -------------------------------------------------------------------------- */
