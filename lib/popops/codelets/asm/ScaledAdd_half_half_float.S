// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "ScaledAddSupervisor.inc"

#define VERTEX_2D_ADD_SCALE_FLOAT_CONST_FAST __runCodelet_popops__ScaledAdd2D___half_half_float_true_true
#define VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST __runCodelet_popops__ScaledAdd2D___half_half_float_false_true
#define VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW __runCodelet_popops__ScaledAdd2D___half_half_float_true_false
#define VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW __runCodelet_popops__ScaledAdd2D___half_half_float_false_false
#define VERTEX_2D_SCALE_FLOAT_COMMON __ScaledAdd2D___half_half_float_common

#define VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST __runCodelet_popops__ScaledAddSupervisor___half_half_float_true_true
#define VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST __runCodelet_popops__ScaledAddSupervisor___half_half_float_false_true
#define VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW __runCodelet_popops__ScaledAddSupervisor___half_half_float_true_false
#define VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW __runCodelet_popops__ScaledAddSupervisor___half_half_float_false_false
#define VERTEX_SV_SCALE_FLOAT_COMMON __ScaledAddSupervisor___half_half_float_common
#define VERTEX_SV_SCALE_FLOAT_COMMON_CONST __ScaledAddSupervisor___half_half_float_common_const

#define VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_FAST __runCodelet_popops__ScaledSubtract2D___half_float_true
#define VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW __runCodelet_popops__ScaledSubtract2D___half_float_false
#define VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_FAST __runCodelet_popops__ScaledSubtractSupervisor___half_half_float_true
#define VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW __runCodelet_popops__ScaledSubtractSupervisor___half_half_float_false

// The bulk of the Supervisor task processing which is common to all scaled-add
// variants has been implemented in a different file: ScaledAddSupervisor_fp.S.
// They use this define to produce labels
#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty

// A C function to check the accuracy when casting.  Extracted from the elf file
#define CHECK_ACCURACY_WHEN_CAST _ZN6popops32checkAccuracyWhenCastComputeImplIfDhEEbT_f
//******************************************************************************
// Common definitions and subroutines used by both the 2D and Supervisor cases
//******************************************************************************

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR64
#define SCALED_PTR64_SHL_BITS 3
#endif

// Integer variables
#define dataPtr m1
#define dataBPtr m5
#define dataSizeD2 m4
#define stride m9
#define strideX2 m6

// Float variables
#define data0 a0:1
#define data0i0 a0
#define data0i1 a1
#define dataB0 a2:3
#define dataB0i0 a2
#define dataB0i1 a3
#define data1 a4:5
#define data1i0 a4
#define data1i1 a5
#define dataB1 a6:7
#define dataB1i0 a6
#define dataB1i1 a7

// Scratch variables
#define mscratch m10
#define mscratch2 m6
#define ascratch a6


// Subroutine: Loop Kernel for ScaledAdd(data_half, dataB_half, factor_float) case:
//             Every iteration processes 2 halves
//
// The calculation for each input half value pair d, dB and float scaling_factor:
//     d' = Cast_Float(d)
//     dB' = Cast_Float(dB)
//     r' = d' + (scaling_factor * dB')
//     r = Cast_Half(r')
//     d = r                    // The 'd' is updated in-place
//
// TAS is loaded with the value of the scaling factor.
//     $TAS <- scaling_factor
//
// The function takes the following inputs.
//  1. $TAS is the scaling factor
//  2. $dataPtr points to the first input array pointer
//  3. $dataBPtr points to the second input array pointer
//  4. $dataSizeD2 is the number of 2xhalfs to process
//  5. $stride is the fixed offset between consecutive half pairs
//
//  NOTE: The final store instruction should be executed by the calling program
//        immediately after the function has returned.
//
DEF_STACK_USAGE  0  scaled_add_data_half_factor_float
.section .text.scaled_add_data_half_factor_float, FUNCTION_IS_WORKER
.type scaled_add_data_half_factor_float, @function

.align 8
scaled_add_data_half_factor_float:
  ld32 $data0i0, $mzero, $dataPtr, 0

  // Cast_Float data0
  {ld32step $dataB0i0, $mzero, $dataBPtr+=, $stride
   f16v2tof32 $data0, $data0i0}

  // Cast_Float dataB0
  {cmpeq $mscratch2, $dataSizeD2, 1
   f16v2tof32 $dataB0, $dataB0i0}

  // Use a 2-deep pipeline
  // handle the single-iteration case
  {brnz $mscratch2, .Lscale_float_flush
   f32v2axpy $azeros, $dataB0, $data0}

  // Repeat loop for N-2 iterations
  add $dataSizeD2, $dataSizeD2, -2

  ld32 $data0i0, $mzero, $dataPtr, $stride

  // Cast_Float data0
  {ld32step $dataB0i0, $mzero, $dataBPtr+=, $stride
   f16v2tof32 $data0, $data0i0}

  // The first array is an input/output and is pointed to by $dataPtr. The code
  // has been designed to only increment $dataPtr using the store instruction.
  // A total of 3 array values will be read out before the store instruction
  // gets executed for the first instruction. Therefore, in order to load the
  // 3rd array value, an offset of 2 x $stride will be required.
  mul $strideX2, $stride, 2

  // Cast_Float dataB0
  {rpt $dataSizeD2, (2f-1f)/8-1
   f16v2tof32 $dataB0, $dataB0i0}
1:
  {ld32 $data0i0, $mzero, $dataPtr, $strideX2
   f32v2axpy $data1, $dataB0, $data0}

  // Cast_Half the result of the previous iteration
  {ld32step $dataB0i0, $mzero, $dataBPtr+=, $stride
   f32v2tof16 $data1i0, $data1}

  // Store half-casted result
  // Cast_Float data0
  {st32step $data1i0, $mzero, $dataPtr+=, $stride
   f16v2tof32 $data0, $data0i0}

  // Cast_Float dataB0
  {nop
   f16v2tof32 $dataB0, $dataB0i0}

2:
  // Obtain the 32-bit result for the second from last iteration
  f32v2axpy $data0, $dataB0, $data0

  // Cast_Half the result of the second from last iteration and store
  f32v2tof16 $data0i0, $data0
  st32step $data0i0, $mzero, $dataPtr+=, $stride

.Lscale_float_flush:
  // Flush the Accumulators to get the final 32-bit result
  f32v2axpy $data0, $azeros, $azeros

  // Cast_Half the result of the final iteration and store
  //
  // Due to the use of bundling with the final return branch instruction,
  // the final store instruction must be executed by the calling program.
  {
    br $lr
    f32v2tof16 $data0i0, $data0
  }

// Subroutine: Process ScaledAdd(data_half, dataB_half, factor_float) for a
//             single half
//
// The calculation:
//     d' = Cast_Float(d)
//     dB' = Cast_Float(dB)
//     r' = d' + (scaling_factor * dB')
//     r = Cast_Half(r')
//     d = r                    // The 'd' is updated in-place
//
// TAS is loaded with the value of the scaling factor.
//     $TAS <- scaling_factor
//
// The function takes the following inputs.
//  1. $TAS is the scaling factor
//  2. $dataPtr points to the first input array pointer
//  3. $dataBPtr points to the second input array pointer
//
DEF_STACK_USAGE 0 scaled_add_data_half_factor_float_scalar
.section .text.scaled_add_data_half_factor_float_scalar
.type scaled_add_data_half_factor_float_scalar, @function

.align 4
scaled_add_data_half_factor_float_scalar:
  ldb16 $data1i0, $mzero, $dataPtr, 0

  // Cast_Float data0
  // Only a single half needs to be cast. However the f32axpy instruction
  // for a single float is not available. In order to provide a well-defined
  // value to the 64-bit accumulators, we have chosen to cast two halves. Note
  // that both halves would be identical.
  {
    ldb16 $dataB1i0, $mzero, $dataBPtr, 0
    f16v2tof32 $data1, $data1i0
  }

  // Cast_Float dataB0
  f16v2tof32 $dataB1, $dataB1i0

  f32v2axpy $azeros, $dataB1, $data1

  // Only 32-bit stores are supported. Hence in order to store 16-bits,
  // perform extra half-read for read-modify-write
  //
  // Flush the Accumulators to get the 32-bit result
  {
    ldb16 $ascratch, $mzero, $dataPtr, 1
    f32v2axpy $data1, $azeros, $azeros
  }

  // Cast the result to Half and modify-write
  f32tof16 $data1i0, $data1i0

  {
    br $lr
    roll16 $data1i0, $data1i0, $ascratch
  }

//******************************************************************************
// 2D case
//******************************************************************************

// Variable offsets (in bytes)
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 4
#define VERTEX_DATA_B_OFFSET 8
#define VERTEX_SCALE_OFFSET 12
#define VERTEX_TOLERANCE_OFFSET 16

// Integer variables
#define outData m0
#define outDataSize m11
#define outDataB m2
#define dataSize m4
#define origDataSize m3

// Float variables

#define factor a7
#define factorTmp a6

// Shared with the all half version - be careful!
#define memConstraints m11

// Registers used in calling the C function checkAccuracyWhenCast
#define C_CALL_PARAM_TOLERANCE a1
#define C_CALL_SCALE a0
#define C_CALL_RETURN m0

.globl VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW
.type VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW, @function
.globl VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST
.type VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST, @function

.macro CALL_CHECK_ACCURACY_WHEN_CAST_2D
  mov $sp, $m12
  // Call the C function to check if we can use scale as a half
  // Pass in scale and tolerance
  ld32  $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $C_CALL_SCALE, $mzero, $dataPtr, 0
  // a7 is a callee save register so is safe to store the FP_CTL register
  // which will be modified inside the called function
  {ld32  $C_CALL_PARAM_TOLERANCE, $mvertex_base, $mzero, VERTEX_TOLERANCE_OFFSET/4
  uget  $a7, CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK }
  call $lr, CHECK_ACCURACY_WHEN_CAST
  uput CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK, $a7
.endm

// Common code used by both the plus/subtract workers. The optional argument
// is intended to allow the scale factor argument to be negated by the subtract
// worker.
.macro PERFORM_VERTEX_2D_SCALE_FLOAT_TENSOR OPTIONAL_SCALE_FACTOR_MODIFIER:vararg
  ld32  $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $factor, $mzero, $dataPtr, 0
  \OPTIONAL_SCALE_FACTOR_MODIFIER
  
  { brz   $C_CALL_RETURN, VERTEX_2D_SCALE_FLOAT_COMMON
    f32tof16 $factorTmp, $factor}
  // Otherwise cast, broadcast and branch into the "all half" version
  { bri   __ScaledAdd2D___half_common
    roll16 $factor, $factorTmp, $factorTmp}
.endm

DEF_STACK_SIZE_OWN 0 .text.VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW
.section .text.VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW
.align 4
VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW:
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D
  // For use when we enter the half, half, half version
  setzi $memConstraints, 0
  bri   1f
VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST:
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D

  setzi $memConstraints, 1
1:
  PERFORM_VERTEX_2D_SCALE_FLOAT_TENSOR

.size VERTEX_2D_SCALE_FLOAT, .-VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW

.globl VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW
.type VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW, @function
.globl VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_FAST
.type VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_FAST, @function

DEF_STACK_SIZE_OWN 0 .text.VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW
.section .text.VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW
.align 4
VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW:
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D
  // For use when we enter the half, half, half version
  setzi $memConstraints, 0
  bri   1f
VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_FAST:
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D

  setzi $memConstraints, 1
1:
  PERFORM_VERTEX_2D_SCALE_FLOAT_TENSOR f32sub $factor, $azero, $factor
  
.size VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW, .-VERTEX_2D_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW

.globl VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW
.type VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW, @function
.globl VERTEX_2D_ADD_SCALE_FLOAT_CONST_FAST
.type VERTEX_2D_ADD_SCALE_FLOAT_CONST_FAST, @function

DEF_STACK_USAGE 0 .text.VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW
.section .text.VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW
.align 4
// The fastest implementation happens to use more Aux instructions than Main
// instructions. Since memory access instructions use the Main path, the
// efficiency of these instructions in a memory-constrained scenario do not lead
// to a speed up of the loop kernel. Hence, this variant of ScaledAdd has only
// a single implementation regardless of the placement of the inputs in memory.
VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW:
VERTEX_2D_ADD_SCALE_FLOAT_CONST_FAST:
  // load vertex state specific to this version of the vertex : k, constant
  ld32 $factor, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4

VERTEX_2D_SCALE_FLOAT_COMMON:
  // load common vertex state
 ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
 ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET/4

  {
    ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
    // setup $TAS for the f16v4mix instructions below.
    uput $TAS, $factor
  }

  // All the data is allocated contiguously for this worker. So use stride=1
  // when traversing the input tensors for the inner loop.
  setzi $stride, 1

  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Lscale_float_outer_loop:
#ifdef VECTOR_AVAIL_SHORT_SPAN
  ld32step $dataPtr, $mzero, $outData+=, 1
  shr $origDataSize, $dataPtr, SHORT_SPAN_PTR_SIZE
  shl $dataPtr, $dataPtr, SHORT_SPAN_LENGTH_SIZE
  shr $dataPtr, $dataPtr, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $dataPtr, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16step $dataBPtr, $mzero, $outDataB+=, 1
  shl $dataBPtr, $dataBPtr, SCALED_PTR64_SHL_BITS
#else
  ld32step $dataBPtr, $mzero, $outDataB+=, 1
#endif

  // process 2 at a time
  {
    shr $dataSizeD2, $origDataSize, 1
    setzi $a0, ZAACC_BITMASK
  }
  {
    brz $dataSizeD2, .Lscale_float_vector2_loop_end
    uput $FP_CLR, $a0
  }

  // Execute storage of final result value immediately after looping function
  // has completed
  call $lr, scaled_add_data_half_factor_float
  st32step $data0i0, $mzero, $dataPtr+=, $stride

.Lscale_float_vector2_loop_end:
  // Do we have a single element remaining to be done?
  and $dataSize, $origDataSize, 0x1
  brz $dataSize, .Lscale_float_end

  // There is one more element that needs to be stored, do a read/modify/write
  // so we do not trash anything else may be stored in the same word.
  //
  // Execute storage of the result value immediately after looping function
  // has completed
  call $lr, scaled_add_data_half_factor_float_scalar
  st32 $data1i0, $mzero, $dataPtr, 0

.Lscale_float_end:
  brnzdec $outDataSize, .Lscale_float_outer_loop
  exitz $mzero

.size VERTEX_2D_SCALE_FLOAT_COMMON, .-VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW

// Undefine 2D register definitions
#undef VERTEX_SCALE_OFFSET
#undef outData
#undef outDataSize
#undef outDataB
#undef dataSize
#undef origDataSize
#undef factor
#undef memConstraints

//******************************************************************************
// Supervisor case
//******************************************************************************

// Variable offsets (in bytes)
#ifdef VECTOR_AVAIL_SCALED_PTR64
#define VERTEX_PACKED_COUNT_OFFSET 2
#define VERTEX_SCALE_OFFSET 6
#define VERTEX_SCALE_OFFSET_FLOAT_CONST 8
#define VERTEX_TOLERANCE_OFFSET_SV 8
#else
#define VERTEX_PACKED_COUNT_OFFSET 4
#define VERTEX_SCALE_OFFSET 12
#define VERTEX_SCALE_OFFSET_FLOAT_CONST 12
#define VERTEX_TOLERANCE_OFFSET_SV 16
#endif


// Integer variables.
// Registers prefixed tmp_ indicate the same register being used but as a temp register - not
// with the meaning that the name implies.
#define vertexPtr m0
#define tmp_countD2 m1
#define final m2
#define remM1 m3
#define tmp_remM1 m3
#define factorPtr m4
#define tmp_log2AtomSize m7
#define tmp_atomSizeMask m8
#define workerIdM1 m8

// Float variables
#define factor a7

#define LC_SET 2
#define LC_MASK 1
#define SSR_REGISTER 1
#define CR_REGISTER 2


.globl VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW
.type VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW, @function
.globl VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_FAST
.type VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_FAST, @function

// ScaledSubtract Supervisor entry point. Just negates the scale value
// and lets the ScaledAdd Supervisor do the rest.
DEF_STACK_SIZE_OWN STACK_SIZE .text.VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW
.section .text.VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW
.align 4
.supervisor
VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW:
   setzi $sv_memConstraints, 0
   bri   1f
VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_FAST:
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16 $factorPtr, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl $factorPtr, $factorPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $factorPtr, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/4
#endif
  ld32 $sv_scale, $mzero, $factorPtr, 0
  or   $tmp_countD2, $mzero, NEGATE_FLOAT_BY_XOR
  xor  $sv_scale, $sv_scale, $tmp_countD2
 
  bri VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW_RUN_WORKERS
.size VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW, .-VERTEX_SV_SUBTRACT_SCALE_FLOAT_TENSOR_SLOW


.globl VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW
.type VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW, @function
.globl VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST
.type VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST, @function

DEF_STACK_SIZE_OWN  STACK_SIZE  .text.VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW
.section .text.VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW
.align 4
.supervisor
VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW:
   setzi $sv_memConstraints, 0
   bri   1f

VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST:
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16 $factorPtr, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl $factorPtr, $factorPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $factorPtr, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/4
#endif
  ld32 $sv_scale, $mzero, $factorPtr, 0

VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW_RUN_WORKERS:
  // Use a worker to test if the float scale is accurate enough to use when cast
  // to a half. Don't stress too much about pipe hits as we would only pad it with
  // code from after the `run` which runs for free while the worker runs anyhow.
  setzi $tmp_remM1, TMEM_REGION0_BASE_ADDR
  setzi $tmp_log2AtomSize, VERTEX_SV_SCALE_FLOAT_CHECK
  sub $tmp_remM1,$m0, $tmp_remM1
  // Save SSR register for later in case other vertices in this compute set
  // check/rely on it
  get $tmp_atomSizeMask, SSR_REGISTER // Pipe flush which would happen before run instruction anyhow

  // Call the worker which will check the accuracy of the scale
  run $tmp_log2AtomSize, $tmp_remM1, 0
  // Do work while the worker is working, we don't care much about pipe hits as
  // we should beat it. The 1-bit exit status of the FLOAT_CHECK worker thread
  // will be AND-ed to the LC bit in $SSR register, so we need to set the LC bit
  // to 1 (through CR register) to find the exit status.
  add $sp, $sp, -STACK_SIZE
  // Set the LC bit in SSR to 1 (through CR register)
  setzi $tmp_countD2, LC_SET
  put CR_REGISTER, $tmp_countD2


  // Based on the returned isHalfAccurate flag being zero, we could select the
  // path that casts everything to float or casts scale to half.
  // Load scale - the same in both cases
  ldz16  $sv_count, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  // Set up assuming that we'll do the all half version
  setzi $sv_log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $sv_atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1
  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX(half_half_float_continue)


  // Done all we can while the worker runs.  Now wait for it to complete
  sync TEXCH_SYNCZONE_LOCAL
  // Fetch and isolate the return value "isHalfAccurate"
  get $tmp_remM1, SSR_REGISTER // This get flushes the pipline
  and $tmp_remM1, $tmp_remM1, LC_MASK
  put SSR_REGISTER, $tmp_atomSizeMask // Restore SSR. This flushes the pipeline

  brnz $tmp_remM1, VERTEX(supervisor)  // 6 Cycles if branch taken, 1 otherwise

  setzi $sv_log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $sv_atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1
  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX_SV_SCALE_FLOAT_COMMON
  bri   VERTEX(supervisor) // 6 cycles
.size VERTEX_SV_SCALE_TENSOR, .-VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW

.globl VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW
.type VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW, @function
.globl VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST
.type VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST, @function

DEF_STACK_SIZE_OWN STACK_SIZE .text.VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW
.section .text.VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW
.align 4
.worker
VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW:
VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST:
  ld32  $sv_scale, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET_FLOAT_CONST/4

  add   $sp, $sp, -STACK_SIZE
  setzi $sv_log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $sv_atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX_SV_SCALE_FLOAT_COMMON_CONST
  ldz16  $sv_count, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  bri   VERTEX(supervisor) // 6 cycles
.size VERTEX_SV_SCALE_CONST, .-VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW

//******************************************************************************

.type VERTEX_SV_SCALE_FLOAT_COMMON, @function

DEF_STACK_SIZE_OWN  0  VERTEX_SV_SCALE_FLOAT_COMMON
.section .text.VERTEX_SV_SCALE_FLOAT_COMMON, FUNCTION_IS_WORKER
.align 4
// Worker entry point used first by the supervisor to check the accuracy of
// the float scale when cast to a half.
VERTEX_SV_SCALE_FLOAT_CHECK:
  // Just call the C function which will check if scale is accurate
  // enough as a half precision value
  mov $sp, $m12
  // passing scale and tolerance.
  // FP_CTL will be modified in the call but as we exit as a worker this won't matter
#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16 $factorPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $factorPtr, $factorPtr, SCALED_PTR64_SHL_BITS
#else
  ld32 $factorPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
#endif
  ld32  $C_CALL_SCALE, $mzero, $factorPtr, 0

  ld32  $C_CALL_PARAM_TOLERANCE, $mvertex_base, $mzero, VERTEX_TOLERANCE_OFFSET_SV/4
  call $lr, CHECK_ACCURACY_WHEN_CAST
  // Return value from C which will get and-ed with SSR bit0
  exitnz $C_CALL_RETURN


//******************************************************************************
// Normal worker code entry to do the scaledAdd
VERTEX_SV_SCALE_FLOAT_COMMON:

VERTEX_SV_SCALE_FLOAT_COMMON_CONST:
  // load vertex state
  {
    ld32 $dataSizeD2, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4

  ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET/4
  ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4

  ld32 $factor, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
  get $workerIdM1, $WSR

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $factor
  }

  // process 2 at a time first as this is the optimal scenario
  shr $dataSizeD2, $dataSizeD2, 1

  // if worker id is less than the remainder this worker can process an extra 1.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $dataSizeD2, $dataSizeD2, $mscratch

  // offset each worker's pointer into the data to interleave them.
  ld32step $azero, $mzero, $dataPtr+=, $workerIdM1
  ld32step $azero, $mzero, $dataBPtr+=, $workerIdM1

  // process 2 at a time
  brz $dataSizeD2, .Lhalf_half_float_loop_epilogue

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // Execute storage of final result value immediately after looping function
  // has completed
  call $mscratch, scaled_add_data_half_factor_float
  st32step $data0i0, $mzero, $dataPtr+=, $stride

.Lhalf_half_float_loop_epilogue:
  // at most one of our workers will have to do the remaining elements. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 1 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_half_float_epilogue
  brz $final, .Lhalf_half_float_epilogue

  // there is one more element that needs to be stored, do a read/modify/write
  // so we do not trash anything else may be stored in the same word.
  //
  // Execute storage of the result value immediately after looping function
  // has completed
  call $mscratch, scaled_add_data_half_factor_float_scalar
  st32 $data1i0, $mzero, $dataPtr, 0

.Lhalf_half_float_epilogue:
  exitz $mzero

.size VERTEX_SV_SCALE_FLOAT_COMMON, .-VERTEX_SV_SCALE_FLOAT_COMMON

// Undefine Supervisor register definitions
#undef vertexPtr
#undef final
#undef remM1
#undef factorPtr
#undef workerIdM1


#endif // __IPU__
