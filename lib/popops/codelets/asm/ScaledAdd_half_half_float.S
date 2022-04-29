// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "ScaledAddSupervisor.inc"
#include "CommonPoplibsMacros.h.S"
#include "workDivision.h.S"
#include "MathConstants.S"

#define VERTEX_2D_ADD_SCALE_FLOAT_FAST __runCodelet_popops__ScaledAdd2D___half_half_float_false_true
#define VERTEX_2D_ADD_SCALE_FLOAT_SLOW __runCodelet_popops__ScaledAdd2D___half_half_float_false_false
#define VERTEX_2D_SCALE_FLOAT_COMMON __ScaledAdd2D___half_half_float_common

#define VERTEX_SV_ADD_SCALE_FLOAT_FAST __runCodelet_popops__ScaledAddSupervisor___half_half_float_false_true
#define VERTEX_SV_ADD_SCALE_FLOAT_SLOW __runCodelet_popops__ScaledAddSupervisor___half_half_float_false_false
#define VERTEX_SV_SCALE_FLOAT_COMMON __ScaledAddSupervisor___half_half_float_common
#define VERTEX_SV_NEG_SCALE_FLOAT_COMMON __ScaledSubtractSupervisor___half_half_float_common

#define VERTEX_2D_SUBTRACT_SCALE_FLOAT_FAST __runCodelet_popops__ScaledSubtract2D___half_float_true
#define VERTEX_2D_SUBTRACT_SCALE_FLOAT_SLOW __runCodelet_popops__ScaledSubtract2D___half_float_false
#define VERTEX_SV_SUBTRACT_SCALE_FLOAT_FAST __runCodelet_popops__ScaledSubtractSupervisor___half_half_float_true
#define VERTEX_SV_SUBTRACT_SCALE_FLOAT_SLOW __runCodelet_popops__ScaledSubtractSupervisor___half_half_float_false

// The bulk of the Supervisor task processing which is common to all scaled-add
// variants has been implemented in a different file: ScaledAddSupervisor_fp.S.
// They use this define to produce labels
#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty


//******************************************************************************
// 2d Case vertex state definition
//******************************************************************************
// Variable offsets (in bytes)
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 4
#define VERTEX_DATA_B_OFFSET 8
#define VERTEX_SCALE_OFFSET 12
#define VERTEX_TOLERANCE_OFFSET 16

//******************************************************************************
// Common definitions and subroutines used by both the 2D and Supervisor cases
//******************************************************************************

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

#define SCALED_PTR128_SHL_BITS 4

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

// A C function to check the accuracy when casting.  Extracted from the elf file
#define CHECK_ACCURACY_WHEN_CAST _ZN6popops32checkAccuracyWhenCastComputeImplIfDhEEbT_f

// Registers used in calling the C function checkAccuracyWhenCast
#define C_CALL_PARAM_TOLERANCE a1
#define C_CALL_SCALE a0
#define C_CALL_RETURN m0

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

// Check accuracy when cast impelmented in assembler, used in the Supervisor
// version of the code
.macro CHECK_ACCURACY_WHEN_CAST_ASM
  // multiply scale by tolerance
  f32mul    $dataB1i0, $C_CALL_PARAM_TOLERANCE, $dataB0i0
  f32absadd $dataB1i0, $dataB1i0, $azero// abs value: max admissible errors

  // Compute the diff between the input and the input converted to half
  f32tof16  $dataB0i1, $dataB0i0
  f16tof32  $dataB0i1, $dataB0i1
  f32sub    $dataB0i1, $dataB0i1, $dataB0i0 // subtract original input
  f32absadd $dataB0i1, $dataB0i1, $azero   // get absolute value of diff

  // Results in TFPU_FP32_TRUE if difference is less than error
  f32cmplt  $dataB0i0, $dataB1i0, $dataB0i1
.endm



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

FN_SECTION scaled_add_data_half_factor_float 8
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
FN_SIZE scaled_add_data_half_factor_float

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

FN_SECTION scaled_add_data_half_factor_float_scalar
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
FN_SIZE scaled_add_data_half_factor_float_scalar
//******************************************************************************
// 2D case
//******************************************************************************

// Integer variables
#define outData m0
#define outDataB m2
#define dataSize m4
#define origDataSize m3

// Float variables

#define factor a7
#define factorTmp a6

// Shared with the all half version - be careful!
//#define memConstraints m11 //defined in shared file


// Common code used by both the plus/subtract workers. The optional argument
// is intended to allow the scale factor argument to be negated by the subtract
// worker.
.macro PERFORM_VERTEX_2D_SCALE_FLOAT OPTIONAL_SCALE_FACTOR_MODIFIER:vararg
  ld32  $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $factor, $mzero, $dataPtr, 0
  \OPTIONAL_SCALE_FACTOR_MODIFIER

  { brz   $C_CALL_RETURN, VERTEX_2D_SCALE_FLOAT_COMMON
    f32tof16 $factorTmp, $factor}
  // Otherwise cast, broadcast and branch into the "all half" version
  { bri   __ScaledAdd2D___half_common
    roll16 $factor, $factorTmp, $factorTmp}
.endm

DEF_STACK_SIZE_OWN 0 .text.VERTEX_2D_ADD_SCALE_FLOAT_SLOW
FN_WORKER_ENTRY_POINT VERTEX_2D_ADD_SCALE_FLOAT_SLOW 4 "" NONE
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D
  // For use when we enter the half, half, half version
  setzi $memConstraints, 0
  bri   1f
FN_EXPORT VERTEX_2D_ADD_SCALE_FLOAT_FAST
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D

  setzi $memConstraints, 1
1:
  PERFORM_VERTEX_2D_SCALE_FLOAT

FN_SIZE VERTEX_2D_ADD_SCALE_FLOAT_SLOW


DEF_STACK_SIZE_OWN 0 .text.VERTEX_2D_SUBTRACT_SCALE_FLOAT_SLOW
FN_WORKER_ENTRY_POINT VERTEX_2D_SUBTRACT_SCALE_FLOAT_SLOW 4 "" NONE
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D
  // For use when we enter the half, half, half version
  setzi $memConstraints, 0
  bri   1f

FN_EXPORT VERTEX_2D_SUBTRACT_SCALE_FLOAT_FAST
  // Do this first as we don't need to worry as much about the calling convention
  CALL_CHECK_ACCURACY_WHEN_CAST_2D

  setzi $memConstraints, 1
1:
  PERFORM_VERTEX_2D_SCALE_FLOAT f32sub $factor, $azero, $factor

FN_SIZE VERTEX_2D_SUBTRACT_SCALE_FLOAT_SLOW


#define outDataSize m11

// The fastest implementation happens to use more Aux instructions than Main
// instructions. Since memory access instructions use the Main path, the
// efficiency of these instructions in a memory-constrained scenario do not lead
// to a speed up of the loop kernel. Hence, this variant of ScaledAdd has only
// a single implementation regardless of the placement of the inputs in memory.
FN_SECTION VERTEX_2D_SCALE_FLOAT_COMMON
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

  ld32step $dataBPtr, $mzero, $outDataB+=, 1

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

FN_SIZE VERTEX_2D_SCALE_FLOAT_COMMON

// Undefine 2D register definitions
#undef VERTEX_SCALE_OFFSET
#undef outData
#undef outDataSize
#undef outDataB
#undef dataSize
#undef origDataSize
#undef factor

//******************************************************************************
// Supervisor case
//******************************************************************************
#undef VERTEX_DATA_A_OFFSET
#undef VERTEX_DATA_B_OFFSET

// Variable offsets (in bytes)
// Data pointers are 32 bits
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_B_OFFSET 4
#define VERTEX_SCALE_OFFSET 8
#define VERTEX_PACKED_COUNT_OFFSET 10
#define VERTEX_TOLERANCE_OFFSET_SV 12


// Integer variables.
// Registers prefixed tmp_ indicate the same register being used but as a temp register - not
// with the meaning that the name implies.

//#define memConstaints m0 // define/ shared between files
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

// ScaledSubtract Supervisor entry point. Just negates the scale value
// and lets the ScaledAdd Supervisor do the rest.
DEF_STACK_SIZE_OWN 0 .text.VERTEX_SV_SUBTRACT_SCALE_FLOAT_SLOW
FN_SUPERVISOR_ENTRY_POINT VERTEX_SV_SUBTRACT_SCALE_FLOAT_SLOW 4 "" NONE
  setzi $sv_workerFunction, VERTEX_SV_NEG_SCALE_FLOAT_COMMON
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE VERTEX_SV_SUBTRACT_SCALE_FLOAT_SLOW


DEF_STACK_SIZE_OWN 0 .text.VERTEX_SV_SUBTRACT_SCALE_FLOAT_FAST
FN_SUPERVISOR_ENTRY_POINT VERTEX_SV_SUBTRACT_SCALE_FLOAT_FAST 4 "" NONE
  setzi $sv_workerFunction, VERTEX_SV_NEG_SCALE_FLOAT_COMMON_FAST
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE VERTEX_SV_SUBTRACT_SCALE_FLOAT_FAST


DEF_STACK_SIZE_OWN 0 .text.VERTEX_SV_ADD_SCALE_FLOAT_SLOW
FN_SUPERVISOR_ENTRY_POINT VERTEX_SV_ADD_SCALE_FLOAT_SLOW 4 "" NONE
  setzi $sv_workerFunction, VERTEX_SV_SCALE_FLOAT_COMMON
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE VERTEX_SV_ADD_SCALE_FLOAT_SLOW

DEF_STACK_SIZE_OWN 0 .text.VERTEX_SV_ADD_SCALE_FLOAT_FAST
FN_SUPERVISOR_ENTRY_POINT VERTEX_SV_ADD_SCALE_FLOAT_FAST 4 "" NONE
  setzi $sv_workerFunction, VERTEX_SV_SCALE_FLOAT_COMMON_FAST
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles

FN_SIZE VERTEX_SV_ADD_SCALE_FLOAT_FAST

//******************************************************************************

FN_WORKER_ENTRY_POINT VERTEX_SCALE_FLOAT_CHECK 4

//******************************************************************************
// Normal worker code entry to do the scaledAdd
VERTEX_SV_SCALE_FLOAT_COMMON_FAST:
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
  { bri 1f
    or    $factor,$azero,FLOAT_1_0}

VERTEX_SV_NEG_SCALE_FLOAT_COMMON_FAST:
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
  { bri 1f
    or    $factor,$azero,FLOAT_NEG_1_0}

VERTEX_SV_SCALE_FLOAT_COMMON:
  setzi $memConstraints, 0
  { bri 1f
    or    $factor,$azero,FLOAT_1_0}

VERTEX_SV_NEG_SCALE_FLOAT_COMMON:
  {setzi $memConstraints, 0
  or    $factor,$azero,FLOAT_NEG_1_0}
1:
  // Based on the accuracy given when using scale cast to half -
  // branch and do that instead.  The $memConstraints parameter will be
  // valid in that case

  ldz16  $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl    $dataPtr, $dataPtr, SCALED_PTR128_SHL_BITS
  // Store FP_CTL while we check scale accuracy
  {ld32   $dataB0i0, $mzero, $dataPtr, 0
   uget   $data1i0, CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK }

  f32mul $factor, $factor, $dataB0i0
  {ld32  $C_CALL_PARAM_TOLERANCE, $mvertex_base, $mzero, VERTEX_TOLERANCE_OFFSET_SV/4
   uput   $FP_CTL, $azero}   // disable FP exceptions for macro

  // Check the accuracy of the scale when cast to half to see if we
  // can run that path
  CHECK_ACCURACY_WHEN_CAST_ASM

  {mov  $C_CALL_RETURN, $dataB0i0 // transfer to MRF register
   uput CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK, $data1i0}

   // Prepare scale for the faster, half scale version
   // held in its `k` register
   f32tof16 $k, $factor
  {get $workerIdM1, $WSR
   or  $data0i0, $azero, HALF_1_0<<16}

   {brz        $C_CALL_RETURN, half_half_half_scales_in_k_continue
    // Pass $k for the f16v4axpby instructions in the half_half_half path.
    sort4x16hi $k, $data0i0, $k}

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $factor
  }

  {
    ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $dataSizeD2, LOG2_FLOAT_ATOM_SIZE

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
  and      $mscratch, $remM1, 1
  brz      $mscratch, .Lhalf_half_float_epilogue
  andc     $remM1, $remM1, 1
  ld32     $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ldb16step $azero, $mzero, $mscratch+=, $remM1
  cmpeq    $mscratch, $dataPtr, $mscratch
  // Use the worker that is pointing at the final element to process it
  brz $mscratch, .Lhalf_half_float_epilogue

  // there is one more element that needs to be stored, do a read/modify/write
  // so we do not trash anything else may be stored in the same word.
  //
  // Execute storage of the result value immediately after looping function
  // has completed
  call $mscratch, scaled_add_data_half_factor_float_scalar
  st32 $data1i0, $mzero, $dataPtr, 0

.Lhalf_half_float_epilogue:
  exitz $mzero

FN_SIZE VERTEX_SCALE_FLOAT_CHECK

// Undefine Supervisor register definitions
#undef vertexPtr
#undef final
#undef remM1
#undef factorPtr
#undef workerIdM1


#endif // __IPU__
