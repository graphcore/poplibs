#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

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

// The bulk of the Supervisor task processing which is common to all scaled-add
// variants has been implemented in a different file: ScaledAddSupervisor_fp.S.
#define VERTEX_SV_SUPERVISOR __runCodelet_popops__ScaledAddSupervisor___supervisor


//******************************************************************************
// Common definitions and subroutines used by both the 2D and Supervisor cases
//******************************************************************************

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define SCALED_PTR64_SHL_BITS 3
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12

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
.section .text.scaled_add_data_half_factor_float
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
#define VERTEX_USE_HALF_SCALE_OFFSET 16

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
#define VERTEX_COMMON __ScaledAdd2D___half_common


.globl VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW
.type VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW, @function
.globl VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST
.type VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST, @function

.section .text.VERTEX_2D_SCALE_FLOAT
.align 4
VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW:
  // For use when we enter the half, half, half version
  setzi $memConstraints, 0
  bri   1f
VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_FAST:
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $factor, $mzero, $dataPtr, 0

  // Can we use half scale? If not then continue into this function for float scale
  ld32  $dataPtr, $mvertex_base, $mzero, VERTEX_USE_HALF_SCALE_OFFSET/4
  ldz8  $dataPtr, $mzero, $dataPtr, 0
  { brz   $dataPtr, VERTEX_2D_SCALE_FLOAT_COMMON
    f32tof16 $factorTmp, $factor}
  // Otherwise cast, broadcast and branch into the "all half" version
  { bri   __ScaledAdd2D___half_common
    roll16 $factor, $factorTmp, $factorTmp}
.size VERTEX_2D_SCALE_FLOAT, .-VERTEX_2D_ADD_SCALE_FLOAT_TENSOR_SLOW


.globl VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW
.type VERTEX_2D_ADD_SCALE_FLOAT_CONST_SLOW, @function
.globl VERTEX_2D_ADD_SCALE_FLOAT_CONST_FAST
.type VERTEX_2D_ADD_SCALE_FLOAT_CONST_FAST, @function

.section .text.VERTEX_2D_SCALE_FLOAT_COMMON
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
  // load inner pointers. A is a SHORT_SPAN, B is a SCALED_PTR64
  ld32step $dataPtr, $mzero, $outData+=, 1
  {
    shr $origDataSize, $dataPtr, SHORT_SPAN_PTR_SIZE
    setzi $a0, ZAACC_BITMASK
  }
  {
    ldz16step $dataBPtr, $mzero, $outDataB+=, 1
    uput $FP_CLR, $a0
  }
  shl $dataPtr, $dataPtr, SHORT_SPAN_LENGTH_SIZE
  shr $dataPtr, $dataPtr, SHORT_SPAN_LENGTH_SIZE

  shl $dataBPtr, $dataBPtr, SCALED_PTR64_SHL_BITS

  // process 2 at a time
  shr $dataSizeD2, $origDataSize, 1
  brz $dataSizeD2, .Lscale_float_vector2_loop_end

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

//******************************************************************************
// Supervisor case
//******************************************************************************

// Variable offsets (in bytes)
#define VERTEX_PACKED_COUNT_OFFSET 2
#define VERTEX_SCALE_OFFSET_FLOAT_CONST 8
#define VERTEX_SCALE_OFFSET 6
#define VERTEX_USE_HALF_SCALE_OFFSET_SV 8

// create a new vertex state on the supervisor stack that has the input values
// preprocessed for all of the workers to use.
#define SV_STATE_DATA_OFFSET 0
#define SV_STATE_COUNT_OFFSET 4
#define SV_STATE_REMM1_OFFSET 8
#define SV_STATE_FINAL_OFFSET 12
#define SV_STATE_SCALES_OFFSET 16
#define SV_STATE_DATA_B_OFFSET 20
#define SV_STATE_MEM_CONSTRAINTS 24

#define SV_STATE_SIZE 28

// total space required on the stack
#define STACK_SIZE (SV_STATE_SIZE)

// to avoid sub-word writes we must make sure that each worker processes
// a number of elements so that we fall exactly into a 64-bit load. for floats
// this is 8/sizeof(float) = 2 and 8/sizeof(half) = 4
#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2

// Integer variables
#define vertexPtr m0
#define countD2 m1
#define final m2
#define remM1 m3
#define factorPtr m4
#define mworkerFunction m6
#define log2AtomSize m7
#define atomSizeMask m8
#define workerIdM1 m8

// Float variables
#define factor a7

#define memConstraintsFloatScale m2
// Flags for memConstraints and floatScale
#define MEM_CONSTRAINTS_MASK 0x1
#define FLOAT_SCALE_MASK 0x2


.globl VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW
.type VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW, @function
.globl VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST
.type VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST, @function

.section .text.VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW
.align 4
VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW:
   ld32 $remM1, $vertexPtr, $mzero, VERTEX_USE_HALF_SCALE_OFFSET_SV/4
   setzi $memConstraintsFloatScale, FLOAT_SCALE_MASK
   bri   1f

VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_FAST:
  ld32 $remM1, $vertexPtr, $mzero, VERTEX_USE_HALF_SCALE_OFFSET_SV/4
  setzi $memConstraintsFloatScale, FLOAT_SCALE_MASK | MEM_CONSTRAINTS_MASK
1:
  ldz16 $factorPtr, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $factorPtr, $factorPtr, SCALED_PTR64_SHL_BITS

  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  // Load a flag to indicate if scale is ok to use as a half
  // and if so, branch into the half, half, half version of this vertex.
  // The memConstraintsFloatScale flag will propogate through to the
  // worker eventually where it will get used to cause a cast
  ldz8 $remM1, $mzero, $remM1, 0
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX_SV_SCALE_FLOAT_COMMON
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ld32  $factorPtr, $mzero, $factorPtr, 0
  nop //Avoid pipe hit for remM1
  brnz $remM1, __runCodelet_popops__ScaledAddSupervisor___half_half_float_continue
  bri   VERTEX_SV_SUPERVISOR // 6 cycles
.size VERTEX_SV_SCALE_TENSOR, .-VERTEX_SV_ADD_SCALE_FLOAT_TENSOR_SLOW


.globl VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW
.type VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW, @function
.globl VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST
.type VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST, @function

.section .text.VERTEX_SV_SCALE_CONST
.align 4
VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW:
VERTEX_SV_ADD_SCALE_FLOAT_CONST_FAST:
  ld32  $factorPtr, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET_FLOAT_CONST/4

  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX_SV_SCALE_FLOAT_COMMON
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  bri   VERTEX_SV_SUPERVISOR // 6 cycles
.size VERTEX_SV_SCALE_CONST, .-VERTEX_SV_ADD_SCALE_FLOAT_CONST_SLOW


.type VERTEX_SV_SCALE_FLOAT_COMMON, @function

.section .text.VERTEX_SV_SCALE_FLOAT_COMMON
.align 4
VERTEX_SV_SCALE_FLOAT_COMMON:

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

  // if worker id is less than the remainder this worker can process an extra 2.
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
#undef countD2
#undef final
#undef remM1
#undef factorPtr
#undef mworkerFunction
#undef log2AtomSize
#undef atomSizeMask
#undef workerIdM1


#endif // __IPU__
