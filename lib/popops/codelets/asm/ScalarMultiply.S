// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include <poplar/TileConstants.hpp>
#include <poplar/StackSizeDefs.hpp>
#include "workDivision.h.S"


// Mangled codelet names.
#define SCALAR_MULTIPLY_1D_HALF_FLOAT         __runCodelet_popops__ScalarMultiply1D___half_float
#define SCALAR_MULTIPLY_1D_INPLACE_HALF_FLOAT __runCodelet_popops__ScalarMultiply1DInplace___half_float
#define SCALAR_MULTIPLY_2D_HALF_FLOAT         __runCodelet_popops__ScalarMultiply2D___half_float
#define SCALAR_MULTIPLY_2D_INPLACE_HALF_FLOAT __runCodelet_popops__ScalarMultiply2DInplace___half_float

// A C function to check the accuracy when casting. Extracted from the elf file.
#define CHECK_ACCURACY_WHEN_CAST _ZN6popops32checkAccuracyWhenCastComputeImplIfDhEEbT_f

// Vertex state offsets in bytes.
#define VOFF_IN1_ADDR  0  // Pointer to `in1`.
#define VOFF_SIZE      4  // Size of `in1`.
#define VOFF_IN2_ADDR  8  // Pointer to `in2`.
#define VOFF_TOL       12 // FP32 to FP16 conversion accuracy tolerance.
#define VOFF_OUT_ADDR  16 // Pointer to `out`.

// Worker registers.
#define MRF_ACC_CHECK_RES   m0   // The return value of `checkAccuracyWhenCastComputeImpl()`.
#define MRF_IN1             m1
#define MRF_IN2             m2
#define MRF_LINK            m2   // Link register used for calls to common code.
#define MRF_OUT             m7
#define MRF_SIZE            m4   // Number of elements in the input array.
#define MRF_OUT_2D          m5   // Stores a pointer to the jagged vector of output vectors.
#define MRF_MEM_STEP        m6   // The number of 4-bytes to step through in the scalar multiply loop.
#define MRF_NITER           m3   // Number of loop iterations for the current worker.
#define MRF_N_REMAINING     m8   // Number of remaining values. Equal to `size(in1) % 4`.
#define MRF_LAST_CHUNK      m9   // Pointer to last chunk of 8-byte aligned data in input array.
#define MRF_WKR_ID          m10  // In the 1D case, `$m10` stores the worker ID.
#define MRF_IN1_2D          m10  // In the 2D case, `$m10` stores a pointer to the jagged vector of input vectors.
#define MRF_SCRATCH         m11
#define ARF_FP32_SCALAR     a0
#define ARF_FP16B_SCALAR    a0
#define ARF_TOLERANCE       a1   // FP32 to FP16 conversion accuracy tolerance for the scalar.
#define ARF_SCRATCH         a1
#define ARF_FP16_V4VALS     a2:3
#define ARF_FP16_V4VALS_LO  a2
#define ARF_FP16_V4VALS_HI  a3
#define ARF_FP16_V4VALS_RES a4:5 // The loops store results here to allow bundling.
#define ARF_FP32_VAL        a4
#define ARF_FP32_V4VALS     a4:7
#define ARF_FP32_V4VALS_LO  a4:5
#define ARF_FP32_V4VALS_HI  a6:7


// Definition of slow scalar multiply. This implementation casts the unscaled
// array to fp32, performs all operations in fp32 and casts the result back to
// fp16.
// -----------------------------------------------------------------------------
DEF_STACK_USAGE 0 scalar_multiply_slow
.section .text.scalar_multiply_slow
.type scalar_multiply_slow, @function
.align 8
scalar_multiply_slow:
  // Jump to `.Lafter_scalar_multiply_loop_slow` if there are no loop iterations
  // to be done.
  brz $MRF_NITER, .Lafter_scalar_multiply_loop_slow

  // Partially unroll the loop and bundle operations.
  ld64step   $ARF_FP16_V4VALS, $mzero, $MRF_IN1+=, $MRF_MEM_STEP
  {
    // Subtract 1 because of partial loop unrolling.
    sub $MRF_NITER, $MRF_NITER, 1
    f16v2tof32 $ARF_FP32_V4VALS_LO, $ARF_FP16_V4VALS_LO
  }

  // Run the scalar multiply loop for the partition of the input/output that
  // belongs to the current worker.
  {
    rpt        $MRF_NITER, (2f - 1f) / 8 - 1
    f16v2tof32 $ARF_FP32_V4VALS_HI, $ARF_FP16_V4VALS_HI
  }
1: // scalar_multiply_loop_begin
  {
    ld64step $ARF_FP16_V4VALS, $mzero, $MRF_IN1+=, $MRF_MEM_STEP
    f32v2mul $ARF_FP32_V4VALS_LO, $ARF_FP32_SCALAR:B, $ARF_FP32_V4VALS_LO
  }
  {
    nop
    f32v2mul $ARF_FP32_V4VALS_HI, $ARF_FP32_SCALAR:B, $ARF_FP32_V4VALS_HI
  }
  {
    nop
    f32v4tof16 $ARF_FP16_V4VALS_RES, $ARF_FP32_V4VALS
  }
  {
    st64step   $ARF_FP16_V4VALS_RES, $mzero, $MRF_OUT+=, $MRF_MEM_STEP
    f16v2tof32 $ARF_FP32_V4VALS_LO, $ARF_FP16_V4VALS_LO
  }
  {
    nop
    f16v2tof32 $ARF_FP32_V4VALS_HI, $ARF_FP16_V4VALS_HI
  }
2: // scalar_multiply_loop_end

  // Partially unroll the loop.
  f32v2mul   $ARF_FP32_V4VALS_LO, $ARF_FP32_SCALAR:B, $ARF_FP32_V4VALS_LO
  f32v2mul   $ARF_FP32_V4VALS_HI, $ARF_FP32_SCALAR:B, $ARF_FP32_V4VALS_HI
  f32v4tof16 $ARF_FP16_V4VALS_RES, $ARF_FP32_V4VALS
  st64step   $ARF_FP16_V4VALS_RES, $mzero, $MRF_OUT+=, $MRF_MEM_STEP

.Lafter_scalar_multiply_loop_slow:

  // End early if there's no remainder.
  brz $MRF_N_REMAINING, .Lend_slow

  // Conditionally handle the case where the last 8-byte aligned chunk of data
  // contains strictly less than 4 elements.
  // Exit if this worker doesn't handle the special case.
  // NOTE: In the supervisor case this means only 1 worker will consider the
  //   remainder. In the 2D case the worker should always point to the remainder
  //   (or where the remainder would start if there is none).
  cmpeq $MRF_SCRATCH, $MRF_IN1, $MRF_LAST_CHUNK
  brz   $MRF_SCRATCH, .Lend_slow

  // Else handle the special case.
  // If there's only one element, skip to `.Lhandle_last_fp16_slow`. Otherwise
  // handle the pair of fp16 values in `.Lhandle_pair_fp16_slow`.
  and   $MRF_SCRATCH, $MRF_N_REMAINING, 0x2
  brz   $MRF_SCRATCH, .Lhandle_last_fp16_slow

.Lhandle_pair_fp16_slow:
  ld32step   $ARF_FP16_V4VALS_LO, $mzero, $MRF_IN1+=, 1
  f16v2tof32 $ARF_FP32_V4VALS_LO, $ARF_FP16_V4VALS_LO
  f32v2mul   $ARF_FP32_V4VALS_LO, $ARF_FP32_SCALAR:B, $ARF_FP32_V4VALS_LO
  f32v2tof16 $ARF_FP16_V4VALS_LO, $ARF_FP32_V4VALS_LO
  st32step   $ARF_FP16_V4VALS_LO, $mzero, $MRF_OUT+=, 1

.Lhandle_last_fp16_slow:
  // If the remainder was 2, then skip to `.Lend_slow`. Otherwise handle the
  // last fp16 value.
  and $MRF_SCRATCH, $MRF_N_REMAINING, 0x1
  brz $MRF_SCRATCH, .Lend_slow

  ldb16      $ARF_FP16_V4VALS_LO, $mzero, $MRF_IN1, 0
  {
    ldb16      $ARF_SCRATCH, $mzero, $MRF_OUT, 1
    f16tof32   $ARF_FP32_VAL, $ARF_FP16_V4VALS_LO
  }
  f32mul     $ARF_FP32_VAL, $ARF_FP32_VAL, $ARF_FP32_SCALAR
  f32tof16   $ARF_FP16_V4VALS_LO, $ARF_FP32_VAL
  sort4x16hi $ARF_FP16_V4VALS_LO, $ARF_FP16_V4VALS_LO, $ARF_SCRATCH
  st32       $ARF_FP16_V4VALS_LO, $mzero, $MRF_OUT, 0

.Lend_slow:
  br $MRF_LINK
.size scalar_multiply_slow, .-scalar_multiply_slow
// -----------------------------------------------------------------------------


// Definition of fast scalar multiply. This implementation casts the scalar to
// fp16, and performs all operations in fp16.
// -----------------------------------------------------------------------------
DEF_STACK_USAGE 0 scalar_multiply_fast
.section .text.scalar_multiply_fast
.type scalar_multiply_fast, @function
.align 8
scalar_multiply_fast:
  // Jump to `.Lafter_scalar_multiply_loop_fast` if there are no loop iterations
  // to be done.
  brz $MRF_NITER, .Lafter_scalar_multiply_loop_fast

  // Subtract 1 because of partial loop unrolling.
  sub $MRF_NITER, $MRF_NITER, 1

  // Partially unroll the loop.
  ld64step $ARF_FP16_V4VALS, $mzero, $MRF_IN1+=, $MRF_MEM_STEP

  // Run the scalar multiply loop for the partition of the input/output that
  // belongs to the current worker.
  rpt $MRF_NITER, (2f - 1f) / 8 - 1
1: // scalar_multiply_loop_begin
  {
    ld64step $ARF_FP16_V4VALS, $mzero, $MRF_IN1+=, $MRF_MEM_STEP
    f16v4mul $ARF_FP16_V4VALS_RES, $ARF_FP16B_SCALAR:BL, $ARF_FP16_V4VALS
  }
  {
    st64step $ARF_FP16_V4VALS_RES, $mzero, $MRF_OUT+=, $MRF_MEM_STEP
    fnop
  }
2: // scalar_multiply_loop_end

  // Partially unroll the loop.
  f16v4mul $ARF_FP16_V4VALS_RES, $ARF_FP16B_SCALAR:BL, $ARF_FP16_V4VALS
  st64step $ARF_FP16_V4VALS_RES, $mzero, $MRF_OUT+=, $MRF_MEM_STEP

.Lafter_scalar_multiply_loop_fast:

  // End early if there's no remainder.
  brz $MRF_N_REMAINING, .Lend_fast

  // Conditionally handle the case where the last 8-byte aligned chunk of data
  // contains strictly less than 4 elements.
  // Exit if this worker doesn't handle the special case.
  // NOTE: In the supervisor case this means only 1 worker will consider the
  //   remainder. In the 2D case the worker should always point to the remainder
  //   (or where the remainder would start if there is none).
  cmpeq $MRF_SCRATCH, $MRF_IN1, $MRF_LAST_CHUNK
  brz   $MRF_SCRATCH, .Lend_fast

  // Else handle the special case.
  // If there's only one element, skip to `.Lhandle_last_fp16_fast`. Otherwise
  // handle the pair of fp16 values in `.Lhandle_pair_fp16_fast`.
  and   $MRF_SCRATCH, $MRF_N_REMAINING, 0x2
  brz   $MRF_SCRATCH, .Lhandle_last_fp16_fast

.Lhandle_pair_fp16_fast:
  ld32step   $ARF_FP16_V4VALS_LO, $mzero, $MRF_IN1+=, 1
  f16v2mul   $ARF_FP16_V4VALS_LO, $ARF_FP16B_SCALAR, $ARF_FP16_V4VALS_LO
  st32step   $ARF_FP16_V4VALS_LO, $mzero, $MRF_OUT+=, 1

.Lhandle_last_fp16_fast:
  // If the remainder was 2, then skip to `.Lend_fast`. Otherwise handle the
  // last fp16 value.
  and $MRF_SCRATCH, $MRF_N_REMAINING, 0x1
  brz $MRF_SCRATCH, .Lend_fast

  ldb16 $ARF_FP16_V4VALS_LO, $mzero, $MRF_IN1, 0
  {
    ldb16    $ARF_SCRATCH, $mzero, $MRF_OUT, 1
    f16v2mul $ARF_FP16_V4VALS_LO, $ARF_FP16B_SCALAR, $ARF_FP16_V4VALS_LO
  }
  sort4x16hi $ARF_FP16_V4VALS_LO, $ARF_FP16_V4VALS_LO, $ARF_SCRATCH
  st32 $ARF_FP16_V4VALS_LO, $mzero, $MRF_OUT, 0

.Lend_fast:
  br $MRF_LINK
.size scalar_multiply_fast, .-scalar_multiply_fast
// -----------------------------------------------------------------------------


// Definition of a 1D scalar multiply worker
// -----------------------------------------------------------------------------
.macro CREATE_SCALAR_MULTIPLY_1D_WORKER NAME NAME_INPLACE NAME_COMMON
.global \NAME
.type \NAME, @function
DEF_STACK_SIZE_OWN 0 \NAME

.global \NAME_INPLACE
.type \NAME_INPLACE, @function
DEF_STACK_SIZE_OWN 0 \NAME_INPLACE

.type \NAME_COMMON, @function
DEF_STACK_SIZE_OWN 0 \NAME_COMMON

.section .text.\NAME, "ax"
.align 4
\NAME:
  .worker
  // NOTE: $MRF_OUT should be a calle save register, because of a function call
  // later in the program.
  ld32 $MRF_OUT, $mvertex_base, $mzero, VOFF_OUT_ADDR / 4
  bri \NAME_COMMON
.size \NAME, .-\NAME

.section .text.\NAME_INPLACE, "ax"
.align 4
\NAME_INPLACE:
  .worker
  // NOTE: $MRF_OUT should be a calle save register, because of a function call
  // later in the program.
  ld32 $MRF_OUT, $mvertex_base, $mzero, VOFF_IN1_ADDR / 4
  bri \NAME_COMMON
.size \NAME_INPLACE, .-\NAME_INPLACE

.section .text.\NAME_COMMON, "ax"
.align 4
\NAME_COMMON:
  .worker

  // Load the value of the scalar and the tolerance.
  ld32 $MRF_IN2, $mvertex_base, $mzero, VOFF_IN2_ADDR / 4
  ld32 $ARF_FP32_SCALAR, $mzero, $MRF_IN2, 0
  ld32 $ARF_TOLERANCE, $mvertex_base, $mzero, VOFF_TOL / 4

  // Check whether converting the scalar to fp16 results in tolerable accuracy.
  // This is achieved by calling the checkAccuracyWhenCastComputeImpl() C++
  // function. This function expects the scalar to be in `$a0` and the tolerance
  // in `$a1`. The result is written in `$m0`.
  // NOTE: Do this first as we don't need to worry as much about the calling
  //   convention.
  {
    mov  $sp, $m12
    // Store the FP ctl.
    uget $a7, CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK
  }
  call $lr, CHECK_ACCURACY_WHEN_CAST

  // Load the vertex state in registers.
  // NOTE: The scalar value is reloaded after the function call.
  {
    ld32 $MRF_IN1, $mvertex_base, $mzero, VOFF_IN1_ADDR / 4
    // Restore the FP ctl.
    uput CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK, $a7
  }
  ld32 $MRF_IN2, $mvertex_base, $mzero, VOFF_IN2_ADDR / 4
  ld32 $MRF_SIZE, $mvertex_base, $mzero, VOFF_SIZE / 4
  ld32 $ARF_FP32_SCALAR, $mzero, $MRF_IN2, 0

  // Store a pointer to the beginning of the last 8-byte aligned "chunk" of data
  // in the array. This chunk may contain 1, 2, or 3 values that need to be
  // processed separately. If a worker points to this chunk after the loop, it
  // should process these remaining values.
  and       $MRF_N_REMAINING, $MRF_SIZE, 0x3
  sub       $MRF_SCRATCH, $MRF_SIZE, $MRF_N_REMAINING
  mov       $MRF_LAST_CHUNK, $MRF_IN1
  ldz16step $mzero, $mzero, $MRF_LAST_CHUNK+=, $MRF_SCRATCH

  // Get the worker ID.
  get $MRF_WKR_ID, $WSR
  and $MRF_WKR_ID, $MRF_WKR_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Step to the section that will be processed by the current worker.
  ld64step $ARF_FP16_V4VALS, $mzero, $MRF_IN1+=, $MRF_WKR_ID
  ld64step $ARF_FP16_V4VALS, $mzero, $MRF_OUT+=, $MRF_WKR_ID

  // Set the number of 4-bytes to step through in the scalar multiply loop.
  setzi $MRF_MEM_STEP, 6

  // Calculate the number of loop iterations for the current worker.
  DIVIDE_BY_WORKER $MRF_SIZE $MRF_WKR_ID $MRF_SCRATCH $MRF_NITER LOG2_HALF_ATOM_SIZE

  // Do operations in fp16 or fp32 depending on the result of the
  // checkAccuracyWhenCastComputeImpl() call.
  brnz $MRF_ACC_CHECK_RES, .Lscalar_multiply_fast\@

.Lscalar_multiply_slow\@:
  call $MRF_LINK, scalar_multiply_slow
  exitz $mzero
.Lscalar_multiply_fast\@:
  // Cast and broadcast the scalar to fp16.
  {
    call $MRF_LINK, scalar_multiply_fast
    f32tof16 $ARF_FP16B_SCALAR, $ARF_FP32_SCALAR
  }
  exitz $mzero

.size \NAME_COMMON, .-\NAME_COMMON
.endm
// -----------------------------------------------------------------------------


// Definition of a 2D scalar multiply worker
// -----------------------------------------------------------------------------
.macro CREATE_SCALAR_MULTIPLY_2D_WORKER NAME INPLACE
.global \NAME
.type \NAME, @function
DEF_STACK_SIZE_OWN 0 \NAME

.section .text.\NAME, "ax"
.align 4
\NAME:
  .worker

  // Load the value of the scalar and the tolerance.
  ld32 $MRF_IN2, $mvertex_base, $mzero, VOFF_IN2_ADDR / 4
  ld32 $ARF_FP32_SCALAR, $mzero, $MRF_IN2, 0
  ld32 $ARF_TOLERANCE, $mvertex_base, $mzero, VOFF_TOL / 4

  // Check whether converting the scalar to fp16 results in tolerable accuracy.
  // This is achieved by calling the checkAccuracyWhenCastComputeImpl() C++
  // function. This function expects the scalar to be in `$a0` and the tolerance
  // in `$a1`. The result is written in `$m0`.
  // NOTE: Do this first as we don't need to worry as much about the calling
  //   convention.
  {
    mov  $sp, $m12
    // Store the FP ctl.
    uget $a7, CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK
  }
  call $lr, CHECK_ACCURACY_WHEN_CAST


  // Load the vertex state in registers.
  // NOTE: The scalar value is reloaded after the function call.
  {
    ld32 $MRF_IN1_2D, $mvertex_base, $mzero, VOFF_IN1_ADDR / 4
    // Restore the FP ctl.
    uput CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK, $a7
  }
  ld32 $MRF_IN2, $mvertex_base, $mzero, VOFF_IN2_ADDR / 4
  ld32 $MRF_SIZE, $mvertex_base, $mzero, VOFF_SIZE / 4
.ifc \INPLACE, 0
  ld32 $MRF_OUT_2D, $mvertex_base, $mzero, VOFF_OUT_ADDR / 4
.endif
  ld32 $ARF_FP32_SCALAR, $mzero, $MRF_IN2, 0

  // Set the number of 4-bytes to step through in the scalar multiply loop.
  setzi $MRF_MEM_STEP, 1

  // Conditionally cast and broadcast the scalar to fp16.
  brz $MRF_ACC_CHECK_RES, .Lafter_scalar_cast\@
  f32tof16 $ARF_FP16B_SCALAR, $ARF_FP32_SCALAR

.Lafter_scalar_cast\@:

  // Decrement the size of elements in the jagged vector so that iteration
  // counting works correctly in the outer loop.
  add $MRF_SIZE, $MRF_SIZE, -1

.Louter_loop\@:
  ld32step $MRF_IN1, $mzero, $MRF_IN1_2D+=, 1
  ld32step $MRF_NITER, $mzero, $MRF_IN1_2D+=, 1
.ifc \INPLACE, 0
  ld32step $MRF_OUT, $mzero, $MRF_OUT_2D+=, 1
.else
  mov $MRF_OUT, $MRF_IN1
.endif

  // Store a pointer to the beginning of the last 8-byte aligned "chunk" of data
  // in the array. This chunk may contain 1, 2, or 3 values that need to be
  // processed separately. If a worker points to this chunk after the loop, it
  // should process these remaining values.
  and       $MRF_N_REMAINING, $MRF_NITER, 0x3
  sub       $MRF_SCRATCH, $MRF_NITER, $MRF_N_REMAINING
  mov       $MRF_LAST_CHUNK, $MRF_IN1
  ldz16step $mzero, $mzero, $MRF_LAST_CHUNK+=, $MRF_SCRATCH

  // Divide the number of iterations by 4, since values are processed in groups
  // of four.
  shr $MRF_NITER, $MRF_NITER, 2

  // Do operations in fp16 or fp32 depending on the result of the
  // checkAccuracyWhenCastComputeImpl() call.
  brnz $MRF_ACC_CHECK_RES, .Lscalar_multiply_fast\@

.Lscalar_multiply_slow\@:
  call $MRF_LINK, scalar_multiply_slow
  brnzdec $MRF_SIZE, .Louter_loop\@
  exitz $mzero
.Lscalar_multiply_fast\@:
  call $MRF_LINK, scalar_multiply_fast
  brnzdec $MRF_SIZE, .Louter_loop\@
  exitz $mzero
.size \NAME, .-\NAME
.endm
// -----------------------------------------------------------------------------


// Create vertices.
// -----------------------------------------------------------------------------
CREATE_SCALAR_MULTIPLY_1D_WORKER SCALAR_MULTIPLY_1D_HALF_FLOAT \
                                 SCALAR_MULTIPLY_1D_INPLACE_HALF_FLOAT \
                                 "SCALAR_MULTIPLY_1D_COMMON_HALF_FLOAT"
CREATE_SCALAR_MULTIPLY_2D_WORKER SCALAR_MULTIPLY_2D_HALF_FLOAT         0
CREATE_SCALAR_MULTIPLY_2D_WORKER SCALAR_MULTIPLY_2D_INPLACE_HALF_FLOAT 1
// -----------------------------------------------------------------------------

#endif // __IPU__
