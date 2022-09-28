// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Macros and defines for SWISH nonlinearity

#ifndef _NonLinearitySwishCommon_S_
#define _NonLinearitySwishCommon_S_

// DELTAN
#define DELTAN_BASE_PTR_BITS 24
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)


// Register aliases
#define ACTS_0                  a0
#define ACTS_1                  a1
#define ACTS_PAIR               a0:1
#define SIGM_0                  a2
#define SIGM_1                  a3
#define SIGM_PAIR               a2:3
#define GRAD_0                  a2
#define GRAD_1                  a3
#define GRAD_PAIR               a2:3
#define RESULT_0                a4
#define RESULT_1                a5
#define RESULT_PAIR             a4:5
#define ONE                     a6
#define ASCRATCH                a7

// used for splitting work in supervisor vertices
#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define LOG2_24_OVER_3 3
#define LOG2_12_OVER_3 2

#-------------------------------------------------------------------------------

// All inputs must be separate registers
// Splits 64-bit chunks of n elements between workers.
// The result we want is n / (no. of worker contexts * elements per-64-bits).
// We achieve this by dividing by 3 first, by multiplying n by the reciprocal
// of 3 shifted left. This value is then shifted right by the same amount + any
// further division by powers of 2 to get the actual divisor we want.
// As an example, in this half case there are 4 halves per-64-bits and
// 6 worker contexts so the divisor we want is 24.
// (n / 3) / 8 = n / 24 so the extra divisor is 8, meaning an extra shift of 3.
.macro HALF_SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_24_OVER_3)
    mul \rem, \size, 24
    sub \rem, \n, \rem
.endm

#-------------------------------------------------------------------------------

// All inputs must be separate registers
// As described above in HALF_SPLIT_BETWEEN_WORKERS with different
// divisor.
.macro FLOAT_SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_12_OVER_3)
    mul \rem, \size, 12
    sub \rem, \n, \rem
.endm

#-------------------------------------------------------------------------------

// Macro to do innermost loop gor a V4 variant of half. The loop count is
// assumed to be already subtracted by 1 with LOOP_CNT >= 0
.macro SwishActivationLoopHalfV4 IN_PTR, OUT_PTR, IO_BASE, LOOP_CNT, STRIDE
ld64step  $ACTS_PAIR, $\IO_BASE, $\IN_PTR+=, \STRIDE
{
  rpt $\LOOP_CNT, 2
  f16v2sigm $RESULT_0, $ACTS_0
}
  {
    nop
    f16v2sigm $RESULT_1, $ACTS_1
  }
  {
    ld64step  $ACTS_PAIR, $\IO_BASE, $\IN_PTR+=, \STRIDE
    f16v4mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
  }
  {
    st64step $RESULT_PAIR, $\IO_BASE, $\OUT_PTR+=, \STRIDE
    f16v2sigm $RESULT_0, $ACTS_0
  }
f16v2sigm $RESULT_1, $ACTS_1
f16v4mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
st64step $RESULT_PAIR, $\IO_BASE, $\OUT_PTR+=, \STRIDE
.endm

#-------------------------------------------------------------------------------

// Macro to compute V2 variant of half.
// input in ACT
// output in RESULT
.macro SwishActivationHalfV2 RESULT, ACT
f16v2sigm $\RESULT, $\ACT
f16v2mul $\RESULT, $\RESULT, $\ACT
.endm

#-------------------------------------------------------------------------------

// Macro to do innermost loop gor a V2 variant of float. The loop count is
// assumed to be already subtracted by 1 with LOOP_CNT >= 0
.macro SwishActivationLoopFloatV2 IN_PTR, OUT_PTR, IO_BASE, LOOP_CNT, STRIDE
ld64step $ACTS_PAIR, $\IO_BASE, $\IN_PTR+=, \STRIDE
{
  rpt $\LOOP_CNT, 2
  f32sigm $RESULT_0, $ACTS_0
}
  {
    nop
    f32sigm $RESULT_1, $ACTS_1
  }
  {
    ld64step $ACTS_PAIR, $\IO_BASE, $\IN_PTR+=, \STRIDE
    f32v2mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
  }
  {
    st64step $RESULT_PAIR, $\IO_BASE, $\OUT_PTR+=, \STRIDE
    f32sigm $RESULT_0, $ACTS_0
  }
f32sigm $RESULT_1, $ACTS_1
f32v2mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
st64step $RESULT_PAIR, $\IO_BASE, $\OUT_PTR+=, \STRIDE
.endm

#-------------------------------------------------------------------------------

// Macro to compute V1 variant of float.
.macro SwishActivationFloatV1 RESULT, ACT
// input in ACT
// output in RESULT
f32sigm $\RESULT, $\ACT
f32mul $\RESULT, $\RESULT, $\ACT
.endm

#-------------------------------------------------------------------------------

// Macro to compute gradient for multiple of 4 halves. All pointers must be
// aligned at 64-bit
.macro SwishGradLoopHalfV4 ACTS_PTR, OUTGRAD_PTR, INGRAD_BASE, INGRAD_PTR, ONE, LOOP_CNT, STRIDE
  ld64step $ACTS_PAIR, $mzero, $\ACTS_PTR+=, \STRIDE
  {
    rpt $\LOOP_CNT, 6
    f16v2sigm $SIGM_0, $ACTS_0
  }
    {
      nop
      f16v2sigm $SIGM_1, $ACTS_1
    }
    {
      nop
      f16v4sub $RESULT_PAIR, $\ONE:BL, $SIGM_PAIR
    }
    {
      nop
      f16v4mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
    }
    {
      nop
      f16v4add $RESULT_PAIR, $\ONE:BL, $RESULT_PAIR
    }
    {
      ld64step $GRAD_PAIR, $mzero, $\OUTGRAD_PTR+=, \STRIDE
      f16v4mul $RESULT_PAIR, $RESULT_PAIR, $SIGM_PAIR
    }
    {
      ld64step $ACTS_PAIR, $mzero, $\ACTS_PTR+=, \STRIDE
      f16v4mul $RESULT_PAIR, $RESULT_PAIR, $GRAD_PAIR
    }
    {
      st64step $RESULT_PAIR, $\INGRAD_BASE, $\INGRAD_PTR+=, \STRIDE
      f16v2sigm $SIGM_0, $ACTS_0
    }
  f16v2sigm $SIGM_1, $ACTS_1
  f16v4sub $RESULT_PAIR, $\ONE:BL, $SIGM_PAIR
  f16v4mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
  f16v4add $RESULT_PAIR, $\ONE:BL, $RESULT_PAIR
  {
    ld64step $GRAD_PAIR, $mzero, $\OUTGRAD_PTR+=, \STRIDE
    f16v4mul $RESULT_PAIR, $RESULT_PAIR, $SIGM_PAIR
  }
  f16v4mul $RESULT_PAIR, $RESULT_PAIR, $GRAD_PAIR
  st64step $RESULT_PAIR, $\INGRAD_BASE, $\INGRAD_PTR+=, \STRIDE
.endm

#-------------------------------------------------------------------------------

// Macro to compute V2 variant of half.
.macro SwishGradHalfV2 RESULT, ACT, SIGM, OUTGRAD, ONE
f16v2sigm $\SIGM, $\ACT
f16v2sub $\RESULT, $\ONE, $\SIGM
f16v2mul $\RESULT, $\RESULT,  $\ACT
f16v2add $\RESULT, $\RESULT, $\ONE
f16v2mul $\RESULT, $\RESULT, $\SIGM
f16v2mul $\RESULT, $\RESULT, $\OUTGRAD
.endm

#-------------------------------------------------------------------------------

// Macro to compute gradient for multiple of 2 floats. All pointers must be
// aligned at 64-bit
.macro SwishGradLoopFloatV2 ACTS_PTR, OUTGRAD_PTR, INGRAD_BASE, INGRAD_PTR, ONE, LOOP_CNT, STRIDE

ld64step $ACTS_PAIR, $mzero, $\ACTS_PTR+=, \STRIDE
{
  rpt $\LOOP_CNT, 6
  f32sigm  $SIGM_0, $ACTS_0
}
  {
    nop
    f32sigm $SIGM_1, $ACTS_1
  }
  {
    nop
    f32v2sub $RESULT_PAIR, $\ONE:B, $SIGM_PAIR
  }
  {
    nop
    f32v2mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
  }
  {
    nop
    f32v2add $RESULT_PAIR, $\ONE:B, $RESULT_PAIR
  }
  {
    ld64step $GRAD_PAIR, $mzero, $\OUTGRAD_PTR+=, \STRIDE
    f32v2mul $RESULT_PAIR, $RESULT_PAIR, $SIGM_PAIR
  }
  {
    ld64step $ACTS_PAIR, $mzero, $\ACTS_PTR+=, \STRIDE
    f32v2mul $RESULT_PAIR, $RESULT_PAIR, $GRAD_PAIR
  }
  {
    st64step $RESULT_PAIR, $\INGRAD_BASE, $\INGRAD_PTR+=, \STRIDE
    f32sigm $SIGM_0, $ACTS_0
  }
f32sigm $SIGM_1, $ACTS_1
f32v2sub $RESULT_PAIR, $\ONE:B, $SIGM_PAIR
f32v2mul $RESULT_PAIR, $RESULT_PAIR, $ACTS_PAIR
f32v2add $RESULT_PAIR, $\ONE:B, $RESULT_PAIR
{
  ld64step $GRAD_PAIR, $mzero, $\OUTGRAD_PTR+=, \STRIDE
  f32v2mul $RESULT_PAIR, $RESULT_PAIR, $SIGM_PAIR
}
f32v2mul $RESULT_PAIR, $RESULT_PAIR, $GRAD_PAIR
st64step $RESULT_PAIR, $\INGRAD_BASE, $\INGRAD_PTR+=, \STRIDE
.endm

#-------------------------------------------------------------------------------

// Macro to compute V1 variant of float.
.macro SwishGradFloatV1 RESULT, ACT, SIGM, OUTGRAD, ONE
f32sigm $\SIGM, $\ACT
f32sub $\RESULT, $\ONE, $\SIGM
f32mul $\RESULT, $\RESULT, $\ACT
f32add $\RESULT, $\RESULT, $\ONE
f32mul $\RESULT, $\RESULT, $\SIGM
f32mul $\RESULT, $\RESULT, $\OUTGRAD
.endm

#endif // _NonLinearitySwishCommon_S_
