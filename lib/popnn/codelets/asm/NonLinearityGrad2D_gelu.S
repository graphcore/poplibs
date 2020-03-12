// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::NonLinearityGrad2D vertex template instantiations.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.
//  * All input/output regions 8-byte aligned.

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___half_popnn__NonLinearityType__GELU
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___float_popnn__NonLinearityType__GELU

// Force all accumulators to zero
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// Constants
#define OUTGRAD_PTR_VOFFSET 0
#define OUT_PTR_VOFFSET 4
#define INGRAD_BASE_AND_N0_VOFFSET 8
#define INGRAD_DELTAN_PTR_VOFFSET 12

#define PTR64_SHL_BITS 3

#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
#define DELTAN_BASE_PTR_BITS 20
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)
#else
#define DELTAN_BASE_PTR_BITS 24
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)
#endif

#define HALF_1_0 0x3C00
#define HALF_0_5 0x3800
#define HALF_MINUS_0_5 0xB800
#define HALF_ALPHA 0x3A62  // sqrt(2 / PI) ~= 0.7980
#define HALF_BETA 0x2989   // 0.0447
#define HALF_ALPHA_TIMES_BETA 0x2891 // 3.568E-2
#define HALF_6_0 0x4600
#define HALF_MINUS_6_0 0xC600

#define FLOAT_12_0 0x41400000
#define FLOAT_MINUS_12_0 0xC1400000
#define FLOAT_MINUS_0_5 0xBF000000
#define FLOAT_0_5 0x3F000000
#define FLOAT_ALPHA 0x3F4C422A  // sqrt(2 / PI) ~= 0.79788456
#define FLOAT_BETA 0x3D372713   // 0.044715
#define FLOAT_ALPHA_TIMES_BETA 0x3D122279   // 3.5677407E-2

// Scratch Offsets in 2xFloats
#define SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5 (0 / 2)
#define SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA    (2 / 2)
#define SCRATCH_OFFSET_FLOAT_CLAMP           (4 / 2)
#define SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF  (6 / 2)

// Scratch Offsets in 1xFloats
#define SCRATCH_OFFSET_CONST_HALF_HI_1_0_LO_0_5   8
#define SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5  9
#define SCRATCH_OFFSET_HALF_CLAMP_LIMITS     10
#define SCRATCH_OFFSET_CONST_M0_5            11
#define SCRATCH_OFFSET_N0                    12
#define SCRATCH_OFFSET_CONST_ALPHA_TIMES_BETA (SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA * 2)
#define SCRATCH_OFFSET_CONST_ALPHA           ((SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA * 2) + 1)
#define SCRATCH_OFFSET_CONST_0_5             (SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5 * 2)
#define SCRATCH_OFFSET_CONST_1_0             ((SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5 * 2) + 1)

// Worker register aliases
#define MASK m0
#define OUTGRAD_OUTER_PTR m1
#define OUTGRAD_PTR m2
#define OUT_OUTER_PTR m3
#define OUT_PTR m4
#define INGRAD_BASE_PTR m5
#define INGRAD_DELTAN_PTR m6
#define INGRAD_PTR m7
#define N0 m8
#define N0_B m9
#define N1 m9
#define MSCRATCH m10
#define MSCRATCH2 m8
#define N1_64BIT m11

// Output gradient, which is an input to the subroutines in this file
#define OUTGRAD a1
#define OUTGRAD_PAR a1

// Result accumulator
// Accumulator also functions as factor0
#define ACC_0 a4
#define ACC_1 a5
#define ACC_PAIR a4:5

// Clamped version of activations
#define XCLAMPED_0 a6
#define XCLAMPED_1 a7
#define XCLAMPED_PAIR a6:7


// Activations
#define OUT a0

// Scratch for Constants
#define CONST_SCRATCH a0

// Packed Constant, used only for the Half implementation
#define HALF_CLAMP_LIMITS a6
#define CONST_HI_1_0_LO_0_5 a0

// Packed Constant, used only for the Float implementation
#define FLOAT_CLAMP_LIMITS_PAIR a6:7

#define FACTOR1 a2
#define FACTOR1_PAIR a2:3

// Subroutine: Calculate GELU non-linearity gradient for a single 2xHalf
//
//   x' = clamp(activation)
//   alpha = sqrt(2 / PI)
//   beta = 0.044715
//   phi = tanh(x' * alpha * (1 + beta * x' * x'))
//   g = 1 + phi + (sqrt(2 / PI) * x' * exp(-x' * x' / 2))
//   grad_in = grad_out * 0.5 * g
//
// The above calculation can be further factorized as follows:
//
//   x' = clamp(activation)
//   phi = tanh(alpha * [x'] + (alpha * beta) * [x'^3])
//   factor1 = alpha * x' * exp(-x' * x' / 2)
//   g = 0.5 * (1 + phi + factor1)
//   grad_in = grad_out * g
//
// The f16v4mix instruction is used to calculate phi as follows:
//
//     r = a.x + b.y,    where a = alpha * beta
//                             b = alpha
//                             x = x'^3
//                             y = x'
//     phi = tanh(r)
//
//     $TAS <- (a & 0xffff) | ((b & 0xffff) << 16)
//
//  Note: 1. $OUT should contain the 16x2 activation inputs
//        2. $HALF_CLAMP_LIMITS should contain clamp limits.
//           This register is used within the function and is restored
//           before the function exits.
//        3. $OUTGRAD should contain the 16x2 gradout inputs
//        4. $ACC_1 and $XCLAMPED_1 must be initialised to any
//           non-Inf/non-NaN value before this function is called
//
.section .text.calc_GELU_half
.globl calc_GELU_half
.type calc_GELU_half, @function

.align 4
calc_GELU_half:
  // XCLAMPED (x') = clamp(x)
  f16v2clamp $XCLAMPED_0, $OUT, $HALF_CLAMP_LIMITS

  // FACTOR1 = x'^2
  f16v2mul $FACTOR1, $XCLAMPED_0, $XCLAMPED_0

  // ACC = x'^3
  {
    ld32 $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5
    f16v2mul $ACC_0, $XCLAMPED_0, $FACTOR1
  }

  // FACTOR1 = -0.5 . [x'^2]
  f16v2mul $FACTOR1, $CONST_SCRATCH:BL, $FACTOR1

  // FACTOR1 = exp[-0.5 . x'^2]
  f16v2exp $FACTOR1, $FACTOR1

  // FACTOR1 = x' . [exp(-0.5 . x'^2)]
  f16v2mul $FACTOR1, $FACTOR1, $XCLAMPED_0

  // FACTOR1 = ALPHA . [x' . exp(-0.5 . x'^2)]
  f16v2mul $FACTOR1, $CONST_SCRATCH:BU, $FACTOR1

  // ACC  =  (alpha . beta . [x^3]) + alpha . [x']
  // Only the lower registers such as ACC_0 are of interest in the following 2 instructions
  f16v4mix $azeros, $ACC_PAIR, $XCLAMPED_PAIR
  f16v4mix $ACC_PAIR, $azeros, $azeros

  // ACC = tanh[(alpha . beta . x^3) + (alpha . x')]
  f16v2tanh $ACC_0, $ACC_0

  // ACC = phi + FACTOR1
  {
    ld32   $CONST_HI_1_0_LO_0_5, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_1_0_LO_0_5
    f16v2add $ACC_0, $ACC_0, $FACTOR1
  }

  // ACC = 1 + [phi + FACTOR1]
  f16v2add $ACC_0, $CONST_HI_1_0_LO_0_5:BU, $ACC_0

  // ACC = 0.5 . [1 + phi + FACTOR1]
  // Load HALF_CLAMP_LIMITS for use after the end of this subroutine.
  {
    ld32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
    f16v2mul $ACC_0, $CONST_HI_1_0_LO_0_5:BL, $ACC_0
  }

  // ACC = OUTGRAD . ACC
  {
    br $MSCRATCH
    f16v2mul $ACC_0, $OUTGRAD, $ACC_0
  }


// Subroutine: Calculate GELU non-linearity gradient for a single Float
//
//   x' = clamp(activation)
//   alpha = sqrt(2 / PI)
//   beta = 0.044715
//   phi = tanh(x' * alpha * (1 + beta * x' * x'))
//   g = 1 + phi + (sqrt(2 / PI) * x' * exp(-x' * x' / 2))
//   grad_in = grad_out * 0.5 * g
//
// The above calculation can be further factorized as follows:
//
//   x' = clamp(activation)
//   phi = tanh(alpha * [x'] + (alpha * beta) * [x'^3])
//   factor1 = alpha * x' * exp(-x' * x' / 2)
//   g = 0.5 * (1 + phi + factor1)
//   grad_in = grad_out * g
//
// The f32v2axpy instruction is used to calculate phi as follows:
//
//     r = a.x + y,      where a = alpha
//                             x = x'
//                             y = alpha * beta * x'^3
//     phi = tanh(r)
//
//     $TAS <- a
//
//  Note: 1. $OUT should contain the activation input
//        2. $FLOAT_CLAMP_LIMITS_PAIR should contain clamp limits
//           This register is used within the function and is restored
//           before the function exits.
//        3. $OUTGRAD should contain the gradout input
//        4. $ACC_1 and $XCLAMPED_1 must be initialised to any
//           non-Inf/non-NaN value before this function is called
//
.section .text.calc_GELU_float
.globl calc_GELU_float
.type calc_GELU_float, @function

.align 4
calc_GELU_float:
  // XCLAMPED (x') = clamp(x)
  f32clamp $XCLAMPED_0, $OUT, $FLOAT_CLAMP_LIMITS_PAIR

  // FACTOR1 = x'^2
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_M0_5
    f32mul $FACTOR1, $XCLAMPED_0, $XCLAMPED_0
  }

  // ACC = x'^3
  {
    nop
    f32mul $ACC_0, $XCLAMPED_0, $FACTOR1
  }

  // FACTOR1 = -0.5 . [x'^2]
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_TIMES_BETA
    f32mul $FACTOR1, $CONST_SCRATCH, $FACTOR1
  }

  // ACC = (alpha . beta) . [x^3]
  f32mul $ACC_0, $CONST_SCRATCH, $ACC_0

  // FACTOR1 = exp[-0.5 . x'^2]
  f32exp $FACTOR1, $FACTOR1

  // FACTOR1 = x' . [exp(-0.5 . x'^2)]
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32mul $FACTOR1, $FACTOR1, $XCLAMPED_0
  }

  // FACTOR1 = ALPHA . [x' . exp(-0.5 . x'^2)]
  f32mul $FACTOR1, $CONST_SCRATCH, $FACTOR1

  // ACC  =  alpha . [x'] + [alpha . beta . x^3]
  // Only the lower registers such as ACC_0 are of interest in the following 2 instructions
  // Note that $ACC_1 should be a valid element since it has been used for OUTGRAD.
  f32v2axpy $azeros, $XCLAMPED_PAIR, $ACC_PAIR
  f32v2axpy $ACC_PAIR, $azeros, $azeros

  f32tanh $ACC_0, $ACC_0

  // ACC = phi + FACTOR1
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
    f32add $ACC_0, $FACTOR1, $ACC_0
  }

  // ACC = 1 + [phi + FACTOR1]
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5
    f32add $ACC_0, $CONST_SCRATCH, $ACC_0
  }

  // ACC = 0.5 . [1 + phi + FACTOR1]
  {
    ld64    $FLOAT_CLAMP_LIMITS_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP
    f32mul $ACC_0, $CONST_SCRATCH, $ACC_0
  }

  // ACC = OUTGRAD . ACC
  {
    br $MSCRATCH
    f32mul $ACC_0, $OUTGRAD, $ACC_0
  }


#undef CONST_SCRATCH
#undef FACTOR1
#undef FACTOR1_PAIR
#undef FLOAT_CLAMP_LIMIT_PAIR
#undef HALF_CLAMP_LIMITS
#undef CONST_HI_1_0_LO_0_5
#undef OUTGRAD
#undef OUT

// NOTE: Register definitions as well as Loop implementation are included here
#include "NonLinearityGradLoop_gelu.S"

.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

.align 8
  // For rpt alignment below.
  nop
HALF_SYMBOL:
  // Initialise $ACC_1 as well as $XCLAMPED_1 as required
  // by function calc_GELU_half()
  {
    ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
    zero $ACC_PAIR
  }

  {
    ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
    zero $XCLAMPED_PAIR
  }

  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET/4

  // Unpack base pointer and n0
#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  setzi $MASK, DELTAN_BASE_PTR_MASK
#else
  ldconst $MASK, DELTAN_BASE_PTR_MASK
#endif
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  // DeltaN table pointer is a ScaledPtr32, gives offset in
  // 32-bit units from TMEM_REGION0_BASE_ADDR.
  // Actually added offset to reclaim a register rather than
  // using extra offset parameters for loads
  ldz16 $INGRAD_DELTAN_PTR, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/2
  shl $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, 2
  setzi $MSCRATCH, TMEM_REGION0_BASE_ADDR
  add $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, $MSCRATCH
#else
  // DeltaN table pointer contains a 24 bit absolute address
  // followed by the upper 8 bits of N0. Combine with the
  // lower N0 bits which has alraedy been loaded from the
  // upper 8 bits of the Base pointer
  {
    ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/4
    fnop // rpt alignment
  }
  and $INGRAD_DELTAN_PTR, $MSCRATCH, $MASK
  shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
  shl $N0, $N0, 8
  or $N0, $N0, $N0_B
#endif

  setzi $MASK, DELTAN_OFFSET_MASK

  // Load clamp values
  ldconst $HALF_CLAMP_LIMITS, (HALF_MINUS_6_0 | (HALF_6_0 << 16))
  ldconst $ASCRATCH_0, (HALF_ALPHA_TIMES_BETA) | (HALF_ALPHA << 16)

  {
    st32 $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
    uput $TAS, $ASCRATCH_0
  }

  ldconst $ASCRATCH_0, (HALF_0_5) | (HALF_1_0 << 16)
  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_1_0_LO_0_5
  ldconst $ASCRATCH_0, (HALF_MINUS_0_5) | (HALF_ALPHA << 16)

  {
    st32  $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5
    setzi $ASCRATCH_0, ZAACC_BITMASK
  }

  // Sub 1 to use post-decrementing brnzdec
  // Force all accumulators to zero
  {
    add $N0, $N0, -1
    uput $FP_CLR, $ASCRATCH_0
  }

.Lhalf_n0_loop:
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  and $INGRAD_PTR, $MSCRATCH, $MASK
#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  ldz16step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ldz16step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, PTR64_SHL_BITS
  shl $OUT_PTR, $OUT_PTR, PTR64_SHL_BITS
#else
  shl $INGRAD_PTR, $INGRAD_PTR, PTR64_SHL_BITS
  ld32step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ld32step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
#endif
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 2

  brz $N1_64BIT, .Lhalf_32_bit_remainder

  // Save register $N0
  st32 $N0, $mworker_base, $mzero, SCRATCH_OFFSET_N0

  // Inner Loop kernel
  NONLINEARITY_GELU_HALF $N1_64BIT $mzero $INGRAD_BASE_PTR $mzero 1

  // Restore register $N0
  ld32 $N0, $mworker_base, $mzero, SCRATCH_OFFSET_N0

.Lhalf_32_bit_remainder:
  and $MSCRATCH, $N1, 0x2
  brz $MSCRATCH, .Lhalf_16_bit_remainder

  // Handle remaining 32-bit value
  ld32step $OUT_0, $mzero, $OUT_PTR+=, 1
  ld32step $OUTGRAD_PAR, $mzero, $OUTGRAD_PTR+=, 1
  call $MSCRATCH, calc_GELU_half
  st32step $INGRAD_0, $INGRAD_BASE_PTR, $INGRAD_PTR+=, 1

.Lhalf_16_bit_remainder:
  and $MSCRATCH, $N1, 0x1
  brz $MSCRATCH, .Lhalf_n0_loop_cond

  ldb16 $OUT_0, $OUT_PTR, $mzero, 0

  // Handle remaining 16-bit value
  // Broadcasting lower 16-bits of remaining input words to
  // ensure no exceptions when calculating last gradient.
  ldb16 $OUTGRAD_PAR, $OUTGRAD_PTR, $mzero, 0
  call $MSCRATCH, calc_GELU_half
  ldb16 $INGRAD_1, $INGRAD_BASE_PTR, $INGRAD_PTR, 1
  sort4x16lo $INGRAD_0, $INGRAD_0, $INGRAD_1
  st32 $INGRAD_0, $INGRAD_BASE_PTR, $INGRAD_PTR, 0

.Lhalf_n0_loop_cond:
  brnzdec $N0, .Lhalf_n0_loop
  exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL


.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.align 8
  // For rpt alignment below.
   nop
FLOAT_SYMBOL:
  {
    ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET/4
    zero $ACC_PAIR
  }

  {
    ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
    zero $XCLAMPED_PAIR
  }

  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET/4

  // Unpack base pointer and n0
#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  setzi $MASK, DELTAN_BASE_PTR_MASK
#else
  ldconst $MASK, DELTAN_BASE_PTR_MASK
#endif
  // Load constants
  ldconst $ASCRATCH_0, FLOAT_ALPHA_TIMES_BETA
  ldconst $ASCRATCH_1, FLOAT_ALPHA

  {
    and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
    uput $TAS, $ASCRATCH_1
  }

  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  // DeltaN table pointer is a ScaledPtr32, gives offset in
  // 32-bit units from TMEM_REGION0_BASE_ADDR.
  // Actually added offset to reclaim a register rather than
  // using extra offset parameters for loads
  ldz16 $INGRAD_DELTAN_PTR, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/2
  shl $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, 2
  setzi $MSCRATCH, TMEM_REGION0_BASE_ADDR
  add $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, $MSCRATCH
#else
  // DeltaN table pointer contains a 24 bit absolute address
  // followed by the upper 8 bits of N0. Combine with the
  // lower N0 bits which has alraedy been loaded from the
  // upper 8 bits of the Base pointer
  {
    ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET/4
    fnop // rpt alignment
  }
  and $INGRAD_DELTAN_PTR, $MSCRATCH, $MASK
  shr $N0_B, $MSCRATCH, DELTAN_BASE_PTR_BITS
  shl $N0, $N0, 8
  or $N0, $N0, $N0_B
#endif

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_FLOAT_ALPHA_ALPHA_TIMES_BETA
    or      $ASCRATCH_0, $azero, FLOAT_0_5
  }

  {
    setzi $MASK, DELTAN_OFFSET_MASK
    f32exp  $ASCRATCH_1, $azero
  }

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5
    or      $ASCRATCH_0, $azero, FLOAT_MINUS_0_5
  }

  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_M0_5

  ldconst $FLOAT_CLAMP_LIMITS_0, FLOAT_MINUS_12_0
  ldconst $FLOAT_CLAMP_LIMITS_1, FLOAT_12_0

  {
    st64    $FLOAT_CLAMP_LIMITS_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP
    setzi $ASCRATCH_0, ZAACC_BITMASK
  }

  // Sub 1 to use post-decrementing brnzdec
  {
    add $N0, $N0, -1
    uput $FP_CLR, $ASCRATCH_0
  }

.Lfloat_n0_loop:
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  and $INGRAD_PTR, $MSCRATCH, $MASK
#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  ldz16step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ldz16step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, PTR64_SHL_BITS
  shl $OUT_PTR, $OUT_PTR, PTR64_SHL_BITS
#else
  shl $INGRAD_PTR, $INGRAD_PTR, PTR64_SHL_BITS
  ld32step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ld32step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
#endif
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 1

  brz $N1_64BIT, .Lfloat_32_bit_remainder

  // Save register $N0
  st32 $N0, $mworker_base, $mzero, SCRATCH_OFFSET_N0

  // Inner Loop kernel
  NONLINEARITY_GELU_FLOAT $N1_64BIT $mzero $INGRAD_BASE_PTR $mzero 1

  // Restore register $N0
  ld32 $N0, $mworker_base, $mzero, SCRATCH_OFFSET_N0

.Lfloat_32_bit_remainder:
  and $MSCRATCH, $N1, 0x1
  brz $MSCRATCH, .Lfloat_n0_loop_condition

  // Handle 32-bit remainder
  ld32 $OUT_0, $OUT_PTR, $azero, 0
  ld32 $OUTGRAD_PAR, $OUTGRAD_PTR, $mzero, 0
  call $MSCRATCH, calc_GELU_float
  st32 $INGRAD_0, $INGRAD_BASE_PTR, $INGRAD_PTR, 0

.Lfloat_n0_loop_condition:
  brnzdec $N0, .Lfloat_n0_loop
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
