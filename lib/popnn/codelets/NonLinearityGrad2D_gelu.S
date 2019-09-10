#ifdef __IPU__

// Assembly implementation of popnn::NonLinearityGrad2D vertex template instantiations.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.
//  * All input/output regions 8-byte aligned.

#include "poplibs_support/TileConstants.hpp"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___half_popnn__NonLinearityType__GELU
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGrad2D___float_popnn__NonLinearityType__GELU

// Constants
#define OUTGRAD_PTR_VOFFSET 0
#define OUT_PTR_VOFFSET 1
#define INGRAD_BASE_AND_N0_VOFFSET 2
#define INGRAD_DELTAN_PTR_VOFFSET (3 * 2) // *2 for 16-bit offset

#define DELTAN_BASE_PTR_BITS 20
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)

#define HALF_3_0 0x4200
#define HALF_1_0 0x3C00
#define HALF_0_5 0x3800

// The actual values used her for alpha and beta are chosen to
// reduce the error in the gradient and hence are slightly off from
// the actual.
#define HALF_ALPHA 0x3A6C  // 0.7978845608f
#define HALF_BETA 0x29C8   // 0.044715
#define HALF_6_0 0x4600
#define HALF_MINUS_6_0 0xC600

#define FLOAT_12_0 0x41400000
#define FLOAT_MINUS_12_0 0xC1400000
    
#define FLOAT_0_5 0x3F000000
#define FLOAT_3_0 0x40400000
#define FLOAT_ALPHA 0x3F4C422A  // 0.7978845608f
#define FLOAT_BETA 0x3D372713   // 0.044715

// Scratch Offsets in 1xFloats
#define SCRATCH_OFFSET_CONST_ALPHA_BETA      0
#define SCRATCH_OFFSET_CONST_HI_1_0_LO_0_5   1
    
#define SCRATCH_OFFSET_CONST_ALPHA           2    
#define SCRATCH_OFFSET_CONST_BETA            3
#define SCRATCH_OFFSET_CONST_0_5             4
#define SCRATCH_OFFSET_CONST_1_0             5
#define SCRATCH_OFFSET_CONST_3_0             6
#define SCRATCH_OFFSET_HALF_CLAMP_LIMITS     7

// Scratch Offsets in 2xFloats
#define SCRATCH_OFFSET_ONE_PLUS_PHI          (8 / 2)
#define SCRATCH_OFFSET_ONE_MINUS_PHI_TIMES_X (10 / 2)
#define SCRATCH_OFFSET_FLOAT_CLAMP           (12 / 2)
#define SCRATCH_OFFSET_BETA_X_SQR            (14 / 2)

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
#define N1 m9
#define MSCRATCH m10
#define N1_64BIT m11

/* The GELU non-linearity calulation of input gradient "grad_in" */
//
//   x' = clamp(activation)
//   phi = tanh(x' * alpha * (1 + beta * x' * x'))
//   g = (1 + phi) * (1  +  (1 - phi) * x' * alpha * (1 +  3 * x' * x' * beta);
//   grad_in = grad_out * 0.5 * g
//
// The above calculation can be factorized in order to reduce redundant operations.
// Each factor is stored away until a suitable time when the value is ready to 
// be accumulated to the results:
//
//   factor0 = beta * x' * x'
//   factor1 = 1 + phi
//   factor2 = x' * (1 - phi)
//
//   phi = tanh(x' * alpha * (1 + factor0))
//   g = factor1 * (1  +  factor2 * alpha * (1 +  3 * factor0);
//   grad_in = grad_out * 0.5 * g
//
// Every step of the calculation is accumulated to the $ACC register until the
// final "grad_in" result is obtained. The contents of $ACC is then stored to
// the appropriate index in the grad_in tensor
//
    
// Activations
#define OUT a0

/* Register allocation for 2 x Half Implmentation */

#define CONST_HI_1_0_LO_0_5 a1
#define CONST_SCRATCH a0
    
// Accumulator in which the output of the function is returned
// Accumulator also functions as factor0
#define ACC a4

// phi
#define PHI a3
    
// Factor1 = (1 + phi)
#define FACTOR_1 a5

// Factor2 = x' * (1 - phi)
#define FACTOR_2 a2
    
#define HALF_CLAMP_LIMITS a6

// Clamped version of activation, denoted by x'
#define XCLAMPED a6

// Output gradient, which is an input to the function.
#define OUTGRAD a7
    
.section .text.calc_GELU_half
.globl calc_GELU_half
.type calc_GELU_half, @function

.align 4
calc_GELU_half:
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
    f16v2clamp $XCLAMPED, $OUT, $HALF_CLAMP_LIMITS
  }
  
  {
    ld32   $CONST_HI_1_0_LO_0_5, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HI_1_0_LO_0_5
    f16v2mul $ACC, $XCLAMPED, $XCLAMPED
  }

  // ACC = (beta . x'^2) and store to a temporary register
  //        ------
  f16v2mul $ACC, $CONST_SCRATCH:BL, $ACC

  // Factor_1 = 1 + (beta . x'^2)
  //            ---
  f16v2add $FACTOR_1, $CONST_HI_1_0_LO_0_5:BU, $ACC
    
  // Factor_1 = alpha . (1 + (beta . x'^2))
  //            -------
  f16v2mul $FACTOR_1, $CONST_SCRATCH:BU, $FACTOR_1

  // Factor_1 = x' . alpha . (1 + (beta . x'^2))
  //            ----
  f16v2mul $FACTOR_1, $FACTOR_1, $XCLAMPED

  // phi = tanh(x' . alpha . (1 + (beta . x'^2)))
  //       ----
  f16v2tanh $PHI, $FACTOR_1

  // Factor_1 = 1 + phi and store to a temporary register
  //            ---
  {
    ld32 $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0
    f16v2add $FACTOR_1, $CONST_HI_1_0_LO_0_5:BU, $PHI
  }

  // Factor_2 = 1 - phi
  //            -------
  f16v2sub $FACTOR_2, $CONST_HI_1_0_LO_0_5:BU, $PHI

  // Factor_2 = (x' . (1 - phi)) and store to  a temporary register
  //             -----
  f16v2mul $FACTOR_2, $XCLAMPED, $FACTOR_2
    
  // ACC = (3 . beta . x'^2)
  //        ----
  f16v2mul $ACC, $CONST_SCRATCH:BU, $ACC
    
  // ACC = (1 + (3 . beta . x'^2))
  //        ---
  {
    ld32   $CONST_SCRATCH, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
    f16v2add $ACC, $CONST_HI_1_0_LO_0_5:BU, $ACC
  }
    
  // ACC = alpha . (1 + (3 . beta . x'^2))
  //       -------
  f16v2mul $ACC, $CONST_SCRATCH:BU, $ACC
    
  // ACC = (1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))
  //       -----------
  f16v2mul $ACC, $FACTOR_2, $ACC
    
  // ACC = 1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2)))
  //       ----
  f16v2add $ACC, $CONST_HI_1_0_LO_0_5:BU, $ACC
    
  // ACC = (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //       -----------
  f16v2mul $ACC, $ACC, $FACTOR_1

  // ACC = 0.5 . (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //       -----
  {
    ld32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
    f16v2mul $ACC, $CONST_HI_1_0_LO_0_5:BL, $ACC
  }

  // Calculate final result and restore Clamp limit register values for subsequent iterations
  {
    br $MSCRATCH
    f16v2mul $ACC, $OUTGRAD, $ACC
  }
  

#undef CONST_HI_1_0_LO_0_5
#undef CONST_SCRATCH
#undef ACC
#undef PHI
#undef FACTOR_1
#undef FACTOR_2
#undef HALF_CLAMP_LIMITS
#undef XCLAMPED
#undef OUTGRAD

/* Register allocation for 1 x Float Implmentation */
    
// Accumulator in which the output of the function is returned
// Accumulator also functions as factor0
#define ACC a4

// phi
#define PHI a3
    
// Factor1 = (1 + phi)
#define FACTOR_1 a5

// Factor2 = x' * (1 - phi)
#define FACTOR_2 a2
    
#define FLOAT_CLAMP_LIMITS_PAIR a2:3

// Clamped version of activation, denoted by x'
#define XCLAMPED a6

// Output gradient, which is an input to the function.
#define OUTGRAD a7
    
// Scratch for constants
#define CONST_SCRATCH_0 a0
#define CONST_SCRATCH_1 a1

.section .text.calc_GELU_float
.globl calc_GELU_float
.type calc_GELU_float, @function

// Assume OUT is loaded
// FLOAT_CLAMP_LIMITS_0 should contain clamp limits
.align 4
calc_GELU_float:
  {
    ld32   $CONST_SCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32clamp $XCLAMPED, $OUT, $FLOAT_CLAMP_LIMITS_PAIR
  }
  
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_BETA
    f32mul $ACC, $XCLAMPED, $XCLAMPED
  }

  // ACC = (beta . x'^2) and store to a temporary register
  //        ------
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
    f32mul $ACC, $CONST_SCRATCH_0, $ACC
  }

  // Factor_1 = 1 + (beta . x'^2)
  //            ---
  f32add $FACTOR_1, $CONST_SCRATCH_0, $ACC
    
  // Factor_1 = alpha . (1 + (beta . x'^2))
  //            -------
  f32mul $FACTOR_1, $CONST_SCRATCH_1, $FACTOR_1

  // Factor_1 = x' . alpha . (1 + (beta . x'^2))
  //            ----
  f32mul $FACTOR_1, $FACTOR_1, $XCLAMPED

  // phi = tanh(x' . alpha . (1 + (beta . x'^2)))
  //       ----
  f32tanh $PHI, $FACTOR_1

  // Calculate 1 + phi and store to a temporary register
  //           ---
  {
    ld32 $CONST_SCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0
    f32add $FACTOR_1, $CONST_SCRATCH_0, $PHI
  }

  // Factor_2 = 1 - phi
  //           ---------
  f32sub $FACTOR_2, $CONST_SCRATCH_0, $PHI

  // Factor_2 =  (x' . (1 - phi)) and store to a temporary register
  //              -----
  f32mul $FACTOR_2, $XCLAMPED, $FACTOR_2
    
  // ACC = (3 . beta . x'^2)
  //       ----
  f32mul $ACC, $CONST_SCRATCH_1, $ACC
    
  // ACC = (1 + (3 . beta . x'^2))
  //        ---
  {
    ld32   $CONST_SCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32add $ACC, $CONST_SCRATCH_0, $ACC
  }
    
  // ACC = alpha . (1 + (3 . beta . x'^2))
  //       -------
  f32mul $ACC, $CONST_SCRATCH_1, $ACC
    
  // ACC = (1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))
  //       -----------
  f32mul $ACC, $FACTOR_2, $ACC
    
  // ACC =  1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2)))
  //        ----
  f32add $ACC, $CONST_SCRATCH_0, $ACC
    
  // ACC = (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //       ---------
  {
    ld32 $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5
    f32mul $ACC, $ACC, $FACTOR_1
  }

  // ACC =  0.5 . (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //        -----
  {
    ld64    $FLOAT_CLAMP_LIMITS_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP
    f32mul $ACC, $CONST_SCRATCH_0, $ACC
  }

  // Calculate final result and restore Clamp limit register values for subsequent iterations
  {
    br $MSCRATCH
    f32mul $ACC, $OUTGRAD, $ACC
  }
  

#undef CONST_SCRATCH_0
#undef CONST_SCRATCH_1
#undef ACC
#undef PHI
#undef FACTOR_1
#undef FACTOR_2
#undef HALF_CLAMP_LIMITS_PAIR
#undef XCLAMPED
#undef OUTGRAD
#undef OUT

// NOTE: Register definitions as well as Loop implementation are included here
#include "NonLinearityGradLoop_gelu.S"
    
.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

.align 8
   nop
HALF_SYMBOL:
  ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET
  ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET
  ldz16 $INGRAD_DELTAN_PTR, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET

  // Unpack base pointer and n0
  setzi $MASK, DELTAN_BASE_PTR_MASK
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

  // DeltaN table pointer is a ScaledPtr32, gives offset in
  // 32-bit units from TMEM_REGION0_BASE_ADDR.
  // Actually added offset to reclaim a register rather than
  // using extra offset parameters for loads
  shl $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, 2
  setzi $MSCRATCH, TMEM_REGION0_BASE_ADDR
  add $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, $MSCRATCH

  setzi $MASK, DELTAN_OFFSET_MASK

  // Load clamp values
  ldconst $HALF_CLAMP_LIMITS, (HALF_MINUS_6_0 | (HALF_6_0 << 16))
  st32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
  ldconst $ASCRATCH_1, (HALF_0_5) | (HALF_1_0 << 16)
  ldconst $ASCRATCH_0, (HALF_BETA) | (HALF_ALPHA << 16)

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, (SCRATCH_OFFSET_CONST_ALPHA_BETA/2)
    or      $ASCRATCH_0, $azero, (HALF_3_0 << 16)
  }
    
  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0

  // Sub 1 to use post-decrementing brnzdec
  add $N0, $N0, -1
.Lhalf_n0_loop:
  ldz16step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ldz16step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 3
  shl $OUT_PTR, $OUT_PTR, 3
  and $INGRAD_PTR, $MSCRATCH, $MASK
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 2

  rpt $N1_64BIT, (2f - 1f) / 8 - 1
  
1:
  NONLINEARITY_GELU_HALF $mzero $INGRAD_BASE_PTR $mzero 1
2:

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
FLOAT_SYMBOL:
  ld32 $OUTGRAD_OUTER_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET
  ld32 $OUT_OUTER_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET
  ld32 $MSCRATCH, $mvertex_base, $mzero, INGRAD_BASE_AND_N0_VOFFSET
  ldz16 $INGRAD_DELTAN_PTR, $mvertex_base, $mzero, INGRAD_DELTAN_PTR_VOFFSET

  // Unpack base pointer and n0
  setzi $MASK, DELTAN_BASE_PTR_MASK
  and $INGRAD_BASE_PTR, $MSCRATCH, $MASK
  shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

  // DeltaN table pointer is a ScaledPtr32, gives offset in
  // 32-bit units from TMEM_REGION0_BASE_ADDR.
  // Actually added offset to reclaim a register rather than
  // using extra offset parameters for loads
  shl $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, 2
  setzi $MSCRATCH, TMEM_REGION0_BASE_ADDR
  add $INGRAD_DELTAN_PTR, $INGRAD_DELTAN_PTR, $MSCRATCH

  // Load clamp values
  ldconst $ASCRATCH_0, FLOAT_ALPHA
  ldconst $ASCRATCH_1, FLOAT_BETA

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, (SCRATCH_OFFSET_CONST_ALPHA/2)
    or      $ASCRATCH_0, $azero, FLOAT_0_5
  }

  { 
    setzi $MASK, DELTAN_OFFSET_MASK
    f32exp  $ASCRATCH_1, $azero
  }

  {    
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, (SCRATCH_OFFSET_CONST_0_5/2)
    or      $ASCRATCH_0, $azero, FLOAT_3_0
  }
    
  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0
  ldconst $ASCRATCH_0, FLOAT_MINUS_12_0
  ldconst $ASCRATCH_1, FLOAT_12_0
  st64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP

  // Sub 1 to use post-decrementing brnzdec
  add $N0, $N0, -1
.Lfloat_n0_loop:
  ldz16step $OUTGRAD_PTR, $mzero, $OUTGRAD_OUTER_PTR+=, 1
  ldz16step $OUT_PTR, $mzero, $OUT_OUTER_PTR+=, 1
  ld32step $MSCRATCH, $mzero, $INGRAD_DELTAN_PTR+=, 1
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 3
  shl $OUT_PTR, $OUT_PTR, 3
  and $INGRAD_PTR, $MSCRATCH, $MASK
  shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
  shr $N1_64BIT, $N1, 1

  rpt $N1_64BIT, (2f - 1f) / 8 - 1
  
1:
  NONLINEARITY_GELU_FLOAT $mzero $INGRAD_BASE_PTR $mzero 1
2:

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
