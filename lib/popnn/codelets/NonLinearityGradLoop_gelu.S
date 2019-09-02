#ifdef __IPU__


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

/*****************************************************************/
/* Register allocation for 4 x Half and 2 x Float Implmentations */
/*                                                               */
/* Factor0, Factor1, Factor2, x' are stored away in memory until */
/* later, when they will are combined with the contents of the   */
/* accumulator $ACC using the appropriate operation.             */
/*                                                               */
/*****************************************************************/

// Activations
#define OUT_0 a0
#define OUT_1 a1
#define OUT_PAIR a0:1

// Gradient input values, which are the output of the non-linearity function
#define INGRAD_0 a4
#define INGRAD_1 a5
#define INGRAD_PAIR a4:5

// Output gradient, which is an input to the function.
#define OUTGRAD_0 a0
#define OUTGRAD_1 a1
#define OUTGRAD_PAIR a0:1

// Result accumulator
// Accumulator also functions as factor0
#define ACC_0 a4
#define ACC_1 a5
#define ACC_PAIR a4:5

// Scratch for Constants
#define CONST_SCRATCH_0 a0
#define CONST_SCRATCH_1 a1

// Clamped version of activations
#define XCLAMPED_0 a6
#define XCLAMPED_1 a7
#define XCLAMPED_PAIR a6:7

// Temporary locations
#define ASCRATCH_0 a2
#define ASCRATCH_1 a3
#define ASCRATCH_PAIR a2:3

// Packed Constant, used only in the 4 x Half implementation
#define HALF_CLAMP_LIMITS a6
#define CONST_HI_1_0_LO_0_5 a1

// Called Function parameters
#define OUTGRAD_PAR a7


.macro NONLINEARITY_GELU_HALF act_base ingrad_base outgrad_base stride
  {
    ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride
    fnop
  }

  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
    f16v4clamp $XCLAMPED_PAIR, $OUT_PAIR, $HALF_CLAMP_LIMITS
  }

  {
    ld32   $CONST_HI_1_0_LO_0_5, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HI_1_0_LO_0_5
    f16v4mul $ACC_PAIR, $XCLAMPED_PAIR, $XCLAMPED_PAIR
  }

  // ACC = (beta . x'^2) and store to scratch as Factor0
  //       -------
  {
    nop
    f16v4mul $ACC_PAIR, $CONST_SCRATCH_0:BL, $ACC_PAIR
  }

  // ACC = 1 + (beta . x'^2)
  //       ---
  {
    st64 $ACC_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_BETA_X_SQR
    f16v4add $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BU, $ACC_PAIR
  }

  // ACC = alpha . (1 + (beta . x'^2))
  //       -------
  {
    nop
    f16v4mul $ACC_PAIR, $CONST_SCRATCH_0:BU, $ACC_PAIR
  }

  // ACC = x' . alpha . (1 + (beta . x'^2))
  //       ----
  {
    nop
    f16v4mul $ACC_PAIR, $ACC_PAIR, $XCLAMPED_PAIR
  }

  // phi = tanh(x' . alpha . (1 + (beta . x'^2)))
  //       ----
  {
    nop
    f16v2tanh $ASCRATCH_0, $ACC_0
  }

  {
    nop
    f16v2tanh $ASCRATCH_1, $ACC_1
  }

  // ACC = 1 + phi and store to scratch as Factor1
  //       -------
  {
    ld32 $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0
    f16v4add $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BU, $ASCRATCH_PAIR
  }

  // ACC = 1 - phi
  //       -------
  {
    st64 $ACC_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_PLUS_PHI
    f16v4sub $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BU, $ASCRATCH_PAIR
  }

  // ACC = (x' . (1 - phi)) and store to scratch as Factor2
  //       -----
  {
    ld64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_BETA_X_SQR
    f16v4mul $ACC_PAIR, $XCLAMPED_PAIR, $ACC_PAIR
  }

  // ACC = (3 . beta . x'^2)
  //       ----
  {
    st64 $ACC_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_MINUS_PHI_TIMES_X
    f16v4mul $ACC_PAIR, $CONST_SCRATCH_0:BU, $ASCRATCH_PAIR
  }

  // ACC = (1 + (3 . beta . x'^2))
  //        ---
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
    f16v4add $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BU, $ACC_PAIR
  }

  // ACC =  alpha . (1 + (3 . beta . x'^2))
  //        -------
  {
    ld64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_MINUS_PHI_TIMES_X
    f16v4mul $ACC_PAIR, $CONST_SCRATCH_0:BU, $ACC_PAIR
  }

  // ACC =  (1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))
  //        --------------
  {
    nop
    f16v4mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  }

  // ACC =  1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2)))
  //        ---
  {
    ld64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_PLUS_PHI
    f16v4add $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BU, $ACC_PAIR
  }

  // ACC =  (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //        ---------
  {
    ld32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
    f16v4mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  }

  // ACC =  0.5 . (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //        -----
  {
    ld64step $OUTGRAD_PAIR, \outgrad_base, $OUTGRAD_PTR+=, \stride
    f16v4mul $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BL, $ACC_PAIR
  }

  {
    nop
    f16v4mul $ACC_PAIR, $OUTGRAD_PAIR, $ACC_PAIR
  }

  {
    st64step $ACC_PAIR, \ingrad_base, $INGRAD_PTR+=, \stride
    fnop
  }
.endm

.macro NONLINEARITY_GELU_FLOAT act_base ingrad_base outgrad_base stride
  {
    ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride
    fnop
  }

  {
    ld32   $CONST_SCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32v2clamp $XCLAMPED_PAIR, $OUT_PAIR, $ASCRATCH_PAIR
  }

  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_BETA
    f32v2mul $ACC_PAIR, $XCLAMPED_PAIR, $XCLAMPED_PAIR
  }

  // ACC = (beta . x'^2) and store to scratch
  //       -------
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  // ACC = 1 + (beta . x'^2)
  //           ----
  {
    st64 $ACC_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_BETA_X_SQR
    f32v2add $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  // ACC = alpha . (1 + (beta . x'^2))
  //      --------
  {
    nop
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_1:B, $ACC_PAIR
  }

  // ACC = x' . alpha . (1 + (beta . x'^2))
  //      -----
  {
    nop
    f32v2mul $ACC_PAIR, $ACC_PAIR, $XCLAMPED_PAIR
  }

  // phi = tanh(x' . alpha . (1 + (beta . x'^2)))
  //       ----
  {
    nop
    f32tanh $ASCRATCH_0, $ACC_0
  }

  {
    nop
    f32tanh $ASCRATCH_1, $ACC_1
  }

  // ACC = 1 + phi and store to scratch
  //       ---
  {
    ld32 $CONST_SCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0
    f32v2add $ACC_PAIR, $CONST_SCRATCH_0:B, $ASCRATCH_PAIR
  }

  // ACC = 1 - phi
  //       ---
  {
    st64 $ACC_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_PLUS_PHI
    f32v2sub $ACC_PAIR, $CONST_SCRATCH_0:B, $ASCRATCH_PAIR
  }

  // ACC = (x' . (1 - phi)) and store to scratch
  //        ----
  {
    ld64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_BETA_X_SQR
    f32v2mul $ACC_PAIR, $XCLAMPED_PAIR, $ACC_PAIR
  }

  // ACC = (3 . beta . x'^2)
  //        ----
  {
    st64 $ACC_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_MINUS_PHI_TIMES_X
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_1:B, $ASCRATCH_PAIR
  }

  // ACC = (1 + (3 . beta . x'^2))
  //        ---
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32v2add $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  // ACC =  alpha . (1 + (3 . beta . x'^2))
  //        -------
  {
    ld64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_MINUS_PHI_TIMES_X
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  // ACC =  (1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))
  //        ----------------
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
    f32v2mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  }

  // ACC =  1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2)))
  //        ---
  {
    ld64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_ONE_PLUS_PHI
    f32v2add $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  // ACC =  (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //        -----------
  {
    ld32 $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5
    f32v2mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  }

  // ACC =  0.5 . (1 + phi) * (1 + ((1 - phi) . x' . alpha . (1 + (3 . beta . x'^2))))
  //        -----
  {
    ld64step $OUTGRAD_PAIR, \outgrad_base, $OUTGRAD_PTR+=, \stride
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  {
    ld64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP
    f32v2mul $ACC_PAIR, $OUTGRAD_PAIR, $ACC_PAIR
  }

  {
    st64step $ACC_PAIR, \ingrad_base, $INGRAD_PTR+=, \stride
    fnop
  }
.endm

#endif // __IPU__
