#ifdef __IPU__

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
// $ACC_0, $ACC_1, $ACC_PAIR should be implemented in the file that calls the macro

// Scratch for Constants
#define CONST_SCRATCH_0 a0
#define CONST_SCRATCH_1 a1
#define CONST_SCRATCH_PAIR a0:1

// Clamped version of activations
// $XCLAMPED_0, $XCLAMPED_1, $XCLAMPED_PAIR should be implemented in the file that calls the macro

// Temporary locations
#define ASCRATCH_0 a2
#define ASCRATCH_1 a3
#define ASCRATCH_PAIR a2:3

// Packed Constant, used only for the Half implementation
#define HALF_CLAMP_LIMITS a6
#define CONST_HI_1_0_LO_0_5 a1

// Constants used only for the Float implementation
#define FLOAT_CLAMP_LIMITS_0 a6
#define FLOAT_CLAMP_LIMITS_1 a7
#define FLOAT_CLAMP_LIMITS_PAIR a6:7

#define FACTOR1_PAIR a6:7
	
// Macro: Calculate GELU non-linearity gradient for a multiple of 4xHalf
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
.macro NONLINEARITY_GELU_HALF n_64bit act_base ingrad_base outgrad_base stride

  ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride
    
  // Use $MSCRATCH as a flag.
  //    $MSCRATCH = 1 indicates that N-1 iterations have executed
  //    $MSCRATCH = 0 indicates that all iterations have executed
  //
  // The $MSCRATCH flag must be initialised to zero in order to support the
  // case when n_64bit is 1, in which case a single pass through the 
  // repeat loop instructions is sufficient.
  {
    zero $MSCRATCH
    f16v4clamp $XCLAMPED_PAIR, $OUT_PAIR, $HALF_CLAMP_LIMITS
  }

  // For efficient use of the f16v4mix instruction, a 2-stage pipeline has
  // been used in the implementation of the loop. 
  //
  // The first and the last iteration are executed using the instructions 
  // within the repeat block but without the use of the rpt instruction 
  // explicitly.
  //
  // Ensure that if N1 is less than 8 (i.e., if N1_64BIT==1), do not branch to
  // the repeat instruction.
  //
  // ASCRATCH = x'^2
  {
    add $MSCRATCH2, \n_64bit, -1	
    f16v4mul $ASCRATCH_PAIR, $XCLAMPED_PAIR, $XCLAMPED_PAIR
  }
    
  // ACC = x'^3
  {
    ld32 $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5
    f16v4mul $ACC_PAIR, $XCLAMPED_PAIR, $ASCRATCH_PAIR
  }

  // ASCRATCH = -0.5 . [x'^2]
  f16v4mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:BL, $ASCRATCH_PAIR

  // ASCRATCH = exp[-0.5 . x'^2]
  f16v2exp $ASCRATCH_0, $ASCRATCH_0

  f16v2exp $ASCRATCH_1, $ASCRATCH_1

  // ASCRATCH = x' . [exp(-0.5 . x'^2)]
  f16v4mul $ASCRATCH_PAIR, $ASCRATCH_PAIR, $XCLAMPED_PAIR

  // ASCRATCH = ALPHA . [x' . exp(-0.5 . x'^2)]
  f16v4mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:BU, $ASCRATCH_PAIR

  // Start "(alpha . beta . [x'^3]) + (alpha . [x'])" calculation
  {
    st64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF
    f16v4mix $azeros, $ACC_PAIR, $XCLAMPED_PAIR
  }
    
  // Since this macro can be used within a Supervisor codelet, over-reads need to be avoided.
  // if N1_64BIT == 1, avoid loading the pipeline with the next set of inputs 
  brz $MSCRATCH2, .Lhalf_loop_last

  // Load pipeline with next set of inputs
  ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride

  ld32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
    
  // $MSCRATCH2 is used as a flag to decide to run the repeat instruction
  // for all the loop iterations besides the first and the last.
  {
    bri .Lhalf_loop_first
    f16v4clamp $XCLAMPED_PAIR, $OUT_PAIR, $HALF_CLAMP_LIMITS
  }

.Lhalf_execute_rpt_block:

  // Reinitialise the $MSCRATCH flag to 1 to ensure that the repeat loop
  // instructions are executed for a last time after the repeat block has
  // fully executed.
  setzi $MSCRATCH, 1

  // Reset flag to indicate that repeat instruction is not to be called again.
  zero $MSCRATCH2

  // Do not execute the repeat instruction for the first or the last iteration.
  add \n_64bit, \n_64bit, -2
    
  rpt \n_64bit, (2f - 1f) / 8 - 1
    
1:
    
  // Load inputs for iteration N
  //
  // ACC = OUTGRAD . ACC
  {
    ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride
    f16v4mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  }

  // Store ouputs for iteration N-2
  {
    st64step $ACC_PAIR, \ingrad_base, $INGRAD_PTR+=, \stride
    f16v4clamp $XCLAMPED_PAIR, $OUT_PAIR, $HALF_CLAMP_LIMITS
  }

.Lhalf_loop_first:
  // Pipeline stage-1 processing for iteration N
  //
  // ASCRATCH = x'^2
  {
    nop
    f16v4mul $ASCRATCH_PAIR, $XCLAMPED_PAIR, $XCLAMPED_PAIR
  }
    
  // ACC = x'^3
  {
    ld32 $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_ALPHA_LO_M0_5
    f16v4mul $ACC_PAIR, $XCLAMPED_PAIR, $ASCRATCH_PAIR
  }

  // ASCRATCH = -0.5 . [x'^2]
  {
    nop
    f16v4mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:BL, $ASCRATCH_PAIR
  }

  // ASCRATCH = exp[-0.5 . x'^2]
  {
    nop
    f16v2exp $ASCRATCH_0, $ASCRATCH_0
  }

  {
    nop
    f16v2exp $ASCRATCH_1, $ASCRATCH_1
  }

  // ASCRATCH = x' . [exp(-0.5 . x'^2)]
  {
    nop
    f16v4mul $ASCRATCH_PAIR, $ASCRATCH_PAIR, $XCLAMPED_PAIR
  }

  // ASCRATCH = ALPHA . [x' . exp(-0.5 . x'^2)]
  {
    nop
    f16v4mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:BU, $ASCRATCH_PAIR
  }

.Lhalf_loop_last:
  // Pipeline stage-2 processing for iteration N-1
  //
  // On the last iteration, $ACC_PAIR and $XCLAMPED_PAIR will be "dummy" values
  // which have no effect on the output of the function. This instruction is 
  // only to flush out the result of the last iteration into $FACTOR1_PAIR.
  {
    ld64 $FACTOR1_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF
    f16v4mix $ACC_PAIR, $ACC_PAIR, $XCLAMPED_PAIR
  }
    
  {
    ld32   $CONST_HI_1_0_LO_0_5, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_HALF_HI_1_0_LO_0_5
    f16v2tanh $ACC_0, $ACC_0
  }

  {
    st64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF
    f16v2tanh $ACC_1, $ACC_1
  }

  // ACC = phi + FACTOR1
  {
    nop
    f16v4add $ACC_PAIR, $ACC_PAIR, $FACTOR1_PAIR
  }

  // ACC = 1 + [phi + FACTOR1]
  {
    ld64step $ASCRATCH_PAIR, \outgrad_base, $OUTGRAD_PTR+=, \stride
    f16v4add $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BU, $ACC_PAIR
  }

  // ACC = 0.5 . [1 + phi + FACTOR1]
  {
    ld32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
    f16v4mul $ACC_PAIR, $CONST_HI_1_0_LO_0_5:BL, $ACC_PAIR
  }

2:

  // Execute repeat block only if \n1_64bit is at least 2
  brnz $MSCRATCH2, .Lhalf_execute_rpt_block
    
  // ACC = OUTGRAD . ACC
  f16v4mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  
  // Store for the last iteration of the repeat block as well as for the very
  // last iteration of \n1_64bit
  st64step $ACC_PAIR, \ingrad_base, $INGRAD_PTR+=, \stride

  // Use instructions in the repeat block for flushing out the pipeline.
  // Flush the mix instruction using zeros in $ACC_PAIR, to ensure that the output does not overflow.
  {
    brnzdec $MSCRATCH, .Lhalf_loop_last
    zero $ACC_PAIR
  }
    
.endm

    
// Macro: Calculate GELU non-linearity gradient for a multiple of 4xHalf
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
.macro NONLINEARITY_GELU_FLOAT n_64bit act_base ingrad_base outgrad_base stride

  ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride
    
  // Use $MSCRATCH as a flag.
  //    $MSCRATCH = 1 indicates that N-1 iterations have executed
  //    $MSCRATCH = 0 indicates that all iterations have executed
  //
  // The $MSCRATCH flag must be initialised to zero in order to support the
  // case when n_64bit is 1, in which case a single pass through the 
  // repeat loop instructions is sufficient.
  {
    zero $MSCRATCH
    f32v2clamp $XCLAMPED_PAIR, $OUT_PAIR, $FLOAT_CLAMP_LIMITS_PAIR
  }

  // For efficient use of the f32v2axpy instruction, a 2-stage pipeline has
  // been used in the implementation of the loop. 
  //
  // The first and the last iteration are executed using the instructions 
  // within the repeat block but without the use of the rpt instruction 
  // explicitly.
  //
  // Ensure that if N1 is less than 8 (i.e., if N1_64BIT==1), do not branch to
  // the repeat instruction.
  //
  // ASCRATCH = x'^2
  {
    add $MSCRATCH2, \n_64bit, -1
    f32v2mul $ASCRATCH_PAIR, $XCLAMPED_PAIR, $XCLAMPED_PAIR
  }
    
  // ACC = x'^3
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_M0_5
    f32v2mul $ACC_PAIR, $XCLAMPED_PAIR, $ASCRATCH_PAIR
  }

  // ASCRATCH = -0.5 . [x'^2]
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_TIMES_BETA
    f32v2mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:B, $ASCRATCH_PAIR
  }

  // ACC= (alpha . beta) . [x'^3]
  f32v2mul $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
    
  // ASCRATCH = exp[-0.5 . x'^2]
  f32exp $ASCRATCH_0, $ASCRATCH_0

  f32exp $ASCRATCH_1, $ASCRATCH_1

  // ASCRATCH = x' . [exp(-0.5 . x'^2)]
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32v2mul $ASCRATCH_PAIR, $ASCRATCH_PAIR, $XCLAMPED_PAIR
  }

  // ASCRATCH = ALPHA . [x' . exp(-0.5 . x'^2)]
  {
    nop
    f32v2mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:B, $ASCRATCH_PAIR
  }

  // Start "alpha . [x'] + (alpha . beta . [x'^3])" calculation
  {
    st64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF
    f32v2axpy $ACC_PAIR, $XCLAMPED_PAIR, $ACC_PAIR
  }
    
  // Since this macro can be used within a Supervisor codelet, over-reads need to be avoided.
  // if N1_64BIT == 1, avoid loading the pipeline with the next set of inputs 
  brz $MSCRATCH2, .Lfloat_loop_last

  // Load pipeline with next set of inputs
  ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride

  ld64    $FLOAT_CLAMP_LIMITS_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP
    
  // $MSCRATCH2 is used as a flag to decide to run the repeat instruction
  // for all the loop iterations besides the first and the last.
  {
    bri .Lfloat_loop_first
    f32v2clamp $XCLAMPED_PAIR, $OUT_PAIR, $FLOAT_CLAMP_LIMITS_PAIR
  }

.Lfloat_execute_rpt_block:

  // Reinitialise the $MSCRATCH flag to 1 to ensure that the repeat loop
  // instructions are executed for a last time after the repeat block has
  // fully executed.
  setzi $MSCRATCH, 1

  // Reset flag to indicate that repeat instruction is not to be called again.
  zero $MSCRATCH2

  // Do not execute the repeat instruction for the first or the last iteration.
  add \n_64bit, \n_64bit, -2
    
  rpt \n_64bit, (2f - 1f) / 8 - 1

1:
  // Load inputs for iteration N
  //
  // ACC = OUTGRAD . ACC
  {
    ld64step $OUT_PAIR, \act_base, $OUT_PTR+=, \stride
    f32v2mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  }

  // Store outputs for iteration N-2
  {
    st64step $ACC_PAIR, \ingrad_base, $INGRAD_PTR+=, \stride
    f32v2clamp $XCLAMPED_PAIR, $OUT_PAIR, $FLOAT_CLAMP_LIMITS_PAIR
  }

.Lfloat_loop_first:
  // Pipeline stage-1 processing for iteration N
  //
  // ASCRATCH = x'^2
  {
    nop
    f32v2mul $ASCRATCH_PAIR, $XCLAMPED_PAIR, $XCLAMPED_PAIR
  }

  // ACC = x'^3
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_M0_5
    f32v2mul $ACC_PAIR, $XCLAMPED_PAIR, $ASCRATCH_PAIR
  }

  // ASCRATCH = -0.5 . [x'^2]
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_TIMES_BETA
    f32v2mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:B, $ASCRATCH_PAIR
  }

  // ACC = (alpha . beta) . [x'^3)
  {
    nop
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

  // ASCRATCH = exp[-0.5 . x'^2]
  {
    nop
    f32exp $ASCRATCH_0, $ASCRATCH_0
  }

  {
    nop
    f32exp $ASCRATCH_1, $ASCRATCH_1
  }

  // ASCRATCH = x' . [exp(-0.5 . x'^2)]
  {
    ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    f32v2mul $ASCRATCH_PAIR, $ASCRATCH_PAIR, $XCLAMPED_PAIR
  }

  // ASCRATCH = ALPHA . [x' . exp(-0.5 . x'^2)]
  {
    nop
    f32v2mul $ASCRATCH_PAIR, $CONST_SCRATCH_0:B, $ASCRATCH_PAIR
  }

.Lfloat_loop_last:
  // Pipeline stage-2 processing for iteration N-1
  //
  // On the last iteration, $ACC_PAIR and $XCLAMPED_PAIR will be "dummy" values
  // which have no effect on the output of the function. This instruction is 
  // only to flush out the result of the last iteration into $FACTOR1_PAIR.
  {
    ld64 $FACTOR1_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF
    f32v2axpy $ACC_PAIR, $XCLAMPED_PAIR, $ACC_PAIR
  }
    
  {
    st64 $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_X_TIMES_GAUSSIAN_PDF
    f32tanh $ACC_0, $ACC_0
  }

  {
    nop
    f32tanh $ACC_1, $ACC_1
  }

  // ACC = phi + $FACTOR1
  {
    ld64   $CONST_SCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_FLOAT_HI_1_0_LO_0_5
    f32v2add $ACC_PAIR, $ACC_PAIR, $FACTOR1_PAIR
  }

  // ACC = 1 + [phi + $FACTOR1]
  {
    ld64    $FLOAT_CLAMP_LIMITS_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP
    f32v2add $ACC_PAIR, $CONST_SCRATCH_1:B, $ACC_PAIR
  }

  // ACC = 0.5 . [1 + phi + $FACTOR1]
  {
    ld64step $ASCRATCH_PAIR, \outgrad_base, $OUTGRAD_PTR+=, \stride
    f32v2mul $ACC_PAIR, $CONST_SCRATCH_0:B, $ACC_PAIR
  }

2:
  // Execute 2nd iteration if \n1_64bit is at least 2
  brnz $MSCRATCH2, .Lfloat_execute_rpt_block
    
  // ACC = OUTGRAD . ACC
  f32v2mul $ACC_PAIR, $ASCRATCH_PAIR, $ACC_PAIR
  
  // Store for the last iteration of the repeat block as well as for the very
  // last iteration of \n1_64bit
  st64step $ACC_PAIR, \ingrad_base, $INGRAD_PTR+=, \stride

  // Use instructions in the repeat block for flushing out the pipeline.
  // Flush the mix instruction using zeros in $ACC_PAIR, to ensure that the output does not overflow.
  {
    brnzdec $MSCRATCH, .Lfloat_loop_last
    zero $ACC_PAIR
  }
    
.endm

#endif // __IPU__
