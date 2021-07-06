// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__


// The GELU non-linearity calculation of output activation
//
//           phi = x' * alpha * (1 + beta * x' * x')
//           factor0 = tanh(phi)
//   Output, y = 0.5 * x * (1 + factor0)
//
// The Mix instruction calculates phi using the expression "a.x + b.y" as follows:
//         
//         alpha * x' + (alpha * beta) * (x')^3
//  
//  $TAS should be loaded with "alpha" in the upper half and "alpha * beta"
//  in the lower half.
//

.macro NONLINEARITY_GELU_HALF n_64bit act_base stride

    
    // Use $MSCRATCH as a flag.
    //    $MSCRATCH = 1 indicates that N-1 iterations have executed
    //    $MSCRATCH = 0 indicates that all iterations have executed
    //
    // The $MSCRATCH flag must be initialised to zero in order to support the
    // case when n_64bit is 1, in which case a single pass through the 
    // repeat loop instructions is sufficient.
    ld64 $ACTS_PAIR, $DATA_PTR, \act_base, 0
    {
      zero $MSCRATCH
      f16v4mul $RESULTS_PAIR, $ACTS_PAIR, $ACTS_PAIR
    }
    // For efficient use of the f16v4mix instruction, a 2-deep pipeline has
    // been used in the implementation of the loop. The 2nd pair of values
    // is loaded here.
    //
    // The first and the last iteration are executed using the instructions 
    // within the repeat block but without the use of the repeat instruction 
    // explicitly.
    //
    // Ensure that if N1 is less than 8 (i.e., if N_64BIT==1), do not branch to
    // the repeat instruction.
    {
      add $MSCRATCH2, \n_64bit, -1
      f16v4mul $RESULTS_PAIR, $ACTS_PAIR, $RESULTS_PAIR
    }

    // $ACTS_PAIR has been loaded with the contents of the DATA_PTR+1, so avoid
    // processing these if N_64BIT == 1.
    {
      brz $MSCRATCH2, .Lhalf_loop_last
      f16v4mix $azeros, $RESULTS_PAIR, $ACTS_PAIR
    }

    ld64 $ACTS_PAIR, $DATA_PTR, \act_base, \stride
    
    // $MSCRATCH2 is used as a flag to decide to run the repeat instruction
    // for all the loop iterations besides the first and the last.
    {
      bri .Lhalf_loop_first
      f16v4mul $RESULTS_PAIR, $ACTS_PAIR, $ACTS_PAIR
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
    
    // Perform the following calculations on the 4 x halves:
    //   - 0.5 * x * (1 + tanh(x * alpha * (1 + (beta * x * x))))

    rpt \n_64bit, (2f - 1f) / 8 - 1

    // All the instructions in the repeat block until the f16v4mix instruction
    // besides st64step process inputs from the most recent iteration.
1:
    
    {
      ld64 $ACTS_PAIR, $DATA_PTR, \act_base, (2 * \stride)
      f16v4mul $RESULTS_PAIR, $RESULTS_PAIR, $ACTS_PAIR
    }

    {
      st64step $RESULTS_PAIR, \act_base, $DATA_PTR+=, \stride
      f16v4mul $RESULTS_PAIR, $ACTS_PAIR, $ACTS_PAIR
    }

.Lhalf_loop_first:
    {
      nop
      f16v4mul $RESULTS_PAIR, $ACTS_PAIR, $RESULTS_PAIR
    }

.Lhalf_loop_last:
    // The instructions following the f16v4mix instruction operate on the 
    // inputs of the previous iteration.
    //
    // On the last iteration, $RESULTS_PAIR will be "dummy" values
    // which have no effect on the output of the function. This instruction is 
    // only to flush out the result of the last iteration into $RESULTS_PAIR.
    {
      nop
      f16v4mix $RESULTS_PAIR, $RESULTS_PAIR, $ACTS_PAIR
    }
    
    {
      nop
      f16v2tanh $RESULTS_0, $RESULTS_0
    }
    
    {
      nop
      f16v2tanh $RESULTS_1, $RESULTS_1
    }
    
    {
      ld64 $ACTS_PAIR, $DATA_PTR, \act_base, 0
      f16v4add $RESULTS_PAIR, $CONST_HI_1_0_LO_0_5:BU, $RESULTS_PAIR
    }
    
    {
      nop
      f16v4mul $RESULTS_PAIR, $CONST_HI_1_0_LO_0_5:BL, $RESULTS_PAIR
    }
    
2:
    // Execute 2nd iteration if \n_64bit is at least 2
    brnz $MSCRATCH2, .Lhalf_execute_rpt_block

    f16v4mul $RESULTS_PAIR, $RESULTS_PAIR, $ACTS_PAIR
    
    // Store for the last iteration of the repeat block as well as for the very
    // last iteration of \n_64bit
    st64step $RESULTS_PAIR, \act_base, $DATA_PTR+=, \stride

    // Use instructions in the repeat block for flushing out the pipeline.
    // Flush the mix instruction using zeros in $RESULTS_PAIR, to ensure that the output does not overflow.
    {
      brnzdec $MSCRATCH, .Lhalf_loop_last
      zero $RESULTS_PAIR
    }

.endm

#endif // __IPU__
