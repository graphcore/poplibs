// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Common header to enable/disable SR in supervisor

#ifndef conv_sr_ctrl_h_S_
#define conv_sr_ctrl_h_S_
#ifdef __IPU__

// To be used in supervisor. Save $FP_ICTL CSR to a given offset in stack
// and disable Stochastic rounding.
.macro SAVE_FPICTL_AND_DISABLE_SR OFFSET_IN_STACK, CTRL_REG, MASK
  get           $\CTRL_REG, CSR_S_FP_ICTL__INDEX
  st32          $\CTRL_REG, $sp, \OFFSET_IN_STACK
  setzi         $\MASK, (CSR_S_FP_ICTL__ESR__MASK << CSR_S_FP_ICTL__ESR__SHIFT)
  andc          $\CTRL_REG,  $\CTRL_REG, $\MASK
  put           CSR_S_FP_ICTL__INDEX, $\CTRL_REG 
.endm

// To be used in supervisor. Restore $FP_ICTL from register value in stack.
.macro RESTORE_FPICTL OFFSET_IN_STACK, TEMP_REG
  ld32          $\TEMP_REG, $sp, \OFFSET_IN_STACK
  put           CSR_S_FP_ICTL__INDEX, $\TEMP_REG 
.endm


// =============================================================================

#endif // #ifdef __IPU__
#endif // #ifdef _conv_sr_ctrl_h_S_

// =============================================================================
// =============================================================================
