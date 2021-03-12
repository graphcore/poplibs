// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Zero  out partials before Nx1 workers start
// Uses same vertex state as the convolution workers
//
// Performance: 14 + (partials == HALF ? 1 : 0) + num_samples / 2
//
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_zero_output_stack.h"

// =============================================================================

.macro CONV_Nx1_ZERO_OUT_WORKER PARTIALS_TYPE

.ifc \PARTIALS_TYPE, float
  .equ LOG2_SIZEOF_OUT_ATOM, 2
.endif
.ifc \PARTIALS_TYPE, half
  .equ LOG2_SIZEOF_OUT_ATOM, 1
.endif

DEF_STACK_USAGE 0 convNx1ZeroOutField_\PARTIALS_TYPE\()
.section ".text.convNx1ZeroOutField_\PARTIALS_TYPE\()", FUNCTION_IS_WORKER
.global convNx1ZeroOutField_\PARTIALS_TYPE\()
.type convNx1ZeroOutField_\PARTIALS_TYPE\(), @function
.align 8

#define wkr_id_zv                       m0
#define zero_info_zv                    m1
#define zero_info_div_12_zv             m2
// Registers above must be retained between calls
#define outchan_ptr_zv                  m3

.ifc \PARTIALS_TYPE, float
nop // rpt alignment
.endif
convNx1ZeroOutField_\PARTIALS_TYPE\():
get           $wkr_id_zv, $WSR
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
ld32          $zero_info_zv, $mvertex_base, WKR_ZERO_INFO/4

shr           $zero_info_div_12_zv, $zero_info_zv, (3 - LOG2_SIZEOF_OUT_ATOM)

// For n with 0 <= n <= 65533 this does a division by 6 with the remainder
// split amongst workers.
add           $zero_info_div_12_zv, $zero_info_div_12_zv, 6
sub           $zero_info_div_12_zv, $zero_info_div_12_zv, $wkr_id_zv
mul           $zero_info_div_12_zv, $zero_info_div_12_zv, 21845
shr           $zero_info_div_12_zv, $zero_info_div_12_zv, 17

.ifc \PARTIALS_TYPE, half
// Divide by 2 followed by minus 1 below so we can quickly store the last 2
// elements below. We always have a multiple of 2 halves to store as this is
// guaranteed when planning all the convolutions that use this function.
shr           $zero_info_zv, $zero_info_zv, 1
.endif

// Minus 1 so we can quickly store the last element below
add           $zero_info_zv, $zero_info_zv, -1

.global convNx1ZeroOutFieldReentry_\PARTIALS_TYPE\()
.type convNx1ZeroOutFieldReentry_\PARTIALS_TYPE\(), @function
convNx1ZeroOutFieldReentry_\PARTIALS_TYPE\():
ld32          $outchan_ptr_zv, $mvertex_base, WKR_OUTCHAN_PTR/4
// Unconditionally write the last element in all the workers
st32          $azero, $outchan_ptr_zv, $mzero, $zero_info_zv
ld64step      $azeros, $mzero, $outchan_ptr_zv+=, $wkr_id_zv
rpt           $zero_info_div_12_zv, (Loop_end_zero_64_\PARTIALS_TYPE\() - Loop_start_zero_64_\PARTIALS_TYPE\())/8 - 1
Loop_start_zero_64_\PARTIALS_TYPE\():
  {
    st64step      $azeros, $mzero, $outchan_ptr_zv+=, 6
    fnop
  }
Loop_end_zero_64_\PARTIALS_TYPE\():
exitz         $mzero

.size convNx1ZeroOutField_\PARTIALS_TYPE\(), . - convNx1ZeroOutField_\PARTIALS_TYPE\()
.endm

// =============================================================================
// Instantiate codelets
// Workers
CONV_Nx1_ZERO_OUT_WORKER half
CONV_Nx1_ZERO_OUT_WORKER float

// =============================================================================
#endif
// =============================================================================
