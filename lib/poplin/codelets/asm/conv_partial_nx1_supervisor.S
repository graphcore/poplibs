// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Computes an nx1 convolution using AMP. A contiguous field is partitioned
// between workers for each position of the kernel element.
//
// Requires a total stack size of 80 bytes in the supervisor
//
#ifdef __IPU__
#ifndef __CONV_PARTIAL_NX1_SUPERVISOR_S__
#define __CONV_PARTIAL_NX1_SUPERVISOR_S__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_zero_output_stack.h"
#include "conv_sr_ctrl.h.S"

// =============================================================================

#define CODELET_NAME __runCodelet_poplin__ConvPartialnx1___\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_true_\LD128\()_\AMP_OUTPUTS\()_\INPUT_ELEMS\()_\DISABLE_SR\()

#define SYMBOL_SUFFIX \ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_true_\LD128\()_\AMP_OUTPUTS\()_\INPUT_ELEMS\()_\DISABLE_SR\()

// =============================================================================

// Worklist partition fields: must match the order of entries in worklists
#define PARTITION_ELEM_BYTES            2
#define PARTITION_OUTOFFSET             0    // short
#define PARTITION_NUM_ELEMS             2    // short
#define PARTITION_INOFFSET              4    // short
#define PARTITION_SIZE                 ((PARTITION_INOFFSET + PARTITION_ELEM_BYTES)/PARTITION_ELEM_BYTES)


// DeltaN decoding constants
#if defined(VECTORLIST_AVAIL_DELTAN)
#define DELTAN_OFFSET_BITS       18
#define DELTAN_COUNT_BITS        14
#define ELEM_TO_BYTES_CONV       0
#else
#define DELTAN_OFFSET_BITS       20
#define DELTAN_COUNT_BITS        (32 - DELTAN_OFFSET_BITS)
#define ELEM_TO_BYTES_CONV       (21 - DELTAN_OFFSET_BITS)
#endif

// =============================================================================


// Supervisor vertex state: offsets and the number must match vertex field
// ordering and sizes
#if defined(VECTORLIST_AVAIL_DELTAN)
#define SUP_INSTRIDE                    0    // word
#define SUP_OUTSTRIDE                   4    // word
#define SUP_INCHAN_VECTORS              8    // word
#define SUP_WEIGHTS_VECTORS             12   // word
#define SUP_OUTCHAN_VECTORS             16   // word
#define SUP_ZERO_INFO                   20   // word
#define SUP_PARTITION_TABLES            24   // VectorList::DELTA
#define SUP_NUM_OUTCHAN_M1              30   // short
#define SUP_NUM_INCHAN                  32   // short
#define SUP_NUM_KERNEL_Y_M1             34   // short (Kx-1)
#define SUP_NUM_KERNEL_X_M1             36   // short (Ky-1)
#define SUP_NUM_CONVGROUPS_M1           38   // short
#define SUP_NUM_FILTER_HEIGHT_M1        40   // short
#define SUP_IN_ROW_STRIDE               42   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      44   // short
#define SUP_NUM_INCHANS_PER_GROUP       46   // short

#else
#define SUP_INSTRIDE                    0    // word
#define SUP_OUTSTRIDE                   4    // word
#define SUP_INCHAN_VECTORS              8    // word
#define SUP_WEIGHTS_VECTORS             12   // word
#define SUP_OUTCHAN_VECTORS             16   // word
#define SUP_ZERO_INFO                   20   // word
#define SUP_PARTITION_TABLES            24   // VectorList::DeltaNElements
#define SUP_NUM_OUTCHAN_M1              32   // short
#define SUP_NUM_INCHAN                  34   // short
#define SUP_NUM_KERNEL_Y_M1             36   // short (Kx-1)
#define SUP_NUM_KERNEL_X_M1             38   // short (Ky-1)
#define SUP_NUM_CONVGROUPS_M1           40   // short
#define SUP_NUM_FILTER_HEIGHT_M1        42   // short
#define SUP_IN_ROW_STRIDE               44   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      46   // short
#define SUP_NUM_INCHANS_PER_GROUP       48   // short
#endif

// DeltaN decoding constants
#if defined(VECTORLIST_AVAIL_DELTAN)
#define SCALED_PTR32_SHL         2
#define DELTAN_DELTAS_COUNT_BITS 12
#else
#define DELTAN_DELTAS_COUNT_BITS 8
#endif

// =============================================================================
// Vertex state shared between workers
// The supervisor sets up a common state for workers to use
// Allocate codelet specific worker stack after zero output worker (WKR_ZERO_OUTPUT_STACK)
#define WKR_INOUTSTRIDES1               WKR_ZERO_OUTPUT_STACK       // word
#define WKR_INOUTSTRIDES2               (WKR_INOUTSTRIDES1 + 4)     // word
#define WKR_INCHAN_PTR                  (WKR_INOUTSTRIDES2 + 4)     // word
#define WKR_PARTITION_PTR               (WKR_INCHAN_PTR + 4)        // word
#define WKR_PARTITION_BASE              (WKR_PARTITION_PTR + 4)     // word
#define WKR_SECOND_PTR_OFFSET           (WKR_PARTITION_BASE + 4)    // word
#define WKR_STATE_SIZE                  (WKR_SECOND_PTR_OFFSET + 4) // bytes

// =============================================================================
// Performance:
// Kernel height dependent cycles in the innermost loop (i.e. AmpOutGroupLoop)
//   Kernel Height = 1: 15 + LOADCYLES + workerCycles
//   Kernel Height = 2: 46 + LOADCYLES + workerCycles
//   Kernel Height = 4: 41 + 19 + 32 + workerCycles
// 99 + numConvGroups * (16 + numInChanGroups * (14 + numOutChanGroups * (14 + innerLoopCycles)))
//
// Where innerLoopCycles are:
// Ky * (14 + Kx * (17 + AmpOutGroupLoop cycles ))
// for 8 AMPs engines:
// LOADCYLES = 16 (if LDTYPE = 128)
//           = 32 (if LDTYPE = 64)
// for 16 AMPs engines:
// LOADCYLES = 32 (if LDTYPE = 128)
//           = 64 (if LDTYPE = 64)
// STACK VARIABLES for supervisor
#define STACK_BASE                      WKR_STATE_SIZE
#define STACK_AMP_OUTSUBGROUP           0      // word
#define STACK_CONV_GROUP_COUNT          4      // word
#define STACK_OUTCHAN_COUNT             8      // word
#define STACK_INNER_LOOP_ADDR           12     // word
#define STACK_WEIGHTS_PTR_INCR          16     // word ptr increment in amp kernel height
#define STACK_PARTITION_PTR             20
#define STACK_KY_COUNT                  24
#define STACK_INCHAN_COUNT              28     // word
#define STACK_NUM_INCHAN                32
#define STACK_NUM_OUTCHAN_M1            36
#define STACK_NUM_KERNEL_X_M1           40
#define STACK_NUM_KERNEL_Y_M1           44
#define STACK_WKR_FUNCTION              48
#define STACK_FPICTL                    52
#define STACK_SIZE                      (STACK_FPICTL + 4) // bytes

#define TOT_STACK_SIZE                  (WKR_STATE_SIZE + 8 + STACK_SIZE)

// registers
#define sup_base                        m0
#define kx_count_s                      m0

#define fpictl_s                        m1
#define weights_ptr_h_incr              m1
#define temp_5_s                        m1
#define convgroup_count_s               m1
#define outchan_vectors_z_s             m1 // for zeroing partials

#define temp_1_s                        m2
#define inchan_count_s                  m2
#define partition_ptr_s                 m2
#define outchan_count_z_s               m2 // for zeroing partials

#define inchan_ptr_s                    m3
#define jmp_addr_s                      m3
#define temp_4_s                        m3
#define zero_info_s_z                   m3 // for zeroing partials

#define amp_out_subgroup_count_s        m4
#define ky_count_s                      m4
#define temp_7_s                        m4
#define wkr_function_s_z                m4 // for zeroing partials

#define wkr_function_s                  m5
#define weights_ptr_s_h                 m5
#define outchan_ptr_z_s                 m5 // for zeroing partials

#define outchan_count_s                 m5
#define amp_height_jump_s               m5
#define temp_3_s                        m5

#define outchan_ptr_s                   m6
#define partition_ptr_b_s               m6

#define weights_ptr_s                   m7
#define temp_2_s                        m7
#define outstride_s                     m7
#define convgroup_count_s_z             m7 // for zeroing partials

#define inchan_vectors_s                m8
#define in_row_stride_s                 m8
#define temp_6_s                        m8

#define outchan_vectors_s               m9
#define inchans_per_group_s             m9
#define amp_kernel_height_s             m9

#define temp_8_s                        m10
#define outchans_per_group_s            m10
#define weights_vectors_s               m10

#define wkr_base                        sp

.macro CONV_Nx1_SUPERVISOR ACTIVATIONS_TYPE PARTIALS_TYPE LD128 AMP_OUTPUTS INPUT_ELEMS DISABLE_SR COMMAND

.ifc \ACTIVATIONS_TYPE, half
    .equ SIZEOF_IN_ATOM,             2
    .equ AMP_INPUTS,                 16
    .equ SIZEOF_WEIGHTS_ATOM,        2
.endif
.ifc \ACTIVATIONS_TYPE, float
    .equ SIZEOF_IN_ATOM,             4
    .equ AMP_INPUTS,                 8
    .equ SIZEOF_WEIGHTS_ATOM,        4
.endif

.ifc \PARTIALS_TYPE, half
    .equ SIZEOF_OUT_ATOM,            2
    .equ LOG2_SIZEOF_OUT_ATOM,       1
.endif
.ifc \PARTIALS_TYPE, float
    .equ SIZEOF_OUT_ATOM,            4
    .equ LOG2_SIZEOF_OUT_ATOM,       2
.endif

.ifc \LD128, true
    .equ LDTYPE, 128
.else
    .equ LDTYPE, 64
.endif

.if \AMP_OUTPUTS == 8
    .equ LOG2_AMP_OUTPUTS,   3
.elseif \AMP_OUTPUTS == 16
    .equ LOG2_AMP_OUTPUTS,   4
.else
    .error "AMP output channels not supported"
.endif

DEF_STACK_USAGE  TOT_STACK_SIZE  CODELET_NAME
.section .text.CODELET_NAME
.globl CODELET_NAME
.type CODELET_NAME, @function
CODELET_NAME:

.supervisor
//-----------------------------------------------------------------------------
ld32          $temp_1_s, $sup_base, SUP_INSTRIDE/4
ld32          $outstride_s, $sup_base, SUP_OUTSTRIDE/4
ld32          $partition_ptr_b_s, $sup_base, SUP_PARTITION_TABLES/4
// space for worker vertex state, supervisor state and callee-save registers
add           $sp, $sp, -TOT_STACK_SIZE

// We don't care about the cost when SR is disabled or where SR doesn't matter
// (for example in the case of float data) as it is only provisioned for
// testing. Hence the code is bunched up for easy maintenance.
.ifc \DISABLE_SR, true
  SAVE_FPICTL_AND_DISABLE_SR  (STACK_BASE + STACK_FPICTL)/4  fpictl_s temp_3_s
.endif

lds16         $in_row_stride_s, $sup_base, SUP_IN_ROW_STRIDE/2
ld32          $zero_info_s_z, $sup_base, SUP_ZERO_INFO/4

//-----------------------------------------------------------------------------
setzi         $wkr_function_s, convPartialNx1Flattened_\COMMAND\()
ldz16         $temp_5_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
shl           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
ldz16         $temp_7_s, $sup_base, SUP_NUM_KERNEL_Y_M1/2
st32          $temp_1_s, $wkr_base, WKR_INOUTSTRIDES1/4
st32          $outstride_s, $wkr_base, WKR_INOUTSTRIDES2/4

//-----------------------------------------------------------------------------
st32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
st32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1
st32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
shr           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
st32          $temp_5_s, $sp, (STACK_BASE + STACK_NUM_OUTCHAN_M1)/4
st32          $temp_7_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_Y_M1)/4

//-----------------------------------------------------------------------------
st32          $in_row_stride_s, $wkr_base, WKR_SECOND_PTR_OFFSET/4
st32          $zero_info_s_z, $wkr_base, WKR_ZERO_INFO/4
ldz16         $outchans_per_group_s, $sup_base, SUP_NUM_OUTCHANS_PER_GROUP/2

// move pointer back by 1 partition as we address from the last partition down to
// the first in the worker. Moving the address down by 1 partition helps in
// offset addressing and using the partition offset as an effective counter in
// the partition loop.
sub           $partition_ptr_b_s, $partition_ptr_b_s, PARTITION_SIZE * PARTITION_ELEM_BYTES
// We could schedule this to avoid the penalty here but would require a lot
// of rescheduling
st32          $partition_ptr_b_s, $wkr_base, WKR_PARTITION_BASE/4

#if defined(VECTORLIST_AVAIL_DELTAN)
ldz16         $partition_ptr_b_s, $sup_base, (SUP_PARTITION_TABLES + 4)/2
#else
ld32          $partition_ptr_b_s, $sup_base, (SUP_PARTITION_TABLES + 4)/4
#endif

//ldz16         $inchans_per_group_s, $sup_base, SUP_NUM_INCHANS_PER_GROUP/2
ldz16         $amp_kernel_height_s, $sup_base, SUP_NUM_FILTER_HEIGHT_M1/2

//-----------------------------------------------------------------------------
nop
// Scale output and input channels per group to scale input and output offsets.
// Scaling depends on sizeof(input) and sizeof(output)
ld32          $outchan_vectors_z_s, $sup_base, SUP_OUTCHAN_VECTORS/4
shr           $temp_2_s, $outchans_per_group_s, LOG2_AMP_OUTPUTS
ldz16         $temp_6_s, $sup_base, SUP_NUM_KERNEL_X_M1/2
setzi         $wkr_function_s_z, convNx1ZeroOutField_\PARTIALS_TYPE\()

#if defined(VECTORLIST_AVAIL_DELTAN)
shl           $partition_ptr_b_s, $partition_ptr_b_s, (SCALED_PTR32_SHL + 13)
#else
shl           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
#endif

//-----------------------------------------------------------------------------
nop           // keep nop for 6 instructions pipeline

#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#else
ld32step      $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#endif

add           $temp_4_s, $temp_2_s, -1
st32          $temp_6_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_X_M1)/4
ldz16         $convgroup_count_s_z, $sup_base, SUP_NUM_CONVGROUPS_M1/2 // temp_2

#if defined(VECTORLIST_AVAIL_DELTAN)
or            $partition_ptr_b_s, $partition_ptr_b_s, (TMEM_REGION0_BASE_ADDR << 13)
#else
nop           // keep nop for 6 instructions pipeline
#endif

//-----------------------------------------------------------------------------
nop           // keep nop for 6 instructions pipeline
nop           // keep nop for 6 instructions pipeline
nop           // keep nop for 6 instructions pipeline
add           $temp_6_s, $temp_6_s, 1
nop           // keep nop for 6 instructions pipeline

#if defined(VECTORLIST_AVAIL_DELTAN)
shr           $partition_ptr_b_s, $partition_ptr_b_s, 13
#else
shr           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
#endif

//-----------------------------------------------------------------------------



ZeroConvGroup_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
  ldz16         $outchan_count_z_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
ZeroOutChanGroup_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    // expand scaled pointer
    shl           $outchan_ptr_z_s, $outchan_ptr_z_s, 3
#endif
    sync          TEXCH_SYNCZONE_LOCAL
    st32          $outchan_ptr_z_s, $wkr_base, WKR_OUTCHAN_PTR/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#else
    ld32step      $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#endif
    runall        $wkr_function_s_z, $wkr_base, 0
    setzi         $wkr_function_s_z, convNx1ZeroOutFieldReentry_\PARTIALS_TYPE\()
    brnzdec       $outchan_count_z_s, ZeroOutChanGroup_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
  brnzdec       $convgroup_count_s_z, ZeroConvGroup_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

mul           $temp_2_s, $temp_6_s, $outchans_per_group_s
st32          $temp_4_s, $sp, (STACK_BASE + STACK_AMP_OUTSUBGROUP)/4
st32          $partition_ptr_b_s, $sp, (STACK_BASE + STACK_PARTITION_PTR)/4
// Jump address for amp height == 1 is set to 0 to take the fast path
setzi         $jmp_addr_s, 0

brz           $amp_kernel_height_s, JmpAddrSelEnd_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
mul           $temp_6_s, $temp_2_s, AMP_INPUTS / 2 * SIZEOF_WEIGHTS_ATOM
setzi         $jmp_addr_s, AmpHeightEq2_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
// a bubble here instead of using 3 nop instructions
add           $amp_kernel_height_s, $amp_kernel_height_s, -1
brz           $amp_kernel_height_s, JmpAddrSelEnd_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
setzi         $jmp_addr_s, AmpHeightEq4_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
mul           $temp_6_s, $temp_2_s, AMP_INPUTS / 4 * SIZEOF_WEIGHTS_ATOM
JmpAddrSelEnd_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():

// The only call to this function may be zero out partials
ldz16         $inchan_count_s, $mzero, $sup_base, SUP_NUM_INCHAN/2
ldz16         $convgroup_count_s, $sup_base, SUP_NUM_CONVGROUPS_M1/2
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#else
ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
mov           $outchan_count_s, $mzero
st32          $jmp_addr_s, $sp, (STACK_BASE + STACK_INNER_LOOP_ADDR)/4
st32          $temp_6_s, $sp, (STACK_BASE + STACK_WEIGHTS_PTR_INCR) / 4
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
st32          $inchan_count_s, $sp, (STACK_BASE + STACK_NUM_INCHAN)/4

brz           $inchan_count_s,  L_sup_conv_end_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
add           $inchan_count_s, $inchan_count_s, -1
ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif

ConvGroupLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
  // Output channel vectors are increased by numOutChanGroups for each
  // convolution group. The increment at the start is 0. The increment
  // of numOutChanGroups is done before branching here for each conv group.
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $mzero, $mzero, $outchan_vectors_s+=, $outchan_count_s
#else
  ld32step      $mzero, $mzero, $outchan_vectors_s+=, $outchan_count_s
#endif
  ld32          $outchan_count_s, $sp, (STACK_BASE +  STACK_NUM_OUTCHAN_M1)/4
  st32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4

InChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
    st32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    shl           $inchan_ptr_s, $inchan_ptr_s, 3
#else
    ld32step      $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif
OutChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      shl           $weights_ptr_s, $weights_ptr_s, 3
      nop
      nop
#endif

      ld32          $partition_ptr_s, $sp, (STACK_BASE + STACK_PARTITION_PTR)/4
      // This load of CCCS is for the case of Amp Kernel Height = 1. In the
      // other two cases, the weights are not loaded in order
      st32          $outchan_count_s, $sp, (STACK_BASE + STACK_OUTCHAN_COUNT)/4
      ld32          $ky_count_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_Y_M1)/4
      put           $CCCSLOAD, $weights_ptr_s

KyLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
        st32          $ky_count_s, $sp, (STACK_BASE + STACK_KY_COUNT)/4
        ld32          $kx_count_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_X_M1)/4
KxLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
          // expand scaled pointer
          ldz16         $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
          shl           $outchan_ptr_s, $outchan_ptr_s, 3
#else
          ld32          $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
#endif
          ld32          $amp_out_subgroup_count_s, $sp, (STACK_BASE + STACK_AMP_OUTSUBGROUP)/4
          ld32          $weights_ptr_h_incr, $sp, (STACK_BASE + STACK_WEIGHTS_PTR_INCR)/4
AmpOutGroupLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
            // registers $m1 and $m8 are free to use in this function as long
            // as wkr_function_s is set before the run
            ld32          $amp_height_jump_s, $sp, (STACK_BASE + STACK_INNER_LOOP_ADDR)/4
            sync          TEXCH_SYNCZONE_LOCAL
            st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4
            st32          $outchan_ptr_s, $wkr_base, WKR_OUTCHAN_PTR/4
            st32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
            // Different AMP kernel heights have different coefficient loading
            // schemes and they all branch back to AmpHeightRun
            // If there were an brnz instruction with target branch
            // address as a register rather than immediate, for the fallthrough
            // case we could have set the condition to check fail (eg using
            // brnz would become brnz $amp_height_jump_s, $amp_height_jump_s)
            brnz          $amp_height_jump_s, AmpHeightJumpToNeq1_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

AmpHeightEq1_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
            LOAD_WEIGHTS_NX1_AMP_HEIGHT_1 LDTYPE \AMP_OUTPUTS \COMMAND

AmpHeightRun_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
            runall        $wkr_function_s, $wkr_base, 0

            // wind pointer to point to next amp subgroup
            add           $outchan_ptr_s, $outchan_ptr_s, SIZEOF_OUT_ATOM * \AMP_OUTPUTS
            brnzdec       $amp_out_subgroup_count_s, AmpOutGroupLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

          add           $partition_ptr_s, $partition_ptr_s, 6 * 4
          ld32          $outchan_count_s, $sp, (STACK_BASE + STACK_OUTCHAN_COUNT)/4
          brnzdec       $kx_count_s, KxLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

        get           $weights_ptr_s, $CCCSLOAD
        ld32          $ky_count_s, $sp, (STACK_BASE + STACK_KY_COUNT)/4
        brnzdec       $ky_count_s, KyLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#else
      ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
      brnzdec       $outchan_count_s, OutChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
    ld32          $outchan_count_s, $sp, (STACK_BASE + STACK_NUM_OUTCHAN_M1)/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif
    ld32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
    brnzdec       $inchan_count_s, InChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
  ld32          $inchan_count_s, $sp, (STACK_BASE + STACK_NUM_INCHAN)/4
  ld32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4
  // cannot use delay instruction here because workers may be active
  nop
  nop
  nop
  add           $outchan_count_s, $outchan_count_s, 1
  add           $inchan_count_s, $inchan_count_s, -1
  brnzdec       $convgroup_count_s, ConvGroupLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

L_sup_conv_end_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():

// Restore saved registers

.ifc \DISABLE_SR, true
  RESTORE_FPICTL  (STACK_BASE + STACK_FPICTL)/4 fpictl_s
.endif

ld32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
ld32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1
add           $sp, $sp, TOT_STACK_SIZE
sync          TEXCH_SYNCZONE_LOCAL
br            $lr

// Jump to code fragment to handle kernel heights = 2 and 4
AmpHeightJumpToNeq1_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
br            $amp_height_jump_s

// =============================================================================

// Code fragement to load CWEI with coefficients when the weights tensor
// innermost dimensions are [AMP_OUTPUTS * m][AMP_INPUTS/2]
AmpHeightEq2_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
put           $CCCSLOAD, $weights_ptr_s
add           $weights_ptr_s_h, $weights_ptr_s, $weights_ptr_h_incr

LOAD_WEIGHTS_NX1_AMP_HEIGHT_2 LDTYPE \AMP_OUTPUTS \COMMAND

// update weight pointer by the amount of coefficient
add           $weights_ptr_s, $weights_ptr_s, \AMP_OUTPUTS * SIZEOF_WEIGHTS_ATOM * AMP_INPUTS / 2
bri           AmpHeightRun_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

// =============================================================================

// Code fragement to load CWEI with coefficients when the weights tensor
// innermost dimensions are [AMP_OUTPUTS * m][AMP_INPUTS/4]
AmpHeightEq4_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
// we reuse $inchan_ptr_s as it has already been saved to vertex
// state

setzi         $inchan_ptr_s, convPartialNx1FlattenedStateRetained_\COMMAND\()

.if \AMP_OUTPUTS == 8
put           $CCCSLOAD, $weights_ptr_s
add           $weights_ptr_s_h, $weights_ptr_s, $weights_ptr_h_incr
ld64putcs     0
ld64putcs     4
ld64putcs     8
ld64putcs     12
ld64putcs     16
ld64putcs     20
ld64putcs     24
ld64putcs     28

put           $CCCSLOAD, $weights_ptr_s_h
add           $weights_ptr_s_h, $weights_ptr_s_h, $weights_ptr_h_incr
ld64putcs     1
ld64putcs     5
ld64putcs     9
ld64putcs     13
ld64putcs     17
ld64putcs     21
ld64putcs     25
ld64putcs     29

put           $CCCSLOAD, $weights_ptr_s_h
add           $weights_ptr_s_h, $weights_ptr_s_h, $weights_ptr_h_incr
ld64putcs     2
ld64putcs     6
ld64putcs     10
ld64putcs     14
ld64putcs     18
ld64putcs     22
ld64putcs     26
ld64putcs     30

put           $CCCSLOAD, $weights_ptr_s_h
ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
ld64putcs     3
ld64putcs     7
ld64putcs     11
ld64putcs     15
ld64putcs     19
ld64putcs     23
ld64putcs     27
ld64putcs     31


.elseif \AMP_OUTPUTS == 16
put           $CCCSLOAD, $weights_ptr_s
add           $weights_ptr_s_h, $weights_ptr_s, $weights_ptr_h_incr
ld64putcs     0
ld64putcs     4
ld64putcs     32
ld64putcs     36
ld64putcs     8
ld64putcs     12
ld64putcs     40
ld64putcs     44
ld64putcs     16
ld64putcs     20
ld64putcs     48
ld64putcs     52
ld64putcs     24
ld64putcs     28
ld64putcs     56
ld64putcs     60

put           $CCCSLOAD, $weights_ptr_s_h
add           $weights_ptr_s_h, $weights_ptr_s_h, $weights_ptr_h_incr
ld64putcs     1
ld64putcs     5
ld64putcs     33
ld64putcs     37
ld64putcs     9
ld64putcs     13
ld64putcs     41
ld64putcs     45
ld64putcs     17
ld64putcs     21
ld64putcs     49
ld64putcs     53
ld64putcs     25
ld64putcs     29
ld64putcs     57
ld64putcs     61

put           $CCCSLOAD, $weights_ptr_s_h
add           $weights_ptr_s_h, $weights_ptr_s_h, $weights_ptr_h_incr
ld64putcs     2
ld64putcs     6
ld64putcs     34
ld64putcs     38
ld64putcs     10
ld64putcs     14
ld64putcs     42
ld64putcs     46
ld64putcs     18
ld64putcs     22
ld64putcs     50
ld64putcs     54
ld64putcs     26
ld64putcs     30
ld64putcs     58
ld64putcs     62

put           $CCCSLOAD, $weights_ptr_s_h
ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
ld64putcs     3
ld64putcs     7
ld64putcs     35
ld64putcs     39
ld64putcs     11
ld64putcs     15
ld64putcs     43
ld64putcs     47
ld64putcs     19
ld64putcs     23
ld64putcs     51
ld64putcs     55
ld64putcs     27
ld64putcs     31
ld64putcs     59
ld64putcs     63

.else
.error "Number of outputs are not supported"
.endif
// update weight pointer by the number of coefficients used up as the next time
add           $weights_ptr_s, $weights_ptr_s, \AMP_OUTPUTS\() * SIZEOF_WEIGHTS_ATOM * AMP_INPUTS / 4
bri           AmpHeightRun_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

.size CODELET_NAME, . - CODELET_NAME
.endm



// =============================================================================
// Macro to load weights for Nx1 kernel when AMP Height is 1

.macro LOAD_WEIGHTS_NX1_AMP_HEIGHT_1 LDTYPE NUM_ENGINES COMMAND
            // we reuse $inchan_ptr_s as it has already been saved to vertex
            // state
            setzi         $inchan_ptr_s, convPartialNx1FlattenedStateRetained_\COMMAND\()
.if \LDTYPE == 128 && \NUM_ENGINES == 8
            ld128putcs    0
            ld128putcs    2
            ld128putcs    4
            ld128putcs    6
            ld128putcs    8
            ld128putcs    10
            ld128putcs    12
            ld128putcs    14
            ld128putcs    16
            ld128putcs    18
            ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
            ld128putcs    20
            ld128putcs    22
            ld128putcs    24
            ld128putcs    26
            ld128putcs    28
            ld128putcs    30

.elseif \LDTYPE == 128 && \NUM_ENGINES == 16
            ld128putcs    0
            ld128putcs    2
            ld128putcs    4
            ld128putcs    6
            ld128putcs    32
            ld128putcs    34
            ld128putcs    36
            ld128putcs    38
            ld128putcs    8
            ld128putcs    10
            ld128putcs    12
            ld128putcs    14
            ld128putcs    40
            ld128putcs    42
            ld128putcs    44
            ld128putcs    46
            ld128putcs    16
            ld128putcs    18
            ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
            ld128putcs    20
            ld128putcs    22
            ld128putcs    48
            ld128putcs    50
            ld128putcs    52
            ld128putcs    54
            ld128putcs    24
            ld128putcs    26
            ld128putcs    28
            ld128putcs    30
            ld128putcs    56
            ld128putcs    58
            ld128putcs    60
            ld128putcs    62

.elseif \LDTYPE == 64 && \NUM_ENGINES == 8
            ld64putcs    0
            ld64putcs    1
            ld64putcs    2
            ld64putcs    3
            ld64putcs    4
            ld64putcs    5
            ld64putcs    6
            ld64putcs    7
            ld64putcs    8
            ld64putcs    9
            ld64putcs    10
            ld64putcs    11
            ld64putcs    12
            ld64putcs    13
            ld64putcs    14
            ld64putcs    15
            ld64putcs    16
            ld64putcs    17
            ld64putcs    18
            ld64putcs    19
            ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
            ld64putcs    20
            ld64putcs    21
            ld64putcs    22
            ld64putcs    23
            ld64putcs    24
            ld64putcs    25
            ld64putcs    26
            ld64putcs    27
            ld64putcs    28
            ld64putcs    29
            ld64putcs    30
            ld64putcs    31

.elseif \LDTYPE == 64 && \NUM_ENGINES == 16
            ld64putcs    0
            ld64putcs    1
            ld64putcs    2
            ld64putcs    3
            ld64putcs    4
            ld64putcs    5
            ld64putcs    6
            ld64putcs    7
            ld64putcs    32
            ld64putcs    33
            ld64putcs    34
            ld64putcs    35
            ld64putcs    36
            ld64putcs    37
            ld64putcs    38
            ld64putcs    39
            ld64putcs    8
            ld64putcs    9
            ld64putcs    10
            ld64putcs    11
            ld64putcs    12
            ld64putcs    13
            ld64putcs    14
            ld64putcs    15
            ld64putcs    40
            ld64putcs    41
            ld64putcs    42
            ld64putcs    43
            ld64putcs    44
            ld64putcs    45
            ld64putcs    46
            ld64putcs    47
            ld64putcs    16
            ld64putcs    17
            ld64putcs    18
            ld64putcs    19
            ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
            ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
            ld64putcs    20
            ld64putcs    21
            ld64putcs    22
            ld64putcs    23
            ld64putcs    48
            ld64putcs    49
            ld64putcs    50
            ld64putcs    51
            ld64putcs    52
            ld64putcs    53
            ld64putcs    54
            ld64putcs    55
            ld64putcs    24
            ld64putcs    25
            ld64putcs    26
            ld64putcs    27
            ld64putcs    28
            ld64putcs    29
            ld64putcs    30
            ld64putcs    31
            ld64putcs    56
            ld64putcs    57
            ld64putcs    58
            ld64putcs    59
            ld64putcs    60
            ld64putcs    61
            ld64putcs    62
            ld64putcs    63
.else
// Error if LDTYPE is not supported
.error "Load type not supported"
.endif
.endm // LOAD_WEIGHTS_NX1_AMP_HEIGHT_1


// =============================================================================
// Macro to load weights for Nx1 kernel when AMP Height is 2

.macro LOAD_WEIGHTS_NX1_AMP_HEIGHT_2 LDTYPE NUM_ENGINES COMMAND
// we reuse $inchan_ptr_s as it has already been saved to vertex
// state
setzi         $inchan_ptr_s, convPartialNx1FlattenedStateRetained_\COMMAND\()
.if \LDTYPE == 128 && \NUM_ENGINES == 8
ld128putcs    0
ld128putcs    4
ld128putcs    8
ld128putcs    12
ld128putcs    16
ld128putcs    20
ld128putcs    24
ld128putcs    28
put           $CCCSLOAD, $weights_ptr_s_h
ld128putcs    2
ld128putcs    6
ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
ld128putcs    10
ld128putcs    14
ld128putcs    18
ld128putcs    22
ld128putcs    26
ld128putcs    30

.elseif \LDTYPE == 128 && \NUM_ENGINES == 16
ld128putcs    0
ld128putcs    4
ld128putcs    32
ld128putcs    36
ld128putcs    8
ld128putcs    12
ld128putcs    40
ld128putcs    44
ld128putcs    16
ld128putcs    20
ld128putcs    48
ld128putcs    52
ld128putcs    24
ld128putcs    28
ld128putcs    56
ld128putcs    60
put           $CCCSLOAD, $weights_ptr_s_h
ld128putcs    2
ld128putcs    6
ld128putcs    34
ld128putcs    38
ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
ld128putcs    10
ld128putcs    14
ld128putcs    42
ld128putcs    46
ld128putcs    18
ld128putcs    22
ld128putcs    50
ld128putcs    54
ld128putcs    26
ld128putcs    30
ld128putcs    58
ld128putcs    62

.elseif \LDTYPE == 64 && \NUM_ENGINES == 8
ld64putcs     0
ld64putcs     1
ld64putcs     4
ld64putcs     5
ld64putcs     8
ld64putcs     9
ld64putcs     12
ld64putcs     13
ld64putcs     16
ld64putcs     17
ld64putcs     20
ld64putcs     21
ld64putcs     24
ld64putcs     25
ld64putcs     28
ld64putcs     29
put           $CCCSLOAD, $weights_ptr_s_h
ld64putcs     2
ld64putcs     3
ld64putcs     6
ld64putcs     7
ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
ld64putcs     10
ld64putcs     11
ld64putcs     14
ld64putcs     15
ld64putcs     18
ld64putcs     19
ld64putcs     22
ld64putcs     23
ld64putcs     26
ld64putcs     27
ld64putcs     30
ld64putcs     31

.elseif \LDTYPE == 64 && \NUM_ENGINES == 16
ld64putcs     0
ld64putcs     1
ld64putcs     4
ld64putcs     5
ld64putcs     32
ld64putcs     33
ld64putcs     36
ld64putcs     37
ld64putcs     8
ld64putcs     9
ld64putcs     12
ld64putcs     13
ld64putcs     40
ld64putcs     41
ld64putcs     44
ld64putcs     45
ld64putcs     16
ld64putcs     17
ld64putcs     20
ld64putcs     21
ld64putcs     48
ld64putcs     49
ld64putcs     52
ld64putcs     53
ld64putcs     24
ld64putcs     25
ld64putcs     28
ld64putcs     29
ld64putcs     56
ld64putcs     57
ld64putcs     60
ld64putcs     61
put           $CCCSLOAD, $weights_ptr_s_h
ld64putcs     2
ld64putcs     3
ld64putcs     6
ld64putcs     7
ld64putcs     34
ld64putcs     35
ld64putcs     38
ld64putcs     39
ld32          $wkr_function_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
st32          $inchan_ptr_s, $sp, (STACK_BASE + STACK_WKR_FUNCTION)/4
ld32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
ld64putcs     10
ld64putcs     11
ld64putcs     14
ld64putcs     15
ld64putcs     42
ld64putcs     43
ld64putcs     46
ld64putcs     47
ld64putcs     18
ld64putcs     19
ld64putcs     22
ld64putcs     23
ld64putcs     50
ld64putcs     51
ld64putcs     54
ld64putcs     55
ld64putcs     26
ld64putcs     27
ld64putcs     30
ld64putcs     31
ld64putcs     58
ld64putcs     59
ld64putcs     62
ld64putcs     63
.else
// Only 128 and 64 bit loads supported
.error "Load type not supported"
.endif
.endm // LOAD_WEIGHTS_NX1_AMP_HEIGHT_2


// =============================================================================
#endif // __CONV_PARTIAL_NX1_SUPERVISOR_S__
#endif // __IPU__
// =============================================================================
