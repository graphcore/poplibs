// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
//
// Contains functions to calculate partials for convolution. Partials and Output
// are used interchangeably in this file. Each worker may process part of a
// contiguous field. This is  done by setting up a partition which contains an
// input offset in the input channel group, an output offset in the output field
// and the number of output field elements to process. Both input and output
// may be strided and the output flipped
//


#ifdef __IPU__
#ifndef __CONV_PARTIAL_1X1_SUPERVISOR_S__
#define __CONV_PARTIAL_1X1_SUPERVISOR_S__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_sr_ctrl.h.S"

// =============================================================================
#define CODELET_NAME __runCodelet_poplin__ConvPartial1x1Out___\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_true_\LD128\()_\AMP_OUTPUTS\()_\INPUT_ELEMS\()_\DISABLE_SR

// =============================================================================

//// Supervisor vertex state
////
// Pointer to input channel group deltas
#define SUP_INCHAN_VECTORS        0  // one_ptr, unsigned
// Pointer to weights vectors
#define SUP_WEIGHTS_VECTORS       4  // one_ptr, unsigned
// Pointer to partials
#define SUP_OUTCHAN_VECTORS       8  // one_ptr, unsigned
// Pointer to worker partition table
#define SUP_PARTITION             12  // one_ptr, unsigned
// Number of convolution groups
#define SUP_NUM_CONV_GROUPS_M1    16  // short
// Number of contiguous output channel group fields. Value is 1 less.
#define SUP_NUM_OUTCHAN_GROUPS_M1 18  // short
// Number of contiguous input channel group fields
#define SUP_NUM_INCHAN_GROUPS     20  // short
// input and output strides
#define SUP_INPUT_STRIDE          22  // short
// number of output channels per group
#define SUP_OUTCHANS_PER_GROUP    24  // short
// Processed out stride: the value depends on whether output is flipped or
// not. This vertex does not actually support striding but this field is used
// to directly create a stride for the flipped output
#define SUP_OUTSTRIDE             26 // short


// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////
// Pointer to input channel group field
#define WKR_INCHAN_PTR            0    // word
// Pointer to output/partial channel group field
#define WKR_OUTCHAN_PTR           4    // word
// Input stride
#define WKR_IN_OUT_STRIDES        8   // word
// Gap to be left in bytes after all partials are written for a field position
#define WKR_PARTITION             12
#define WKR_VERTEX_SIZE           16   // bytes

// =============================================================================

// supervisor base is $m0 - passed to this function
#define sup_base                  m0
#define stride_s                  m1
#define conv_group_s              m1
#define invec_s                   m2
#define outvec_s                  m2
#define inchan_group_s            m3
#define partition_s               m3
#define fpictl_s                  m4
#define wkr_function              m4
#define wkr_function_post         m5
#define amp_group_s               m5
#define add_zero_in_stride_s      m5
#define outchan_group_s           m6
#define temp_s                    m6
#define outstride_s               m6
#define const0x80000000           m7
#define weight_vec_s              m7
#define outchan_vectors_s         m10
#define inchan_vectors_s          m9
#define weights_vectors_s         m8
#define wkr_vertex                sp

#define SUP_STACK_CALLEE_SAVE0    0  //(word)
#define SUP_STACK_CALLEE_SAVE1    4  //(word)
#define SUP_STACK_CONV_GROUP      8  //(word)
// The number of output channels per group is divided by the number of number
// of channels the AMP processes. The actual value kept is one less.
#define SUP_STACK_AMP_GROUPS      12 //(word)
#define SUP_WKR_FUNCTION          16 //(word)
#define SUP_FPICTL                20 //(word)
#define SUP_STACK_SIZE            (SUP_FPICTL + 4)
#define TOT_STACK_SIZE            (WKR_VERTEX_SIZE + SUP_STACK_SIZE)

// ZERO_PARTIALS_PATH : set to true if zero path exists in the worker function

.macro CONV_1x1_SUPERVISOR ACTIVATIONS_TYPE PARTIALS_TYPE LD128 AMP_OUTPUTS INPUT_ELEMS COMMAND ZERO_PARTIALS_PATH DISABLE_SR

.ifc \ACTIVATIONS_TYPE, half
    .equ SIZE_OF_ACTIVATIONS,        2
    .equ NUM_INCHAN_GROUPS,         16
.endif
.ifc \ACTIVATIONS_TYPE, float
    .equ SIZE_OF_ACTIVATIONS,        4
    .equ NUM_INCHAN_GROUPS,          8
.endif

.ifc \PARTIALS_TYPE, half
    .equ SIZE_OF_PARTIALS_TYPE,      2
    .equ LOG2_SIZE_OF_PARTIALS_TYPE, 1
    .equ OUTSTRIDE_TO_LOADS,         2
.endif
.ifc \PARTIALS_TYPE, float
    .equ SIZE_OF_PARTIALS_TYPE,      4
    .equ LOG2_SIZE_OF_PARTIALS_TYPE, 2
    .equ OUTSTRIDE_TO_LOADS,         1
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
// Performance:
// 50 + numConvGroups * (12 +
//                       inChanGroups * (15 + inChanOverhead +
//                                       outChanGroups * (19 +
//                                                        AmpGroups * 16 + LOADCYCLES))))
// Where AMP groups = OutChansPerGroup / (8 or 16)
// and number of weights load depending on AMP engines:
// for 8 AMP engines:
// LOADCYLES = 16 (if LDTYPE = 128)
//           = 32 (if LDTYPE = 64)
// for 16 AMP engines:
// LOADCYLES = 32 (if LDTYPE = 128)
//           = 64 (if LDTYPE = 64)
//
// inChanOverhead = 0 if worker state is not retained
//                = 1 if worker state is retained
// ----------------------------------------------------------------------------
  ld32          $partition_s, $sup_base, SUP_PARTITION/4
  add           $sp, $sp, -TOT_STACK_SIZE

// We don't care about the cost when SR is disabled or where SR doesn't matter
// (for example in the case of float data) as it is only provisioned for
// testing. Hence the code is bunched up for easy maintenance.
.ifc \DISABLE_SR, true
  SAVE_FPICTL_AND_DISABLE_SR (WKR_VERTEX_SIZE/4 + SUP_FPICTL/4) fpictl_s temp_s
.endif

  lds16         $stride_s, $sup_base, SUP_INPUT_STRIDE/2
  ldz16         $amp_group_s, $sup_base, SUP_OUTCHANS_PER_GROUP/2
  lds16         $outstride_s, $sup_base, SUP_OUTSTRIDE/2
// ----------------------------------------------------------------------------
  nop
  ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
  st32          $m9, $sp, WKR_VERTEX_SIZE/4 + SUP_STACK_CALLEE_SAVE0/4
  st32          $m10, $sp, WKR_VERTEX_SIZE/4 + SUP_STACK_CALLEE_SAVE1/4
  shr           $amp_group_s, $amp_group_s, LOG2_AMP_OUTPUTS
  shr           $outstride_s, $outstride_s, OUTSTRIDE_TO_LOADS
// ----------------------------------------------------------------------------
  ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
  ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
  nop           // keep nop for 6 instructions pipeline
  nop           // keep nop for 6 instructions pipeline
  nop
  and           $temp_s, $outstride_s, 0x3FF
// ----------------------------------------------------------------------------
  nop
  nop           // keep nop for 6 instructions pipeline
  st32          $partition_s, $sp, WKR_PARTITION/4
  nop           // keep nop for 6 instructions pipeline
  ldz16         $inchan_group_s, $mzero, $sup_base, SUP_NUM_INCHAN_GROUPS/2
  shl           $temp_s, $temp_s, 10
// ----------------------------------------------------------------------------
  add           $amp_group_s, $amp_group_s, -1
  nop           // keep nop for 6 instructions pipeline
  nop           // keep nop for 6 instructions pipeline
  nop
  ld32step      $weight_vec_s, $mzero, $weights_vectors_s+=, 1
  or            $stride_s, $stride_s, $temp_s
// ----------------------------------------------------------------------------
  st32          $amp_group_s, $sp, WKR_VERTEX_SIZE/4 + SUP_STACK_AMP_GROUPS/4
  ld32step      $invec_s, $mzero, $inchan_vectors_s+=, 1
  nop           // keep nop for 6 instructions pipeline
  nop
  nop           // keep nop for 6 instructions pipeline
  st32          $stride_s, $wkr_vertex, WKR_IN_OUT_STRIDES/4
// ----------------------------------------------------------------------------
.ifc \ZERO_PARTIALS_PATH, true
  setzi         $wkr_function_post, convPartialFlattenedFieldStateRetainedInChanPtr_\COMMAND\()_Z
.else
  setzi         $wkr_function, convPartialFlattenedField_\COMMAND\()
.endif
  ldz16         $conv_group_s, $sup_base, SUP_NUM_CONV_GROUPS_M1/2
  ldz16         $outchan_group_s, $mzero, $sup_base, SUP_NUM_OUTCHAN_GROUPS_M1/2
  nop           // keep nop for 6 instructions pipeline
  put           $CCCSLOAD, $weight_vec_s
.ifc \ZERO_PARTIALS_PATH, true
  setzi         $wkr_function, convPartialFlattenedField_\COMMAND\()_Z
.else
  nop
.endif
// ----------------------------------------------------------------------------

convGroupsLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
.ifc \ZERO_PARTIALS_PATH, true
  st32          $wkr_function_post, $sp, WKR_VERTEX_SIZE/4 + SUP_WKR_FUNCTION/4
.endif
  add           $inchan_group_s, $inchan_group_s, -1

inChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
    st32          $invec_s, $wkr_vertex,  WKR_INCHAN_PTR/4

outChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
      ld32step     $weight_vec_s, $mzero, $weights_vectors_s+=, 1
      ld32         $outvec_s, $mzero, $outchan_vectors_s, $outchan_group_s
      ld32         $amp_group_s, $sp, WKR_VERTEX_SIZE/4 + SUP_STACK_AMP_GROUPS/4

AmpGroupLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\():
        // must wait for all workers to load CWEI as it is shared
        sync          TEXCH_SYNCZONE_LOCAL

        // load weights for 1x1 kernel
        LOAD_WEIGHTS_1X1 LDTYPE \AMP_OUTPUTS

        add           $outvec_s, $outvec_s, \AMP_OUTPUTS * SIZE_OF_PARTIALS_TYPE
        runall        $wkr_function, $wkr_vertex, 0
.ifc \ZERO_PARTIALS_PATH, true
        ld32          $wkr_function, $sp, WKR_VERTEX_SIZE/4 + SUP_WKR_FUNCTION/4
.else
        setzi         $wkr_function, convPartialFlattenedFieldStateRetainedInChanPtr_\COMMAND\()
.endif
        // increment output vector pointer for the AMP loop
        brnzdec $amp_group_s, AmpGroupLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
.ifc \ZERO_PARTIALS_PATH, false
      ld32          $add_zero_in_stride_s, $wkr_vertex, WKR_IN_OUT_STRIDES/4
.endif
      put           $CCCSLOAD, $weight_vec_s
      brnzdec       $outchan_group_s, outChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
.ifc \ZERO_PARTIALS_PATH, true
    setzi         $wkr_function_post, convPartialFlattenedFieldStateRetainedInChanPtr_\COMMAND\()
.else
    // Use the MSB of stride which is unused to flag zeroing of partials
    or            $const0x80000000, $mzero, 0x80000000
    or            $add_zero_in_stride_s, $add_zero_in_stride_s, 0x80000000
.endif
    ld32step      $invec_s, $mzero, $inchan_vectors_s+=, 1
    ldz16         $outchan_group_s, $mzero, $sup_base, SUP_NUM_OUTCHAN_GROUPS_M1/2
    // cannot write to worker vertex state until all workers have finished
    // processing the output channel group
    sync          TEXCH_SYNCZONE_LOCAL
.ifc \ZERO_PARTIALS_PATH, false
    st32          $add_zero_in_stride_s, $wkr_vertex, WKR_IN_OUT_STRIDES/4
    setzi         $wkr_function, convPartialFlattenedFieldStateRetained_\COMMAND\()
.endif
.ifc \ZERO_PARTIALS_PATH, true
    setzi         $wkr_function, convPartialFlattenedFieldStateRetained_\COMMAND\()
    nop
    st32          $wkr_function_post, $sp, WKR_VERTEX_SIZE/4 + SUP_WKR_FUNCTION/4
.endif
    brnzdec       $inchan_group_s, inChanLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()
.ifc \ZERO_PARTIALS_PATH, false
  // This should clear the msb
  xor           $add_zero_in_stride_s, $add_zero_in_stride_s, $const0x80000000
.endif
  ld32step      $mzero, $mzero, $outchan_vectors_s+=, $outchan_group_s
  add           $outchan_vectors_s, $outchan_vectors_s, 4
  ldz16         $inchan_group_s, $mzero, $sup_base, SUP_NUM_INCHAN_GROUPS/2
.ifc \ZERO_PARTIALS_PATH, true
  setzi         $wkr_function_post, convPartialFlattenedFieldStateRetainedInChanPtr_\COMMAND\()_Z
  setzi         $wkr_function, convPartialFlattenedFieldStateRetained_\COMMAND\()_Z
.else
  st32          $add_zero_in_stride_s, $wkr_vertex, WKR_IN_OUT_STRIDES/4
.endif
  brnzdec       $conv_group_s, convGroupsLoop_\ACTIVATIONS_TYPE\()_\PARTIALS_TYPE\()_\LD128\()_\AMP_OUTPUTS\()_\DISABLE_SR\()

.ifc \DISABLE_SR, true
  RESTORE_FPICTL (WKR_VERTEX_SIZE/4 + SUP_FPICTL/4) fpictl_s
.endif

// restore stack
ld32          $m9, $sp, WKR_VERTEX_SIZE/4 + SUP_STACK_CALLEE_SAVE0/4
ld32          $m10, $sp, WKR_VERTEX_SIZE/4 + SUP_STACK_CALLEE_SAVE1/4
add           $sp, $sp, TOT_STACK_SIZE
br            $lr

.size CODELET_NAME, . - CODELET_NAME
.endm


// =============================================================================
// Macro to load weights for 1x1 kernel

.macro LOAD_WEIGHTS_1X1 LDTYPE NUM_ENGINES
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
        ld128putcs    20
        ld128putcs    22
        st32          $outvec_s, $mzero, $wkr_vertex, WKR_OUTCHAN_PTR/4
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
        ld128putcs    20
        ld128putcs    22
        ld128putcs    48
        ld128putcs    50
        ld128putcs    52
        ld128putcs    54
        st32          $outvec_s, $mzero, $wkr_vertex, WKR_OUTCHAN_PTR/4
        ld128putcs    24
        ld128putcs    26
        ld128putcs    28
        ld128putcs    30
        ld128putcs    56
        ld128putcs    58
        ld128putcs    60
        ld128putcs    62
.elseif \LDTYPE == 64 && \NUM_ENGINES == 8
        ld64putcs     0
        ld64putcs     1
        ld64putcs     2
        ld64putcs     3
        ld64putcs     4
        ld64putcs     5
        ld64putcs     6
        ld64putcs     7
        ld64putcs     8
        ld64putcs     9
        ld64putcs     10
        ld64putcs     11
        ld64putcs     12
        ld64putcs     13
        ld64putcs     14
        ld64putcs     15
        ld64putcs     16
        ld64putcs     17
        ld64putcs     18
        ld64putcs     19
        ld64putcs     20
        ld64putcs     21
        ld64putcs     22
        ld64putcs     23
        st32          $outvec_s, $mzero, $wkr_vertex, WKR_OUTCHAN_PTR/4
        ld64putcs     24
        ld64putcs     25
        ld64putcs     26
        ld64putcs     27
        ld64putcs     28
        ld64putcs     29
        ld64putcs     30
        ld64putcs     31
.elseif \LDTYPE == 64 && \NUM_ENGINES == 16
        ld64putcs     0
        ld64putcs     1
        ld64putcs     2
        ld64putcs     3
        ld64putcs     4
        ld64putcs     5
        ld64putcs     6
        ld64putcs     7
        ld64putcs     32
        ld64putcs     33
        ld64putcs     34
        ld64putcs     35
        ld64putcs     36
        ld64putcs     37
        ld64putcs     38
        ld64putcs     39
        ld64putcs     8
        ld64putcs     9
        ld64putcs     10
        ld64putcs     11
        ld64putcs     12
        ld64putcs     13
        ld64putcs     14
        ld64putcs     15
        ld64putcs     40
        ld64putcs     41
        ld64putcs     42
        ld64putcs     43
        ld64putcs     44
        ld64putcs     45
        ld64putcs     46
        ld64putcs     47
        ld64putcs     16
        ld64putcs     17
        ld64putcs     18
        ld64putcs     19
        ld64putcs     20
        ld64putcs     21
        ld64putcs     22
        ld64putcs     23
        st32          $outvec_s, $mzero, $wkr_vertex, WKR_OUTCHAN_PTR/4
        ld64putcs     48
        ld64putcs     49
        ld64putcs     50
        ld64putcs     51
        ld64putcs     52
        ld64putcs     53
        ld64putcs     54
        ld64putcs     55
        ld64putcs     24
        ld64putcs     25
        ld64putcs     26
        ld64putcs     27
        ld64putcs     28
        ld64putcs     29
        ld64putcs     30
        ld64putcs     31
        ld64putcs     56
        ld64putcs     57
        ld64putcs     58
        ld64putcs     59
        ld64putcs     60
        ld64putcs     61
        ld64putcs     62
        ld64putcs     63
.else
.error "Weight load type not supported"
.endif
.endm

// =============================================================================
#endif // __CONV_PARTIAL_1X1_SUPERVISOR_S__
#endif // __IPU__
// =============================================================================
