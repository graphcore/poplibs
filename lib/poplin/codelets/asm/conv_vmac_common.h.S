// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Common header for Vertical MAC

#ifndef _conv_vmac_common_h_S_
#define _conv_vmac_common_h_S_
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define HALF_HALF_VERTEX_NAME __runCodelet_poplin__ConvPartialVerticalMac___half_half_true_4
#define HALF_FLOAT_VERTEX_NAME __runCodelet_poplin__ConvPartialVerticalMac___half_float_true_4

// =============================================================================

// =============================================================================

// Supervisor vertex state: offsets and the number must match vertex field
// ordering and sizes
#if defined(VECTORLIST_AVAIL_DELTAN)
#define SUP_INCHAN_VECTORS              0    // word
#define SUP_WEIGHTS_VECTORS             4    // word
#define SUP_OUTCHAN_VECTORS             8    // word
#define SUP_PARTIALS_VECTOR             12   // word
#define SUP_ZERO_INFO                   16   // word
#define SUP_NUM_INCHAN                  20   // word
#define SUP_NUM_CONVGROUPS_M1           24   // word
#define SUP_PARTITION_TABLES            28   // VectorList::DeltaN/DeltaNElements
#define SUP_IN_STRIDE                   34   // short
#define SUP_WEIGHTS_STRIDE              36   // short
#else
#define SUP_INCHAN_VECTORS              0    // word
#define SUP_WEIGHTS_VECTORS             4    // word
#define SUP_OUTCHAN_VECTORS             8    // word
#define SUP_PARTIALS_VECTOR             12   // word
#define SUP_ZERO_INFO                   16   // word
#define SUP_NUM_INCHAN                  20   // word
#define SUP_NUM_CONVGROUPS_M1           24   // word
#define SUP_PARTITION_TABLES            28   // VectorList::DeltaN/DeltaNElements
#define SUP_IN_STRIDE                   36   // short
#define SUP_WEIGHTS_STRIDE              38   // short
#endif

// Worklist partition fields: must match the order of entries in worklists
#define PARTITION_OUT_OFFSET           0    // word
#define PARTITION_WEIGHTS_OFFSET       2    // word
#define PARTITION_IN_OFFSET            4    // word
#define PARTITION_NUM_ELEMS            6    // word

// DeltaN decoding constants
#if defined(VECTORLIST_AVAIL_DELTAN)
#define SCALED_PTR32_SHL         2
#define DELTAN_DELTAS_ADDR_BITS  20
#define DELTAN_DELTAS_COUNT_BITS 12
#define DELTAN_OFFSET_BITS       18
#define DELTAN_COUNT_BITS        14
#define ELEM_TO_BYTES_CONV       0
#else
#define DELTAN_DELTAS_ADDR_BITS  24
#define DELTAN_DELTAS_COUNT_BITS 8
#define DELTAN_OFFSET_BITS       20
#define DELTAN_COUNT_BITS        (32 - DELTAN_OFFSET_BITS)
#define ELEM_TO_BYTES_CONV       (21 - DELTAN_OFFSET_BITS)
#endif

// =============================================================================
// Vertex state shared between workers
// The supervisor sets up a common state for workers to use
// Allocate codelet specific worker stack after zero output worker (WKR_ZERO_OUTPUT_STACK)
#define WKR_ZERO_LEN                    0
#define WKR_PARTIALS_PTR                (WKR_ZERO_LEN + 4)
#define WKR_IN_STRIDE                   (WKR_PARTIALS_PTR + 4)
#define WKR_WEIGHTS_STRIDE              (WKR_IN_STRIDE + 4)
#define WKR_INCHAN_PTR                  (WKR_WEIGHTS_STRIDE + 4)
#define WKR_OUTCHAN_PTR                 (WKR_INCHAN_PTR + 4)
#define WKR_WEIGHTS_PTR                 (WKR_OUTCHAN_PTR + 4)
#define WKR_PARTITION_PTR               (WKR_WEIGHTS_PTR + 4)
#define WKR_PARTITION_BASE              (WKR_PARTITION_PTR + 4)
#define WKR_STATE_SIZE                  (WKR_PARTITION_BASE + 4)

// Worker stack preserved for all worker functions
#define WKR_STACK_PARTIALS_PTR           0
#define WKR_STACK_TOTAL_ELEMS_X2         (WKR_STACK_PARTIALS_PTR + 4)
#define WKR_STACK_TOTAL_ELEMS_X4         (WKR_STACK_TOTAL_ELEMS_X2 + 4)
#define WKR_STACK_NUM_ELEMS_X4           (WKR_STACK_TOTAL_ELEMS_X4 + 4)
#define WKR_STACK_REDUCTION_PARTIALS_PTR (WKR_STACK_NUM_ELEMS_X4 + 4)
#define WKR_STACK_REDUCTION_ELEM_OFFSET  (WKR_STACK_REDUCTION_PARTIALS_PTR + 4)
#define WKR_STACK_COMMON                 (WKR_STACK_REDUCTION_ELEM_OFFSET + 4)

// common registers aliases
#define partition_v                    m2
#define in_offset_v                    m7
#define weights_offset_v               m9
#define num_fp_m1_v                    m3
#define out_delta_v                    m8

// =============================================================================
// Performance:
// 56 + numConvGroups * (42 + numInChanGroups * (34 + innerLoopCycles))
//
// Where innerLoopCycles are vertex cycles
//

// Total stack size
#define TOT_STACK_SIZE                  (WKR_STATE_SIZE + 8)

// registers
#define sup_base                        m0
#define partials_vectors_z_s            m1
#define wkr_reduction_s                 m1
#define wkr_zero_s                      m1
#define convgroup_count_s               m2
#define inchan_ptr_s                    m3
#define weights_ptr_s                   m4
#define in_stride_s                     m4
#define weights_stride_s                m5
#define inchan_count_s                  m5
#define partition_ptr_s                 m6
#define outchan_ptr_s                   m6
#define wkr_core_s                      m7
#define zero_info_s_z                   m8
#define inchan_vectors_s                m8
#define outchan_vectors_s               m9
#define worklist_count_s                m9
#define weights_vectors_s               m10
#define wkr_base                        sp

.macro CONV_VMAC_SUPERVISOR VNAME, INTYPE, PTYPE

DEF_STACK_USAGE  TOT_STACK_SIZE  \VNAME\()

.section .text.\VNAME\()
.align 4
.type \VNAME\(), @function
.globl \VNAME\()
.supervisor

\VNAME\():
// From base and delta, create base and expand pointer to vector containing
// deltaN (i.e. 18 bit delta offset and 14 bit size in number of elements)
// The instructions used in the calculation of the partition parameters are
// spread in the code to avoid
ld32          $partition_ptr_s, $sup_base, SUP_PARTITION_TABLES/4

// space for worker vertex state, supervisor state and callee save
add           $sp, $sp, -TOT_STACK_SIZE
st32          $m9, $sp, WKR_STATE_SIZE/4 + 0
st32          $m10, $sp, WKR_STATE_SIZE/4 + 1
shr           $worklist_count_s, $partition_ptr_s, DELTAN_DELTAS_ADDR_BITS
ld32          $zero_info_s_z, $sup_base, SUP_ZERO_INFO/4
shl           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
add           $worklist_count_s, $worklist_count_s, -1
sync          TEXCH_SYNCZONE_LOCAL
shr           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16         $partials_vectors_z_s, $sup_base, SUP_PARTIALS_VECTOR/2
#else
ld32          $partials_vectors_z_s, $sup_base, SUP_PARTIALS_VECTOR/4
#endif
st32          $zero_info_s_z, $wkr_base, WKR_ZERO_LEN/4
st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_BASE/4

// The only call to the convolution may be to zero out partials
brz           $worklist_count_s,  L_sup_conv_end_\INTYPE\()_\PTYPE\()
#if defined(VECTORLIST_AVAIL_DELTAN)
ldz16         $partition_ptr_s, $sup_base, (SUP_PARTITION_TABLES + 4)/2
#else
ld32          $partition_ptr_s, $sup_base, (SUP_PARTITION_TABLES + 4)/4
#endif

ld32          $convgroup_count_s, $sup_base, SUP_NUM_CONVGROUPS_M1/4

lds16         $in_stride_s, $sup_base, SUP_IN_STRIDE/2
lds16         $weights_stride_s, $sup_base, SUP_WEIGHTS_STRIDE/2
#if defined(VECTORLIST_AVAIL_DELTAN)
shl           $partition_ptr_s, $partition_ptr_s, (SCALED_PTR32_SHL + 13)
shl           $partials_vectors_z_s, $partials_vectors_z_s, 3
or            $partition_ptr_s, $partition_ptr_s, (TMEM_REGION0_BASE_ADDR << 13)
#else
shl           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
nop
#endif
st32          $in_stride_s, $wkr_base, WKR_IN_STRIDE/4
st32          $weights_stride_s, $wkr_base, WKR_WEIGHTS_STRIDE/4
ld32          $inchan_count_s, $sup_base, SUP_NUM_INCHAN/4
setzi         $wkr_core_s, convVerticalMacFlattened_\INTYPE\()_\PTYPE\()
st32          $partials_vectors_z_s, $wkr_base, WKR_PARTIALS_PTR/4
#if defined(VECTORLIST_AVAIL_DELTAN)
shr           $partition_ptr_s, $partition_ptr_s, 13
#else
shr           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
#endif
setzi         $wkr_zero_s, convVerticalMacWorkerStateRetention_\PTYPE\()
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
add           $inchan_count_s, $inchan_count_s, -1
ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4

ConvGroupLoop_\INTYPE\()_\PTYPE\():
  sync          TEXCH_SYNCZONE_LOCAL
  runall        $wkr_zero_s, $wkr_base, 0
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $outchan_ptr_s, $mzero, $outchan_vectors_s+=, 1
#endif
  setzi         $wkr_reduction_s, convVerticalMacReduce_\PTYPE\()
InChanLoop_\INTYPE\()_\PTYPE\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
    ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
    shl           $inchan_ptr_s, $inchan_ptr_s, 3
    shl           $weights_ptr_s, $weights_ptr_s, 3
#else
    ld32step      $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
    ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
    // partials pointer should already be known worker
    sync          TEXCH_SYNCZONE_LOCAL
    st32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
    st32          $weights_ptr_s, $wkr_base, WKR_WEIGHTS_PTR/4
    runall        $wkr_core_s, $wkr_base, 0
    setzi         $wkr_core_s, convVerticalMacFlattenedReentry_\INTYPE\()_\PTYPE\()
    brnzdec       $inchan_count_s, InChanLoop_\INTYPE\()_\PTYPE\()
  ld32          $inchan_count_s, $sup_base, SUP_NUM_INCHAN/4

  // Reduce the partials and store results to output tensor
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  shl           $outchan_ptr_s, $outchan_ptr_s, 3
#else
  ld32step      $outchan_ptr_s, $mzero, $outchan_vectors_s+=, 1
#endif
  sync          TEXCH_SYNCZONE_LOCAL
  st32          $outchan_ptr_s, $wkr_base, WKR_OUTCHAN_PTR/4
  runall        $wkr_reduction_s, $wkr_base, 0

  // Prepare for zeroing out partials for the next conv group
  setzi         $wkr_zero_s, convVerticalMacZeroPartials_\PTYPE\()
  add           $inchan_count_s, $inchan_count_s, -1
  brnzdec       $convgroup_count_s, ConvGroupLoop_\INTYPE\()_\PTYPE\()

L_sup_conv_end_\INTYPE\()_\PTYPE\():

// Restore saved registers
ld32          $m9, $sp, WKR_STATE_SIZE/4 + 0
ld32          $m10, $sp, WKR_STATE_SIZE/4 + 1
add           $sp, $sp, TOT_STACK_SIZE
sync          TEXCH_SYNCZONE_LOCAL
br            $lr
.size \VNAME\(), . - \VNAME\()
.endm

// =============================================================================

// load worklist entry parametrs into registers
.macro LOAD_WORKLIST_ENTRY_EXCEPT_OUT IN_ATOM_SIZE, WEIGHTS_ATOM_SIZE, NUM_CHANS
  ldz16step     $weights_offset_v, $mzero, $partition_v+=, 1

  // form input address
  ldz16step     $in_offset_v, $mzero, $partition_v+=, 1
  mul           $in_offset_v, $in_offset_v, \IN_ATOM_SIZE * \NUM_CHANS

  {
    ldz16step     $num_fp_m1_v, $mzero, $partition_v+=, 1
    mov           $a0:1, $azeros
  }

  // form weights address
  {
    mul           $weights_offset_v, $weights_offset_v, \WEIGHTS_ATOM_SIZE * \NUM_CHANS
    mov           $a2:3, $azeros
  }
.endm

#endif // #ifdef __IPU__
#endif // #ifdef _conv_vmac_common_h_S_


// =============================================================================
// =============================================================================
