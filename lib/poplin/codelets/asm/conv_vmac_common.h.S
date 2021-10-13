// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Common header for Vertical MAC

#ifndef _conv_vmac_common_h_S_
#define _conv_vmac_common_h_S_
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================
#define CODELET_NAME __runCodelet_poplin__ConvPartialVerticalMac___\INTYPE\()_\PTYPE\()_true_\NUM_CHANS\()

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
#define in_stride_v                    m0
#define weights_stride_v               m1
#define partition_v                    m2
#define num_fp_m1_v                    m3
#define num_partitions_v               m4
#define weights_ptr_v                  m6
#define in_offset_v                    m7
#define out_delta_v                    m8
#define weights_offset_v               m9
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11

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

.macro CONV_VMAC_SUPERVISOR INTYPE PTYPE NUM_CHANS

DEF_STACK_USAGE  TOT_STACK_SIZE  CODELET_NAME

.section .text.CODELET_NAME
.align 4
.type CODELET_NAME, @function
.globl CODELET_NAME
.supervisor

CODELET_NAME:
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
brz           $worklist_count_s,  L_sup_conv_end_\INTYPE\()_\PTYPE\()_\NUM_CHANS\()
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
setzi         $wkr_core_s, convVerticalMacFlattened_\INTYPE\()_\PTYPE\()_\NUM_CHANS\()
st32          $partials_vectors_z_s, $wkr_base, WKR_PARTIALS_PTR/4
#if defined(VECTORLIST_AVAIL_DELTAN)
shr           $partition_ptr_s, $partition_ptr_s, 13
#else
shr           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
#endif
setzi         $wkr_zero_s, convVerticalMacWorkerStateRetention_\PTYPE\()_\NUM_CHANS\()
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
add           $inchan_count_s, $inchan_count_s, -1
ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4

ConvGroupLoop_\INTYPE\()_\PTYPE\()_\NUM_CHANS\():
  sync          TEXCH_SYNCZONE_LOCAL
  runall        $wkr_zero_s, $wkr_base, 0
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $outchan_ptr_s, $mzero, $outchan_vectors_s+=, 1
#endif
  setzi         $wkr_reduction_s, convVerticalMacReduce_\PTYPE\()_\NUM_CHANS\()
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
    setzi         $wkr_core_s, convVerticalMacFlattenedReentry_\INTYPE\()_\PTYPE\()_\NUM_CHANS\()
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
  setzi         $wkr_zero_s, convVerticalMacZeroPartials_\PTYPE\()_\NUM_CHANS\()
  add           $inchan_count_s, $inchan_count_s, -1
  brnzdec       $convgroup_count_s, ConvGroupLoop_\INTYPE\()_\PTYPE\()_\NUM_CHANS\()

L_sup_conv_end_\INTYPE\()_\PTYPE\()_\NUM_CHANS\():

// Restore saved registers
ld32          $m9, $sp, WKR_STATE_SIZE/4 + 0
ld32          $m10, $sp, WKR_STATE_SIZE/4 + 1
add           $sp, $sp, TOT_STACK_SIZE
sync          TEXCH_SYNCZONE_LOCAL
br            $lr

.size CODELET_NAME, . - CODELET_NAME
.endm

// =============================================================================
// Performance: 5 cycles
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


// =============================================================================
// Performance: 3 + (3 * num_fp_m1_v) cycles
.macro PARTITION_CORE_LOOP_8CHANS ID NUM_CHANS
  {
    ld128step    $a0:3, $inchan_ptr_v, $in_offset_v+=, $in_stride_v  
    f16v8acc     $a0:3
  }
  ld128step      $a4:7, $weights_ptr_v, $weights_offset_v+=, $weights_stride_v
  {
    rpt          $num_fp_m1_v, (Loop_end_Vmac_\NUM_CHANS\()_\ID - Loop_start_Vmac_\NUM_CHANS\()_\ID) / 8 - 1
    fnop
  }
Loop_start_Vmac_\NUM_CHANS\()_\ID:
  {
    nop
    f16v4mul       $a2:3, $a2:3, $a6:7
  }
  {
    ld128step      $a4:7, $weights_ptr_v, $weights_offset_v+=, $weights_stride_v
    f16v4mul       $a0:1, $a0:1, $a4:5
  }
  {
    ld128step      $a0:3, $inchan_ptr_v, $in_offset_v+=, $in_stride_v    
    f16v8acc       $a0:3
  }
Loop_end_Vmac_\NUM_CHANS\()_\ID:
.endm

// =============================================================================
// Performance for PARTITION_LOOP_8CHANS macro:
//      partition_loop = LOAD_WORKLIST_ENTRY_EXCEPT_OUT(5) + 
//                       6 + 
//                       PARTITION_CORE_LOOP_8CHANS(3 + (3 * num_fp_m1_v)) + 
//                       3
//      overall = 2 + num_partitions_v * (partition_loop) + 5
// Requires:
// - partition_v       
// - num_partitions_v  
// Next pointers must be adjusted by: 
//          0 - to process first 8 channels (0...7)
//          CHAN_GROUPS_PER_GROUP / 2 - to process second 8 channels (8...15)
// - inchan_ptr_v
// - weights_ptr_v
// - outchan_ptr_v

.macro PARTITION_LOOP_8CHANS ID SIZEOF_IN_ATOM SIZEOF_WEIGHTS_ATOM NUM_CHANS

// reload the accumulators with current output values
ld128          $a0:3, $mzero, $outchan_ptr_v, 0
{
  setzi         $out_delta_v, 0
  f16v8acc      $a0:3
}

Partition_loop_start_\ID:
  LOAD_WORKLIST_ENTRY_EXCEPT_OUT \SIZEOF_IN_ATOM \SIZEOF_WEIGHTS_ATOM \NUM_CHANS
  
  brz           $out_delta_v, Partition_acc_\ID

  // store the accumulator result except if:
  // - this is the very first pass
  // - out_delta_v remains zero then keep accumulating
  {
    ld128         $a0:3, $mzero, $outchan_ptr_v, $out_delta_v
    f16v2gina     $a4, $azero, 0
  }
  {
    // Adjust out_delta as we load with ld128 but store with st64
    mul           $out_delta_v, $out_delta_v, 2
    f16v2gina     $a5, $azero, 0
  }
  {
    st64step      $a4:5, $mzero, $outchan_ptr_v+=, 1
    f16v2gina     $a4, $azero, 0
  }
  {
    add           $out_delta_v,  $out_delta_v, -1 // because we already stored first 4 values
    f16v2gina     $a5, $azero, 0
  }
  st64step      $a4:5, $mzero, $outchan_ptr_v+=, $out_delta_v

Partition_acc_\ID:
  PARTITION_CORE_LOOP_8CHANS \ID \NUM_CHANS

  {
    ldz16step     $out_delta_v, $mzero, $partition_v+=, 1
    f16v4mul      $a2:3, $a2:3, $a6:7
  }
  {
    // Adjust out_delta_v because ld128 can load only 8 channels per call
    mul           $out_delta_v, $out_delta_v, \NUM_CHANS / 8
    f16v4mul      $a0:1, $a0:1, $a4:5
  }
  {
    brnzdec       $num_partitions_v, Partition_loop_start_\ID
    f16v8acc      $a0:3
  }

  // Store accumulator results as final output
  f16v2gina     $a4, $azero, 0
  f16v2gina     $a5, $azero, 0
  {
    st64step      $a4:5, $mzero, $outchan_ptr_v+=, 1
    f16v2gina     $a4, $azero, 0
  }
  f16v2gina     $a5, $azero, 0
  st64          $a4:5, $mzero, $outchan_ptr_v, 0

Partition_loop_end_\ID:
.endm


// =============================================================================
// Performance:
// half_8 or half_16: 
//    31 + (2 * num_elements / 8)
// half_4: 
//    31 + (num_elements / 4)
.macro VMAC_STATE_RETENTION_HALF NUM_CHANS

.if \NUM_CHANS == 4
  .equ SHIFT_NUM_CHANS,      2
// NUM_CHANS  == 16 shall use same config as 8 because we limited by
// how many elements we can process by one load/store instruction
.elseif \NUM_CHANS == 8 || \NUM_CHANS  == 16
  .equ SHIFT_NUM_CHANS,      3
.else
    .error "VMAC (half) groups per group not supported"
.endif

#define VMAC_STATE_RETENTION_CODELET convVerticalMacWorkerStateRetention_half_\NUM_CHANS\()
#define VMAC_ZERO_PARTIALS convVerticalMacZeroPartials_half_\NUM_CHANS\()

.section ".text.VMAC_STATE_RETENTION_CODELET", "ax"
.type VMAC_STATE_RETENTION_CODELET, @function
.align 8
.worker
VMAC_STATE_RETENTION_CODELET:
// worker register mapping
#define wkr_id_v                       m0
#define partials_offset                m1
#define mscratch                       m1
#define partials_ptr                   m2
#define total_elems                    m3
#define total_elems_by_ld_st           m3
#define num_elems_by_ld_st             m4
#define num_elems_whole                m5
#define remainder_elems                m5
#define remainder_cumul                m5
#define single_elem                    m6
#define out_ptr_v                      m6
#define elems_offset                   m7
#define fp_clr_reg                     a0

// Save worker state for later calls to convVerticalMacZeroPartials
get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK
ld32          $total_elems, $mvertex_base, WKR_ZERO_LEN/4

// form partials address for worker
ld32          $partials_ptr, $mvertex_base, WKR_PARTIALS_PTR/4
mul           $partials_offset, $total_elems, $wkr_id_v
shr           $total_elems_by_ld_st, $total_elems, SHIFT_NUM_CHANS
ldb16step     $azero, $mzero, $partials_ptr+=, $partials_offset

// store partials address for other worker functions
st32          $partials_ptr, $mworker_base, WKR_STACK_PARTIALS_PTR/4
st32          $total_elems_by_ld_st, $mworker_base, WKR_STACK_TOTAL_ELEMS_X4/4

// Save worker state for later calls to convVerticalMacReduce_half

// Calculate the number of (4 or 8)-element blocks to be reduced by this worker
// for the rest calculate n / 6 and n % 6 by reciprocal multiplcation
//   n/6 = (n * 0xAAAB) >> 18
//   n%6 = n - (n/6)*6
// where n = count/(4*atomSize)
setzi         $mscratch, 0xAAAB
mul           $num_elems_by_ld_st, $total_elems_by_ld_st, $mscratch
shr           $num_elems_by_ld_st, $num_elems_by_ld_st, 18
mul           $elems_offset, $wkr_id_v, $num_elems_by_ld_st
mul           $num_elems_whole, $num_elems_by_ld_st, CTXT_WORKERS
sub           $remainder_elems, $total_elems_by_ld_st, $num_elems_whole
cmpult        $single_elem, $wkr_id_v, $remainder_elems
add           $num_elems_by_ld_st, $num_elems_by_ld_st, $single_elem

// calculate the beginning offset
min           $remainder_cumul, $wkr_id_v, $remainder_elems

// form the output address
{
  add           $elems_offset, $elems_offset, $remainder_cumul
  setzi         $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}

// clear accumulator: only required once because we always feed zeros
// with gina instruction
{
  shl           $elems_offset, $elems_offset, SHIFT_NUM_CHANS
  uput          $FP_CLR, $fp_clr_reg
}

// position worker offset into partials vectors for reduction
ld32          $partials_ptr, $mvertex_base, WKR_PARTIALS_PTR/4
ldb16step $azero, $mzero, $partials_ptr+=, $elems_offset

// store worker state for reduction step
st32          $num_elems_by_ld_st, $mworker_base, WKR_STACK_NUM_ELEMS_X4/4
st32          $partials_ptr, $mworker_base, WKR_STACK_REDUCTION_PARTIALS_PTR/4
st32          $elems_offset, $mworker_base, WKR_STACK_REDUCTION_ELEM_OFFSET/4

VMAC_ZERO_PARTIALS:
ld32          $partials_ptr, $mworker_base, WKR_STACK_PARTIALS_PTR/4
ld32          $total_elems_by_ld_st, $mworker_base, WKR_STACK_TOTAL_ELEMS_X4/4
rpt           $total_elems_by_ld_st, (2f - 1f) / 8 - 1
1:
  {
    st64step      $azeros, $mzero, $partials_ptr+=, 1
    fnop
  }
.if \NUM_CHANS >= 8  
  {
    st64step      $azeros, $mzero, $partials_ptr+=, 1
    fnop
  }
.endif
2:
exitz         $mzero

#undef wkr_id_v
#undef partials_offset
#undef mscratch
#undef partials_ptr
#undef total_elems
#undef total_elems_by_ld_st
#undef num_elems_by_ld_st
#undef num_elems_whole
#undef remainder_elems
#undef remainder_cumul
#undef single_elem
#undef out_ptr_v
#undef elems_offset
#undef fp_clr_reg

.size VMAC_STATE_RETENTION_CODELET, . - VMAC_STATE_RETENTION_CODELET
.endm
// =============================================================================


// =============================================================================
// Performance:
// 10 + ((9 + (num_contexts - 1)) * (n/8))

.macro VMAC_REDUCE_HALF_BY_8 NUM_CHANS

.if \NUM_CHANS != 8 && \NUM_CHANS != 16
  .error "VMAC reduce codelet for half partials not supported for requested channels"
.endif

#define VMAC_REDUCE_BY_8_CODELET convVerticalMacReduce_half_\NUM_CHANS\()

.section ".text.VMAC_REDUCE_BY_8_CODELET", "ax"
.type VMAC_REDUCE_BY_8_CODELET, @function
.align 8
.worker
VMAC_REDUCE_BY_8_CODELET:
// worker register mapping
#define out_ptr_v                      m1
#define elems_offset                   m2
#define num_elems_by_ld_st             m3
#define total_elems_by_ld_st           m4
#define partials_base                  m5
#define partials_context               m6

ld32          $num_elems_by_ld_st, $mworker_base, WKR_STACK_NUM_ELEMS_X4/4
brz           $num_elems_by_ld_st, LReduce_end
add           $num_elems_by_ld_st, $num_elems_by_ld_st, -1
ld32          $total_elems_by_ld_st, $mworker_base, WKR_STACK_TOTAL_ELEMS_X4/4
ld32          $partials_base, $mworker_base, WKR_STACK_REDUCTION_PARTIALS_PTR/4
ld32          $elems_offset, $mworker_base, WKR_STACK_REDUCTION_ELEM_OFFSET/4

ld32          $out_ptr_v, $mvertex_base, WKR_OUTCHAN_PTR/4
ldb16step     $azero, $mzero, $out_ptr_v+=, $elems_offset
mov           $partials_context, $partials_base

// accumulate partials four at a time
LRowLoop:
  ld128step     $a0:3, $mzero, $partials_context+=, $total_elems_by_ld_st
  {
    rpt         CTXT_WORKERS-1, (2f - 1f) / 8 - 1
    fnop
  }
1:
    {
      ld128step   $a0:3, $mzero, $partials_context+=, $total_elems_by_ld_st
      f16v8acc    $a0:3
    }
2:
  {
    // shift base to the next 8 channels
    ldb16step    $azero, $mzero, $partials_base+=, 8
    f16v8acc     $a0:3
  }
  {
    mov          $partials_context, $partials_base
    f16v2gina    $a0, $azero, 0
  }
  f16v2gina   $a1, $azero, 0
  {
    st64step    $a0:1, $mzero, $out_ptr_v+=, 1
    f16v2gina   $a2, $azero, 0
  }
  f16v2gina   $a3, $azero, 0
  st64step    $a2:3, $mzero, $out_ptr_v+=, 1

  brnzdec     $num_elems_by_ld_st, LRowLoop

LReduce_end:
exitz         $mzero

#undef out_ptr_v
#undef elems_offset
#undef num_elems_by_ld_st
#undef total_elems_by_ld_st
#undef partials_base
#undef context
#undef partials_context
.size VMAC_REDUCE_BY_8_CODELET, . - VMAC_REDUCE_BY_8_CODELET
.endm
// =============================================================================

#endif // #ifdef __IPU__
#endif // #ifdef _conv_vmac_common_h_S_


// =============================================================================
// =============================================================================
