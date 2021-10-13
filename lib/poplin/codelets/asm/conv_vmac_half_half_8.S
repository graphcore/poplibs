// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Computes an nx1 convolution using Vertical MAC with half input and half partials
// for 8 convolution groups per group.
// A contiguous field is partitioned for each position of the output element. The
// partitions are distributed between workers. Each worker is provided a
// temporary tensor to store its VMAC outputs. The worker outputs are finally
// reduced and stored in the user-specified output buffer.

#ifdef __IPU__

#include "conv_vmac_common.h.S"

// =============================================================================

// Vertex input, output and weight data atom size constants

// sizeof(input) in bytes
#define SIZEOF_IN_ATOM                  2
// sizeof(output) in bytes
#define SIZEOF_OUT_ATOM                 2
// sizeof(weights) in bytes
#define SIZEOF_WEIGHTS_ATOM             2

#define CHAN_GROUPS_PER_GROUP           8

// =============================================================================
// Performance:
//   overall = one_time_overhead + 6 + (9 + PARTITION_LOOP_8CHANS) + 1
//     where one_time_overhead= 14
//     and PARTITION_LOOP_8CHANS = 7 + num_partitions_v * (16 + (3 * num_fp_m1_v))
//
// Pre-requisite:
//  - The worklists should be sorted by output offset.
//

// worker register mapping
#define wkr_id_v                       m0
#define in_stride_v                    m0
#define weights_stride_v               m1
#define partition_table_v              m2
#define partition_struct_v             m2 // partition_v
#define defined_in_common_1            m3 // num_fp_m1_v
#define num_partitions_v               m4
#define stored_outchan_ptr_v           m5 // This can be store on the stack
#define weights_ptr_v                  m6
#define partition_base_v               m7 // in_offset_v
#define defined_in_common_3            m8 // out_delta_v
#define defined_in_common_4            m9 // weights_offset_v
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11
#define fp_clr_reg                     a0

// This is in the worker stack
#define WKR_STACK_TOTAL_PARTITIONS      WKR_STACK_COMMON
#define WKR_STACK_NUM_PARTITIONS        (WKR_STACK_TOTAL_PARTITIONS + 4)
#define WKR_STACK_PARTITION_BASE        (WKR_STACK_NUM_PARTITIONS + 4)

.section ".text.convVerticalMacFlattened_half_half_8", "ax"
.type convVerticalMacFlattened_half_half_8, @function
.align 8
.worker
nop
convVerticalMacFlattened_half_half_8:
get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK

ld32          $partition_table_v, $mvertex_base, WKR_PARTITION_PTR/4
ld32          $partition_struct_v, $partition_table_v, $wkr_id_v

// extract offset and delta: upper X bits gives number of partitions
// lower Y bits give offset to common base for the vector list
// For DeltaN: X = 18, Y = 14
// For DeltaNElements: X and Y are based on memory alignment used
shr           $num_partitions_v, $partition_struct_v, DELTAN_OFFSET_BITS
st32          $num_partitions_v, $mworker_base, WKR_STACK_TOTAL_PARTITIONS/4
// exit if no work to be done for worker
shl           $partition_v, $partition_struct_v, DELTAN_COUNT_BITS
// Offset needs to be converted from elements to bytes.
// Hence shift rigth is less by ELEM_TO_BYTES_CONV
shr           $partition_v, $partition_v, (DELTAN_COUNT_BITS - ELEM_TO_BYTES_CONV)
ld32          $partition_base_v, $mvertex_base, WKR_PARTITION_BASE/4

// This following approximates num_partitions/4 - 1 for values
// [3:3:2^14-1]. The case of zero is handled above
{
  add           $partition_v, $partition_v, $partition_base_v
  setzi         $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}

// clear accumulator: only required once because we always feed zeros
// with gina instruction
{
  shr           $num_partitions_v, $num_partitions_v, 2
  uput          $FP_CLR, $fp_clr_reg
}

add           $num_partitions_v, $num_partitions_v, -1
st32          $num_partitions_v, $mworker_base, WKR_STACK_NUM_PARTITIONS/4
st32          $partition_v, $mworker_base, WKR_STACK_PARTITION_BASE/4

convVerticalMacFlattenedReentry_half_half_8:

// Compensate for extra load in the loops. This could be done in the supervisor
ld32          $num_partitions_v, $mworker_base, WKR_STACK_TOTAL_PARTITIONS/4
brz           $num_partitions_v, L_Conv_end

ld32          $in_stride_v, $mvertex_base, WKR_IN_STRIDE/4
ld32          $weights_stride_v, $mvertex_base, WKR_WEIGHTS_STRIDE/4

ld32          $num_partitions_v, $mworker_base, WKR_STACK_NUM_PARTITIONS/4
ld32          $partition_v, $mworker_base, WKR_STACK_PARTITION_BASE/4
ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
ld32          $inchan_ptr_v, $mvertex_base, WKR_INCHAN_PTR/4
ld32          $outchan_ptr_v, $mworker_base, WKR_STACK_PARTIALS_PTR/4

// Load first delta, adjust outchan pointer with delta
ldz16step     $out_delta_v, $mzero, $partition_v+=, 1
mul           $out_delta_v, $out_delta_v, CHAN_GROUPS_PER_GROUP
ldz16step     $mzero, $mzero, $outchan_ptr_v+=, $out_delta_v

PARTITION_LOOP_8CHANS 0 SIZEOF_IN_ATOM SIZEOF_WEIGHTS_ATOM CHAN_GROUPS_PER_GROUP

L_Conv_end:
exitz         $mzero

.size convVerticalMacFlattened_half_half_8, . - convVerticalMacFlattened_half_half_8

// =============================================================================

VMAC_STATE_RETENTION_HALF CHAN_GROUPS_PER_GROUP
VMAC_REDUCE_HALF_BY_8 CHAN_GROUPS_PER_GROUP

// Instantiate supervisor codelet
CONV_VMAC_SUPERVISOR half half CHAN_GROUPS_PER_GROUP

#endif

// =============================================================================
// =============================================================================
