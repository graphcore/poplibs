// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Computes an nx1 convolution using Vertical MAC with half input and float partials.
// A contiguous field is partitioned for each position of the output element. The
// partitions are distributed between between workers. Each worker is provided a
// temporary tensor to store its VMAC outputs. The worker outputs are finally
// reduced and stored in the user-specified output buffer.

#ifdef __IPU__

#include "conv_vmac_common.h.S"

// =============================================================================

// Vertex input, output and weight data atom size constants

// sizeof(input) in bytes
#define SIZEOF_IN_ATOM                  2
// sizeof(output) in bytes
#define SIZEOF_OUT_ATOM                 4
// sizeof(weights) in bytes
#define SIZEOF_WEIGHTS_ATOM             2

#define CHAN_GROUPS_PER_GROUP           4

// =============================================================================
// Performance:
// 9 + (1 * num_elements)
.section ".text.convVerticalMacWorkerStateRetention_float_4", "ax"
.type convVerticalMacWorkerStateRetention_float_4, @function
.align 8
.worker
convVerticalMacWorkerStateRetention_float_4:
// worker register mapping
#define wkr_id_v                       m0
#define partials_offset                m1
#define mscratch                       m1
#define partials_ptr                   m2
#define total_elems                    m3
#define total_elemsx2                  m3
#define total_elemsx4                  m3
#define num_elemsx4                    m4
#define num_elems_whole                m5
#define remainder_elems                m5
#define remainder_cumul                m5
#define single_elem                    m6
#define out_ptr_v                      m6
#define elems_offset                   m7
#define fp_clr_reg                     a0

// Save worker state for later calls to convVerticalMacZeroPartials_float_4
get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK
ld32          $total_elems, $mvertex_base, WKR_ZERO_LEN/4

// form partials address for worker
ld32          $partials_ptr, $mvertex_base, WKR_PARTIALS_PTR/4
mul           $partials_offset, $total_elems, $wkr_id_v
shr           $total_elemsx2, $total_elems, 1
ld32step      $azero, $mzero, $partials_ptr+=, $partials_offset

// store partials address for other worker functions
st32          $partials_ptr, $mworker_base, WKR_STACK_PARTIALS_PTR/4
st32          $total_elemsx2, $mworker_base, WKR_STACK_TOTAL_ELEMS_X2/4

// Save worker state for later calls to convVerticalMacReduce_float_4

// Calculate the number of 4-element blocks to be reduced by this worker
// for the rest calculate n / 6 and n % 6 by reciprocal multiplcation
//   n/6 = (n * 0xAAAB) >> 18
//   n%6 = n - (n/6)*6
// where n = count/(4*atomSize)
shr           $total_elemsx4, $total_elemsx2, 1
setzi         $mscratch, 0xAAAB
mul           $num_elemsx4, $total_elemsx4, $mscratch
shr           $num_elemsx4, $num_elemsx4, 18
mul           $elems_offset, $wkr_id_v, $num_elemsx4
mul           $num_elems_whole, $num_elemsx4, CTXT_WORKERS
sub           $remainder_elems, $total_elemsx4, $num_elems_whole
cmpult        $single_elem, $wkr_id_v, $remainder_elems
add           $num_elemsx4, $num_elemsx4, $single_elem

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
  shl           $elems_offset, $elems_offset, 2
  uput          $FP_CLR, $fp_clr_reg
}

// position worker offset into partials vectors for reduction
ld32          $partials_ptr, $mvertex_base, WKR_PARTIALS_PTR/4
ld32step      $azero, $mzero, $partials_ptr+=, $elems_offset

// store worker state for reduction step
st32          $num_elemsx4, $mworker_base, WKR_STACK_NUM_ELEMS_X4/4
st32          $partials_ptr, $mworker_base, WKR_STACK_REDUCTION_PARTIALS_PTR/4
st32          $elems_offset, $mworker_base, WKR_STACK_REDUCTION_ELEM_OFFSET/4

convVerticalMacZeroPartials_float_4:
ld32          $partials_ptr, $mworker_base, WKR_STACK_PARTIALS_PTR/4
ld32          $total_elemsx2, $mworker_base, WKR_STACK_TOTAL_ELEMS_X2/4
{
  rpt           $total_elemsx2, (2f - 1f) / 8 - 1
  fnop
}
1:
  {
    st64step      $azeros, $mzero, $partials_ptr+=, 1
    fnop
  }
2:
exitz         $mzero

#undef wkr_id_v
#undef partials_offset
#undef mscratch
#undef partials_ptr
#undef total_elems
#undef total_elemsx2
#undef total_elemsx4
#undef num_elemsx4
#undef num_elems_whole
#undef remainder_elems
#undef remainder_cumul
#undef single_elem
#undef out_ptr_v
#undef elems_offset
#undef fp_clr_reg

.size convVerticalMacWorkerStateRetention_float_4, . - convVerticalMacWorkerStateRetention_float_4

// =============================================================================
// Performance:
// 9 + ((9 + (2 * (num_contexts - 1))) * (n/4))
.section ".text.convVerticalMacReduce_float_4", "ax"
.type convVerticalMacReduce_float_4, @function
.align 8
.worker
convVerticalMacReduce_float_4:
// worker register mapping
#define out_ptr_v                      m1
#define elems_offset                   m2
#define num_elemsx4                    m3
#define total_elemsx2                  m4
#define partials_base                  m5
#define partials_context               m6

ld32          $num_elemsx4, $mworker_base, WKR_STACK_NUM_ELEMS_X4/4
brz           $num_elemsx4, LReduce_end
add           $num_elemsx4, $num_elemsx4, -1
ld32          $total_elemsx2, $mworker_base, WKR_STACK_TOTAL_ELEMS_X2/4
ld32          $partials_base, $mworker_base, WKR_STACK_REDUCTION_PARTIALS_PTR/4
ld32          $elems_offset, $mworker_base, WKR_STACK_REDUCTION_ELEM_OFFSET/4

ld32          $out_ptr_v, $mvertex_base, WKR_OUTCHAN_PTR/4
ld32step      $azero, $mzero, $out_ptr_v+=, $elems_offset

// accumulate partials four at a time
LRowLoop:
  mov         $partials_context, $partials_base
  ld64        $a2:3, $mzero, $partials_context, 1
  {
    rpt         CTXT_WORKERS-1, (2f - 1f) / 8 - 1
    fnop
  }
1:
    {
      ld64step    $a0:1, $mzero, $partials_context+=, $total_elemsx2
      fnop
    }
    {
      ld64        $a2:3, $mzero, $partials_context, 1
      f32v4acc    $a0:3
    }
2:
  ld64step    $a0:1, $mzero, $partials_context+=, $total_elemsx2
  f32v4acc    $a0:3
  {
    ld32step    $azero, $mzero, $partials_base+=, 4
    f32v2gina   $a0:1, $azeros, 0
  }
  {
    st64step    $a0:1, $mzero, $out_ptr_v+=, 1
    f32v2gina   $a2:3, $azeros, 0
  }
  st64step    $a2:3, $mzero, $out_ptr_v+=, 1
  brnzdec     $num_elemsx4, LRowLoop

LReduce_end:
exitz         $mzero

#undef out_ptr_v
#undef elems_offset
#undef num_elemsx4
#undef total_elemsx2
#undef partials_base
#undef context
#undef partials_context
.size convVerticalMacReduce_float_4, . - convVerticalMacReduce_float_4

// =============================================================================
// Performance:
//   one_time_overhead + 20 + load_acc + (11 + 2*(num_fp_m1_v + 1)))*num_partitions_v + reload_acc*(num_outputs_processed-1)
//     where one_time_overhead= 13
//     where reload_acc = 5 cycles
//
// Pre-requisite:
//  - The worklists should be sorted by output offset.
//

// worker register mapping
#define wkr_id_v                       m0
#define in_stride_v                    m0
#define mscratch                       m0
#define weights_stride_v               m1
#define partition_table_v              m2
#define partition_struct_v             m2
#define num_partitions_v               m4
#define weights_ptr_v                  m6
#define partition_base_v               m7
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11
#define fp_clr_reg                     a0

// This is in the worker stack
#define WKR_STACK_TOTAL_PARTITIONS      WKR_STACK_COMMON
#define WKR_STACK_NUM_PARTITIONS        (WKR_STACK_TOTAL_PARTITIONS + 4)
#define WKR_STACK_PARTITION_BASE        (WKR_STACK_NUM_PARTITIONS + 4)

.section ".text.convVerticalMacFlattened_half_float_4", "ax"
.type convVerticalMacFlattened_half_float_4, @function
.align 8
nop  // rpt align
.worker
convVerticalMacFlattened_half_float_4:
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

convVerticalMacFlattenedReentry_half_float_4:

// Compensate for extra load in the loops. This could be done in the supervisor
ld32          $num_partitions_v, $mworker_base, WKR_STACK_TOTAL_PARTITIONS/4
brz           $num_partitions_v, L_Conv_end
ld32          $num_partitions_v, $mworker_base, WKR_STACK_NUM_PARTITIONS/4
ld32          $weights_stride_v, $mvertex_base, WKR_WEIGHTS_STRIDE/4
ld32          $partition_v, $mworker_base, WKR_STACK_PARTITION_BASE/4

// load previously stored partials address
ld32          $outchan_ptr_v, $mworker_base, WKR_STACK_PARTIALS_PTR/4
ld32          $inchan_ptr_v, $mvertex_base, WKR_INCHAN_PTR/4
ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4

ld32          $in_stride_v, $mvertex_base, WKR_IN_STRIDE/4
ldz16step     $out_delta_v, $mzero, $partition_v+=, 1
mul           $out_delta_v, $out_delta_v, CHAN_GROUPS_PER_GROUP
ld32step      $mzero, $mzero, $outchan_ptr_v+=, $out_delta_v

// reload the accumulators with current output values
ld64          $a0:1, $mzero, $outchan_ptr_v, 0
ld64          $a2:3, $mzero, $outchan_ptr_v, 1
{
  setzi         $out_delta_v, 0
  f32v4acc      $a0:3
}

PartitionLoop:
  LOAD_WORKLIST_ENTRY_EXCEPT_OUT SIZEOF_IN_ATOM SIZEOF_WEIGHTS_ATOM CHAN_GROUPS_PER_GROUP

  brz           $out_delta_v, L_Partition_Acc

  // store the accumulator result except if this is the very first pass
  {
    mul           $out_delta_v, $out_delta_v, CHAN_GROUPS_PER_GROUP / 2
    f32v2gina     $a4:5, $azeros, 0
  }

  // reload the accumulators with current output values
  {
    ld64          $a0:1, $mzero, $outchan_ptr_v, $out_delta_v
    f32v2gina     $a6:7, $azeros, 0
  }
  st64          $a6:7, $mzero, $outchan_ptr_v, 1
  st64step      $a4:5, $mzero, $outchan_ptr_v+=, $out_delta_v

  // continue reloading the accumulators
  ld64          $a2:3, $mzero, $outchan_ptr_v, 1

L_Partition_Acc:
  ld64step      $a4:5, $inchan_ptr_v, $in_offset_v+=, $in_stride_v
  ld64step      $a6:7, $weights_ptr_v, $weights_offset_v+=, $weights_stride_v
  {
    rpt          $num_fp_m1_v, (2f - 1f) / 8 - 1
    f32v4acc     $a0:3
  }
1:
    {
      ld64step      $a4:5, $inchan_ptr_v, $in_offset_v+=, $in_stride_v
      f16v4mul      $a0:1, $a4:5, $a6:7
    }
    {
      ld64step      $a6:7, $weights_ptr_v, $weights_offset_v+=, $weights_stride_v
      f16v4acc      $a0:1
    }
2:
  {
    ldz16step     $out_delta_v, $mzero, $partition_v+=, 1
    f16v4mul      $a0:1, $a4:5, $a6:7
  }
  {
    brnzdec       $num_partitions_v, PartitionLoop
    f16v4acc      $a0:1
  }

  // Store accumulator results as final output
  f32v2gina     $a0:1, $azeros, 0
  {
    st64          $a0:1, $mzero, $outchan_ptr_v, 0
    f32v2gina     $a2:3, $azeros, 0
  }
  st64          $a2:3, $mzero, $outchan_ptr_v, 1

L_Conv_end:
exitz         $mzero

.size convVerticalMacFlattened_half_float_4, . - convVerticalMacFlattened_half_float_4


// =============================================================================

// Instantiate supervisor codelet
CONV_VMAC_SUPERVISOR half float CHAN_GROUPS_PER_GROUP

#endif

// =============================================================================
// =============================================================================
