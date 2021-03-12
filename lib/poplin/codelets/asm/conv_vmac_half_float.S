// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Computes an nx1 convolution using Vertical MAC. A contiguous field is partitioned
// for each position of the output element. The partitions are distributed between
// between workers. Each worker is provided a temporary tensor to store its VMAC outputs.
// The worker outputs are finally reduced and stored in the user-specified output buffer.
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define conv_vmac_sup_half_float __runCodelet_poplin__ConvPartialVerticalMac___half_float_true_4

// =============================================================================


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
#define WKR_STACK_NUM_ELEMS_X4           (WKR_STACK_TOTAL_ELEMS_X2 + 4)
#define WKR_STACK_REDUCTION_PARTIALS_PTR (WKR_STACK_NUM_ELEMS_X4 + 4)
#define WKR_STACK_REDUCTION_ELEM_OFFSET  (WKR_STACK_REDUCTION_PARTIALS_PTR + 4)
#define WKR_STACK_COMMON                 (WKR_STACK_REDUCTION_ELEM_OFFSET + 4)

// =============================================================================
// Performance:
// 9 + (1 * num_elements)
.section ".text.convVerticalMacWorkerStateRetention_float", "ax"
.type convVerticalMacWorkerStateRetention_float, @function
.align 8
.worker
convVerticalMacWorkerStateRetention_float:
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

// Save worker state for later calls to convVerticalMacZeroPartials_float
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

// Save worker state for later calls to convVerticalMacReduce_float

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

convVerticalMacZeroPartials_float:
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

.size convVerticalMacWorkerStateRetention_float, . - convVerticalMacWorkerStateRetention_float

// =============================================================================
// Performance:
// 24 + ((9 + (2 * num_contexts)) * (n/4))
.section ".text.convVerticalMacReduce_float", "ax"
.type convVerticalMacReduce_float, @function
.align 8
.worker
convVerticalMacReduce_float:
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
.size convVerticalMacReduce_float, . - convVerticalMacReduce_float

// =============================================================================
// Performance:
//   one_time_overhead + 16 + load_acc + (20 + 2*(num_fp_m1_v + 1)))*num_partitions_v + reload_acc*num_outputs_processed
//     where one_time_overhead= 13
//     where load_acc = 4 cycles
//     where reload_acc = 7 cycles
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
#define partition_v                    m2
#define num_fp_m1_v                    m3
#define num_partitions_v               m4
#define prev_out_offset                m5
#define weights_ptr_v                  m6
#define partition_base_v               m7
#define in_offset_v                    m7
#define out_offset_v                   m8
#define weights_offset_v               m9
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11
#define fp_clr_reg                     a0

// This is in the worker stack
#define WKR_STACK_TOTAL_PARTITIONS      WKR_STACK_COMMON
#define WKR_STACK_NUM_PARTITIONS        (WKR_STACK_TOTAL_PARTITIONS + 4)
#define WKR_STACK_PARTITION_BASE        (WKR_STACK_NUM_PARTITIONS + 4)

// macro to store accumulators to 64-bit memory
.macro STORE_ACC_FLOAT base, addr
  f32v2gina     $a0:1, $azeros, 0
  {
    st64          $a0:1, \base, \addr, 0
    f32v2gina     $a2:3, $azeros, 0
  }
  st64          $a2:3, \base, \addr, 1
.endm

// load worklist entry parametrs into registers
.macro LOAD_WORKLIST_ENTRY index
  // form input address
  ldz16         $in_offset_v, $partition_v, PARTITION_IN_OFFSET/2
  mul           $in_offset_v, $in_offset_v, SIZEOF_IN_ATOM * CHAN_GROUPS_PER_GROUP

  // form weights address
  ldz16         $weights_offset_v, $partition_v, PARTITION_WEIGHTS_OFFSET/2
  mul           $weights_offset_v, $weights_offset_v, SIZEOF_WEIGHTS_ATOM * CHAN_GROUPS_PER_GROUP

  // obtain number of consecutive accumulations
  // Note this relies on the fact that out offset is the first entry in the
  // partition. This allows us to wind the pointer to the next partition
  {
    ldz16         $num_fp_m1_v, $partition_v, PARTITION_NUM_ELEMS/2
    mov           $a0:1, $azeros
  }
  {
    ldz16step     $out_offset_v, $mzero, $partition_v+=, 4
    mov           $a2:3, $azeros
  }

  // form output address and
  mul           $out_offset_v, $out_offset_v, SIZEOF_OUT_ATOM * CHAN_GROUPS_PER_GROUP
.endm


.section ".text.convVerticalMacFlattened_half_float", "ax"
.type convVerticalMacFlattened_half_float, @function
.align 8
.worker
convVerticalMacFlattened_half_float:
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

// This following approximates num_partitions/3 - 1 for values
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

st32          $num_partitions_v, $mworker_base, WKR_STACK_NUM_PARTITIONS/4
st32          $partition_v, $mworker_base, WKR_STACK_PARTITION_BASE/4

convVerticalMacFlattenedReentry_half_float:

// Compensate for extra load in the loops. This could be done in the supervisor
ld32          $num_partitions_v, $mworker_base, WKR_STACK_TOTAL_PARTITIONS/4
brz           $num_partitions_v, L_Conv_end
ld32          $num_partitions_v, $mworker_base, WKR_STACK_NUM_PARTITIONS/4
add           $num_partitions_v, $num_partitions_v, -1
ld32          $weights_stride_v, $mvertex_base, WKR_WEIGHTS_STRIDE/4
ld32          $partition_v, $mworker_base, WKR_STACK_PARTITION_BASE/4

// load previously stored partials address
ld32          $outchan_ptr_v, $mworker_base, WKR_STACK_PARTIALS_PTR/4
ld32          $inchan_ptr_v, $mvertex_base, WKR_INCHAN_PTR/4
ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4

LOAD_WORKLIST_ENTRY 1
bri           L_Partition_load_acc

PartitionLoop:
  LOAD_WORKLIST_ENTRY 2

  // detect if the output has changed. Store accumulator to output and reload
  // the accumulator with the new output entry.
  cmpne         $mscratch, $prev_out_offset, $out_offset_v
  brz           $mscratch, L_Partition_Acc

  // store the accumulator result except if this is the very first pass
  STORE_ACC_FLOAT $outchan_ptr_v, $prev_out_offset

L_Partition_load_acc:
  // reload the accumulators with current output values
  ld64          $a0:1, $outchan_ptr_v, $out_offset_v, 0
  ld64          $a2:3, $outchan_ptr_v, $out_offset_v, 1

L_Partition_Acc:
  ld32          $in_stride_v, $mvertex_base, WKR_IN_STRIDE/4
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
    mov           $prev_out_offset, $out_offset_v
    f16v4mul      $a0:1, $a4:5, $a6:7
  }
L_Partition_end:
   {
     brnzdec       $num_partitions_v, PartitionLoop
     f16v4acc      $a0:1
   }

  // Store accumulator results as final output
  STORE_ACC_FLOAT $outchan_ptr_v, $prev_out_offset

L_Conv_end:
exitz         $mzero

.size convVerticalMacFlattened_half_float, . - convVerticalMacFlattened_half_float


// =============================================================================


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

DEF_STACK_USAGE  TOT_STACK_SIZE  conv_vmac_sup_half_float
.section .text.conv_vmac_sup_half_float
.align 4
.type conv_vmac_sup_half_float, @function
.globl conv_vmac_sup_half_float
.supervisor

conv_vmac_sup_half_float:
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
brz           $worklist_count_s,  L_sup_conv_end
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
setzi         $wkr_core_s, convVerticalMacFlattened_half_float
st32          $partials_vectors_z_s, $wkr_base, WKR_PARTIALS_PTR/4
#if defined(VECTORLIST_AVAIL_DELTAN)
shr           $partition_ptr_s, $partition_ptr_s, 13
#else
shr           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
#endif
setzi         $wkr_zero_s, convVerticalMacWorkerStateRetention_float //convVerticalMacZeroPartials_float
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
add           $inchan_count_s, $inchan_count_s, -1
ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4

ConvGroupLoop:
  sync          TEXCH_SYNCZONE_LOCAL
  runall        $wkr_zero_s, $wkr_base, 0
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $outchan_ptr_s, $mzero, $outchan_vectors_s+=, 1
#endif
  setzi         $wkr_reduction_s, convVerticalMacReduce_float
InChanLoop:
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
    setzi         $wkr_core_s, convVerticalMacFlattenedReentry_half_float
    brnzdec       $inchan_count_s, InChanLoop
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
  setzi         $wkr_zero_s, convVerticalMacZeroPartials_float
  add           $inchan_count_s, $inchan_count_s, -1
  brnzdec       $convgroup_count_s, ConvGroupLoop

L_sup_conv_end:

// Restore saved registers
ld32          $m9, $sp, WKR_STATE_SIZE/4 + 0
ld32          $m10, $sp, WKR_STATE_SIZE/4 + 1
add           $sp, $sp, TOT_STACK_SIZE
sync          TEXCH_SYNCZONE_LOCAL
br            $lr
.size conv_vmac_sup_half_float, . - conv_vmac_sup_half_float

#endif

// =============================================================================
// =============================================================================
