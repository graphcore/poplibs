// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
// Common headers for the HZMAC codelets
//

#ifndef conv_hzmac_common_h_s
#define conv_hzmac_common_h_s
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"

// =============================================================================


// Worklist partition fields: must match the order of entries in worklists
#define PARTITION_ELEM_BYTES            2
#define PARTITION_OUTOFFSET             0    // short
#define PARTITION_NUM_ELEMS             2    // short
#define PARTITION_INOFFSET              4    // short
#define PARTITION_SIZE                 ((PARTITION_INOFFSET + PARTITION_ELEM_BYTES)/PARTITION_ELEM_BYTES)


// =============================================================================

// Supervisor vertex state: offsets and the number must match vertex field
// ordering and sizes
#if defined(VECTORLIST_AVAIL_DELTAN)
#define SUP_INCHAN_VECTORS              0    // word
#define SUP_WEIGHTS_VECTORS             4    // word
#define SUP_OUTCHAN_VECTORS             8    // word
#define SUP_ZERO_INFO                   12   // word
#define SUP_PARTITION_TABLES            16   // VectorList::DELTA
#define SUP_NUM_OUTCHAN_M1              22   // short
#define SUP_INSTRIDE                    24   // word
#define SUP_OUTSTRIDE                   28   // word
#define SUP_NUM_INCHAN                  32   // short
#define SUP_NUM_KERNEL_M1               34   // short (K-1)
#define SUP_NUM_CONVGROUPS_M1           36   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      38   // short
#define SUP_NUM_INCHANS_PER_GROUP       40   // short

#else
#define SUP_INCHAN_VECTORS              0    // word
#define SUP_WEIGHTS_VECTORS             4    // word
#define SUP_OUTCHAN_VECTORS             8    // word
#define SUP_ZERO_INFO                   12   // word
#define SUP_PARTITION_TABLES            16   // VectorList::DeltaNElements
#define SUP_NUM_OUTCHAN_M1              24   // short
#define SUP_INSTRIDE                    28   // word
#define SUP_OUTSTRIDE                   32   // word
#define SUP_NUM_INCHAN                  36   // short
#define SUP_NUM_KERNEL_M1               38   // short (K-1)
#define SUP_NUM_CONVGROUPS_M1           40   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      42   // short
#define SUP_NUM_INCHANS_PER_GROUP       44   // short
#endif

// DeltaN decoding constants
#if defined(VECTORLIST_AVAIL_DELTAN)
#define SCALED_PTR32_SHL         2
#define DELTAN_DELTAS_COUNT_BITS 12
#define DELTAN_OFFSET_BITS       18
#define DELTAN_COUNT_BITS        14
#define ELEM_TO_BYTES_CONV       0
#else
#define DELTAN_DELTAS_COUNT_BITS 8
#define DELTAN_OFFSET_BITS       20
#define DELTAN_COUNT_BITS        (32 - DELTAN_OFFSET_BITS)
#define ELEM_TO_BYTES_CONV       (21 - DELTAN_OFFSET_BITS)
#endif

// =============================================================================

// Vertex state shared between workers
// The supervisor sets up a common state for workers to use
// Allocate codelet specific worker stack after zero output worker (WKR_ZERO_OUTPUT_STACK)
#define WKR_INSTRIDE                    WKR_ZERO_OUTPUT_STACK
#define WKR_OUTSTRIDE                   (WKR_INSTRIDE + 4)
#define WKR_INCHAN_PTR                  (WKR_OUTSTRIDE + 4)
#define WKR_PARTITION_PTR               (WKR_INCHAN_PTR + 4)
#define WKR_OUTCHANS_PER_GROUP_IN_BYTES (WKR_PARTITION_PTR + 4)
#define WKR_PARTITION_BASE              (WKR_OUTCHANS_PER_GROUP_IN_BYTES + 4)
#define WKR_INCHANS_PER_GROUP_IN_BYTES  (WKR_PARTITION_BASE + 4)
#define WKR_OCG_COUNT                   (WKR_INCHANS_PER_GROUP_IN_BYTES + 4)
#define WKR_ICG_COUNT                   (WKR_OCG_COUNT + 4)
#define WKR_WEIGHTS_PTR                 (WKR_ICG_COUNT + 4)
#define WKR_STATE_SIZE                  (WKR_WEIGHTS_PTR + 4) // bytes

// =============================================================================

// worker register mapping
#define wkr_id_v                       m11 //retained
#define instride_v                     m0  //retained
#define partition_table_v              m3
#define partition_struct_v             m3
#define partition_v                    m3
#define out_addr_v                     m1
#define num_partitions_v               m2
#define num_fp_v                       m5
#define outstride_v                    m4 //retained
#define in_offset_v                    m5
#define out_offset_v                   m5
#define out_chans_per_group_in_bytes   m6
#define ocg_count_v                    m6
#define partition_base_v               m10 //retained
#define icg_count_v                    m7
#define in_chans_per_group_in_bytes    m7
#define fp_clr_reg                     a0
#define in_addr_v                      m8
#define weights_ptr_v                  m9
#define tripacked_addr                 m8:9

// =============================================================================

// Stack parameters
#define WKR_STACK_INADDR                4


// =============================================================================

// Macro to prepare vertex state. Vertex state is ordered such that retained
// state is loaded first
.macro PREPARE_FOR_PARTITION_LOOP VERTEX_NAME
get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK
ld32          $partition_base_v, $mvertex_base, WKR_PARTITION_BASE/4
add           $partition_base_v, $partition_base_v, -PARTITION_SIZE*PARTITION_ELEM_BYTES
ld32          $instride_v, $mvertex_base, WKR_INSTRIDE/4
ld32          $outstride_v, $mvertex_base, WKR_OUTSTRIDE/4

\VERTEX_NAME\()_retained:
ld32          $partition_table_v, $mvertex_base, WKR_PARTITION_PTR/4
ld32          $partition_struct_v, $partition_table_v, $wkr_id_v

// extract offset and delta: upper X bits gives number of partitions
// lower Y bits give offset to common base for the vector list
// For DeltaN: X = 18, Y = 14
// For DeltaNElements: X and Y are based on memory alignment used
shr           $num_partitions_v, $partition_struct_v, DELTAN_OFFSET_BITS
// exit if no work to be done for worker
brz           $num_partitions_v, LConvEnd
shl           $partition_v, $partition_struct_v, DELTAN_COUNT_BITS

// Offset needs to be converted from elements to bytes.
// Hence shift rigth is less by ELEM_TO_BYTES_CONV
shr           $partition_v, $partition_v, (DELTAN_COUNT_BITS - ELEM_TO_BYTES_CONV)
{
  add           $partition_v, $partition_v, $partition_base_v
  setzi         $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}
{
  shl           $num_partitions_v, $num_partitions_v, 1
  // clear accumulator: only required once because we always feed zeros
  // with gina instruction
  uput          $FP_CLR, $fp_clr_reg
}
.endm // PREPARE_FOR_PARTION_LOOP



// Set up partition for doing the main channel and field loops
.macro PARTITION_LOOP_SETUP 

  // we are addressing partitions from last to first. Move pointer to the previous
  // partition. The other elements in the partition are addressed with an
  // offset of a partition.
  ldz16step     $out_offset_v, $partition_v, $num_partitions_v+=, -PARTITION_SIZE

  ld32          $in_addr_v, $mvertex_base, WKR_INCHAN_PTR/4
  ld32          $out_addr_v, $mvertex_base, WKR_OUTCHAN_PTR/4

  ld32          $in_chans_per_group_in_bytes, $mvertex_base, WKR_INCHANS_PER_GROUP_IN_BYTES/4
  ld32          $out_chans_per_group_in_bytes, $mvertex_base, WKR_OUTCHANS_PER_GROUP_IN_BYTES/4


  mul           $out_offset_v, $out_offset_v, $out_chans_per_group_in_bytes
  add           $out_addr_v, $out_addr_v, $out_offset_v

  // form input address
  ldz16         $in_offset_v, $partition_v, $num_partitions_v, (PARTITION_SIZE + PARTITION_INOFFSET/PARTITION_ELEM_BYTES)
  mul           $in_offset_v, $in_offset_v, $in_chans_per_group_in_bytes
  add           $in_addr_v, $in_addr_v, $in_offset_v

  lds16         $num_fp_v, $partition_v,  $num_partitions_v, (PARTITION_SIZE + PARTITION_NUM_ELEMS/PARTITION_ELEM_BYTES)
  brnzdec       $num_fp_v, LNonZeroFieldPos
  bri           LPartitionEnd
LNonZeroFieldPos:  
  st32          $in_addr_v, $mworker_base, WKR_STACK_INADDR/4

  and           $in_chans_per_group_in_bytes, $in_chans_per_group_in_bytes, 0x7
  brnz          $in_chans_per_group_in_bytes, LProcessTwoInputsFp

  
  // Process a vector of 4 input channels at a time.
  // Optim: We could still use this for input channels > 4 by doing residual
  //        channels. This would require checks on stride as well.
  ld32          $icg_count_v, $mvertex_base, WKR_ICG_COUNT/4
.endm // PARTITION_LOOP_SETUP
 
#endif // __IPU__
#endif // conv_hzmac_common_h_s

// =============================================================================
// =============================================================================
