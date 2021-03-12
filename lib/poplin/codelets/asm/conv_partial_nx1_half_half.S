// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Computes an nx1 convolution using AMP. A contiguous field is partitioned
// between workers for each position of the kernel element.
//
// Requires a total stack size of 80 bytes in the supervisor
//
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_nx1_supervisor.S"

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// =============================================================================

// Worklist partition fields: must match the order of entries in worklists
#define PARTITION_OUTOFFSET             0    // half
#define PARTITION_NUM_ELEMS             2    // half
#define PARTITION_INOFFSET              4    // half

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
// Non loop overhead:
//      zero Partitions:         7
//      non-zero partitions:     17
//
// Loop performance:
//      Number of field elems = 0
//             18 cycles
//      Number of field elems = 1
//             31 cycles
//      Number of field elems = 2
//             35 cycles
//      Number of field elems >= 3
//             36 + (num_field_elems - 3) * 4

.section ".text.convPartialNx1Flattened_f16v4hihoamp", "ax"
.type convPartialNx1Flattened_f16v4hihoamp, @function
.align 8
.worker
convPartialNx1FlattenedAligned_f16v4hihoamp:
convPartialNx1Flattened_f16v4hihoamp:
// worker register mapping
#define wkr_id_v                       m0
#define partition_table_v              m1
#define partition_struct_v             m1
#define partition_v                    m1
#define num_partitions_v               m2
#define num_fp_v                       m3
#define inoutstrides1_v                m4
#define inoutstrides2_v                m5
#define inoutstrides3_v                m6
#define partition_base_v               m7
#define tripacked_addr                 m8:9
#define in_addr_v                      m8
#define in_offset_v                    m8
#define out_addr_v                     m9
#define out_offset_v                   m9
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11

get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK

// The number of input strides depends on the AMP height used. The height
// supported is 4 and hence there can be at most 4 input stides. There is
// only one output stride as the output channels are always grouped together.
// inoutstrides1_v = [0][in-stride][out-stride]
ld32          $inoutstrides1_v, $mvertex_base, WKR_INOUTSTRIDES1/4
// inoutstrides2_v = [in-row-stride1][in-row-stride0][out-stride]
ld32          $inoutstrides2_v, $mvertex_base, WKR_INOUTSTRIDES2/4
// inoutstrides3_v = [0][in-row-stride2][out-stride]
ld32          $inoutstrides3_v, $mvertex_base, WKR_INOUTSTRIDES3/4

ld32          $partition_base_v, $mvertex_base, WKR_PARTITION_BASE/4

convPartialNx1FlattenedStateRetained_f16v4hihoamp:
ld32          $partition_table_v, $mvertex_base, WKR_PARTITION_PTR/4
ld32          $partition_struct_v, $partition_table_v, $wkr_id_v

// extract offset and delta: upper 14 bits gives number of partitions
// lower 18 bits give offset to common base for the vector list
shr           $num_partitions_v, $partition_struct_v, DELTAN_OFFSET_BITS
// exit if no work to be done for worker
brz           $num_partitions_v, L_Conv_end
shl           $partition_v, $partition_struct_v, DELTAN_COUNT_BITS
// Offset needs to be converted from elements to bytes.
// Hence shift right is less by ELEM_TO_BYTES_CONV
shr           $partition_v, $partition_v, (DELTAN_COUNT_BITS - ELEM_TO_BYTES_CONV)
add           $partition_v, $partition_v, $partition_base_v

ld32          $inchan_ptr_v, $mvertex_base, WKR_INCHAN_PTR/4
ld32          $outchan_ptr_v, $mvertex_base, WKR_OUTCHAN_PTR/4

// This following approximates num_partitions/3 - 1 for values
// [3:3:2^14-1]. The case of zero is handled above
{
  mul           $num_partitions_v, $num_partitions_v, 21845
  setzi         $a0, ZAACC_BITMASK
}
{
  shr           $num_partitions_v, $num_partitions_v, 16
  uput          $FP_CLR, $a0
}
// Code fragments to handle number of field samples equal to 1 and 2 are
// handled differently inorder to avoid excess loads which could potentially
// cause memory conflicts given that striding on both inputs and output are
// used. It is possible to use the same code for all cases but this would
// require a lot more stride registers and selective setting of them depending
// on number of field samples.
PartitionLoop:
  lds16         $num_fp_v, $partition_v, PARTITION_NUM_ELEMS/2

  // form input address
  ldz16         $in_offset_v, $partition_v, PARTITION_INOFFSET/2
  shl           $in_offset_v, $in_offset_v, 3
  add           $in_addr_v, $inchan_ptr_v, $in_offset_v

  // form output address
  // Note this relies on the fact that out offset is the first entry in the
  // partition. This allows us to wind the pointer to the next partition
  ldz16step     $out_offset_v, $mzero, $partition_v+=, 3

  shl           $out_offset_v, $out_offset_v, 3
  add           $out_addr_v, $outchan_ptr_v, $out_offset_v

  // Form packed address
  tapack        $tripacked_addr, $in_addr_v, $out_addr_v, $out_addr_v
  // *input-ptr += 0; *partials-ptr += 1
  ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $mzero, 0b0011
  f16v4hihoamp  $a6, $azeros, $a2, TAMP_F16V4_E4_P0

  {
    // *input-ptr += 0; *partials-ptr += outstride
    ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0111
    f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P1
  }
  {
    // jump to specialisation for number of field samples equal to 0, 1 and 2
    brneg         $num_fp_v, ConvNumFpLt3
    f16v4hihoamp  $a6, $azeros, $a2, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1 (2 elements)
    ld64a32pace   $a0:1, $a2, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-row-stride1; *partials-ptr += 1 (2 elements)
    ld64a32pace   $a0:1, $a3, $tripacked_addr+=, $inoutstrides2_v, 0b0011
    f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride2; *partials-ptr += outstride
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides3_v, 0b0110
    f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
  }
  {
    // *input-ptr += in-stride; *partials_ptr += 0
    ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides1_v, 0b1110
    f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1 (2 elements)
    ld64a32pace   $a0:1, $a2, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-row-stride1; *partials_ptr += 1 (2 elements)
    ld64a32pace   $a0:1, $a3, $tripacked_addr+=, $inoutstrides2_v, 0b0011
    f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride2; *partials-ptr += outstride
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides3_v, 0b0110
    f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
  }
  rpt $num_fp_v, (Loop_end_Amp-Loop_start_Amp)/8-1
Loop_start_Amp:
    {
      // *input-ptr += in-stride; *out-ptr += 1
      ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $inoutstrides1_v, 0b0010
      f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
    }
    {
      // *input-ptr += in-row-stride0; *partials-ptr += 1
      ld2x64pace  $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
      f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
    }
    {
      // *input-ptr += in-row-stride1; *out-ptr += out-stride
      ldst64pace  $a0:1, $a6:7, $tripacked_addr+=, $inoutstrides2_v, 0b0111
      f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
    }
    {
      // *input-ptr += in-row-stride2; *partials-ptr += out-stride
      ld2x64pace  $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides3_v, 0b0110
      f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
    }
Loop_end_Amp:
  {
    // *input-ptr += in-stride; out-ptr += 0
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $inoutstrides1_v, 0b1110
    f16v4hihoamp  $a4, $a0:1, $a2, TAMP_F16V4_E4_P2
  }
  {
    // input-ptr += in-row-stride0;  out-ptr += 1
    // out-ptr written again with same value. Done to avoid overreading
    // partials
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4hihoamp  $a5, $a0:1, $a3, TAMP_F16V4_E4_P3
  }
  {
    // input-ptr += in-row-stride1;  out-ptr += out-stride
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $inoutstrides2_v, 0b0111
    f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P0
  }
  {
    // input-ptr += in-row-stride2;  out-ptr += 0
    // out-ptr written again with the same value. Required to use a pace instr
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $inoutstrides3_v, 0b1110
    f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P1
  }
StoreFinalAmpOutputs2:
  {
    // *input-ptr += 1; *out-ptr += 1
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $mzero, 0b0000
    f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P2
  }
  f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P3
  {
    // *out-ptr += out-stride
    st64pace  $a6:7, $tripacked_addr+=, $inoutstrides1_v, 0b01
    f16v4hihoamp  $a6, $azeros, $azero, TAMP_F16V4_E4_P0
  }
  f16v4hihoamp  $a7, $azeros, $azero, TAMP_F16V4_E4_P1
StoreFinalAmpOutputs1:
  // The partials for the next iteration may be loaded here but would require
  // the input and output addresses to be computed.
  {
    // *out-ptr += 1
    st64pace      $a6:7, $tripacked_addr+=, $mzero, 0b00
    f16v4hihoamp  $a6, $azeros, $azero, TAMP_F16V4_E4_P2
  }
  f16v4hihoamp  $a7, $azeros, $azero, TAMP_F16V4_E4_P3
  // *out-ptr += 1
  st64pace      $a6:7, $tripacked_addr+=, $mzero, 0b00

L_Partition_end:
  brnzdec       $num_partitions_v, PartitionLoop

L_Conv_end:
exitz         $mzero

// =============================================================================

// This handles the case of number of field positions equal to 1. The first
// three partials are already assumed to be loaded and fed to the AMP
// 6 extra partials are loaded to allow use of pace instruction all with
// post increment of 1
ConvNumFpLt3:
add           $num_fp_v, $num_fp_v, 1
brz           $num_fp_v, ConvNumFpEq2
add           $num_fp_v, $num_fp_v, 1
brneg         $num_fp_v, L_Partition_end

// we need a zero as stride alongwith store partials pointer
#define inoutstrides2_v_tmp  num_fp_v
shr $inoutstrides2_v_tmp, $inoutstrides2_v, 10

{
  // *input-ptr += in-row-stride0; *out-ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_v_tmp, 0b1101
  f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P3
}
{
  // *input-ptr += in-row-stride1; *out-ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_v_tmp, 0b1110
  f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride2; *out-ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides3_v, 0b1110
  f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += in-stride; *out_ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides1_v, 0b1110
  f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P2
}
f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P3
f16v4hihoamp  $a6, $azeros, $azero, TAMP_F16V4_E4_P0
{
  // Jump to common part to store final samples
  bri           StoreFinalAmpOutputs1
  f16v4hihoamp  $a7, $azeros, $azero, TAMP_F16V4_E4_P1
}

// =============================================================================


// This handles the case of number of field positions equal to 2. The first
// seven partials and the first three input data are assumed to be loaded and
// fed to the AMP
// 6 extra partials are already loaded with an increment of 1.

// clear output stride to allow post increment by 0 in the dummy loads of
// partials
ConvNumFpEq2:
{
  // *input-ptr += in-row-stride0; *partials-ptr += 1
  ld64a32pace   $a0:1, $a2, $tripacked_addr+=, $inoutstrides2_v, 0b0010
  f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P3
}
{
  // *input-ptr += in-row-stride1; partials-ptr += 1
  ld64a32pace   $a0:1, $a3, $tripacked_addr+=, $inoutstrides2_v, 0b0011
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride2; *partials-ptr += 1
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides3_v, 0b0010
  f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
}

#define inoutstrides2_eq2_v   num_fp_v
andc $inoutstrides2_eq2_v, $inoutstrides2_v, 0x3ff

{
  // *input-ptr += in-stride; *partials_ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides1_v, 0b1110
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
}
{
  // *input-ptr += in-row-stride0; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_eq2_v, 0b0110
  f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
}
{
  // input-ptr += in-row-stride1; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_eq2_v, 0b0111
  f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P0
}
{
  // input-ptr += in-row-stride2; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides3_v, 0b1110
  f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P1
}
bri           StoreFinalAmpOutputs2

.size convPartialNx1Flattened_f16v4hihoamp, . - convPartialNx1Flattened_f16v4hihoamp

// =============================================================================
// Instantiate codelets

CONV_Nx1_SUPERVISOR half half false 8 f16v4hihoamp
CONV_Nx1_SUPERVISOR half half true  8 f16v4hihoamp

// =============================================================================
#endif
// =============================================================================
