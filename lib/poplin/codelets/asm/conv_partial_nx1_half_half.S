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
//      zero Partitions:         13
//      non-zero partitions:     23
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
nop // rpt alignment
// worker register mapping
#define wkr_id_v                       m0   // retained
#define partition_table_v              m1
#define partition_struct_v             m1
#define partition_v                    m1
#define num_partitions_v               m2
#define num_fp_v                       m3
#define par_addr_v                     m3
#define inoutstrides1_v                m4   // retained
#define inoutstrides2_v                m5   // retained
#define tripacked_addr2                m6:7
#define in_addr_v                      m6
#define out_addr_v                     m7
#define tripacked_addr1                m8:9
#define tripacked_addr                 m8:9
#define in_offset_v                    m8
#define out_addr_adj_v                 m8
#define out_offset_v                   m9
#define inoutstrides3_v                m10 // retained
#define second_ptr_offset_v            m11 
#define partition_base_v               m11 // retained


get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK

// The number of input strides depends on the AMP height used. The height
// supported is 4 and hence there can be at most 4 input stides. There is
// only one output stride as the output channels are always grouped together.
// inoutstrides1_v = [0][in-stride][out-stride]
ld32          $inoutstrides1_v, $mvertex_base, WKR_INOUTSTRIDES1/4
// inoutstrides2_v = [0][in-row-stride][out-stride]
ld32          $inoutstrides2_v, $mvertex_base, WKR_INOUTSTRIDES2/4

ld32          $second_ptr_offset_v, $mvertex_base, WKR_SECOND_PTR_OFFSET/4

// Special stride for initial preloads and wind downs. Form 3 strides in 
// a register: [second_ptr_offset_v | 0x0 | inoutstrides3_v]
and           $inoutstrides3_v, $inoutstrides1_v, 0x3FF
shl           $second_ptr_offset_v, $second_ptr_offset_v, 20
or            $inoutstrides3_v, $inoutstrides3_v, $second_ptr_offset_v

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
  ldz16step     $out_offset_v, $partition_base_v, $partition_v+=, 2

  // form input address
  ld32          $in_addr_v, $mvertex_base, WKR_INCHAN_PTR/4
  ldz16step     $in_offset_v, $partition_base_v, $partition_v+=, -1
  // dummy load to do in_addr_v = in_addr_v + in_offset_v*8
  ld64step      $azeros, $mzero,  $in_addr_v+=, $in_offset_v

  // form output address
  ld32          $out_addr_v, $mvertex_base, WKR_OUTCHAN_PTR/4
  ld64step      $azeros, $mzero, $out_addr_v+=, $out_offset_v 

  // Move partials-out1 pointer by one write (16v4) to get correct 
  // store alignment
  add           $out_addr_adj_v, $out_addr_v, 0x8

  // Form packed address
  tapack        $tripacked_addr1, $in_addr_v, $out_addr_v, $out_addr_adj_v

  // Move partials-in2 pointer by one read (16v4) to get correct 
  // load alignment
  add           $par_addr_v, $out_addr_v, 0x8
  tapack        $tripacked_addr2, $in_addr_v, $par_addr_v, $out_addr_v

  lds16step     $num_fp_v, $partition_base_v, $partition_v+=, 2

  // *input-ptr += 0; *partials-ptr += out-stride (4 partials)
  ld2x64pace    $azeros, $a2:3, $tripacked_addr1+=, $inoutstrides3_v, 0b0110
  f16v4hihoamp  $a6, $azeros, $a2, TAMP_F16V4_E4_P0

  {
    // *input-ptr += $second_ptr_offset_v; *partials-ptr += out-stride (4 partials)
    ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides3_v, 0b0111
    f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P1
  }
  {
    // jump to specialisation for number of field samples equal to 0, 1 and 2
    brneg         $num_fp_v, ConvNumFpLt3
    f16v4hihoamp  $a6, $azeros, $a2, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += 0
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
    f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-stride; *partials_ptr += out-stride
    ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
    f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += 0
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
    f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
  }
  {
    // *input-ptr += in-stride; *partials_ptr += out-stride
    ld2x64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides1_v, 0b0110
    f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += 0
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
    f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-stride; *partials_ptr += out-stride
    ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
    f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += out-stride
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0110
    f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
  }
  rpt $num_fp_v, (Loop_end_Amp-Loop_start_Amp)/8-1
Loop_start_Amp:
    {
      // *input-ptr += in-stride;
      // *out-ptr += out-stride
      ldst64pace    $a0:1, $a6:7, $tripacked_addr2+=, $inoutstrides1_v, 0b0110
      f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
    }
    {
      // *input-ptr += in-row-stride; 
      // *partials-ptr += out-stride
      ld2x64pace  $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b0110
      f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
    }
    {
      // *input-ptr += in-stride; 
      // *out-ptr += out-stride
      ldst64pace  $a0:1, $a6:7, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
      f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
    }
    {
      // *input-ptr += in-row-stride; 
      // *partials-ptr += out-stride
      ld2x64pace  $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0110
      f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
    }
Loop_end_Amp:
  {
    // *input-ptr += in-stride; out-ptr += out-stride;
    ldst64pace    $a0:1, $a6:7, $tripacked_addr2+=, $inoutstrides1_v, 0b0110
    f16v4hihoamp  $a4, $a0:1, $a2, TAMP_F16V4_E4_P2
  }
  {
    // input-ptr += in-row-stride;  out-ptr += 0
    ldst64pace    $a0:1, $a6:7, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
    f16v4hihoamp  $a5, $a0:1, $a3, TAMP_F16V4_E4_P3
  }
  {
    // input-ptr += in-stride;  out-ptr += out-stride
    ldst64pace    $a0:1, $a4:5, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
    f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P0
  }
  {
    // input-ptr += in-row-stride;  out-ptr += 0
    ldst64pace    $a0:1, $a4:5, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
    f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P1
  }
StoreFinalAmpOutputs2:
  {
    // *input-ptr += 1; *out-ptr += out-stride
    ldst64pace    $a0:1, $a6:7, $tripacked_addr2+=, $inoutstrides1_v, 0b0100
    f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P2
  }
  f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P3
  {
    // *out-ptr += out-stride
    st64pace  $a6:7, $tripacked_addr1+=, $inoutstrides1_v, 0b01
    f16v4hihoamp  $a6, $azeros, $azero, TAMP_F16V4_E4_P0
  }
  f16v4hihoamp  $a7, $azeros, $azero, TAMP_F16V4_E4_P1
StoreFinalAmpOutputs1:
  // The partials for the next iteration may be loaded here but would require
  // the input and output addresses to be computed.
  {
    // *out-ptr += 0
    st64pace      $a6:7, $tripacked_addr2+=, $mzero, 0b11
    f16v4hihoamp  $a6, $azeros, $azero, TAMP_F16V4_E4_P2
  }
  f16v4hihoamp  $a7, $azeros, $azero, TAMP_F16V4_E4_P3
  // *out-ptr += 0
  st64pace      $a6:7, $tripacked_addr1+=, $mzero, 0b11

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

{
  // *input-ptr += in-row-stride0; *out-ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
  f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P3
}
{
  // *input-ptr += in-stride; *out-ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b1110
  f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride; *out-ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
  f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += in-stride; *out_ptr += 0
  ldst64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides1_v, 0b1110
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
  // *input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
  f16v4hihoamp  $a7, $azeros, $a3, TAMP_F16V4_E4_P3
}
{
  // *input-ptr += in-stride; partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b1110
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
  f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += in-stride; *partials_ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides1_v, 0b1110
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
}
{
  // *input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
  f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
}
{
  // input-ptr += in-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b1110
  f16v4hihoamp  $a6, $a0:1, $azero, TAMP_F16V4_E4_P0
}
{
  // input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
  f16v4hihoamp  $a7, $a0:1, $azero, TAMP_F16V4_E4_P1
}
bri           StoreFinalAmpOutputs2

.size convPartialNx1Flattened_f16v4hihoamp, . - convPartialNx1Flattened_f16v4hihoamp

// =============================================================================
// Instantiate codelets

CONV_Nx1_SUPERVISOR half half false 8 false f16v4hihoamp
CONV_Nx1_SUPERVISOR half half true  8 false f16v4hihoamp
CONV_Nx1_SUPERVISOR half half false 8 true f16v4hihoamp
CONV_Nx1_SUPERVISOR half half true  8 true f16v4hihoamp

// =============================================================================
#endif
// =============================================================================
