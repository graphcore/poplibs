// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

// Non loop overhead:
//      zero Partitions:         13
//      non-zero partitions:     23
//
// Loop performance:
//      Number of field elems = 0
//             18 cycles
//      Number of field elems = 1
//             34 cycles
//      Number of field elems = 2
//             45 cycles
//      Number of field elems >= 3
//             46 + (num_field_elems - 3) * 8

.section ".text.convPartialNx1Flattened_f32sisov2amp", "ax"
.type convPartialNx1Flattened_f32sisov2amp, @function
.align 8
convPartialNx1FlattenedAligned_f32sisov2amp:
convPartialNx1Flattened_f32sisov2amp:
nop // rpt alignment
// worker register mapping
#define wkr_id_v                       m0   // retained
#define partition_table_v              m1
#define partition_struct_v             m1
#define partition_v                    m1
#define num_partitions_v               m2
#define num_fp_v                       m3
#define inoutstrides3_tmp_v            m3
#define second_ptr_offset_v            m3
#define inoutstrides1_v                m4   // retained
#define inoutstrides2_v                m5   // retained
#define tripacked_addr2                m6:7
#define tripacked_addr2_hi             m7
#define in_addr_v                      m6
#define out_addr_v                     m7
#define tripacked_addr1                m8:9
#define tripacked_addr                 m8:9
#define in_offset_v                    m8
#define out_offset_v                   m9
#define inoutstrides3_v                m10 // retained
#define partition_base_v               m11 // retained

#define X_IN_0                         $a0
#define X_IN_1                         $a1
#define X_IN                           $a0:1
#define P_IN                           $a2:3
#define X_IN__P_IN                     $a0:3
#define P_OUT                          $a4:5
#define P2_OUT                         $a6:7

#define NULL2                          $a10:11
#define NULL1                          $a14
#define NULL                           $azeros

get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK

// inoutstrides1_v = [out-stride-step][in-stride][out-stride-p4]
ld32          $inoutstrides1_v, $mvertex_base, WKR_INOUTSTRIDES1/4
// inoutstrides2_v = [0][in-row-stride][out-stride]
ld32          $inoutstrides2_v, $mvertex_base, WKR_INOUTSTRIDES2/4
and           $inoutstrides3_v, $inoutstrides1_v, 0x3FF
ld32          $second_ptr_offset_v, $mvertex_base, WKR_SECOND_PTR_OFFSET/4
shl           $second_ptr_offset_v, $second_ptr_offset_v, 20
// inoutstrides3_v = [second-offset][0][out-stride-p4]
or            $inoutstrides3_v, $inoutstrides3_v, $second_ptr_offset_v

ld32          $partition_base_v, $mvertex_base, WKR_PARTITION_BASE/4

convPartialNx1FlattenedStateRetained_f32sisov2amp:
ld32          $partition_table_v, $mvertex_base, WKR_PARTITION_PTR/4
ld32          $partition_struct_v, $partition_table_v, $wkr_id_v

// extract offset and delta: upper 14 bits gives number of partitions
// lower 18 bits give offset to common base for the vector list
shr           $num_partitions_v, $partition_struct_v, DELTAN_OFFSET_BITS
shl           $num_partitions_v, $num_partitions_v, 1
// exit if no work to be done for worker
brz           $num_partitions_v, L_Conv_end
shl           $partition_v, $partition_struct_v, DELTAN_COUNT_BITS
// Offset needs to be converted from elements to bytes.
// Hence shift right is less by ELEM_TO_BYTES_CONV
{
  shr           $partition_v, $partition_v, (DELTAN_COUNT_BITS - ELEM_TO_BYTES_CONV)
  setzi         $a0, ZAACC_BITMASK
}
{
  add           $partition_v, $partition_base_v, $partition_v
  uput          $FP_CLR, $a0
}

// Code fragments to handle number of field samples equal to 1 and 2 are
// handled differently inorder to avoid excess loads which could potentially
// cause memory conflicts given that striding on both inputs and output are
// used. It is possible to use the same code for all cases but this would
// require a lot more stride registers and selective setting of them depending
// on number of field samples.
PartitionLoop:

  // we are addressing partitions from last to first. Move pointer to the previous
  // partition. The other elements in the partition are addressed with an
  // offset of a partition.
  ldz16step     $out_offset_v, $partition_v, $num_partitions_v+=, -PARTITION_SIZE

  // form input address
  ld32          $in_addr_v, $mvertex_base, WKR_INCHAN_PTR/4
  ldz16         $in_offset_v, $partition_v, $num_partitions_v, (PARTITION_SIZE + PARTITION_INOFFSET/PARTITION_ELEM_BYTES)
  // dummy load to do in_addr_v = in_addr_v + in_offset_v*8
  ld64step      $azeros, $mzero,  $in_addr_v+=, $in_offset_v

  // form output address
  ld32          $out_addr_v, $mvertex_base, WKR_OUTCHAN_PTR/4
  ld64step      $azeros, $mzero, $out_addr_v+=, $out_offset_v

  // Form packed address
  tapack        $tripacked_addr, $in_addr_v, $out_addr_v, $out_addr_v

  // Also move partials-out pointer by 1 write because first output
  // will be saved by using tripacked_addr1
  add           $out_addr_v, $out_addr_v, 0x8  // = 1 * 8

  tapack        $tripacked_addr2, $in_addr_v, $out_addr_v, $out_addr_v

  // Move partials-in by 4 reads because worker calls for four time for
  // tripacked_addr1 and then four times for tripacked_addr2
  // subtract 0x8 because we have already incremented out-ptr by 0x8 before
  // packing
  add           $tripacked_addr2_hi, $tripacked_addr2_hi, (0x20 - 0x8)

  lds16         $num_fp_v, $partition_v, $num_partitions_v, (PARTITION_SIZE + PARTITION_NUM_ELEMS/PARTITION_ELEM_BYTES)

  // jump to specialisation for number of field samples equal to 1
  // Note: when num_fpv = 0, 3 partials will be loaded but with an post
  //       increment of 8 bytes
  brneg         $num_fp_v, ConvNumFpEq1

  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace    NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
  {
    // *input-ptr += 0
    // partials-ptr += 1;
    ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P0
  }
  {
    // *input-ptr += 0
    // *partials-ptr += 1
    ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P1
  }
  {
    // *input-ptr += 0
    // partials-ptr += out-stride-p4;
    ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $inoutstrides3_v, 0b0110
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P2
  }
  {
    // *input-ptr += in-row-stride (part 1 of second ptr offset)
    // *partials-ptr += 1
    ld2x64pace      NULL, P_IN, $tripacked_addr2+=,  $inoutstrides2_v, 0b0010
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P3
  }

  {
    // *input-ptr += second-offset (part 2 of second ptr offst)
    // *partials-ptr += 1;
    ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b0011
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P4
  }
  {
    // *input-ptr += 0
    // *partials-ptr += 1
    ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $mzero, 0b0011
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P5
  }
  {
    // *input-ptr += 0
    // *partials-ptr += out-stride-p4
    ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b0110
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P6
  }
  {
    // *input-ptr += in-row-stride
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
    f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P7
  }

  // Start providing inputs ----------------------------------------------------
  {
    // *input-ptr += 0
    // *partials-ptr += 1
    ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
  }
  {
    // *input-ptr += in-stride
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $inoutstrides1_v, 0b0010
    f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P1
  }
  {
    // *input-ptr += 0
    // *partials-ptr += out-stride-p4
    ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $inoutstrides3_v, 0b0110
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
  }
  {
    // *input-ptr += in-row-stride
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
    f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P3
  }
  {
    // *input-ptr += 0
    // *partials-ptr += 1
    ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $mzero, 0b0011
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
  }
  {
    // *input-ptr += in-stride
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr2+=, $inoutstrides1_v, 0b0010
    f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P5
  }
  {
    // *input-ptr += 0
    // *partials-ptr += out-stride-p4
    ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b0110
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
  }
  {
    // *input-ptr += 0 // don't increment here cause will need to read from this address again
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $mzero, 0b0011
    f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P7
  }

  // Start recording output ----------------------------------------------------
  {
    // *input-ptr += in-row-stride
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
  }
  {
    // *input-ptr += 0 // don't increment here cause will need to read from this address again
    // *partials-ptr += 1
    ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $mzero, 0b0011
    f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P1
  }
  {
    // *input-ptr += in-stride
    // *partials-ptr += out-stride-p4
    // *out-ptr += out-stride-step
    ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr1+=, $inoutstrides1_v, 0b110110
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
  }
  {
    // *input-ptr += 0 // don't increment here cause will need to read from this address again
    // *partials-ptr += 1
    // *out-ptr += 1
    ld2xst64pace    X_IN__P_IN, P2_OUT, $tripacked_addr2+=, $mzero, 0b000011
    f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P3
  }

  rpt $num_fp_v, (Loop_end_Amp-Loop_start_Amp)/8-1
Loop_start_Amp:
    {
      // *input-ptr += in-row-stride
      // *partials-ptr += 1
      // *out-ptr += 1
      ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr2+=, $inoutstrides2_v, 0b000010
      f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
    }
    {
      // *input-ptr += 0 // don't increment here cause will need to read from this address again
      // *partials-ptr += 1
      // *out-ptr += 1
      ld2xst64pace    X_IN__P_IN, P2_OUT, $tripacked_addr2+=, $mzero, 0b000011
      f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P5
    }
    {
      // *input-ptr += in-stride
      // *partials-ptr += out-stride-p4
      // *out-ptr += out-stride-p4
      ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr2+=, $inoutstrides1_v, 0b010110
      f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
    }
    {
      // *input-ptr += 0 // don't increment here cause will need to read from this address again
      // *partials-ptr += 1
      // *out-ptr += 1
      ld2xst64pace    X_IN__P_IN, P2_OUT, $tripacked_addr1+=, $mzero, 0b000011
      f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P7
    }
    {
      // *input-ptr += in-row-stride0
      // *partials-ptr += 1
      // *out-ptr += 1
      ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr1+=, $inoutstrides2_v, 0b000010
      f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
    }
    {
      // *input-ptr += 0 // don't increment here cause will need to read from this address again
      // *partials-ptr += 1
      // *out-ptr += out-stride
      ld2xst64pace    X_IN__P_IN, P2_OUT, $tripacked_addr1+=, $inoutstrides2_v, 0b010011
      f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P1
    }
    {
      // *input-ptr += in-stride
      // *partials-ptr += out-stride-p4
      // *out-ptr += out-stride-step
      ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr1+=, $inoutstrides1_v, 0b110110
      f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
    }
    {
      // *input-ptr += 0 // don't increment here cause will need to read from this address again
      // *partials-ptr += 1
      // *out-ptr += 1
      ld2xst64pace    X_IN__P_IN, P2_OUT, $tripacked_addr2+=, $mzero, 0b000011
      f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P3
    }
Loop_end_Amp:
  {
    // *input-ptr += in-row-stride
    // *partials-ptr += 1
    // *out-ptr += 1
    ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr2+=, $inoutstrides2_v, 0b000010
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
  }
  {
    // *input-ptr += 0 // don't increment here cause will need to read from this address again
    // *partials-ptr += 1
    // *out-ptr += 1
    ld2xst64pace    X_IN__P_IN, P2_OUT, $tripacked_addr2+=, $mzero, 0b000011
    f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P5
  }
  {
    // *input-ptr += in-stride
    // *partials-ptr += 0
    // *out-ptr += out-stride-step
    ld2xst64pace    X_IN__P_IN, P_OUT, $tripacked_addr2+=, $inoutstrides1_v, 0b011110
    f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
  }
  {
    // *input-ptr += in-row-stride
    // *out-ptr += 1
    ldst64pace      X_IN, P2_OUT, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
    f32sisov2amp    P2_OUT, X_IN_1, P_IN, TAMP_F32_E4_P7
  }
  // Stop providing partials ---------------------------------------------------
  {
    // *input-ptr += 0
    // *out-ptr += 1
    ldst64pace      NULL, P_OUT, $tripacked_addr1+=, $mzero, 0b0011
    f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P0
  }
  {
    // input-ptr += 0
    // out-ptr += out-stride
    ldst64pace      X_IN, P2_OUT, $tripacked_addr1+=, $inoutstrides2_v, 0b0111
    f32sisov2amp    P2_OUT, X_IN_1, NULL, TAMP_F32_E4_P1
  }
  {
    // out-ptr += out-stride-step
    st64pace        P_OUT, $tripacked_addr1+=, $inoutstrides1_v, 0b11
    f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P2
  }
  {
    // input-ptr += in-row-stride
    // out-ptr += 1
    ldst64pace      X_IN, P2_OUT, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
    f32sisov2amp    P2_OUT, X_IN_1, NULL, TAMP_F32_E4_P3
  }
StoreFinalAmpOutputs2:
  {
    // *out-ptr += 1
    st64pace        P_OUT, $tripacked_addr2+=, $mzero, 0b00
    f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P4
  }
  {
    // *input-ptr += 0
    // *out-ptr += 1
    ldst64pace      X_IN, P2_OUT, $tripacked_addr2+=, $mzero, 0b0011
    f32sisov2amp    P2_OUT, X_IN_1, NULL, TAMP_F32_E4_P5
  }
  {
    // out-ptr += out-stride-p4
    st64pace        P_OUT, $tripacked_addr2+=, $inoutstrides1_v, 0b01
    f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P6
  }
  {
    // *out-ptr += 1
    st64pace        P2_OUT, $tripacked_addr1+=, $mzero, 0b00
    f32sisov2amp    P2_OUT, X_IN_1, NULL, TAMP_F32_E4_P7
  }
  // Stop providing input ------------------------------------------------------
  {
    // *out-ptr += 1
    st64pace        P_OUT, $tripacked_addr1+=, $mzero, 0b00
    f32sisov2amp    P_OUT, NULL1, NULL, TAMP_F32_E4_P0
  }
    {
    // *out-ptr += out-stride
    st64pace        P2_OUT, $tripacked_addr1+=, $inoutstrides2_v, 0b01
    f32sisov2amp    P2_OUT, NULL1, NULL, TAMP_F32_E4_P1
  }
  {
    // *out-ptr += out-stride-step
    st64pace        P_OUT, $tripacked_addr1+=, $inoutstrides1_v, 0b11
    f32sisov2amp    P_OUT, NULL1, NULL, TAMP_F32_E4_P2
  }

StoreFinalAmpOutputs1:
  // The partials for the next iteration may be loaded here but would require
  // the input and output addresses to be computed.
  {
    // *out-ptr += 1
    st64pace        P2_OUT, $tripacked_addr2+=, $mzero, 0b00
    f32sisov2amp    P2_OUT, NULL1, NULL, TAMP_F32_E4_P3
  }
  {
    // *out-ptr += 1
    st64pace        P_OUT, $tripacked_addr2+=, $mzero, 0b00
    f32sisov2amp    P_OUT, NULL1, NULL, TAMP_F32_E4_P4
  }
  {
    // *out-ptr += 1
    st64pace        P2_OUT, $tripacked_addr2+=, $mzero, 0b00
    f32sisov2amp    P2_OUT, NULL1, NULL, TAMP_F32_E4_P5
  }
  {
    // *out-ptr += out-stride-p4
    st64pace        P_OUT, $tripacked_addr2+=, $inoutstrides1_v, 0b01
    f32sisov2amp    P_OUT, NULL1, NULL, TAMP_F32_E4_P6
  }
  {
    // *out-ptr += 1
    st64pace        P2_OUT, $tripacked_addr1+=, $mzero, 0b00
    f32sisov2amp    P2_OUT, NULL1, NULL, TAMP_F32_E4_P7
  }
  // *out-ptr += 1
  st64pace          P_OUT, $tripacked_addr1+=, $mzero, 0b00
  // *out-ptr += 0
  st64pace          P2_OUT, $tripacked_addr1+=, $mzero, 0b11


L_Partition_end:
  brnz              $num_partitions_v, PartitionLoop

L_Conv_end:
exitz         $mzero

// =============================================================================
// This handles the case of number of field positions equal to 1.

ConvNumFpEq1:
add           $num_fp_v, $num_fp_v, 1
brz           $num_fp_v, ConvNumFpEq2
add           $num_fp_v, $num_fp_v, 1
brneg         $num_fp_v, L_Partition_end

// *input-ptr += 0
// *partials-ptr += 1
ld2x64pace    NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
{
  // *input-ptr += 0
  // partials-ptr += 1;
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P0
}
{
  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P1
}
{
  // *input-ptr += 0
  // partials-ptr += 0;
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $inoutstrides3_v, 0b1010
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P2
}
{
  // *input-ptr += in-row-stride (part 1 of second ptr offset)
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=,  $inoutstrides2_v, 0b0010
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P3
}
{
  // *input-ptr += second-offset  (part 2 of second ptr offset)
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P4
}
{
  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P5
}
{
  // *input-ptr += 0
  // *partials-ptr += 0
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $mzero, 0b1111
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P6
}
{
  // *input-ptr += in-row-stride
  // *partials-ptr += 0
  ld2x64pace      X_IN, NULL, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P7
}
// ---- Stop providing partials -------------------------------

f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P0
{
  // *input-ptr += in-stride
  // *partials-ptr += (1)
  ld2x64pace      X_IN, NULL, $tripacked_addr1+=, $inoutstrides1_v, 0b0010
  f32sisov2amp    NULL2, X_IN_1, NULL, TAMP_F32_E4_P1
}
f32sisov2amp      P_OUT, X_IN_0, NULL, TAMP_F32_E4_P2
{
  // *input-ptr += in-row-stride
  // *partials-ptr += 0
  ld2x64pace      X_IN, NULL, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
  f32sisov2amp    NULL2, X_IN_1, NULL, TAMP_F32_E4_P3
}
f32sisov2amp      P_OUT, X_IN_0, NULL, TAMP_F32_E4_P4
{
  // *input-ptr += 0
  // *partials-ptr += 0
  ld2x64pace      X_IN, NULL, $tripacked_addr2+=, $mzero, 0b1111
  f32sisov2amp    NULL2, X_IN_1, NULL, TAMP_F32_E4_P5
}
f32sisov2amp      P_OUT, X_IN_0, NULL, TAMP_F32_E4_P6
f32sisov2amp      NULL2, X_IN_1, NULL, TAMP_F32_E4_P7

// Start recording output ----------------------------------------------------
f32sisov2amp      P_OUT, NULL1, NULL, TAMP_F32_E4_P0
f32sisov2amp      P2_OUT, NULL1, NULL, TAMP_F32_E4_P1
{
  // *out-ptr += out-stride-step
  st64pace        P_OUT, $tripacked_addr1+=, $inoutstrides1_v, 0b11
  f32sisov2amp    P_OUT, NULL1, NULL, TAMP_F32_E4_P2
}
// Jump to common part to store final samples
bri           StoreFinalAmpOutputs1

// =============================================================================
// This handles the case of number of field positions equal to 2.

// clear output stride to allow post increment by 0 in the dummy loads of
// partials
ConvNumFpEq2:

// *input-ptr += 0
// *partials-ptr += 1
ld2x64pace    NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
{
  // *input-ptr += 0
  // partials-ptr += 1;
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P0
}
{
  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P1
}
{
  // *input-ptr += 0
  // partials-ptr += out-stride-p4;
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $inoutstrides3_v, 0b0110
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P2
}
{
  // *input-ptr += in-row-stride (part 1 of second pointer offset)
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=,  $inoutstrides2_v, 0b0010
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P3
}
{
  // *input-ptr += second-ptr-off (part 2 of second pointer offset)
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P4
}
{
  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P5
}
{
  // *input-ptr += 0
  // *partials-ptr += out-stride-p4
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b0110
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P6
}
{
  // *input-ptr += in-row-stride
  // *partials-ptr += 1
  ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
  f32sisov2amp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P7
}
{
  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
}
{
  // *input-ptr += in-stride
  // *partials-ptr += 1
  ld2x64pace      X_IN, P_IN, $tripacked_addr1+=, $inoutstrides1_v, 0b0010
  f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P1
}
{
  // *input-ptr += 0
  // *partials-ptr += 0
  ld2x64pace      NULL, P_IN, $tripacked_addr1+=, $inoutstrides3_v, 0b1010
  f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
}
{
  // *input-ptr += in-row-stride
  // *partials-ptr += 1
  ld2x64pace      X_IN, P_IN, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
  f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P3
}
{
  // *input-ptr += 0
  // *partials-ptr += 1
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $mzero, 0b0011
  f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
}
{
  // *input-ptr += in-stride
  // *partials-ptr += 1
  ld2x64pace      X_IN, P_IN, $tripacked_addr2+=, $inoutstrides1_v, 0b0010
  f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P5
}
{
  // *input-ptr += 0
  // *partials-ptr += 0
  ld2x64pace      NULL, P_IN, $tripacked_addr2+=, $inoutstrides3_v, 0b1010
  f32sisov2amp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
}
{
  // *input-ptr += in-row-stride
  // *partials-ptr += 0
  ld2x64pace      X_IN, NULL, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
  f32sisov2amp    NULL2, X_IN_1, P_IN, TAMP_F32_E4_P7
}
// ---- Stop providing partials -------------------------------

f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P0
{
  // *input-ptr += in-stride
  // *partials-ptr += (1)
  ld2x64pace      X_IN, NULL, $tripacked_addr1+=, $inoutstrides1_v, 0b0010
  f32sisov2amp    P2_OUT, X_IN_1, NULL, TAMP_F32_E4_P1
}
{
  // *out-ptr += out-stride-step
  st64pace        P_OUT, $tripacked_addr1+=, $inoutstrides1_v, 0b11
  f32sisov2amp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P2
}
{
  // *input-ptr += in-row-stride
  // *out-ptr += 1
  ldst64pace      X_IN, P2_OUT, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
  f32sisov2amp    P2_OUT, X_IN_1, NULL, TAMP_F32_E4_P3
}
bri           StoreFinalAmpOutputs2

.size convPartialNx1Flattened_f32sisov2amp, . - convPartialNx1Flattened_f32sisov2amp


// =============================================================================
// Instantiate codelets

CONV_Nx1_SUPERVISOR float float false 16 2 false f32sisov2amp
CONV_Nx1_SUPERVISOR float float true  16 2 false f32sisov2amp
CONV_Nx1_SUPERVISOR float float false 16 2 true f32sisov2amp
CONV_Nx1_SUPERVISOR float float true  16 2 true f32sisov2amp

// =============================================================================
#endif
// =============================================================================
