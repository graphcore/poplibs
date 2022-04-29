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
//             30 cycles
//      Number of field elems = 2
//             34 cycles
//      Number of field elems >= 3
//             35 + (num_field_elems - 3) * 4

.macro CONV_Nx1_WORKER COMMAND

.section ".text.convPartialNx1Flattened_\COMMAND\()", "ax"
.type convPartialNx1Flattened_\COMMAND\(), @function
.align 8
nop // rpt alignment
.worker
convPartialNx1FlattenedAligned_\COMMAND\():
convPartialNx1Flattened_\COMMAND\():
// worker register mapping
#define wkr_id_v                       m0   // retained
#define partition_table_v              m1
#define partition_struct_v             m1
#define partition_v                    m1
#define num_partitions_v               m2
#define num_fp_v                       m3
#define second_ptr_offset_v            m3
#define inoutstrides3_tmp_v            m3
#define inoutstrides1_v                m4   // retained
#define inoutstrides2_v                m5   // retained
#define tripacked_addr2                m6:7
#define tripacked_addr2_hi             m7
#define in_addr_v                      m6
#define out_addr_v                     m7
#define tripacked_addr1                m8:9
#define in_offset_v                    m8
#define out_offset_v                   m9
#define inoutstrides3_v                m10 // retained
#define partition_base_v               m11 // retained

get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK

// inoutstrides1_v = [out-stride-back][in-stride][out-stride-p2]
ld32          $inoutstrides1_v, $mvertex_base, WKR_INOUTSTRIDES1/4
// inoutstrides2_v = [0][in-row-stride][out-stride]
ld32          $inoutstrides2_v, $mvertex_base, WKR_INOUTSTRIDES2/4

// Special stride for initial preloads and wind downs. Form 3 strides in 
// a register: [second-offset | 0x0 | inoutstrides3_v]
and           $inoutstrides3_v, $inoutstrides1_v, 0x3FF

ld32          $second_ptr_offset_v, $mvertex_base, WKR_SECOND_PTR_OFFSET/4
shl           $second_ptr_offset_v, $second_ptr_offset_v, 20
or            $inoutstrides3_v, $inoutstrides3_v, $second_ptr_offset_v

ld32          $partition_base_v, $mvertex_base, WKR_PARTITION_BASE/4

convPartialNx1FlattenedStateRetained_\COMMAND\():
ld32          $partition_table_v, $mvertex_base, WKR_PARTITION_PTR/4
ld32          $partition_struct_v, $partition_table_v, $wkr_id_v

// extract offset and delta: upper X bits gives number of partitions
// lower Y bits give offset to common base for the vector list
// For DeltaN: X = 18, Y = 14
// For DeltaNElements: X and Y are based on memory alignment used
shr           $num_partitions_v, $partition_struct_v, DELTAN_OFFSET_BITS
shl           $num_partitions_v, $num_partitions_v, 1
// exit if no work to be done for worker
brz           $num_partitions_v, L_Conv_end_\COMMAND\()
shl           $partition_v, $partition_struct_v, DELTAN_COUNT_BITS
// Offset needs to be converted from elements to bytes.
// Hence shift rigth is less by ELEM_TO_BYTES_CONV
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
PartitionLoop_\COMMAND\():
  // we are addressing partitions from last to first. Move pointer to the previous
  // partition. The other elements in the partition are addressed with an
  // offset of a partition.
  ldz16step     $out_offset_v, $partition_v, $num_partitions_v+=, -PARTITION_SIZE

  // form input address
  ld32          $in_addr_v, $mvertex_base, WKR_INCHAN_PTR/4
  ldz16         $in_offset_v, $partition_v, $num_partitions_v, (PARTITION_SIZE + PARTITION_INOFFSET/PARTITION_ELEM_BYTES)
  // dummy load to do in_addr_v = in_addr_v + in_offset_v*8
  ld64step      $azeros, $mzero, $in_addr_v+=, $in_offset_v

  // form output address
  ld32          $out_addr_v, $mvertex_base, WKR_OUTCHAN_PTR/4
  // dummy load to do out_addr_v = out_addr_v + out_offset_v*8
  ld64step      $azeros, $mzero, $out_addr_v+=, $out_offset_v

  // Form packed address
  tapack        $tripacked_addr1, $in_addr_v, $out_addr_v, $out_addr_v

  tapack        $tripacked_addr2, $in_addr_v, $out_addr_v, $out_addr_v
  // Move partials-in by 2 reads because worker calls for twice for
  // tripacked_addr1 and then twice for tripacked_addr2
  add           $tripacked_addr2_hi, $tripacked_addr2_hi, 0x10

  lds16         $num_fp_v, $partition_v,  $num_partitions_v, (PARTITION_SIZE + PARTITION_NUM_ELEMS/PARTITION_ELEM_BYTES)

  // jump to specialisation for number of field samples equal to 1
  // Note: when num_fpv = 0, 3 partials will be loaded but with an post
  //       increment of 8 bytes
  brneg         $num_fp_v, ConvNumFpEq1_\COMMAND\()

  // *input-ptr += 0; *partials-ptr += 1
  ld2x64pace    $azeros, $a4:5, $tripacked_addr1+=, $mzero, 0b0011
  {
    // Need to move output-ptr-1 by 1 atom to keep store addresses in order
    // input-ptr += 0;
    // partials-ptr += out-stride-p2;
    // out-ptr += 1;
    ld2xst64pace  $a0:3, $a4:5, $tripacked_addr1+=, $inoutstrides3_v, 0b000110
    \COMMAND      $a6:7, $azeros, $a4:5, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride (first part of second ptr offset) 
    // *partials-ptr += 1
    ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
    \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P1
  }
  {
    // *input-ptr += second-offset (second part of second ptr offset) 
    // *partials-ptr += out-stride-p2;
    ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides3_v, 0b0111
    \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
    \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-stride; partials-ptr += out-stride-p2
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
    \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
    \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }
  {
    // *input-ptr += in-stride; *partials_ptr += out-stride-p2
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides1_v, 0b0110
    \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
    \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-stride; *partials-ptr += out-stride-p2
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
    \COMMAND      $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride; *partials += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
    \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }

  {
    rpt $num_fp_v, (Loop_end_Amp_\COMMAND\()-Loop_start_Amp_\COMMAND\())/8-1
    fnop
  }
Loop_start_Amp_\COMMAND\():
    {
      // *input-ptr += in-stride;
      // *partials-ptr += out-stride-p2;
      // *out-ptr += out-stride-step;
      ld2xst64pace  $a0:3, $a4:5, $tripacked_addr2+=, $inoutstrides1_v, 0b110110
      \COMMAND      $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P2
    }
    {
      // *input-ptr += in-row-stride;
      // *partials-ptr += 1;
      // *out-ptr += 1;
      ld2xst64pace  $a0:3, $a6:7, $tripacked_addr1+=, $inoutstrides2_v, 0b000010
      \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
    }
    {
      // *input-ptr += in-stride;
      // *partials-ptr += out-stride-p2;
      // *out-ptr += out-stride-p2;
      ld2xst64pace  $a0:3, $a4:5, $tripacked_addr1+=, $inoutstrides1_v, 0b010110
      \COMMAND      $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
    }
    {
      // *input-ptr += in-row-stride;
      // *partials-ptr += 1;
      // *out-ptr += out-stride;
      ld2xst64pace  $a0:3, $a6:7, $tripacked_addr2+=, $inoutstrides2_v, 0b010010
      \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
    }
Loop_end_Amp_\COMMAND\():
  {
    // *input-ptr += in-stride;
    // *partials-ptr += out-stride-p2;
    // *out-ptr += out-stride-step;
    ld2xst64pace  $a0:3, $a4:5, $tripacked_addr2+=, $inoutstrides1_v, 0b110110
    \COMMAND      $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // input-ptr += in-row-stride; out-ptr += 1
    ldst64pace    $a0:1, $a6:7, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
    \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // input-ptr += in-stride; out-ptr += out-stride-p2
    ldst64pace    $a0:1, $a4:5, $tripacked_addr1+=, $inoutstrides1_v, 0b0110
    \COMMAND      $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride;
    // *out-ptr += out-stride;
    ldst64pace    $a0:1, $a6:7, $tripacked_addr2+=, $inoutstrides2_v, 0b0110
    \COMMAND      $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
  }
StoreFinalAmpOutputs2_\COMMAND\():
  {
    // *input-ptr += (1);   // Last read so it's safe to post increment input pointer
    // *out-ptr += out-stride-step;
    ldst64pace    $a0:1, $a4:5, $tripacked_addr2+=, $inoutstrides1_v, 0b1100
    \COMMAND      $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P2
  }
  {
    // *out-ptr1 += 1
    st64pace  $a6:7, $tripacked_addr1+=, $mzero, 0b00
    \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
  }
  {
    // *out-ptr1 += out-stride-p2
    st64pace  $a4:5, $tripacked_addr1+=, $inoutstrides1_v, 0b01
    \COMMAND  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P0
  }
  {
    // *out-ptr += out-stride * 2;
    st64pace  $a6:7, $tripacked_addr2+=, $inoutstrides2_v, 0b01
    \COMMAND  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P1
  }
StoreFinalAmpOutputs1_\COMMAND\():
  // The partials for the next iteration may be loaded here but would require
  // the input and output addresses to be computed.
  {
    // *out-ptr += out-stride-step;
    st64pace      $a4:5, $tripacked_addr2+=, $inoutstrides1_v, 0b11
    \COMMAND  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P2
  }
  {
    // *out-ptr1 += 1
    st64pace      $a6:7, $tripacked_addr1+=, $mzero, 0b00
    \COMMAND  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P3
  }
  // *out-ptr1 += 0
  st64pace      $a4:5, $tripacked_addr1+=, $mzero, 0b11
  // *out-ptr2 += 0
  st64pace      $a6:7, $tripacked_addr2+=, $mzero, 0b11

L_Partition_end_\COMMAND\():
  brnz          $num_partitions_v, PartitionLoop_\COMMAND\()

L_Conv_end_\COMMAND\():
exitz         $mzero

// =============================================================================
// This handles the case of number of field positions equal to 1.

ConvNumFpEq1_\COMMAND\():
add           $num_fp_v, $num_fp_v, 1
brz           $num_fp_v, ConvNumFpEq2_\COMMAND\()
add           $num_fp_v, $num_fp_v, 1
brneg         $num_fp_v, L_Partition_end_\COMMAND\()

// *input-ptr += 0; *partials-ptr += 1
ld2x64pace    $azeros, $a4:5, $tripacked_addr1+=, $mzero, 0b0011
{
  // Need to move output-ptr-1 by 1 atom to keep store addresses in order
  // input-ptr += 0;
  // partials-ptr += 0;
  // out-ptr += 1;
  ld2xst64pace  $a0:3, $a4:5, $tripacked_addr1+=, $inoutstrides3_v, 0b001010
  \COMMAND      $a6:7, $azeros, $a4:5, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride (first part of second ptr offset)
  // *partials-ptr += 1
  ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
  \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += second offset (second part of second ptr offset) 
  // *partials-ptr += 0
  ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides3_v, 0b1011
  \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  // *input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides2_v, 0b1110
  \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // *input-ptr += in-stride; *partials-ptr += (1)
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b0010
  \COMMAND      $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides2_v, 0b1110
  \COMMAND      $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += in-stride; *partials_ptr += (1)
  ld2x64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides1_v, 0b0110
  \COMMAND      $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P2
}
\COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
\COMMAND  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P0
{
  // Jump to common part to store final samples
  bri           StoreFinalAmpOutputs1_\COMMAND\()
  \COMMAND      $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P1
}

// =============================================================================
// This handles the case of number of field positions equal to 2.

// clear output stride to allow post increment by 0 in the dummy loads of
// partials
ConvNumFpEq2_\COMMAND\():
// *input-ptr += 0; *partials-ptr += 1
ld2x64pace    $azeros, $a4:5, $tripacked_addr1+=, $mzero, 0b0011
{
  // Need to move output-ptr-1 by 1 atom to keep store addresses in order
  // input-ptr += 0;
  // partials-ptr += out-stride-p2;
  // out-ptr += 1;
  ld2xst64pace  $a0:3, $a4:5, $tripacked_addr1+=, $inoutstrides3_v, 0b000110
  \COMMAND      $a6:7, $azeros, $a4:5, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride (first part of second ptr offset)
  // *partials-ptr += 1
  ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
  \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += second-offset (second part of second ptr offset) 
  // *partials-ptr += out-stride-p2
  ld2x64pace    $azeros, $a2:3, $tripacked_addr2+=, $inoutstrides3_v, 0b0111
  \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  // *input-ptr += in-row-stride; *partials-ptr += 1
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides2_v, 0b0010
  \COMMAND      $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
shl $inoutstrides3_tmp_v, $inoutstrides1_v, 10
{
  // *input-ptr += in-stride; partials-ptr += 0
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr1+=, $inoutstrides3_tmp_v, 0b0111
  \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride; *partials-ptr += 1
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
  \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += in-stride; *partials_ptr += 0
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr2+=, $inoutstrides3_tmp_v, 0b0111
  \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P2
}
#define inoutstrides2_eq2_v   num_fp_v
andc    $inoutstrides2_eq2_v, $inoutstrides2_v, 0x3ff
{
  // *input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides2_eq2_v, 0b0110
  \COMMAND      $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
}
{
  // input-ptr += in-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr1+=, $inoutstrides1_v, 0b0010
  \COMMAND      $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // input-ptr += in-row-stride; *partials-ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr2+=, $inoutstrides2_v, 0b0010
  \COMMAND      $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
bri           StoreFinalAmpOutputs2_\COMMAND\()

.size convPartialNx1Flattened_\COMMAND\(), . - convPartialNx1Flattened_\COMMAND\()
.endm


// =============================================================================
// Instantiate codelets

#if (__IPU_ARCH_VERSION__ == 2) || (__IPU_ARCH_VERSION__ == 21)
// Worker
CONV_Nx1_WORKER f16v4hihov4amp

// Supervisor
CONV_Nx1_SUPERVISOR half half false 16 4 false f16v4hihov4amp
CONV_Nx1_SUPERVISOR half half true  16 4 false f16v4hihov4amp
CONV_Nx1_SUPERVISOR half half false 16 4 true f16v4hihov4amp
CONV_Nx1_SUPERVISOR half half true  16 4 true f16v4hihov4amp
#endif


// Worker
CONV_Nx1_WORKER f16v4sisoamp

// Supervisors
CONV_Nx1_SUPERVISOR half float false 8 4 false f16v4sisoamp
CONV_Nx1_SUPERVISOR half float true  8 4 false f16v4sisoamp
CONV_Nx1_SUPERVISOR half float false 8 4 true f16v4sisoamp
CONV_Nx1_SUPERVISOR half float true  8 4 true f16v4sisoamp


// =============================================================================
#endif
// =============================================================================
