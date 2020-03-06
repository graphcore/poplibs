// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Computes an nx1 convolution using AMP. A contiguous field is partitioned
// between workers for each position of the kernel element.
//
// Requires a total stack size of 80 bytes in the supervisor
//
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)


// =============================================================================

// Types of load variants
#define LD128                           128  // Use ld128putcs
#define LD64                            64   // Use ld64putcs

// =============================================================================

#define CODELET_NAME_128 __runCodelet_poplin__ConvPartialnx1___half_float_true_true
#define CODELET_NAME_64 __runCodelet_poplin__ConvPartialnx1___half_float_true_false

// =============================================================================

// Vertex input, output and weight data atom size constants

// The number of AMP outputs with input and weight data == half
#define AMP_OUTPUTS_HALF                8
#define LOG2_AMP_OUTPUTS_HALF           3
// log2(AMP_OUTPUTS_HALF)
#define AMP_INPUTS_HALF                 16
// sizeof(input) in bytes
#define SIZEOF_IN_ATOM                  2
// sizeof(output) in bytes
#define SIZEOF_OUT_ATOM                 4
#define LOG2_SIZEOF_OUT_ATOM            2
// sizeof(weights) in bytes
#define SIZEOF_WEIGHTS_ATOM             2

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
#define SUP_NUM_INCHAN                  24   // short
#define SUP_NUM_KERNEL_Y_M1             26   // short (Kx-1)
#define SUP_NUM_KERNEL_X_M1             28   // short (Ky-1)
#define SUP_INSTRIDE                    30   // short
#define SUP_OUTSTRIDE                   32   // short
#define SUP_NUM_CONVGROUPS_M1           34   // short
#define SUP_NUM_FILTER_HEIGHT_M1        36   // short
#define SUP_IN_ROW_STRIDE               38   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      40   // short
#define SUP_NUM_INCHANS_PER_GROUP       42   // short

#else
#define SUP_INCHAN_VECTORS              0    // word
#define SUP_WEIGHTS_VECTORS             4    // word
#define SUP_OUTCHAN_VECTORS             8    // word
#define SUP_ZERO_INFO                   12   // word
#define SUP_PARTITION_TABLES            16   // VectorList::DeltaNElements
#define SUP_NUM_OUTCHAN_M1              24   // short
#define SUP_NUM_INCHAN                  26   // short
#define SUP_NUM_KERNEL_Y_M1             28   // short (Kx-1)
#define SUP_NUM_KERNEL_X_M1             30   // short (Ky-1)
#define SUP_INSTRIDE                    32   // short
#define SUP_OUTSTRIDE                   34   // short
#define SUP_NUM_CONVGROUPS_M1           36   // short
#define SUP_NUM_FILTER_HEIGHT_M1        38   // short
#define SUP_IN_ROW_STRIDE               40   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      42   // short
#define SUP_NUM_INCHANS_PER_GROUP       44   // short
#endif

// Worklist partition fields: must match the order of entries in worklists
#define PARTITION_OUTOFFSET             0    // half
#define PARTITION_NUM_ELEMS             2    // half
#define PARTITION_INOFFSET              4    // half

// Zero partition fields: must match the order of entries in zero worklist
#define ZERO_PARTITION_OUTOFFSET        0
#define ZERO_PARTITION_NUM_ELEMS        2

// Zero partition constants
#define ZERO_PARTITION_ATOM_SIZE        2
#define ZERO_PARTITION_NUM_ENTRIES      2

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
#define WKR_INOUTSTRIDES1               0      // word
#define WKR_INOUTSTRIDES2               4      // word
#define WKR_INCHAN_PTR                  8      // word
#define WKR_OUTCHAN_PTR                 12     // word
#define WKR_PARTITION_PTR               16     // word
#define WKR_ZERO_INFO                   20     // word
#define WKR_OUTCHANS_PER_GROUP_X4       24     // word
#define WKR_PARTITION_BASE              28     // word
#define WKR_INCHANS_PER_GROUP_X2        32     // word
#define WKR_STATE_SIZE                  (WKR_INCHANS_PER_GROUP_X2 + 4) // bytes

// =============================================================================

// Zero output/partials
// Uses same vertex state as the convolution
//
// Performance: 14 + num_samples / 2

.section ".text.convNx1ZeroOutField", "ax"
.global convNx1ZeroOutField
.type convNx1ZeroOutField, @function
.align 8

#define wkr_id_zv                       m0
#define outchan_ptr_zv                  m1
#define zero_info_zv                    m2
#define rem_zv                          m3
#define final_write_zv                  m4
#define div_by_12_zv                    m5
#define add_one_zv                      m6
convNx1ZeroOutField:
{
get           $wkr_id_zv, $WSR
setzi         $a0, ZAACC_BITMASK
}
{
and           $wkr_id_zv, $wkr_id_zv, CSR_W_WSR__CTXTID_M1__MASK
uput          $FP_CLR, $a0
}
ld32          $outchan_ptr_zv, $mvertex_base, WKR_OUTCHAN_PTR/4
ld32          $zero_info_zv, $mvertex_base, WKR_ZERO_INFO/4

// One worker has to do the remainder
and           $final_write_zv, $zero_info_zv, 0x1
shr           $zero_info_zv, $zero_info_zv, (3 - LOG2_SIZEOF_OUT_ATOM)

// floor(x/6) = floor(21846 * x / 2^17) for values of x in the range
// [0, 32768)
mul           $div_by_12_zv, $zero_info_zv, 21846
shr           $div_by_12_zv, $div_by_12_zv, 17
mul           $rem_zv, $div_by_12_zv, 6
sub           $rem_zv, $zero_info_zv, $rem_zv
// each remaining double word is assigned to a worker
cmpult        $add_one_zv, $wkr_id_zv, $rem_zv
add           $div_by_12_zv, $div_by_12_zv, $add_one_zv

ld64step      $azeros, $mzero, $outchan_ptr_zv+=, $wkr_id_zv
rpt           $div_by_12_zv, (Loop_end_zero_64 - Loop_start_zero_64)/8 - 1
Loop_start_zero_64:
  {
    st64step      $azeros, $mzero, $outchan_ptr_zv+=, 6
    fnop
  }
Loop_end_zero_64:
brz           $final_write_zv, ExitZeroFn
cmpeq         $rem_zv, $rem_zv, $wkr_id_zv
brz           $rem_zv, ExitZeroFn
st32step      $azero, $mzero, $outchan_ptr_zv+=, 1

ExitZeroFn:
exitz         $mzero

.size convNx1ZeroOutField, . - convNx1ZeroOutField

// =============================================================================


// Non loop overhead:
//      zero Partitions:         7
//      non-zero partitions:     19
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

.section ".text.convPartialNx1Flattened", "ax"
.type convPartialNx1Flattened, @function
.align 8
.worker
convPartialNx1FlattenedAligned:
convPartialNx1Flattened:
// worker register mapping
#define wkr_id_v                       m0
#define inoutstrides2_v                m0
#define partition_table_v              m1
#define partition_struct_v             m1
#define partition_v                    m1
#define num_partitions_v               m2
#define num_fp_v                       m3
#define inoutstrides1_v                m4
#define in_offset_v                    m5
#define out_offset_v                   m5
#define out_chans_per_group_x4         m6
#define partition_base_v               m7
#define in_chans_per_group_x2          m7
#define tripacked_addr                 m8:9
#define in_addr_v                      m8
#define out_addr_v                     m9
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11

{
get           $wkr_id_v, $WSR
setzi         $a0, ZAACC_BITMASK
}
{
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK
uput          $FP_CLR, $a0
}

ld32          $partition_table_v, $mvertex_base, WKR_PARTITION_PTR/4
ld32          $partition_struct_v, $partition_table_v, $wkr_id_v

// extract offset and delta: upper X bits gives number of partitions
// lower Y bits give offset to common base for the vector list
// For DeltaN: X = 18, Y = 14
// For DeltaNElements: X and Y are based on memory alignment used
shr           $num_partitions_v, $partition_struct_v, DELTAN_OFFSET_BITS
// exit if no work to be done for worker
brz           $num_partitions_v, L_Conv_end
shl           $partition_v, $partition_struct_v, DELTAN_COUNT_BITS
// Offset needs to be converted from elements to bytes.
// Hence shift rigth is less by ELEM_TO_BYTES_CONV
shr           $partition_v, $partition_v, (DELTAN_COUNT_BITS - ELEM_TO_BYTES_CONV)
ld32          $partition_base_v, $mvertex_base, WKR_PARTITION_BASE/4
add           $partition_v, $partition_v, $partition_base_v

ld32          $inchan_ptr_v, $mvertex_base, WKR_INCHAN_PTR/4
ld32          $outchan_ptr_v, $mvertex_base, WKR_OUTCHAN_PTR/4

// inoutstrides1_v = [0][in-stride][out-stride]
ld32          $inoutstrides1_v, $mvertex_base, WKR_INOUTSTRIDES1/4
// inoutstrides2_v = [in-row-stride1][in-row-stride0][out-stride]
ld32          $inoutstrides2_v, $mvertex_base, WKR_INOUTSTRIDES2/4
ld32          $in_chans_per_group_x2, $mvertex_base, WKR_INCHANS_PER_GROUP_X2/4
ld32          $out_chans_per_group_x4, $mvertex_base, WKR_OUTCHANS_PER_GROUP_X4/4

// This following approximates num_partitions/3 - 1 for values
// [3:3:2^14-1]. The case of zero is handled above
mul           $num_partitions_v, $num_partitions_v, 21845
shr           $num_partitions_v, $num_partitions_v, 16


// Code fragments to handle number of field samples equal to 1 and 2 are
// handled differently inorder to avoid excess loads which could potentially
// cause memory conflicts given that striding on both inputs and output are
// used. It is possible to use the same code for all cases but this would
// require a lot more stride registers and selective setting of them depending
// on number of field samples.
PartitionLoop:
  ldz16         $num_fp_v, $partition_v, PARTITION_NUM_ELEMS/2

  // form input address
  ldz16         $in_offset_v, $partition_v, PARTITION_INOFFSET/2
  mul           $in_offset_v, $in_offset_v, $in_chans_per_group_x2
  add           $in_addr_v, $inchan_ptr_v, $in_offset_v

  // form output address
  // Note this relies on the fact that out offset is the first entry in the
  // partition. This allows us to wind the pointer to the next partition
  ldz16step     $out_offset_v, $mzero, $partition_v+=, 3

  mul           $out_offset_v, $out_offset_v, $out_chans_per_group_x4
  add           $out_addr_v, $outchan_ptr_v, $out_offset_v

  // Form packed address
  tapack        $tripacked_addr, $in_addr_v, $out_addr_v, $out_addr_v
  // *input-ptr += 0; *partials-ptr += 1
  ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0011
  {
    // input-ptr += 0; partials-ptr += 1
    ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0011
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += 0; *partials-ptr += 1
    ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0011
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P1
  }
  // jump to specialisation for number of field samples equal to 1
  // Note: when num_fpv = 0, 3 partials will be loaded but with an post
  //       increment of 8 bytes
  add           $num_fp_v, $num_fp_v, -3
  brneg         $num_fp_v, ConvNumFpEq1
  {
    // *input-ptr += 0; *partials-ptr += out-stride
    ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0111
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-row-stride1; partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0011
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }

  {
    // *input-ptr += in-stride; *partials_ptr += out-stride
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0110
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-row-stride1; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0011
    f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride0; *partials += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }
  rpt $num_fp_v, (Loop_end_Amp-Loop_start_Amp)/8-1
Loop_start_Amp:
    {
      // *input-ptr += in-stride; *partials-ptr += out-stride; *out-ptr += 1
      ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $inoutstrides1_v, 0b000110
      f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P2
    }
    {
      // *input-ptr += in-row-stride0; *partials-ptr += 1; *out-ptr += 1
      ld2xst64pace  $a0:3, $a6:7, $tripacked_addr+=, $inoutstrides2_v, 0b000010
      f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
    }
    {
      // *input-ptr += in-row-stride1; *partials-ptr += 1; *out-ptr += 1
      ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $inoutstrides2_v, 0b000011
      f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
    }
    {
      // *input-ptr += in-row-stride0; *partials-ptr += 1; *out-ptr += out-stride
      ld2xst64pace  $a0:3, $a6:7, $tripacked_addr+=, $inoutstrides2_v, 0b010010
      f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
    }
Loop_end_Amp:
  {
    // *input-ptr += in-stride; *partials-ptr += out-stride; out-ptr += 1
    ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $inoutstrides1_v, 0b000110
    f16v4sisoamp  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // input-ptr += in-row-stride0; partials-ptr += 1; out-ptr += 1
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // input-ptr += in-row-stride1; partials-ptr += 1; out-ptr += 1
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $inoutstrides2_v, 0b0011
    f16v4sisoamp  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
  }
  {
    // input-ptr += in-row-stride0; partials-ptr += 1; out-ptr += out-stride
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $inoutstrides2_v, 0b0110
    f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
  }
StoreFinalAmpOutputs2:
  {
    // *input-ptr += in-stride; *out-ptr += 1
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $mzero, 0b0000
    f16v4sisoamp  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P2
  }
  {
    // *out-ptr += 1
    st64pace  $a6:7, $tripacked_addr+=, $mzero, 0b00
    f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
  }
  {
    // *out-ptr += 1
    st64pace  $a4:5, $tripacked_addr+=, $mzero, 0b00
    f16v4sisoamp  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P0
  }
  {
    // *out-ptr += out-stride
    st64pace  $a6:7, $tripacked_addr+=, $inoutstrides1_v, 0b01
    f16v4sisoamp  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P1
  }
StoreFinalAmpOutputs1:
  // The partials for the next iteration may be loaded here but would require
  // the input and output addresses to be computed.
  {
    // *out-ptr += 1
    st64pace      $a4:5, $tripacked_addr+=, $mzero, 0b00
    f16v4sisoamp  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P2
  }
  {
    // *out-ptr += 1
    st64pace      $a6:7, $tripacked_addr+=, $mzero, 0b00
    f16v4sisoamp  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P3
  }
  // *out-ptr += 1
  st64pace      $a4:5, $tripacked_addr+=, $mzero, 0b00
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
ConvNumFpEq1:
add           $num_fp_v, $num_fp_v, 1
brz           $num_fp_v, ConvNumFpEq2
add           $num_fp_v, $num_fp_v, 1
brneg         $num_fp_v, L_Partition_end
{
  // *input-ptr += 0; *partials-ptr += 0
  ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b1111
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
}
{
  // *input-ptr += in-row-stride0; *partials-ptr += 1
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_v, 0b0010
  f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // *input-ptr += in-row-stride1; *partials-ptr += 1
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_v, 0b0011
  f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // *input-ptr += in-row-stride0; *partials-ptr += 1
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_v, 0b0010
  f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
{
  // *input-ptr += in-stride; *partials_ptr += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides1_v, 0b1110
  f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P2
}
f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
f16v4sisoamp  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P0
{
  // Jump to common part to store final samples
  bri           StoreFinalAmpOutputs1
  f16v4sisoamp  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P1
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
    // *input-ptr += 0; *partials-ptr += out-stride
    ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b0111
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // *input-ptr += in-row-stride1; partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0011
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // *input-ptr += in-row-stride0; *partials-ptr += 1
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides2_v, 0b0010
    f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }

#define inoutstrides2_eq2_v   num_fp_v
andc          $inoutstrides2_eq2_v, $inoutstrides2_v, 0x3ff

{
  // *input-ptr += in-stride; *partials_ptr += 0
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $inoutstrides1_v, 0b1110
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P2
}
{
  // *input-ptr += in-row-stride0; *partials-ptr += 1
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_eq2_v, 0b0110
  f16v4sisoamp  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
}
{
  // input-ptr += in-row-stride1; partials-ptr += 1
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_eq2_v, 0b0111
  f16v4sisoamp  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // input-ptr += in-row-stride0; partials += 1
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $inoutstrides2_eq2_v, 0b0110
  f16v4sisoamp  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
bri           StoreFinalAmpOutputs2

.size convPartialNx1Flattened, . - convPartialNx1Flattened


// =============================================================================


// Performance:
// Kernel height dependent cycles in the innermost loop (i.e. AmpOutGroupLoop)
//   Kernel Height = 1: 15 + LOADCYLES + workerCycles
//   Kernel Height = 2: 46 + LOADCYLES + workerCycles
//   Kernel Height = 4: 41 + 19 + 32 + workerCycles
// 101 + numConvGroups * (16 + numInChanGroups * (14 + numOutChanGroups * (14 + innerLoopCycles)))
//
// Where innerLoopCycles are:
// Ky * (14 + Kx * (17 + AmpOutGroupLoop cycles ))
// LOADCYLES = 16 (if LDTYPE = 128)
//           = 32 (if LDTYPE = 64)
// STACK VARIABLES for supervisor
#define STACK_BASE                      WKR_STATE_SIZE
#define STACK_AMP_OUTSUBGROUP           0      // word
#define STACK_CONV_GROUP_COUNT          4      // word
#define STACK_OUTCHAN_COUNT             8      // word
#define STACK_INNER_LOOP_ADDR           12     // word
#define STACK_WEIGHTS_PTR_INCR          16     // word ptr increment in amp kernel height
#define STACK_PARTITION_PTR             20
#define STACK_KY_COUNT                  24
#define STACK_INCHAN_COUNT              28     // word
#define STACK_NUM_INCHAN                32
#define STACK_NUM_OUTCHAN_M1            36
#define STACK_NUM_KERNEL_X_M1           40
#define STACK_NUM_KERNEL_Y_M1           44
// Additional 4 bytes to make total stack size a multiple of 8
#define STACK_SIZE                      (STACK_NUM_KERNEL_Y_M1 + 8) // bytes


// registers
#define sup_base                        m0
#define kx_count_s                      m0
#define weights_ptr_h_incr              m1
#define temp_5_s                        m1
#define convgroup_count_s               m1
#define temp_1_s                        m2
#define inchan_count_s                  m2
#define partition_ptr_s                 m2
#define strides_s                       m2
#define inchan_ptr_s                    m3
#define jmp_addr_s                      m3
#define temp_4_s                        m3
#define amp_out_subgroup_count_s        m4
#define ky_count_s                      m4
#define temp_7_s                        m4
#define wkr_function_s                  m5
#define weights_ptr_s_h                 m5
#define outchan_count_s                 m5
#define amp_height_jump_s               m5
#define temp_3_s                        m5
#define outchan_ptr_s                   m6
#define partition_ptr_b_s               m6
#define weights_ptr_s                   m7
#define temp_2_s                        m7
#define outstride_s                     m7
#define inchan_vectors_s                m8
#define in_row_stride_s                 m8
#define temp_6_s                        m8
#define outchan_vectors_s               m9
#define inchans_per_group_s             m9
#define amp_kernel_height_s             m9
#define outchans_per_group_s            m10
#define weights_vectors_s               m10
#define wkr_base                        sp


// for zeroing partials
#define outchan_vectors_z_s             m1
#define outchan_count_z_s               m2
#define zero_info_s_z                   m3
#define outchan_ptr_z_s                 m5
#define wkr_function_s_z                m4
#define convgroup_count_s_z             m7

.macro CONV_NX1 LDTYPE

.if \LDTYPE == LD64
.section .text.CODELET_NAME_64
.align 4
.type CODELET_NAME_64, @function
.globl CODELET_NAME_64
CODELET_NAME_64:

.elseif \LDTYPE == LD128
.section .text.CODELET_NAME_128
.type CODELET_NAME_128, @function
.globl CODELET_NAME_128
CODELET_NAME_128:

.else
.error "Load type not supported"
.endif

.supervisor
//-----------------------------------------------------------------------------
lds16         $temp_1_s, $sup_base, SUP_INSTRIDE/2
lds16         $outstride_s, $sup_base, SUP_OUTSTRIDE/2
ld32          $partition_ptr_b_s, $sup_base, SUP_PARTITION_TABLES/4
// space for worker vertex state, supervisor state and callee-save registers
add           $sp, $sp, -(WKR_STATE_SIZE + 8 + STACK_SIZE)
lds16         $in_row_stride_s, $sup_base, SUP_IN_ROW_STRIDE/2
ld32          $zero_info_s_z, $sup_base, SUP_ZERO_INFO/4
//-----------------------------------------------------------------------------
and           $temp_1_s, $temp_1_s, 0x3FF

ldz16         $temp_5_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
ldz16         $temp_7_s, $sup_base, SUP_NUM_KERNEL_Y_M1/2
st32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
st32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1


// in-stride and out-stride are packed in one register. Out-stride
// must be scaled by 1/2 because 64 bit writes are used and atom size is 32
// bits
// inoutstrides = [0][in-stride][out-stride]
shr           $outstride_s, $outstride_s, 1
shl           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS

st32          $temp_5_s, $sp, (STACK_BASE + STACK_NUM_OUTCHAN_M1)/4
st32          $temp_7_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_Y_M1)/4

st32          $zero_info_s_z, $wkr_base, WKR_ZERO_INFO/4
//-----------------------------------------------------------------------------
shl           $temp_1_s, $temp_1_s, 10
and           $outstride_s, $outstride_s, 0x3FF
and           $in_row_stride_s, $in_row_stride_s, 0x3FF
ldz16         $temp_5_s, $sup_base, SUP_NUM_FILTER_HEIGHT_M1/2
ldz16         $inchans_per_group_s, $sup_base, SUP_NUM_INCHANS_PER_GROUP/2
ldz16         $outchans_per_group_s, $sup_base, SUP_NUM_OUTCHANS_PER_GROUP/2
//-----------------------------------------------------------------------------
shr           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
or            $temp_4_s, $outstride_s, $temp_1_s
shl           $in_row_stride_s, $in_row_stride_s, 20

// A second input strides registers has packed strides
// [in-row-stride1][in-row-stride0][out-stride]
// in-row-stride0 = in-row-stride for kernel height = 4
//                = 1 for other heights
cmpult        $temp_1_s, $temp_5_s, 3
brz           $temp_1_s, SetInStridesHeightEq4_\LDTYPE\()
setzi         $strides_s, 0x400
bri           StridesSet_\LDTYPE\()
SetInStridesHeightEq4_\LDTYPE\():
shr           $strides_s, $in_row_stride_s, 10
StridesSet_\LDTYPE\():
st32          $partition_ptr_b_s, $wkr_base, WKR_PARTITION_BASE/4
#if defined(VECTORLIST_AVAIL_DELTAN)
ldz16         $partition_ptr_b_s, $sup_base, (SUP_PARTITION_TABLES + 4)/2
#else
ld32          $partition_ptr_b_s, $sup_base, (SUP_PARTITION_TABLES + 4)/4
#endif
st32          $temp_4_s, $wkr_base, WKR_INOUTSTRIDES1/4
mul           $temp_5_s, $inchans_per_group_s, SIZEOF_IN_ATOM
//-----------------------------------------------------------------------------
mul           $temp_4_s, $outchans_per_group_s, SIZEOF_OUT_ATOM
or            $strides_s, $strides_s, $outstride_s
shr           $temp_2_s, $outchans_per_group_s, LOG2_AMP_OUTPUTS_HALF
#if defined(VECTORLIST_AVAIL_DELTAN)
shl           $partition_ptr_b_s, $partition_ptr_b_s, (SCALED_PTR32_SHL + 13)
#else
shl           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
#endif
setzi         $wkr_function_s_z, convNx1ZeroOutField

// Scale output and input channels per group to scale input and output offsets.
// Scaling depends on sizeof(input) and sizeof(output)
st32          $temp_5_s, $wkr_base, WKR_INCHANS_PER_GROUP_X2/4
//-----------------------------------------------------------------------------
ld32          $outchan_vectors_z_s,$sup_base, SUP_OUTCHAN_VECTORS/4
or            $strides_s, $strides_s, $in_row_stride_s
ldz16         $temp_6_s, $sup_base, SUP_NUM_KERNEL_X_M1/2
#if defined(VECTORLIST_AVAIL_DELTAN)
or            $partition_ptr_b_s, $partition_ptr_b_s, (TMEM_REGION0_BASE_ADDR << 13)
#else
nop           // keep nop for 6 instructions pipeline
#endif
st32          $temp_4_s, $wkr_base, WKR_OUTCHANS_PER_GROUP_X4/4
add           $temp_4_s, $temp_2_s, -1
//-----------------------------------------------------------------------------
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#else
ld32step      $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#endif
ldz16         $convgroup_count_s_z, $sup_base, SUP_NUM_CONVGROUPS_M1/2
st32          $strides_s, $wkr_base, WKR_INOUTSTRIDES2/4
#if defined(VECTORLIST_AVAIL_DELTAN)
shr           $partition_ptr_b_s, $partition_ptr_b_s, 13
#else
shr           $partition_ptr_b_s, $partition_ptr_b_s, DELTAN_DELTAS_COUNT_BITS
#endif
st32          $temp_6_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_X_M1)/4
add           $temp_6_s, $temp_6_s, 1
ldz16         $amp_kernel_height_s, $sup_base, SUP_NUM_FILTER_HEIGHT_M1/2
//-----------------------------------------------------------------------------

ZeroConvGroup_\LDTYPE\():
  ldz16         $outchan_count_z_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
ZeroOutChanGroup_\LDTYPE\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    // expand scaled pointer
    shl           $outchan_ptr_z_s, $outchan_ptr_z_s, 3
#endif
    sync          TEXCH_SYNCZONE_LOCAL
    st32          $outchan_ptr_z_s, $wkr_base, WKR_OUTCHAN_PTR/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#else
    ld32step      $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#endif
    runall        $wkr_function_s_z, $wkr_base, 0
    brnzdec       $outchan_count_z_s, ZeroOutChanGroup_\LDTYPE\()
  brnzdec       $convgroup_count_s_z, ZeroConvGroup_\LDTYPE\()

mul           $temp_2_s, $temp_6_s, $outchans_per_group_s
st32          $temp_4_s, $sp, (STACK_BASE + STACK_AMP_OUTSUBGROUP)/4
st32          $partition_ptr_b_s, $sp, (STACK_BASE + STACK_PARTITION_PTR)/4
// Jump address for amp height == 1 is set to 0 to take the fast path
setzi         $jmp_addr_s, 0

brz           $amp_kernel_height_s, JmpAddrSelEnd_\LDTYPE\()
mul           $temp_6_s, $temp_2_s, AMP_INPUTS_HALF/2 * SIZEOF_WEIGHTS_ATOM
setzi         $jmp_addr_s, AmpHeightEq2_\LDTYPE\()
// a bubble here instead of using 3 nop instructions
add           $amp_kernel_height_s, $amp_kernel_height_s, -1
brz           $amp_kernel_height_s, JmpAddrSelEnd_\LDTYPE\()
setzi         $jmp_addr_s, AmpHeightEq4_\LDTYPE\()
mul           $temp_6_s, $temp_2_s, AMP_INPUTS_HALF/4 * SIZEOF_WEIGHTS_ATOM
JmpAddrSelEnd_\LDTYPE\():

// The only call to this function may be zero out partials
ldz16         $inchan_count_s, $mzero, $sup_base, SUP_NUM_INCHAN/2
ldz16         $convgroup_count_s, $sup_base, SUP_NUM_CONVGROUPS_M1/2
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#else
ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
mov           $outchan_count_s, $mzero
st32          $jmp_addr_s, $sp, (STACK_BASE + STACK_INNER_LOOP_ADDR)/4
st32          $temp_6_s, $sp, (STACK_BASE + STACK_WEIGHTS_PTR_INCR) / 4
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
st32          $inchan_count_s, $sp, (STACK_BASE + STACK_NUM_INCHAN)/4

brz           $inchan_count_s,  L_sup_conv_end_\LDTYPE\()
add           $inchan_count_s, $inchan_count_s, -1
ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif

ConvGroupLoop_\LDTYPE\():
  // Output channel vectors are increased by numOutChanGroups for each
  // convolution group. The increment at the start is 0. The increment
  // of numOutChanGroups is done before branching here for each conv group.
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $mzero, $mzero, $outchan_vectors_s+=, $outchan_count_s
#else
  ld32step      $mzero, $mzero, $outchan_vectors_s+=, $outchan_count_s
#endif
  ld32          $outchan_count_s, $sp, (STACK_BASE +  STACK_NUM_OUTCHAN_M1)/4
  st32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4

InChanLoop_\LDTYPE\():
    st32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    shl           $inchan_ptr_s, $inchan_ptr_s, 3
#else
    ld32step      $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif
OutChanLoop_\LDTYPE\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      shl           $weights_ptr_s, $weights_ptr_s, 3
      nop
      nop
#endif

      ld32          $partition_ptr_s, $sp, (STACK_BASE + STACK_PARTITION_PTR)/4
      // This load of CCCS is for the case of Amp Kernel Height = 1. In the
      // other two cases, the weights are not loaded in order
      st32          $outchan_count_s, $sp, (STACK_BASE + STACK_OUTCHAN_COUNT)/4
      ld32          $ky_count_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_Y_M1)/4
      put           $CCCSLOAD, $weights_ptr_s

KyLoop_\LDTYPE\():
        st32          $ky_count_s, $sp, (STACK_BASE + STACK_KY_COUNT)/4
        ld32          $kx_count_s, $sp, (STACK_BASE + STACK_NUM_KERNEL_X_M1)/4
KxLoop_\LDTYPE\():
#if defined(VECTOR_AVAIL_SCALED_PTR64)
          // expand scaled pointer
          ldz16         $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
          shl           $outchan_ptr_s, $outchan_ptr_s, 3
#else
          ld32          $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
#endif
          ld32          $amp_out_subgroup_count_s, $sp, (STACK_BASE + STACK_AMP_OUTSUBGROUP)/4
          ld32          $weights_ptr_h_incr, $sp, (STACK_BASE + STACK_WEIGHTS_PTR_INCR)/4
AmpOutGroupLoop_\LDTYPE\():
            // registers $m1 and $m8 are free to use in this function as long
            // as wkr_function_s is set before the run
            ld32          $amp_height_jump_s, $sp, (STACK_BASE + STACK_INNER_LOOP_ADDR)/4
            sync          TEXCH_SYNCZONE_LOCAL
            st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4
            st32          $outchan_ptr_s, $wkr_base, WKR_OUTCHAN_PTR/4
            st32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
            // Different AMP kernel heights have different coefficient loading
            // schemes and they all branch back to AmpHeightRun
            // If there were an brnz instruction with target branch
            // address as a register rather than immediate, for the fallthrough
            // case we could have set the condition to check fail (eg using
            // brnz would become brnz $amp_height_jump_s, $amp_height_jump_s)
            brnz          $amp_height_jump_s, AmpHeightJumpToNeq1_\LDTYPE\()
 AmpHeightEq1_\LDTYPE\():

.if \LDTYPE == LD128

            ld128putcs    0
            ld128putcs    2
            ld128putcs    4
            ld128putcs    6
            ld128putcs    8
            ld128putcs    10
            ld128putcs    12
            ld128putcs    14
            ld128putcs    16
            ld128putcs    18
            setzi         $wkr_function_s, convPartialNx1Flattened
            ld128putcs    20
            ld128putcs    22
            ld128putcs    24
            ld128putcs    26
            ld128putcs    28
            ld128putcs    30

.elseif \LDTYPE == LD64

            ld64putcs    0
            ld64putcs    1
            ld64putcs    2
            ld64putcs    3
            ld64putcs    4
            ld64putcs    5
            ld64putcs    6
            ld64putcs    7
            ld64putcs    8
            ld64putcs    9
            ld64putcs    10
            ld64putcs    11
            ld64putcs    12
            ld64putcs    13
            ld64putcs    14
            ld64putcs    15
            ld64putcs    16
            ld64putcs    17
            ld64putcs    18
            ld64putcs    19
            setzi         $wkr_function_s, convPartialNx1Flattened
            ld64putcs    20
            ld64putcs    21
            ld64putcs    22
            ld64putcs    23
            ld64putcs    24
            ld64putcs    25
            ld64putcs    26
            ld64putcs    27
            ld64putcs    28
            ld64putcs    29
            ld64putcs    30
            ld64putcs    31

.else
// Error if LDTYPE is not supported
.error "Load type not supported"
.endif
AmpHeightRun_\LDTYPE\():
            runall        $wkr_function_s, $wkr_base, 0

            // wind pointer to point to next amp subgroup
            add           $outchan_ptr_s, $outchan_ptr_s, SIZEOF_OUT_ATOM * AMP_OUTPUTS_HALF
            brnzdec       $amp_out_subgroup_count_s, AmpOutGroupLoop_\LDTYPE\()

          add           $partition_ptr_s, $partition_ptr_s, 6 * 4
          ld32          $outchan_count_s, $sp, (STACK_BASE + STACK_OUTCHAN_COUNT)/4
          brnzdec       $kx_count_s, KxLoop_\LDTYPE\()

        get           $weights_ptr_s, $CCCSLOAD
        ld32          $ky_count_s, $sp, (STACK_BASE + STACK_KY_COUNT)/4
        brnzdec       $ky_count_s, KyLoop_\LDTYPE\()
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#else
      ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
      brnzdec       $outchan_count_s, OutChanLoop_\LDTYPE\()
    ld32          $outchan_count_s, $sp, (STACK_BASE + STACK_NUM_OUTCHAN_M1)/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif
    ld32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
    brnzdec       $inchan_count_s, InChanLoop_\LDTYPE\()
  ld32          $inchan_count_s, $sp, (STACK_BASE + STACK_NUM_INCHAN)/4
  ld32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4
  // cannot use delay instruction here because workers may be active
  nop
  nop
  nop
  add           $outchan_count_s, $outchan_count_s, 1
  add           $inchan_count_s, $inchan_count_s, -1
  brnzdec       $convgroup_count_s, ConvGroupLoop_\LDTYPE\()

L_sup_conv_end_\LDTYPE\():
// Restore saved registers
ld32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
ld32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1
add           $sp, $sp, (WKR_STATE_SIZE + STACK_SIZE + 8)
sync          TEXCH_SYNCZONE_LOCAL
br            $lr

// Jump to code fragment to handle kernel heights = 2 and 4
AmpHeightJumpToNeq1_\LDTYPE\():
br            $amp_height_jump_s

// =============================================================================

// Code fragement to load CWEI with coefficients when the weights tensor
// innermost dimensions are [AMP_OUTPUTS_HALF * m][AMP_INPUTS_HALF/2]
AmpHeightEq2_\LDTYPE\():
put           $CCCSLOAD, $weights_ptr_s
add           $weights_ptr_s_h, $weights_ptr_s, $weights_ptr_h_incr

.if \LDTYPE == LD128

ld128putcs    0
ld128putcs    4
ld128putcs    8
ld128putcs    12
ld128putcs    16
ld128putcs    20
ld128putcs    24
ld128putcs    28
put           $CCCSLOAD, $weights_ptr_s_h
ld128putcs    2
ld128putcs    6
setzi         $wkr_function_s, convPartialNx1Flattened
ld128putcs    10
ld128putcs    14
ld128putcs    18
ld128putcs    22
ld128putcs    26
ld128putcs    30

.elseif \LDTYPE == LD64

ld64putcs     0
ld64putcs     1
ld64putcs     4
ld64putcs     5
ld64putcs     8
ld64putcs     9
ld64putcs     12
ld64putcs     13
ld64putcs     16
ld64putcs     17
ld64putcs     20
ld64putcs     21
ld64putcs     24
ld64putcs     25
ld64putcs     28
ld64putcs     29
put           $CCCSLOAD, $weights_ptr_s_h
ld64putcs     2
ld64putcs     3
ld64putcs     6
ld64putcs     7
setzi         $wkr_function_s, convPartialNx1Flattened
ld64putcs     10
ld64putcs     11
ld64putcs     14
ld64putcs     15
ld64putcs     18
ld64putcs     19
ld64putcs     22
ld64putcs     23
ld64putcs     26
ld64putcs     27
ld64putcs     30
ld64putcs     31

.else
// Only 128 and 64 bit loads supported
.error "Load type not supported"
.endif

// update weight pointer by the amount of coefficient
add           $weights_ptr_s, $weights_ptr_s, AMP_OUTPUTS_HALF * SIZEOF_WEIGHTS_ATOM * AMP_INPUTS_HALF/2
bri           AmpHeightRun_\LDTYPE\()

// =============================================================================

// Code fragement to load CWEI with coefficients when the weights tensor
// innermost dimensions are [AMP_OUTPUTS_HALF * m][AMP_INPUTS_HALF/4]
// TODO: T12867 If code is shared between the 64-bit and 128-bit weight load
// versions, then this part of the code can be shared. It needs to be in a
// common section.
AmpHeightEq4_\LDTYPE\():
put           $CCCSLOAD, $weights_ptr_s
add           $weights_ptr_s_h, $weights_ptr_s, $weights_ptr_h_incr
ld64putcs     0
ld64putcs     4
ld64putcs     8
ld64putcs     12
ld64putcs     16
ld64putcs     20
ld64putcs     24
ld64putcs     28
put           $CCCSLOAD, $weights_ptr_s_h
add           $weights_ptr_s_h, $weights_ptr_s_h, $weights_ptr_h_incr
ld64putcs     1
ld64putcs     5
ld64putcs     9
ld64putcs     13
ld64putcs     17
ld64putcs     21
ld64putcs     25
ld64putcs     29
put           $CCCSLOAD, $weights_ptr_s_h
add           $weights_ptr_s_h, $weights_ptr_s_h, $weights_ptr_h_incr
ld64putcs     2
ld64putcs     6
ld64putcs     10
ld64putcs     14
ld64putcs     18
ld64putcs     22
ld64putcs     26
ld64putcs     30
put           $CCCSLOAD, $weights_ptr_s_h
setzi         $wkr_function_s, convPartialNx1Flattened
ld64putcs     3
ld64putcs     7
ld64putcs     11
ld64putcs     15
ld64putcs     19
ld64putcs     23
ld64putcs     27
ld64putcs     31
// update weight pointer by the number of coefficients used up as the next time
add           $weights_ptr_s, $weights_ptr_s, AMP_OUTPUTS_HALF * SIZEOF_WEIGHTS_ATOM * AMP_INPUTS_HALF/4
bri           AmpHeightRun_\LDTYPE\()

.if \LDTYPE == LD64
.size CODELET_NAME_64, . - CODELET_NAME_64
.else
.size CODELET_NAME_128, . - CODELET_NAME_128
.endif
.endm

// =============================================================================

// Instantiate codelets
CONV_NX1 LD128
CONV_NX1 LD64

// =============================================================================
#endif
// =============================================================================
