// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
// Computes an 1x1 convolution using MAC. A contiguous field is partitioned
// between workers for each position of the kernel element.
//
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_zero_output_stack.h"

#define conv_hzmac_sup_half_float __runCodelet_poplin__ConvPartialHorizontalMac1x1___half_float_true

// =============================================================================

// Vertex input, output and weight data atom size constants

// sizeof(input) in bytes
#define SIZEOF_IN_ATOM                  2
// sizeof(output) in bytes
#define SIZEOF_OUT_ATOM                 4
// sizeof(weights) in bytes
#define SIZEOF_WEIGHTS_ATOM             2

// =============================================================================

// Vertex state offsets
#define SUP_INCHAN_VECTORS              0    // word
#define SUP_WEIGHTS_VECTORS             4    // word
#define SUP_OUTCHAN_VECTORS             8    // word
#define SUP_ZERO_INFO                   12   // word
#define SUP_WORKLIST                    16   // word
#define SUP_NUM_OUTCHAN_M1              20   // short
#define SUP_INSTRIDE                    24   // word
#define SUP_OUTSTRIDE                   28   // word
#define SUP_NUM_INCHAN                  32   // short
#define SUP_NUM_CONVGROUPS_M1           34   // short
#define SUP_NUM_OUTCHANS_PER_GROUP      36   // short
#define SUP_NUM_INCHANS_PER_GROUP       38   // short

// =============================================================================


// Worklist partition fields: must match the order of entries in worklists
#define PARTITION_ELEM_BYTES            2
#define PARTITION_OUTOFFSET             0    // short
#define PARTITION_NUM_ELEMS             2    // short
#define PARTITION_INOFFSET              4    // short
#define PARTITION_SIZE                 ((PARTITION_INOFFSET + PARTITION_ELEM_BYTES)/PARTITION_ELEM_BYTES)

// =============================================================================

// =============================================================================

// Vertex state shared between workers
// The supervisor sets up a common state for workers to use
// Allocate codelet specific worker stack after zero output worker (WKR_ZERO_OUTPUT_STACK)
#define WKR_INSTRIDE                    WKR_ZERO_OUTPUT_STACK
#define WKR_OUTSTRIDE                   (WKR_INSTRIDE + 4)
#define WKR_INCHAN_PTR                  (WKR_OUTSTRIDE + 4)
#define WKR_PARTITION_PTR               (WKR_INCHAN_PTR + 4)
#define WKR_OUTCHANS_PER_GROUP_IN_BYTES (WKR_PARTITION_PTR + 4)
#define WKR_NO_ZEROS_PATH               (WKR_OUTCHANS_PER_GROUP_IN_BYTES + 4)
#define WKR_INCHANS_PER_GROUP_IN_BYTES  (WKR_NO_ZEROS_PATH + 4)
#define WKR_OCG_COUNT                   (WKR_INCHANS_PER_GROUP_IN_BYTES + 4)
#define WKR_ICG_COUNT                   (WKR_OCG_COUNT + 4)
#define WKR_WEIGHTS_PTR                 (WKR_ICG_COUNT + 4)
#define WKR_STATE_SIZE                  (WKR_WEIGHTS_PTR + 4) // bytes

// =============================================================================
// Stack parameters
#define WKR_STACK_INADDR                4

// =============================================================================

// worker register mapping
#define wkr_id_v                       m0
#define instride_v                     m11
#define outstride_v                    m10
#define partition_v                    m1
#define out_chans_per_group_in_bytes   m2
#define num_fp_saved_v                 m9
#define in_addr_v                      m0
#define weights_ptr_v                  m1
#define out_addr_v                     m3
#define in_offset_v                    m6
#define in_chans_per_group_in_bytes    m8
#define in_chan_mask_v                 m8
#define out_offset_v                   m7
#define icg_count_v                    m5
#define ocg_count_v                    m2
#define num_fp_v                       m4
#define tripacked_addr                 m0:1
#define fp_clr_reg                     a0

// =============================================================================

.section ".text.convHorizMac1x1Flattened_half_float", "ax"
.type convHorizMac1x1Flattened_half_float, @function
.align 8
.worker
convHorizMac1x1Flattened_half_float:
get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK
mul           $wkr_id_v, $wkr_id_v, PARTITION_SIZE*PARTITION_ELEM_BYTES
ld32          $instride_v, $mvertex_base, WKR_INSTRIDE/4
ld32          $outstride_v, $mvertex_base, WKR_OUTSTRIDE/4
ld32          $partition_v, $mvertex_base, WKR_PARTITION_PTR/4

ldz16         $out_offset_v, $wkr_id_v, $partition_v, PARTITION_OUTOFFSET/PARTITION_ELEM_BYTES
ldz16         $num_fp_saved_v, $wkr_id_v, $partition_v, PARTITION_NUM_ELEMS/PARTITION_ELEM_BYTES
ldz16         $in_offset_v, $wkr_id_v, $partition_v, PARTITION_INOFFSET/PARTITION_ELEM_BYTES

ld32          $in_chans_per_group_in_bytes, $mvertex_base, WKR_INCHANS_PER_GROUP_IN_BYTES/4
ld32          $out_chans_per_group_in_bytes, $mvertex_base, WKR_OUTCHANS_PER_GROUP_IN_BYTES/4
mul           $out_offset_v, $out_offset_v, $out_chans_per_group_in_bytes
mul           $in_offset_v, $in_offset_v, $in_chans_per_group_in_bytes
ld32          $icg_count_v, $mvertex_base, WKR_ICG_COUNT/4
and           $in_chan_mask_v, $in_chans_per_group_in_bytes, 0x7

convHorizMac1x1Flattened_half_float_retained:

// form input address
{
  ld32          $in_addr_v, $mvertex_base, WKR_INCHAN_PTR/4
  setzi         $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}
{
  ld32          $out_addr_v, $mvertex_base, WKR_OUTCHAN_PTR/4
  // clear accumulator: only required once because we always feed zeros
  // with gina instruction
  uput          $FP_CLR, $fp_clr_reg
}
add           $in_addr_v, $in_addr_v, $in_offset_v

add           $num_fp_v, $num_fp_saved_v, -1
brneg         $num_fp_v, LConvEnd

st32          $in_addr_v, $mworker_base, WKR_STACK_INADDR/4
brnz          $in_chan_mask_v, LProcessTwoInputsFp

ld32          $ocg_count_v, $mvertex_base, WKR_OCG_COUNT/4
brz           $ocg_count_v, LProcessFourInputsFpOcgZero

LProcessFourInputsFpLoop:
    ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
    ld32          $ocg_count_v, $mvertex_base, WKR_OCG_COUNT/4
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
    rpt $icg_count_v, (2f - 1f)/8 - 1
1:
      {
        ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
        f16v4cmac     $a0:1, $a2:3
      }
2:
    {
      // reset source pointer back to start for the next output channel
      ld32          $in_addr_v, $mworker_base, WKR_STACK_INADDR/4
      f16v4cmac     $a0:1, $a2:3
    }
    {
      // load previous partial
      ld32          $a6, $out_offset_v, $out_addr_v, 0
      f32v2gina     $a4:5, $azeros, 0
    }
    {
      ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
      f32add        $a4, $a4, $a5
    }
    {
      // ocg_count is always >=1 and this branch is always taken
      brnzdec       $ocg_count_v, LProcessFourInputsOcgLoop
      f32add        $a7, $a4, $a6
    }


// The specialisation with ocg=zero is required for the case of matrix multiplications where
// one of the dimension is of size 1.
LProcessFourInputsFpOcgZero:
#define  no_zeros_path_v  ocg_count_v
 ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
 ld32          $no_zeros_path_v, $mvertex_base, WKR_NO_ZEROS_PATH/4
 brz           $no_zeros_path_v, LProcessFourInputsFpOcgZeroLoopNoLoadPartials
 #define instride_ocg0_v ocg_count_v


LProcessFourInputsFpOcgZeroLoop:
   // Need to avoid over-reading the input due to input striding. Hence,
   // read at the beginning of loop (avoid pre-loading at the end of loop)
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
    {
      rpt $icg_count_v, (2f - 1f)/8 - 1
      fnop
    }
1:
      {
        ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
        f16v4cmac     $a0:1, $a2:3
      }
2:
    {
      sub           $instride_ocg0_v, $instride_v, $icg_count_v
      f16v4cmac     $a0:1, $a2:3
    }
    {
      ld64step      $azeros, $mzero, $in_addr_v+=, $instride_ocg0_v
      f32v2gina     $a4:5, $azeros, 0
    }
    {
      ld32          $a6, $out_offset_v, $out_addr_v, 0
      f32add        $a4, $a4, $a5
    }
    {
      ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
      f32add        $a4, $a4, $a6
    }
    st32step      $a4, $out_offset_v, $out_addr_v+=, $outstride_v
    brnzdec       $num_fp_v, LProcessFourInputsFpOcgZeroLoop
exitz         $mzero



LProcessFourInputsFpOcgZeroLoopNoLoadPartials:
   // Need to avoid over-reading the input due to input striding. Hence,
   // read at the beginning of loop (avoid pre-loading at the end of loop)
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
    {
      rpt $icg_count_v, (2f - 1f)/8 - 1
      fnop
    }
1:
      {
        ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
        f16v4cmac     $a0:1, $a2:3
      }
2:
    {
      sub           $instride_ocg0_v, $instride_v, $icg_count_v
      f16v4cmac     $a0:1, $a2:3
    }
    {
      ld64step      $azeros, $mzero, $in_addr_v+=, $instride_ocg0_v
      f32v2gina     $a4:5, $azeros, 0
    }
    {
      ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
      f32add        $a4, $a4, $a5
    }
    st32step      $a4, $out_offset_v, $out_addr_v+=, $outstride_v
    brnzdec       $num_fp_v, LProcessFourInputsFpOcgZeroLoopNoLoadPartials
exitz         $mzero


LProcessFourInputsOcgLoop:
    rpt $icg_count_v, (2f - 1f)/8 - 1
1:
      {
        ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
        f16v4cmac     $a0:1, $a2:3
      }
2:
    {
      // reset source pointer back to start for the next output channel
      ld32           $in_addr_v, $mworker_base, WKR_STACK_INADDR/4
      f16v4cmac      $a0:1, $a2:3
    }
    {
      st32step        $a7, $out_offset_v, $out_addr_v+=, 1
      // load previous partial
      f32v2gina       $a4:5, $azeros, 0
    }
    {
      ld32            $a6, $out_offset_v, $out_addr_v, 0
      f32add          $a4, $a4, $a5
    }
    {
      ld2x64pace      $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
      f32add          $a7, $a4, $a6
    }
    brnzdec         $ocg_count_v, LProcessFourInputsOcgLoop

  st32step        $a7, $out_offset_v, $out_addr_v+=, $outstride_v
  ld64step        $azeros, $mzero, $in_addr_v+=, $instride_v
  st32            $in_addr_v, $mzero, $mworker_base, WKR_STACK_INADDR/4
  brnzdec         $num_fp_v, LProcessFourInputsFpLoop
exitz         $mzero


// Process a vector of 2 input channels at a time
LProcessTwoInputsFp:

LProcessTwoInputsFpLoop:
    ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
    ld32          $ocg_count_v, $mvertex_base, WKR_OCG_COUNT/4
    ld32step      $a0, $mzero, $in_addr_v+=, 1

LProcessTwoInputsOcgLoop:
      ld32step      $a1, $mzero, $weights_ptr_v+=, 1
      {
        rpt $icg_count_v, (2f - 1f)/8 - 1
        fnop
      }
1:
        {
          ld32step      $a0, $mzero, $in_addr_v+=, 1
          f16v2cmac     $a0, $a1
        }
        {
          ld32step      $a1, $mzero, $weights_ptr_v+=, 1
          fnop
        }
2:
      {
        // load previous partial
        ld32          $a6, $out_offset_v, $out_addr_v, 0
        f16v2cmac     $a0, $a1
      }
      {
        ld32          $in_addr_v, $mworker_base, WKR_STACK_INADDR/4
        f32v2gina     $a4:5, $azeros, 0
      }
      {
        ld32step      $a0, $mzero, $in_addr_v+=, 1
        f32add        $a4, $a4, $a6
      }
      st32step        $a4, $out_offset_v, $out_addr_v+=, 1
      brnzdec         $ocg_count_v, LProcessTwoInputsOcgLoop

    ld32step        $azero, $out_offset_v, $out_addr_v+=, $outstride_v
    add             $out_addr_v, $out_addr_v, -4
    ld32step        $azero, $mzero, $in_addr_v+=, $instride_v
    st32            $in_addr_v, $mzero, $mworker_base, WKR_STACK_INADDR/4
    brnzdec         $num_fp_v, LProcessTwoInputsFpLoop

LConvEnd:
exitz         $mzero

.size convHorizMac1x1Flattened_half_float, . - convHorizMac1x1Flattened_half_float


// =============================================================================
// =============================================================================


// STACK VARIABLES for supervisor
#define STACK_BASE                      WKR_STATE_SIZE
#define STACK_CONV_GROUP_COUNT          0      // word
#define STACK_OUTCHAN_COUNT             4      // word
#define STACK_PARTITION_PTR             8      // word
#define STACK_INCHAN_GROUPS             12     // word
#define STACK_INCHAN_COUNT              16     // word
#define STACK_SIZE                      (STACK_INCHAN_COUNT + 8) // bytes
#define TOT_STACK_SIZE                  (WKR_STATE_SIZE + 8 + STACK_SIZE)

// registers
#define sup_base                        m0
#define outchan_countp_s                m1
#define temp_3_s                        m1
#define outchan_ptr_s                   m2
#define icg_count_s                     m2
#define convgroup_count_s               m2
#define inchan_ptr_s                    m3
#define weights_ptr_s                   m4
#define temp_2_s                        m4
#define instride_s                      m5
#define inchan_count_s                  m5
#define partition_ptr_s                 m6
#define outchan_count_s                 m6
#define wkr_function_s                  m7
#define outstride_s                     m7
#define ocg_count_s                     m7
#define inchan_vectors_s                m8
#define outchan_vectors_s               m9
#define outchans_per_group_s            m9
#define inchans_per_group_s             m10
#define weights_vectors_s               m10
#define wkr_base                        sp

DEF_STACK_USAGE  TOT_STACK_SIZE  conv_hzmac_sup_half_float
.section .text.conv_hzmac_sup_half_float
.align 4
.type conv_hzmac_sup_half_float, @function
.globl conv_hzmac_sup_half_float
.supervisor

conv_hzmac_sup_half_float:

#define outchan_vectors_z_s             m1
#define outchan_count_z_s               m2
#define outchan_ptr_z_s                 m5
#define wkr_function_s_z                m3
#define zero_info_s_z                   m8
#define convgroup_count_s_z             m7

ld32          $outchan_vectors_z_s,$sup_base, SUP_OUTCHAN_VECTORS/4
ldz16         $convgroup_count_s_z, $sup_base, SUP_NUM_CONVGROUPS_M1/2
ldz16         $temp_2_s, $mzero, $sup_base, SUP_NUM_INCHAN/2
ld32          $partition_ptr_s, $sup_base, SUP_WORKLIST/4
// space for worker vertex state, supervisor state and callee save
add           $sp, $sp, -TOT_STACK_SIZE
ld32          $zero_info_s_z, $sup_base, SUP_ZERO_INFO/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#else
ld32step      $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#endif
// Run worker to zero partials
setzi         $wkr_function_s_z, convNx1ZeroOutField_float
nop
st32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
st32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1
st32          $zero_info_s_z, $wkr_base, WKR_ZERO_INFO/4
st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4
brz           $zero_info_s_z, LZeroingComplete

ZeroConvGroup:
  ldz16         $outchan_count_z_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
ZeroOutChanGroup:
#if defined(VECTOR_AVAIL_SCALED_PTR64)
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
    setzi         $wkr_function_s_z, convNx1ZeroOutFieldReentry_float
    brnzdec       $outchan_count_z_s, ZeroOutChanGroup
  brnzdec       $convgroup_count_s_z, ZeroConvGroup

LZeroingComplete:

ldz16         $inchans_per_group_s, $sup_base, SUP_NUM_INCHANS_PER_GROUP/2
st32          $zero_info_s_z, $wkr_base, WKR_NO_ZEROS_PATH/4
// The only call to the convolution may be to zero out partials
brz           $temp_2_s,  L_sup_conv_end
ld32          $outstride_s, $sup_base, SUP_OUTSTRIDE/4
ldz16         $outchans_per_group_s, $sup_base, SUP_NUM_OUTCHANS_PER_GROUP/2


// instride = actualStride * inChansPerGroup / 4 for inChansPerGroup a multiple of 4
//             (divide by 4 because ld64 is used to wind the pointer)
//           = actualStride * inChansPerGroup / 2 for all other values of inChansPerGroup
//             (divide by 2 because ld32 is used to wind the pointer)
ld32          $instride_s, $sup_base, SUP_INSTRIDE/4
and           $temp_2_s, $inchans_per_group_s, 0x3
nop
shr           $icg_count_s, $inchans_per_group_s, 1
add           $outstride_s, $outstride_s, 1
ldz16         $outchan_count_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
nop
brnz          $temp_2_s, IcgNotMult4
shr           $instride_s, $instride_s, 1
shr           $icg_count_s, $icg_count_s, 1
nop
nop
nop

IcgNotMult4:
st32          $outstride_s, $wkr_base, WKR_OUTSTRIDE/4
shr           $instride_s, $instride_s, 1
add           $icg_count_s, $icg_count_s, -1


// Scale output and input channels per group to scale input and output offsets.
// Scaling depends on sizeof(input) and sizeof(output)
mul           $inchans_per_group_s, $inchans_per_group_s, SIZEOF_IN_ATOM
add           $ocg_count_s, $outchans_per_group_s, -1

mul           $outchans_per_group_s, $outchans_per_group_s, SIZEOF_OUT_ATOM
add           $instride_s, $instride_s, -1
ldz16         $temp_3_s, $sup_base, SUP_NUM_INCHAN/2

st32          $icg_count_s, $wkr_base, WKR_ICG_COUNT/4
st32          $inchans_per_group_s, $wkr_base, WKR_INCHANS_PER_GROUP_IN_BYTES/4
st32          $ocg_count_s, $mzero, $wkr_base, WKR_OCG_COUNT/4
st32          $outchans_per_group_s, $mzero, $wkr_base, WKR_OUTCHANS_PER_GROUP_IN_BYTES/4

ldz16         $convgroup_count_s, $sup_base, SUP_NUM_CONVGROUPS_M1/2

st32          $instride_s, $wkr_base, WKR_INSTRIDE/4
add           $inchan_count_s, $temp_3_s, -1
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4

ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
setzi         $wkr_function_s, convHorizMac1x1Flattened_half_float
add           $outchan_countp_s, $outchan_count_s, 1
st32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4


ConvGroupLoop:
  st32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4
InChanLoop:
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
    // expand scaled pointer
    shl           $inchan_ptr_s, $inchan_ptr_s, 3
#else
    ld32step      $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif
OutChanLoop:
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
      shl           $weights_ptr_s, $weights_ptr_s, 3
#else
      ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      ldz16         $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
      shl           $outchan_ptr_s, $outchan_ptr_s, 3
#else
      ld32          $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
#endif
      sync          TEXCH_SYNCZONE_LOCAL
      st32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
      st32          $weights_ptr_s, $wkr_base, WKR_WEIGHTS_PTR/4
      st32          $outchan_ptr_s, $wkr_base, WKR_OUTCHAN_PTR/4
      runall        $wkr_function_s, $wkr_base, 0
      setzi         $wkr_function_s, convHorizMac1x1Flattened_half_float_retained
      brnzdec       $outchan_count_s, OutChanLoop
    ldz16         $outchan_count_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
    brnzdec       $inchan_count_s, InChanLoop
  ld32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4
  ld32          $inchan_count_s,  $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $mzero, $mzero, $outchan_vectors_s+=, $outchan_countp_s
#else
  ld32step      $mzero, $mzero, $outchan_vectors_s+=, $outchan_countp_s
#endif
  nop
  nop
  nop
  brnzdec       $convgroup_count_s, ConvGroupLoop


L_sup_conv_end:
// Restore saved registers
ld32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
ld32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1
add           $sp, $sp, TOT_STACK_SIZE
sync          TEXCH_SYNCZONE_LOCAL
br            $lr

.size conv_hzmac_sup_half_float, . - conv_hzmac_sup_half_float

#endif

// =============================================================================
// =============================================================================
