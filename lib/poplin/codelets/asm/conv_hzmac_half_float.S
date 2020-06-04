// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Computes an nx1 convolution using AMP. A contiguous field is partitioned
// between workers for each position of the kernel element.
//
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define conv_hzmac_sup_half_float __runCodelet_poplin__ConvPartialHorizontalMac___half_float_true

// =============================================================================


// =============================================================================

// Vertex input, output and weight data atom size constants

// sizeof(input) in bytes
#define SIZEOF_IN_ATOM                  2
// sizeof(output) in bytes
#define SIZEOF_OUT_ATOM                 4
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

// Worklist partition fields: must match the order of entries in worklists
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


// Vertex state shared between workers
// The supervisor sets up a common state for workers to use
#define WKR_INSTRIDE                    0      // word
#define WKR_OUTSTRIDE                   4      // word
#define WKR_INCHAN_PTR                  8      // word
#define WKR_OUTCHAN_PTR                 12     // word
#define WKR_PARTITION_PTR               16     // word
#define WKR_ZERO_INFO                   20     // word
#define WKR_OUTCHANS_PER_GROUP_X4       24     // word
#define WKR_PARTITION_BASE              28     // word
#define WKR_INCHANS_PER_GROUP_X2        32     // word
#define WKR_OCG_COUNT                   36
#define WKR_ICG_COUNT                   40
#define WKR_WEIGHTS_PTR                 44
#define WKR_STATE_SIZE                  (WKR_WEIGHTS_PTR + 4) // bytes

// =============================================================================


// Non loop overhead:
//      zero Partitions:         7
//      non-zero partitions:     16
//
// Loop performance:
//   With input channels per group % 4 == 0:
//     partition loop: 19 + num_fp * (7 + outChansPerGroup * (7 + inChansPerGroup/4))
//   Otherwise:
//     partition loop: 19 + num_fp * (7 + outChansPerGroup * (7 + 2*(inChansPerGroup/2 - 1)))


.section ".text.convHorizMacFlattened", "ax"
.type convHorizMacFlattened, @function
.align 8
.worker
convHorizMacFlattened:
// worker register mapping
#define wkr_id_v                       m0
#define instride_v                     m0
#define partition_table_v              m1
#define partition_struct_v             m1
#define partition_v                    m1
#define out_addr_v                     m1
#define num_partitions_v               m2
#define num_fp_v                       m3
#define outstride_v                    m4
#define in_offset_v                    m5
#define out_offset_v                   m5
#define out_chans_per_group_x4         m6
#define ocg_count_v                    m6
#define partition_base_v               m7
#define icg_count_v                    m7
#define in_chans_per_group_x2          m7
#define fp_clr_reg                     a0
#define in_addr_v                      m8
#define weights_ptr_v                  m9
#define tripacked_addr                 m8:9
#define inchan_ptr_v                   m10
#define outchan_ptr_v                  m11

// This is in the worker stack
#define WKR_STACK_PARTITIONS_PTR        0
#define WKR_STACK_INADDR                4

get           $wkr_id_v, $WSR
and           $wkr_id_v, $wkr_id_v, CSR_W_WSR__CTXTID_M1__MASK

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

// Compensate for extra load in the loops. This could be done in the supervisor
ld32          $instride_v, $mvertex_base, WKR_INSTRIDE/4
add           $instride_v, $instride_v, -1
ld32          $outstride_v, $mvertex_base, WKR_OUTSTRIDE/4

// This following approximates num_partitions/3 - 1 for values
// [3:3:2^14-1]. The case of zero is handled above
{
  mul           $num_partitions_v, $num_partitions_v, 21845
  setzi         $fp_clr_reg, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
}

// clear accumulator: only required once because we always feed zeros
// with gina instruction
{
  shr           $num_partitions_v, $num_partitions_v, 16
  uput          $FP_CLR, $fp_clr_reg
}

PartitionLoop:
  ldz16         $num_fp_v, $partition_v, PARTITION_NUM_ELEMS/2

  ld32          $in_chans_per_group_x2, $mvertex_base, WKR_INCHANS_PER_GROUP_X2/4
  ld32          $out_chans_per_group_x4, $mvertex_base, WKR_OUTCHANS_PER_GROUP_X4/4

  // form input address
  ldz16         $in_offset_v, $partition_v, PARTITION_INOFFSET/2
  mul           $in_offset_v, $in_offset_v, $in_chans_per_group_x2
  add           $in_addr_v, $inchan_ptr_v, $in_offset_v

  // form output address
  // Note this relies on the fact that out offset is the first entry in the
  // partition. This allows us to wind the pointer to the next partition
  ldz16step     $out_offset_v, $mzero, $partition_v+=, 3
  st32          $partition_v, $mworker_base, WKR_STACK_PARTITIONS_PTR/4
  brz           $num_fp_v, L_Partition_end
  add           $num_fp_v, $num_fp_v, -1

  st32          $in_addr_v, $mworker_base, WKR_STACK_INADDR/4
  mul           $out_offset_v, $out_offset_v, $out_chans_per_group_x4
  add           $out_addr_v, $outchan_ptr_v, $out_offset_v

  and           $in_chans_per_group_x2, $in_chans_per_group_x2, 0x7
  brnz          $in_chans_per_group_x2, LProcessTwoInputsFp

  // Process a vector of 4 input channels at a time.
  // Optim: We could still use this for input channels > 4 by doing residual
  //        channels. This would require checks on stride as well.
  ld32          $icg_count_v, $mvertex_base, WKR_ICG_COUNT/4


LProcessFourInputsFpLoop:
    ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
    ld32          $ocg_count_v, $mvertex_base, WKR_OCG_COUNT/4
    ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000

LProcessFourInputsOcgLoop:
    {
      rpt $icg_count_v, (LProcessFourInputsIcgLoopEnd - LProcessFourInputsIcgLoopBegin)/8 - 1
      fnop
    }
LProcessFourInputsIcgLoopBegin:
        {
          ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
          f16v4cmac     $a0:1, $a2:3
        }
LProcessFourInputsIcgLoopEnd:
      {
        // reset source pointer back to start for the next output channel
        ld32           $in_addr_v, $mworker_base, WKR_STACK_INADDR/4
        f16v4cmac      $a0:1, $a2:3
      }
      {
        // load previous partial
        ld32          $a6, $mzero, $out_addr_v, 0
        f32v2gina     $a4:5, $azeros, 0
      }
      {
        ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
        f32add          $a4, $a4, $a5
      }
      f32add          $a4, $a4, $a6
      st32step        $a4, $mzero, $out_addr_v+=, 1
      brnzdec         $ocg_count_v, LProcessFourInputsOcgLoop

    ld32step        $azero, $mzero, $out_addr_v+=, $outstride_v
    ld64step        $azeros, $mzero, $in_addr_v+=, $instride_v
    st32            $in_addr_v, $mzero, $mworker_base, WKR_STACK_INADDR/4
    brnzdec         $num_fp_v, LProcessFourInputsFpLoop

  ld32          $partition_v, $mworker_base, WKR_STACK_PARTITIONS_PTR/4
  brnzdec       $num_partitions_v, PartitionLoop
  exitz         $mzero


// Process a vector of 2 input channels at a time
LProcessTwoInputsFp:
  ld32          $icg_count_v, $mvertex_base, WKR_ICG_COUNT/4

LProcessTwoInputsFpLoop:
    ld32          $weights_ptr_v, $mvertex_base, WKR_WEIGHTS_PTR/4
    ld32          $ocg_count_v, $mvertex_base, WKR_OCG_COUNT/4
    ld32step      $a0, $mzero, $in_addr_v+=, 1

LProcessTwoInputsOcgLoop:
      ld32step      $a1, $mzero, $weights_ptr_v+=, 1

      rpt $icg_count_v, (LProcessTwoInputsLoopEnd - LProcessTwoInputsLoopBegin)/8 - 1
LProcessTwoInputsLoopBegin:
        {
          ld32step      $a0, $mzero, $in_addr_v+=, 1
          f16v2cmac     $a0, $a1
        }
        {
          ld32step      $a1, $mzero, $weights_ptr_v+=, 1
          fnop
        }
LProcessTwoInputsLoopEnd:
      {
        // load previous partial
        ld32          $a6, $mzero, $out_addr_v, 0
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
      st32step        $a4, $mzero, $out_addr_v+=, 1
      brnzdec         $ocg_count_v, LProcessTwoInputsOcgLoop

    ld32step        $azero, $mzero, $out_addr_v+=, $outstride_v
    ld32step        $azero, $mzero, $in_addr_v+=, $instride_v
    st32            $in_addr_v, $mzero, $mworker_base, WKR_STACK_INADDR/4
    brnzdec         $num_fp_v, LProcessTwoInputsFpLoop

L_Partition_end:
  ld32            $partition_v, $mworker_base, WKR_STACK_PARTITIONS_PTR/4
  brnzdec         $num_partitions_v, PartitionLoop

L_Conv_end:
exitz         $mzero

.size convHorizMacFlattened, . - convHorizMacFlattened


// =============================================================================


// =============================================================================

// Performance:
// 63 + numConvGroups * (23 + numInChanGroups
//                       * (15 + numOutChanGroups
//                         * (10 + KernelElems
//                           * (16 + innerLoopCycles)))
//
// Where innerLoopCycles are vertex cycles
//

// STACK VARIABLES for supervisor
#define STACK_BASE                      WKR_STATE_SIZE
#define STACK_CONV_GROUP_COUNT          0      // word
#define STACK_OUTCHAN_COUNT             4      // word
#define STACK_PARTITION_PTR             8      // word
#define STACK_K_COUNT                   12     // word
#define STACK_INCHAN_COUNT              16     // word
#define STACK_SIZE                      (STACK_INCHAN_COUNT + 8) // bytes
#define TOT_STACK_SIZE                  (WKR_STATE_SIZE + 8 + STACK_SIZE)

// registers
#define sup_base                        m0
#define weights_ptr_h_incr              m1
#define outchan_ptr_s                   m2
#define icg_count_s                     m2
#define convgroup_count_s               m2
#define outchan_count_s                 m2
#define inchan_ptr_s                    m3
#define weights_ptr_s                   m4
#define temp_2_s                        m4
#define instride_s                      m5
#define k_count_s                       m5
#define inchan_count_s                  m5
#define partition_ptr_s                 m6
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
// From base and delta, create base and expand pointer to vector containing
// deltaN (i.e. 18 bit delta offset and 14 bit size in number of elements)
// The instructions used in the calculation of the partition parameters are
// spread in the code to avoid
ld32          $partition_ptr_s, $sup_base, SUP_PARTITION_TABLES/4
// space for worker vertex state, supervisor state and callee save
add           $sp, $sp, -TOT_STACK_SIZE

#define outchan_vectors_z_s             m1
#define outchan_count_z_s               m2
#define outchan_ptr_z_s                 m5
#define wkr_function_s_z                m3
#define zero_info_s_z                   m8
#define convgroup_count_s_z             m7

ld32          $outchan_vectors_z_s,$sup_base, SUP_OUTCHAN_VECTORS/4
ld32          $zero_info_s_z, $sup_base, SUP_ZERO_INFO/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
ldz16step     $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#else
ld32step      $outchan_ptr_z_s, $mzero, $outchan_vectors_z_s+=, 1
#endif
// Run worker to zero partials
setzi         $wkr_function_s_z, convNx1ZeroOutField_float
shl           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS

st32          $m9, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 0
st32          $m10, $sp, (WKR_STATE_SIZE + STACK_SIZE)/4 + 1
st32          $zero_info_s_z, $wkr_base, WKR_ZERO_INFO/4

ldz16         $convgroup_count_s_z, $sup_base, SUP_NUM_CONVGROUPS_M1/2
ldz16         $temp_2_s, $mzero, $sup_base, SUP_NUM_INCHAN/2
shr           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS

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

ldz16         $inchans_per_group_s, $sup_base, SUP_NUM_INCHANS_PER_GROUP/2
st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_BASE/4

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
#if defined(VECTORLIST_AVAIL_DELTAN)
ldz16         $partition_ptr_s, $sup_base, (SUP_PARTITION_TABLES + 4)/2
#else
ld32          $partition_ptr_s, $sup_base, (SUP_PARTITION_TABLES + 4)/4
#endif

shr           $icg_count_s, $inchans_per_group_s, 1
st32          $outstride_s, $wkr_base, WKR_OUTSTRIDE/4
nop
nop
brnz          $temp_2_s, IcgNotMult4
shr           $instride_s, $instride_s, 1
shr           $icg_count_s, $icg_count_s, 1
nop
nop

IcgNotMult4:
#if defined(VECTORLIST_AVAIL_DELTAN)
shl           $partition_ptr_s, $partition_ptr_s, (SCALED_PTR32_SHL + 13)
#else
shl           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
#endif

// weight pointer increment per kernel element = inChansPerGroup * outChansPerGroup
mul           $weights_ptr_h_incr, $inchans_per_group_s, $outchans_per_group_s
shr           $instride_s, $instride_s, 1
add           $icg_count_s, $icg_count_s, -1


// Scale output and input channels per group to scale input and output offsets.
// Scaling depends on sizeof(input) and sizeof(output)
mul           $inchans_per_group_s, $inchans_per_group_s, SIZEOF_IN_ATOM
add           $ocg_count_s, $outchans_per_group_s, -1
#if defined(VECTORLIST_AVAIL_DELTAN)
or            $partition_ptr_s, $partition_ptr_s, (TMEM_REGION0_BASE_ADDR << 13)
#else
nop           // keep nop for 6 instructions pipeline
#endif
mul           $outchans_per_group_s, $outchans_per_group_s, SIZEOF_OUT_ATOM
st32          $instride_s, $wkr_base, WKR_INSTRIDE/4
st32          $icg_count_s, $wkr_base, WKR_ICG_COUNT/4
st32          $inchans_per_group_s, $wkr_base, WKR_INCHANS_PER_GROUP_X2/4
st32          $ocg_count_s, $mzero, $wkr_base, WKR_OCG_COUNT/4
#if defined(VECTORLIST_AVAIL_DELTAN)
shr           $partition_ptr_s, $partition_ptr_s, 13
#else
shr           $partition_ptr_s, $partition_ptr_s, DELTAN_DELTAS_COUNT_BITS
#endif
st32          $outchans_per_group_s, $mzero, $wkr_base, WKR_OUTCHANS_PER_GROUP_X4/4

ldz16         $convgroup_count_s, $sup_base, SUP_NUM_CONVGROUPS_M1/2
ldz16         $inchan_count_s, $sup_base, SUP_NUM_INCHAN/2
ld32          $inchan_vectors_s, $sup_base, SUP_INCHAN_VECTORS/4
ld32          $outchan_vectors_s, $sup_base, SUP_OUTCHAN_VECTORS/4
setzi         $wkr_function_s, convHorizMacFlattened
st32          $partition_ptr_s, $sp, (STACK_BASE + STACK_PARTITION_PTR)/4
ld32          $weights_vectors_s, $sup_base, SUP_WEIGHTS_VECTORS/4
add           $inchan_count_s, $inchan_count_s, -1

ConvGroupLoop:
  st32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4
InChanLoop:
    st32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
#if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
    // expand scaled pointer
    shl           $inchan_ptr_s, $inchan_ptr_s, 3
#else
    ld32step      $inchan_ptr_s, $mzero, $inchan_vectors_s+=, 1
#endif
    ldz16         $outchan_count_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
OutChanLoop:
      ld32          $partition_ptr_s, $sp, (STACK_BASE + STACK_PARTITION_PTR)/4
      ldz16         $k_count_s, $sup_base, SUP_NUM_KERNEL_M1/2
#if defined(VECTOR_AVAIL_SCALED_PTR64)
      ldz16step     $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
      shl           $weights_ptr_s, $weights_ptr_s, 3
#else
      ld32step      $weights_ptr_s, $mzero, $weights_vectors_s+=, 1
#endif
      st32          $outchan_count_s, $sp, (STACK_BASE + STACK_OUTCHAN_COUNT)/4
KLoop:
#if defined(VECTOR_AVAIL_SCALED_PTR64)
        ldz16         $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
        shl           $outchan_ptr_s, $outchan_ptr_s, 3
#else
        ld32          $outchan_ptr_s, $mzero, $outchan_vectors_s, $outchan_count_s
#endif
        sync          TEXCH_SYNCZONE_LOCAL
        st32          $partition_ptr_s, $wkr_base, WKR_PARTITION_PTR/4
        st32          $inchan_ptr_s, $wkr_base, WKR_INCHAN_PTR/4
        st32          $weights_ptr_s, $wkr_base, WKR_WEIGHTS_PTR/4
        st32          $outchan_ptr_s, $wkr_base, WKR_OUTCHAN_PTR/4
        ld32          $outchan_count_s, $sp, (STACK_BASE + STACK_OUTCHAN_COUNT)/4
        runall        $wkr_function_s, $wkr_base, 0
        // NUM_CONTEXTS(6) * SIZEOF_UINTPTR(4)
        add           $partition_ptr_s,$partition_ptr_s, 6 * 4
        ldz16step     $mzero, $mzero, $weights_ptr_s+=, $weights_ptr_h_incr
        brnzdec       $k_count_s, KLoop
      brnzdec       $outchan_count_s, OutChanLoop
    ld32          $inchan_count_s, $sp, (STACK_BASE + STACK_INCHAN_COUNT)/4
    brnzdec       $inchan_count_s, InChanLoop
  ldz16         $inchan_count_s, $sup_base, SUP_NUM_INCHAN/2
  ldz16         $outchan_count_s, $sup_base, SUP_NUM_OUTCHAN_M1/2
  add           $outchan_count_s, $outchan_count_s, 1
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step     $mzero, $mzero, $outchan_vectors_s+=, $outchan_count_s
#else
  ld32step      $mzero, $mzero, $outchan_vectors_s+=, $outchan_count_s
#endif
  ld32          $convgroup_count_s, $sp, (STACK_BASE + STACK_CONV_GROUP_COUNT)/4
  nop
  add           $inchan_count_s, $inchan_count_s, -1
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
