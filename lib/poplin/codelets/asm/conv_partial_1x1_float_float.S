// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
//
// Contains functions to calculate partials for convolution. Partials and Output
// are used interchangebly in this file. Each worker may process part of a
// contiguous field. This is done by setting up a partition which contains an
// input offset in the input channel group, an output offset in the output field
// and the number of field elements to process.
//

#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_1x1_supervisor.S"

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// =============================================================================

.section ".text.convPartialFlattenedField_f32sisoamp", "ax"
.type convPartialFlattenedField_f32sisoamp, @function
.align 8
.worker
// worker code:
//     num_field_pos = 0
//         24
//     num_field_pos == 1
//         46 + (2 + zeroCyclesPerGroup * num_field_pos) * first_input_group
//     num_field_pos == 2
//         46 + (2 + zeroCyclesPerGroup * num_field_pos) * first_input_group
//     num_field_pos >= 3
//         46  + (2 + zeroCyclesPerGroup * num_field_pos) * first_input_group
//                          + (num_field_pos - 3) * 8
//
// where zeroCyclesPerGroup = 4
//
// first_in_group is set to a 1 for the first input channel group of every
// convolution, in which case no partial is loaded.
//
// Note: 3 extra double word reads are done for num_field_pos={0,1,2}. These are
//       guaranteed to be non-strided
//
// The very first call of the worker requires 11 more cycles
// Otherwise all calls of workers requires 11 cycles more for a given input channel
//                                         8 cycles when an input channel changes

// Total:
convPartialFlattenedField_f32sisoamp:

#define wkr_id                      m0
#define outchan_ptr                 m1
#define tripacked_addr              m0:1
#define partition_w                 m2
#define in_off                      m3
#define out_off                     m4
#define num_elems                   m5
#define stride1                     m6
#define zero_outchan                m6
#define cmp_res                     m7
#define stride2                     m8
#define inchan_ptr                  m9
#define stride3                     m10
#define num_elems_plus_3            m11

#define X_IN_0                      $a0
#define X_IN_1                      $a1
#define X_IN                        $a0:1
#define P_IN                        $a2:3
#define NULL2                       $a4:5
#define P_OUT                       $a6:7
#define NULL1                       $a14
#define NULL                        $azeros

{
get           $wkr_id, $WSR
setzi         $a0, ZAACC_BITMASK
}
{
and           $wkr_id, $wkr_id, CSR_W_WSR__CTXTID_M1__MASK
uput          $FP_CLR, $a0
}
mul           $wkr_id, $wkr_id, 6
ld32          $partition_w, $mvertex_base, WKR_PARTITION/4
// There are always as many partitions as the number of worker contexts. Each
// partition has 3 entries.
ldz16step     $out_off, $wkr_id, $partition_w+=, 1
lds16step     $num_elems, $wkr_id, $partition_w+=, 1
ldz16step     $in_off, $wkr_id, $partition_w+=, 1
add           $num_elems_plus_3, $num_elems, 3

convPartialFlattenedFieldStateRetained_f32sisoamp:

ld32          $inchan_ptr, $mvertex_base, WKR_INCHAN_PTR/4
// Do x8 and add at the same time, n.b. we aren't actually loading here
ld64step      $azeros, $mzero, $inchan_ptr+=, $in_off

convPartialFlattenedFieldStateRetainedInChanPtr_f32sisoamp:

ld32          $stride1, $mvertex_base, WKR_IN_OUT_STRIDES/4

ld32          $outchan_ptr, $mvertex_base, WKR_OUTCHAN_PTR/4
// Do x8 and add at the same time, n.b. we aren't actually loading here
ld64step      $azeros, $mzero, $outchan_ptr+=, $out_off

brneg         $zero_outchan, Amp_start
// The two input pointers will be retained even though store pointer is incremented
tapack        $tripacked_addr, $inchan_ptr, $outchan_ptr, $outchan_ptr

rpt         $num_elems_plus_3, (LZeroLoopEnd - LZeroLoopBegin)/8 - 1
LZeroLoopBegin:
  {
    st64pace      $azeros,          $tripacked_addr+=, $mzero, 0b00
    fnop
  }
  {
    st64pace      $azeros,          $tripacked_addr+=, $mzero, 0b00
    fnop
  }
  {
    st64pace      $azeros,          $tripacked_addr+=, $mzero, 0b00
    fnop
  }
  {
    st64pace      $azeros,          $tripacked_addr+=, $stride1, 0b10
    fnop
  }
LZeroLoopEnd:
Amp_start:

// Check strides to use in the AMP loop. The strides to use are dependent on
// the number of elements to avoid excess strided reads
{
  brneg         $num_elems, SetStridesNumElemsLt3
  fnop
}
// case of num_elems >= 3
// Stride1 = Stride2 = [0][out index][in index] are the default
mov           $stride2, $stride1
setzi         $stride3, 1

AfterStrideSet:

// stride3 is 0 if number of elements = 1 else it is 1
// This avoids 6 extra loads of the output(partials)
// Note that when number of elements = 2, 3 extra non-strided loads are done
// In all other cases, no extra loads are done

// Get compact representation of physical addresses
tapack        $tripacked_addr, $inchan_ptr, $outchan_ptr, $outchan_ptr

// Assumption that groups in conv is directly stored as actual value -1
//   &input += 0, &partials += 1
ld2x64pace      NULL, P_IN, $tripacked_addr+=, $stride1, 0b0011
{
  // &input += 0, &partials += 1
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b0011
  f32sisoamp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P0
}
{
  // &input += 0, &partials += 1
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b0011
  f32sisoamp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P2
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : [out index])
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b1011
  f32sisoamp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P4
}
{
  // &input += 1, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    X_IN, P_IN, $tripacked_addr+=, $stride3, 0b0100
  f32sisoamp    P_OUT, NULL1, P_IN, TAMP_F32_E4_P6
}

// Start providing inputs ----------------------------------------------------
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride3, 0b0111
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
}
{
  // &input += 1, &partials += 0
  ld2x64pace    X_IN, NULL2, $tripacked_addr+=, $stride1, 0b1100
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P1
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride3, 0b0111
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
}
{
  // &input += 1, &partials += 0
  ld2x64pace    X_IN, NULL2, $tripacked_addr+=, $stride1, 0b1100
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P3
}
{
  // &input += 0, &partials += (num_elems >= 3 ? [out index] : 0)
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride2, 0b1011
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
}
{
  // &input += (num_elems == 1 ? 0 : [in index]), &partials += 0
  ld2x64pace    X_IN, NULL2, $tripacked_addr+=, $stride2, 0b1101
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P5
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride3, 0b0111
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
}
{
  // &input += (num_elems == 1 ? 0 : 1), &partials += 0
  ld2x64pace    X_IN, NULL2, $tripacked_addr+=, $stride3, 0b1101
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P7
}
// Start recording output ----------------------------------------------------
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride3, 0b0111
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
}
{
  // &input += (num_elems == 1 ? 0 : 1), &output += 1
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride3, 0b0001
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P1
}
{
  // &input += 0, &partials += (num_elems == 1 ? 0 : 1)
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride3, 0b0111
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
}
{
  // &input += (num_elems == 1 ? 0 : 1), &output += 1
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride3, 0b0001
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P3
}

// exit paths for special cases of num_elems = {1, 2}
brneg         $num_elems, JumpPaths

rpt $num_elems, (Loop_end_Amp - Loop_start_Amp)/8-1
Loop_start_Amp:
  // The reads in the last pass are effectively dummy to avoid code bloat
  {
    // &input += 0, &partials += [out index]
    ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b1011
    f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
  }
  {
    // &input += [in index], &output += 1
    ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0001
    f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P5
  }
  {
    // &input += 0, &partials += 1
    ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b0011
    f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
  }
  {
    // &input += 1, &output += [out index]
    ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b1000
    f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P7
  }
  {
    // &input += 0, &partials += 1
    ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b0011
    f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P0
  }
  {
    // &input += 1, &output += 1
    ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0000
    f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P1
  }
  {
    // &input += 0, &partials += 1
    ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b0011
    f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P2
  }
  {
    // &input += 1, &output += 1
    ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0000
    f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P3
  }
Loop_end_Amp:

{
  // &input += 0, &partials += 0
  ld2x64pace    NULL, P_IN, $tripacked_addr+=, $stride1, 0b1111
  f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P4
}
{
  // &input += [in index], &output += 1
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0001
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P5
}
f32sisoamp    P_OUT, X_IN_0, P_IN, TAMP_F32_E4_P6
{
  // &input += 1, &output += [out index]
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b1000
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P7
}
// Stop providing partials ---------------------------------------------------
f32sisoamp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P0
{
  // &input += 1, &output += 1
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0000
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P1
}
f32sisoamp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P2
{
  // &input += 1, &output += 1
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0000
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P3
}

LNumElemsEq2:

f32sisoamp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P4
{
  // &input += 0, &output += 1
  ldst64pace    X_IN, P_OUT, $tripacked_addr+=, $stride1, 0b0011
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P5
}
f32sisoamp    P_OUT, X_IN_0, NULL, TAMP_F32_E4_P6
{
  // &output += [out index]
  st64pace      P_OUT, $tripacked_addr+=, $stride1, 0b10
  f32sisoamp    NULL, X_IN_1, NULL, TAMP_F32_E4_P7
}
// Stop providing input ------------------------------------------------------
f32sisoamp    P_OUT, NULL1, NULL, TAMP_F32_E4_P0
{
  // &output += 1
  st64pace      P_OUT, $tripacked_addr+=, $stride1, 0b00
  f32sisoamp    P_OUT, NULL1, NULL, TAMP_F32_E4_P2
}

// &output += 1
st64pace      P_OUT, $tripacked_addr+=, $stride1, 0b00

LNumElemsEq1:

// This may need to change if partials for the next loop could be loaded
// with the store of old results
f32sisoamp    P_OUT, NULL1, NULL, TAMP_F32_E4_P4
{
  // &output += 1
  st64pace      P_OUT,          $tripacked_addr+=, $stride1, 0b00
  f32sisoamp    P_OUT, NULL1, NULL, TAMP_F32_E4_P6
}
// &output += 0
st64pace      P_OUT,          $tripacked_addr+=, $stride1, 0b11

L_end_fn:
exitz         $m15

// Code fragment to set strides for num_elems = {0, 1, 2}
// Jumps back to main program after setting strides
SetStridesNumElemsLt3:
add           $cmp_res, $num_elems, 1
brz           $cmp_res, LStrideCheckElemsEq2
add           $cmp_res, $num_elems, 2
brneg         $cmp_res, L_end_fn
// Number of elems = 1
// Stride1 = [0][0][in index]
// Stride2 = Stride3 = [0][0][0]
// Need to zero the [out index] bitfield (bits 10-20) inside $stride1, passed
// by supervisor, to avoid accessing elements outside the 'out/partial' vector.
and           $stride1, $stride1, 0x3FF

mov           $stride2, $mzero
mov           $stride3, $mzero
bri           AfterStrideSet

LStrideCheckElemsEq2:
// Number of elems = 2
// Stride1 = [0][out index][in index]
// Stride2 = [0][0][in index]
// Stride3 = [0][0][1]
and           $stride2, $stride1, 0x3FF
setzi         $stride3, 1
bri           AfterStrideSet

// This code fragment jumps to the appropriate point in the main program for
// number of elements = {1, 2}
JumpPaths:
add           $cmp_res, $num_elems, 1
brz           $cmp_res, LNumElemsEq2
bri           LNumElemsEq1


.size convPartialFlattenedField_f32sisoamp, . - convPartialFlattenedField_f32sisoamp

// =============================================================================
// Instantiate codelets

CONV_1x1_SUPERVISOR float float false 8 f32sisoamp
CONV_1x1_SUPERVISOR float float true  8 f32sisoamp

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
