// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
//
// Contains functions to calculate partials for convolution. Partials and Output
// are used interchangeably in this file. Each worker may process part of a
// contiguous field. This is  done by setting up a partition which contains an
// input offset in the input channel group, an output offset in the output field
// and the number of output field elements to process. Both input and output
// may be strided and the output flipped

#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_1x1_supervisor.S"

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// =============================================================================

.section ".text.convPartialFlattenedField_f16v4hihoamp", "ax"
.type convPartialFlattenedField_f16v4hihoamp, @function
.align 8
.worker

// worker code:
//       num_field_pos = 0
//                24
//       num_field_pos == 1
//                39 + (2 + zeroCyclesPerGroup) * first_input_group
//       num_field_pos == 2
//                40 + (2 + zeroCyclesPerGroup * num_field_pos) * first_input_group
//       num_field_pos >= 3
//                41  + (2 + zeroCyclesPerGroup * num_field_pos) * first_input_group
//                    + (num_field_pos - 3) * 4
//
// where zeroCyclesPerGroup = 2
//
// first_in_group is set to a 1 for the first input channel group of every
// convolution, in which case no partial is loaded.
//
// Note: 2 extra double word reads are done for num_field_pos={2}. These are
//       guaranteed to be non-strided
//
// The very first call of the worker requires 11 more cycles if
// Otherwise all calls of workers requires 11 cycles more for a given input channel
//                                         9 cycles when an input channel changes

// Total:
convPartialFlattenedField_f16v4hihoamp:

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
ldz16step     $out_off, $wkr_id, $partition_w+=, 1
lds16step     $num_elems, $wkr_id, $partition_w+=, 1
ldz16step     $in_off, $wkr_id, $partition_w+=, 1
add           $num_elems_plus_3, $num_elems, 3

convPartialFlattenedFieldStateRetained_f16v4hihoamp:

ld32          $inchan_ptr, $mvertex_base, WKR_INCHAN_PTR/4
// Do x8 and add at the same time, n.b. we aren't actually loading here
ld64step      $azeros, $mzero, $inchan_ptr+=, $in_off

convPartialFlattenedFieldStateRetainedInChanPtr_f16v4hihoamp:

ld32          $stride1, $mvertex_base, WKR_IN_OUT_STRIDES/4
ld32          $outchan_ptr, $mvertex_base, WKR_OUTCHAN_PTR/4
// Do x8 and add at the same time, n.b. we aren't actually loading here
ld64step      $azeros, $mzero, $outchan_ptr+=, $out_off

brneg         $zero_outchan, Amp_start
// The two input pointers will be retained even though store pointer is incremented
tapack        $tripacked_addr, $inchan_ptr, $outchan_ptr, $outchan_ptr
rpt           $num_elems_plus_3, (LZeroLoopEnd - LZeroLoopBegin)/8 - 1
LZeroLoopBegin:
  {
    // partials += 1
    st64pace      $azeros,          $tripacked_addr+=, $mzero, 0b00
    fnop
  }
  {
    // partials += outstride
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
// This reduces extra loads of the output(partials)
// Note that when number of elements = 2, 2 extra non-strided loads are done
// In all other cases, no extra loads are done

// Get compact representation of physical addresses
tapack        $tripacked_addr, $inchan_ptr, $outchan_ptr, $outchan_ptr

// Assumption that groups in conv is directly stored as actual value -1

// Note: dummy loads used in the code as there is no ld64pace instruction
// inchan_ptr   += 0
// partials_ptr += 1
ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $stride1, 0b0011
f16v4hihoamp  $a6, $azeros, $a2, TAMP_F16V4_E4_P0

{
  // inchan_ptr += 0
  // partials_ptr += 0         (0 for num_elems=1)
  //              += outstride (for num_elems>=2)
  ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $stride1, 0b1011
  f16v4hihoamp  $a6, $azeros, $a3, TAMP_F16V4_E4_P1
}
f16v4hihoamp  $a6, $azeros, $a2, TAMP_F16V4_E4_P2
{
  // load input += 1
  // load partials += 0 (for num_elems = 1)
  //               += 1 (for num_elems = 2)
  //               += 1 (for num_elems > 2)
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b0100
  f16v4hihoamp  $a6, $azeros, $a3, TAMP_F16V4_E4_P3
}
{
  // load input += 1
  // load partials += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b1100
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P0
}
{
  // load input += 1
  // load partials += 0         (for num_elems = 1)
  //               += 0         (for num_elems = 2)
  //               += outstride (for num_elems > 2)
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride2, 0b1000
  f16v4hihoamp  $a6, $a0:1, $a3, TAMP_F16V4_E4_P1
}
{
  // load input += 0        (for num_elems = 1)
  //            += instride (for num_elems >= 2)
  // load partials += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride2, 0b1101
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
}
{
  // load input += 0       (for num_elems = 1)
  //            += 1       (for num_elems >= 2)
  // load partials += 0    (for num_elems = 0)
  //               += 1    (for num_elems >= 2)
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b0101
  f16v4hihoamp  $a6, $a0:1, $a3, TAMP_F16V4_E4_P3
}
{
  // load input += 0 (for num_elems = 1)
  //            += 1 (for num_elems >= 2)
  // load partials += 0
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b1101
  f16v4hihoamp  $a4, $a0:1, $a2, TAMP_F16V4_E4_P0
}
{
  // load input += 0            (for num_elems = 1)
  //            += 1            (for num_elems >= 2)
  // load partials += 0         (for numelems = 1)
  //               += 0         (for numelems = 2)
  //               += outstride (for numelems > 2)
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride2, 0b1001
  f16v4hihoamp  $a5, $a0:1, $a3, TAMP_F16V4_E4_P1
}

// exit paths for special cases of num_elems = {1, 2}
brneg         $num_elems, JumpPaths

// repeat loop entered only if num_elems > 3
rpt $num_elems, (Loop_end_Amp-Loop_start_Amp)/8-1
Loop_start_Amp:
  {
    // load input += stride,
    // store partials += 1
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b0001
    f16v4hihoamp  $a4, $a0:1, $a2, TAMP_F16V4_E4_P2
  }
  {
    // load input += 1,
    // load partials += 1
    ld2x64pace  $a0:1, $a2:3, $tripacked_addr+=, $stride1, 0b000000
    f16v4hihoamp  $a5, $a0:1, $a3, TAMP_F16V4_E4_P3
  }
  {
    // load input += 1
    // store partials += stride
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b1000
    f16v4hihoamp  $a4, $a0:1, $a2, TAMP_F16V4_E4_P0
  }
  {
    // load input += 1,
    // load partials += stride
    ld2x64pace  $a0:1, $a2:3, $tripacked_addr+=, $stride1, 0b1000
    f16v4hihoamp  $a5, $a0:1, $a3, TAMP_F16V4_E4_P1
  }
Loop_end_Amp:

{
  // load input += instride
  // store partials += 0
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b1101
  f16v4hihoamp  $a6, $a0:1, $a2, TAMP_F16V4_E4_P2
}
{
  // load input += 1, store partials += 1
  // partial stored again to use pace instruction (partials load cannot be done
  // because the read pointer is already increased by stride)
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b0000
  f16v4hihoamp  $a7, $a0:1, $a3, TAMP_F16V4_E4_P3
}
{
  // load input += 1
  // store partials += 0
  ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $stride1, 0b1100
  f16v4hihoamp  $a4, $a0:1, $azero, TAMP_F16V4_E4_P0
}
{
  // load input += 1
  // store partials += stride
  // partial stored again to use pace instruction (partials load cannot be done
  // because the read pointer is already increased by stride)
  ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $stride1, 0b1000
  f16v4hihoamp  $a5, $a0:1, $azero, TAMP_F16V4_E4_P1
}
LNumElemsEq2:
{
  // load input += stride
  // store partials += 1
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b0001
  f16v4hihoamp  $a4, $a0:1, $azero, TAMP_F16V4_E4_P2
}
f16v4hihoamp  $a5, $a0:1, $azero, TAMP_F16V4_E4_P3
{
  // store partials += stride
  st64pace      $a4:5, $tripacked_addr+=, $stride1, 0b10
  f16v4hihoamp  $a4, $a0:1, $azero, TAMP_F16V4_E4_P0
}
f16v4hihoamp  $a5, $a0:1, $azero, TAMP_F16V4_E4_P1

LNumElemsEq1:

// This may need to change if partials for the next loop could be loaded
// with the store of old results
{
  // store partials += 1
  st64pace      $a4:5,          $tripacked_addr+=, $stride1, 0b00
  f16v4hihoamp  $a4, $azeros, $azero, TAMP_F16V4_E4_P2
}
f16v4hihoamp  $a5, $azeros, $azero, TAMP_F16V4_E4_P3
// store partials += 0
st64pace      $a4:5,          $tripacked_addr+=, $stride1, 0b00

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
// Stride1 = Stride2 = Stride3 = [0][0][0]
mov           $stride1, $mzero
mov           $stride2, $mzero
mov           $stride3, $mzero
bri           AfterStrideSet

LStrideCheckElemsEq2:
// Number of elems = 2
// Stride1 = [0][out index][in index]
// Stride2 = [0][0][in index]
// Stride2 = [0][0][1]
and           $stride2, $stride1, 0x3FF
setzi         $stride3, 1
bri           AfterStrideSet

// This code fragment jumps to the appropriate point in the main program for
// number of elements = {1, 2}
JumpPaths:
add           $cmp_res, $num_elems, 1
brz           $cmp_res, LNumElemsEq2
bri           LNumElemsEq1

.size convPartialFlattenedField_f16v4hihoamp, . - convPartialFlattenedField_f16v4hihoamp

// =============================================================================
// Instantiate codelets

CONV_1x1_SUPERVISOR half half false 8 f16v4hihoamp
CONV_1x1_SUPERVISOR half half true  8 f16v4hihoamp

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
