// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Contains functions to calculate partials for convolution. This is a worker
// function to process a contiguous field for specific AMP variant. This is
// done by setting up a partition which contains an input offset in the input
// channel group, an output offset in the output field and the number of output
// field elements to process. Both input and output may be strided and the
// output flipped

#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "conv_partial_1x1_supervisor.S"

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// =============================================================================


.macro CONV_1x1_WORKER COMMAND

.section ".text.convPartialFlattenedField_\COMMAND\()", "ax"
.type convPartialFlattenedField_\COMMAND\(), @function
.align 8
.worker

// worker code:
//
// first_in_group is set to a 1 for the first input channel group of every
// convolution, in which case no partial is loaded
//
// first_in_group = 1: (i.e. zeroing of partials)
//        num_field_pos = 0
//                      11
//        num_field_pos == 1
//                      24
//        num_field_pos == 2
//                      26
//        num_field_pos >= 3
//                      26  + (num_field_pos - 3) * 4
//
//
// first_in_group = 0: (i.e. no zeroing of partials)
//        num_field_pos = 0
//                      14
//        num_field_pos == 1
//                      28
//        num_field_pos == 2
//                      30
//        num_field_pos >= 3
//                      29  + (num_field_pos - 3) * 4
//
//
// The very first call of the worker requires 11 more cycles
// Otherwise all calls of workers requires 11 cycles more for a given input channel
//                                         8 cycles when an input channel changes

// Total:
convPartialFlattenedField_\COMMAND\():

#define wkr_id                      m0
#define outchan_ptr                 m1
#define tripacked_addr              m0:1
#define partition_w                 m2
#define eq2flag                     m2
#define in_off                      m3
#define out_off                     m4
#define num_elems                   m5
#define stride1                     m6
#define cmp_res                     m7
#define out_off_x8                  m8
#define inchan_ptr                  m9
#define stride3                     m10
#define const_stride_m2             m11

convPartialFlattenedFieldStateRetained_\COMMAND\():
ld32          $inchan_ptr, $mvertex_base, WKR_INCHAN_PTR/4
// Do x8 and add at the same time, n.b. we aren't actually loading here
ld64step      $azeros, $mzero, $inchan_ptr+=, $in_off

convPartialFlattenedFieldStateRetainedInChanPtr_\COMMAND\():
{
  ld32          $outchan_ptr, $mvertex_base, WKR_OUTCHAN_PTR/4
  setzi         $a0, ZAACC_BITMASK
}

Amp_start_\COMMAND\():

// can do a ld128 as partials are always in interleaved memory and the
// second pointer increment doesn not affect the store pointer
{
  ld128step     $a0:3, $out_off_x8, $outchan_ptr+=, $out_off
  uput          $FP_CLR, $a0
}
{
  // Get compact representation of physical addresses
  tapack        $tripacked_addr, $inchan_ptr, $outchan_ptr, $outchan_ptr
  \COMMAND  $a6:7, $azeros, $a0:1, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 0
  // partials_ptr1 += 1         -> load into a4:5
  ld2x64pace    $azeros, $a4:5, $tripacked_addr+=, $mzero, 0b0011
  \COMMAND  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P1
}
brneg         $num_elems, NumElemsEq1AndEq2_\COMMAND\()
{
  // input_ptr += 0
  // partials_ptr1 += outstride -> load into a2:3
  ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $stride1, 0b1011
  \COMMAND  $a6:7, $azeros, $a4:5, TAMP_F16V4_E4_P2
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // restore write pointer to correct address after writing original value
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  // partials_ptr2 += (-2048)   -> store from a4:5
  ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $const_stride_m2, 0b100000
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
{
  // input_ptr += instride      -> load into a0:1
  // partials_ptr1 += outstride -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride1, 0b1001
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P2
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr1 += 1         -> load into a2:3
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}


rpt $num_elems, (Loop_end_Amp_\COMMAND\()-Loop_start_Amp_\COMMAND\())/8-1
Loop_start_Amp_\COMMAND\():
  // The reads in the last pass are effectively dummy to avoid code bloat
  {
    // input_ptr += instride      -> load into a0:1
    // partials_ptr1 += outstride -> load into a2:3
    // partials_ptr2 += 1         -> store from a4:5
    ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $stride1, 0b001001
    \COMMAND  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P2
  }
  {
    // input_ptr += 1             -> load into a0:1
    // partials_ptr1 += 1         -> load into a2:3
    // partials_ptr2 += 1         -> store from a6:7
    ld2xst64pace  $a0:3, $a6:7, $tripacked_addr+=, $mzero, 0b000000
    \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
  }
  {
    // input_ptr += 1             -> load into a0:1
    // partials_ptr1 += 1         -> load into a2:3
    // partials_ptr2 += 1         -> store from a4:5
    ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $mzero, 0b000000
    \COMMAND  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
  }
  {
    // input_ptr += 1             -> load into a0:1
    // partials_ptr1 += 1         -> load into a2:3
    // partials_ptr2 += outstride -> store from a4:5
    ld2xst64pace  $a0:3, $a6:7, $tripacked_addr+=, $stride1, 0b100000
    \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
  }
Loop_end_Amp_\COMMAND\():

{
  // input_ptr += instride      -> load into a0:1
  // partials_ptr1 += 0         -> load into a2:3
  // partials_ptr2 += 1         -> store from a4:5
  ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $stride1, 0b001101
  \COMMAND  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P2
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr2 += 1         -> store from a6:7
  ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr2 += 1         -> store from a4:5
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr2 += outstride -> store from a6:7
  ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $stride1, 0b1000
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
LNumElemsEq2_\COMMAND\():
{
  // input_ptr += 1             -> load into a0:1
  // partials_ptr2 += 1         -> store from a4:5
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $mzero, 0b000
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P2
}
{
  // partials_ptr2 += 1         -> store from a6:7
  st64pace      $a6:7, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
}
{
  // partials_ptr2 += 1         -> store from a4:5
  st64pace      $a4:5, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  // partials_ptr2 += outstride -> store from a6:7
  st64pace      $a6:7, $tripacked_addr+=, $stride1, 0b10
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}

LNumElemsEq1_\COMMAND\():

// This may need to change if partials for the next loop could be loaded
// with the store of old results
{
  // partials_ptr2 += 1         -> store from a4:5
  st64pace      $a4:5,          $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a4:5, $azeros, $azeros, TAMP_F16V4_E4_P2
}
{
  // partials_ptr2 += 1 -> store from a6:7
  st64pace      $a6:7,          $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a6:7, $azeros, $azeros, TAMP_F16V4_E4_P3
}

// partials_ptr2 += 1         -> store from a4:5
st64pace      $a4:5,          $tripacked_addr+=, $mzero, 0b00

// partials_ptr2 += 1 -> store from a6:7
st64pace      $a6:7,          $tripacked_addr+=, $mzero, 0b00

L_end_fn_\COMMAND\():
exitz         $m15

// Handles the case of number of elements <=2
// stride1 at any point contains strides for both input and output. These
// may be modified to avoid overreading partials
NumElemsEq1AndEq2_\COMMAND\():
// This code fragment is called if number of elements are 0, 1, or 2
add           $cmp_res, $num_elems, 1
cmpeq         $eq2flag, $cmp_res, $mzero
or            $stride3, $eq2flag, $const_stride_m2
brz           $cmp_res, SetStride1NumElemsEq1_\COMMAND\()
add           $cmp_res, $cmp_res, 1
brneg         $cmp_res, L_end_fn_\COMMAND\()
setzi         $stride1, 0

SetStride1NumElemsEq1_\COMMAND\():
{
  ld2x64pace    $azeros, $a2:3, $tripacked_addr+=, $stride1, 0b1011
  \COMMAND  $a6:7, $azeros, $a4:5, TAMP_F16V4_E4_P2
}
{
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b0100
  \COMMAND  $a6:7, $azeros, $a2:3, TAMP_F16V4_E4_P3
}
{
  // restore write pointer to correct address after writing original value
  ld2xst64pace  $a0:3, $a4:5, $tripacked_addr+=, $stride3, 0b100100
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b0100
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
{
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride1, 0b1101
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P2
}
{
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b1101
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P3
}
{
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b1101
  \COMMAND  $a4:5, $a0:1, $a2:3, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $a0:1, $a2:3, $tripacked_addr+=, $stride3, 0b1101
  \COMMAND  $a6:7, $a0:1, $a2:3, TAMP_F16V4_E4_P1
}
brz             $eq2flag, LNumElemsEq1_\COMMAND\()
bri             LNumElemsEq2_\COMMAND\()

//------------------------------------------------------------------------------
// Code path to deal with case when partials are zero.
convPartialFlattenedField_\COMMAND\()_Z:

get           $wkr_id, $WSR
and           $wkr_id, $wkr_id, CSR_W_WSR__CTXTID_M1__MASK

// each partition is 3 ushorts
mul           $wkr_id, $wkr_id, 6
ld32          $partition_w, $mvertex_base, WKR_PARTITION/4

ldz16step     $out_off, $wkr_id, $partition_w+=, 1
lds16step     $num_elems, $wkr_id, $partition_w+=, 1
ldz16step     $in_off, $wkr_id, $partition_w+=, 1
setzi         $const_stride_m2, 0xFF800  // -2 * 1024

// We could keep out_off / 16 + 1 in the worklist and save a thread cycle here.
// Kept as is for consistency between different specialisations.
shl           $out_off_x8, $out_off, 3
shr           $out_off, $out_off, 1
add           $out_off, $out_off, 1

ld32          $stride1, $mvertex_base, WKR_IN_OUT_STRIDES/4

convPartialFlattenedFieldStateRetained_\COMMAND\()_Z:

ld32          $inchan_ptr, $mvertex_base, WKR_INCHAN_PTR/4
// Do x8 and add at the same time, n.b. we aren't actually loading here
ld64step      $azeros, $mzero, $inchan_ptr+=, $in_off

convPartialFlattenedFieldStateRetainedInChanPtr_\COMMAND\()_Z:
{
  ld32          $outchan_ptr, $mvertex_base, WKR_OUTCHAN_PTR/4
  setzi         $a0, ZAACC_BITMASK
}
{
  // Do x8 and add at the same time, n.b. we aren't actually loading here
  add           $outchan_ptr, $outchan_ptr, $out_off_x8
  uput          $FP_CLR, $a0
}

LNoPartialsToLoad_\COMMAND\():
tapack        $tripacked_addr, $inchan_ptr, $outchan_ptr, $outchan_ptr
brneg         $num_elems, NumElemsEq1AndEq2_Z_\COMMAND\()
ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $mzero, 0b00
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride1, 0b01
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P2
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $mzero, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}

rpt $num_elems, (Loop_end_Amp_Z_\COMMAND\()-Loop_start_Amp_Z_\COMMAND\())/8-1
Loop_start_Amp_Z_\COMMAND\():
  // The reads in the last pass are effectively dummy to avoid code bloat
  {
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b0001
    \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P2
  }
  {
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $mzero, 0b0000
    \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
  }
  {
    ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $mzero, 0b0000
    \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
  }
  {
    ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $stride1, 0b1000
    \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
  }
Loop_end_Amp_Z_\COMMAND\():

{
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $stride1, 0b0001
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P2
}
{
  ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
}
{
  ldst64pace    $a0:1, $a4:5, $tripacked_addr+=, $mzero, 0b0000
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  ldst64pace    $a0:1, $a6:7, $tripacked_addr+=, $stride1, 0b1000
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
bri LNumElemsEq2_\COMMAND\()

// Handles the case of number of elements <=2
// stride1 at any point contains strides for both input and output. These
// may be modified to avoid overreading partials
NumElemsEq1AndEq2_Z_\COMMAND\():
// This code fragment is called if number of elements are 0, 1, or 2
add           $cmp_res, $num_elems, 1
cmpeq         $stride3, $cmp_res, $mzero
brz           $cmp_res, SetStride1NumElemsEq1_Z_\COMMAND\()
add           $cmp_res, $cmp_res, 1
brneg         $cmp_res, L_end_fn_\COMMAND\()
// preserve msb
shr           $stride1, $stride1, 31
shl           $stride1, $stride1, 31

SetStride1NumElemsEq1_Z_\COMMAND\():
ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b00
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b00
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride1, 0b01
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P2
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b01
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P3
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b01
  \COMMAND  $a4:5, $a0:1, $azeros, TAMP_F16V4_E4_P0
}
{
  ld2x64pace    $a0:1, $azeros, $tripacked_addr+=, $stride3, 0b01
  \COMMAND  $a6:7, $a0:1, $azeros, TAMP_F16V4_E4_P1
}
brz             $stride3, LNumElemsEq1_\COMMAND\()
bri             LNumElemsEq2_\COMMAND\()

.size convPartialFlattenedField_\COMMAND\(), . - convPartialFlattenedField_\COMMAND\()
.endm


// =============================================================================
// Instantiate codelets
// Workers
#if (__IPU_ARCH_VERSION__ == 2) || (__IPU_ARCH_VERSION__ == 21)
  CONV_1x1_WORKER f16v4hihov4amp
#endif
CONV_1x1_WORKER f16v4sisoamp


// Instantiate codelets
// Supervisors
#if (__IPU_ARCH_VERSION__ == 2) || (__IPU_ARCH_VERSION__ == 21)
  CONV_1x1_SUPERVISOR half half false 16 4 f16v4hihov4amp true false
  CONV_1x1_SUPERVISOR half half true  16 4 f16v4hihov4amp true false
  CONV_1x1_SUPERVISOR half half false 16 4 f16v4hihov4amp true true
  CONV_1x1_SUPERVISOR half half true  16 4 f16v4hihov4amp true true
#endif

CONV_1x1_SUPERVISOR half float false 8 4 f16v4sisoamp true false
CONV_1x1_SUPERVISOR half float true  8 4 f16v4sisoamp true false
CONV_1x1_SUPERVISOR half float false 8 4 f16v4sisoamp true true
CONV_1x1_SUPERVISOR half float true  8 4 f16v4sisoamp true true

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
