// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

#define VERTEX_XMINUSAXPLUSBY_CONST_FAST __runCodelet_popops__XMinusaXPlusbY2D___half_true_true
#define VERTEX_XMINUSAXPLUSBY_TENSOR_FAST __runCodelet_popops__XMinusaXPlusbY2D___half_false_true

#define VERTEX_XMINUSAXPLUSBY_CONST_SLOW __runCodelet_popops__XMinusaXPlusbY2D___half_true_false
#define VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW __runCodelet_popops__XMinusaXPlusbY2D___half_false_false

// This is used as an entry point from the half, half, float version, when it
// is decided that the float scale is OK when cast to half
#define VERTEX_COMMON __XMinusaXPlusbY2D___half_common

// constants
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 1
#define VERTEX_DATA_B_OFFSET 2
#define VERTEX_SCALE_OFFSET 3
#define VERTEX_SCALE_B_OFFSET 4
// The XMINUSAXPLUSBY variants have 2 16-bit constants stored at VERTEX_SCALE_AB_OFFSET
// which can be accessed as one 32 bit word
#define VERTEX_SCALE_AB_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define data m3
#define dataSize m4
#define dataSizeD4 m5
#define dataB m6
#define origDataSize m7
#define triPtr m8:9
#define triPtrData m8
#define triPtrDataB m9
#define offset m10
#define memConstraints m11

// float variables
#define data0 a0:1
#define dataB0 a2:3
#define data1 a4:5
#define data1i0 a4
#define data1i1 a5
#define dataB1 a6:7
#define dataB1i0 a6
#define dataB1i1 a7

// scratch variables
#define mscratch m8
#define ascratch a6
#define ascratch2 a7
#define ascratch3 a0

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR64
#define SCALED_PTR64_SHIFTS 3
#endif

.globl VERTEX_XMINUSAXPLUSBY_CONST_FAST
.type VERTEX_XMINUSAXPLUSBY_CONST_FAST, @function

.globl VERTEX_XMINUSAXPLUSBY_TENSOR_FAST
.type VERTEX_XMINUSAXPLUSBY_TENSOR_FAST, @function

.globl VERTEX_XMINUSAXPLUSBY_CONST_SLOW
.type VERTEX_XMINUSAXPLUSBY_CONST_SLOW, @function

.globl VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW
.type VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW, @function

.macro CHOOSE_FAST_OR_SLOW_PATH FAST_PATH_LABEL
  // The fast path is always OK if constraints were applied
  brnz $memConstraints, \FAST_PATH_LABEL
  // Or if the data start is far enough apart.  It could be ok in some other
  // circumstances but this is time consuming to check correctly.
  sub $mscratch, $data, $dataB
  abs $mscratch, $mscratch
  // +8 is to account for really wanting a <= instruction
  cmpult $mscratch, $mscratch, (2 * TMEM_ELEMSIZE) + 8
  brz $mscratch, \FAST_PATH_LABEL
.endm

DEF_STACK_USAGE 0 .text.VERTEX_XMINUSAXPLUSBY_CONST_SLOW
.section .text.VERTEX_XMINUSAXPLUSBY_CONST_SLOW
.align 4
VERTEX_XMINUSAXPLUSBY_CONST_SLOW:
  setzi $memConstraints, 0
  bri 1f
VERTEX_XMINUSAXPLUSBY_CONST_FAST:
  setzi $memConstraints, 1
1:
  // Implement (1 - a) X + b * Y as follows:
  //     -aX + bY + X
  //
  // The implementation is to be performed in 32-bit precision in the
  // accumulators.

  // load vertex state specific to this version of the vertex :
  // constant A,B pair of halves.  32 bit aligned
  //
  //negate scale_a
  {
    ld32  $ascratch3, $mvertex_base, $mzero, VERTEX_SCALE_AB_OFFSET
    setzi $ascratch, -1.0h
  }
  {
    ld32  $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
    f16v2exp  $ascratch2, $azero
  }
  {
    ld32  $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
    sort4x16lo $ascratch, $ascratch, $ascratch2
  }
  {
    bri   xminusaxplusby_continue
    f16v2mul $ascratch, $ascratch, $ascratch3
  }
.size VERTEX_XMINUSAXPLUSBY_TRUE, .-VERTEX_XMINUSAXPLUSBY_CONST_SLOW


DEF_STACK_USAGE 0 .text.VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW
.section .text.VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW
.align 4
  // Implement (1 - a) X + b * Y as follows:
  //     -aX + bY + X
  //
  // The implementation is to be performed in 32-bit precision in the
  // accumulators.
VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW:
  setzi $memConstraints, 0
  bri 1f
VERTEX_XMINUSAXPLUSBY_TENSOR_FAST:
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensors A,B
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  //negate scale_a
  {
    ldb16 $ascratch, $mzero, $data, 0
    setzi $ascratch2, -1.0h
  }
  {
    ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
    f16v2mul $ascratch, $ascratch, $ascratch2
  }
  ldb16 $ascratch2, $mzero, $data, 0
  ld32  $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  {
    ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
    // vertex->k should have the form of {1, k} for scaled add/sub
    // or {A, B} for axpymbY variants
    sort4x16lo $ascratch, $ascratch, $ascratch2
  }
  bri   xminusaxplusby_continue
.size VERTEX_XMINUSAXPLUSBY_FALSE, .-VERTEX_XMINUSAXPLUSBY_TENSOR_SLOW


.section .text.VERTEX_COMMON
.align 8
xminusaxplusby_continue:
  {
    ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
    // setup $TAS for the f16v4mix instructions below.
    uput $TAS, $ascratch
  }

  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Louter_loop:
#ifdef VECTOR_AVAIL_SHORT_SPAN
  ld32step $data, $mzero, $outData+=, 1
  shr $origDataSize, $data, SHORT_SPAN_PTR_SIZE
  shl $data, $data, SHORT_SPAN_LENGTH_SIZE
  shr $data, $data, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $data, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR64
  ldz16step $dataB, $mzero, $outDataB+=, 1
  shl $dataB, $dataB, SCALED_PTR64_SHIFTS
#else
  {
    ld32step $dataB, $mzero, $outDataB+=, 1
    fnop
  }
#endif

  // process 4 at a time first as this is the optimal scenario
  {
    shr $dataSizeD4, $origDataSize, 2
    setzi $a0, CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT
  }
  {
    brz $dataSizeD4, .Lvector4_loop_end
    uput $FP_CLR, $a0
  }

  // Choose the fast or slow path, based on flag set at the entry point
  CHOOSE_FAST_OR_SLOW_PATH .Lfast_path
  // Use tapack to copy the 2 addresses into working registers for the loop
  tapack $triPtr, $data, $dataB, $mzero

  // Load first inputs
  ld64 $data0, $mzero, $triPtrData, 0
  ld64step $dataB0, $mzero, $triPtrDataB+=, 1
  // minus 1 because we pipeline the first value.
  {
    add $dataSizeD4, $dataSizeD4, -1
    f16v4mix $azeros, $data0, $dataB0
  }

  rpt $dataSizeD4, (2f-1f)/8-1
1:
  {
    // Load new X input
    ld64 $data0, $mzero, $triPtrData, 1
    // Add previous X input to the accumulator
    f16v4acc $data0
  }
  {
    // Load new Y input
    ld64step $dataB0, $mzero, $triPtrDataB+=, 1
    // Obtain the result from the accumulator for the previous input
    f16v4mix $data1, $azeros, $azeros
  }
  {
    // Store the result for the previous input
    st64step $data1, $mzero, $triPtrData+=, 1
    // Perform -aX + bY for current inputs
    f16v4mix $data1, $data0, $dataB0
  }
2:
  // Process and store final result
  f16v4acc $data0
  f16v4mix $data1, $azeros, $azeros
  st64step $data1, $mzero, $triPtrData+=, 1
  bri .Lvector4_loop_end

.Lfast_path:
  // pack out/in pointers
  tapack $triPtr, $data, $dataB, $data
  // load the first values and push them into the accumulators.
  ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
  {
    // minus 1 from our count because of the preloading above.
    add $dataSizeD4, $dataSizeD4, -1
    f16v4mix $azeros, $data0, $dataB0
  }

  brz $dataSizeD4, .LFast_one_remaining

  {
    // Load second pair of inputs X and Y
    ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
    // Add previous X input to the accumulator
    f16v4acc $data0
  }
  {
    // Decrement loop count due to depth-2 pipelining
    add $dataSizeD4, $dataSizeD4, -1
    // Obtain first result, process previous inputs
    f16v4mix $data1, $data0, $dataB0
  }

  rpt $dataSizeD4, (2f-1f)/8-1
1:
  {
    // Load the next inputs
    ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
    // Add previous X input to the accumulator
    f16v4acc $data0
  }
  {
    // Store the current result
    st64pace $data1, $triPtr+=, $mzero, 0
    // Obtain result for previous inputs and process the current inputs
    f16v4mix $data1, $data0, $dataB0
  }
2:
  // Store the last-but-one result
  st64pace $data1, $triPtr+=, $mzero, 0
.LFast_one_remaining:
  // Finish processing and store the final result
  f16v4acc $data0
  f16v4mix $data1, $azeros, $azeros
  st64pace $data1, $triPtr+=, $mzero, 0

.Lvector4_loop_end:
  // how many left do we have? maximum of 3.
  and $dataSize, $origDataSize, 0x3
  brz $dataSize, .Lend

  // we need to calculate what our out pointer is because the value is hidden
  // inside the $triPtr with no easy way of extracting it. we do this by using
  // how many elements we have processed (origDataSize-currentDataSize), then
  // doubled as we do one 32-bit load for every 2 halves and we want the offset
  // to be number of bytes, not items.
  sub $offset, $origDataSize, $dataSize
  shl $offset, $offset, 1

  // zero the second half of the $data1 and $dataB1 registers because we will
  // only be loading into the first half from now on but processing them using
  // a v4 instruction.
  {
    // if we have at least 2 left we can use a st32 variation for at least some
    // of the remaining values.
    cmpult $mscratch, $dataSize, 2
    zero $data1i1
  }
  {
    brnz $mscratch, .Lscalar
    zero $dataB1i1
  }
.Lvector2:
  ld32 $data1i0, $data, $offset, 0
  ld32 $dataB1i0, $dataB, $offset, 0
  f16v4mix $azeros, $data1, $dataB1
  f16v4acc $data1
  {
    add $dataSize, $dataSize, -2
    f16v4mix $data1, $azeros, $azeros
  }
  st32step $data1i0, $data, $offset+=, 1
  brz $dataSize, .Lend

.Lscalar:
  // there is one more element that needs to be stored, do a read/modify/write
  // so we do not trash anything else may be stored in the same word.
  ldb16 $data1i0, $data, $offset, 0
  ldb16 $dataB1i0, $dataB, $offset, 0

  f16v4mix $azeros, $data1, $dataB1
  f16v4acc $data1
  {
    ldb16 $ascratch, $data, $offset, 1
    f16v4mix $data1, $azeros, $azeros
  }
  roll16 $data1i0, $data1i0, $ascratch

  st32 $data1i0, $data, $offset, 0

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX_COMMON, .-xminusaxplusby_continue

#endif // __IPU__
