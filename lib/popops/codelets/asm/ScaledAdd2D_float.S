// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"

#define VERTEX_ADD_CONST_FAST __runCodelet_popops__ScaledAdd2D___float_float_float_true_true
#define VERTEX_ADD_TENSOR_FAST __runCodelet_popops__ScaledAdd2D___float_float_float_false_true
#define VERTEX_SUBTRACT_FAST __runCodelet_popops__ScaledSubtract2D___float_true

#define VERTEX_ADD_CONST_SLOW __runCodelet_popops__ScaledAdd2D___float_float_float_true_false
#define VERTEX_ADD_TENSOR_SLOW __runCodelet_popops__ScaledAdd2D___float_float_float_false_false
#define VERTEX_SUBTRACT_SLOW __runCodelet_popops__ScaledSubtract2D___float_false

#define VERTEX_COMMON __ScaledAdd2D___float_common

// constants
#define FLOAT_MINUS_ONE 0xbf800000
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 1
#define VERTEX_DATA_B_OFFSET 2
// 2 versions: one has a constant, which is a float
// the other a pointer to a tensor float
#define VERTEX_SCALE_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define data m3
#define dataSize m4
#define dataSizeD2 m5
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

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR64
#define SCALED_PTR64_SHIFTS 3
#endif

.globl VERTEX_ADD_CONST_FAST
.type VERTEX_ADD_CONST_FAST, @function

.globl VERTEX_ADD_TENSOR_FAST
.type VERTEX_ADD_TENSOR_FAST, @function

.globl VERTEX_SUBTRACT_FAST
.type VERTEX_SUBTRACT_FAST, @function

.globl VERTEX_ADD_CONST_SLOW
.type VERTEX_ADD_CONST_SLOW, @function

.globl VERTEX_ADD_TENSOR_SLOW
.type VERTEX_ADD_TENSOR_SLOW, @function

.globl VERTEX_SUBTRACT_SLOW
.type VERTEX_SUBTRACT_SLOW, @function


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
1:
.endm


.section .text.VERTEX_ADD_TENSOR_FAST
.align 4
VERTEX_ADD_TENSOR_SLOW:
  setzi $memConstraints, 0
  bri 1f
VERTEX_ADD_TENSOR_FAST:
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ld32  $ascratch, $mzero, $data, 0
  bri   VERTEX_COMMON
.size VERTEX_ADD_TENSOR_FAST, .-VERTEX_ADD_TENSOR_FAST

.section .text.VERTEX_SUBTRACT_SLOW
.align 4
VERTEX_SUBTRACT_SLOW:
  setzi $memConstraints, 0
  bri   1f
VERTEX_SUBTRACT_FAST:
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  {ld32  $ascratch, $mzero, $data, 0
   or    $data1i0, $azero, FLOAT_MINUS_ONE}
  {bri  VERTEX_COMMON
   f32mul $ascratch, $ascratch, $data1i0}
.size VERTEX_SUBTRACT_FAST, .-VERTEX_SUBTRACT_SLOW

.section .text.VERTEX_COMMON
.align 8
#ifndef VECTOR_AVAIL_SCALED_PTR64
  nop //rpt align
#endif
VERTEX_ADD_CONST_SLOW:
  setzi $memConstraints, 0
  bri 1f
VERTEX_ADD_CONST_FAST:
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : k, constant
  ld32  $ascratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET

VERTEX_COMMON:
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
  ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
  {
    // minus 1 for the brnzdec
    add $outDataSize, $outDataSize, -1
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $ascratch
  }
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
  ld32step $dataB, $mzero, $outDataB+=, 1
#endif

  // process 2 at a time first as this is the optimal scenario
  shr $dataSizeD2, $origDataSize, 1
  brz $dataSizeD2, .Lvector2_loop_end

  // Choose the fast or slow path, based on flag set at the entry point
  CHOOSE_FAST_OR_SLOW_PATH .Lfast_path

  // Use tapack to copy the 2 addresses into working registers for the loop
  tapack $triPtr, $data, $dataB, $mzero

  ld64 $data0, $mzero, $triPtrData, 0
  ld64step $dataB0, $mzero, $triPtrDataB+=, 1
  {add $dataSizeD2, $dataSizeD2, -1
   f32v2axpy $azeros, $dataB0, $data0}

  rpt $dataSizeD2, (2f-1f)/8-1
1:
  {ld64 $data0, $mzero, $triPtrData, 1
   f32v2axpy $data1, $azeros, $azeros}

  {ld64step $dataB0, $mzero, $triPtrDataB+=, 1
   fnop}

  {st64step $data1, $mzero, $triPtrData+=, 1
   f32v2axpy $azeros, $dataB0, $data0}
2:
  f32v2axpy $data1, $azeros, $azeros
  st64step $data1, $mzero, $triPtrData+=, 1
  bri .Lvector2_loop_end

.Lfast_path:
  // pack out/in pointers
  tapack $triPtr, $data, $dataB, $data
  // load the first values and push them into the accumulators.
  ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
  {
    // minus 1 from our count because of the preloading above.
    add $dataSizeD2, $dataSizeD2, -1
    f32v2axpy $azeros, $dataB0, $data0
  }

  rpt $dataSizeD2, (2f-1f)/8-1
1:
  {
    // load the next values and retrieve the current from the accumulators.
    ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
    f32v2axpy $data1, $azeros, $azeros
  }
  {
    // store the current result and process the next ones.
    st64pace $data1, $triPtr+=, $mzero, 0
    f32v2axpy $azeros, $dataB0, $data0
  }
2:
  // process and store the final values.
  f32v2axpy $data1, $azeros, $azeros
  st64pace $data1, $triPtr+=, $mzero, 0

.Lvector2_loop_end:
  // how many left do we have? maximum of 1.
  and $dataSize, $origDataSize, 0x1
  brz $dataSize, .Lend

  // we need to calculate what our out pointer is because the value is hidden
  // inside the $triPtr with no easy way of extracting it. we do this by using
  // how many elements we have processed (origDataSize-currentDataSize), then
  // times 4 as we do one 32-bit load for every float and we want the offset
  // to be number of bytes, not items.
  sub $offset, $origDataSize, $dataSize
  shl $offset, $offset, 2

.Lscalar:
  // zero the second half of the $data1 and $dataB1 registers because we will
  // only be loading into the first half from now on but processing them using
  // a v2 instruction.
  {
    ld32 $data1i0, $data, $offset, 0
    zero $data1i1
  }
  {
    ld32 $dataB1i0, $dataB, $offset, 0
    zero $dataB1i1
  }
  f32v2axpy $azeros, $dataB1, $data1
  f32v2axpy $data1, $azeros, $azeros
  st32step $data1i0, $data, $offset+=, 1

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX_COMMON, .-VERTEX_ADD_CONST_SLOW

#endif // __IPU__
