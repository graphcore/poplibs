// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#define VERTEX_AXPBY_CONST_FAST __runCodelet_popops__aXPlusbY2D___half_half_true_true
#define VERTEX_AXPBY_TENSOR_FAST __runCodelet_popops__aXPlusbY2D___half_half_false_true
#define VERTEX_AX_MINUS_BY_TENSOR_FAST __runCodelet_popops__aXMinusbY2D___half_half_false_true
#define VERTEX_ADD_CONST_FAST __runCodelet_popops__ScaledAdd2D___half_half_half_true_true
#define VERTEX_ADD_TENSOR_FAST __runCodelet_popops__ScaledAdd2D___half_half_half_false_true
#define VERTEX_SUBTRACT_FAST __runCodelet_popops__ScaledSubtract2D___half_half_true

#define VERTEX_AXPBY_CONST_SLOW __runCodelet_popops__aXPlusbY2D___half_half_true_false
#define VERTEX_AXPBY_TENSOR_SLOW __runCodelet_popops__aXPlusbY2D___half_half_false_false
#define VERTEX_AX_MINUS_BY_TENSOR_SLOW __runCodelet_popops__aXMinusbY2D___half_half_false_false
#define VERTEX_ADD_CONST_SLOW __runCodelet_popops__ScaledAdd2D___half_half_half_true_false
#define VERTEX_ADD_TENSOR_SLOW __runCodelet_popops__ScaledAdd2D___half_half_half_false_false
#define VERTEX_SUBTRACT_SLOW __runCodelet_popops__ScaledSubtract2D___half_half_false

// This is used as an entry point from the half, half, float version, when it
// is decided that the float scale is OK when cast to half
#define VERTEX_COMMON __ScaledAdd2D___half_common

// constants
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 1
#define VERTEX_DATA_B_OFFSET 2
// 2 versions: one has a constant, which is a half
// the other a pointer to a tensor half
#define VERTEX_SCALE_CONST_OFFSET 6
#define VERTEX_SCALE_OFFSET 3
#define VERTEX_SCALE_B_OFFSET 4
// The AXPBY variants have 2 16-bit constants stored at VERTEX_SCALE_AB_OFFSET
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

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR64
#define SCALED_PTR64_SHIFTS 3
#endif

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


FN_WORKER_ENTRY_POINT VERTEX_ADD_TENSOR_SLOW
  setzi $memConstraints, 0
  bri 1f

FN_EXPORT VERTEX_ADD_TENSOR_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET

  ldb16 $ascratch2, $mzero, $data, 0
  bri   VERTEX_COMMON
FN_SIZE VERTEX_ADD_TENSOR_SLOW

FN_WORKER_ENTRY_POINT VERTEX_SUBTRACT_SLOW
  setzi $memConstraints, 0
  bri 1f
FN_EXPORT VERTEX_SUBTRACT_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  {ldb16 $ascratch2, $mzero, $data, 0
   setzi $ascratch, -1.0h}
  {
    bri   VERTEX_COMMON
    f16v2mul $ascratch2, $ascratch:BL, $ascratch2
  }
FN_SIZE VERTEX_SUBTRACT_SLOW

FN_WORKER_ENTRY_POINT VERTEX_AXPBY_CONST_SLOW
  setzi $memConstraints, 0
  bri 1f
FN_EXPORT VERTEX_AXPBY_CONST_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex :
  // constant A,B pair of halves.  32 bit aligned
  ld32  $ascratch, $mvertex_base, $mzero, VERTEX_SCALE_AB_OFFSET
  ld32  $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  ld32  $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
  bri   axplusby_true_continue
FN_SIZE VERTEX_AXPBY_CONST_SLOW


FN_WORKER_ENTRY_POINT VERTEX_AX_MINUS_BY_TENSOR_SLOW
  setzi $memConstraints, 0
  bri 1f
FN_EXPORT VERTEX_AX_MINUS_BY_TENSOR_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensors B
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  ldb16 $ascratch2, $mzero, $data, 0
  //negate scale_b
  setzi $ascratch, -1.0h

  {bri ax_plus_minus_by_common
  f16v2mul $ascratch2, $ascratch, $ascratch2}
FN_SIZE VERTEX_AX_MINUS_BY_TENSOR_SLOW


FN_WORKER_ENTRY_POINT VERTEX_AXPBY_TENSOR_SLOW
  setzi $memConstraints, 0
  bri 1f
FN_EXPORT VERTEX_AXPBY_TENSOR_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensors A,B
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  ldb16 $ascratch2, $mzero, $data, 0
ax_plus_minus_by_common:
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ldb16 $ascratch, $mzero, $data, 0
  ld32  $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  bri   axplusby_false_continue
FN_SIZE VERTEX_AXPBY_TENSOR_SLOW

FN_WORKER_ENTRY_POINT VERTEX_ADD_CONST_SLOW 8
  setzi $memConstraints, 0
  bri 1f
#ifndef VECTOR_AVAIL_SCALED_PTR64
  nop //rpt align
#endif

FN_EXPORT VERTEX_ADD_CONST_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : k, constant
  // set ascratch2 to {k, k}
  ldb16 $ascratch2, $mvertex_base, $mzero, VERTEX_SCALE_CONST_OFFSET

FN_EXPORT VERTEX_COMMON
  // load common vertex state
  {
    ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
    // set ascratch to {1, 1}
    f16v2exp $ascratch, $azero
  }
axplusby_false_continue:
  {
    ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
    // vertex->k should have the form of {1, k} for scaled add/sub
    // or {A, B} for axpby variants
    sort4x16lo $ascratch, $ascratch, $ascratch2
  }
axplusby_true_continue:
  {
    ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
    // setup $TAS for the f16v4mix instructions below.
    uput $TAS, $ascratch
  }

FN_EXPORT axplusby_half_half_common  // This is called from ScaledAddaXPlusbY_mixed.S
  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Louter_loop:
#if defined(VECTOR_AVAIL_SHORT_SPAN)
  ld32step $data, $mzero, $outData+=, 1
  shr $origDataSize, $data, SHORT_SPAN_PTR_SIZE
  shl $data, $data, SHORT_SPAN_LENGTH_SIZE
  shr $data, $data, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $data, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16step $dataB, $mzero, $outDataB+=, 1
  shl $dataB, $dataB, SCALED_PTR64_SHIFTS
#else
  ld32step $dataB, $mzero, $outDataB+=, 1
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

  ld64 $data0, $mzero, $triPtrData, 0
  ld64step $dataB0, $mzero, $triPtrDataB+=, 1
  {add $dataSizeD4, $dataSizeD4, -1
   f16v4mix $azeros, $data0, $dataB0}

  rpt $dataSizeD4, (2f-1f)/8-1
1:
  {ld64 $data0, $mzero, $triPtrData, 1
   f16v4mix $data1, $azeros, $azeros}

  {ld64step $dataB0, $mzero, $triPtrDataB+=, 1
   fnop}

  {st64step $data1, $mzero, $triPtrData+=, 1
   f16v4mix $azeros, $data0, $dataB0}
2:
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
  rpt $dataSizeD4, (2f-1f)/8-1
1:
  {
    // load the next values and retrieve the current from the accumulators.
    ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
    f16v4mix $data1, $azeros, $azeros
  }
  {
    // store the current result and process the next ones.
    st64pace $data1, $triPtr+=, $mzero, 0
    f16v4mix $azeros, $data0, $dataB0
  }
2:
  // process and store the final values.
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
  {
    ldb16 $ascratch, $data, $offset, 1
    f16v4mix $data1, $azeros, $azeros
  }
  roll16 $data1i0, $data1i0, $ascratch

  st32 $data1i0, $data, $offset, 0

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

FN_SIZE VERTEX_ADD_CONST_SLOW

#endif // __IPU__
