// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#define VERTEX_ADD_CONST_HALF_SCALE __runCodelet_popops__ScaledAdd2D___half_float_half_true_false
#define VERTEX_ADD_CONST_FLOAT_SCALE __runCodelet_popops__ScaledAdd2D___half_float_float_true_false

#define VERTEX_ADD_TENSOR_HALF_SCALE __runCodelet_popops__ScaledAdd2D___half_float_half_false_false
#define VERTEX_ADD_TENSOR_FLOAT_SCALE __runCodelet_popops__ScaledAdd2D___half_float_float_false_false

#define VERTEX_COMMON __ScaledAdd2D___half_float_common


// constants
// Vertex state offsets in bytes
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 4
#define VERTEX_DATA_B_OFFSET 8
// 2 versions: one has a constant, which is a float
// the other a pointer to a tensor float
#define VERTEX_SCALE_OFFSET 12

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define dataA m3
#define dataSize m4
#define dataSizeD4 m5
#define subVCalc m5
#define dataB m6
#define origDataSize m7

// float variables
#define aA0123  a0:1  //f16v4
#define aA01    a0    //f16v2
#define aBScale a2    //f32v1
#define aTmp    a3
#define aB01    a4:5  //f32v2
#define aB0     a4    //f32v1
#define aB23    a6:7  //f32v2, also used as f16v4
#define aB01h   a4
#define aB0123  a4:7  //f32v4

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif


FN_WORKER_ENTRY_POINT VERTEX_ADD_TENSOR_FLOAT_SCALE
  // load vertex state specific to this version of the vertex : Tensor(float): via a pointer
  ld32  $m0, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $aBScale, $mzero, $m0, 0
  bri   VERTEX_COMMON
FN_SIZE VERTEX_ADD_TENSOR_FLOAT_SCALE

FN_WORKER_ENTRY_POINT VERTEX_ADD_TENSOR_HALF_SCALE
  // load vertex state specific to this version of the vertex : Tensor(half): via a pointer
  ld32  $dataA, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ldb16  $aBScale, $mzero, $dataA, 0
  {bri   VERTEX_COMMON
   f16tof32 $aBScale, $aBScale}
FN_SIZE VERTEX_ADD_TENSOR_HALF_SCALE

FN_WORKER_ENTRY_POINT VERTEX_ADD_CONST_HALF_SCALE 8
  // load vertex state specific to this version of the vertex : k, constant(half)
  ldb16  $aBScale, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  {bri VERTEX_COMMON
   f16tof32 $aBScale, $aBScale}

FN_EXPORT VERTEX_ADD_CONST_FLOAT_SCALE
  // load vertex state specific to this version of the vertex : k, constant(float)
  ld32  $aBScale, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4

VERTEX_COMMON:
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET/4
  ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
  // minus 1 for the outer loop brnzdec
  add $outDataSize, $outDataSize, -1

.Louter_loop:
#ifdef VECTOR_AVAIL_SHORT_SPAN
  ld32step $dataA, $mzero, $outData+=, 1
  shr $origDataSize, $dataA, SHORT_SPAN_PTR_SIZE
  shl $dataA, $dataA, SHORT_SPAN_LENGTH_SIZE
  shr $dataA, $dataA, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $dataA, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

  ld32step $dataB, $mzero, $outDataB+=, 1

  // process 4 at a time first as this is the optimal scenario
  shr $dataSizeD4, $origDataSize, 2
  brz $dataSizeD4, .Lvector4_loop_end

  ld64step    $aB01,    $mzero,     $dataB+=, 1
  {ld64step   $aB23,    $mzero,     $dataB+=, 1
   f32v2mul   $aB01,    $aBScale:B, $aB01
  }
  // Unrolling
  add $dataSizeD4, $dataSizeD4, -1
  rpt $dataSizeD4, (2f-1f)/8-1
1:
  {ld64       $aA0123,  $mzero,     $dataA, 0
   f32v2mul   $aB23,    $aBScale:B, $aB23
  }
  {ld64step   $aB01,    $mzero,     $dataB+=, 1
   f32v4tof16 $aB23,    $aB0123
  }
  {ld64step   $aB23,    $mzero,     $dataB+=, 1
   f16v4add   $aA0123,  $aA0123,    $aB23
  }
  {st64step   $aA0123,  $mzero,     $dataA+=, 1
   f32v2mul   $aB01,    $aBScale:B, $aB01
  }
2:
  {ld64       $aA0123,  $mzero,     $dataA, 0
   f32v2mul   $aB23,    $aBScale:B, $aB23
  }
  f32v4tof16  $aB23,    $aB0123
  f16v4add    $aA0123,  $aA0123,    $aB23
  st64step    $aA0123,  $mzero,     $dataA+=, 1
  // All full/4 vectors have now been processed and stored.
.Lvector4_loop_end:
  // Any remaining partials must be loaded.
  and $subVCalc, $origDataSize, 0x2
  brz $subVCalc, .LhandleLastElement

  // process next 32bit of result
  ld64step    $aB01,    $mzero,     $dataB+=, 1
  {ld32       $aA01,    $mzero,     $dataA, 0
   f32v2mul   $aB01,    $aBScale:B, $aB01
  }
  f32v2tof16  $aB01h,   $aB01
  f16v2add    $aA01,    $aA01,      $aB01h
  st32step    $aA01,    $mzero,     $dataA+=, 1

.LhandleLastElement:
  // how many left do we have? maximum of 1.
  and $subVCalc, $origDataSize, 0x1
  brz $subVCalc, .LouterEnd
  ld32        $aB0,     $mzero,     $dataB, 0 // [ B(float) ]
  {ld32       $aA01,    $mzero,     $dataA, 0 // [ overrun | addend]
   f32mul      $aB0,     $aBScale,  $aB0
  }
  f32tof16    $aB01h,   $aB0                  // [ addend  | addend]
  sort4x16lo  $aTmp,    $azero,     $aA01     // [ value   | 0]
  f16v2add    $aTmp,    $aTmp,      $aB01h    // [ result  | addend]
  sort4x16hi  $aTmp,    $aTmp,      $aA01
  st32        $aTmp,    $mzero,     $dataA, 0

.LouterEnd:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

FN_SIZE VERTEX_ADD_CONST_HALF_SCALE

#endif // __IPU__
