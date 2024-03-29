// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#define VERTEX_ADD_TENSOR_HALF_SCALE __runCodelet_popops__ScaledAdd2D___float_half_half_false
#define VERTEX_ADD_TENSOR_FLOAT_SCALE __runCodelet_popops__ScaledAdd2D___float_half_float_false

#define VERTEX_COMMON __ScaledAdd2D___float_half_common


// constants
// Vertex state offsets in bytes
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 4
#define VERTEX_DATA_B_OFFSET 8
#define VERTEX_SCALE_OFFSET 12

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define data m3
#define dataSize m4
#define dataSizeD2 m5
#define dataB m6
#define origDataSize m7

// float variables
#define data0 a0:1
#define dataBHalf a2
#define dataB0 a2:3
#define data1 a4:5
#define data1i0 a4
#define data1i1 a5
#define dataB1 a6:7
#define dataB1i0 a6
#define dataB1i1 a7

// scratch variables
#define ascratch a6

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

FN_WORKER_ENTRY_POINT VERTEX_ADD_TENSOR_FLOAT_SCALE
  // load vertex state specific to this version of the vertex : Tensor(float): via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $ascratch, $mzero, $data, 0
  bri   VERTEX_COMMON
FN_SIZE VERTEX_ADD_TENSOR_FLOAT_SCALE

FN_WORKER_ENTRY_POINT VERTEX_ADD_TENSOR_HALF_SCALE 8 nop
  // load vertex state specific to this version of the vertex : Tensor(half): via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ldb16  $ascratch, $mzero, $data, 0
  f16tof32 $ascratch, $ascratch

VERTEX_COMMON:
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET/4
  ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
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

  ld32step $dataB, $mzero, $outDataB+=, 1

  // process 2 at a time first as this is the optimal scenario
  shr $dataSizeD2, $origDataSize, 1
  brz $dataSizeD2, .Lvector2_loop_end

  ld32step $dataBHalf, $mzero, $dataB+=, 1
  {ld64 $data0, $mzero, $data, 0
   f16v2tof32 $dataB0, $dataBHalf}

  {add $dataSizeD2, $dataSizeD2, -1
   f32v2axpy $azeros, $dataB0, $data0}

  rpt $dataSizeD2, (2f-1f)/8-1
1:
 {ld32step $dataBHalf, $mzero, $dataB+=, 1
  f32v2axpy $data1, $azeros, $azeros}

  {ld64 $data0, $mzero, $data, 1
   f16v2tof32 $dataB0, $dataBHalf}

  {st64step $data1, $mzero, $data+=, 1
   f32v2axpy $azeros, $dataB0, $data0}
2:
  f32v2axpy $data1, $azeros, $azeros
  st64step $data1, $mzero, $data+=, 1

.Lvector2_loop_end:
  // how many left do we have? maximum of 1.
  and $dataSize, $origDataSize, 0x1
  brz $dataSize, .Lend

.Lscalar:
  // zero the second half of the $data1 and $dataB1 registers because we will
  // only be loading into the first half from now on but processing them using
  // a v2 instruction.
  {
    ldb16 $dataB1i0, $mzero, $dataB, 0
    zero $dataB1i1
  }
  {
    ld32 $data1i0, $mzero, $data, 0
    zero $data1i1
  }
  f16tof32 $dataB1i0, $dataB1i0
  f32v2axpy $azeros, $dataB1, $data1
  f32v2axpy $data1, $azeros, $azeros
  st32 $data1i0, $mzero, $data, 0

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

FN_SIZE VERTEX_ADD_TENSOR_HALF_SCALE

#endif // __IPU__
